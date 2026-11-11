"""Training, checkpointing, independent-subject evaluation and label-free inference."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from continuous_mmwave.data_parrad import assert_disjoint_splits

from .config import RadarTextConfig
from .data import (
    OfflineCaptionDataset, RecordingWindowSampler, caption_from_actions,
    collapse_actions, collate_windows, load_phrases,
)
from .metrics import score_caption
from .model import RadarPretrainer, RadarTextModel


def make_tiny_model(config):
    """Actual random Qwen2, purely for CPU plumbing tests; never a trained reader."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "system", "user", "assistant",
             "The", "person", "walks", "sits", "down", "lies", "then", "and", ".", ",",
             "radar", "activities", "in", "order", "human", "Describe", "the", "recording"]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + "
        "message['content'] + '<|im_end|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        llm = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=1024, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, attention_dropout=0.0,
        ))
    return RadarTextModel(config, llm, tokenizer)


def build_model(settings, stage, classes):
    config = RadarTextConfig(**settings.get("model", {}))
    if stage == "stage1":
        return RadarPretrainer(config, classes)
    llm_settings = settings.get("llm", {})
    if settings.get("tiny", False):
        if stage == "lora":
            raise ValueError("tiny smoke mode does not simulate LoRA")
        return make_tiny_model(config)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = llm_settings.get("name", "Qwen/Qwen2.5-0.5B-Instruct")
    precision = llm_settings.get("dtype", "float32")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("LLM dtype must be float32, float16 or bfloat16")
    local = llm_settings.get("local_files_only", True)
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local)
    llm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, precision),
                                              local_files_only=local)
    rank = int(llm_settings.get("lora_rank", 16)) if stage == "lora" else 0
    if stage == "lora" and rank <= 0:
        raise ValueError("LoRA stage requires positive lora_rank")
    return RadarTextModel(config, llm, tokenizer, lora_rank=rank,
                          lora_alpha=llm_settings.get("lora_alpha", 32),
                          lora_dropout=llm_settings.get("lora_dropout", 0.05))


def save_checkpoint(path, model, *, epoch, stage, settings):
    payload = dict(format_version=1, stage=stage, epoch=epoch, config=model.config.to_dict(),
                   encoder=model.encoder.state_dict(), settings=settings,
                   training_subjects=sorted(getattr(model, "training_subjects", set())),
                   selection_subjects=sorted(getattr(model, "selection_subjects", set())))
    if isinstance(model, RadarPretrainer):
        payload["classifier"] = model.classifier.state_dict()
    else:
        payload.update(projector=model.projector.state_dict(), prompt=model.prompt,
                       lora_rank=model.lora_rank, lora_alpha=model.lora_alpha,
                       lora_dropout=model.lora_dropout)
        if model.lora_rank:
            from peft import get_peft_model_state_dict
            payload["adapter"] = get_peft_model_state_dict(model.llm)
        if settings.get("tiny"):
            payload["tiny_llm"] = model.llm.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("unsupported Radar-to-Text checkpoint format")
    if not {"training_subjects", "selection_subjects"}.issubset(saved):
        raise ValueError("checkpoint lacks subject provenance; use a newly trained checkpoint")
    return saved


def load_weights(path, model):
    saved = read_checkpoint(path)
    if saved["config"] != model.config.to_dict():
        raise ValueError("checkpoint encoder config does not match model config")
    if isinstance(model, RadarTextModel) and "prompt" in saved and saved["prompt"] != model.prompt:
        raise ValueError("checkpoint prompt does not match inference prompt")
    model.encoder.load_state_dict(saved["encoder"])
    model.training_subjects = set(getattr(model, "training_subjects", set())) | set(saved["training_subjects"])
    model.selection_subjects = set(getattr(model, "selection_subjects", set())) | set(saved["selection_subjects"])
    if isinstance(model, RadarPretrainer):
        if "classifier" not in saved:
            raise ValueError("classification baseline requires a stage1 checkpoint")
        model.classifier.load_state_dict(saved["classifier"])
    elif "projector" in saved:
        model.projector.load_state_dict(saved["projector"])
        if "adapter" in saved:
            if (model.lora_rank, model.lora_alpha, model.lora_dropout) != (
                saved["lora_rank"], saved["lora_alpha"], saved["lora_dropout"],
            ):
                raise ValueError("checkpoint LoRA configuration does not match model")
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model.llm, saved["adapter"])
        if "tiny_llm" in saved:
            model.llm.load_state_dict(saved["tiny_llm"])
    return saved


def check_holdout_subjects(saved, rows, *, test=False):
    seen = set(saved["training_subjects"])
    if test:
        seen.update(saved["selection_subjects"])
    overlap = seen & {str(row["subject_id"]) for row in rows}
    if overlap:
        raise ValueError(f"checkpoint training/selection subjects overlap evaluation subjects: {sorted(overlap)}")


def datasets(settings, names):
    data = settings["data"]
    phrases = load_phrases(data["phrases"])
    rows = {}
    # Check metadata in every supplied split before accessing any training values.
    for name in ("train", "validation", "test"):
        path = data.get(f"{name}_manifest")
        rows[name] = [] if not path else [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line.strip()]
    assert_disjoint_splits(**rows)
    options = {key: data[key] for key in ("window_seconds", "stride_seconds", "min_seconds", "max_actions")
               if key in data}
    result = {name: OfflineCaptionDataset(data[f"{name}_manifest"], phrases,
                                          captions=data.get(f"{name}_captions"), **options) for name in names}
    return result, phrases


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def batch_loss(model, batch):
    if isinstance(model, RadarPretrainer):
        logits = model(batch["rd"], batch["md"], batch["lengths"])
        return F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=-100), int(
            batch["labels"].ne(-100).sum())
    output = model(batch["rd"], batch["md"], batch["lengths"], batch["captions"])
    count = sum(len(model.answer_ids(caption)) for caption in batch["captions"])
    return output.loss, count


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(settings, stage, checkpoint=None):
    options = settings.get("training", {})
    seed_everything(int(options.get("seed", 17)))
    device = torch.device(options.get("device", "cpu"))
    sets, phrases = datasets(settings, ("train", "validation"))
    model = build_model(settings, stage, len(phrases)).to(device)
    model.training_subjects = {str(row["subject_id"]) for row in sets["train"].recordings.rows}
    model.selection_subjects = {str(row["subject_id"]) for row in sets["validation"].recordings.rows}
    if checkpoint:
        saved = read_checkpoint(checkpoint)
        for name in ("validation", "test"):
            path = settings["data"].get(f"{name}_manifest")
            if path:
                rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                        if line.strip()]
                check_holdout_subjects(saved, rows, test=name == "test")
        old_phrases = saved["settings"].get("phrase_vocabulary")
        if old_phrases and old_phrases != {str(k): v for k, v in phrases.items()}:
            raise ValueError("checkpoint phrase vocabulary differs from training")
        if saved["stage"] != "stage1" and saved["settings"].get("llm", {}).get("name") != (
            settings.get("llm", {}).get("name")
        ):
            raise ValueError("checkpoint base LLM does not match requested LLM")
        load_weights(checkpoint, model)
    elif stage == "lora":
        raise ValueError("LoRA stage requires an aligned caption checkpoint")
    if stage == "lora" and saved["stage"] == "stage1":
        raise ValueError("run align before lora")
    epochs = int(options.get("epochs", 5))
    batch_size = int(options.get("batch_size", 2))
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    loaders = {name: DataLoader(dataset, batch_size=batch_size,
                                sampler=RecordingWindowSampler(dataset) if name == "train" else None,
                                num_workers=0, collate_fn=collate_windows) for name, dataset in sets.items()}
    groups = [{"params": list(model.encoder.parameters()), "lr": float(options.get("encoder_lr", 1e-4))}]
    if isinstance(model, RadarPretrainer):
        groups.append({"params": list(model.classifier.parameters()), "lr": float(options.get("lr", 1e-3))})
    else:
        groups.append({"params": list(model.projector.parameters()), "lr": float(options.get("lr", 1e-3))})
        if model.lora_rank:
            groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad],
                           "lr": float(options.get("lora_lr", 2e-5))})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(options.get("weight_decay", 0.01)))
    accumulation = int(options.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    output_dir = Path(options.get("output_dir", f"outputs/radar-text-{stage}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    report = dict(stage=stage, windows={name: len(ds) for name, ds in sets.items()},
                  skipped={name: ds.skipped for name, ds in sets.items()},
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  weak_supervision=True, tiny_random_llm=bool(settings.get("tiny")))
    print(json.dumps(report), flush=True)
    for epoch in range(1, epochs + 1):
        losses = {}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for index, raw in enumerate(loader):
                batch = move_batch(raw, device)
                with torch.set_grad_enabled(split == "train"):
                    loss, weight = batch_loss(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss in {split} epoch {epoch}")
                    if split == "train":
                        # Last partial accumulation group uses its actual batch count.
                        group_size = min(accumulation, len(loader) - (index // accumulation) * accumulation)
                        (loss / group_size).backward()
                        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in model.parameters() if p.requires_grad],
                                float(options.get("grad_clip", 1.0)), error_if_nonfinite=True,
                            )
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach()) * weight
                count += weight
                log_every = int(options.get("log_every", 20))
                if split == "train" and log_every > 0 and (index + 1) % log_every == 0:
                    print(json.dumps(dict(epoch=epoch, batch=index + 1, batches=len(loader), loss=total / count)),
                          flush=True)
            losses[split] = total / count
        history.append(dict(epoch=epoch, **losses))
        save_checkpoint(output_dir / "last.pt", model, epoch=epoch, stage=stage, settings=settings)
        if losses["validation"] < best:
            best = losses["validation"]
            save_checkpoint(output_dir / "best.pt", model, epoch=epoch, stage=stage, settings=settings)
        report["history"] = history
        (output_dir / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(history[-1]), flush=True)
    return output_dir / "best.pt"


def restore_caption_model(checkpoint, device, model_name=None):
    saved = read_checkpoint(checkpoint)
    if saved["stage"] == "stage1":
        raise ValueError("caption inference requires align/lora checkpoint")
    settings = {**saved["settings"], "model": saved["config"]}
    if model_name:
        settings = {**settings, "llm": {**settings.get("llm", {}), "name": model_name}}
    model = build_model(settings, saved["stage"], 0).to(device)
    load_weights(checkpoint, model)
    return model.eval(), saved


@torch.no_grad()
def evaluate(settings, checkpoint, *, split="test", baseline_checkpoint=None, output=None):
    sets, phrases = datasets(settings, (split,))
    options = settings.get("training", {})
    device = torch.device(options.get("device", "cpu"))
    model, saved = restore_caption_model(checkpoint, device)
    trained_phrases = saved["settings"].get("phrase_vocabulary")
    if trained_phrases and trained_phrases != {str(k): v for k, v in phrases.items()}:
        raise ValueError("evaluation phrase vocabulary differs from training")
    check_holdout_subjects(saved, sets[split].recordings.rows, test=split == "test")
    baseline = None
    if baseline_checkpoint:
        baseline_saved = read_checkpoint(baseline_checkpoint)
        check_holdout_subjects(baseline_saved, sets[split].recordings.rows, test=split == "test")
        baseline_phrases = baseline_saved["settings"].get("phrase_vocabulary")
        if baseline_phrases and baseline_phrases != {str(k): v for k, v in phrases.items()}:
            raise ValueError("baseline phrase vocabulary differs from evaluation")
        baseline = RadarPretrainer(RadarTextConfig(**baseline_saved["config"]), len(phrases)).to(device)
        load_weights(baseline_checkpoint, baseline)
        baseline.eval()
    loader = DataLoader(sets[split], batch_size=int(options.get("batch_size", 2)), collate_fn=collate_windows)
    results = []
    for raw in loader:
        batch = move_batch(raw, device)
        texts = model.generate(batch["rd"], batch["md"], batch["lengths"],
                               max_new_tokens=int(settings.get("max_new_tokens", 96)))
        predictions = None if baseline is None else baseline(batch["rd"], batch["md"], batch["lengths"]).argmax(-1)
        for i, text in enumerate(texts):
            row = dict(sample_id=batch["sample_ids"][i], subject_id=batch["subjects"][i],
                       reference=batch["captions"][i], reference_actions=batch["actions"][i],
                       generated=text, **score_caption(text, batch["actions"][i], phrases))
            if predictions is not None:
                actions = collapse_actions(predictions[i, :int(batch["lengths"][i])].tolist())
                baseline_text = caption_from_actions(actions, phrases)
                row["baseline_text"] = baseline_text
                row["baseline"] = score_caption(baseline_text, batch["actions"][i], phrases)
            results.append(row)
    keys = ("action_recall", "action_precision", "sequence_exact", "order_lcs", "template_exact", "unmatched")
    def mean(rows):
        return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    report = dict(split=split, windows=len(results), metrics=mean(results),
                  metric_scope="restricted phrase matching; not fine-grained language quality",
                  tiny_random_llm=bool(saved["settings"].get("tiny")), results=results)
    if baseline:
        report["baseline_metrics"] = mean([row["baseline"] for row in results])
    report["by_subject"] = {
        subject: mean([row for row in results if row["subject_id"] == subject])
        for subject in sorted({row["subject_id"] for row in results})
    }
    report["by_action_count"] = {
        str(count): dict(windows=len(group), metrics=mean(group))
        for count in sorted({len(row["reference_actions"]) for row in results})
        if (group := [row for row in results if len(row["reference_actions"]) == count])
    }
    destination = Path(output or f"outputs/radar-text-{split}-evaluation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    return report


def infer(checkpoint, input_path, *, device="cpu", model_name=None, max_new_tokens=96):
    model, _ = restore_caption_model(checkpoint, device, model_name)
    # Labels, captions, timestamps and class IDs are not used by the generation path.
    with np.load(input_path, allow_pickle=False) as arrays:
        rd = torch.tensor(arrays["rd"], dtype=torch.float32, device=device).unsqueeze(0)
        md = torch.tensor(arrays["md"], dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([rd.shape[1]], device=device)
    return model.generate(rd, md, lengths, max_new_tokens=max_new_tokens)[0]


def smoke(output_dir):
    """Exercise both stages, checkpoint reload and generation with no downloads."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    phrases = {0: "walks", 1: "sits down", 2: "lies down"}
    phrase_path = output / "phrases.json"
    phrase_path.write_text(json.dumps(phrases), encoding="utf-8")
    data = dict(phrases=str(phrase_path), window_seconds=2, stride_seconds=2, min_seconds=2, max_actions=3)
    for index, split in enumerate(("train", "validation", "test")):
        path = output / f"{split}.npz"
        np.savez(path, rd=rng.normal(size=(8, 1, 93, 128)).astype("float32"),
                 md=rng.normal(size=(8, 128)).astype("float32"), labels=np.repeat([0, 1], 4),
                 timestamps=np.arange(8) * 0.25)
        manifest = output / f"{split}.jsonl"
        row = dict(path=str(path), subject_id=f"synthetic-{index}", session_id=f"session-{index}",
                   sensor_id="60ghz", environment_id="synthetic")
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        data[f"{split}_manifest"] = str(manifest)
    settings = dict(tiny=True, data=data, model=dict(dropout=0),
                    phrase_vocabulary={str(k): v for k, v in phrases.items()},
                    training=dict(epochs=1, batch_size=1, device="cpu", output_dir=str(output / "stage1")),
                    max_new_tokens=5)
    first = train(settings, "stage1")
    settings["training"] = {**settings["training"], "output_dir": str(output / "align")}
    second = train(settings, "align", first)
    report = evaluate(settings, second, baseline_checkpoint=first, output=output / "evaluation.json")
    description = infer(second, output / "test.npz", max_new_tokens=5)
    print(json.dumps(dict(smoke="passed", generated=description, meaningful_caption=False)), flush=True)
    return report


def read_settings(path):
    """Resolve all data/output paths relative to the YAML, not the caller's cwd."""
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = settings["data"]
    for key in ("train_manifest", "validation_manifest", "test_manifest", "phrases",
                "train_captions", "validation_captions", "test_captions"):
        if data.get(key):
            data[key] = str((path.parent / data[key]).resolve())
    output = settings.setdefault("training", {}).get("output_dir")
    if output:
        settings["training"]["output_dir"] = str((path.parent / output).resolve())
    name = settings.get("llm", {}).get("name", "")
    if name.startswith(("./", "../", ".\\", "..\\")):
        settings["llm"]["name"] = str((path.parent / name).resolve())
    phrases = load_phrases(data["phrases"])
    settings["phrase_vocabulary"] = {str(k): v for k, v in phrases.items()}
    return settings


def prepare(settings, split, output):
    sets, _ = datasets(settings, (split,))
    dataset = sets[split]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for index, start, stop, sample_id, actions in dataset.windows:
            row = dict(sample_id=sample_id, recording=dataset.recordings.rows[index]["path"],
                       start_frame=start, stop_frame=stop, actions=actions,
                       caption=dataset.overrides.get(sample_id, caption_from_actions(actions, dataset.phrases)))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(output=str(destination), windows=len(dataset), skipped=dataset.skipped)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--stage", choices=["stage1", "align", "lora"], required=True)
    training.add_argument("--checkpoint")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--baseline-checkpoint")
    evaluation.add_argument("--output")
    inference = sub.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--model-name")
    inference.add_argument("--max-new-tokens", type=int, default=96)
    smoking = sub.add_parser("smoke")
    smoking.add_argument("--output-dir", default="outputs/radar-text-smoke")
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--config", required=True)
    preparation.add_argument("--split", choices=["train", "validation", "test"], default="train")
    preparation.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        print(train(read_settings(args.config), args.stage, args.checkpoint))
    elif args.command == "evaluate":
        evaluate(read_settings(args.config), args.checkpoint, split=args.split,
                 baseline_checkpoint=args.baseline_checkpoint, output=args.output)
    elif args.command == "infer":
        print(infer(args.checkpoint, args.input, device=args.device,
                    model_name=args.model_name, max_new_tokens=args.max_new_tokens))
    elif args.command == "prepare":
        prepare(read_settings(args.config), args.split, args.output)
    else:
        torch.set_num_threads(1)
        smoke(args.output_dir)


if __name__ == "__main__":
    main()
