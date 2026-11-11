# Remote Training Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the remote SEER checkout reproducible, Git-tracked, CUDA-capable, and runnable with a project-local Conda environment and the downloaded Qwen model.

**Architecture:** Keep deployment-specific state isolated in a dedicated prefix on the large data disk, exposed through `/home/xibeidaxue/SEER/.conda/envs/seer`; model weights remain outside Git, and training outputs remain ignored. Update the Stage 2 model path, GPU-compatible dtype, tested dependency floors, and deployment-specific path assertion. Validate the environment with dependency imports, CUDA tensor work, the non-Gemma repository test scope requested by the user, offline Qwen inference, and the synthetic smoke pipeline; never fabricate production manifests or label synthetic checkpoints as production artifacts.

**Tech Stack:** Git, Conda, Python 3.10, PyTorch with CUDA, Transformers, PEFT, pytest, YAML.

**Spec:** User request in the 2026-09-17 remote-server task.

## Global Constraints

- Do not modify any existing Conda environment.
- Keep the new environment and package cache isolated under `/home/xibeidaxue/newdisk2/WJ/.seer-conda`; expose only an ignored symlink at `/home/xibeidaxue/SEER/.conda/envs/seer`.
- Use `/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct` for Stage 2.
- Do not commit model weights, Conda files, generated data, or outputs.
- Do not invent real training manifests or present synthetic checkpoints as production artifacts.
- Preserve the working CUDA workaround for the unpowered GPU at PCI `0000:19:00.0`.

---

### Task 1: Deployment Configuration and Ignore Rules

**Files:**
- Modify: `configs/stage2.yaml`
- Modify: `.gitignore`
- Create: `docs/superpowers/plans/2026-09-17-remote-training-bootstrap.md`

**Interfaces:**
- Consumes: Downloaded model directory at `/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct`.
- Produces: A Stage 2 configuration that resolves locally and a clean Git boundary.

- [ ] **Step 1: Assert the current Stage 2 model path is the online repository ID**

Run a YAML query and verify it reports `Qwen/Qwen2.5-0.5B-Instruct`.

- [ ] **Step 2: Change only `llm.name_or_path` to the absolute local model directory**

Set it to `/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct`.

- [ ] **Step 3: Add `.conda/` and `Qwen2.5-0.5B-Instruct/` to `.gitignore`**

Keep the existing generated-data and output exclusions.

- [ ] **Step 4: Validate YAML parsing, directory existence, and ignore behavior**

Use Python/YAML plus `git check-ignore` after repository initialization.

### Task 2: Isolated Python/CUDA Environment

**Files:**
- Create: `/home/xibeidaxue/newdisk2/WJ/.seer-conda/envs/seer/`
- Create: `/home/xibeidaxue/newdisk2/WJ/.seer-conda/pkgs/`
- Create: `.conda/envs/seer` symlink

**Interfaces:**
- Consumes: Conda/pip packages and the healthy RTX 2080 Ti.
- Produces: `/home/xibeidaxue/SEER/.conda/envs/seer/bin/python` with compatible runtime dependencies.

- [ ] **Step 1: Build or clone a Python 3.10 prefix without changing existing environments**

Set `CONDA_ENVS_PATH` and `CONDA_PKGS_DIRS` to the dedicated data-disk directories for every Conda operation.

- [ ] **Step 2: Install pinned compatible dependencies from an offline wheelhouse**

Use NumPy 1.26.x, PyTorch 2.4.1 with CUDA 12.1 support, Transformers 5.4.x, PEFT 0.18.x, PyYAML 6.x, and pytest 8.x.

- [ ] **Step 3: Remove the wheelhouse after installation**

Keep only the isolated environment and project-local Conda package cache.

- [ ] **Step 4: Verify imports and CUDA execution**

Assert the package versions, `torch.cuda.is_available()`, one visible healthy GPU, and a synchronized CUDA matrix multiplication.

### Task 3: Repository Initialization

**Files:**
- Create: `.git/`

**Interfaces:**
- Consumes: Validated ignore rules and the current copied source tree.
- Produces: A local Git repository with a baseline commit and no large runtime artifacts tracked.

- [ ] **Step 1: Initialize Git with a `main` branch**

Use the installed Git 2.7-compatible sequence and configure repository-local author identity if needed.

- [ ] **Step 2: Verify ignored artifacts before staging**

Confirm the model, `.conda`, outputs, caches, and generated synthetic datasets are ignored.

- [ ] **Step 3: Create a baseline commit**

Stage only source, tests, configs, documentation, and small metadata files; verify no staged file is unexpectedly large.

### Task 4: Runnable Validation Without Fabricated Production Data

**Files:**
- Create (ignored): `data/synthetic/`
- Create (ignored): `outputs/tiny_stage1/`
- Create (ignored): `outputs/tiny_stage2/`

**Interfaces:**
- Consumes: The isolated environment and existing synthetic-data generator/tiny configs.
- Produces: Passing tests and smoke checkpoints proving the copied code and environment run end to end.

- [ ] **Step 1: Run the repository test suite in the isolated environment, excluding the two user-deferred Gemma compatibility cases**

Run with `PYTHONPATH=src`; require zero failures in the selected 51-test scope and record the two explicit deselections.

- [ ] **Step 2: Generate the standard synthetic dataset**

Run `scripts/make_synthetic_data.py` and validate its manifest paths.

- [ ] **Step 3: Run `scripts/smoke_train.sh`**

Require Stage 1 and Stage 2 tiny outputs and checkpoints to be created successfully.

- [ ] **Step 4: Verify the local Qwen model offline on CUDA**

Force Hugging Face offline mode, load the local model in float16 on the RTX 2080 Ti, and require a successful logits-producing forward pass.

- [ ] **Step 5: Report the remaining production-data boundary accurately**

Confirm that `data/train-stage1.jsonl`, `data/train-stage2.jsonl`, and `outputs/stage1/radar_encoder.pt` remain intentionally absent until real RD/point-cloud/skeleton/caption data is supplied and Stage 1 is trained. Do not symlink synthetic artifacts into production paths.

## Self-Review

- Spec coverage: local model path, isolated environment, Git history, tests, CUDA verification, and missing-data handling are covered.
- Placeholder scan: no implementation placeholders or fabricated artifacts are included.
- Type consistency: this plan changes deployment configuration and environment state only; no new application interfaces are introduced.
