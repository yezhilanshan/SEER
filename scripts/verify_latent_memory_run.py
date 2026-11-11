"""Read back new-run evidence and optionally verify the original workspace file manifest."""
import argparse
import json
from continuous_mmwave.latent_memory.verification import verify_run,verify_original_files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run'); parser.add_argument('--original-files')
    args=parser.parse_args(); report=verify_run(args.run)
    if args.original_files: report['original_files_unchanged']=verify_original_files(args.original_files)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
