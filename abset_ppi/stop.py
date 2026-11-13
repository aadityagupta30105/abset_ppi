"""Ask a running training job to stop cleanly (works from any terminal).

    python -m abset_ppi.stop --run main

Creates runs/<run>/STOP. The trainer notices it within one optimisation step, keeps the last saved
epoch and exits. Resume later with:  python -m abset_ppi.train --run main
"""
import argparse
import sys

from .utils import run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    rd = run_dir(a.run)
    if not rd.exists():
        raise SystemExit(f"No run named '{a.run}' in {rd.parent}")
    (rd / "STOP").write_text("stop requested\n")
    print(f"Stop requested for run '{a.run}'. The trainer will save and exit within a few seconds.\n"
          f"Check with:  python -m abset_ppi.status --run {a.run}\n"
          f"Resume with: python -m abset_ppi.train --run {a.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
