"""AbSet antibody–antigen pose scorer.

Commands (run from the package root folder):
  python -m abset_ppi.check      environment and data check
  python -m abset_ppi.embed      compute language-model embeddings (once, resumable)
  python -m abset_ppi.train      train / resume a run
  python -m abset_ppi.stop       ask a running training job to stop cleanly
  python -m abset_ppi.status     progress and metrics of a run
  python -m abset_ppi.finalize   calibrate and evaluate on the test set (train runs this automatically)
  python -m abset_ppi.predict    score complexes with a trained run
"""
__version__ = "1.0.0"
