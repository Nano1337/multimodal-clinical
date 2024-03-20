import argparse

parser = argparse.ArgumentParser(description="which directory to run")
parser.add_argument("--dir", type=str, default=None, help="directory to run")
ar = parser.parse_args()

if ar.dir == "cremad": 
    from cremad.run_trainer import run_training

if ar.dir is None: 
    raise NotImplementedError("Please specify a directory to run")
else: 
    run_training()
