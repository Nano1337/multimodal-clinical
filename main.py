import argparse

parser = argparse.ArgumentParser(description="which directory to run")
parser.add_argument("--dir", type=str, default=None, help="directory to run")
arg = parser.parse_args()


if arg.dir == "cremad":
    from cremad.run_trainer import run_training
elif arg.dir == "food101":
    from food101.run_training import run_training
elif arg.dir == "enrico":
    from enrico.run_training import run_training
elif arg.dir == "mimic":
    from mimic.run_training import run_training
else: 
    raise NotImplementedError("Please specify a directory to run")

run_training()