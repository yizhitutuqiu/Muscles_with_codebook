import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--predemg", default="True")
args = parser.parse_args(["--predemg", "False"])
print(repr(args.predemg))
print(args.predemg == 'True')
