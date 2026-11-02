import argparse

from track_counter import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="track-counter", description=__doc__)
    parser.add_argument("--version", action="version", version=f"track-counter {__version__}")
    parser.parse_args(argv)
    parser.print_help()
