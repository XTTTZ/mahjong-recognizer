"""Parse one prepared ML argv and stop before entrypoint work; never train/evaluate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class ParsedOnly(BaseException):
    pass


def validate(argv):
    if len(argv) < 4 or argv[1] != '-m' or not argv[2].startswith(('ml.train.', 'ml.eval.', 'ml.data.')):
        raise ValueError('Expected a prepared python -m ML entrypoint argv')
    original = argparse.ArgumentParser.parse_args
    def parse(parser, args=None, namespace=None):
        original(parser, argv[3:] if args is None else args, namespace)
        raise ParsedOnly()
    with patch.object(argparse.ArgumentParser, 'parse_args', parse):
        try:
            runpy.run_module(argv[2], run_name='__main__')
        except ParsedOnly:
            return {'state': 'argv_parsed_only', 'module': argv[2], 'trainingStarted': False}
    raise ValueError('ML entrypoint did not expose a parse_args boundary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--argv-json', required=True)
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.argv_json))))


if __name__ == '__main__':
    main()
