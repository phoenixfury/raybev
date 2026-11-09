#!/usr/bin/env python3
"""Fetch and verify the pinned official nano checkpoint before offline/DDP training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raybev.yolo26 import DEFAULT_WEIGHTS,ensure_official_weights,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(DEFAULT_WEIGHTS))
    args=parser.parse_args()
    path=ensure_official_weights(args.output)
    print(f'{path}\nsha256={sha256(path)}')


if __name__=='__main__':main()
