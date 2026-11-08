#!/usr/bin/env python3
"""Chain your existing append/extension script and the new ROSA organizer.

The extension code pasted by the user imports three private local modules:
build_improved_detzero_rosa_dataset, append_improved_detzero_rosa_dataset,
and ocamcamera. Run it in that environment. This wrapper does not attempt
to replace or pretend to have those unavailable project modules.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extender',type=Path,required=True,
                   help='Path to the supplied extend_combined.py (with its local dependencies)')
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--extended-output',type=Path,required=True)
    p.add_argument('--prepared-output',type=Path,required=True)
    p.add_argument('--add-sequence',nargs=2,action='append',required=True,metavar=('BAG','DATASET'))
    p.add_argument('--sequence-boundaries',nargs=3,action='append',default=[],metavar=('BAG','TRAIN_FRAMES','VAL_START_FRAME'))
    p.add_argument('--native-dataset',nargs=2,action='append',default=[],metavar=('EXISTING_BAG','DATASET'),help='original native JPG dataset for each existing bag')
    p.add_argument('--storage',choices=['copy','hardlink'],default='hardlink')
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--purge-seconds',type=float,default=5.)
    args=p.parse_args()
    for path in (args.extended_output,args.prepared_output):
        if path.exists():raise FileExistsError(f'Refusing to modify existing dataset: {path}')
    cmd=[sys.executable,str(args.extender.resolve()),'--base',str(args.base.resolve()),
         '--output',str(args.extended_output.resolve()),'--storage',args.storage,
         '--val-fraction',str(args.val_fraction),'--purge-seconds',str(args.purge_seconds)]
    for name,src in args.add_sequence:cmd+=['--add-sequence',name,str(Path(src).resolve())]
    for name,train,val in args.sequence_boundaries:cmd+=['--sequence-boundaries',name,train,val]
    subprocess.run(cmd,check=True)
    prepare=Path(__file__).resolve().with_name('prepare_combined_rosa.py')
    cmd=[sys.executable,str(prepare),'--base',str(args.extended_output.resolve()),
         '--output',str(args.prepared_output.resolve()),'--storage',args.storage]
    for name,src in args.native_dataset+args.add_sequence:cmd+=['--native-dataset',name,str(Path(src).resolve())]
    # Supply existing native datasets for the base bags via additional calls
    # to prepare_combined_rosa.py if their base images had been resized to 800x800.
    subprocess.run(cmd,check=True)

if __name__=='__main__':main()
