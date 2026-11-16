#!/usr/bin/env python3
"""Read-only four-camera contact sheet for a flagged annotation track."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from review_heavy_annotations import ReviewSession,fit_preview,DEFAULT_CALIBRATION,CLASSES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--track',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=3)
    a=p.parse_args();release=a.release.resolve()
    with tempfile.TemporaryDirectory(prefix='raybev-preview-state-') as temporary:
        s=ReviewSession(a.root,release/'fisheye3dod_infos_train.pkl',release/'fisheye3dod_infos_val.pkl',
            release/'train_yolo_extents_v1.json',DEFAULT_CALIBRATION,Path(temporary)/'unused.json',
            classes=CLASSES)
        records=[rec for rec in s.records if rec['sequence']==a.sequence and rec['track_id']==a.track]
        records.sort(key=lambda rec:rec['token'])  # Interleave TRAIN/VAL in time.
        if not records:raise ValueError('Track not in the requested review class set')
        indices=np.unique(np.linspace(0,len(records)-1,min(a.samples,len(records))).round().astype(int))
        sheet=Image.new('RGB',(1600,330*len(indices)),(35,35,35));draw=ImageDraw.Draw(sheet)
        try:font=ImageFont.truetype('LiberationSans-Regular.ttf',22)
        except OSError:font=ImageFont.load_default()
        for ri,index in enumerate(indices):
            rec=records[index];top=ri*330
            draw.text((10,top+4),f"{a.sequence}/{a.track} | {rec['split']} {rec['token']} | {rec['original_class']} | LWH {np.round(rec['box_CLIDAR'][3:6],2)}",font=font,fill='white')
            for ci,(name,preview,_) in enumerate(s.render(rec['id'],context=.25)):
                scaled=fit_preview(preview,392,275);left=ci*400+(400-scaled.width)//2
                sheet.paste(scaled,(left,top+48+(275-scaled.height)//2))
                draw.text((ci*400+6,top+28),name,font=font,fill='yellow')
        a.output.parent.mkdir(parents=True,exist_ok=True);sheet.save(a.output,quality=92)
        print(a.output)


if __name__=='__main__':main()
