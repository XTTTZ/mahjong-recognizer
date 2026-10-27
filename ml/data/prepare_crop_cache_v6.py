"""Lossless RGB pixel envelopes for train/valid crops; test remains unopened."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict
from PIL import Image
from ml.train.provenance import atomic_json, dataset_contract, sha256
from ml.data.prepare_crops import verify_prepared as verify_indices


def envelope(box, width, height, margin=.35):
    x,y,w,h=box
    return [max(0,math.floor(x-w*margin)),max(0,math.floor(y-h*margin)),
            min(width,math.ceil(x+w*(1+margin))),min(height,math.ceil(y+h*(1+margin)))]


def verify_cache(out, contract, crops):
    out=Path(out);meta=json.loads((out/'prepared.json').read_text())
    if meta.get('schemaVersion')!=2 or meta.get('kind')!='lossless_crop_envelopes_v6' or meta['dataContract']!=contract or meta['sourceIndicesSha256']!=sha256(Path(crops)/'prepared.json'):
        raise ValueError('Crop envelope cache belongs to different source indices/pixels')
    for split in ('train','valid'):
        index=json.loads((out/(split+'.json')).read_text())
        if sha256(out/(split+'.json'))!=meta['indices'][split]['sha256']:
            raise ValueError('Cached crop index was changed')
        for row in index['records']:
            required={'bbox','cacheEnvelopeXYXY','sourceImageSize','cacheFile','cacheSha256'}
            if not required.issubset(row):raise ValueError('Cache index lacks global-coordinate geometry; rebuild as schema2')
            rectangle=row['cacheEnvelopeXYXY'];size=row['sourceImageSize']
            if len(rectangle)!=4 or len(size)!=2 or not all(isinstance(v,int)for v in rectangle+size):raise ValueError('Invalid cache geometry')
            if not(0<=rectangle[0]<rectangle[2]<=size[0]and 0<=rectangle[1]<rectangle[3]<=size[1]):raise ValueError('Cache envelope escapes its source image')
            path=out/row['cacheFile']
            if not path.resolve().is_relative_to(out.resolve()) or sha256(path)!=row['cacheSha256']:
                raise ValueError('Cached lossless pixels were changed')
    return meta


def prepare(dataset,crops,out,preflight_only=False):
    dataset,crops,out=map(Path,(dataset,crops,out));contract=dataset_contract(dataset,verify_images=True)
    verify_indices(crops,contract)
    if preflight_only:
        ready=(out/'prepared.json').is_file()
        if ready:verify_cache(out,contract,crops)
        return {'derivativeReady':ready,'testOpened':False,'trainingStarted':False}
    if out.exists():raise FileExistsError('Use a fresh cache directory')
    out.mkdir(parents=True)
    indices={}
    for split in ('train','valid'):
        records=json.loads((crops/(split+'.json')).read_text())['records'];by_image=defaultdict(list)
        for record in records:
            if not isinstance(record['annotationId'],int) or record['annotationId']<0:raise ValueError('Cache annotation IDs must be nonnegative integers')
            by_image[record['image']].append(record)
        cached=[];(out/split).mkdir()
        for filename,rows in by_image.items():
            source=dataset/split/filename
            with Image.open(source) as photo:
                photo=photo.convert('RGB')
                for row in rows:
                    rect=envelope(row['bbox'],photo.width,photo.height)
                    target=out/split/(str(row['annotationId'])+'.png')
                    photo.crop(rect).save(target,compress_level=1)
                    x,y,w,h=row['bbox']
                    cached.append(dict(row,cacheFile=target.relative_to(out).as_posix(),cacheSha256=sha256(target),
                        cacheEnvelopeXYXY=rect,sourceImageSize=[photo.width,photo.height],cacheBBox=[x-rect[0],y-rect[1],w,h]))
        atomic_json(out/(split+'.json'),{'records':cached})
        indices[split]={'sha256':sha256(out/(split+'.json')),'crops':len(cached)}
    meta={'schemaVersion':2,'kind':'lossless_crop_envelopes_v6','dataContract':contract,
          'sourceIndicesSha256':sha256(crops/'prepared.json'),'indices':indices,'envelopeMargin':.35,
          'maximumSupportedContext':.16,'preparationCodeSha256':sha256(__file__),'testOpened':False,
          'note':'Lossless pixel storage only; no new photographs, labels, groups, or train/valid split changes.'}
    atomic_json(out/'prepared.json',meta)
    return {'derivativeReady':True,'indices':indices,'testOpened':False,'trainingStarted':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--crops',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();print(json.dumps(prepare(args.dataset,args.crops,args.out,args.preflight_only)))

if __name__=='__main__':main()
