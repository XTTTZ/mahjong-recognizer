"""Prove public ONNX differs only by metadata; scan all protobuf string fields."""
import argparse
import hashlib
import json
from pathlib import Path
import re

PRIVATE=re.compile(r'(?i)(/Users/|(?<![A-Za-z0-9])[A-Z]:[\\/]|\\Users\\|192\.168\.|TRAINING_USER|tianyi|-----BEGIN[^\n]*PRIVATE|(?:api[_-]?key|token|password|secret)\s*[:=])')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def string_fields(message,prefix=''):
    for desc,value in message.ListFields():
        field=prefix+'.'+desc.name
        if desc.type==desc.TYPE_STRING:
            if desc.is_repeated:
                for i,item in enumerate(value):yield field+f'[{i}]',item
            else:yield field,value
        elif desc.type==desc.TYPE_MESSAGE:
            if desc.is_repeated:
                for i,item in enumerate(value):yield from string_fields(item,field+f'[{i}]')
            else:yield from string_fields(value,field)


def audit(private_dir,public_dir):
    import onnx
    private_dir,public_dir=Path(private_dir),Path(public_dir)
    before=json.loads((private_dir/'manifest.json').read_text());after=json.loads((public_dir/'manifest.json').read_text())
    for key in ['sourceModelFilesSha256','sourceModelManifestSha256','classes','resolution','classifierResolution','cropContext','threshold','acceptThreshold','detectorThreshold','nmsIou','ioContracts']:
        if before[key]!=after[key]:raise ValueError('Portable manifest contract changed: '+key)
    proof={}
    for name in ['detector.onnx','classifier.onnx']:
        a=onnx.load(private_dir/name);b=onnx.load(public_dir/name)
        onnx.checker.check_model(b)
        if a.graph.SerializeToString()!=b.graph.SerializeToString()or a.opset_import!=b.opset_import:
            raise ValueError('Graph or initializers changed: '+name)
        am={x.key:x.value for x in a.metadata_props};bm={x.key:x.value for x in b.metadata_props}
        changes=sorted(k for k in set(am)|set(bm)if am.get(k)!=bm.get(k))
        if set(changes)-{'description','date'}:raise ValueError('Unexpected metadata change: '+str(changes))
        # Values that could contain private data are never copied into audit output.
        findings=[key for key,value in string_fields(b)if PRIVATE.search(value)]
        if findings:raise ValueError('Private marker detected at protobuf fields '+str(findings))
        initializers=[x.SerializeToString()for x in b.graph.initializer]
        proof[name]={'privateSha256':sha(private_dir/name),'publicSha256':sha(public_dir/name),'bytes':(public_dir/name).stat().st_size,
            'graphProtoExactlyEqual':True,'allInitializersExactlyEqual':True,'initializers':len(initializers),'opsetExactlyEqual':True,
            'modifiedMetadataKeys':changes,'publicProtobufStringsScanned':sum(1 for _ in string_fields(b)),
            'privateMarkersFound':0,'publicManifestFileHashMatches':after['filesSha256'][name]==sha(public_dir/name)}
        if not proof[name]['publicManifestFileHashMatches']:raise ValueError('Manifest hash mismatch')
    for key,value in string_fields_json(after):
        if PRIVATE.search(value):raise ValueError('Private marker in manifest field '+key)
    return {'schemaVersion':1,'scope':'public/private export inference-graph byte equality and private text marker audit',
        'sourceModelFilesSha256':after['sourceModelFilesSha256'],'sourceModelManifestSha256':after['sourceModelManifestSha256'],
        'classesAndInferenceContractExactlyEqual':True,'files':proof,'modelBytesTotal':sum(v['bytes']for v in proof.values()),
        'graphForwardExecuted':False,'trainingExecuted':False,'testOpened':False,
        'auditScriptSha256':sha(__file__),
        'limitations':['Text-marker scan is limited to known identifying/path/credential patterns; not a universal secrecy proof.',
                       'Evidence measured using private export hashes applies to public graphs by exact GraphProto/initializer equality; runtime source changes still require their own validation.']}

def string_fields_json(value,prefix=''):
    if isinstance(value,dict):
        for key,v in value.items():yield from string_fields_json(v,prefix+'.'+key)
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from string_fields_json(v,prefix+f'[{i}]')
    elif isinstance(value,str):yield prefix,value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve prior audit evidence')
    result=audit(a.private_dir,a.public_dir);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
