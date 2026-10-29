"""Measure protected cold/warm endpoints without printing credentials or image data."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--token-file', type=Path, default=Path('.secrets/modal-proxy-token'))
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--idle-seconds', type=int, default=40)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Preserve old measurements; choose a new output')
    if not 1 <= args.cycles <= 5 or args.idle_seconds < 35: raise ValueError('Invalid probe budget')
    token = args.token_file.read_text().strip()
    base = args.endpoint.rstrip('/').removesuffix('/recognize')
    boundary = uuid.uuid4().hex
    image = args.image.read_bytes()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="validation.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
            +image+f'\r\n--{boundary}--\r\n'.encode())
    report = {'endpoint':base, 'observedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'idleSeconds':args.idle_seconds, 'modelTraining':False, 'complete':False, 'probes':[]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def call(path, data=None):
        headers={'Authorization':'Bearer '+token}
        if data is not None: headers['Content-Type']='multipart/form-data; boundary='+boundary
        request=urllib.request.Request(base+path, data=data, headers=headers)
        started=time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=110) as response:
                status=response.status; value=json.loads(response.read(1024*1024))
        except urllib.error.HTTPError as error:
            status=error.code; value={'error':'http_'+str(status)}
        elapsed=round((time.perf_counter()-started)*1000,1)
        row={'path':path,'status':status,'wallMs':elapsed}
        if path=='/health': row['health']=value
        else:
            row.update({'modelVersion':value.get('modelVersion'),'tiles':len(value.get('tiles',[])),
                        'timings':value.get('timings'), 'quality':value.get('quality'),
                        'classIds':[t['classId'] for t in value.get('tiles',[])]})
        return row
    for cycle in range(args.cycles):
        if cycle: time.sleep(args.idle_seconds)
        row=call('/health'); row.update({'cycle':cycle,'phase':'cold_after_idle' if cycle else 'initial'})
        report['probes'].append(row); print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        if row['status'] != 200: raise RuntimeError('Health did not become ready; no recognition submitted')
        for repeat in range(2):
            row=call('/recognize',body); row.update({'cycle':cycle,'phase':'warm','repeat':repeat})
            report['probes'].append(row); print(json.dumps(row), flush=True)
            args.out.write_text(json.dumps(report, indent=2)+'\n')
            if row['status'] != 200: raise RuntimeError('Recognition failed; no automatic retry')
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    report['complete']=True
    args.out.write_text(json.dumps(report, indent=2)+'\n')

if __name__=='__main__': main()
