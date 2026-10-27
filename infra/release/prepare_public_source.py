"""Build a clean corresponding-source snapshot without publishing or reading secrets."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--license-file',type=Path,required=True)
    parser.add_argument('--ref',default='HEAD',help='An already committed, reviewed source revision')
    parser.add_argument('--release',action='store_true',help='Write the authorized public release notice')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh review directory')
    root=args.out/'source';root.mkdir(parents=True)
    commit=subprocess.check_output(['git','rev-parse',args.ref],text=True).strip()
    archive=subprocess.check_output(['git','archive',commit])
    excludes=('artifacts/','docs/models/runs/','docs/models/experiments/','infra/windows/specs/',
              'infra/windows/verified/','docs/deploy/')
    substitutions={'TRAINING_USER':'TRAINING_USER','MODAL_WORKSPACE':'MODAL_WORKSPACE','TRAINING_PC':'TRAINING_PC','TRAINING_PC':'TRAINING_PC',
                   '/Users/PROJECT_USER':'/Users/PROJECT_USER',
                   'YOUR_CLOUDFLARE_ACCOUNT_ID':'YOUR_CLOUDFLARE_ACCOUNT_ID',
                   'YOUR_CLOUDFLARE_ZONE_ID':'YOUR_CLOUDFLARE_ZONE_ID'}
    copied=[];omitted=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for item in bundle.getmembers():
            name=item.name
            if not item.isfile():continue
            historical_model_record=(name.startswith('docs/models/') and name.endswith('.json')
                                     and name != 'docs/models/yolo-cascade-pretrained.json')
            if (name.startswith(excludes) or historical_model_record or name in ('INFRA.md','PROGRESS.md')
                or '/.secrets/' in '/'+name or name.startswith('.env') and not name.endswith('.example')):
                omitted.append(name);continue
            data=bundle.extractfile(item).read()
            try:
                text=data.decode('utf-8')
            except UnicodeDecodeError:
                if name.endswith(('.png','.jpg','.webp','.ico')):
                    # Ship only public product assets; experimental phone screenshots are omitted above.
                    target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);copied.append(name)
                    continue
                omitted.append(name);continue
            for before,after in substitutions.items():text=text.replace(before,after)
            text=text.replace('https://github.com/XTTTZ/mahjong-recognizer.git', 'https://github.com/XTTTZ/mahjong-recognizer.git')
            text=re.sub(r'https://github\.com/XTTTZ/mahjong(?=[/#?\s)\"\']|$)',
                        'https://github.com/XTTTZ/mahjong-recognizer',text)
            text=re.sub(r'192\.168\.\d{1,3}\.\d{1,3}','TRAINING_HOST',text)
            text=re.sub(r'(?m)^.*ssh-ed25519 [A-Za-z0-9+/=]+.*$','[SSH public-key configuration omitted; use your own key]',text)
            if name.startswith('infra/windows/verified/') and name.endswith('.json'):
                try:
                    value=json.loads(text)
                    if isinstance(value,dict):
                        for key in ('gpu','platform','gpu_name','hostname','host','device_name'):
                            if key in value:value[key]='REDACTED_MACHINE_INFO'
                        text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
                except ValueError:pass
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8');copied.append(name)
    (root/'LICENSE').write_bytes(args.license_file.read_bytes())
    notice='''# Public release candidate — review required\n\nThis is a clean source snapshot of the Mahjong project, proposed for AGPL-3.0 publication. It has not been published.\n\nIt includes the web frontend, upload gateway, inference adapters, model training/evaluation and deployment tools. No Git history, credentials, raw photos, model weights, SSH keys, experiment screenshots or private run receipts are included. Hardware/user/account placeholders must be configured on a new installation. Poker and Avalon repositories are not included.\n\nThe proposed public release will also provide the selected model weights and exact deployment version, with license and dataset attribution. Machine-specific experiment plan receipts are historical evidence and must be regenerated after configuring paths; do not reuse their hashes as launch authorization.\n\nUltralytics states that its AGPL route requires disclosure of the complete corresponding project, including source/configuration/training components and applicable weights. User permission is required before creating a public repository or publishing weights. No commercial license is purchased by this tool.\n'''
    if args.release:
        notice='''# Public source and model release\n\nThis independent Mahjong application is released under AGPL-3.0-only. The source includes the mobile frontend, upload gateway, inference adapters, training, evaluation, data preparation, tests and deployment tools. The selected model weights and their exact hashes are available in the [model release](https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027). See LICENSE and NOTICE for attribution.\n\nThis is a clean snapshot with no private repository history, credentials, raw photographs, SSH keys, experiment screenshots or private run logs. Machine/account placeholders must be configured on a new installation. Poker and Avalon are separate projects and are not included. Historical machine-specific receipts are omitted; regenerate plans after configuring your installation. Data comes from the attributed public source; see docs/data for the recorded preparation and split manifests.\n\nThe online service uses the same inference source and exact model weights. The model is experimental: the locked validation winner scored 72/82 automatic exact hands (at least 10 tiles); its one-time sealed test scored 31/47. These source-group holdouts do not establish accuracy in unfamiliar real-world rooms. Users must review every tile. The classifier was recovered from an intact checkpoint after a time budget expired; this is not a claim of training convergence.\n\nA future 12-hour experiment queue is prepared separately and is not automatically launched by this release. Published commands require explicit local submission. No paid license was purchased.\n'''
        template=Path(__file__).with_name('PUBLIC_README.md')
        (root/'README.md').write_text(template.read_text(encoding='utf-8'),encoding='utf-8')
    (root/'PUBLIC_RELEASE.md').write_text(notice)
    suspicious=[]
    pattern=re.compile(rb'(gh[pousr]_[A-Za-z0-9]{30,}|wk-[A-Za-z0-9]{15,}\.ws-|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    files=[]
    for file in sorted(root.rglob('*')):
        if not file.is_file():continue
        data=file.read_bytes()
        if file.suffix=='.py':compile(data,str(file),'exec')
        if pattern.search(data):suspicious.append(str(file.relative_to(root)))
        files.append({'path':str(file.relative_to(root)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    if suspicious:raise ValueError('Credential-like data requires review in: '+','.join(suspicious))
    zip_path=args.out/'source-review.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in files:bundle.write(root/item['path'],item['path'])
    report={'status':'authorized_release_snapshot_not_published' if args.release else 'local_review_candidate_not_published','sourceCommit':commit,'sourceFiles':len(files),
            'archiveSha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'archiveBytes':zip_path.stat().st_size,
            'credentialPatternMatches':suspicious,'omittedPaths':omitted,'files':files,'publishingPerformed':False,
            'note':'Prepared for user review; final selected weights, latest website fixes and exact serving sources will be refreshed before any publication.'}
    (args.out/'review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','sourceCommit','sourceFiles','archiveBytes','archiveSha256','publishingPerformed')}))


if __name__=='__main__':main()
