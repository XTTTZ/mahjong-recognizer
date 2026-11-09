import { describe,expect,it } from 'vitest';
import { DEFAULT_PROFILE, headPoints, modelInfo, profilePath, validateManifest, validateSessionIO, type Manifest } from './profiles';
function fixture(resolution=800,classifierResolution=224,original=false):Manifest{
  const tokens=Array.from({length:9},(_,i)=>['m','p','s'].map(s=>`${i+1}${s}`)).flat().concat(Array.from({length:7},(_,i)=>`${i+1}z`),Array.from({length:8},(_,i)=>`${i+1}h`));
  return {schemaVersion:1,format:original?'onnx-fp32-cascade':'research-onnx-fp32-cascade',version:'test-profile',ontology:'faces42',precision:'float32',opset:17,researchOnly:!original,
    detectorArchitecture:'yolov8n-generic',classifierArchitecture:classifierResolution===320?'efficientnet-v2-s':'mobilenet-v3-large',
    resolution,classifierResolution,cropContext:.08,detectorThreshold:.05,nmsIou:.6,detectorNmsMaxDetections:300,maxDetections:200,maxClassifierBatch:32,threshold:.35,acceptThreshold:.7,
    classes:Object.fromEntries(tokens.map((t,i)=>[String(i),t])),filesBytes:{'detector.onnx':12_000_000,'classifier.onnx':17_000_000},filesSha256:{'detector.onnx':'a'.repeat(64),'classifier.onnx':'b'.repeat(64)},
    sourceEvidence:{detector:{runSha256:'c'.repeat(64),checkpointSha256:'d'.repeat(64),signatureSha256:'e'.repeat(64)},classifier:{runSha256:'f'.repeat(64),checkpointSha256:'a'.repeat(64),signatureSha256:'b'.repeat(64)},testOpened:false},
    ioContracts:{'detector.onnx':{inputs:{images:[1,3,resolution,resolution]},outputs:{output0:[1,5,headPoints(resolution)]},float32:true},'classifier.onnx':{inputs:{images:['batch',3,classifierResolution,classifierResolution]},outputs:{logits:['batch',42]},float32:true}}};
}
describe('explicit portable profiles',()=>{
 it('prefers configured compact profile and accepts only fixedsameorigin resource paths',()=>{
  expect(DEFAULT_PROFILE).toBe('compact-v10');expect(profilePath('compact-v10','manifest.json')).toBe('/models/compact-v10/manifest.json');
  expect(profilePath('portable-v1','detector.onnx')).toBe('/models/portable-v1/detector.onnx');
  expect(()=>profilePath('https://remote.test/models' as never,'manifest.json')).toThrow();expect(()=>profilePath('compact-v10','../../private' as never)).toThrow();
 });
 it('preserves originalprofile and supportsreviewedcompactgeometries',()=>{
  const old=fixture(960,320,true);expect(validateManifest(old,'portable-v1')).toEqual(old);
  for(const resolution of [640,800,960])for(const size of [192,224,320]){const m=fixture(resolution,size);expect(validateManifest(m,'compact-v10')).toEqual(m);}
  expect([640,800,960].map(headPoints)).toEqual([8400,13125,18900]);
  expect(()=>validateManifest(fixture(),'portable-v1')).toThrow();expect(()=>validateManifest(old,'compact-v10')).toThrow();expect(()=>headPoints(1280)).toThrow();
 });
 it('reports exact model bytesfromverifiedmanifest',()=>{const m=fixture();expect(modelInfo(validateManifest(m,'compact-v10'),'compact-v10')).toMatchObject({modelBytes:29_000_000,profileId:'compact-v10',detectorResolution:800,classifierResolution:224});});
 it('rejects changedclasses,precision,threshold,geometry,IO andunsignedsourceproof',()=>{
  const cases:((m:Manifest)=>void)[]=[m=>{m.classes['0']=m.classes['1'];},m=>{m.filesBytes['detector.onnx']=NaN;},m=>{m.filesBytes['classifier.onnx']=160*1024*1024+1;},m=>{m.filesSha256['classifier.onnx']='bad';},m=>{m.threshold=.4;},m=>{m.cropContext=.12;},m=>{m.precision='float16';},m=>{m.classifierResolution=128;},m=>{m.detectorArchitecture='unknown';},m=>{m.ioContracts['detector.onnx'].outputs.output0=[1,5,8400];},m=>{m.ioContracts['classifier.onnx'].inputs.images=[1,3,224,224];},m=>{m.sourceEvidence!.classifier.checkpointSha256='';},m=>{m.sourceEvidence!.testOpened=true;}];
  for(const mutate of cases){const m=fixture();mutate(m);expect(()=>validateManifest(m,'compact-v10')).toThrow();}
 });
 it('comparesloadedIO metadata against actual selected profile',()=>{
  const m=fixture(),session={inputNames:['images'],outputNames:['logits'],inputMetadata:[{name:'images',isTensor:true,type:'float32',shape:['batch',3,224,224]}],outputMetadata:[{name:'logits',isTensor:true,type:'float32',shape:['batch',42]}]};
  expect(()=>validateSessionIO(session,m,'classifier.onnx')).not.toThrow();
  session.inputMetadata[0].shape=['batch',3,192,192];expect(()=>validateSessionIO(session,m,'classifier.onnx')).toThrow();
 });
});
