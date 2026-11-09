export type ProfileId = 'compact-v10' | 'portable-v1';
export type ModelFile = 'detector.onnx' | 'classifier.onnx';
export const MODEL_FILES: readonly ModelFile[] = ['detector.onnx', 'classifier.onnx'];
export const PROFILES: readonly {id:ProfileId;label:string;description:string;available:boolean}[] = [
  {id:'compact-v10',label:'轻量模型',description:'适合比较手机与 CPU 速度，识别结果请人工核对。',available:true},
  {id:'portable-v1',label:'原始模型（对照）',description:'原线上权重的浏览器对照，下载较大。',available:true},
];
export const DEFAULT_PROFILE = PROFILES.find(p=>p.available)!.id;
export function profileDefinition(id:unknown){
  const value=PROFILES.find(p=>p.id===id&&p.available);
  if(!value)throw new Error('本地模型选项不可用');
  return value;
}
export function profilePath(id:ProfileId,file:ModelFile|'manifest.json'){
  profileDefinition(id);
  if(file!=='manifest.json'&&!MODEL_FILES.includes(file))throw new Error('模型文件选项无效');
  return `/models/${id}/${file}`;
}
export function headPoints(resolution:number){
  if(![640,800,960].includes(resolution))throw new Error('检测分辨率不受支持');
  return [8,16,32].reduce((sum,stride)=>sum+(resolution/stride)**2,0);
}
interface IOContract {inputs:Record<string,(string|number)[]>;outputs:Record<string,(string|number)[]>;float32:boolean}
export interface Manifest {
  schemaVersion:number;format:string;version:string;ontology:string;precision:string;opset:number;
  researchOnly?:boolean;detectorArchitecture?:string;classifierArchitecture?:string;
  classes:Record<string,string>;filesBytes:Record<ModelFile,number>;filesSha256:Record<ModelFile,string>;
  resolution:number;classifierResolution:number;cropContext:number;detectorThreshold:number;nmsIou:number;
  detectorNmsMaxDetections:number;maxDetections:number;maxClassifierBatch:number;threshold:number;acceptThreshold:number;
  ioContracts:Record<ModelFile,IOContract>;sourceEvidence?:{detector:Record<string,unknown>;classifier:Record<string,unknown>;testOpened:boolean};
}
export interface ModelInfo {profileId:ProfileId;version:string;modelBytes:number;detectorResolution:number;classifierResolution:number}
const object=(value:unknown):value is Record<string,unknown>=>value!==null&&typeof value==='object'&&!Array.isArray(value);
const keys=(value:unknown,expected:readonly string[]):value is Record<string,unknown>=>object(value)&&Object.keys(value).length===expected.length&&expected.every(k=>Object.hasOwn(value,k));
const sha=(value:unknown)=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
function io(value:unknown,input:(string|number)[],outputName:string,output:(string|number)[]){
  if(!object(value)||value.float32!==true||!keys(value.inputs,['images'])||!keys(value.outputs,[outputName]))return false;
  return Array.isArray(value.inputs.images)&&String(value.inputs.images)===String(input)&&Array.isArray(value.outputs[outputName])&&String(value.outputs[outputName])===String(output);
}
export function validateManifest(value:unknown,profileId:ProfileId):Manifest{
  profileDefinition(profileId);
  if(!object(value))throw new Error('模型清单无效');
  const m=value as unknown as Manifest;
  if(m.schemaVersion!==1||m.ontology!=='faces42'||m.precision!=='float32'||m.opset!==17||typeof m.version!=='string'||!m.version||m.version.length>200||
    m.cropContext!==.08||m.detectorThreshold!==.05||m.nmsIou!==.6||m.detectorNmsMaxDetections!==300||m.maxDetections!==200||m.maxClassifierBatch!==32||m.threshold!==.35||m.acceptThreshold!==.7)throw new Error('模型配置不受支持');
  const resolution=m.resolution as number,size=m.classifierResolution as number;
  const points=headPoints(resolution);
  if(profileId==='portable-v1'){
    if(m.format!=='onnx-fp32-cascade'||resolution!==960||size!==320)throw new Error('原始对照模型配置不符');
  }else{
    if(m.format!=='research-onnx-fp32-cascade'||m.researchOnly!==true||!['yolov8n-generic','yolov8s-generic','yolo26s-generic'].includes(String(m.detectorArchitecture))||
      !(m.classifierArchitecture==='mobilenet-v3-large'&&[192,224].includes(size)||m.classifierArchitecture==='efficientnet-v2-s'&&size===320))throw new Error('轻量模型配置不受支持');
    const source=m.sourceEvidence;
    if(!object(source)||source.testOpened!==false||!(['detector','classifier'] as const).every(stage=>object(source[stage])&&['runSha256','checkpointSha256','signatureSha256'].every(k=>sha(source[stage][k]))))throw new Error('轻量模型来源校验信息不完整');
  }
  const tokens=Array.from({length:9},(_,i)=>['m','p','s'].map(s=>`${i+1}${s}`)).flat().concat(Array.from({length:7},(_,i)=>`${i+1}z`),Array.from({length:8},(_,i)=>`${i+1}h`));
  if(!keys(m.classes,Array.from({length:42},(_,i)=>String(i)))||new Set(Object.values(m.classes)).size!==42||Object.values(m.classes).some(t=>!tokens.includes(String(t))))throw new Error('模型牌种配置不符');
  if(!keys(m.filesBytes,MODEL_FILES)||!keys(m.filesSha256,MODEL_FILES)||!MODEL_FILES.every(name=>Number.isInteger(m.filesBytes[name])&&(m.filesBytes[name]as number)>0&&(m.filesBytes[name]as number)<=160*1024*1024&&sha(m.filesSha256[name])))throw new Error('模型文件校验信息不符');
  if(!keys(m.ioContracts,MODEL_FILES)||!io(m.ioContracts['detector.onnx'],[1,3,resolution,resolution],'output0',[1,5,points])||!io(m.ioContracts['classifier.onnx'],['batch',3,size,size],'logits',['batch',42]))throw new Error('模型输入输出配置不符');
  return value as unknown as Manifest;
}
export function modelInfo(manifest:Manifest,profileId:ProfileId):ModelInfo{
  return {profileId,version:manifest.version,modelBytes:MODEL_FILES.reduce((sum,name)=>sum+manifest.filesBytes[name],0),detectorResolution:manifest.resolution,classifierResolution:manifest.classifierResolution};
}
export const megabytes=(bytes:number)=>(bytes/1_000_000).toFixed(1);

interface SessionMetadata {
  readonly inputNames:readonly string[];readonly outputNames:readonly string[];
  readonly inputMetadata:readonly {readonly name:string;readonly isTensor:boolean;readonly type?:string;readonly shape?:readonly (string|number)[]}[];
  readonly outputMetadata:readonly {readonly name:string;readonly isTensor:boolean;readonly type?:string;readonly shape?:readonly (string|number)[]}[];
}
export function validateSessionIO(session:SessionMetadata,manifest:Manifest,name:ModelFile){
  const expected=manifest.ioContracts[name],input=session.inputMetadata,output=session.outputMetadata;
  const outputName=name==='detector.onnx'?'output0':'logits';
  if(String(session.inputNames)!=='images'||String(session.outputNames)!==outputName||input.length!==1||output.length!==1||
    !input[0].isTensor||!output[0].isTensor||input[0].name!=='images'||output[0].name!==outputName||input[0].type!=='float32'||output[0].type!=='float32'||
    String(input[0].shape)!==String(expected.inputs.images)||String(output[0].shape)!==String(expected.outputs[outputName]))throw new Error('已加载模型的输入输出配置不符');
}
