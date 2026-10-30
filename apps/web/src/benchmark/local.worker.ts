import * as ort from 'onnxruntime-web/webgpu';
import { classify, nms, tensorPixels } from './pipeline';
interface Manifest { version:string; classes:Record<string,string>; filesBytes:Record<string,number>; filesSha256:Record<string,string>; resolution:number; classifierResolution:number; cropContext:number; detectorThreshold:number; nmsIou:number; detectorNmsMaxDetections:number; maxDetections:number; maxClassifierBatch:number; threshold:number; acceptThreshold:number; }
let detector:ort.InferenceSession|null=null,classifier:ort.InferenceSession|null=null,manifest:Manifest|null=null;
let provider='',loadStats:Record<string,unknown>={},deviceInfo:Record<string,unknown>={},busy=false;
const send=(type:string,value:unknown)=>postMessage({type,value});
async function download(name:string,m:Manifest):Promise<Uint8Array>{
  send('status',`下载${name==='detector.onnx'?'定位':'分类'}模型…`);
  const response=await fetch(`/models/portable-v1/${name}`,{credentials:'omit'});
  if(!response.ok||!response.body)throw new Error('模型下载失败，请稍后再试');
  const expected=m.filesBytes[name];if(!Number.isInteger(expected)||expected<1||expected>160*1024*1024)throw new Error('模型清单大小无效');
  const result=new Uint8Array(expected),reader=response.body.getReader();let pos=0,last=0;
  try{while(true){const{value,done}=await reader.read();if(done)break;if(pos+value.length>expected)throw new Error('模型文件大小不符');result.set(value,pos);pos+=value.length;
    if(performance.now()-last>250){send('status',`下载${name==='detector.onnx'?'定位':'分类'}模型 ${Math.round(pos/expected*100)}%`);last=performance.now();}}}
  catch(error){await reader.cancel();throw error;}
  if(pos!==expected)throw new Error('模型文件下载不完整');
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',result))].map(v=>v.toString(16).padStart(2,'0')).join('');
  if(hash!==m.filesSha256[name])throw new Error('模型校验失败，请清除浏览器缓存后重试');
  return result;
}
async function load(requested:string){
  const started=performance.now();provider=requested;
  if(!['wasm','webgpu'].includes(requested))throw new Error('运行方式无效');
  if(typeof OffscreenCanvas==='undefined'||typeof createImageBitmap!=='function')throw new Error('当前浏览器不支持后台图片处理，请更新浏览器');
  if(requested==='webgpu'&&!('gpu'in navigator))throw new Error('当前浏览器未提供 WebGPU，可改选 CPU/WASM');
  if(requested==='webgpu'){
    const gpu=(navigator as unknown as {gpu:{requestAdapter:(options:Record<string,string>)=>Promise<unknown>}}).gpu;
    const adapter=await gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)throw new Error('没有可用的 WebGPU 设备，可改选 CPU/WASM');
    ort.env.webgpu.adapter=adapter as typeof ort.env.webgpu.adapter;
    const a=adapter as {info?:{vendor?:string;architecture?:string;device?:string;description?:string;isFallbackAdapter?:boolean};isFallbackAdapter?:boolean};
    deviceInfo={vendor:a.info?.vendor??'',architecture:a.info?.architecture??'',device:a.info?.device??'',description:a.info?.description??'',isFallbackAdapter:a.isFallbackAdapter??a.info?.isFallbackAdapter??null};
  }
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths=new URL('/ort/',self.location.origin).href;
  ort.env.logLevel='error';
  const response=await fetch('/models/portable-v1/manifest.json',{credentials:'omit'});if(!response.ok)throw new Error('模型清单暂不可用');
  const m:Manifest=await response.json();
  const tokens=Array.from({length:9},(_,i)=>['m','p','s'].map(s=>`${i+1}${s}`)).flat().concat(Array.from({length:7},(_,i)=>`${i+1}z`),Array.from({length:8},(_,i)=>`${i+1}h`));
  if(m.resolution!==960||m.classifierResolution!==320||m.cropContext!==.08||m.detectorThreshold!==.05||m.nmsIou!==.6||m.detectorNmsMaxDetections!==300||m.maxDetections!==200||m.maxClassifierBatch!==32||m.threshold!==.35||m.acceptThreshold!==.7||Object.keys(m.classes).length!==42||new Set(Object.values(m.classes)).size!==42||Array.from({length:42},(_,i)=>m.classes[String(i)]).some(t=>!tokens.includes(t)))throw new Error('模型配置不受支持');
  manifest=m;
  let downloaded=0,compiled=0;
  for(const name of ['detector.onnx','classifier.onnx']){
    const begin=performance.now();const bytes=await download(name,m);downloaded+=performance.now()-begin;
    send('status',`正在初始化${name==='detector.onnx'?'定位':'分类'}模型…`);const init=performance.now();
    const session=await ort.InferenceSession.create(bytes,{executionProviders:[requested],graphOptimizationLevel:'all'});compiled+=performance.now()-init;
    if(name==='detector.onnx')detector=session;else classifier=session;
  }
  loadStats={downloadAndHashMs:downloaded,sessionCreateMs:compiled,totalLoadMs:performance.now()-started,modelBytes:Object.values(m.filesBytes).reduce((a,b)=>a+b,0)};
  send('ready',loadStats);
}
async function run(file:Blob,repeats:number){
  if(!detector||!classifier||!manifest)throw new Error('请先加载模型');
  const decode=performance.now(),image=await createImageBitmap(file),decodeMs=performance.now()-decode;
  if(image.width*image.height>2560*2560){image.close();throw new Error('请先缩小图片');}
  const inputImage={width:image.width,height:image.height,bytes:file.size},runs=[];
  try{for(let iteration=0;iteration<Math.min(3,Math.max(1,repeats));iteration++){
    send('status',`本地识别 ${iteration+1}/${repeats}，可随时取消`);const start=performance.now();
    const pixels=tensorPixels(image,960),input=new ort.Tensor('float32',pixels,[1,3,960,960]);const pre=performance.now();
    let output:ort.InferenceSession.ReturnType|undefined;
    let boxes:ReturnType<typeof nms>,detEnd:number;
    try{output=await detector.run({images:input});detEnd=performance.now();
      if(output.output0.type!=='float32'||String(output.output0.dims)!=='1,5,18900')throw new Error('检测输出维度不符');
      boxes=nms(output.output0.data as Float32Array,image.width,image.height);
    }finally{input.dispose();if(output)Object.values(output).forEach(t=>t.dispose());}
    const post=performance.now();let cropMs=0,clsMs=0;const tiles:{bbox:[number,number,number,number];score:number;classId:string}[]=[];
    for(let first=0;first<boxes.length;first+=4){
      const part=boxes.slice(first,first+4),t=performance.now(),batch=new Float32Array(part.length*3*320*320);
      part.forEach((box,i)=>batch.set(tensorPixels(image,320,box),i*3*320*320));
      const tensor=new ort.Tensor('float32',batch,[part.length,3,320,320]),prepared=performance.now();
      let out:ort.InferenceSession.ReturnType|undefined;
      try{out=await classifier.run({images:tensor});const end=performance.now();cropMs+=prepared-t;clsMs+=end-prepared;
        if(out.logits.type!=='float32'||String(out.logits.dims)!==`${part.length},42`)throw new Error('分类输出维度不符');
        const logits=out.logits.data as Float32Array;
        part.forEach((box,i)=>{const c=classify(logits,i*42),score=box.score*c.probability;if(score>.35)tiles.push({...box,score,classId:manifest!.classes[String(c.index)]});});
      }finally{tensor.dispose();if(out)Object.values(out).forEach(t=>t.dispose());}
    }
    runs.push({totalMs:performance.now()-start,detectorPreprocessMs:pre-start,detectorForwardMs:detEnd-pre,detectorPostprocessMs:post-detEnd,cropPreprocessMs:cropMs,classifierForwardMs:clsMs,detectedFaces:boxes.length,allTilesMeetAcceptThreshold:tiles.length>0&&tiles.every(tile=>tile.score>=.7),tiles});
  }}finally{image.close();}
  send('result',{createdAt:new Date().toISOString(),modelVersion:manifest.version,modelSha256:manifest.filesSha256,
    requestedProvider:provider,providerNote:provider==='webgpu'?'WebGPU requested; unsupported operators may fall back to WASM':'CPU/WASM single thread',
    runtime:'onnxruntime-web 1.23.2',wasmThreads:1,classifierBatch:4,userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,device:deviceInfo,
    crossOriginIsolated:self.crossOriginIsolated,preprocessing:'Browser Canvas bilinear, JPEG .94 longest-edge2560; not bit-identical to OpenCV/PIL',
    inputImage,threshold:.35,acceptThreshold:.7,decodeMs,load:loadStats,runs,physicalPhoneVerified:false,photoUploaded:false});
}
self.onmessage=async(event:MessageEvent)=>{if(busy){send('error','本地测试正在运行');return;}busy=true;try{if(event.data.type==='load')await load(event.data.provider);else if(event.data.type==='run')await run(event.data.file,event.data.repeats);}catch(error){send('error',error instanceof Error?error.message:'当前浏览器无法完成本地推理');}finally{busy=false;}};
