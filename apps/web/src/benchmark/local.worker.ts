import type { InferenceSession } from 'onnxruntime-web';
let ort: typeof import('onnxruntime-web');
import { classify, classifierPixels, nms, tensorPixels } from './pipeline';
import { headPoints, MODEL_FILES, modelInfo, profileDefinition, profilePath, validateManifest, validateSessionIO, type Manifest, type ProfileId, type ModelFile } from './profiles';
let detector:InferenceSession|null=null,classifier:InferenceSession|null=null,manifest:Manifest|null=null;
let profileId:ProfileId='portable-v1';
let provider='',loadStats:Record<string,unknown>={},deviceInfo:Record<string,unknown>={},busy=false;
const send=(type:string,value:unknown)=>postMessage({type,value});
async function digestMatches(bytes: Uint8Array<ArrayBuffer>, expected: string): Promise<boolean> {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('') === expected;
}

async function download(name: ModelFile, model: Manifest): Promise<Uint8Array<ArrayBuffer>> {
  const url = profilePath(profileId, name);
  const expected = model.filesBytes[name];
  const hash = model.filesSha256[name];
  const label = name === 'detector.onnx' ? '定位模型' : '分类模型';
  // Some mobile browsers evict large HTTP responses even with immutable headers.
  // Only the compact, explicitly requested model is retained across visits.
  const key = new URL(url, self.location.origin);
  key.searchParams.set('sha256', hash);
  let cache: Cache | undefined;
  if (profileId === 'compact-v10') {
    try {
      cache = await caches.open('mahjong-compact-weights-v1');
      const stored = await cache.match(key.href);
      if (stored) {
        send('status', `正在校验已保存的${label}…`);
        const bytes = new Uint8Array(await stored.arrayBuffer());
        if (bytes.byteLength === expected && await digestMatches(bytes, hash)) return bytes;
        await cache.delete(key.href);
      }
    } catch { /* Private browsing or exhausted storage still allows an in-memory session. */ }
  }
  send('status', `下载${label}…`);
  const response = await fetch(url, { credentials: 'omit' });
  if (!response.ok || !response.body) throw new Error('模型下载失败，请稍后再试');
  const bytes = new Uint8Array(expected);
  const reader = response.body.getReader();
  let offset = 0;
  let lastUpdate = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (offset + value.length > expected) throw new Error('模型文件大小不符');
      bytes.set(value, offset);
      offset += value.length;
      if (performance.now() - lastUpdate > 250) {
        send('status', `下载${label} ${Math.round(offset / expected * 100)}%`);
        lastUpdate = performance.now();
      }
    }
  } catch (error) {
    await reader.cancel();
    throw error;
  }
  if (offset !== expected || !await digestMatches(bytes, hash)) throw new Error('模型校验失败，请重新准备本机识别');
  if (cache) {
    try { await cache.put(key.href, new Response(bytes)); } catch { /* Storage is optional. */ }
  }
  return bytes;
}
async function release(){
  const sessions=[detector,classifier];detector=null;classifier=null;manifest=null;
  await Promise.allSettled(sessions.filter(s=>s!==null).map(s=>s.release()));
}
async function load(requested:string,requestedProfile:unknown){
  const started=performance.now();provider=requested;profileId=profileDefinition(requestedProfile).id;
  await release();deviceInfo={};
  if(!['wasm','webgpu'].includes(requested))throw new Error('运行方式无效');
  if(typeof OffscreenCanvas==='undefined'||typeof createImageBitmap!=='function')throw new Error('当前浏览器不支持后台图片处理，请更新浏览器');
  if(requested==='webgpu'&&!('gpu'in navigator))throw new Error('当前浏览器未提供 WebGPU，可改选 CPU/WASM');
  ort = requested === 'wasm' ? await import('onnxruntime-web/wasm') : await import('onnxruntime-web/webgpu');
  if(requested==='webgpu'){
    const gpu=(navigator as unknown as {gpu:{requestAdapter:(options:Record<string,string>)=>Promise<unknown>}}).gpu;
    const adapter=await gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)throw new Error('没有可用的 WebGPU 设备，可改选 CPU/WASM');
    ort.env.webgpu.adapter=adapter as typeof ort.env.webgpu.adapter;
    const a=adapter as {info?:{vendor?:string;architecture?:string;device?:string;description?:string;isFallbackAdapter?:boolean};isFallbackAdapter?:boolean};
    deviceInfo={vendor:a.info?.vendor??'',architecture:a.info?.architecture??'',device:a.info?.device??'',description:a.info?.description??'',isFallbackAdapter:a.isFallbackAdapter??a.info?.isFallbackAdapter??null};
  }
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths=new URL(requested === 'wasm' ? '/ort/1.23.2/' : '/ort/',self.location.origin).href;
  ort.env.logLevel='error';
  const response=await fetch(profilePath(profileId,'manifest.json'),{credentials:'omit'});if(!response.ok)throw new Error('所选模型尚未就绪，可稍后重试或手动选择原始对照');
  const m=validateManifest(await response.json(),profileId);
  manifest=m;send('manifest',modelInfo(m,profileId));
  let downloaded=0,compiled=0;
  for(const name of MODEL_FILES){
    const begin=performance.now();const bytes=await download(name,m);downloaded+=performance.now()-begin;
    send('status',`正在初始化${name==='detector.onnx'?'定位':'分类'}模型…`);const init=performance.now();
    const session=await ort.InferenceSession.create(bytes,{executionProviders:[requested],graphOptimizationLevel:'all'});compiled+=performance.now()-init;
    if(name==='detector.onnx')detector=session;else classifier=session;
    validateSessionIO(session,m,name);
  }
  loadStats={...modelInfo(m,profileId),downloadAndHashMs:downloaded,sessionCreateMs:compiled,totalLoadMs:performance.now()-started,modelBytes:Object.values(m.filesBytes).reduce((a,b)=>a+b,0)};
  send('ready',loadStats);
}
async function run(file:Blob,repeats:number){
  if(!detector||!classifier||!manifest)throw new Error('请先加载模型');
  const decode=performance.now(),image=await createImageBitmap(file),decodeMs=performance.now()-decode;
  if(image.width*image.height>2560*2560){image.close();throw new Error('请先缩小图片');}
  const inputImage={width:image.width,height:image.height,bytes:file.size},runs=[];
  try{for(let iteration=0;iteration<Math.min(3,Math.max(1,repeats));iteration++){
    send('status',`本地识别 ${iteration+1}/${repeats}，可随时取消`);const start=performance.now();
    const resolution=manifest.resolution,size=manifest.classifierResolution;
    const pixels=tensorPixels(image,resolution),input=new ort.Tensor('float32',pixels,[1,3,resolution,resolution]);const pre=performance.now();
    let output:InferenceSession.ReturnType|undefined;
    let boxes:ReturnType<typeof nms>,detEnd:number;
    try{output=await detector.run({images:input});detEnd=performance.now();
      if(output.output0.type!=='float32'||String(output.output0.dims)!==`1,5,${headPoints(resolution)}`)throw new Error('检测输出维度不符');
      boxes=nms(output.output0.data as Float32Array,image.width,image.height,resolution);
    }finally{input.dispose();if(output)Object.values(output).forEach(t=>t.dispose());}
    const post=performance.now();let cropMs=0,clsMs=0;const tiles:{bbox:[number,number,number,number];score:number;classId:string}[]=[];
    for(let first=0;first<boxes.length;first+=4){
      const part=boxes.slice(first,first+4),t=performance.now(),batch=classifierPixels(image,part,size);
      const tensor=new ort.Tensor('float32',batch,[part.length,3,size,size]),prepared=performance.now();
      let out:InferenceSession.ReturnType|undefined;
      try{out=await classifier.run({images:tensor});const end=performance.now();cropMs+=prepared-t;clsMs+=end-prepared;
        if(out.logits.type!=='float32'||String(out.logits.dims)!==`${part.length},42`)throw new Error('分类输出维度不符');
        const logits=out.logits.data as Float32Array;
        part.forEach((box,i)=>{const c=classify(logits,i*42),score=box.score*c.probability;if(score>.35)tiles.push({...box,score,classId:manifest!.classes[String(c.index)]});});
      }finally{tensor.dispose();if(out)Object.values(out).forEach(t=>t.dispose());}
    }
    runs.push({totalMs:performance.now()-start,detectorPreprocessMs:pre-start,detectorForwardMs:detEnd-pre,detectorPostprocessMs:post-detEnd,cropPreprocessMs:cropMs,classifierForwardMs:clsMs,detectedFaces:boxes.length,allTilesMeetAcceptThreshold:tiles.length>0&&tiles.every(tile=>tile.score>=.7),tiles});
  }}finally{image.close();}
  send('result',{createdAt:new Date().toISOString(),profileId,detectorResolution:manifest.resolution,classifierResolution:manifest.classifierResolution,modelBytes:loadStats.modelBytes,modelVersion:manifest.version,modelSha256:manifest.filesSha256,
    requestedProvider:provider,providerNote:provider==='webgpu'?'WebGPU requested; unsupported operators may fall back to WASM':'CPU/WASM single thread',
    runtime:'onnxruntime-web 1.23.2',wasmThreads:1,classifierBatch:4,userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,device:deviceInfo,
    crossOriginIsolated:self.crossOriginIsolated,preprocessing:'Browser Canvas bilinear, JPEG .94 longest-edge2560; not bit-identical to OpenCV/PIL',
    inputImage,threshold:.35,acceptThreshold:.7,decodeMs,load:loadStats,runs,physicalPhoneVerified:false,photoUploaded:false});
}
self.onmessage=async(event:MessageEvent)=>{if(busy){send('error','本地测试正在运行');return;}busy=true;try{if(event.data.type==='load')await load(event.data.provider,event.data.profileId);else if(event.data.type==='run')await run(event.data.file,event.data.repeats);}catch(error){if(event.data.type==='load')await release();send('error',error instanceof Error?error.message:'当前浏览器无法完成本地推理');}finally{busy=false;}};
