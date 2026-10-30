import { useEffect, useRef, useState } from 'react';
import { preparePhoto } from '../photo';
import { TileFace } from '../TileFace';
import { tileLabel } from './labels';
import './benchmark.css';
interface Result { requestedProvider:string; decodeMs:number; load:{totalLoadMs:number}; runs:{totalMs:number;detectorForwardMs:number;classifierForwardMs:number;allTilesMeetAcceptThreshold:boolean;tiles:{classId:string;score:number}[]}[] }
export default function BrowserBenchmark(){
  const [provider,setProvider]=useState('wasm'),[busy,setBusy]=useState(false),[ready,setReady]=useState(false),[status,setStatus]=useState('尚未下载模型'),[error,setError]=useState('');
  const [file,setFile]=useState<File|null>(null),[preview,setPreview]=useState(''),[result,setResult]=useState<Result|null>(null),[repeats,setRepeats]=useState(3);
  const worker=useRef<Worker|null>(null),timer=useRef<number|undefined>(undefined),url=useRef(''),alive=useRef(true);
  const stop=()=>{worker.current?.terminate();worker.current=null;clearTimeout(timer.current);setBusy(false);setReady(false);setStatus('已释放本地模型，可重新加载');};
  useEffect(()=>{alive.current=true;return()=>{alive.current=false;worker.current?.terminate();clearTimeout(timer.current);if(url.current)URL.revokeObjectURL(url.current);};},[]);
  const timeout=(ms:number)=>{clearTimeout(timer.current);timer.current=window.setTimeout(()=>{stop();setError('测试超时，已释放模型。可尝试另一运行方式或返回云端识别。');},ms);};
  const load=()=>{
    stop();setBusy(true);setError('');setResult(null);setStatus('准备下载约119 MB模型和约25.5 MB运行库…');timeout(480000);
    const w=new Worker(new URL('./local.worker.ts',import.meta.url),{type:'module'});worker.current=w;
    w.onmessage=event=>{if(!alive.current||worker.current!==w)return;const{type,value}=event.data;if(type==='status')setStatus(value);else if(type==='ready'){clearTimeout(timer.current);setBusy(false);setReady(true);setStatus('模型已就绪，照片只在本机处理');}else if(type==='result'){clearTimeout(timer.current);setBusy(false);setResult(value);setStatus('测试完成，可更换照片继续测试');}else if(type==='error'){stop();setError(String(value));}};
    w.onerror=()=>{if(!alive.current||worker.current!==w)return;stop();setError('本地运行环境初始化失败，请改选另一方式。');};w.postMessage({type:'load',provider});
  };
  const choose=async(next?:File)=>{if(!next)return;setError('');setResult(null);setBusy(true);try{const photo=await preparePhoto(next);if(!alive.current){URL.revokeObjectURL(photo.url);return;}if(url.current)URL.revokeObjectURL(url.current);url.current=photo.url;setPreview(photo.url);setFile(photo.file);}catch(e){setError(e instanceof Error?e.message:'照片处理失败');}finally{if(alive.current)setBusy(false);}};
  const run=()=>{if(!file||!ready)return;setBusy(true);setError('');setResult(null);timeout(180000);worker.current?.postMessage({type:'run',file,repeats});};
  const download=()=>{const data=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=data;a.download='mahjong-local-benchmark.json';a.click();setTimeout(()=>URL.revokeObjectURL(data),1000);};
  return <main className="bench-shell"><a className="bench-back" href="/">← 返回拍照识牌</a><header><span className="eyebrow">设备性能测试</span><h1>在这台设备上识牌</h1><p>照片不上传，模型在浏览器内运行。这是实验入口，识别结果仍需核对。</p></header>
    <section className="bench-card"><h2>1 · 加载本地模型</h2><p>首次需要约145 MB下载（模型约119 MB、运行库约25.5 MB），建议使用Wi-Fi；手机内存不足时可能无法运行。加载后可连续测试。</p><label>运行方式<select aria-label="运行方式" value={provider} disabled={busy||ready} onChange={e=>setProvider(e.target.value)}><option value="wasm">CPU / WASM（兼容优先）</option><option value="webgpu">WebGPU（设备支持时）</option></select></label><button className="primary-button" disabled={busy||ready} onClick={load}>下载并加载模型</button>{(busy||ready)&&<button className="secondary-button" onClick={stop}>{busy?'取消并释放模型':'释放模型 / 切换方式'}</button>}</section>
    <p className="bench-status" role="status">{status}</p>{error&&<p className="error-notice" role="alert">{error}</p>}
    <section className="bench-card"><h2>2 · 选择照片并测试</h2><label className="bench-file">选择手牌照片<input type="file" accept="image/jpeg,image/png,image/webp" aria-label="本地测试照片" disabled={busy} onChange={e=>void choose(e.target.files?.[0])}/></label>{preview&&<img className="bench-preview" src={preview} alt="仅在本机处理的测试照片"/>}<label>连续次数<select aria-label="连续次数" value={repeats} disabled={busy} onChange={e=>setRepeats(Number(e.target.value))}><option value={1}>1次（快速检查）</option><option value={3}>3次（区分首次与后续）</option></select></label><button className="primary-button" disabled={!ready||!file||busy} onClick={run}>开始本地测试</button></section>
    {result&&<section className="bench-card"><h2>测试结果</h2><p>模型加载：{(result.load.totalLoadMs/1000).toFixed(1)}秒。下表为识别流水线耗时，不含下载和选图。</p><div className="bench-table"><table><thead><tr><th>次数</th><th>整图识别</th><th>定位</th><th>分类</th></tr></thead><tbody>{result.runs.map((r,i)=><tr key={i}><td>{i===0?'首次':`后续${i}`}</td><td>{(r.totalMs/1000).toFixed(2)}秒</td><td>{(r.detectorForwardMs/1000).toFixed(2)}秒</td><td>{(r.classifierForwardMs/1000).toFixed(2)}秒</td></tr>)}</tbody></table></div><p>识别到 {result.runs.at(-1)?.tiles.length} 张，以下为实验结果：</p>{!result.runs.at(-1)?.allTilesMeetAcceptThreshold&&<p className="error-notice">部分牌面置信分数较低或未识别到牌，请核对结果或换一张更清晰的照片。</p>}<div className="bench-tiles">{result.runs.at(-1)?.tiles.map((tile,i)=><span key={i} role="img" aria-label={`${tileLabel(tile.classId)}，置信分数${Math.round(tile.score*100)}%`}><TileFace classId={tile.classId} label={tileLabel(tile.classId)} small/><small>{Math.round(tile.score*100)}%</small></span>)}</div><button className="secondary-button" onClick={download}>保存测试结果</button><p>保存的文件含设备、耗时和牌面结果，不含照片。浏览器的缩放方式与服务器不同，尚不能视为同等准确率。</p></section>}
    <footer><p>真机速度以您当前设备实测为准；电脑上的手机尺寸预览不能代表手机性能。</p></footer>
  </main>;
}
