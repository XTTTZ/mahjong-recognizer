#!/usr/bin/env node
/** Explicit, local-photo-only opt-in smoke for the optional /benchmark route. */
import { chromium } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

function argumentsFrom(argv) {
  const options={baseUrl:'http://127.0.0.1:4174',provider:'wasm',browser:'chromium',headed:false,repeats:3};
  const names={'--base-url':'baseUrl','--image':'image','--provider':'provider','--out':'out','--browser':'browser','--repeats':'repeats'};
  for(let i=0;i<argv.length;i++){
    if(argv[i]==='--headed'){options.headed=true;continue;}
    if(argv[i]==='--help'){console.log('node infra/portable/probe_browser.mjs --image PHOTO --out RESULT.json [--base-url http://127.0.0.1:4174] [--provider wasm|webgpu] [--browser chromium|chrome] [--headed] [--repeats 1|3]');process.exit(0);}
    const key=names[argv[i]];if(!key||!argv[i+1]||argv[i+1].startsWith('--'))throw Error(`Unknown or missing argument: ${argv[i]}`);
    options[key]=argv[++i];
  }
  options.repeats=Number(options.repeats);
  if(!options.image||!options.out)throw Error('--image and --out are required; no default private photo is used');
  if(!['wasm','webgpu'].includes(options.provider)||!['chromium','chrome'].includes(options.browser)||![1,3].includes(options.repeats))throw Error('Invalid provider/browser/repeats');
  const base=new URL(options.baseUrl);
  if(base.username||base.password||base.search||base.hash||!['/','/benchmark','/benchmark/'].includes(base.pathname))throw Error('--base-url must be an origin or its /benchmark route without credentials/query');
  if(base.protocol!=='https:'&&!(base.protocol==='http:'&&['localhost','127.0.0.1','[::1]'].includes(base.hostname)))throw Error('Use HTTPS or a loopback HTTP origin');
  return {...options,url:new URL('/benchmark',base.origin).href,origin:base.origin};
}
const opt=argumentsFrom(process.argv.slice(2));
const out=path.resolve(opt.out),receiptFile=`${out}.probe.json`;
for(const file of [out,receiptFile]){try{await fs.access(file);throw Error(`Refusing to overwrite ${file}`);}catch(e){if(e.code!=='ENOENT')throw e;}}
const imageStat=await fs.stat(opt.image);if(!imageStat.isFile()||imageStat.size<1||imageStat.size>40*1024*1024)throw Error('Choose a nonempty photo of at most 40 MiB');
await fs.mkdir(path.dirname(out),{recursive:true});
const browser=await chromium.launch({...(opt.browser==='chrome'?{channel:'chrome'}:{}),headless:!opt.headed});
const page=await browser.newPage({viewport:{width:1280,height:900},acceptDownloads:true});
const requests=[],responses=[],responseTasks=[],blockedRequests=[],errors=[];let inferenceRequestsBeforeOptIn=0,optedIn=false,workerClosed=0;
// Only the benchmark document, local static runtime and three fixed model paths
// may be requested. Abort API calls, non-GET requests and external telemetry
// before they can send data, even if a future page change accidentally adds one.
await page.route('**/*',async route=>{
  const request=route.request(),url=new URL(request.url());
  const allowed=url.origin===opt.origin&&request.method()==='GET'&&(
    ['/benchmark','/benchmark/','/icon.svg','/favicon.ico'].includes(url.pathname)||
    url.pathname.startsWith('/assets/')||url.pathname.startsWith('/ort/')||
    ['/models/portable-v1/manifest.json','/models/portable-v1/detector.onnx','/models/portable-v1/classifier.onnx'].includes(url.pathname));
  if(!allowed){blockedRequests.push({method:request.method(),url:request.url()});await route.abort('blockedbyclient');return;}
  requests.push({method:request.method(),url:request.url()});
  if(!optedIn&&(/\/models\/|\/ort\/|\/assets\/local.worker-/.test(url.pathname)))inferenceRequestsBeforeOptIn++;
  await route.continue();
});
page.on('worker',worker=>worker.on('close',()=>workerClosed++));
page.on('pageerror',error=>errors.push(error.message));
page.on('response',response=>{
  responseTasks.push((async()=>{
    const headers=await response.allHeaders(),safeHeaders={};
    for(const name of ['content-type','content-length','content-encoding','content-security-policy','cross-origin-embedder-policy','cross-origin-opener-policy','cross-origin-resource-policy','cache-control'])if(headers[name])safeHeaders[name]=headers[name];
    responses.push({url:response.url(),status:response.status(),headers:safeHeaders});
  })());
});
async function waitFor(condition,timeout,label){
  const start=Date.now();let lastLog=0;
  while(Date.now()-start<timeout){
    if(blockedRequests.length||errors.length)throw Error('Unexpected network request or browser error; inspect the probe receipt');
    if(await page.getByRole('alert').count())throw Error(await page.getByRole('alert').first().innerText());
    if(await condition())return;
    if(Date.now()-lastLog>15000){console.log(`${label}: ${await page.getByRole('status').innerText()}`);lastLog=Date.now();}
    await page.waitForTimeout(200);
  }
  throw Error(`${label} timed out`);
}
const receipt={schemaVersion:1,url:opt.url,requestedProvider:opt.provider,requestedBrowser:opt.browser,headed:opt.headed,viewport:{width:1280,height:900},mobileEmulation:false,physicalPhoneVerified:false,userSuppliedLocalPhoto:true,photoUploaded:false,repeats:opt.repeats,requests,responses,blockedRequests,errors};
try{
  await page.goto(opt.url,{waitUntil:'networkidle',timeout:60000});
  if(inferenceRequestsBeforeOptIn)throw Error('Inference resources were requested before explicit opt-in');
  await page.getByLabel('运行方式').selectOption(opt.provider);
  optedIn=true;await page.getByRole('button',{name:'下载并加载模型'}).click();
  await waitFor(async()=>((await page.getByRole('status').innerText()).includes('模型已就绪')),480000,'Model load');
  await page.getByLabel('本地测试照片').setInputFiles(opt.image);
  await page.getByLabel('连续次数').selectOption(String(opt.repeats));
  await page.getByRole('button',{name:'开始本地测试'}).click();
  await waitFor(async()=>await page.getByRole('button',{name:'保存测试结果'}).count()>0,180000,'Inference');
  const pending=page.waitForEvent('download',{timeout:10000});await page.getByRole('button',{name:'保存测试结果'}).click();
  const download=await pending,downloadPath=await download.path();
  if(!downloadPath)throw Error('Browser download did not produce a local result');
  const bytes=await fs.readFile(downloadPath),report=JSON.parse(bytes.toString('utf8'));
  if(report.photoUploaded!==false||report.requestedProvider!==opt.provider||report.runs.length!==opt.repeats)throw Error('Downloaded report does not match the requested local-only run');
  await fs.writeFile(out,bytes,{flag:'wx'});
  await page.getByRole('button',{name:'释放模型 / 切换方式'}).click();
  await page.waitForTimeout(100);
  Object.assign(receipt,{state:'passed',inferenceRequestsBeforeOptIn,modelReleaseClosedWorker:workerClosed>0,reportSha256:createHash('sha256').update(bytes).digest('hex'),browserVersion:browser.version()});
  if(!workerClosed||blockedRequests.length||errors.length)throw Error('Resource release or browser/network guard failed');
  console.log(JSON.stringify({state:'passed',out:opt.out,provider:report.requestedProvider,device:report.device,totalMs:report.runs.map(run=>run.totalMs),retainedTiles:report.runs.map(run=>run.tiles.length),allTilesMeetAcceptThreshold:report.runs.map(run=>run.allTilesMeetAcceptThreshold),phoneHardwareVerified:false}));
}catch(error){Object.assign(receipt,{state:'failed',error:String(error)});process.exitCode=1;console.error(String(error));}
finally{await Promise.allSettled(responseTasks);await browser.close();await fs.writeFile(receiptFile,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});}
