import { expect, test } from '@playwright/test';

// Cold-route checks intentionally never opt into model loading or inference.
for (const pathname of ['/', '/benchmark']) {
  test(`${pathname}: local inference resources stay unloaded before explicit opt-in`, async ({ page }) => {
    const localResources: string[] = [];
    const gpuRequests: string[] = [];
    let warmups = 0;
    page.on('request', request => {
      const url = new URL(request.url());
      if (/\/models\/|\/ort\/|local[._-]worker|onnxruntime[-_]web/i.test(url.pathname)) localResources.push(url.pathname);
      if (url.pathname === '/api/recognize' || url.hostname.endsWith('.modal.run')) gpuRequests.push(request.url());
    });
    await page.route('**/api/warmup', route => { warmups++; return route.fulfill({ json: { ready: true } }); });
    await page.route('**/api/health', route => route.fulfill({ json: { configured: true, inferenceProbed: false } }));
    // These guards also prevent accidental heavy downloads if lazy loading regresses.
    await page.route('**/models/**', route => route.abort());
    await page.route('**/ort/**', route => route.abort());
    await page.route('**/api/recognize', route => route.abort());
    await page.route('https://**.modal.run/**', route => route.abort());
    await page.goto(pathname);
    if (pathname === '/') {
      await expect(page.getByRole('button', { name: '拍一张', exact: true })).toBeVisible();
      await expect(page.getByRole('link', { name: '设备性能测试', exact: true })).toHaveAttribute('href', '/benchmark');
    } else {
      await expect(page.getByRole('button', { name: '下载并加载模型', exact: true })).toBeEnabled();
      await expect(page.getByRole('button', { name: '开始本地测试', exact: true })).toBeDisabled();
    }
    await page.waitForLoadState('networkidle');
    expect(localResources).toEqual([]);
    expect(gpuRequests).toEqual([]);
    if (pathname === '/benchmark') expect(warmups).toBe(0);
  });
}

for (const width of [320,390]) {
  test(`benchmark profile choice stays explicit at ${width}px`, async ({ page }) => {
    await page.setViewportSize({width,height:844});
    const remote: string[]=[];
    page.on('request',request=>{if(/\/api\/|\.modal\.run|\/models\//.test(request.url()))remote.push(request.url());});
    await page.route('**/api/**',route=>route.abort());
    await page.route('**/models/**',route=>route.abort());
    await page.addInitScript(()=>{
      const calls:{profileId:string;terminated:boolean}[]=[];
      Object.assign(window,{__profileCalls:calls});
      class InertWorker {
        onmessage:((event:{data:unknown})=>void)|null=null;onerror:null=null;
        index=-1;
        postMessage(value:{type:string;profileId:string}){
          if(value.type!=='load')throw new Error('Only inert loading is allowed');
          this.index=calls.length;calls.push({profileId:value.profileId,terminated:false});
          const compact=value.profileId==='compact-v10';
          setTimeout(()=>{
            this.onmessage?.({data:{type:'manifest',value:{profileId:value.profileId,version:'inert-fixture',modelBytes:compact?29_000_000:119_092_864,detectorResolution:compact?800:960,classifierResolution:compact?224:320}}});
            this.onmessage?.({data:{type:'ready',value:{totalLoadMs:10}}});
          },0);
        }
        terminate(){if(this.index>=0)calls[this.index].terminated=true;}
      }
      Object.defineProperty(window,'Worker',{value:InertWorker,configurable:true});
    });
    await page.goto('/benchmark');
    const choice=page.getByLabel('本地模型');
    await expect(choice).toHaveValue('compact-v10');
    await page.getByRole('button',{name:'下载并加载模型',exact:true}).click();
    await expect(page.getByRole('status')).toContainText('模型已就绪');
    await expect(page.getByTestId('model-download-size')).toContainText('29.0 MB');
    await expect(choice).toBeDisabled();
    await page.getByRole('button',{name:'释放模型 / 切换方式',exact:true}).click();
    await expect(choice).toBeEnabled();
    await choice.selectOption('portable-v1');
    await page.getByRole('button',{name:'下载并加载模型',exact:true}).click();
    await expect(page.getByTestId('model-download-size')).toContainText('119.1 MB');
    await page.getByRole('button',{name:'释放模型 / 切换方式',exact:true}).click();
    expect(await page.evaluate(()=>(window as unknown as {__profileCalls:unknown}).__profileCalls)).toEqual([
      {profileId:'compact-v10',terminated:true},{profileId:'portable-v1',terminated:true},
    ]);
    expect(remote).toEqual([]);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  });
}
