export interface Box { bbox: [number, number, number, number]; score: number }
/** Python/OpenCV export geometry uses round-to-even for resize dimensions. */
export function roundEven(value: number): number {
  const floor = Math.floor(value), fraction = value - floor;
  return fraction === .5 ? floor + (floor % 2) : Math.round(value);
}
export function detectorGeometry(width:number,height:number) {
  const scale=Math.min(960/width,960/height), resizedWidth=roundEven(width*scale), resizedHeight=roundEven(height*scale);
  return { gainX:resizedWidth/width, gainY:resizedHeight/height,
    padX:roundEven((960-resizedWidth)/2-.1),padY:roundEven((960-resizedHeight)/2-.1) };
}
export function nms(values: Float32Array, width: number, height: number): Box[] {
  if (values.length !== 5 * 18900 || values.some(v => !Number.isFinite(v))) throw new Error('检测输出格式不符');
  const count = 18900, boxes: Box[] = [];
  for (let i = 0; i < count; i++) if (values[count * 4 + i] > .05) {
    const w = values[count * 2 + i], h = values[count * 3 + i];
    boxes.push({ bbox: [values[i] - w / 2, values[count + i] - h / 2, w, h], score: values[count * 4 + i] });
  }
  boxes.sort((a, b) => b.score - a.score);
  const kept: Box[] = [];
  const overlap = (a: Box, b: Box) => {
    const [x,y,w,h] = a.bbox, [u,v,s,t] = b.bbox;
    const area = Math.max(0, Math.min(x+w,u+s)-Math.max(x,u))*Math.max(0, Math.min(y+h,v+t)-Math.max(y,v));
    return area / (w*h+s*t-area);
  };
  for (const box of boxes) {
    if (kept.every(other => overlap(box, other) <= .6)) kept.push(box);
    if (kept.length === 300) break;
  }
  if (kept.length > 200) throw new Error('画面过于拥挤，请裁剪手牌后再测试');
  const {gainX,gainY,padX:px,padY:py}=detectorGeometry(width,height);
  return kept.map(box => {
    const [x,y,w,h] = box.bbox;
    const left = Math.max(0, Math.min(width,(x-px)/gainX)), top = Math.max(0,Math.min(height,(y-py)/gainY));
    const right = Math.max(0,Math.min(width,(x+w-px)/gainX)), bottom = Math.max(0,Math.min(height,(y+h-py)/gainY));
    return { score: box.score, bbox: [left,top,right-left,bottom-top] };
  });
}
export function classify(logits: Float32Array, start: number): { index: number; probability: number } {
  let max = -Infinity, index = 0;
  for (let i=0;i<42;i++) { const value=logits[start+i]; if (!Number.isFinite(value)) throw new Error('分类输出无效'); if(value>max){max=value;index=i;} }
  let sum=0;for(let i=0;i<42;i++)sum+=Math.exp(logits[start+i]-max);
  return {index,probability:1/sum};
}

/** Canvas interpolation is recorded as a separate runtime, not PIL equivalence. */
export function tensorPixels(image: ImageBitmap, size: number, box?: Box): Float32Array {
  let left=0,top=0,right=image.width,bottom=image.height;
  if(box){const[x,y,w,h]=box.bbox;left=Math.max(0,Math.floor(x-w*.08));top=Math.max(0,Math.floor(y-h*.08));right=Math.min(image.width,Math.ceil(x+w*1.08));bottom=Math.min(image.height,Math.ceil(y+h*1.08));}
  const w=right-left,h=bottom-top;if(w<=0||h<=0)throw new Error('牌框超出照片');
  const scale=Math.min(size/w,size/h),nw=Math.max(1,roundEven(w*scale)),nh=Math.max(1,roundEven(h*scale));
  const canvas=new OffscreenCanvas(size,size),ctx=canvas.getContext('2d',{willReadFrequently:true});
  if(!ctx)throw new Error('当前浏览器不支持后台图片处理');
  ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,size,size);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='low';
  ctx.drawImage(image,left,top,w,h,Math.floor((size-nw)/2),Math.floor((size-nh)/2),nw,nh);
  const pixels=ctx.getImageData(0,0,size,size).data,n=size*size,result=new Float32Array(3*n);
  const mean=[.485,.456,.406],std=[.229,.224,.225];
  for(let c=0;c<3;c++)for(let i=0;i<n;i++){const v=pixels[i*4+c]/255;result[c*n+i]=box?(v-mean[c])/std[c]:v;}
  return result;
}
