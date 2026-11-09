import { afterEach,describe,expect,it,vi } from 'vitest';
import { classifierPixels,tensorPixels,type Box } from './pipeline';
afterEach(()=>vi.unstubAllGlobals());
describe('contiguousboundedclassificationcrops',()=>{
 it('keepscompleteROIcontext andnormalizesall192/224/320 pixelsincludingtailbatches',()=>{
  const draws:unknown[][]=[];
  class Canvas {
   width:number;height:number;red=0;
   constructor(width:number,height:number){this.width=width;this.height=height;}
   getContext(){return {fillStyle:'',imageSmoothingEnabled:true,imageSmoothingQuality:'low',fillRect:()=>{},drawImage:(...args:unknown[])=>{draws.push(args);this.red=40+Number(args[1]);},getImageData:()=>{const data=new Uint8ClampedArray(this.width*this.height*4);for(let i=0;i<data.length;i+=4){data[i]=this.red;data[i+1]=100;data[i+2]=200;data[i+3]=255;}return {data};}};}
  }
  vi.stubGlobal('OffscreenCanvas',Canvas);
  const image={width:100,height:100}as ImageBitmap;
  const a:Box={bbox:[10,20,10,20],score:.9},b:Box={bbox:[30,20,10,20],score:.8};
  for(const size of [192,224,320])for(const boxes of [[a],[a,b],[a,b,a,b]]){
   const values=classifierPixels(image,boxes,size),n=size*size;
   expect(values.length).toBe(boxes.length*3*n);
   boxes.forEach((box,i)=>{const left=Math.floor(box.bbox[0]-.08*box.bbox[2]);expect(values[i*3*n]).toBeCloseTo(((40+left)/255-.485)/.229,5);expect(values[i*3*n+n]).toBeCloseTo((100/255-.456)/.224,5);expect(values[i*3*n+2*n]).toBeCloseTo((200/255-.406)/.225,5);});
  }
  expect(draws[0]).toEqual([image,9,18,12,24,48,0,96,192]);
  expect(()=>classifierPixels(image,[],224)).toThrow();expect(()=>classifierPixels(image,[a,a,a,a,a],224)).toThrow();
  expect(()=>tensorPixels(image,128,a)).toThrow();
 });
});
