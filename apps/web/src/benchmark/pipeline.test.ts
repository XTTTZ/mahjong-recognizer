import { describe, expect, it } from 'vitest';
import { classify, detectorGeometry, nms, roundEven } from './pipeline';
import { tileLabel } from './labels';
const raw=()=>new Float32Array(5*18900);
function detection(a:Float32Array,i:number,b:[number,number,number,number],score=.9){b.forEach((v,j)=>{a[j*18900+i]=v;});a[4*18900+i]=score;}
describe('portable browser postprocess contract',()=>{
 it('uses the reviewed honor/flower names',()=>{expect(tileLabel('1z')).toBe('东');expect(tileLabel('7z')).toBe('中');expect(tileLabel('7h')).toBe('菊');expect(tileLabel('8h')).toBe('竹');});
 it('uses even resize rounding, including half-pixel cases',()=>{expect([.5,1.5,2.5,3.5].map(roundEven)).toEqual([0,2,2,4]);});
 it('inverts each rounded detector axis, not a shared nominal gain',()=>{
  const g=detectorGeometry(1234,777);expect(g.gainX).not.toBe(g.gainY);
  const a=raw();detection(a,0,[100*g.gainX+g.padX,200*g.gainY+g.padY,20*g.gainX,40*g.gainY]);
  const [b]=nms(a,1234,777);expect(b.bbox[0]).toBeCloseTo(90,3);expect(b.bbox[1]).toBeCloseTo(180,3);expect(b.bbox[2]).toBeCloseTo(20,3);expect(b.bbox[3]).toBeCloseTo(40,3);
 });
 it('does not suppress adjacent repeated tile faces; suppresses duplicate boxes',()=>{const a=raw();detection(a,0,[40,50,20,20],.9);detection(a,1,[40,50,20,20],.8);detection(a,2,[65,50,20,20],.7);expect(nms(a,960,960)).toHaveLength(2);});
 it('preserves the crowd guard instead of truncating to 200',()=>{const a=raw();for(let i=0;i<201;i++)detection(a,i,[(i%20)*40+10,Math.floor(i/20)*40+10,10,10]);expect(()=>nms(a,960,960)).toThrow('过于拥挤');});
 it('rejects malformed/nonfinite outputs',()=>{expect(()=>nms(new Float32Array(3),960,960)).toThrow();const a=raw();a[0]=NaN;expect(()=>nms(a,960,960)).toThrow();});
 it('stable softmax returns the actual top probability',()=>{const a=new Float32Array(42).fill(-1000);a[6]=1000;a[8]=1000;expect(classify(a,0)).toEqual({index:6,probability:.5});a[10]=NaN;expect(()=>classify(a,0)).toThrow();});
});
