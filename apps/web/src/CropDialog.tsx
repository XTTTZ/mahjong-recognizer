import { useEffect, useRef, useState } from 'react';
import { cropRect, FULL_CROP, MIN_CROP_PIXELS, type CropEdges } from './crop';
import { Icon } from './icons';
import type { PreparedPhoto } from './types';

interface Props {
  open: boolean;
  source: PreparedPhoto | null;
  initial: CropEdges | null;
  onApply: (edges: CropEdges) => Promise<boolean>;
  onClose: () => void;
}

export function CropDialog({ open, source, initial, onApply, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [edges, setEdges] = useState<CropEdges>(FULL_CROP);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const operation = useRef(0);
  const initialRef = useRef(initial);
  initialRef.current = initial;
  useEffect(() => {
    operation.current += 1;
    setApplying(false);
    if (open && source) {
      setEdges(initialRef.current ?? FULL_CROP);
      setError(null);
      dialog.current?.showModal();
    } else dialog.current?.close();
  }, [open, source]);
  useEffect(() => () => { operation.current += 1; }, []);

  const update = (side: keyof CropEdges, percent: number) => {
    if (!source) return;
    const minW = MIN_CROP_PIXELS / source.width;
    const minH = MIN_CROP_PIXELS / source.height;
    const value = Math.min(1, Math.max(0, percent / 100));
    setEdges(previous => ({ ...previous, [side]: side === 'left' ? Math.min(value, previous.right - minW)
      : side === 'right' ? Math.max(value, previous.left + minW)
        : side === 'top' ? Math.min(value, previous.bottom - minH)
          : Math.max(value, previous.top + minH) }));
    setError(null);
  };
  const apply = async () => {
    if (!source || applying) return;
    const current = operation.current;
    try {
      cropRect(edges, source.width, source.height);
      setApplying(true);
      const applied = await onApply(edges);
      if (current === operation.current && applied) onClose();
    } catch (cause) {
      if (current === operation.current) setError(cause instanceof Error ? cause.message : '裁剪未完成，请重新选择区域。');
    } finally { if (current === operation.current) setApplying(false); }
  };

  return <dialog ref={dialog} className="crop-dialog" aria-labelledby="crop-title"
    onCancel={event => { event.preventDefault(); if (!applying) onClose(); }}
    onClose={event => { if (open && !event.currentTarget.open && !applying) onClose(); }}
    onClick={event => { if (event.target === event.currentTarget && !applying) onClose(); }}>
    {source && <div className="dialog-body"><div className="dialog-heading"><div><span className="eyebrow">手动选取</span><h2 id="crop-title">要识别哪一组牌？</h2></div><button className="icon-button" aria-label="关闭区域选择" disabled={applying} onClick={onClose}><Icon name="close" /></button></div>
      <p className="dialog-description">调整四边，把需要识别的牌面放进亮框内。框内所有可见的牌都会识别。</p>
      <div className="crop-image-stage"><div className="crop-image-frame"><img src={source.url} alt="调整要识别的牌面区域" /><svg className="crop-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <path d={`M0 0H100V100H0Z M${edges.left * 100} ${edges.top * 100}V${edges.bottom * 100}H${edges.right * 100}V${edges.top * 100}Z`} fillRule="evenodd" fill="#10271db3" />
        <rect x={edges.left * 100} y={edges.top * 100} width={(edges.right - edges.left) * 100} height={(edges.bottom - edges.top) * 100} fill="none" stroke="#f2ffe3" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      </svg></div></div>
      <div className="crop-sliders">{([['left', '左边'], ['right', '右边'], ['top', '上边'], ['bottom', '下边']] as const).map(([side, label]) => <label key={side}><span>{label}</span><input aria-label={`裁剪${label}`} type="range" min="0" max="100" step="0.5" value={edges[side] * 100} disabled={applying} onChange={event => update(side, Number(event.target.value))} /><output>{Math.round(edges[side] * 100)}%</output></label>)}</div>
      {error && <p className="crop-error" role="alert">{error}</p>}
      <div className="crop-dialog-actions"><button className="text-button" disabled={applying} onClick={() => { setEdges(FULL_CROP); setError(null); }}>重置整图</button><div><button className="secondary-button" disabled={applying} onClick={onClose}>取消</button><button className="primary-button" disabled={applying} onClick={() => void apply()}>{applying ? '正在裁剪…' : '应用区域'}</button></div></div>
    </div>}
  </dialog>;
}
