import { useEffect, useRef, useState } from 'react';
import { Icon } from './icons';
import { recognize, RecognitionError, serviceConfigured } from './api';
import { cropPhoto, preparePhoto, rotatePhoto } from './photo';
import { canonicalCropEdges, isFullCrop, rotateCropClockwise, type CropEdges } from './crop';
import { CropDialog } from './CropDialog';
import { TileFace, WelcomeTiles } from './TileFace';
import type { PreparedPhoto, RecognitionResult, TileCorrection, TilePrediction } from './types';

const numbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];
const correctionGroups = [
  { title: '万', values: numbers.map((n, i) => ({ classId: `${i + 1}m`, label: `${n}万` })) },
  { title: '筒', values: numbers.map((n, i) => ({ classId: `${i + 1}p`, label: `${n}筒` })) },
  { title: '条', values: numbers.map((n, i) => ({ classId: `${i + 1}s`, label: `${n}条` })) },
  // Visually checked against the fixed v2 dataset class contact sheets.
  { title: '字牌', values: ['东', '南', '西', '北', '白板', '发', '中'].map((label, i) => ({ classId: `${i + 1}z`, label })) },
  { title: '其他', values: [...['春', '夏', '秋', '冬', '梅', '兰', '菊', '竹'].map((label, i) => ({ classId: `${i + 1}h`, label })), { classId: 'back', label: '牌背' }] },
];
const displayLabels = new Map(correctionGroups.flatMap(group => group.values).map(tile => [tile.classId, tile.label]));

type RequestState = 'idle' | 'preparing' | 'recognizing';

function App() {
  const [photo, setPhoto] = useState<PreparedPhoto | null>(null);
  const [originalPhoto, setOriginalPhoto] = useState<PreparedPhoto | null>(null);
  const [cropRegion, setCropRegion] = useState<CropEdges | null>(null);
  const [cropOpen, setCropOpen] = useState(false);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [state, setState] = useState<RequestState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [slow, setSlow] = useState(false);
  const [offline, setOffline] = useState(!navigator.onLine);
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [infoOpen, setInfoOpen] = useState(false);
  const [selectedTile, setSelectedTile] = useState<string | null>(null);
  const [corrections, setCorrections] = useState<Record<string, TileCorrection>>({});
  const [hiddenTiles, setHiddenTiles] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [zoomed, setZoomed] = useState(false);
  const cameraRef = useRef<HTMLInputElement>(null);
  const libraryRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<AbortController | null>(null);
  const generationRef = useRef(0);
  const photoRef = useRef<PreparedPhoto | null>(null);
  const originalPhotoRef = useRef<PreparedPhoto | null>(null);
  const correctionDialog = useRef<HTMLDialogElement>(null);
  const busy = state !== 'idle';

  useEffect(() => {
    const healthController = new AbortController();
    const refreshHealth = () => {
      void serviceConfigured(healthController.signal).then(value => {
        if (!healthController.signal.aborted) setConfigured(value);
      });
    };
    const update = () => { setOffline(!navigator.onLine); if (navigator.onLine) refreshHealth(); };
    refreshHealth();
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    return () => {
      window.removeEventListener('online', update);
      window.removeEventListener('offline', update);
      healthController.abort();
      generationRef.current += 1;
      requestRef.current?.abort();
      for (const url of new Set([photoRef.current?.url, originalPhotoRef.current?.url])) if (url) URL.revokeObjectURL(url);
    };
  }, []);

  useEffect(() => {
    if (selectedTile) correctionDialog.current?.showModal();
    else correctionDialog.current?.close();
  }, [selectedTile]);

  const replacePhotoPair = (next: PreparedPhoto | null, original: PreparedPhoto | null) => {
    const nextUrls = new Set([next?.url, original?.url]);
    for (const url of new Set([photoRef.current?.url, originalPhotoRef.current?.url])) {
      if (url && !nextUrls.has(url)) URL.revokeObjectURL(url);
    }
    photoRef.current = next;
    originalPhotoRef.current = original;
    setPhoto(next);
    setOriginalPhoto(original);
    setResult(null);
    setCorrections({});
    setHiddenTiles([]);
    setSelectedTile(null);
    setZoomed(false);
  };

  const replacePhoto = (next: PreparedPhoto | null) => {
    replacePhotoPair(next, next);
    setCropRegion(null);
    setCropOpen(false);
  };

  const invalidate = () => {
    generationRef.current += 1;
    requestRef.current?.abort();
    requestRef.current = null;
    setSlow(false);
    return generationRef.current;
  };

  const choosePhoto = async (file?: File) => {
    if (!file) return;
    const generation = invalidate();
    setState('preparing');
    setError(null);
    setResult(null);
    try {
      const next = await preparePhoto(file);
      if (generation !== generationRef.current) { URL.revokeObjectURL(next.url); return; }
      replacePhoto(next);
    } catch (cause) {
      if (generation === generationRef.current) setError(cause instanceof Error ? cause.message : '照片处理失败，请重新选择。');
    } finally {
      if (generation === generationRef.current) setState('idle');
    }
  };

  const rotate = async () => {
    if (!photo || !originalPhoto || busy) return;
    const generation = invalidate();
    setState('preparing');
    setError(null);
    let nextOriginal: PreparedPhoto | null = null;
    let next: PreparedPhoto | null = null;
    try {
      nextOriginal = await rotatePhoto(originalPhoto);
      const region = cropRegion ? canonicalCropEdges(rotateCropClockwise(cropRegion), nextOriginal.width, nextOriginal.height) : null;
      next = region ? await cropPhoto(nextOriginal, region) : nextOriginal;
      if (generation !== generationRef.current) {
        for (const url of new Set([next.url, nextOriginal.url])) URL.revokeObjectURL(url);
        return;
      }
      replacePhotoPair(next, nextOriginal);
      setCropRegion(region);
    } catch (cause) {
      for (const url of new Set([next?.url, nextOriginal?.url])) if (url) URL.revokeObjectURL(url);
      if (generation === generationRef.current) setError(cause instanceof Error ? cause.message : '旋转失败，请重新选择照片。');
    } finally {
      if (generation === generationRef.current) setState('idle');
    }
  };

  const applyCrop = async (region: CropEdges) => {
    const original = originalPhotoRef.current;
    if (!original) throw new Error('请先选择照片。');
    const generation = invalidate();
    setState('preparing');
    setError(null);
    try {
      const canonical = canonicalCropEdges(region, original.width, original.height);
      const next = isFullCrop(canonical) ? original : await cropPhoto(original, canonical);
      if (generation !== generationRef.current) {
        if (next.url !== original.url) URL.revokeObjectURL(next.url);
        return false;
      }
      replacePhotoPair(next, original);
      setCropRegion(isFullCrop(canonical) ? null : canonical);
      return true;
    } finally { if (generation === generationRef.current) setState('idle'); }
  };

  const restoreWholePhoto = () => {
    if (!originalPhotoRef.current) return;
    invalidate();
    replacePhotoPair(originalPhotoRef.current, originalPhotoRef.current);
    setCropRegion(null);
    setError(null);
    setState('idle');
  };

  const startRecognition = async () => {
    if (!photo || busy) return;
    const generation = invalidate();
    const controller = new AbortController();
    requestRef.current = controller;
    let timedOut = false;
    const slowTimer = window.setTimeout(() => {
      if (generation === generationRef.current) setSlow(true);
    }, 8_000);
    const timeout = window.setTimeout(() => { timedOut = true; controller.abort(); }, 65_000);
    setState('recognizing');
    setResult(null);
    setCorrections({});
    setHiddenTiles([]);
    setError(null);
    try {
      const response = await recognize(photo.file, controller.signal);
      if (generation !== generationRef.current) return;
      setResult(response);
    } catch (cause) {
      if (generation !== generationRef.current) return;
      if (timedOut) setError('等待时间较长，已停止等待。原请求可能仍在处理，请稍后再试。');
      else if (cause instanceof Error && cause.name === 'AbortError') return;
      else if (cause instanceof RecognitionError) setError(cause.message);
      else setError(navigator.onLine ? '连接中断，请检查网络后重新识别。' : '网络已断开，连接恢复后可重新识别。');
    } finally {
      window.clearTimeout(slowTimer);
      window.clearTimeout(timeout);
      if (generation === generationRef.current) {
        requestRef.current = null;
        setState('idle');
        setSlow(false);
      }
    }
  };

  const cancel = () => {
    invalidate();
    setState('idle');
    setError('已停止等待。需要时可重新选择照片。');
  };

  const visibleTiles = result?.tiles.filter(tile => !hiddenTiles.includes(tile.id)) ?? [];
  const changed = Object.keys(corrections).length + hiddenTiles.length;
  const allRemoved = Boolean(result?.tiles.length && visibleTiles.length === 0);
  const activeTile = result?.tiles.find(tile => tile.id === selectedTile);
  const corrected = (tile: TilePrediction) => corrections[tile.id]
    ?? { classId: tile.classId, label: displayLabels.get(tile.classId) ?? tile.label };
  const uncertainCount = visibleTiles.filter(tile => tile.uncertain && !corrections[tile.id]).length;

  return <>
    <main className="shell">
      <header className="header">
        <a className="brand" href="/" aria-label="识麻将首页"><img src="/icon.svg" width="37" height="37" alt="" /><span>识麻将<span className="brand-description">拍照识牌</span></span></a>
        <button className={`icon-button info-button ${infoOpen ? 'active' : ''}`} aria-label="拍摄提示与隐私" aria-expanded={infoOpen} onClick={() => setInfoOpen(!infoOpen)}><Icon name="info" size={21} /></button>
      </header>

      {infoOpen && <aside className="info-card">
        <strong>让每张牌面都清楚</strong>
        <p>对准手牌，尽量避开反光和遮挡。可用“裁剪”选取目标区域，区域内所有可见的牌都会识别。</p>
        <p>照片仅用于本次识别，默认不保存。结果可逐张核对；看不清的牌建议重新拍摄。</p>
        <p>当前为试用模型，可能漏牌或认错牌面，请逐张核对后使用。</p>
      </aside>}

      {!photo && <section className="intro"><span className="eyebrow">把手牌，看清楚</span><h1>拍下手牌，<br />一眼看清。</h1><p>对准麻将牌面，拍照或上传即可识别。</p></section>}

      <p className="trial-note">试用模型 · 结果请逐张核对</p>
      {configured === false && <p className="notice service-notice" role="status"><Icon name="info" size={18} />识别服务准备中，暂可拍照、选图和裁剪。</p>}

      <section className={`photo-card ${photo ? 'has-photo' : ''} ${dragOver ? 'drag-over' : ''}`} aria-label="手牌照片"
        onDragOver={event => { event.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={event => { event.preventDefault(); setDragOver(false); void choosePhoto(event.dataTransfer.files[0]); }}>
        {photo ? <>
          <div className="photo-toolbar"><span><span className="status-dot" />{result ? '识别照片' : '准备好了'}</span><div>
            <button className="text-button" onClick={() => setCropOpen(true)} disabled={busy} aria-label="选取要识别的牌面区域"><Icon name="crop" size={17} /><span>裁剪</span></button>
            <button className="text-button" onClick={() => void rotate()} disabled={busy} aria-label="向右旋转照片"><Icon name="rotate" size={17} /><span>旋转</span></button>
            <button className="icon-button" onClick={() => { invalidate(); replacePhoto(null); setState('idle'); setError(null); }} aria-label="移除照片"><Icon name="close" size={19} /></button>
          </div></div>
          <div className={`photo-stage ${zoomed ? 'is-zoomed' : ''}`}>
            <div className="photo-canvas" style={{ aspectRatio: `${photo.width} / ${photo.height}` }}>
              <img className="source-photo" src={photo.url} alt="本次识别的麻将照片" />
              {result && <svg className="annotation-layer" viewBox={`0 0 ${result.image.width} ${result.image.height}`} aria-label="麻将牌位置">
                {visibleTiles.map((tile, i) => <g key={tile.id} className={`annotation ${tile.uncertain && !corrections[tile.id] ? 'uncertain' : ''}`} role="button" aria-label={`第 ${i + 1} 张：${corrected(tile).label}，查看或修改`} tabIndex={0} onClick={() => setSelectedTile(tile.id)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedTile(tile.id); } }}>
                  <rect x={tile.bbox[0]} y={tile.bbox[1]} width={tile.bbox[2]} height={tile.bbox[3]} vectorEffect="non-scaling-stroke" />
                  <text x={tile.bbox[0] + 3} y={tile.bbox[1] + Math.min(24, result.image.width * 0.028)} fontSize={Math.max(14, result.image.width * 0.024)}>{i + 1}</text>
                </g>)}
              </svg>}
            </div>
            {result && <button className="zoom-button icon-button" aria-label={zoomed ? '缩小照片' : '放大照片'} aria-pressed={zoomed} onClick={() => setZoomed(!zoomed)}><Icon name="zoom" size={19} /></button>}
          </div>
          <div className="photo-caption"><span>{result ? '点照片中的牌框，或下方牌面，逐张核对' : cropRegion ? '识别当前选区内的所有可见牌' : '默认识别整张照片，可裁剪选取目标区域'}</span>{cropRegion && <button className="text-button" disabled={state === 'preparing'} onClick={restoreWholePhoto}>恢复整图</button>}</div>
        </> : <div className="empty-photo">
          <div className="viewfinder"><span /><span /><span /><span /><WelcomeTiles /></div>
          <div className="empty-copy"><strong>把一手好牌，放进画面</strong><span>牌面朝上 · 光线均匀 · 避开反光</span></div>
        </div>}
      </section>

      <div className="inputs">
        <input ref={cameraRef} type="file" accept="image/*" capture="environment" aria-label="拍摄麻将照片" onChange={e => { void choosePhoto(e.target.files?.[0]); e.target.value = ''; }} />
        <input ref={libraryRef} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif" aria-label="上传麻将照片" onChange={e => { void choosePhoto(e.target.files?.[0]); e.target.value = ''; }} />
      </div>

      {offline && <p className="notice" role="status"><Icon name="alert" size={18} />当前没有网络，连接后即可识别。</p>}
      {error && <div className="error-notice" role="alert"><Icon name="alert" size={19} /><p>{error}</p><button className="icon-button" onClick={() => setError(null)} aria-label="关闭提示"><Icon name="close" size={17} /></button></div>}

      {state === 'recognizing' ? <div className="recognition-progress" role="status" aria-live="polite"><span className="spinner" /><div><strong>正在仔细看牌…</strong><p>{slow ? '首次识别可能需要多等一会儿' : '请稍候，结果马上呈现'}</p></div><button className="text-button" onClick={cancel}>取消</button></div>
        : state === 'preparing' ? <div className="recognition-progress" role="status"><span className="spinner" /><strong>正在准备照片…</strong></div>
          : photo && !result ? <div className="actions photo-actions"><button className="primary-button" onClick={() => void startRecognition()} disabled={offline || configured === false}><Icon name="spark" />开始识别<Icon name="arrow" size={19} /></button><button className="secondary-button" onClick={() => libraryRef.current?.click()}><Icon name="image" size={19} />换一张</button></div>
            : !photo ? <div className="actions"><button className="primary-button" onClick={() => cameraRef.current?.click()}><Icon name="camera" />拍一张<Icon name="arrow" size={19} /></button><button className="secondary-button" onClick={() => libraryRef.current?.click()}><Icon name="image" size={19} />从相册选</button></div> : null}

      {result && <section className="results" aria-label="识别结果" aria-live="polite">
        <div className="results-heading"><div><span className="eyebrow">这次识别</span><h1>{visibleTiles.length ? `${visibleTiles.length} 张麻将` : allRemoved ? '已移除全部牌' : '暂未找到麻将'}</h1></div><span className={`result-badge ${!allRemoved && (uncertainCount || result.quality.status === 'needs_retake' || !visibleTiles.length) ? 'needs-check' : ''}`}><Icon name={!allRemoved && (uncertainCount || result.quality.status === 'needs_retake' || !visibleTiles.length) ? 'alert' : 'check'} size={15} />{allRemoved ? '人工调整' : uncertainCount ? `${uncertainCount} 张待核对` : result.quality.status === 'needs_retake' || !visibleTiles.length ? '建议重拍' : '请核对牌面'}</span></div>
        {!allRemoved && (result.quality.status === 'needs_retake' || result.quality.reasons.length > 0 || !visibleTiles.length) && <div className="quality-note"><Icon name="alert" size={18} /><p>{result.quality.reasons.length ? result.quality.reasons.join('；') : '没有找到清楚的牌面，请靠近一些重新拍摄。'}</p></div>}
        {allRemoved && <button className="secondary-button" onClick={() => { setCorrections({}); setHiddenTiles([]); }}>撤销人工调整</button>}
        {visibleTiles.length > 0 && <><div className="tile-list">{visibleTiles.map((tile, index) => {
          const value = corrected(tile);
          return <button key={tile.id} className={`tile-button ${tile.uncertain && !corrections[tile.id] ? 'needs-check' : ''} ${corrections[tile.id] ? 'corrected' : ''}`} aria-label={`第 ${index + 1} 张 ${value.label}${corrections[tile.id] ? '，已手动修改' : tile.uncertain ? '，待核对' : ''}`} onClick={() => setSelectedTile(tile.id)}><span className="tile-index">{index + 1}</span><TileFace classId={value.classId} label={value.label} /><span className="tile-label">{value.label}</span>{tile.uncertain && !corrections[tile.id] ? <span className="tile-flag">待核对</span> : corrections[tile.id] ? <span className="tile-flag manual">已修改</span> : <span className="tile-flag placeholder">核对</span>}</button>;
        })}</div><div className="results-hint"><span>点击牌面可修改{changed > 0 ? ` · ${changed} 处人工调整` : ''}</span>{changed > 0 && <button className="text-button" onClick={() => { setCorrections({}); setHiddenTiles([]); }}>撤销调整</button>}</div></>}
        <div className="result-actions"><button className="primary-button" onClick={() => cameraRef.current?.click()}><Icon name="camera" size={20} />再拍一张</button><button className="secondary-button" onClick={() => libraryRef.current?.click()}><Icon name="image" size={19} />换照片</button></div>
        <details className="result-details"><summary>识别详情</summary><dl><div><dt>模型版本</dt><dd>{result.modelVersion}</dd></div><div><dt>处理耗时</dt><dd>{(result.timings.totalMs / 1000).toFixed(1)} 秒</dd></div><div><dt>请求编号</dt><dd>{result.requestId}</dd></div></dl></details>
      </section>}

      <footer className="footer">
        <p><span className="privacy-dot" />照片仅用于本次识别，默认不保存</p>
        <nav aria-label="开源与模型" className="source-links">
          <a href="https://github.com/XTTTZ/mahjong-recognizer" target="_blank" rel="noopener noreferrer">完整源码 · AGPL-3.0</a>
          <a href="https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027" target="_blank" rel="noopener noreferrer">模型下载</a>
        </nav>
      </footer>
    </main>

    <CropDialog open={cropOpen} source={originalPhoto} initial={cropRegion} onApply={applyCrop} onClose={() => setCropOpen(false)} />
    <dialog ref={correctionDialog} className="correction-dialog" onClose={() => setSelectedTile(null)} onClick={event => { if (event.target === event.currentTarget) setSelectedTile(null); }} aria-labelledby="tile-dialog-title">
      {activeTile && <div className="dialog-body"><div className="dialog-heading"><div><span className="eyebrow">逐张核对</span><h2 id="tile-dialog-title">{corrected(activeTile).label}</h2></div><button className="icon-button" autoFocus aria-label="关闭牌面修改" onClick={() => setSelectedTile(null)}><Icon name="close" /></button></div>
        <p className="dialog-description">选出这张牌的正确牌面。人工修改仅在本次显示。</p>
        <div className="correction-options">{correctionGroups.map(group => <fieldset key={group.title}><legend>{group.title}</legend><div>{group.values.map(option => <button key={option.classId} className={corrected(activeTile).classId === option.classId ? 'selected' : ''} onClick={() => { setCorrections(previous => ({ ...previous, [activeTile.id]: option })); setSelectedTile(null); }}>{option.label}</button>)}</div></fieldset>)}</div>
        <div className="dialog-footer"><button className="text-button danger" onClick={() => { setHiddenTiles(previous => [...previous, activeTile.id]); setSelectedTile(null); }}>这不是麻将牌</button>{corrections[activeTile.id] && <button className="text-button" onClick={() => { setCorrections(previous => { const next = { ...previous }; delete next[activeTile.id]; return next; }); setSelectedTile(null); }}>恢复识别结果</button>}</div>
      </div>}
    </dialog>
  </>;
}

export default App;
