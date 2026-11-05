/** Shared vector artwork across recognition, correction, waits and benchmarks. */
export function TileFace({ classId, label, small = false }: { classId: string; label: string; small?: boolean }) {
  const supported = /^(?:[1-9][mps]|[1-7]z|[1-8]h|back)$/.test(classId);
  return <span className={`tile-face ${small ? 'tile-small' : ''}`} aria-hidden="true">
    {supported ? <img className="tile-artwork" src={`/tiles/${classId}.svg`} alt="" draggable={false} decoding="async" />
      : <span className="tile-other">{label}</span>}
  </span>;
}

export function WelcomeTiles() {
  return <div className="welcome-tiles" aria-hidden="true">
    <div className="welcome-tile tile-left"><TileFace classId="3p" label="三筒" /></div>
    <div className="welcome-tile tile-right"><TileFace classId="6s" label="六条" /></div>
    <div className="welcome-tile tile-center"><TileFace classId="1m" label="一万" /></div>
  </div>;
}
