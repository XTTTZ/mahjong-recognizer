const digits = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];
const circleLayouts: Record<number, [number, number][]> = {
  1: [[18, 24]], 2: [[10, 11], [26, 37]], 3: [[8, 10], [18, 24], [28, 38]],
  4: [[9, 11], [27, 11], [9, 37], [27, 37]],
  5: [[8, 10], [28, 10], [18, 24], [8, 38], [28, 38]],
  6: [[9, 10], [27, 10], [9, 24], [27, 24], [9, 38], [27, 38]],
  7: [[8, 7], [18, 15], [28, 23], [8, 31], [28, 31], [8, 41], [28, 41]],
  8: [[9, 7], [27, 7], [9, 18], [27, 18], [9, 30], [27, 30], [9, 41], [27, 41]],
  9: [[8, 10], [18, 10], [28, 10], [8, 24], [18, 24], [28, 24], [8, 38], [18, 38], [28, 38]],
};

export function TileFace({ classId, label, small = false }: { classId: string; label: string; small?: boolean }) {
  const match = /^([1-9])([mps])$/.exec(classId);
  const number = Number(match?.[1]);
  const suit = match?.[2];
  const knownText = /^(东|南|西|北|中|发|發|白|東)$/.test(label) ? label : undefined;
  const face = suit === 'm' ? <><span className="tile-number">{digits[number - 1]}</span><span className="tile-wan">万</span></>
    : suit === 'p' ? <svg viewBox="0 0 36 48" aria-hidden="true">{circleLayouts[number].map(([x, y], i) => {
      return <g key={i}><circle cx={x} cy={y} r={number === 1 ? 10 : 4.1} fill="none" stroke={i % 3 === 0 ? '#b3473c' : '#1c6253'} strokeWidth="2.2" /><circle cx={x} cy={y} r={number === 1 ? 5 : 1.2} fill={i % 3 === 0 ? '#b3473c' : '#1c6253'} /></g>;
    })}</svg>
      : suit === 's' ? <><span className="tile-number bamboo-number">{digits[number - 1]}</span><span className="tile-bamboo">条</span></>
        : classId === 'back' ? <span className="tile-back" />
          : classId === '5z' ? <span className="tile-white" />
            : /^\dh$/.test(classId) ? <span className="tile-flower">{label}</span>
          : knownText ? <span className={`tile-honor ${/[中]/.test(knownText) ? 'red' : /[发發]/.test(knownText) ? 'green' : ''}`}>{knownText}</span>
            : <span className="tile-other">{label.length > 3 ? classId : label}</span>;
  return <span className={`tile-face ${small ? 'tile-small' : ''}`} aria-hidden="true">{face}</span>;
}

export function WelcomeTiles() {
  return <div className="welcome-tiles" aria-hidden="true">
    <div className="welcome-tile tile-left"><TileFace classId="3p" label="三筒" /></div>
    <div className="welcome-tile tile-right"><TileFace classId="6s" label="六条" /></div>
    <div className="welcome-tile tile-center"><TileFace classId="1m" label="一万" /></div>
  </div>;
}
