import { useId, useMemo, useState } from 'react';
import { analyzeHand, calculateScore, type DeclaredMeld, type MeldKind, type RuleOptionId, type WinMethod } from './scoring';
import './scoring.css';

const chineseNumbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];
const honorLabels = ['东', '南', '西', '北', '白板', '发', '中'];
const suitLabels: Record<string, string> = { m: '万', p: '筒', s: '条' };
const faceCodes = [...['m', 'p', 's'].flatMap(suit => chineseNumbers.map((_, i) => `${i + 1}${suit}`)),
  ...honorLabels.map((_, i) => `${i + 1}z`)];
const meldLabels: Record<MeldKind, string> = { pung: '碰', 'exposed-kong': '明杠', 'concealed-kong': '暗杠' };
const methods: { value: WinMethod; label: string }[] = [
  { value: 'unspecified', label: '未指定' }, { value: 'discard', label: '吃胡' },
  { value: 'self-draw', label: '自摸' }, { value: 'rob-exposed-kong', label: '抢明杠' },
  { value: 'rob-concealed-kong', label: '抢暗杠（仅十三幺）' },
];

function tileLabel(code: string): string {
  const match = /^([1-9])([mps])$/.exec(code);
  if (match) return chineseNumbers[Number(match[1]) - 1] + (suitLabels[match[2]] ?? '');
  return honorLabels[Number(code[0]) - 1] ?? code;
}

function numberInput(raw: string): number { return raw.trim() === '' ? 0 : Number(raw); }
function multiplier(value: number): string { return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2))); }
function optionLabel(label: string): string { return label.replace(/\s*\+\s*\d+(?:\s*倍)?$/, '').trim(); }

export interface ScoringPanelProps {
  /** All physical tiles, including the tiles used by each declared meld. */
  tiles: readonly string[];
  resetKey?: string;
}

/** A new recognition resets declarations synchronously; edits to this hand remain editable. */
export function ScoringPanel({ tiles, resetKey }: ScoringPanelProps) {
  return <ScoringPanelState key={resetKey ?? 'current-hand'} tiles={tiles} />;
}

function ScoringPanelState({ tiles }: { tiles: readonly string[] }) {
  const titleId = useId();
  const [melds, setMelds] = useState<DeclaredMeld[]>([]);
  const [meldKind, setMeldKind] = useState<MeldKind>('pung');
  const [meldTile, setMeldTile] = useState('1m');
  const [meldIssue, setMeldIssue] = useState('');
  const [winMethod, setWinMethod] = useState<WinMethod>('unspecified');
  const [winningTile, setWinningTile] = useState('');
  const [selected, setSelected] = useState<RuleOptionId[]>([]);
  const [codeHits, setCodeHits] = useState('0');
  const [tableBonus, setTableBonus] = useState('0');

  const remaining = useMemo(() => {
    const counts = new Map<string, number>();
    for (const code of tiles) counts.set(code, (counts.get(code) ?? 0) + 1);
    for (const meld of melds) counts.set(meld.tile, (counts.get(meld.tile) ?? 0) - (meld.kind === 'pung' ? 3 : 4));
    return counts;
  }, [tiles, melds]);
  const analysis = useMemo(() => analyzeHand({ tiles, melds, winMethod, winningTile: winningTile || undefined }),
    [tiles, melds, winMethod, winningTile]);
  const score = useMemo(() => calculateScore(analysis,
    { selected, codeHits: numberInput(codeHits), tableBonus: numberInput(tableBonus) }),
  [analysis, selected, codeHits, tableBonus]);
  // A legal zero-point hand remains structurally complete even when it cannot
  // meet the three-point threshold; it can still gain a confirmed menqing bonus.
  const issues = score.status === 'invalid' || analysis.status === 'invalid' ? [...new Set(score.issues)] : [];
  const legalWinningShape = analysis.status === 'winning' && score.status !== 'invalid';
  const ready = legalWinningShape && !score.belowMinimum && typeof score.finalMultiplier === 'number' && Number.isFinite(score.finalMultiplier);
  const total = ready && typeof score.finalMultiplier === 'number' ? score.finalMultiplier : null;
  const availableWinningTiles = faceCodes.filter(code => (remaining.get(code) ?? 0) > 0);
  const chosenCount = selected.length;
  const needed = meldKind === 'pung' ? 3 : 4;

  function declareMeld() {
    const count = Math.max(0, remaining.get(meldTile) ?? 0);
    if (melds.length >= 4) { setMeldIssue('最多声明4组碰杠，请先核对已有分组。'); return; }
    if (count < needed) {
      setMeldIssue(`${meldLabels[meldKind]}需要${needed}张${tileLabel(meldTile)}，当前未分组的只有${count}张。请先补牌或修改牌面。`);
      return;
    }
    setMelds(previous => [...previous, { kind: meldKind, tile: meldTile }]);
    setMeldIssue('');
  }

  function toggleOption(id: RuleOptionId, checked: boolean) {
    const availability = analysis.availableOptions.find(option => option.id === id);
    if (checked && !availability?.eligible) return;
    setSelected(previous => {
      if (!checked) return previous.filter(item => item !== id);
      const retained = id === 'heavenly-hand' || id === 'earthly-hand'
        ? previous.filter(item => item !== 'heavenly-hand' && item !== 'earthly-hand') : previous;
      return [...new Set([...retained, id])];
    });
  }

  return <section className="scoring-panel" aria-labelledby={titleId}>
    <div className="scoring-heading"><h2 id={titleId}>胡牌计倍</h2><span className="scoring-threshold">3倍起胡</span></div>
    <div className={`scoring-summary ${ready ? '' : 'is-pending'}`} role="status" aria-live="polite">
      <div>
        <div className={`scoring-total ${legalWinningShape ? '' : 'is-text'}`}>
          <strong data-testid="score-total">{total !== null ? multiplier(total) : legalWinningShape ? multiplier(score.handMultiplier) : '—'}</strong>
          {legalWinningShape && <span>倍</span>}
        </div>
        <p>{ready ? Number(codeHits) > 0 ? '中码后的最终倍数' : '当前胡牌倍数'
          : score.belowMinimum ? '不足3倍，暂不能胡' : score.status === 'not-winning' ? '当前牌面暂未组成胡牌' : '请先核对牌面与条件'}</p>
      </div>
      <span className="scoring-verdict">{ready ? '达到起胡门槛' : legalWinningShape ? '未满3倍' : '待核对'}</span>
    </div>

    {issues.length > 0 && <div className="scoring-error" role="alert"><ul className="scoring-errors">{issues.map(issue => <li key={issue}>{issue}</li>)}</ul></div>}
    {legalWinningShape && <>
      <ul className="scoring-breakdown" aria-label="计倍拆分">
        {score.patterns.map(pattern => <li key={`pattern-${pattern.id}`}><span>{pattern.label}</span><b>{multiplier(pattern.multiplier)}倍</b></li>)}
        {!score.patterns.length && <li><span>基础牌型暂无计倍</span><b>0倍</b></li>}
        {score.additions.map(addition => <li key={`addition-${addition.id}`}><span>{optionLabel(addition.label)}</span><b>+{multiplier(addition.multiplier)}倍</b></li>)}
      </ul>
      {ready && <p className="scoring-formula">{score.formula}</p>}
      {score.belowMinimum && <p className="scoring-formula">胡牌倍数需先达到3倍，中码不能凑起胡。</p>}
    </>}
    {(score.exclusions.length > 0 || score.warnings.length > 0) && <details className="scoring-exclusions">
      <summary>计倍说明{score.exclusions.length ? ` · ${score.exclusions.length}项不重复计入` : ''}</summary>
      <ul>{score.exclusions.map((item, i) => <li key={`${item.excludedId}-${i}`}>{item.excludedLabel}：{item.reason}</li>)}
        {score.warnings.map((warning, i) => <li key={`warning-${i}`}>{warning}</li>)}</ul>
    </details>}

    <details className="scoring-section">
      <summary>特殊情况<span>{chosenCount ? `已选${chosenCount}项` : '按实际牌局选择'}</span></summary>
      <div className="scoring-section-content">
        <p className="scoring-hint">照片不能判断和牌经过，未选择的特殊条件不加倍。普通自摸本身不额外加3倍。</p>
        <div className="scoring-fields is-pair">
          <label className="scoring-field">和牌方式<select aria-label="和牌方式" value={winMethod} onChange={event => setWinMethod(event.target.value as WinMethod)}>
            {methods.map(method => <option key={method.value} value={method.value}>{method.label}</option>)}
          </select></label>
          <label className="scoring-field">和牌张<select aria-label="和牌张" value={winningTile} onChange={event => setWinningTile(event.target.value)}>
            <option value="">未指定</option>
            {winningTile && !availableWinningTiles.includes(winningTile) && <option value={winningTile}>{tileLabel(winningTile)}（请重选）</option>}
            {availableWinningTiles.map(code => <option key={code} value={code}>{tileLabel(code)}</option>)}
          </select></label>
        </div>
        <div className="scoring-choices">{analysis.availableOptions.map(option => {
          const checked = selected.includes(option.id);
          return <label key={option.id} className={`scoring-choice ${option.eligible ? '' : 'is-disabled'}`} title={option.reason ?? option.description}>
            <input type="checkbox" aria-label={optionLabel(option.label)} checked={checked} disabled={!option.eligible && !checked}
              onChange={event => toggleOption(option.id, event.target.checked)} />
            <span>{optionLabel(option.label)}<small>{option.eligible ? `${option.kind === 'override' ? '' : '+'}${option.multiplier}倍` : option.reason ?? '当前牌型不适用'}</small></span>
          </label>;
        })}</div>
        {chosenCount > 0 && <button type="button" className="scoring-button" onClick={() => setSelected([])}>清空特殊选择</button>}
      </div>
    </details>

    <details className="scoring-section">
      <summary>碰杠<span>{melds.length ? `已声明${melds.length}组` : '未声明'}</span></summary>
      <div className="scoring-section-content">
        <p className="scoring-hint">已碰、明杠、暗杠的牌也要包含在上方牌面中。这里仅分组，不会自动补牌。</p>
        {melds.length > 0 && <ul className="scoring-meld-list" aria-label="已声明碰杠">{melds.map((meld, index) => <li key={`${index}-${meld.tile}-${meld.kind}`}>
          <span>{meldLabels[meld.kind]} · {tileLabel(meld.tile)} ×{meld.kind === 'pung' ? 3 : 4}</span>
          <button type="button" className="scoring-remove" aria-label={`删除第${index + 1}组${meldLabels[meld.kind]}`} onClick={() => {
            setMelds(previous => previous.filter((_, i) => i !== index)); setMeldIssue('');
          }}>移除</button>
        </li>)}</ul>}
        <div className="scoring-kind-options" aria-label="选择碰杠种类">{(Object.keys(meldLabels) as MeldKind[]).map(kind => <button type="button" key={kind}
          aria-pressed={meldKind === kind} onClick={() => { setMeldKind(kind); setMeldIssue(''); }}>{meldLabels[kind]}</button>)}</div>
        <div className="scoring-fields"><label className="scoring-field">分组牌面<select aria-label="分组牌面" value={meldTile} onChange={event => { setMeldTile(event.target.value); setMeldIssue(''); }}>
          {faceCodes.map(code => <option value={code} key={code}>{tileLabel(code)}（未分组{Math.max(0, remaining.get(code) ?? 0)}张）</option>)}
        </select></label></div>
        <button type="button" className="scoring-button is-full" onClick={declareMeld}>声明{meldLabels[meldKind]}</button>
        {meldIssue && <p className="scoring-error" role="alert">{meldIssue}</p>}
      </div>
    </details>

    <details className="scoring-section">
      <summary>中码 / 自定义<span>{numberInput(codeHits) || numberInput(tableBonus) ? '已调整' : '默认不加'}</span></summary>
      <div className="scoring-section-content">
        <div className="scoring-field"><span>中码数</span><div className="scoring-counter">
          <button type="button" aria-label="减少中码" disabled={!Number.isFinite(numberInput(codeHits)) || numberInput(codeHits) <= 0}
            onClick={() => setCodeHits(String(Math.max(0, numberInput(codeHits) - 1)))}>−</button>
          <input type="number" aria-label="中码数" inputMode="numeric" min="0" step="1" value={codeHits} onChange={event => setCodeHits(event.target.value)} />
          <button type="button" aria-label="增加中码" onClick={() => setCodeHits(String((Number.isFinite(numberInput(codeHits)) ? numberInput(codeHits) : 0) + 1))}>+</button>
        </div></div>
        <p className="scoring-hint">胡牌倍数 ×（1＋中码数）。请按本桌规则填写实际中码数。</p>
        <div className="scoring-fields"><label className="scoring-field">本桌另加倍数（可选）
          <input type="number" aria-label="本桌另加倍数" inputMode="numeric" min="0" step="1" value={tableBonus} onChange={event => setTableBonus(event.target.value)} />
        </label></div>
        <p className="scoring-hint">仅用于另行约定的加项，先加入胡牌倍数，再计算中码。默认0。</p>
      </div>
    </details>
    <p className="scoring-policy">广东大牌136张 · 兼容牌型相加，按附图排除重复。先核对全部牌面与碰杠，结果仅按已选条件计算，不代替包赔、杠钱或输赢结算。</p>
  </section>;
}
