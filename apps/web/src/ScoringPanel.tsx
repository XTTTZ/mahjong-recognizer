import { useEffect, useId, useMemo, useState } from 'react';
import { analyzeHand, analyzeCompatibility, analyzeWaitingHand, calculateScore, type DeclaredMeld,
  type MeldKind, type RuleOptionId, type WaitingTileCandidate, type WinMethod } from './scoring';
import './scoring.css';
import { TileFace } from './TileFace';

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

function WaitingTiles({ candidates, label, conditional = false }: {
  candidates: readonly WaitingTileCandidate[]; label: string; conditional?: boolean;
}) {
  if (!candidates.length) return null;
  const renderList = (items: readonly WaitingTileCandidate[], listLabel: string) => <ul className={`scoring-wait-list ${items.length === 1 ? 'has-one' : ''}`} aria-label={listLabel}>{items.map(candidate => <li key={candidate.tile} data-tile={candidate.tile}>
    <TileFace classId={candidate.tile} label={tileLabel(candidate.tile)} small />
    <div className="scoring-wait-copy">
      <div><strong>{tileLabel(candidate.tile)}</strong><b>基础{multiplier(candidate.baseMultiplier)}倍</b></div>
      <span>{candidate.patterns.map(pattern => pattern.label).join('＋') || '基础牌型暂无计倍'}</span>
      {conditional && candidate.conditionHints.length > 0 && <details className="scoring-wait-condition"><summary>条件说明</summary><p>{candidate.conditionHints.join('；')}</p></details>}
    </div>
  </li>)}</ul>;
  return <div className={`scoring-wait-group ${conditional ? 'is-conditional' : ''}`}>
    <h4>{label}</h4>
    {renderList(candidates.slice(0, 6), label)}
    {candidates.length > 6 && <details className="scoring-wait-extra">
      <summary>查看其余{candidates.length - 6}种候选</summary>
      {renderList(candidates.slice(6), `${label} · 更多`)}
    </details>}
  </div>;
}

function ScoringPanelState({ tiles }: { tiles: readonly string[] }) {
  const titleId = useId();
  const [melds, setMelds] = useState<DeclaredMeld[]>([]);
  const [meldKind, setMeldKind] = useState<MeldKind>('pung');
  const [meldTile, setMeldTile] = useState('1m');
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
  const availableWinningTiles = faceCodes.filter(code => (remaining.get(code) ?? 0) > 0);
  const activeWinningTile = availableWinningTiles.includes(winningTile) ? winningTile : '';
  const analysis = useMemo(() => analyzeHand({ tiles, melds, winMethod, winningTile: activeWinningTile || undefined }),
    [tiles, melds, winMethod, activeWinningTile]);
  const compatibility = useMemo(() => analyzeCompatibility(analysis, selected), [analysis, selected]);
  const waiting = useMemo(() => analyzeWaitingHand({ tiles, melds, winMethod }), [tiles, melds, winMethod]);
  const score = useMemo(() => calculateScore(analysis,
    { selected, codeHits: numberInput(codeHits), tableBonus: numberInput(tableBonus) }),
  [analysis, selected, codeHits, tableBonus]);
  // A legal zero-point hand remains structurally complete even when it cannot
  // meet the three-point threshold; it can still gain a confirmed menqing bonus.
  const isWaitingPosition = waiting.status === 'waiting' || waiting.status === 'no-waits';
  const issues = isWaitingPosition ? []
    : score.status === 'invalid' || analysis.status === 'invalid' ? [...new Set(score.issues)] : [];
  const legalWinningShape = analysis.status === 'winning' && score.status !== 'invalid';
  const ready = legalWinningShape && !score.belowMinimum && typeof score.finalMultiplier === 'number' && Number.isFinite(score.finalMultiplier);
  const total = ready && typeof score.finalMultiplier === 'number' ? score.finalMultiplier : null;
  const chosenCount = selected.length;
  const needed = meldKind === 'pung' ? 3 : 4;
  const meldChoices = compatibility.meldChoices.filter(choice => choice.kind === meldKind && choice.eligible && choice.remaining >= needed);
  const blockedMeldChoices = compatibility.meldChoices.filter(choice => choice.kind === meldKind && !choice.eligible && choice.remaining >= needed);
  const activeMeldTile = meldChoices.some(choice => choice.tile === meldTile) ? meldTile : meldChoices[0]?.tile ?? '';
  const activeMeldChoice = meldChoices.find(choice => choice.tile === activeMeldTile);
  const inactiveSelections = compatibility.availableOptions.filter(option => selected.includes(option.id) && !option.eligible);
  useEffect(() => {
    if (winningTile !== activeWinningTile) setWinningTile(activeWinningTile);
  }, [winningTile, activeWinningTile]);
  useEffect(() => {
    if (meldTile !== activeMeldTile) setMeldTile(activeMeldTile);
  }, [meldTile, activeMeldTile]);

  function declareMeld() {
    if (!activeMeldChoice) return;
    setMelds(previous => [...previous, { kind: meldKind, tile: activeMeldTile }]);
    setMeldTile('');
  }

  function toggleOption(id: RuleOptionId, checked: boolean) {
    const availability = compatibility.availableOptions.find(option => option.id === id);
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
        <p>{isWaitingPosition ? waiting.status === 'waiting' ? '再进1张的基础候选见下方' : '当前牌面尚未听牌'
          : ready ? Number(codeHits) > 0 ? '中码后的最终倍数' : '当前胡牌倍数'
          : score.belowMinimum ? '不足3倍，暂不能胡' : score.status === 'not-winning' ? '当前牌面暂未组成胡牌' : '请先核对牌面与条件'}</p>
      </div>
      <span className="scoring-verdict">{isWaitingPosition ? '待进牌' : ready ? '达到起胡门槛' : legalWinningShape ? '未满3倍' : '待核对'}</span>
    </div>

    {issues.length > 0 && <div className="scoring-error" role="alert"><ul className="scoring-errors">{issues.map(issue => <li key={issue}>{issue}</li>)}</ul></div>}
    {isWaitingPosition && <section className="scoring-waits" aria-label="听牌候选" aria-live="polite">
      <div className="scoring-waits-heading"><h3>听牌</h3><span>当前{waiting.physicalTileCount}张 · 再进1张</span></div>
      {waiting.status === 'no-waits' ? <p className="scoring-hint">{waiting.issues.join('；') || '当前没有补1张即可组成胡牌的牌面，请核对手牌与碰杠。'}</p> : <>
        <WaitingTiles candidates={waiting.candidates.filter(candidate => candidate.meetsMinimum)} label="基础已满3倍" />
        <WaitingTiles candidates={waiting.candidates.filter(candidate => !candidate.meetsMinimum)} label="需特殊条件，基础不足3倍" conditional />
        <p className="scoring-hint">仅按牌面与已声明碰杠计算。门清、海捞、天胡等特殊情况、中码和本桌附加均未计入；不足3倍的候选不能直接胡。</p>
      </>}
    </section>}
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
          <label className="scoring-field">和牌张<select aria-label="和牌张" value={activeWinningTile} onChange={event => setWinningTile(event.target.value)}>
            <option value="">未指定</option>
            {availableWinningTiles.map(code => <option key={code} value={code}>{tileLabel(code)}</option>)}
          </select></label>
        </div>
        <div className="scoring-choices">{compatibility.availableOptions.map(option => {
          const checked = selected.includes(option.id);
          return <label key={option.id} className={`scoring-choice ${option.eligible ? '' : 'is-disabled'}`} title={option.reason ?? option.description}>
            <input type="checkbox" aria-label={optionLabel(option.label)} checked={checked} disabled={!option.eligible && !checked}
              onChange={event => toggleOption(option.id, event.target.checked)} />
            <span>{optionLabel(option.label)}<small>{option.eligible ? `${option.kind === 'override' ? '' : '+'}${option.multiplier}倍` : option.reason ?? '当前牌型不适用'}</small></span>
          </label>;
        })}</div>
        {inactiveSelections.length > 0 && <p className="scoring-hint" role="status">{inactiveSelections.map(option => optionLabel(option.label)).join('、')}已不适用于当前牌面，不计入倍数；可取消原选择。</p>}
        {chosenCount > 0 && <button type="button" className="scoring-button" onClick={() => setSelected([])}>清空特殊选择</button>}
      </div>
    </details>

    <details className="scoring-section">
      <summary>碰杠<span>{melds.length ? `已声明${melds.length}组` : '未声明'}</span></summary>
      <div className="scoring-section-content">
        <p className="scoring-hint">已碰、明杠、暗杠的牌也要包含在上方牌面中。这里仅分组，不会自动补牌。</p>
        {compatibility.restrictions.map(restriction => <p className="scoring-hint" key={restriction}>{restriction}</p>)}
        {melds.length > 0 && <ul className="scoring-meld-list" aria-label="已声明碰杠">{melds.map((meld, index) => <li key={`${index}-${meld.tile}-${meld.kind}`}>
          <span>{meldLabels[meld.kind]} · {tileLabel(meld.tile)} ×{meld.kind === 'pung' ? 3 : 4}</span>
          <button type="button" className="scoring-remove" aria-label={`删除第${index + 1}组${meldLabels[meld.kind]}`} onClick={() => {
            setMelds(previous => previous.filter((_, i) => i !== index));
          }}>移除</button>
        </li>)}</ul>}
        <div className="scoring-kind-options" aria-label="选择碰杠种类">{(Object.keys(meldLabels) as MeldKind[]).map(kind => <button type="button" key={kind}
          aria-pressed={meldKind === kind} onClick={() => { setMeldKind(kind); setMeldTile(''); }}>{meldLabels[kind]}</button>)}</div>
        <div className="scoring-fields"><label className="scoring-field">分组牌面<select aria-label="分组牌面" disabled={!meldChoices.length} value={activeMeldTile} onChange={event => setMeldTile(event.target.value)}>
          {!meldChoices.length && <option value="">暂无可声明牌面</option>}
          {meldChoices.map(choice => <option value={choice.tile} key={choice.tile}>{tileLabel(choice.tile)}（未分组{choice.remaining}张）</option>)}
        </select></label></div>
        {!meldChoices.length && <p className="scoring-hint">{melds.length >= 4 ? '已声明4组碰杠，不能继续分组。'
          : blockedMeldChoices[0]?.reason ?? `当前没有未分组满${needed}张的牌，不能声明${meldLabels[meldKind]}。`}</p>}
        <button type="button" className="scoring-button is-full" disabled={!activeMeldChoice} onClick={declareMeld}>声明{meldLabels[meldKind]}</button>
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
