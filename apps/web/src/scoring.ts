/** Rules transcribed from the user's 136-tile table. No recognition/history inference. */
export type CanonicalTile = `${1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9}${'m' | 'p' | 's'}` | `${1 | 2 | 3 | 4 | 5 | 6 | 7}z`;
export type MeldKind = 'pung' | 'exposed-kong' | 'concealed-kong';
export interface DeclaredMeld { kind: MeldKind; tile: string }
export type WinMethod = 'unspecified' | 'discard' | 'self-draw' | 'rob-exposed-kong' | 'rob-concealed-kong';
export interface ScoringHandInput { tiles: readonly string[]; melds?: readonly DeclaredMeld[]; winMethod?: WinMethod; winningTile?: string }
export type RuleOptionId = 'four-concealed-triplets' | 'four-concealed-single-wait' | 'thirteen-orphans-flying' | 'nine-gates-ready' | 'concealed-hand' | 'last-tile' | 'kong-draw' | 'heavenly-hand' | 'earthly-hand';
export interface ScoreOptions { selected?: readonly RuleOptionId[]; codeHits?: number; tableBonus?: number }
export interface PatternMatch { id: string; label: string; multiplier: number; evidence: string[] }
export interface ScoringOption { id: RuleOptionId; label: string; description: string; multiplier: number; kind: 'special' | 'bonus' | 'override' }
export interface RuleOptionAvailability extends ScoringOption { eligible: boolean; reason?: string }
export interface HandMeld { kind: 'sequence' | 'pung' | 'kong'; tile: CanonicalTile; declared?: MeldKind }
export interface HandDecomposition { pair: CanonicalTile; melds: HandMeld[] }
export type HuStatus = 'invalid' | 'not-winning' | 'winning';
export interface HandAnalysis {
  status: HuStatus; issues: string[]; warnings: string[]; input: ScoringHandInput;
  physicalTileCount: number; expectedTileCount: number; kongCount: number;
  counts: Readonly<Record<CanonicalTile, number>>; decompositions: HandDecomposition[];
  sevenPairsQuadCount: number | null; thirteenOrphans: boolean;
  patterns: PatternMatch[]; availableOptions: RuleOptionAvailability[];
}
export interface ScoreAddition { id: string; label: string; multiplier: number; evidence: string[] }
export interface ExclusionEvidence { excludedId: string; excludedLabel: string; by: string; reason: string }
export interface ScoreResult {
  status: HuStatus; analysis: HandAnalysis; baseMultiplier: number; additions: ScoreAddition[];
  handMultiplier: number; codeMultiplier: number; finalMultiplier: number; patterns: PatternMatch[];
  exclusions: ExclusionEvidence[]; issues: string[]; warnings: string[]; formula: string; belowMinimum: boolean;
}
export interface WaitingTileCandidate {
  tile: CanonicalTile; structuralWinning: true; patterns: PatternMatch[];
  /** All tile-only compatible value, including structural 三元另成混碰+3; no context/codes/table bonus. */
  baseMultiplier: number; meetsMinimum: boolean; conditionHints: string[]; exclusions: ExclusionEvidence[];
}
export interface WaitingHandAnalysis {
  status: 'invalid' | 'not-one-away' | 'no-waits' | 'waiting'; issues: string[];
  physicalTileCount: number; requiredTileCount: number; candidates: WaitingTileCandidate[];
}
export interface MeldDeclarationAvailability {
  kind: MeldKind; tile: CanonicalTile; required: number; remaining: number;
  eligible: boolean; reason?: string; notes: string[];
}
export interface HandCompatibility {
  availableOptions: RuleOptionAvailability[]; meldChoices: MeldDeclarationAvailability[]; restrictions: string[];
}
export const RULES_VERSION = 'gd-136-attached-table-compatible-sum-v1';
export const CANONICAL_TILES: readonly CanonicalTile[] = [
  ...(['m', 'p', 's'] as const).flatMap(suit => Array.from({ length: 9 }, (_, i) => `${i + 1}${suit}` as CanonicalTile)),
  ...Array.from({ length: 7 }, (_, i) => `${i + 1}z` as CanonicalTile),
];
export const SCORING_OPTIONS: readonly ScoringOption[] = [
  { id: 'four-concealed-triplets', label: '四暗刻', multiplier: 10, kind: 'special', description: '手动确认四组暗刻；不能有碰/明杠，可有暗杠。不由未声明的刻子推断。' },
  { id: 'four-concealed-single-wait', label: '四暗刻单骑 +3', multiplier: 3, kind: 'bonus', description: '手动确认四暗刻且胡将牌单骑；同时启用四暗刻。' },
  { id: 'thirteen-orphans-flying', label: '十三幺飞 +3', multiplier: 3, kind: 'bonus', description: '须自行确认飞；已指定胡牌张时，扣除后须为十三种幺九各一张。' },
  { id: 'nine-gates-ready', label: '九面听 +3', multiplier: 3, kind: 'bonus', description: '须自行确认九面听；已指定胡牌张时，扣除后须为1112345678999。' },
  { id: 'concealed-hand', label: '门清 +3', multiplier: 3, kind: 'bonus', description: '自行确认未有碰/明杠。七对系列、十三幺、四暗刻不计门清。' },
  { id: 'last-tile', label: '海捞 +3', multiplier: 3, kind: 'bonus', description: '自行确认本次胡牌符合海捞。' },
  { id: 'kong-draw', label: '杠开 +3', multiplier: 3, kind: 'bonus', description: '自行确认本次胡牌为杠后补牌胡；普通自摸不自动加。' },
  { id: 'heavenly-hand', label: '天胡 +50', multiplier: 50, kind: 'special', description: '自行确认第一轮条件；与地胡互斥，按本桌兼容相加口径加50。' },
  { id: 'earthly-hand', label: '地胡 +50', multiplier: 50, kind: 'special', description: '自行确认第一轮条件；与天胡互斥，按本桌兼容相加口径加50。' },
];

const tileSet = new Set<string>(CANONICAL_TILES);
const winds = ['1z', '2z', '3z', '4z'] as const;
const dragons = ['5z', '6z', '7z'] as const; // Existing app: white, green, red.
const orphanTiles: readonly CanonicalTile[] = ['1m', '9m', '1p', '9p', '1s', '9s', ...winds, ...dragons];
const labels: Record<string, [string, number]> = {
  'half-flush': ['混一色', 3], 'all-pungs': ['碰碰胡', 3], 'full-flush': ['清一色', 6],
  'mixed-pungs': ['混碰', 6], 'pure-pungs': ['清碰', 9],
  'mixed-dragon': ['混一条龙', 6], 'pure-dragon': ['清一条龙', 9],
  'seven-pairs': ['七对', 4], 'mixed-seven-pairs': ['混七对', 7],
  'luxury-seven-pairs': ['豪华七对', 8], 'double-luxury-seven-pairs': ['双豪七对', 16],
  'triple-luxury-seven-pairs': ['三豪七对', 32], 'consecutive-seven-pairs': ['连七对', 16],
  'little-three-dragons': ['小三元', 9], 'big-three-dragons': ['大三元', 16],
  'little-four-winds': ['小四喜', 16], 'big-four-winds': ['大四喜', 24],
  'mixed-terminals': ['混幺九', 9], 'pure-terminals': ['清幺九', 20],
  'red-peacock': ['红孔雀', 20], 'green-hand': ['绿一色', 20], 'blue-hand': ['蓝一色', 20],
  'all-honors': ['字一色', 20], 'thirteen-orphans': ['十三幺', 13],
  'nine-gates': ['九莲宝灯', 24], 'big-seven-stars': ['大七星', 24],
  'four-kongs': ['十八罗汉', 32], 'four-concealed-triplets': ['四暗刻', 10],
};
interface Candidate extends PatternMatch { covers: string[]; parts: string[]; priority: number }
interface CandidateSet { candidates: Candidate[]; exclusions: ExclusionEvidence[]; notes: string[] }
function pattern(id: string, evidence: string[], covers: string[], parts = [id], priority = 0): Candidate {
  const [label, multiplier] = labels[id]; return { id, label, multiplier, evidence, covers, parts, priority };
}
function isPungHand(d: HandDecomposition) { return d.melds.length === 4 && d.melds.every(m => m.kind !== 'sequence'); }
function suits(a: HandAnalysis) { return new Set(CANONICAL_TILES.filter(t => a.counts[t] && !t.endsWith('z')).map(t => t[1])); }
function occupied(a: HandAnalysis) { return CANONICAL_TILES.filter(t => a.counts[t] > 0); }
function hasHonor(a: HandAnalysis) { return occupied(a).some(t => t.endsWith('z')); }
function isNineGates(a: HandAnalysis) {
  const set = occupied(a); if (a.input.melds?.length || set.length !== 9 || suits(a).size !== 1 || hasHonor(a)) return false;
  const suit = set[0][1]; return a.counts[`1${suit}` as CanonicalTile] >= 3 && a.counts[`9${suit}` as CanonicalTile] >= 3
    && Array.from({ length: 9 }, (_, i) => a.counts[`${i + 1}${suit}` as CanonicalTile]).every(n => n >= 1);
}
function standardDecompositions(counts: number[], declared: HandMeld[]): HandDecomposition[] {
  const required = 4 - declared.length, results: HandDecomposition[] = [];
  function melds(work: number[], left: number): HandMeld[][] {
    const first = work.findIndex(n => n > 0);
    if (left === 0) return first < 0 ? [[]] : [];
    if (first < 0) return [];
    const found: HandMeld[][] = [];
    if (work[first] >= 3) {
      work[first] -= 3;
      for (const tail of melds(work, left - 1)) found.push([{ kind: 'pung', tile: CANONICAL_TILES[first] }, ...tail]);
      work[first] += 3;
    }
    if (first < 27 && first % 9 <= 6 && work[first + 1] && work[first + 2]) {
      work[first]--; work[first + 1]--; work[first + 2]--;
      for (const tail of melds(work, left - 1)) found.push([{ kind: 'sequence', tile: CANONICAL_TILES[first] }, ...tail]);
      work[first]++; work[first + 1]++; work[first + 2]++;
    }
    return found;
  }
  for (let pair = 0; pair < counts.length; pair++) if (counts[pair] >= 2) {
    counts[pair] -= 2;
    for (const groups of melds(counts, required)) results.push({ pair: CANONICAL_TILES[pair], melds: [...declared, ...groups] });
    counts[pair] += 2;
  }
  const unique = new Map(results.map(d => [JSON.stringify([d.pair, d.melds.map(m => `${m.kind}:${m.tile}:${m.declared ?? ''}`).sort()]), d]));
  return [...unique.values()];
}
function honorPatterns(a: HandAnalysis, kind: 'dragons' | 'winds') {
  const values = kind === 'dragons' ? dragons : winds;
  let small = false, big = false;
  for (const d of a.decompositions) {
    const groups = new Set(d.melds.filter(m => m.kind !== 'sequence').map(m => m.tile));
    const n = values.filter(t => groups.has(t)).length;
    big ||= n === values.length;
    small ||= n === values.length - 1 && (values as readonly string[]).includes(d.pair) && !groups.has(d.pair);
  }
  // In seven-pairs, a quartet is two pairs, never an inferred kong. Only the
  // explicit double/triple-luxury exceptions permit its role in three/four honors.
  const q = a.sevenPairsQuadCount;
  if (q !== null && ((kind === 'dragons' && q >= 2) || (kind === 'winds' && q >= 3))) {
    const quads = values.filter(t => a.counts[t] === 4).length;
    big ||= quads === values.length;
    small ||= quads === values.length - 1 && values.some(t => a.counts[t] === 2);
  }
  return { small, big };
}
function buildCandidates(a: HandAnalysis, selected: ReadonlySet<RuleOptionId>): CandidateSet {
  const candidates: Candidate[] = [], exclusions: ExclusionEvidence[] = [], notes: string[] = [];
  if (a.status !== 'winning') return { candidates, exclusions, notes };
  const set = occupied(a), numberedSuits = suits(a), honor = hasHonor(a), standard = a.decompositions.length > 0;
  const half = numberedSuits.size === 1 && honor, full = numberedSuits.size === 1 && !honor;
  const pungs = a.decompositions.some(isPungHand), q = a.sevenPairsQuadCount;
  const add = (id: string, why: string, covers: string[], parts?: string[], priority?: number) => candidates.push(pattern(id, [why], covers, parts, priority));
  if (half) add('half-flush', '数牌只含一种花色，同时有字牌。', ['color']);
  if (full) add('full-flush', '全部是同一种数牌花色。', ['color']);
  if (pungs) add('all-pungs', '存在四组刻子/已声明杠和一对将的标准分解；不代表四组都是暗刻。', ['pung']);
  if (pungs && half) add('mixed-pungs', '混一色3与碰碰胡3合成6，基础不重复。', ['color', 'pung'], ['half-flush', 'all-pungs'], 2);
  if (pungs && full) add('pure-pungs', '清一色6与碰碰胡3合成9，基础不重复。', ['color', 'pung'], ['full-flush', 'all-pungs'], 2);
  const dragon = standard && numberedSuits.size === 1 && a.decompositions.some(d => {
    const starts = new Set(d.melds.filter(m => m.kind === 'sequence').map(m => m.tile));
    const suit = [...numberedSuits][0]; return [1, 4, 7].every(n => starts.has(`${n}${suit}` as CanonicalTile));
  });
  if (dragon && half) add('mixed-dragon', '同一标准分解含同花123、456、789，余牌只有该花色与字牌。', ['color', 'dragon'], ['half-flush', 'dragon'], 2);
  if (dragon && full) add('pure-dragon', '同一标准分解含同花123、456、789，全部为该花色。', ['color', 'dragon'], ['full-flush', 'dragon'], 2);
  if (q !== null) {
    const pairIds = ['seven-pairs', 'luxury-seven-pairs', 'double-luxury-seven-pairs', 'triple-luxury-seven-pairs'];
    let id = pairIds[q];
    const ranks = set.map(t => Number(t[0])).sort((x, y) => x - y);
    const consecutive = q === 0 && full && set.length === 7 && ranks.every((n, i) => n === ranks[0] + i);
    const stars = q === 0 && set.length === 7 && set.every(t => t.endsWith('z'));
    if (consecutive) id = 'consecutive-seven-pairs';
    if (stars) id = 'big-seven-stars';
    add(id, `${q}种牌各有四张；四张按两对，未推断为杠。${consecutive ? '七种连续同花对子。' : stars ? '东南西北白发中各一对。' : ''}`, ['seven-pairs'], [id], 3);
    if (id !== 'seven-pairs') exclusions.push({ excludedId: 'seven-pairs', excludedLabel: '七对', by: labels[id][0], reason: '同一七对系列只计对应升级档，不再加基本七对4。' });
    if (q === 0 && half) add('mixed-seven-pairs', '七对4与混一色3合成7，基础不重复。', ['seven-pairs', 'color'], ['seven-pairs', 'half-flush'], 2);
  }
  const three = honorPatterns(a, 'dragons'), four = honorPatterns(a, 'winds');
  if (three.big) add('big-three-dragons', '中、发、白分别组成刻子/声明杠，或三豪七对允许的三种四张。', ['three-dragons']);
  else if (three.small) add('little-three-dragons', '中发白中两种刻子/允许的四张，第三种作将/对子。', ['three-dragons']);
  if (four.big) add('big-four-winds', '东南西北四副刻子/声明杠。', ['four-winds']);
  else if (four.small) add('little-four-winds', '三种风牌刻子/三豪允许的四张，第四种风牌作将/对子。', ['four-winds']);
  const terminalsOnly = set.every(t => !t.endsWith('z') && (t[0] === '1' || t[0] === '9'));
  const mixedTerminals = honor && numberedSuits.size > 0 && set.every(t => t.endsWith('z') || t[0] === '1' || t[0] === '9');
  if (terminalsOnly && (standard || q !== null)) add('pure-terminals', '全部数牌均为1或9。', ['terminals']);
  else if (mixedTerminals && (standard || q !== null)) add('mixed-terminals', '由数牌1/9与字牌共同组成。', ['terminals']);
  if (set.every(t => t.endsWith('z'))) add('all-honors', '全手都是字牌；标准结构或七对结构已单独验证。', ['color']);
  if (standard && pungs && set.every(t => new Set(['1s', '5s', '7s', '9s', '7z']).has(t))) add('red-peacock', '全部为1/5/7/9条与红中，按刻子/声明杠和将分解。', ['color'], undefined, 2);
  if (standard && set.every(t => new Set(['2s', '3s', '4s', '6s', '8s', '6z']).has(t))) add('green-hand', '只含2/3/4/6/8条与发，已通过刻子/顺子/将分解。', ['color'], undefined, 2);
  if (standard && pungs && set.every(t => new Set(['1z', '2z', '3z', '4z', '5z', '8p']).has(t))) add('blue-hand', '只含东南西北白与8筒，按刻子/声明杠和将分解。', ['color'], undefined, 2);
  if (a.thirteenOrphans) add('thirteen-orphans', '六种数牌1/9及七种字牌各至少一张，另有一对，无声明副露。', ['orphans']);
  if (isNineGates(a)) add('nine-gates', '单一数牌花色满足1112345678999再加同花一张，无声明副露。', ['nine-gates']);
  if (a.kongCount === 4 && standard) add('four-kongs', '四副杠均已由用户明确声明，余下两张为将。', ['pung'], ['four-kongs'], 4);
  if (selected.has('four-concealed-triplets') || selected.has('four-concealed-single-wait')) add('four-concealed-triplets', '四暗刻由用户确认；结构与已声明碰/杠已核对。', ['pung'], undefined, 2);

  const suppress = (by: string, blocked: string[], reason: string) => {
    for (const candidate of candidates) if (candidate.parts.some(p => blocked.includes(p))) {
      if (!exclusions.some(e => e.excludedId === candidate.id && e.by === by)) exclusions.push({ excludedId: candidate.id, excludedLabel: candidate.label, by, reason });
    }
  };
  if (three.small || three.big) suppress(three.big ? '大三元' : '小三元', ['all-pungs', 'half-flush'], '表内：三元不计碰碰胡、混一色；只有另成混碰才加3。');
  if (four.small || four.big) suppress(four.big ? '大四喜' : '小四喜', ['all-pungs', 'half-flush'], '表内：四喜不计碰碰胡、混一色。');
  if (mixedTerminals || terminalsOnly) suppress(terminalsOnly ? '清幺九' : '混幺九', ['all-pungs'], '表内：幺九不计碰碰胡。');
  for (const color of ['red-peacock', 'green-hand', 'blue-hand', 'all-honors']) if (candidates.some(p => p.id === color)) suppress(labels[color][0], ['all-pungs', 'half-flush'], '表内：该色雀牌型不计碰碰胡、混一色。');
  if (candidates.some(p => p.id === 'consecutive-seven-pairs')) suppress('连七对', ['full-flush'], '表内：连七对不计清一色、基本七对。');
  if (candidates.some(p => p.id === 'big-seven-stars')) suppress('大七星', ['all-honors'], '表内：大七星不计字一色；七对系列升级不重复。');
  if (candidates.some(p => p.id === 'nine-gates')) {
    suppress('九莲宝灯', ['full-flush'], '表内不计清一色；含清一色基础的清龙复合项按本工具去重口径不重复。');
    if (dragon) notes.push('九莲与清龙的交叉未逐字说明；本工具按清一色基础不重复处理，可用本桌附加协商。');
  }
  if (candidates.some(p => p.id === 'four-kongs') && selected.has('four-concealed-triplets')) notes.push('十八罗汉与四暗刻按同一刻杠系列取升级高项，这是本工具去重口径，非图片明文排斥。');
  const blocked = new Set(exclusions.filter(e => candidates.some(c => c.id === e.excludedId)).map(e => e.excludedId));
  return { candidates: candidates.filter(c => !blocked.has(c.id)), exclusions, notes };
}
function selectCandidates(set: CandidateSet) {
  let best: Candidate[] = [], bestScore = -1, bestPriority = -1;
  function search(index: number, selected: Candidate[], covers: Set<string>, score: number, priority: number) {
    if (index === set.candidates.length) {
      if (score > bestScore || (score === bestScore && (selected.length < best.length || (selected.length === best.length && priority > bestPriority)))) {
        best = [...selected]; bestScore = score; bestPriority = priority;
      }
      return;
    }
    const item = set.candidates[index]; search(index + 1, selected, covers, score, priority);
    if (!item.covers.some(key => covers.has(key))) search(index + 1, [...selected, item], new Set([...covers, ...item.covers]), score + item.multiplier, priority + item.priority);
  }
  search(0, [], new Set(), 0, 0);
  const exclusions = [...set.exclusions];
  for (const item of set.candidates) if (!best.includes(item)) {
    const conflicting = best.filter(p => p.covers.some(c => item.covers.includes(c)));
    exclusions.push({ excludedId: item.id, excludedLabel: item.label, by: conflicting.map(p => p.label).join('、') || '同系列去重', reason: item.covers.includes('color') && conflicting.some(p => p.covers.includes('color')) ? '按本桌口径：颜色同族取最高；复合项覆盖的基础不重复。' : '同一牌型系列/复合基础不重复，取最高兼容组合。' });
  }
  return { chosen: best, exclusions, notes: set.notes };
}
function availableOptions(a: HandAnalysis): RuleOptionAvailability[] {
  const pungs = a.decompositions.filter(isPungHand), exposed = (a.input.melds ?? []).some(m => m.kind !== 'concealed-kong');
  const discard = ['discard', 'rob-exposed-kong', 'rob-concealed-kong'].includes(a.input.winMethod ?? 'unspecified');
  const tile = a.input.winningTile, pairWait = !tile || pungs.some(d => d.pair === tile);
  return SCORING_OPTIONS.map(option => {
    let reason: string | undefined;
    if (a.status !== 'winning') reason = '须先组成合法和牌结构。';
    else if (option.id.startsWith('four-concealed')) {
      if (!pungs.length || exposed) reason = '四暗刻需四组刻子/暗杠，且不能有声明的碰/明杠。';
      else if (discard && tile && !pairWait) reason = '所选吃胡张不是将牌，会把一组刻子变成明刻。';
      else if (option.id === 'four-concealed-single-wait' && tile && !pairWait) reason = '所选和牌张不是该全刻分解的将牌，不能计单骑。';
    } else if (option.id === 'thirteen-orphans-flying') {
      if (!a.thirteenOrphans) reason = '只适用于十三幺。';
      else if (tile && a.counts[tile as CanonicalTile] !== 2) reason = '扣除所选和牌张后不是十三种幺九各一张。';
    } else if (option.id === 'nine-gates-ready') {
      if (!isNineGates(a)) reason = '只适用于九莲宝灯。';
      else if (tile) {
        const suit = tile[1];
        if (!Array.from({ length: 9 }, (_, i) => a.counts[`${i + 1}${suit}` as CanonicalTile] - (`${i + 1}${suit}` === tile ? 1 : 0)).every((n, i) => n === (i === 0 || i === 8 ? 3 : 1))) reason = '扣除所选和牌张后不是1112345678999。';
      }
    } else if (option.id === 'concealed-hand') {
      if (exposed) reason = '已声明碰/明杠，不适用门清。';
      else if (a.sevenPairsQuadCount !== null || a.thirteenOrphans) reason = '表内：七对系列、十三幺不计门清。';
      else if ((a.input.melds ?? []).length === 4 && a.input.melds?.every(m => m.kind === 'concealed-kong')) reason = '四副暗杠已明确属于四暗刻结构，不再计门清。';
    }
    return { ...option, eligible: !reason, ...(reason ? { reason } : {}) };
  });
}

function scoringViews(a: HandAnalysis): HandAnalysis[] {
  if (a.status !== 'winning') return [a];
  const views: HandAnalysis[] = [];
  if (a.decompositions.length) views.push({ ...a, sevenPairsQuadCount: null, thirteenOrphans: false });
  if (a.sevenPairsQuadCount !== null) views.push({ ...a, decompositions: [], thirteenOrphans: false });
  if (a.thirteenOrphans) views.push({ ...a, decompositions: [], sevenPairsQuadCount: null });
  return views;
}
function aggregateOptions(a: HandAnalysis): RuleOptionAvailability[] {
  const options = scoringViews(a).map(view => availableOptions(view));
  return SCORING_OPTIONS.map((option, index) => {
    const eligible = options.some(group => group[index].eligible);
    return { ...option, eligible, ...(!eligible ? { reason: options[0][index].reason } : {}) };
  });
}

export function analyzeHand(input: ScoringHandInput): HandAnalysis {
  const tiles = Array.isArray(input.tiles) ? [...input.tiles] : [];
  const meldInput = Array.isArray(input.melds) ? input.melds.map(m => ({ ...m })) : [];
  const normalized: ScoringHandInput = { tiles, melds: meldInput, winMethod: input.winMethod ?? 'unspecified', ...(input.winningTile ? { winningTile: input.winningTile } : {}) };
  const counts = Object.fromEntries(CANONICAL_TILES.map(t => [t, 0])) as Record<CanonicalTile, number>;
  const a: HandAnalysis = { status: 'invalid', issues: [], warnings: [], input: normalized, physicalTileCount: tiles.length, expectedTileCount: 14, kongCount: 0, counts, decompositions: [], sevenPairsQuadCount: null, thirteenOrphans: false, patterns: [], availableOptions: [] };
  for (const token of tiles) {
    if (!tileSet.has(token)) a.issues.push(`不支持牌值“${String(token)}”：136张规则只接受万筒条及七种字牌，请移除花牌/牌背并核对。`);
    else counts[token as CanonicalTile]++;
  }
  for (const tile of CANONICAL_TILES) if (counts[tile] > 4) a.issues.push(`${tile}有${counts[tile]}张，超过一副牌最多4张。`);
  if (meldInput.length > 4) a.issues.push('最多声明四副碰/杠。');
  if (!['unspecified', 'discard', 'self-draw', 'rob-exposed-kong', 'rob-concealed-kong'].includes(normalized.winMethod!)) a.issues.push('和牌方式无效。');
  if (input.winningTile && (!tileSet.has(input.winningTile) || counts[input.winningTile as CanonicalTile] === 0)) a.issues.push('和牌张必须是当前牌中的一种有效牌值。');
  const remaining = CANONICAL_TILES.map(t => counts[t]), declared: HandMeld[] = [];
  for (const meld of meldInput) {
    if (!['pung', 'exposed-kong', 'concealed-kong'].includes(meld.kind) || !tileSet.has(meld.tile)) { a.issues.push('碰/杠声明无效。'); continue; }
    const index = CANONICAL_TILES.indexOf(meld.tile as CanonicalTile), number = meld.kind === 'pung' ? 3 : 4;
    if (remaining[index] < number) a.issues.push(`${meld.tile}的${meld.kind === 'pung' ? '碰' : '杠'}需要${number}张实体牌；声明包含在总牌里，不能额外凭空补牌。`);
    else remaining[index] -= number;
    if (number === 4) a.kongCount++;
    declared.push({ kind: number === 4 ? 'kong' : 'pung', tile: meld.tile as CanonicalTile, declared: meld.kind });
  }
  a.expectedTileCount = 14 + a.kongCount;
  if (tiles.length > a.expectedTileCount) a.issues.push(`当前${tiles.length}张，已声明${a.kongCount}杠时应为${a.expectedTileCount}张；请核对或声明杠，四张同牌不会自动算杠。`);
  if (a.issues.length) { a.availableOptions = availableOptions(a); return a; }
  if (tiles.length !== a.expectedTileCount) {
    a.status = 'not-winning'; a.issues.push(`当前${tiles.length}张，需${a.expectedTileCount}张（14＋已声明杠数）才能构成完整和牌。`);
    a.availableOptions = availableOptions(a); return a;
  }
  a.decompositions = standardDecompositions([...remaining], declared);
  if (!declared.length && tiles.length === 14 && CANONICAL_TILES.every(t => counts[t] % 2 === 0)) a.sevenPairsQuadCount = CANONICAL_TILES.filter(t => counts[t] === 4).length;
  a.thirteenOrphans = !declared.length && tiles.length === 14 && occupied(a).every(t => orphanTiles.includes(t)) && orphanTiles.every(t => counts[t] >= 1);
  a.status = a.decompositions.length || a.sevenPairsQuadCount !== null || a.thirteenOrphans ? 'winning' : 'not-winning';
  if (a.status !== 'winning') a.issues.push('无法分成四组刻子/顺子加一对将，也不满足七对或十三幺。');
  if (normalized.winMethod === 'rob-concealed-kong' && !a.thirteenOrphans) { a.status = 'not-winning'; a.issues.push('表内只允许十三幺抢暗杠。'); }
  if (normalized.winMethod?.startsWith('rob-')) a.warnings.push('抢杠按自摸结算；普通自摸和抢杠本身不自动加3倍。');
  if (a.status === 'winning') a.patterns = scoringViews(a).map(view => selectCandidates(buildCandidates(view, new Set())).chosen).sort((x, y) => y.reduce((sum, p) => sum + p.multiplier, 0) - x.reduce((sum, p) => sum + p.multiplier, 0))[0];
  a.availableOptions = aggregateOptions(a); return a;
}

function calculateSingle(analysis: HandAnalysis, options: ScoreOptions = {}): ScoreResult {
  const codeHits = options.codeHits ?? 0, tableBonus = options.tableBonus ?? 0, selected = new Set(options.selected ?? []);
  const result: ScoreResult = { status: analysis.status, analysis, baseMultiplier: 0, additions: [], handMultiplier: 0, codeMultiplier: 1, finalMultiplier: 0, patterns: [], exclusions: [], issues: [...analysis.issues], warnings: [...analysis.warnings], formula: '尚未形成可计倍的和牌。', belowMinimum: false };
  for (const [name, value] of [['中码数', codeHits], ['本桌附加', tableBonus]] as const) if (!Number.isSafeInteger(value) || value < 0) result.issues.push(`${name}必须是非负安全整数。`);
  if (!Number.isSafeInteger(1 + codeHits)) result.issues.push('中码乘数超出安全整数范围。');
  for (const id of selected) if (!SCORING_OPTIONS.some(o => o.id === id)) result.issues.push(`未知手选项：${String(id)}。`);
  if (selected.has('heavenly-hand') && selected.has('earthly-hand')) result.issues.push('天胡与地胡互斥，不能同时选择。');
  if (result.issues.length > analysis.issues.length) { result.status = 'invalid'; return result; }
  result.codeMultiplier = 1 + codeHits;
  if (analysis.status !== 'winning') return result;
  const active = new Set<RuleOptionId>();
  for (const id of selected) {
    const option = analysis.availableOptions.find(o => o.id === id)!;
    if (option.eligible) active.add(id);
    else result.exclusions.push({ excludedId: id, excludedLabel: option.label, by: '结构/声明约束', reason: option.reason! });
  }
  if (active.has('four-concealed-single-wait')) active.add('four-concealed-triplets');
  const candidateSet = buildCandidates(analysis, active), scored = selectCandidates(candidateSet);
  result.patterns = scored.chosen; result.exclusions.push(...scored.exclusions); result.warnings.push(...scored.notes);
  result.baseMultiplier = result.patterns.reduce((total, item) => total + item.multiplier, 0);
  const three = honorPatterns(analysis, 'dragons'), mixedPungs = analysis.decompositions.some(isPungHand) && suits(analysis).size === 1 && hasHonor(analysis);
  if ((three.small || three.big) && mixedPungs && analysis.sevenPairsQuadCount === null) result.additions.push({ id: 'three-dragons-mixed-pungs', label: '三元另成混碰', multiplier: 3, evidence: ['表内：小/大三元不计普通碰碰/混一色；本手另满足真正混碰，故额外+3。'] });
  for (const option of SCORING_OPTIONS) if (active.has(option.id) && option.id !== 'four-concealed-triplets') {
    let blocked: string | undefined;
    if (option.id === 'concealed-hand' && active.has('four-concealed-triplets')) blocked = '表内：四暗刻不计门清。';
    if (option.id === 'four-concealed-single-wait' && !result.patterns.some(p => p.id === 'four-concealed-triplets')) blocked = '四暗刻被十八罗汉的同系列升级覆盖，单骑附项按本工具口径不再重复。';
    if (blocked) result.exclusions.push({ excludedId: option.id, excludedLabel: option.label, by: '同系列/不叠加规则', reason: blocked });
    else result.additions.push({ id: option.id, label: option.label, multiplier: option.multiplier, evidence: ['用户手动确认；未从照片或普通自摸方式推断。', option.description] });
  }
  if (tableBonus) result.additions.push({ id: 'table-bonus', label: '本桌附加', multiplier: tableBonus, evidence: ['用户自行约定的额外加项，先相加再乘中码。'] });
  result.handMultiplier = result.baseMultiplier + result.additions.reduce((total, item) => total + item.multiplier, 0);
  const product = result.handMultiplier * result.codeMultiplier;
  if (!Number.isSafeInteger(product)) { result.status = 'invalid'; result.issues.push('倍数超出安全计算范围，请减少输入。'); return result; }
  result.belowMinimum = result.handMultiplier < 3;
  result.status = result.belowMinimum ? 'not-winning' : 'winning';
  result.finalMultiplier = result.belowMinimum ? 0 : product;
  result.formula = result.belowMinimum ? `${result.handMultiplier}倍未满3倍起胡；中码不用于凑起胡。` : `${result.handMultiplier} × (1 + ${codeHits}) = ${product}倍`;
  if (result.belowMinimum) result.issues.push('结构可和，但当前牌型及手选加项未满3倍，无鸡胡。');
  result.warnings.push('按用户选定的兼容相加口径；同系列升级/颜色同族/复合基础已去重。包三家、漏胡漏碰、买码命中和暗刻历史仍需人工确认。');
  return result;
}
export function calculateScore(analysis: HandAnalysis, options: ScoreOptions = {}): ScoreResult {
  const results = scoringViews(analysis).map(view => calculateSingle({ ...view, availableOptions: availableOptions(view) }, options));
  const best = results.sort((x, y) => y.handMultiplier - x.handMultiplier)[0];
  best.analysis = analysis;
  if (scoringViews(analysis).length > 1) best.warnings.push('此手有多种合法分解；按最高兼容的一种结构计算，不混加标准分解与七对专属项。');
  return best;
}

/** Enumerate one physical missing tile, with the current method but no guessed history. */
export function analyzeWaitingHand(input: ScoringHandInput): WaitingHandAnalysis {
  // A previously selected winning tile belongs to the completed hand. For a
  // waiting hand the enumerated new tile is the winning tile, even if absent now.
  const waitingInput: ScoringHandInput = { tiles: input.tiles, melds: input.melds, winMethod: input.winMethod };
  const analysis = analyzeHand(waitingInput);
  const result: WaitingHandAnalysis = {
    status: 'invalid', issues: [], physicalTileCount: analysis.physicalTileCount,
    requiredTileCount: analysis.expectedTileCount - 1, candidates: [],
  };
  if (analysis.status === 'invalid') { result.issues = [...analysis.issues]; return result; }
  if (analysis.physicalTileCount !== result.requiredTileCount) {
    result.status = 'not-one-away';
    result.issues.push(`听牌分析需${result.requiredTileCount}张实体牌（13＋已声明杠数），当前为${analysis.physicalTileCount}张；请先补齐或核对碰杠。`);
    return result;
  }
  for (const tile of CANONICAL_TILES) {
    // Counts include every physical tile in declared groups, not only the
    // remaining concealed hand. A fifth copy cannot be a waiting candidate.
    if (analysis.counts[tile] >= 4) continue;
    const completed = analyzeHand({ ...waitingInput, tiles: [...analysis.input.tiles, tile], winningTile: tile });
    if (completed.status !== 'winning') continue;
    const bare = calculateScore(completed);
    const structuralAdditions = bare.additions.map(({ id, label, multiplier, evidence }) => ({ id, label, multiplier, evidence: [...evidence] }));
    const baseMultiplier = bare.handMultiplier;
    result.candidates.push({ tile, structuralWinning: true,
      patterns: [...bare.patterns, ...structuralAdditions], baseMultiplier,
      meetsMinimum: baseMultiplier >= 3,
      conditionHints: baseMultiplier < 3
        ? [`牌面可组成和牌结构，但基础${baseMultiplier}倍未满3倍；须实际符合并确认允许的特殊条件，中码不能凑起胡。`,
          ...(completed.availableOptions.some(o => o.id === 'concealed-hand' && o.eligible)
            ? ['如实际未有碰/明杠，可自行确认门清；这里未自动加3倍。'] : [])]
        : [],
      exclusions: bare.exclusions.map(item => ({ ...item })),
    });
  }
  result.status = result.candidates.length ? 'waiting' : 'no-waits';
  if (!result.candidates.length) result.issues.push(analysis.input.winMethod === 'rob-concealed-kong'
    ? '当前牌面补一张后不满足十三幺；本表只有十三幺允许抢暗杠。'
    : '当前实体牌与碰杠声明补一张后，没有符合本表和牌结构的牌。');
  return result;
}

/** Selection-aware explanations for the UI; declarations remain user-correctable. */
export function analyzeCompatibility(analysis: HandAnalysis, selected: readonly RuleOptionId[] = []): HandCompatibility {
  const selectedSet = new Set(selected), baseOptions = aggregateOptions(analysis);
  const result: HandCompatibility = { availableOptions: [], meldChoices: [], restrictions: [] };
  result.availableOptions = baseOptions.map(option => {
    let reason = option.reason;
    if (option.eligible) {
      if ((option.id === 'heavenly-hand' && selectedSet.has('earthly-hand'))
          || (option.id === 'earthly-hand' && selectedSet.has('heavenly-hand'))) reason = '天胡与地胡互斥，不能同时选择。';
      else if (option.id.startsWith('four-concealed') && selectedSet.has('concealed-hand')
          && baseOptions.some(o => o.id === 'concealed-hand' && o.eligible)) reason = '四暗刻与门清不叠加；请先取消门清，再按实际牌局确认四暗刻。';
      else if (option.id === 'concealed-hand' && (selectedSet.has('four-concealed-triplets') || selectedSet.has('four-concealed-single-wait'))
          && baseOptions.some(o => o.id === 'four-concealed-triplets' && o.eligible)) reason = '表内：四暗刻不计门清。';
      else {
        // Eligibility in one interpretation does not make a bonus compatible
        // with another interpretation that wins the final calculation. Check
        // its actual counterfactual result, rather than unioning route bonuses.
        const proposed = calculateScore(analysis, { selected: [...selectedSet, option.id] });
        const awarded = proposed.patterns.some(p => p.id === option.id) || proposed.additions.some(p => p.id === option.id);
        if (!awarded) reason = proposed.exclusions.find(item => item.excludedId === option.id)?.reason;
      }
    }
    return { ...option, eligible: option.eligible && !reason, ...(reason ? { reason } : {}) };
  });

  const counts = new Map<string, number>();
  for (const tile of analysis.input.tiles) counts.set(tile, (counts.get(tile) ?? 0) + 1);
  const remaining = new Map(counts), melds = analysis.input.melds ?? [];
  let declarationIssue: string | undefined;
  if ([...counts].some(([tile, count]) => !tileSet.has(tile) || count > 4)) declarationIssue = '请先修正花牌、牌背、无效牌值或超过4张的牌。';
  if (melds.length > 4) declarationIssue = '已超过四副碰杠，请先移除错误声明。';
  for (const meld of melds) {
    const needed = meld.kind === 'pung' ? 3 : 4;
    if (!tileSet.has(meld.tile) || !['pung', 'exposed-kong', 'concealed-kong'].includes(meld.kind)
        || (remaining.get(meld.tile) ?? 0) < needed) declarationIssue = '已有碰杠声明与实体牌不符，请先修正或移除。';
    else remaining.set(meld.tile, (remaining.get(meld.tile) ?? 0) - needed);
  }
  const pairRoute = analysis.sevenPairsQuadCount !== null;
  for (const kind of ['pung', 'exposed-kong', 'concealed-kong'] as const) for (const tile of CANONICAL_TILES) {
    const required = kind === 'pung' ? 3 : 4, count = Math.max(0, remaining.get(tile) ?? 0);
    let reason = declarationIssue;
    if (!reason && melds.length >= 4) reason = '最多声明四副碰杠，请先移除或修正已有分组。';
    if (!reason && count < required) reason = `未分组的${tile}只有${count}张，${kind === 'pung' ? '碰' : '杠'}需要${required}张实体牌。`;
    if (!reason && analysis.status === 'winning') {
      const prospective: ScoringHandInput = { ...analysis.input, melds: [...melds, { kind, tile }] };
      const declared = analyzeHand(prospective);
      if (declared.status !== 'winning' && !analyzeWaitingHand(prospective).candidates.length)
        reason = '这项声明不能保留当前和牌或形成差一张的标准结构；请先核对/更正实体牌，不可把七对四张直接算杠。';
    }
    result.meldChoices.push({ kind, tile, required, remaining: count, eligible: !reason, ...(reason ? { reason } : {}),
      notes: pairRoute ? ['新增任何碰/杠声明后，七对及豪华七对系列不再适用；仍会重新检查标准刻顺将分解。'] : [] });
  }
  if (pairRoute || melds.length) result.restrictions.push('七对及豪华七对系列不能有任何碰/明杠/暗杠声明，也不叠加门清。实际有副露时请修正声明并重新计算标准结构。');
  if (analysis.thirteenOrphans) result.restrictions.push('十三幺不能有碰杠声明，也不叠加门清。');
  if (melds.some(m => m.kind !== 'concealed-kong')) result.restrictions.push('已声明碰/明杠，门清与四暗刻不可选；暗杠本身不等于明杠。');
  if (selectedSet.has('four-concealed-triplets') || selectedSet.has('four-concealed-single-wait')) result.restrictions.push('四暗刻与门清不重复计入；四暗刻和单骑仍须确认真实和牌经过。');
  if (selectedSet.has('heavenly-hand') || selectedSet.has('earthly-hand')) result.restrictions.push('天胡与地胡互斥。');
  if (analysis.input.winMethod === 'rob-concealed-kong') result.restrictions.push('本表只允许十三幺抢暗杠。');
  if (declarationIssue) result.restrictions.push(declarationIssue);
  return result;
}
export function scoreHand(input: ScoringHandInput, options: ScoreOptions = {}): ScoreResult { return calculateScore(analyzeHand(input), options); }
