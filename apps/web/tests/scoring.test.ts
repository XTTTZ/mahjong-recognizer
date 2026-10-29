import { describe, expect, it } from 'vitest';
import { analyzeCompatibility, analyzeHand, analyzeWaitingHand, calculateScore, scoreHand, type DeclaredMeld, type RuleOptionId } from '../src/scoring';

function tiles(notation: string): string[] {
  return [...notation.matchAll(/([1-9]+)([mpsz])/g)].flatMap(([, digits, suit]) => [...digits].map(n => n + suit));
}
function score(notation: string, selected: RuleOptionId[] = [], melds: DeclaredMeld[] = []) {
  return scoreHand({ tiles: tiles(notation), melds }, { selected });
}
const ids = (result: ReturnType<typeof score>) => result.patterns.map(p => p.id);

describe('physical hand validation and standard decomposition', () => {
  it('enumerates ambiguous legal decompositions instead of relying on one greedy split', () => {
    const input = { tiles: tiles('11122233344455m') };
    const before = JSON.stringify(input);
    const analysis = analyzeHand(input);
    expect(analysis.status).toBe('winning');
    expect(analysis.decompositions.length).toBeGreaterThan(1);
    expect(scoreHand(input).handMultiplier).toBe(9);
    expect(JSON.stringify(input)).toBe(before);
  });
  it.each(['8z', '1h', 'back', '0m', '10m'])('rejects unsupported token %s without silently removing it', bad => {
    expect(scoreHand({ tiles: [...tiles('123m456p789s111z'), bad, '2z'] }).status).toBe('invalid');
  });
  it('rejects fifth copies and declarations that consume missing physical tiles', () => {
    expect(scoreHand({ tiles: tiles('11111m222p333s44z') }).status).toBe('invalid');
    expect(score('111m222p333s444z55z', [], [{ kind: 'exposed-kong', tile: '1m' }]).status).toBe('invalid');
  });
  it('keeps a thirteen-tile hand nonwinning and cannot award manual options through it', () => {
    const result = score('123m456m789p123s5z', ['heavenly-hand']);
    expect(result.status).toBe('not-winning'); expect(result.finalMultiplier).toBe(0);
  });
  it('does not infer a kong from four matching tiles or allow an undeclared fifteen-tile hand', () => {
    const pairs = analyzeHand({ tiles: tiles('1111m2222p3333s44z') });
    expect(pairs.kongCount).toBe(0); expect(pairs.sevenPairsQuadCount).toBe(3);
    expect(score('1111m222p333s444z55z').status).toBe('invalid');
    expect(score('1111m222p333s444z55z', [], [{ kind: 'concealed-kong', tile: '1m' }]).status).toBe('winning');
  });
  it('subtracts declared pungs and kongs from the complete physical collection', () => {
    const result = score('1111m222p333s444z55z', [], [{ kind: 'exposed-kong', tile: '1m' }, { kind: 'pung', tile: '2p' }]);
    expect(result.analysis.physicalTileCount).toBe(15); expect(result.analysis.expectedTileCount).toBe(15);
    expect(result.analysis.decompositions[0].melds.filter(m => m.declared)).toHaveLength(2);
    expect(result.handMultiplier).toBe(3);
  });
});

describe('one physical tile away, without assumed context or code bonuses', () => {
  it('keeps an ordinary structural wait at zero with a visible three-point condition', () => {
    const result = analyzeWaitingHand({ tiles: tiles('123m456m789p123s5z'), winMethod: 'self-draw' });
    expect(result.status).toBe('waiting'); expect(result.requiredTileCount).toBe(13);
    expect(result.candidates.map(c => c.tile)).toEqual(['5z']);
    const candidate = result.candidates[0];
    expect(candidate.structuralWinning).toBe(true); expect(candidate.baseMultiplier).toBe(0);
    expect(candidate.meetsMinimum).toBe(false); expect(candidate.patterns).toEqual([]);
    expect(candidate.conditionHints.join(' ')).toContain('未满3倍');
    expect(candidate.conditionHints.join(' ')).toContain('未自动加3倍');
  });
  it('ignores stale winningTile and unrelated selected/code/table fields without mutating the input', () => {
    const input = { tiles: tiles('123m456m789p123s5z'), winningTile: '9m',
      selected: ['concealed-hand', 'heavenly-hand'], codeHits: 9, tableBonus: 99 };
    const before = JSON.stringify(input), result = analyzeWaitingHand(input);
    expect(result.candidates[0].tile).toBe('5z'); expect(result.candidates[0].baseMultiplier).toBe(0);
    expect(JSON.stringify(input)).toBe(before);
  });
  it('enumerates a three-sided wait rather than just the first standard decomposition', () => {
    const result = analyzeWaitingHand({ tiles: tiles('123m123p123s4567s') });
    expect(result.candidates.map(c => c.tile)).toEqual(['1s', '4s', '7s']);
    expect(result.candidates.every(c => c.baseMultiplier === 0 && !c.meetsMinimum)).toBe(true);
  });
  it('never offers a fifth copy, including copies present in declared groups', () => {
    const result = analyzeWaitingHand({ tiles: tiles('1111m234m567m899m') });
    expect(result.candidates.map(c => c.tile)).toEqual(['3m', '6m', '9m']);
    expect(result.candidates.some(c => c.tile === '1m')).toBe(false);
  });
  it.each([
    ['1122m3344p5566s7z', '7z', 4, 'seven-pairs'],
    ['1111m22p33p44s55s6z', '6z', 8, 'luxury-seven-pairs'],
    ['1111m2222p33s44s5z', '5z', 16, 'double-luxury-seven-pairs'],
    ['1111z2222z3333z4z', '4z', 68, 'triple-luxury-seven-pairs'],
  ] as const)('recognizes the tile-only seven-pairs tier in %s', (hand, winningTile, value, patternId) => {
    const result = analyzeWaitingHand({ tiles: tiles(hand) });
    expect(result.candidates.map(c => c.tile)).toEqual([winningTile]);
    expect(result.candidates[0].baseMultiplier).toBe(value);
    expect(result.candidates[0].patterns.map(p => p.id)).toContain(patternId);
    expect(result.candidates[0].meetsMinimum).toBe(true);
  });
  it('enumerates all thirteen orphan waits without automatically awarding the flying bonus', () => {
    const result = analyzeWaitingHand({ tiles: tiles('19m19p19s1234567z'), winMethod: 'rob-concealed-kong' });
    expect(result.candidates.map(c => c.tile)).toEqual(['1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z', '4z', '5z', '6z', '7z']);
    expect(result.candidates.every(c => c.baseMultiplier === 13)).toBe(true);
  });
  it('recognizes the single missing orphan and nine-sided nine-gates wait at base24', () => {
    expect(analyzeWaitingHand({ tiles: tiles('119m19p19s123456z') }).candidates.map(c => c.tile)).toEqual(['7z']);
    const result = analyzeWaitingHand({ tiles: tiles('1112345678999m') });
    expect(result.candidates.map(c => c.tile)).toEqual(['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m']);
    expect(result.candidates.every(c => c.baseMultiplier === 24 && c.patterns.some(p => p.id === 'nine-gates'))).toBe(true);
  });
  it('respects rob-concealed-kong instead of projecting ordinary waits as legal', () => {
    const result = analyzeWaitingHand({ tiles: tiles('123m456m789p123s5z'), winMethod: 'rob-concealed-kong' });
    expect(result.status).toBe('no-waits'); expect(result.candidates).toEqual([]);
    expect(result.issues.join(' ')).toContain('十三幺');
  });
  it('accounts for one or multiple explicitly declared kongs in the waiting physical count', () => {
    for (const [hand, melds, count] of [
      ['1111m222p333s444z5z', [{ kind: 'concealed-kong', tile: '1m' }], 14],
      ['1111m2222p333s444z5z', [{ kind: 'exposed-kong', tile: '1m' }, { kind: 'concealed-kong', tile: '2p' }], 15],
    ] as const) {
      const result = analyzeWaitingHand({ tiles: tiles(hand), melds });
      expect(result.status).toBe('waiting'); expect(result.requiredTileCount).toBe(count);
      expect(result.candidates.map(c => c.tile)).toEqual(['5z']); expect(result.candidates[0].baseMultiplier).toBe(3);
    }
    const four = analyzeWaitingHand({ tiles: tiles('1111m2222p3333s4444z5z'),
      melds: ['1m', '2p', '3s', '4z'].map(tile => ({ kind: 'concealed-kong', tile })) });
    expect(four.requiredTileCount).toBe(17); expect(four.candidates[0].baseMultiplier).toBe(32);
  });
  it('subtracts a declared pung and does not suggest menqing to an open ordinary hand', () => {
    const result = analyzeWaitingHand({ tiles: tiles('111m234m456p789s5z'), melds: [{ kind: 'pung', tile: '1m' }] });
    expect(result.candidates.map(c => c.tile)).toEqual(['5z']); expect(result.candidates[0].baseMultiplier).toBe(0);
    expect(result.candidates[0].conditionHints.join(' ')).not.toContain('可自行确认门清');
  });
  it('includes the known three-dragon mixed-pungs plus3 in candidate base and its evidence', () => {
    const result = analyzeWaitingHand({ tiles: tiles('111m222m555z666z7z') });
    const candidate = result.candidates.find(c => c.tile === '7z')!;
    expect(candidate.baseMultiplier).toBe(12);
    expect(candidate.patterns.map(p => p.id)).toEqual(['little-three-dragons', 'three-dragons-mixed-pungs']);
  });
  it.each(['1h', 'back', '8z'])('does not produce waits for invalid tile %s', invalid => {
    const result = analyzeWaitingHand({ tiles: [...tiles('123m456m789p123s'), invalid] });
    expect(result.status).toBe('invalid'); expect(result.candidates).toEqual([]);
  });
  it('rejects fifth copies and invalid declarations, and only accepts exactly one missing tile', () => {
    expect(analyzeWaitingHand({ tiles: tiles('11111m234p567s55z') }).status).toBe('invalid');
    expect(analyzeWaitingHand({ tiles: tiles('123m123p123s4567s'), melds: [{ kind: 'pung', tile: '1m' }] }).status).toBe('invalid');
    expect(analyzeWaitingHand({ tiles: tiles('123m456m789p123s') }).status).toBe('not-one-away');
    expect(analyzeWaitingHand({ tiles: tiles('123m456m789p123s55z') }).status).toBe('not-one-away');
    expect(analyzeWaitingHand({ tiles: tiles('1111m222p333s444z55z') }).status).toBe('invalid');
    expect(analyzeWaitingHand({ tiles: tiles('123456789m1357p') }).status).toBe('no-waits');
  });
});

describe('selection-aware compatibility and correctable declarations', () => {
  const option = (hand: string, id: RuleOptionId, selected: RuleOptionId[] = [], melds: DeclaredMeld[] = []) =>
    analyzeCompatibility(analyzeHand({ tiles: tiles(hand), melds }), selected).availableOptions.find(o => o.id === id)!;
  it('disables menqing for pure seven-pairs and for the final seven-pairs route of an ambiguous hand', () => {
    expect(option('1111m22p33p44s55s66z', 'concealed-hand').eligible).toBe(false);
    const ambiguous = analyzeHand({ tiles: tiles('11112222333344m') });
    expect(ambiguous.decompositions.length).toBeGreaterThan(0);
    expect(ambiguous.sevenPairsQuadCount).toBe(3);
    expect(ambiguous.availableOptions.find(o => o.id === 'concealed-hand')?.eligible).toBe(true);
    const projected = analyzeCompatibility(ambiguous).availableOptions.find(o => o.id === 'concealed-hand')!;
    expect(projected.eligible).toBe(false); expect(projected.reason).toContain('七对');
  });
  it('prevents adding four-concealed and menqing in either order, preserving stale choices for caller correction', () => {
    const hand = '111m222p333s444z55z';
    expect(option(hand, 'concealed-hand', ['four-concealed-triplets']).eligible).toBe(false);
    expect(option(hand, 'four-concealed-triplets', ['concealed-hand']).eligible).toBe(false);
    expect(option(hand, 'four-concealed-single-wait', ['concealed-hand']).eligible).toBe(false);
    const selected: RuleOptionId[] = ['four-concealed-triplets', 'concealed-hand'];
    const before = JSON.stringify(selected); analyzeCompatibility(analyzeHand({ tiles: tiles(hand) }), selected);
    expect(JSON.stringify(selected)).toBe(before);
  });
  it('exposes heavenly/earthly mutual exclusion and open/concealed-kong distinctions', () => {
    expect(option('123m456m789p123s55z', 'earthly-hand', ['heavenly-hand']).eligible).toBe(false);
    expect(option('123m456m789p123s55z', 'heavenly-hand', ['earthly-hand']).eligible).toBe(false);
    expect(option('1111m222p333s444z55z', 'four-concealed-triplets', [], [{ kind: 'exposed-kong', tile: '1m' }]).eligible).toBe(false);
    expect(option('1111m222p333s444z55z', 'four-concealed-triplets', [], [{ kind: 'concealed-kong', tile: '1m' }]).eligible).toBe(true);
    expect(option('111m222p333s444z55z', 'concealed-hand', [], [{ kind: 'pung', tile: '1m' }]).eligible).toBe(false);
  });
  it('allows a real pung declaration to switch an ambiguous luxury hand to a standard interpretation', () => {
    const hand = '11112222333344m', analysis = analyzeHand({ tiles: tiles(hand) });
    const choice = analyzeCompatibility(analysis).meldChoices.find(m => m.kind === 'pung' && m.tile === '1m')!;
    expect(choice.eligible).toBe(true); expect(choice.notes.join(' ')).toContain('七对');
    const declared = analyzeHand({ tiles: tiles(hand), melds: [{ kind: 'pung', tile: '1m' }] });
    expect(declared.status).toBe('winning'); expect(declared.sevenPairsQuadCount).toBeNull();
    expect(calculateScore(declared).patterns.map(p => p.id)).not.toContain('triple-luxury-seven-pairs');
  });
  it('does not label a pure luxury quartet as a legal kong when it cannot form the standard route', () => {
    const choices = analyzeCompatibility(analyzeHand({ tiles: tiles('1111m22p33p44s55s66z') })).meldChoices;
    expect(choices.find(c => c.tile === '1m' && c.kind === 'concealed-kong')?.eligible).toBe(false);
    expect(choices.find(c => c.tile === '1m' && c.kind === 'pung')?.eligible).toBe(false);
  });
  it('keeps actual undeclared-kong corrections available and accounts for already consumed entities', () => {
    const repair = analyzeCompatibility(analyzeHand({ tiles: tiles('1111m222p333s444z55z') }));
    expect(repair.meldChoices.find(c => c.tile === '1m' && c.kind === 'concealed-kong')?.eligible).toBe(true);
    const used = analyzeCompatibility(analyzeHand({ tiles: tiles('111m222p333s444z55z'), melds: [{ kind: 'pung', tile: '1m' }] }));
    expect(used.meldChoices.find(c => c.tile === '1m' && c.kind === 'pung')?.remaining).toBe(0);
    expect(used.meldChoices.find(c => c.tile === '1m' && c.kind === 'pung')?.eligible).toBe(false);
    expect(used.meldChoices.find(c => c.tile === '2p' && c.kind === 'exposed-kong')?.eligible).toBe(false);
  });
  it('requires correction of unsupported entities or existing invalid declarations before adding groups', () => {
    const invalid = analyzeCompatibility(analyzeHand({ tiles: [...tiles('111m222p333s444z5z'), 'back'] }));
    expect(invalid.meldChoices.every(c => !c.eligible)).toBe(true);
    const missing = analyzeCompatibility(analyzeHand({ tiles: tiles('111m222p333s444z55z'), melds: [{ kind: 'exposed-kong', tile: '1m' }] }));
    expect(missing.meldChoices.every(c => !c.eligible)).toBe(true);
  });
});

describe('minimum hand value and explicit context', () => {
  const ordinary = '123m456m789p123s55z';
  it('does not give ordinary self-draw any automatic bonus or use codes to reach three', () => {
    const result = scoreHand({ tiles: tiles(ordinary), winMethod: 'self-draw' }, { codeHits: 100 });
    expect(result.analysis.status).toBe('winning'); expect(result.status).toBe('not-winning');
    expect(result.baseMultiplier).toBe(0); expect(result.finalMultiplier).toBe(0);
    expect(result.belowMinimum).toBe(true);
  });
  it('adds confirmed context before multiplying code hits', () => {
    const result = scoreHand({ tiles: tiles(ordinary) }, { selected: ['concealed-hand'], codeHits: 2 });
    expect(result.baseMultiplier).toBe(0); expect(result.handMultiplier).toBe(3);
    expect(result.codeMultiplier).toBe(3); expect(result.finalMultiplier).toBe(9);
    expect(result.formula).toContain('3 × (1 + 2) = 9');
  });
  it('excludes menqing after a declared open group without calling the remaining hand invalid', () => {
    const result = score('111m222p333s444z55z', ['concealed-hand'], [{ kind: 'pung', tile: '1m' }]);
    expect(result.status).toBe('winning'); expect(result.handMultiplier).toBe(3);
    expect(result.exclusions.some(e => e.excludedId === 'concealed-hand')).toBe(true);
  });
  it('treats heavenly/earthly as mutually exclusive independent fifty additions', () => {
    expect(score(ordinary, ['heavenly-hand']).handMultiplier).toBe(50);
    expect(score('123m234m456m111z22z', ['earthly-hand']).handMultiplier).toBe(53);
    expect(score(ordinary, ['heavenly-hand', 'earthly-hand']).status).toBe('invalid');
  });
  it('rejects unsafe numeric settings and prevents table add-ons from bypassing invalid tiles', () => {
    expect(scoreHand({ tiles: tiles(ordinary) }, { codeHits: -1 }).status).toBe('invalid');
    expect(scoreHand({ tiles: tiles(ordinary) }, { codeHits: Number.MAX_SAFE_INTEGER }).status).toBe('invalid');
    expect(scoreHand({ tiles: tiles(ordinary) }, { tableBonus: NaN }).status).toBe('invalid');
    expect(scoreHand({ tiles: ['back'] }, { tableBonus: 100 }).status).toBe('invalid');
  });
});

describe('table values and composite deduplication', () => {
  it.each([
    ['123m234m456m111z22z', 3, 'half-flush'],
    ['111m222p333s444z55z', 3, 'all-pungs'],
    ['111m222m333m444z55z', 6, 'mixed-pungs'],
    ['123m234m456m678m99m', 6, 'full-flush'],
    ['123m456m789m111z22z', 6, 'mixed-dragon'],
    ['111m222m333m444m55m', 9, 'pure-pungs'],
    ['123m456m789m111m22m', 9, 'pure-dragon'],
    ['111m999p111z222z33z', 9, 'mixed-terminals'],
    ['111m999m111p999p11s', 20, 'pure-terminals'],
    ['111s555s777s999s77z', 20, 'red-peacock'],
    ['222s333s444s666s66z', 20, 'green-hand'],
    ['111z222z333z888p55z', 20, 'blue-hand'],
    ['111z222z333z666z77z', 20, 'all-honors'],
  ] as const)('%s follows the supplied table: %i', (hand, expected, expectedId) => {
    const result = score(hand);
    expect(result.status).toBe('winning'); expect(result.handMultiplier).toBe(expected);
    expect(ids(result)).toContain(expectedId);
  });
  it('folds mixed/pure pungs and dragons instead of adding their color base twice', () => {
    expect(ids(score('111m222m333m444z55z'))).toEqual(['mixed-pungs']);
    expect(score('123m456m789m111z22z').exclusions.some(e => e.excludedId === 'half-flush')).toBe(true);
  });
  it('takes the highest color family rather than green plus pure-flush six', () => {
    const result = score('222s333s444s666s88s');
    expect(result.handMultiplier).toBe(20); expect(ids(result)).not.toContain('full-flush');
  });
});

describe('seven pairs, true quads, and exceptions six/seven', () => {
  it.each([
    ['11m33m55p77p22s44s66z', 4, 0],
    ['1111m22p33p44s55s66z', 8, 1],
    ['1111m2222p33s44z55z', 16, 2],
    ['1111m2222p3333s44z', 32, 3],
  ] as const)('scores %s as %i without inventing declared kongs', (hand, expected, quads) => {
    const result = score(hand);
    expect(result.analysis.sevenPairsQuadCount).toBe(quads); expect(result.analysis.kongCount).toBe(0);
    expect(result.handMultiplier).toBe(expected);
  });
  it('keeps compatible color additions but replaces the base family upgrade', () => {
    expect(score('11m22m44m55m77m88m11z').handMultiplier).toBe(7);
    expect(score('11m22m44m55m77m88m99m').handMultiplier).toBe(10);
    expect(score('1111m22m33m55m77m99m').handMultiplier).toBe(14);
    expect(score('11m22m33m44m55m66m77m').handMultiplier).toBe(16);
  });
  it('does not count honor pairs as pungs for basic or single-luxury seven pairs', () => {
    const result = score('5555z66z77z11m22p33s');
    expect(result.handMultiplier).toBe(8);
    expect(ids(result).some(id => id.includes('three-dragons'))).toBe(false);
  });
  it('permits double luxury little-three and triple luxury big-three only', () => {
    expect(score('5555z6666z77z11m22p').handMultiplier).toBe(25);
    const big = score('5555z6666z7777z22m');
    expect(big.handMultiplier).toBe(48); expect(big.additions.some(a => a.id === 'three-dragons-mixed-pungs')).toBe(false);
  });
  it('retains a separate compatible mixed-terminals item when the triple-dragon pair is terminal', () => {
    const result = score('5555z6666z7777z11m');
    expect(result.handMultiplier).toBe(57);
    expect(ids(result)).toEqual(expect.arrayContaining(['triple-luxury-seven-pairs', 'big-three-dragons', 'mixed-terminals']));
  });
  it('permits triple luxury little-four plus the independent all-honors color, total68', () => {
    const result = score('1111z2222z3333z44z');
    expect(result.handMultiplier).toBe(68);
    expect(ids(result)).toEqual(expect.arrayContaining(['triple-luxury-seven-pairs', 'little-four-winds', 'all-honors']));
  });
  it('makes big-seven-stars a complete seven-pair upgrade and excludes all-honors20', () => {
    const result = score('11z22z33z44z55z66z77z', ['concealed-hand']);
    expect(result.handMultiplier).toBe(24); expect(ids(result)).toEqual(['big-seven-stars']);
    expect(result.exclusions.some(e => e.excludedId === 'all-honors')).toBe(true);
    expect(result.exclusions.some(e => e.excludedId === 'concealed-hand')).toBe(true);
  });
  it('does not mix the standard-green interpretation with an incompatible triple-seven interpretation', () => {
    const result = score('22223333444466s');
    expect(result.analysis.decompositions.length).toBeGreaterThan(0);
    expect(result.analysis.sevenPairsQuadCount).toBe(3);
    expect(result.handMultiplier).toBe(38); expect(ids(result)).not.toContain('green-hand');
  });
});

describe('honors and manual special-win evidence', () => {
  it('adds exactly3 only for a real mixed-pungs hand with little/big-three', () => {
    expect(score('555z666z77z123m456m').handMultiplier).toBe(9);
    expect(score('555z666z77z111m222m').handMultiplier).toBe(12);
    expect(score('555z666z777z123m22m').handMultiplier).toBe(16);
    expect(score('555z666z777z111m22m').handMultiplier).toBe(19);
  });
  it('scores little/big-four without adding ordinary mixed pungs', () => {
    expect(score('111z222z333z44z123m').handMultiplier).toBe(16);
    expect(score('111z222z333z444z55m').handMultiplier).toBe(24);
  });
  it('requires confirmation for four concealed and single wait, and excludes menqing', () => {
    const hand = '111m222p333s444z55z';
    expect(score(hand).handMultiplier).toBe(3);
    const result = score(hand, ['four-concealed-triplets', 'four-concealed-single-wait', 'concealed-hand']);
    expect(result.handMultiplier).toBe(13);
    expect(result.exclusions.some(e => e.excludedId === 'concealed-hand')).toBe(true);
    const badWait = scoreHand({ tiles: tiles(hand), winMethod: 'discard', winningTile: '1m' }, { selected: ['four-concealed-triplets'] });
    expect(badWait.handMultiplier).toBe(3);
  });
  it('keeps thirteen-orphans13 separate from mixed-terminals and flying+3 manual', () => {
    const hand = '19m19p19s1234567z1m';
    expect(score(hand).handMultiplier).toBe(13);
    expect(score(hand, ['thirteen-orphans-flying', 'concealed-hand']).handMultiplier).toBe(16);
    expect(scoreHand({ tiles: tiles(hand), winningTile: '9m' }, { selected: ['thirteen-orphans-flying'] }).handMultiplier).toBe(13);
    expect(scoreHand({ tiles: tiles(hand), winMethod: 'rob-concealed-kong' }).status).toBe('winning');
    expect(scoreHand({ tiles: tiles('111m222p333s444z55z'), winMethod: 'rob-concealed-kong' }).status).toBe('not-winning');
  });
  it('scores nine-gates24, optional ready+3, and known winning-tile contradictions', () => {
    const hand = '1112345678999m5m';
    expect(score(hand).handMultiplier).toBe(24);
    expect(scoreHand({ tiles: tiles(hand), winningTile: '5m' }, { selected: ['nine-gates-ready'] }).handMultiplier).toBe(27);
    expect(scoreHand({ tiles: tiles(hand), winningTile: '3m' }, { selected: ['nine-gates-ready'] }).handMultiplier).toBe(24);
  });
  it('recognizes eighteen-arhats only from four explicitly declared kongs', () => {
    const hand = '1111m2222p3333s4444z55z';
    const melds: DeclaredMeld[] = ['1m', '2p', '3s', '4z'].map(tile => ({ kind: 'concealed-kong', tile }));
    expect(score(hand).status).toBe('invalid');
    const result = score(hand, ['four-concealed-triplets', 'four-concealed-single-wait'], melds);
    expect(result.handMultiplier).toBe(32); expect(ids(result)).toContain('four-kongs');
    expect(result.warnings.join(' ')).toContain('本工具去重口径');
  });
  it('supports the explicit three-stage API and never mutates selections', () => {
    const selected: RuleOptionId[] = ['concealed-hand']; const before = JSON.stringify(selected);
    const analysis = analyzeHand({ tiles: tiles('123m456m789p123s55z') });
    expect(calculateScore(analysis, { selected, tableBonus: 2, codeHits: 1 }).finalMultiplier).toBe(10);
    expect(JSON.stringify(selected)).toBe(before);
  });
});
