// The trainable half: tokens -> scores for every choice a date program can make.
// One tiny transformer. Its first token is the convention vector u, which also serves
// as the summary position the choice heads read from.

import { type Day } from './calendar.ts'
import { H, K, MAX_TOKENS, NO_DATE, type Programs, type Tok, enumerate, lex } from './programs.ts'
import { DURS, NO_TIME, TH, type Times, endOf, enumerateTimes } from './time.ts'
import { T, add, constant, embedBag, hcat, marginalNLL, matmul, pickSum, softmax1, vcat } from './tensor.ts'
import { initTransformer, transformer } from './transformer.ts'

/** timeWords: the corpus words that say when (scripts/event.ts finds them); time.words does not read them. */
export type Config = { d: number; heads: number; layers: number; ff: number; buckets: number; k: number; timeWords?: string[] }
export const DEFAULT_CONFIG: Config = { d: 48, heads: 4, layers: 2, ff: 96, buckets: 2048, k: 8 }

export type Params = Record<string, T>

export function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const NUM_FEATURES = 13
export function numFeatures(t: Tok): number[] {
  const v = t.value, n = t.digits
  return [1, +(n === 1), +(n === 2), +(n === 3), +(n === 4), +(n > 4), +(t.text[0] === '0'), +(v === 0),
    +(v >= 1 && v <= 3), +(v >= 1 && v <= 12), +(v >= 13 && v <= 31), +(v >= 32 && v <= 99), +(v >= 1900 && v <= 2100)]
}

/** A text token's character n-grams (n = 1..3, with word boundaries ^ $) and the whole token. */
export function ngramStrings(text: string): string[] {
  const cs = ['^', ...text, '$']
  const out: string[] = []
  for (let n = 1; n <= 3; n++)
    for (let i = 0; i + n <= cs.length; i++) {
      if (n === 1 && (i === 0 || i === cs.length - 1)) continue
      out.push(cs.slice(i, i + n).join(''))
    }
  out.push(`^${text}$ whole`)
  return out
}
/** The embedding rows those n-grams hash to. */
export function ngrams(text: string, buckets: number): number[] {
  return ngramStrings(text).map(g => {
    const s = g.length + g // n-grams of different lengths never share a key
    let h = 0x811c9dc5
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) }
    return (h >>> 0) % buckets
  })
}

const sets = new WeakMap<string[], Set<string>>() // by the list itself: teach replaces it when an alias adds a word
function timeWords(cfg: Config): Set<string> {
  const list = cfg.timeWords ?? []
  let s = sets.get(list)
  if (!s) sets.set(list, (s = new Set(list)))
  return s
}

/** `event` adds the heads that score time programs: the model then reads a whole event sentence. */
export function init(cfg: Config, seed = 1, event = false): Params {
  const rnd = mulberry32(seed)
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd())
  const P: Params = {}
  const mat = (name: string, r: number, c: number, std: number) => {
    const t = new T(r, c)
    for (let i = 0; i < t.d.length; i++) t.d[i] = gauss() * std
    t.name = name
    return (P[name] = t)
  }
  const { d } = cfg
  mat('ngram', cfg.buckets, d, 0.3)
  mat('num', NUM_FEATURES, d, 0.3)
  mat('type', 2, d, 0.1)
  mat('pos', MAX_TOKENS + 1, d, 0.1)
  mat('conv', cfg.k, d, 0.3)
  initTransformer(mat, cfg)
  mat('choice', d, H.SIZE, 1 / Math.sqrt(d))
  mat('choice.b', 1, H.SIZE, 0)
  mat('role', d, H.ROLES, 1 / Math.sqrt(d))
  if (event) {
    mat('time', d, TH.SIZE, 1 / Math.sqrt(d))
    mat('time.b', 1, TH.SIZE, 0)
    mat('time.role', d, TH.ROLES, 1 / Math.sqrt(d))
    // Words to named durations, read straight from the word embeddings. Zero as shipped: only corrections train it.
    mat('time.words', d, DURS.length, 0)
  }
  return P
}

/**
 * The model's flat output: H.SIZE choice scores, then H.ROLES role scores per row (row 0 unused);
 * an event model appends the same pair for times, TH.SIZE and TH.ROLES wide.
 */
export function forward(P: Params, cfg: Config, u: T, toks: Tok[]): T {
  const n = toks.length
  const text = toks.map((t, i) => ({ t, i })).filter(x => x.t.kind === 'text')
  const nums = toks.map((t, i) => ({ t, i })).filter(x => x.t.kind === 'num')
  // Token embeddings, assembled in token order behind the convention token.
  const rows: T[] = new Array(n + 1)
  rows[0] = matmul(u, P.conv)
  let words: T | undefined
  if (text.length) {
    const e = embedBag(P.ngram, text.map(x => ngrams(x.t.text, cfg.buckets)))
    // The words that do not say when, and not punctuation (the ":" of "9:30"): what is left is mostly the title.
    const rest = text.map((x, j) => ({ x, j })).filter(({ x }) => !timeWords(cfg).has(x.t.text) && /\p{L}/u.test(x.t.text))
    if (P['time.words'] && rest.length) words = vcat(rest.map(({ j }) => rowOf(e, j)))
    text.forEach((x, j) => { rows[x.i + 1] = add(rowOf(e, j), rowOf(P.type, 0)) })
  }
  if (nums.length) {
    const f = constant(nums.length, NUM_FEATURES, nums.flatMap(x => numFeatures(x.t)), 'number features')
    const e = matmul(f, P.num)
    nums.forEach((x, j) => { rows[x.i + 1] = add(rowOf(e, j), rowOf(P.type, 1)) })
  }
  let x = add(vcat(rows), firstRows(P.pos, n + 1))
  x = transformer(P, cfg, x)
  const choices = add(matmul(rowOf(x, 0), P.choice), P['choice.b'])
  choices.name = 'choice scores'
  const roles = matmul(x, P.role)
  roles.name = 'role scores'
  if (!P.time) return hcat([choices, flatten(roles)])
  let times = add(matmul(rowOf(x, 0), P.time), P['time.b'])
  if (words) {
    // The mean word embedding, through time.words, added to the scores of the named durations.
    const lengths = matmul(matmul(constant(1, words.r, new Float32Array(words.r).fill(1 / words.r), ''), words), P['time.words'])
    times = add(times, hcat([constant(1, TH.DUR, new Float32Array(TH.DUR), ''), lengths, constant(1, TH.SIZE - TH.DUR - DURS.length, new Float32Array(TH.SIZE - TH.DUR - DURS.length), '')]))
  }
  times.name = 'time choice scores'
  const timeRoles = matmul(x, P['time.role'])
  timeRoles.name = 'time role scores'
  return hcat([choices, flatten(roles), times, flatten(timeRoles)])
}

// Small shape ops built from the tape primitives' conventions.
function rowOf(a: T, i: number): T { return sliceRows(a, i, 1) }
function firstRows(a: T, n: number): T { return sliceRows(a, 0, n) }
export function sliceRows(a: T, from: number, n: number): T {
  const o = constant(n, a.c, a.d.subarray(from * a.c, (from + n) * a.c), '')
  o.op = 'rows'
  o.inputs = [a]
  o.back = () => { for (let i = 0; i < o.g.length; i++) a.g[from * a.c + i] += o.g[i] }
  return o
}
function flatten(a: T): T {
  const o = constant(1, a.r * a.c, a.d, '')
  o.op = 'flatten'
  o.inputs = [a]
  o.back = () => { for (let i = 0; i < o.g.length; i++) a.g[i] += o.g[i] }
  return o
}

/** target: a day number or NO_DATE. An event also has a start (minute of the day, or NO_TIME for all day) and, unless left open, an end. */
export type Example = { text: string; ref: Day; target: number; start?: number; end?: number }

export type Scored = { toks: Tok[]; programs: Programs; scores: T; start?: Times; startScores?: T; end?: Times; endScores?: T }

export function score(P: Params, cfg: Config, u: T, text: string, ref: Day): Scored {
  const toks = lex(text)
  const programs = enumerate(toks, ref)
  const out = forward(P, cfg, u, toks)
  const scores = pickSum(out, programs.terms, K)
  scores.name = 'program scores'
  if (!P.time) return { toks, programs, scores }
  // An event with no date written is today.
  programs.dates = programs.dates.map(d => (d === NO_DATE ? ref : d))
  const { start, end } = enumerateTimes(toks)
  return { toks, programs, scores, start, startScores: pickSum(out, start.terms, K), end, endScores: pickSum(out, end.terms, K) }
}

/** The programs that compute each part of the example; a part no program computes is absent. */
function marks(s: Scored, ex: Example): [T, Uint8Array][] {
  const part = (scores: T, n: number, hit: (i: number) => boolean): [T, Uint8Array][] => {
    const m = Uint8Array.from({ length: n }, (_, i) => +hit(i))
    return m.includes(1) ? [[scores, m]] : []
  }
  const out = part(s.scores, s.programs.n, i => s.programs.dates[i] === ex.target)
  if (ex.start === undefined || !s.start) return out
  out.push(...part(s.startScores!, s.start.n, i => s.start!.values[i] === ex.start))
  if (ex.start !== NO_TIME && ex.end !== undefined) out.push(...part(s.endScores!, s.end!.n, i => endOf(s.end!.values[i], ex.start!) === ex.end && ex.end! > ex.start!))
  return out
}

/** -log P(target), summed over the example's parts. Null when no program of this input computes any of them. */
export function loss(P: Params, cfg: Config, u: T, ex: Example, weight = 1): T | null {
  const parts = marks(score(P, cfg, u, ex.text, ex.ref), ex).map(([scores, m]) => marginalNLL(scores, m, weight))
  return parts.length ? parts.reduce((a, b) => add(a, b)) : null
}

/** Whether the day, the start and the end read as the example says. A part no program computes, or the example does not have, counts as read. */
export function held(s: Scored, ex: Example): boolean[] {
  const top = (values: ArrayLike<number>, scores: T, hit: (v: number) => boolean) => {
    const rs = rank(values, scores.d)
    return !rs.some(r => hit(r.value)) || hit(rs[0].value)
  }
  const day = top(s.programs.dates, s.scores, v => v === ex.target)
  if (ex.start === undefined || !s.start) return [day, true, true]
  return [day, top(s.start.values, s.startScores!, v => v === ex.start),
    ex.start === NO_TIME || ex.end === undefined || top(s.end!.values.map(v => endOf(v, ex.start!)), s.endScores!, v => v === ex.end && v > ex.start!)]
}
export const holds = (s: Scored, ex: Example) => held(s, ex).every(Boolean)

export type Ranked = { value: number; p: number; programs: { index: number; p: number }[] }

/** Values by total probability, each with the programs that compute it, most likely first. */
export function rank(values: ArrayLike<number>, scores: ArrayLike<number>): Ranked[] {
  const p = softmax1(scores)
  const by = new Map<number, Ranked>()
  for (let i = 0; i < p.length; i++) {
    let r = by.get(values[i])
    if (!r) by.set(values[i], (r = { value: values[i], p: 0, programs: [] }))
    r.p += p[i]
    r.programs.push({ index: i, p: p[i] })
  }
  const out = [...by.values()].sort((a, b) => b.p - a.p)
  for (const r of out) r.programs.sort((a, b) => b.p - a.p)
  return out
}

export type Reading = { date: number; p: number; programs: { index: number; p: number }[] }

/** Dates by total probability. */
export const readings = (s: Scored): Reading[] => rank(s.programs.dates, s.scores.d).map(r => ({ date: r.value, p: r.p, programs: r.programs }))
