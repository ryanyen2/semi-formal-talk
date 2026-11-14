// The model lives here: reading a sentence, and teaching. The weights in force are a function
// of the shipped files and the corrections; nothing else is kept.
import { iso, parseISO } from './core/calendar.ts'
import { type Manifest, unpack } from './core/checkpoint.ts'
import { DEFAULT, TITLES } from './core/events.ts'
import type { Alias } from './core/lexicon.ts'
import { type Config, type Example, type Params, type Ranked, type Scored, held, holds, ngramStrings, ngrams, rank, score } from './core/model.ts'
import { FORM, H, K, describe, lex } from './core/programs.ts'
import { ALL_CONVENTIONS } from './core/seed.ts'
import { EVENT_HYPER, type Prior, type Step, tableLoss, teach } from './core/teach.ts'
import { T, resetTape } from './core/tensor.ts'
import { DURS, NO_TIME, TH, clock, describeTime, endOf } from './core/time.ts'
import type { Api, Chunk, Correction, Event, Frame, Guess, Inspection, Line, Option, Placed, Probe, Solved, Summary } from './protocol.ts'

let cfg: Config, P: Params, prior: Prior, manifest: Manifest, u: T, u0: Float32Array
/** Each correction as the user gave it, and the part of it the weights are taught: what a solve could not make hold is left out of the next. */
let corrections: { given: Correction; taught?: Example }[] = []

const day = (s: string) => parseISO(s)!
const example = (c: Correction): Example => ({ text: c.text, ref: day(c.ref), target: day(c.date), start: c.start ?? NO_TIME, end: c.end ?? 0 })
const read = (text: string, ref: string): Scored => { resetTape(); return score(P, cfg, u, text, day(ref)) }
const same = (a: Correction, b: Correction) => a.text === b.text && a.ref === b.ref
/** The sentence a teach or forget is about, and the probes read beside it at every step: fixed when the solve starts, so every frame reads the same ones. */
let watching: { text: string; ref: string; probes: Probe[] } | undefined
const post = (e: Event) => postMessage(e)

function solve(): Solved {
  const t0 = performance.now(), aliases: Alias[] = [], trace: Step[] = []
  const frame = (step?: Step) => { if (watching) post({ event: 'frame', frame: { step, of: corrections.filter(c => c.taught).length, inspection: inspection(watching.text, watching.ref, watching.probes), lines: ledger() } satisfies Frame }) }
  frame()
  // A part that does not hold within the step budget is not something this model learns from one
  // example. It is dropped and the solve repeated, so the weights depend only on what is taught,
  // and later solves do not spend their steps on it again.
  for (let round = 0; round < 3; round++) {
    aliases.length = 0
    trace.push(...teach(P, cfg, u, prior, corrections.flatMap(c => c.taught ?? []), EVENT_HYPER, step => { post({ event: 'step', step }); frame(step) }, a => aliases.push(a)))
    let trimmed = false
    for (const c of corrections) {
      if (!c.taught) continue
      const [day, start, end] = held(read(c.given.text, c.given.ref), c.taught)
      if (day && start && end) continue
      c.taught = !day ? undefined : !start ? { ...c.taught, start: undefined, end: undefined } : { ...c.taught, end: undefined }
      trimmed = true
    }
    if (!trimmed) break
  }
  return { summary: summary(), trace, ms: Math.round(performance.now() - t0), aliases }
}

const MOVED = 0.01 // a weight this far from the shipped one counts as moved; the page uses the same
/** The moved weights as lines (protocol.ts `Line`). */
function ledger(): Line[] {
  // Which word pieces read each ngram row, from the sentences corrected and the one being taught.
  const pieces = new Map<number, string>()
  for (const text of [...corrections.map(c => c.given.text), watching?.text ?? ''])
    for (const t of lex(text)) {
      if (t.kind !== 'text') continue
      const rows = ngrams(t.text, cfg.buckets)
      ngramStrings(t.text).forEach((g, k) => {
        const piece = g.replace(/ whole$/, ''), was = pieces.get(rows[k])
        if (!was) pieces.set(rows[k], piece)
        else if (!was.split('/').includes(piece)) pieces.set(rows[k], `${was}/${piece}`)
      })
    }
  const lines = new Map<string, Line>()
  const add = (key: string, label: string, i: number, slot: number, d: number) => {
    let l = lines.get(key)
    if (!l) lines.set(key, (l = { key, label, cells: [], size: 0 }))
    l.cells.push({ i, slot: slot < 0 ? l.cells.length : slot, d })
    l.size = Math.max(l.size, Math.abs(d))
  }
  const length = (m: number) => (m % 60 ? `→ ${m} min` : `→ ${m / 60} h`)
  let at = 0
  for (const m of manifest) {
    const now = P[m.name].d, was = prior.theta[m.name]
    for (let k = 0; k < now.length; k++) {
      const d = now[k] - was[k]
      if (Math.abs(d) < MOVED) continue
      if (m.name === 'ngram') { const row = Math.floor(k / m.c); add(`ngram ${row}`, pieces.get(row) ?? `row ${row}`, at + k, k % m.c, d) }
      else if (m.name === 'time.words') add(`length ${k % m.c}`, length(DURS[k % m.c]), at + k, Math.floor(k / m.c), d)
      else add(m.name, m.name, at + k, -1, d)
    }
    at += now.length
  }
  for (let k = 0; k < u.d.length; k++) if (Math.abs(u.d[k] - u0[k]) >= MOVED) add('u', 'u', at + k, k, u.d[k] - u0[k])
  return [...lines.values()].sort((a, b) => b.size - a.size)
}

/** The seed convention nearest to u. */
function nearest(): number {
  let table = 0, near = Infinity
  for (let t = 0; t < ALL_CONVENTIONS.length; t++) {
    let d = 0
    for (let i = 0; i < cfg.k; i++) d += (u.d[i] - prior.tables[t * cfg.k + i]) ** 2
    if (d < near) { near = d; table = t }
  }
  return table
}

function summary(): Summary {
  let changed = 0
  for (const m of manifest) { const now = P[m.name].d, was = prior.theta[m.name]; for (let i = 0; i < now.length; i++) if (now[i] !== was[i]) changed++ }
  return {
    corrections: corrections.map(c => ({ ...c.given, held: holds(read(c.given.text, c.given.ref), example(c.given)) })),
    parameters: manifest.reduce((s, m) => s + m.r * m.c, 0), changed, table: nearest(),
  }
}

/** The three rankings of a reading, the end given the most probable start. */
function rankings(s: Scored) {
  const date = rank(s.programs.dates, s.scores.d), start = rank(s.start!.values, s.startScores!.d)
  const end = (at: number) => rank(s.end!.values.map(v => endOf(v, at)), s.endScores!.d).filter(r => r.value > at && r.value <= 1440)
  return { date, start, end }
}

type Rankings = ReturnType<typeof rankings>
/** The sentence read with each token left out, in token order. */
const without = (text: string, ref: string, s: Scored): Rankings[] => {
  const norm = text.normalize('NFKC')
  return s.toks.map(t => rankings(read(norm.slice(0, t.at) + norm.slice(t.at + t.text.length), ref)))
}

/** The token a term of a program reads (a role score of that token's row), or -1 for a choice made from the whole sentence. */
function pointee(term: number, n: number): number {
  const date = term - H.SIZE - H.ROLES
  if (date >= 0 && date < n * H.ROLES) return Math.floor(date / H.ROLES)
  const time = term - (H.SIZE + (n + 1) * H.ROLES) - TH.SIZE - TH.ROLES
  return time >= 0 && time < n * TH.ROLES ? Math.floor(time / TH.ROLES) : -1
}

/** The sentence's tokens, the five most probable values of each part with what each token gives them, and the probes placed. */
function inspection(text: string, ref: string, probes: Probe[]): Inspection {
  const s = read(text, ref), r = rankings(s), out = without(text, ref, s), role = roles(s, out), n = s.toks.length
  const at = r.start[0].value
  const options = (part: 'date' | 'start' | 'end', label: (v: number) => string): Option[] => {
    const of = (x: Rankings) => (part === 'date' ? x.date : part === 'start' ? x.start : at === NO_TIME ? [] : x.end(at))
    const ps = part === 'date' ? s.programs : part === 'start' ? s.start! : s.end!
    const how = part === 'date' ? (p: number) => describe(s.programs, p, s.toks) : (p: number) => describeTime(ps as typeof s.start & {}, p, s.toks)
    return of(r).slice(0, 5).map(x => {
      const read = s.toks.map(() => 0)
      for (const { index, p } of x.programs) {
        const seen = new Set<number>()
        for (let k = 0; k < K; k++) { const t = pointee(ps.terms[index * K + k], n); if (t >= 0 && !seen.has(t)) { seen.add(t); read[t] += p } }
      }
      const sway = s.toks.map((t, i) => (t.kind === 'text' ? Math.max(0, x.p - (of(out[i]).find(y => y.value === x.value)?.p ?? 0)) : 0))
      return { value: x.value, label: label(x.value), p: x.p, how: how(x.programs[0].index), read, sway }
    })
  }
  return {
    tokens: s.toks.map((t, i) => ({ text: t.text, role: role[i], rows: t.kind === 'text' ? ngrams(t.text, cfg.buckets) : [] })),
    date: options('date', iso),
    start: options('start', v => (v === NO_TIME ? 'all day' : clock(v))),
    end: options('end', clock),
    programs: s.programs.n + s.start!.n + s.end!.n,
    probes, placed: probes.map(p => place(p.text, p.ref)),
  }
}

/** A probe's most probable reading; a start with no end read is given an hour, as on the calendar. */
function place(text: string, ref: string): Placed {
  const r = rankings(read(text, ref)), d = r.date[0], a = r.start[0], e = a.value === NO_TIME ? undefined : r.end(a.value)[0]
  return { date: iso(d.value), start: a.value === NO_TIME ? null : a.value, end: a.value === NO_TIME ? null : e?.value ?? Math.min(a.value + 60, 1440), p: d.p * a.p * (e?.p ?? 1) }
}

const SWAPS = ['dinner', 'yoga', 'call mom'] // titles put in place of a sentence's first title word
/**
 * The sentence with one thing changed, then the six events added last. The changes: each small
 * number one less and one more (four at most), and the first word that is not the date or the
 * time replaced by another title. Whether a correction carries to these is what it taught about
 * numbers and titles.
 */
function probesOf(text: string, ref: string, others: Probe[]): Probe[] {
  const s = read(text, ref), role = roles(s, without(text, ref, s)), norm = text.normalize('NFKC'), out: Probe[] = []
  const swap = (from: number, to: number, by: string) => out.push({ text: norm.slice(0, from) + by + norm.slice(to), ref, mark: [from, from + by.length] })
  for (const t of s.toks)
    if (t.kind === 'num' && t.digits <= 2)
      for (const v of [t.value - 1, t.value + 1]) if (v >= 1 && out.length < 4) swap(t.at, t.at + t.text.length, String(v).padStart(t.text[0] === '0' ? t.digits : 1, '0'))
  const t = s.toks.find((t, i) => t.kind === 'text' && (role[i] === 'title' || role[i] === 'none'))
  if (t) swap(t.at, t.at + t.text.length, SWAPS.find(x => x !== t.text)!)
  return [...out, ...others.filter(o => o.text !== text).slice(-6)]
}

/** Words the corpus writes in titles; a time word among them ("hour" in "happy hour") can end a title. */
const TITLE_WORDS = new Set(TITLES.flatMap(t => t.split(' ')))
const SWAY = 0.2 // how much of a reading's probability a token must move to count as part of it

/**
 * What the reading uses each token for. A token the most probable program points at (the hour,
 * the day number) is part of the time or the date. Any other token is, when the sentence
 * without it gives the days (the starts or ends) a different distribution: the total variation
 * between the two is above SWAY. The title is the longest run of tokens that are not part of
 * the date and are not pointed at (a word that only sways the time, "dinner", can be in it), with
 * what says when taken off its ends: numbers, the corpus words that say when (not those its titles
 * use), and right after a number, a word that says when or changes the end ("hours" in "for 3 hours").
 */
function roles(s: Scored, without: Rankings[]): Chunk['role'][] {
  const r = rankings(s), at = r.start[0].value, timed = at !== NO_TIME
  const out: Chunk['role'][] = s.toks.map(() => 'none')
  const point = (meta: Int32Array, p: number, role: 'date' | 'time', ...fields: number[]) => { for (const f of fields) { const t = meta[p * 4 + f]; if (t >= 0 && t < 100) out[t] = role } }
  const d = r.date[0].programs[0].index, a = r.start[0].programs[0].index, e = timed ? r.end(at)[0]?.programs[0].index : undefined
  if (s.programs.meta[d * 4] === FORM.ABS) point(s.programs.meta, d, 'date', 1, 2, 3)
  if (s.programs.meta[d * 4] === FORM.REL) point(s.programs.meta, d, 'date', 3)
  if (s.start!.meta[a * 4] === 0) point(s.start!.meta, a, 'time', 1, 2)
  if (e !== undefined && s.end!.meta[e * 4] >= 4) point(s.end!.meta, e, 'time', 1, ...(s.end!.meta[e * 4] === 4 ? [2] : []))

  const apart = (x: Ranked[], y: Ranked[]) => { let tv = 0; for (const v of new Set([...x, ...y].map(q => q.value))) tv += Math.abs((x.find(q => q.value === v)?.p ?? 0) - (y.find(q => q.value === v)?.p ?? 0)); return tv / 2 }
  const sways: boolean[] = s.toks.map(() => false), lengthens = [...sways]
  s.toks.forEach((_, i) => {
    const r2 = without[i]
    lengthens[i] = timed && apart(r.end(at), r2.end(at)) > SWAY
    if (out[i] !== 'none') return
    if (apart(r.date, r2.date) > SWAY) out[i] = 'date'
    else if (apart(r.start, r2.start) > SWAY || lengthens[i]) { out[i] = 'time'; sways[i] = true }
  })
  const when = new Set(cfg.timeWords ?? [])
  const edge = (i: number) => {
    const t = s.toks[i]
    return t.kind === 'num' || (when.has(t.text) && !TITLE_WORDS.has(t.text)) || (s.toks[i - 1]?.kind === 'num' && (when.has(t.text) || lengthens[i]))
  }
  let best = [0, 0]
  for (let i = 0, from = 0; i <= out.length; i++)
    if (i === out.length || (out[i] !== 'none' && !sways[i])) {
      let a = from, b = i
      while (a < b && edge(a)) a++
      while (b > a && edge(b - 1)) b--
      if (b - a > best[1] - best[0]) best = [a, b]
      from = i + 1
    }
  for (let i = best[0]; i < best[1]; i++) out[i] = 'title'
  return out
}

/** Runs f with frames of c's sentence and the others posted at every step. */
function watch<R>(c: Correction, others: Probe[], f: () => R): R {
  watching = { text: c.text, ref: c.ref, probes: probesOf(c.text, c.ref, others) }
  try { return f() } finally { watching = undefined }
}

export const api: { [M in keyof Api]: (...a: Parameters<Api[M]>) => ReturnType<Api[M]> | Promise<ReturnType<Api[M]>> } = {
  async init() {
    // the files sit beside the page's root, which is not the domain's root on GitHub Pages
    const get = (file: string) => fetch(import.meta.env.BASE_URL + file)
    const floats = async (file: string) => new Float32Array(await (await get(file)).arrayBuffer())
    const meta: { cfg: Config; manifest: Manifest } = await (await get('event.json')).json()
    const { tables, behaviour, words } = await (await get('event.prior.json')).json()
    cfg = meta.cfg
    manifest = meta.manifest.filter(m => m.name !== 'U')
    const { U: _, ...params } = unpack(meta.manifest, await floats('event.bin'))
    P = params
    const curvature = unpack(manifest, await floats('event.curvature.bin'))
    const meant = ALL_CONVENTIONS.findIndex(c => (Object.keys(DEFAULT) as (keyof typeof DEFAULT)[]).every(k => c[k] === DEFAULT[k]))
    prior = {
      theta: Object.fromEntries(manifest.map(m => [m.name, Float32Array.from(P[m.name].d)])),
      curvature: Object.fromEntries(manifest.map(m => [m.name, curvature[m.name].d])),
      tables: Float32Array.from(tables), tableLoss: tableLoss(behaviour, meant), words,
    }
    u = new T(1, cfg.k)
    const s = solve().summary
    u0 = Float32Array.from(u.d)
    return s
  },

  parse(text, ref) {
    const s = read(text, ref), r = rankings(s)
    const guess = (d: Ranked, a: Ranked, s: Scored, r: ReturnType<typeof rankings>, p: number): Guess => {
      const e = a.value === NO_TIME ? undefined : r.end(a.value)[0]
      return {
        date: iso(d.value), start: a.value === NO_TIME ? null : a.value, end: e?.value ?? null, p,
        how: { date: describe(s.programs, d.programs[0].index, s.toks), start: describeTime(s.start!, a.programs[0].index, s.toks), end: e ? describeTime(s.end!, e.programs[0].index, s.toks) : null },
      }
    }
    const guesses: Guess[] = []
    for (const d of r.date.slice(0, 3)) for (const a of r.start.slice(0, 3)) guesses.push(guess(d, a, s, r, d.p * a.p))
    guesses.sort((x, y) => y.p - x.p)
    const shown = guesses.filter((g, i) => i === 0 || g.p >= 0.04).slice(0, 4)
    // What the sentence would read as under each convention one habit away from the one in force.
    // These are the readings a correction can switch to at once, so they are offered even when
    // the weights give them little probability.
    const here = ALL_CONVENTIONS[nearest()], mine = Float32Array.from(u.d)
    ALL_CONVENTIONS.forEach((c, t) => {
      if ((Object.keys(c) as (keyof typeof c)[]).filter(k => c[k] !== here[k]).length !== 1) return
      u.d.set(prior.tables.subarray(t * cfg.k, (t + 1) * cfg.k))
      const s2 = read(text, ref), r2 = rankings(s2), d = r2.date[0], a = r2.start[0]
      const p = (r.date.find(x => x.value === d.value)?.p ?? 0) * (r.start.find(x => x.value === a.value)?.p ?? 0)
      const g = guess(d, a, s2, r2, p)
      if (!shown.some(x => x.date === g.date && x.start === g.start)) shown.push(g)
    })
    u.d.set(mine)
    const role = roles(s, without(text, ref, s)), norm = text.normalize('NFKC')
    // Runs of tokens with one role, as the text between the first and the last of them.
    const chunks: Chunk[] = []
    for (let i = 0, at = 0; i < role.length; ) {
      let j = i
      while (j + 1 < role.length && role[j + 1] === role[i]) j++
      const to = s.toks[j].at + s.toks[j].text.length
      chunks.push({ text: norm.slice(at, to), role: role[i] })
      at = to
      i = j + 1
    }
    return { text, title: chunks.filter(c => c.role === 'title').map(c => c.text.trim()).join(' '), chunks, guesses: shown.slice(0, 6) }
  },

  teach(c, others = []) {
    const s = read(c.text, c.ref), ex = example(c)
    const learnable = {
      date: s.programs.dates.includes(ex.target),
      start: s.start!.values.includes(ex.start!),
      end: ex.start === NO_TIME || s.end!.values.some(v => endOf(v, ex.start!) === ex.end),
    }
    corrections = [...corrections.filter(x => !same(x.given, c)), { given: c, taught: ex }]
    return { ...watch(c, others, solve), learnable }
  },

  forget(c, others = []) {
    corrections = corrections.filter(x => !same(x.given, c)).map(x => ({ given: x.given, taught: example(x.given) }))
    return watch(c, others, solve)
  },

  weights() {
    const n = manifest.reduce((s, m) => s + m.r * m.c, 0)
    const now = new Float32Array(n), shipped = new Float32Array(n)
    let at = 0
    for (const m of manifest) { now.set(P[m.name].d, at); shipped.set(prior.theta[m.name], at); at += m.r * m.c }
    return { manifest, now, shipped, u: Float32Array.from(u.d), u0, lines: ledger() }
  },

  vectors(words) {
    const d = cfg.d
    const mean = (table: Float32Array, rows: number[]) => {
      const v = new Float32Array(d)
      for (const r of rows) for (let j = 0; j < d; j++) v[j] += table[r * d + j] / rows.length
      return v
    }
    const cos = (a: Float32Array, b: Float32Array) => {
      let ab = 0, aa = 0, bb = 0
      for (let j = 0; j < d; j++) { ab += a[j] * b[j]; aa += a[j] ** 2; bb += b[j] ** 2 }
      return ab / Math.sqrt(aa * bb || 1)
    }
    const known = (prior.words ?? []).map(w => [w, mean(P.ngram.d, ngrams(w, cfg.buckets))] as const)
    return words.map(word => {
      const rows = ngrams(word, cfg.buckets), now = mean(P.ngram.d, rows)
      let near = { word: '', sim: -1, now }
      for (const [k, v] of known) if (k !== word) { const sim = cos(now, v); if (sim > near.sim) near = { word: k, sim, now: v } }
      return { word, known: known.some(([k]) => k === word), pieces: ngramStrings(word), rows, now, shipped: mean(prior.theta.ngram, rows), near }
    })
  },

  inspect(text, ref, others = []) {
    return inspection(text, ref, probesOf(text, ref, others))
  },
}

onmessage = async (e: MessageEvent<{ id: number; method: keyof Api; args: unknown[] }>) => {
  const { id, method, args } = e.data
  try {
    postMessage({ id, result: await (api[method] as (...a: unknown[]) => unknown)(...args) })
  } catch (err) {
    postMessage({ id, error: String(err) })
  }
}
