// Seed corpus: synthetic tables, each written under its own conventions.
// Month names, weekday names, long date formats and relative phrases come from the
// platform's CLDR data (Intl); only "last/next <weekday>" and "weekend" phrases are listed
// by hand. The model never sees a convention, only (text, reference date, date) triples
// and which table a triple came from.

import { type Day, fromDay, toDay, weekStart } from './calendar.ts'
import type { Example } from './model.ts'
import { NO_DATE, execDow, execRel } from './programs.ts'

export type Conventions = {
  order: 0 | 1          // ambiguous numbers: month first, day first
  yrule: 0 | 1 | 2 | 3  // year not written: same year, next occurrence, last occurrence, nearest
  weekend: 5 | 6        // a weekend, as one date: its Saturday or its Sunday
  bare: 3 | 1 | 0       // "Friday": this week's, the next one, the last one
  next: 1 | 4           // "next Friday": the next one, next week's
  last: 0 | 2           // "last Friday": the last one, last week's
}

export const ALL_CONVENTIONS: Conventions[] = []
for (const order of [0, 1] as const)
  for (const yrule of [0, 1, 2, 3] as const)
    for (const weekend of [5, 6] as const)
      for (const bare of [3, 1, 0] as const)
        for (const next of [1, 4] as const)
          for (const last of [0, 2] as const) ALL_CONVENTIONS.push({ order, yrule, weekend, bare, next, last })

export const LOCALES = ['en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'zh', 'ja']

// [last, next, this] with {d} for the weekday name; [last, this, next] weekend.
const PHRASES: Record<string, { dow: [string, string, string]; weekend: [string, string, string] }> = {
  en: { dow: ['last {d}', 'next {d}', 'this {d}'], weekend: ['last weekend', 'this weekend', 'next weekend'] },
  fr: { dow: ['{d} dernier', '{d} prochain', 'ce {d}'], weekend: ['le week-end dernier', 'ce week-end', 'le week-end prochain'] },
  es: { dow: ['el {d} pasado', 'el próximo {d}', 'este {d}'], weekend: ['el fin de semana pasado', 'este fin de semana', 'el próximo fin de semana'] },
  de: { dow: ['letzten {d}', 'nächsten {d}', 'diesen {d}'], weekend: ['letztes Wochenende', 'dieses Wochenende', 'nächstes Wochenende'] },
  zh: { dow: ['上{d}', '下{d}', '这{d}'], weekend: ['上周末', '这周末', '下周末'] },
}

const JUNK = ['n/a', 'tbd', '-', '?', 'none', 'unknown', 'pending', 'x', '…', 'null', 'todo', 'see notes', 'asap', '#ref!', 'ok']

const fmtCache = new Map<string, Intl.DateTimeFormat>()
function fmt(locale: string, opts: Intl.DateTimeFormatOptions) {
  const key = locale + JSON.stringify(opts)
  let f = fmtCache.get(key)
  if (!f) fmtCache.set(key, (f = new Intl.DateTimeFormat(locale, { ...opts, timeZone: 'UTC' })))
  return f
}
const rtfCache = new Map<string, Intl.RelativeTimeFormat>()
function rtf(locale: string, numeric: 'auto' | 'always') {
  const key = locale + numeric
  let f = rtfCache.get(key)
  if (!f) rtfCache.set(key, (f = new Intl.RelativeTimeFormat(locale, { numeric })))
  return f
}
const jsDate = (z: Day) => new Date(z * 86400000)

// Month and weekday names of the corpus's languages, so a random junk word is never one of them.
let words: Set<string> | undefined
function dateWords(): Set<string> {
  if (words) return words
  words = new Set()
  for (const loc of LOCALES)
    for (const style of ['long', 'short'] as const) {
      for (let m = 0; m < 12; m++) for (const t of fmt(loc, { month: style }).format(jsDate(toDay(2026, m + 1, 1))).toLowerCase().match(/\p{L}+/gu) ?? []) words.add(t)
      for (let w = 0; w < 7; w++) for (const t of fmt(loc, { weekday: style }).format(jsDate(toDay(2026, 3, 16) + w)).toLowerCase().match(/\p{L}+/gu) ?? []) words.add(t)
    }
  return words
}

export function yearFor(rule: number, m: number, d: number, ref: Day): Day | null {
  const ry = fromDay(ref).y
  const ok = (y: number) => fromDay(toDay(y, m, d)).d === d // Feb 29 only in leap years
  if (rule === 0) return ok(ry) ? toDay(ry, m, d) : null
  let next: Day | null = null, last: Day | null = null
  for (let y = ry; y < ry + 9 && next === null; y++) if (ok(y) && toDay(y, m, d) >= ref) next = toDay(y, m, d)
  for (let y = ry; y > ry - 9 && last === null; y--) if (ok(y) && toDay(y, m, d) <= ref) last = toDay(y, m, d)
  if (rule === 1) return next
  if (rule === 2) return last
  if (next === null || last === null) return next ?? last
  return next - ref <= ref - last ? next : last
}

export type Kind = 'num2' | 'num3' | 'named+year' | 'named' | 'relative' | 'count' | 'weekday' | 'weekend' | 'junk'
export type Seed = Example & { kind: Kind }

/** One row of a table written under `conv`. `locales` lets a caller hold some out. */
export function sample(conv: Conventions, rnd: () => number, locales = LOCALES): Seed {
  const int = (a: number, b: number) => a + Math.floor(rnd() * (b - a + 1))
  const pick = <X>(xs: readonly X[]) => xs[Math.floor(rnd() * xs.length)]
  const ref = toDay(int(2015, 2035), 1, 1) + int(0, 364)
  const loc = pick(locales)
  const r = rnd()
  const padded = (n: number) => (rnd() < 0.5 ? String(n).padStart(2, '0') : String(n))
  const sep = pick(['/', '/', '-', '.', ' '])
  // The table's order decides ambiguous numbers. A day above 12 can be read one way only, so
  // now and then a row is written against the table's order: validity outranks convention.
  const dayFirst = (d: number) => (d > 12 && rnd() < 0.15 ? !conv.order : !!conv.order)

  for (;;) {
    if (r < 0.14) {
      const near = fromDay(ref + int(-200, 200))
      const target = yearFor(conv.yrule, near.m, near.d, ref)
      if (target === null) continue
      const [a, b] = dayFirst(near.d) ? [near.d, near.m] : [near.m, near.d]
      return { kind: 'num2', text: padded(a) + sep + padded(b), ref, target }
    }
    if (r < 0.28) {
      const two = rnd() < 0.3
      const target = two ? toDay(fromDay(ref).y + int(-60, 15), 1, 1) + int(0, 364) : ref + int(-900, 900)
      const { y, m, d } = fromDay(target)
      const yy = two ? String(y % 100).padStart(2, '0') : String(y)
      const iso = !two && rnd() < 0.3
      const parts = iso ? [yy, padded(m), padded(d)] : dayFirst(d) ? [padded(d), padded(m), yy] : [padded(m), padded(d), yy]
      return { kind: 'num3', text: parts.join(iso && sep === ' ' ? '-' : sep), ref, target }
    }
    if (r < 0.54) {
      const withYear = r < 0.40
      let target: Day
      if (withYear) target = ref + int(-900, 900)
      else {
        const near = fromDay(ref + int(-200, 200))
        const t = yearFor(conv.yrule, near.m, near.d, ref)
        if (t === null) continue
        target = t
      }
      const { y, d } = fromDay(target)
      const date = jsDate(target)
      const mon = fmt(loc, { month: pick(['long', 'long', 'short'] as const) }).format(date)
      const style = int(0, 5)
      let text: string
      if (style < 2) {
        const f = withYear ? fmt(loc, { dateStyle: pick(['long', 'medium', 'full'] as const) }) : fmt(loc, { month: pick(['long', 'short'] as const), day: 'numeric' })
        const numericMonth = f.formatToParts(date).some(p => p.type === 'month' && /^\d+$/.test(p.value))
        if (numericMonth && loc !== 'zh' && loc !== 'ja') continue // a numeric month here would bypass the table's order
        text = f.format(date)
      } else if (style === 2) text = withYear ? `${d} ${mon} ${y}` : `${d} ${mon}`
      else if (style === 3) text = withYear ? `${mon} ${d}, ${y}` : `${mon} ${d}`
      else if (style === 4) text = withYear ? `${y} ${mon} ${d}` : `${mon}${d}`
      else text = withYear ? `${mon} ${d} ${y}` : `${d}-${mon}`
      return { kind: withYear ? 'named+year' : 'named', text, ref, target }
    }
    if (r < 0.66) {
      const unit = pick(['day', 'day', 'day', 'week', 'month', 'year'] as const)
      const n = unit === 'day' ? int(-3, 3) : pick([-1, 1])
      const u = ['day', 'week', 'month', 'year'].indexOf(unit)
      return { kind: 'relative', text: rtf(loc, 'auto').format(n, unit), ref, target: execRel(u, n, ref) }
    }
    if (r < 0.76) {
      const u = int(0, 3)
      const n = int(1, [30, 12, 12, 5][u]) * pick([-1, 1])
      return { kind: 'count', text: rtf(loc, 'always').format(n, (['day', 'week', 'month', 'year'] as const)[u]), ref, target: execRel(u, n, ref) }
    }
    if (r < 0.90) {
      const w = int(0, 6)
      const name = fmt(loc, { weekday: pick(['long', 'long', 'short'] as const) }).format(jsDate(weekStart(ref) + w))
      const ph = PHRASES[loc]
      const which = ph ? int(0, 3) : 3
      if (which === 3) return { kind: 'weekday', text: name, ref, target: execDow(w, conv.bare, ref) }
      const mode = [conv.last, conv.next, 3][which]
      return { kind: 'weekday', text: ph.dow[which].replace('{d}', name), ref, target: execDow(w, mode, ref) }
    }
    if (r < 0.95) {
      const ph = PHRASES[loc] ?? PHRASES.en
      const k = int(0, 2)
      return { kind: 'weekend', text: ph.weekend[k], ref, target: execDow(conv.weekend, 2 + k, ref) }
    }
    const letters = 'abcdefghijklmnopqrstuvwxyz'
    const word = (): string => { const w = Array.from({ length: int(2, 8) }, () => pick([...letters])).join(''); return dateWords().has(w) ? word() : w }
    const j = int(0, 4)
    // Two numbers that are a date in neither order: both above 12, or a day its month does not have.
    const short = pick([2, 4, 6, 9, 11])
    const pair = rnd() < 0.5 ? [int(13, 31), int(13, 99)] : rnd() < 0.5 ? [short === 2 ? int(30, 31) : 31, short] : [short, short === 2 ? int(30, 31) : 31]
    const text = j === 0 ? pick(JUNK) : j === 1 ? word() : j === 2 ? String(int(10000, 9999999)) : j === 3 ? `${word()} ${int(0, 999)}` : pair.map(padded).join(sep)
    return { kind: 'junk', text, ref, target: NO_DATE }
  }
}
