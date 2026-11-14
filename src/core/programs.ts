// The formal half: a lexer, the space of date programs an input admits, and their exact
// execution. Nothing here knows what a word means or which number plays which role;
// that is what the weights score.

import { type Day, addMonths, fromDay, toDay, valid, weekStart, weekday } from './calendar.ts'

export type Tok = { kind: 'num' | 'text'; text: string; value: number; digits: number; at: number } // at: offset in the normalised string

export const MAX_TOKENS = 14

export function lex(s: string, max = MAX_TOKENS): Tok[] {
  const out: Tok[] = []
  for (const m of s.normalize('NFKC').toLowerCase().matchAll(/\d+|[\p{L}\p{M}]+|\S/gu)) {
    const text = m[0]
    const num = /^\d/.test(text)
    out.push({ kind: num ? 'num' : 'text', text, value: num ? Math.min(+text, 1e6) : 0, digits: num ? text.length : 0, at: m.index })
    if (out.length === max) break
  }
  return out
}

// Layout of the head vector the model produces: one score per choice a program can make.
export const H = {
  FORM: 0,    // abs, rel, dow, none
  MONTH: 4,   // month named by the words: 1..12
  YRULE: 16,  // year not written: same year as ref, next occurrence, last occurrence, nearest
  UNIT: 20,   // day, week, month, year
  RELN: 24,   // signed count named by the words: -3..3
  DIR: 31,    // past, future (for a count pointed at in the text)
  DOW: 33,    // Monday..Sunday
  MODE: 40,   // before ref, after ref, in last week, in this week, in next week
  SIZE: 45,
  ROLES: 4,   // per token: is the month, is the day, is the year, is the count
} as const

export const FORM = { ABS: 0, REL: 1, DOW: 2, NONE: 3 } as const
export const YRULES = ['same year', 'next occurrence', 'last occurrence', 'nearest'] as const
export const UNITS = ['day', 'week', 'month', 'year'] as const
export const MODES = ['before today', 'after today', 'last week', 'this week', 'next week'] as const
export const DOWS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] as const

/** Index of token t's role score in the model's flat output (row 0 is the convention token). */
export const ptr = (t: number, role: number) => H.SIZE + (t + 1) * H.ROLES + role

export const NO_DATE = -(2 ** 31)
export const K = 4 // terms per program

export type Programs = {
  n: number
  terms: Int32Array // n × K indices into the model's flat output, -1 for none
  dates: Int32Array // n day numbers, NO_DATE for "not a date"
  meta: Int32Array  // n × 4: form and its choices, for describe()
}

/** A two-digit year is the year with those digits in the hundred years ending twenty after today's. */
export function twoDigitYear(v: number, ref: Day): number {
  const latest = fromDay(ref).y + 20, y = Math.floor(latest / 100) * 100 + v
  return y > latest ? y - 100 : y
}

export function yearByRule(rule: number, m: number, d: number, ref: Day): number | null {
  const ry = fromDay(ref).y
  if (rule === 0) return valid(ry, m, d) ? ry : null
  const find = (from: number, step: number, ok: (z: Day) => boolean) => {
    for (let y = from, i = 0; i < 9; y += step, i++) if (valid(y, m, d) && ok(toDay(y, m, d))) return y
    return null
  }
  const next = find(ry, 1, z => z >= ref), last = find(ry, -1, z => z <= ref)
  if (rule === 1) return next
  if (rule === 2) return last
  if (next === null || last === null) return next ?? last
  return toDay(next, m, d) - ref <= ref - toDay(last, m, d) ? next : last
}

export function execRel(unit: number, n: number, ref: Day): Day {
  return unit === 0 ? ref + n : unit === 1 ? ref + 7 * n : addMonths(ref, unit === 2 ? n : 12 * n)
}

export function execDow(w: number, mode: number, ref: Day): Day {
  if (mode === 0) return ref - ((weekday(ref) - w + 7) % 7 || 7)
  if (mode === 1) return ref + ((w - weekday(ref) + 7) % 7 || 7)
  return weekStart(ref) + 7 * (mode - 3) + w
}

/** Every program the tokens admit, with the date each one computes for this reference date. */
export function enumerate(toks: Tok[], ref: Day): Programs {
  const terms: number[] = [], dates: number[] = [], meta: number[] = []
  const push = (date: number, t: number[], m: number[]) => {
    for (let i = 0; i < K; i++) terms.push(t[i] ?? -1)
    for (let i = 0; i < 4; i++) meta.push(m[i] ?? -1)
    dates.push(date)
  }
  const nums = toks.map((t, i) => ({ ...t, i })).filter(t => t.kind === 'num')

  // abs: date(year, month, day); each field points at a number or is supplied by a head.
  type Src = { term: number; v: number; tok: number; code: number }
  const months: Src[] = [], days: Src[] = [], years: Src[] = []
  for (const t of nums) {
    if (t.value >= 1 && t.value <= 12) months.push({ term: ptr(t.i, 0), v: t.value, tok: t.i, code: t.i })
    if (t.value >= 1 && t.value <= 31) days.push({ term: ptr(t.i, 1), v: t.value, tok: t.i, code: t.i })
    if (t.digits === 4 && t.value >= 1900 && t.value <= 2100) years.push({ term: ptr(t.i, 2), v: t.value, tok: t.i, code: t.i })
    else if (t.digits === 2) years.push({ term: ptr(t.i, 2), v: twoDigitYear(t.value, ref), tok: t.i, code: t.i })
  }
  for (let k = 1; k <= 12; k++) months.push({ term: H.MONTH + k - 1, v: k, tok: -1, code: 100 + k })
  for (const d of days)
    for (const m of months) {
      if (m.tok === d.tok) continue
      for (const y of years) {
        if (y.tok === d.tok || y.tok === m.tok || !valid(y.v, m.v, d.v)) continue
        push(toDay(y.v, m.v, d.v), [H.FORM + FORM.ABS, m.term, d.term, y.term], [FORM.ABS, m.code, d.code, y.code])
      }
      for (let r = 0; r < 4; r++) {
        const y = yearByRule(r, m.v, d.v, ref)
        if (y !== null) push(toDay(y, m.v, d.v), [H.FORM + FORM.ABS, m.term, d.term, H.YRULE + r], [FORM.ABS, m.code, d.code, 100 + r])
      }
    }

  // rel: ref + n units; n is named by the words (-3..3) or is a number in the text with a direction.
  for (let u = 0; u < 4; u++) {
    for (let n = -3; n <= 3; n++)
      if (n !== 0 || u === 0) push(execRel(u, n, ref), [H.FORM + FORM.REL, H.UNIT + u, H.RELN + n + 3], [FORM.REL, u, n, -1])
    for (const t of nums)
      if (t.value >= 1 && t.value <= 366)
        for (let dir = 0; dir < 2; dir++)
          push(execRel(u, dir ? t.value : -t.value, ref), [H.FORM + FORM.REL, H.UNIT + u, H.DIR + dir, ptr(t.i, 3)], [FORM.REL, u, dir ? t.value : -t.value, t.i])
  }

  // dow: a weekday, placed relative to ref.
  for (let w = 0; w < 7; w++)
    for (let mode = 0; mode < 5; mode++)
      push(execDow(w, mode, ref), [H.FORM + FORM.DOW, H.DOW + w, H.MODE + mode], [FORM.DOW, w, mode, -1])

  push(NO_DATE, [H.FORM + FORM.NONE], [FORM.NONE])
  return { n: dates.length, terms: Int32Array.from(terms), dates: Int32Array.from(dates), meta: Int32Array.from(meta) }
}

/** The program as the expression it is, for the inspector. */
export function describe(ps: Programs, p: number, toks: Tok[]): string {
  const [form, a, b, c] = ps.meta.subarray(p * 4, p * 4 + 4)
  const src = (code: number, named: (k: number) => string) => (code >= 100 ? named(code - 100) : `"${toks[code].text}"`)
  if (form === FORM.ABS)
    return `date(year: ${src(c, k => YRULES[k])}, month: ${src(a, k => `named ${k}`)}, day: "${toks[b].text}")`
  if (form === FORM.REL) return `today ${b < 0 ? '−' : '+'} ${Math.abs(b)} ${UNITS[a]}${Math.abs(b) === 1 ? '' : 's'}${c >= 0 ? ` (count "${toks[c].text}")` : ''}`
  if (form === FORM.DOW) return `${DOWS[a]} ${MODES[b]}`
  return 'not a date'
}

/** What a term of a program chooses, as [the question, the answer], for the inspector. */
export function choiceLabel(t: number, toks: Tok[]): [string, string] {
  if (t >= H.SIZE) {
    const row = Math.floor((t - H.SIZE) / H.ROLES) - 1
    return [['month', 'day', 'year', 'count'][(t - H.SIZE) % H.ROLES], `"${toks[row].text}"`]
  }
  if (t >= H.MODE) return ['which one', MODES[t - H.MODE]]
  if (t >= H.DOW) return ['weekday', DOWS[t - H.DOW]]
  if (t >= H.DIR) return ['direction', ['past', 'future'][t - H.DIR]]
  if (t >= H.RELN) return ['count', String(t - H.RELN - 3)]
  if (t >= H.UNIT) return ['unit', UNITS[t - H.UNIT]]
  if (t >= H.YRULE) return ['year', YRULES[t - H.YRULE]]
  if (t >= H.MONTH) return ['month', `named ${t - H.MONTH + 1}`]
  return ['form', ['a written date', 'an offset from today', 'a weekday', 'not a date'][t]]
}
