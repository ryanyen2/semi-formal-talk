// The formal half for times of day: the start and end programs a sentence admits, and the
// minute each one computes. Like the date programs, nothing here knows what a word means or
// which number is the hour; the weights score those choices.

import { H, K, type Tok } from './programs.ts'

// Layout of the time scores, which follow the date scores in the model's flat output.
export const TH = {
  FORM: 0,   // start: at a written hour, at an hour named by the words, no time (all day)
  MER: 3,    // a written hour is: am, pm, as written
  HOUR: 6,   // hour named by the words: 0..23
  MIN: 30,   // minute not written: :00, :15, :30, :45
  END: 34,   // end: a duration named by the words, until a written time, a counted duration
  DUR: 37,   // duration named by the words: 15, 30, … 240 minutes
  EMER: 53,  // the end's written hour is: am, pm, as written
  EMIN: 56,  // the end's minute, not written
  UNIT: 60,  // a counted duration is in: minutes, hours
  SIZE: 62,
  ROLES: 7,  // per token: hour, minute, hhmm, end hour, end minute, end hhmm, duration count
} as const

export const MINS = [0, 15, 30, 45] as const
export const DURS = Array.from({ length: 16 }, (_, i) => 15 * (i + 1))
export const MERS = ['am', 'pm', 'as written'] as const
export const NO_TIME = -1
/** An end program's value is a minute of the day, or DURATION + a length in minutes. */
export const DURATION = 10000
export const endOf = (value: number, start: number) => (value >= DURATION ? start + value - DURATION : value)

export type Times = {
  n: number
  terms: Int32Array  // n × K indices into the model's flat output, -1 for none
  values: Int32Array // start: minute of the day or NO_TIME; end: see DURATION
  meta: Int32Array   // n × 4, for describe
}

const hourOf = (v: number, mer: number) => (mer === 2 ? (v <= 23 ? v : -1) : v >= 1 && v <= 12 ? (v % 12) + 12 * mer : -1)

/** Every start and end program the tokens admit. */
export function enumerateTimes(toks: Tok[]): { start: Times; end: Times } {
  const base = H.SIZE + (toks.length + 1) * H.ROLES
  const role = (t: number, r: number) => base + TH.SIZE + (t + 1) * TH.ROLES + r
  const nums = toks.map((t, i) => ({ ...t, i })).filter(t => t.kind === 'num')
  const list = () => {
    const terms: number[] = [], values: number[] = [], meta: number[] = []
    return {
      push(value: number, t: number[], m: number[]) {
        for (let i = 0; i < K; i++) terms.push(t[i] ?? -1)
        for (let i = 0; i < 4; i++) meta.push(m[i] ?? -1)
        values.push(value)
      },
      done: (): Times => ({ n: values.length, terms: Int32Array.from(terms), values: Int32Array.from(values), meta: Int32Array.from(meta) }),
    }
  }
  // A time read off the digits: an hour token with its minute (a following number, or supplied
  // by a head), or one token holding both (1630). r0 is the first of the three roles involved.
  const written = (out: ReturnType<typeof list>, form: number, formTerm: number, r0: number, MER: number, MIN: number) => {
    for (const t of nums)
      for (let mer = 0; mer < 3; mer++) {
        if (t.digits <= 2) {
          const h = hourOf(t.value, mer)
          if (h < 0) continue
          for (let k = 0; k < 4; k++) out.push(h * 60 + MINS[k], [formTerm, role(t.i, r0), base + MER + mer, base + MIN + k], [form, t.i, 100 + k, mer])
          for (const m of nums)
            if (m.i > t.i && m.i - t.i <= 2 && m.digits <= 2 && m.value < 60)
              out.push(h * 60 + m.value, [formTerm, role(t.i, r0), base + MER + mer, role(m.i, r0 + 1)], [form, t.i, m.i, mer])
        } else if (t.digits <= 4 && t.value % 100 < 60) {
          const h = hourOf(Math.floor(t.value / 100), mer)
          if (h >= 0) out.push(h * 60 + (t.value % 100), [formTerm, role(t.i, r0 + 2), base + MER + mer], [form, t.i, -1, mer])
        }
      }
  }

  const start = list()
  written(start, 0, base + TH.FORM, 0, TH.MER, TH.MIN)
  for (let h = 0; h < 24; h++)
    for (let k = 0; k < 4; k++) start.push(h * 60 + MINS[k], [base + TH.FORM + 1, base + TH.HOUR + h, base + TH.MIN + k], [1, h, 100 + k])
  start.push(NO_TIME, [base + TH.FORM + 2], [2])

  const end = list()
  DURS.forEach((d, i) => end.push(DURATION + d, [base + TH.END, base + TH.DUR + i], [3, d]))
  written(end, 4, base + TH.END + 1, 3, TH.EMER, TH.EMIN)
  for (const t of nums)
    for (let unit = 0; unit < 2; unit++) {
      const d = t.value * (unit ? 60 : 1)
      if (d >= 1 && d <= 1440) end.push(DURATION + d, [base + TH.END + 2, role(t.i, 6), base + TH.UNIT + unit], [5, t.i, unit])
    }
  return { start: start.done(), end: end.done() }
}

export const clock = (m: number) => `${Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')}`

/** The program as the expression it is. */
export function describeTime(ts: Times, p: number, toks: Tok[]): string {
  const [form, a, b, c] = ts.meta.subarray(p * 4, p * 4 + 4)
  const tok = (i: number) => `"${toks[i].text}"`
  const written = () => (b < 0 ? `hhmm: ${tok(a)}, ${MERS[c]}` : `hour: ${tok(a)}, minute: ${b >= 100 ? ':' + String(MINS[b - 100]).padStart(2, '0') : tok(b)}, ${MERS[c]}`)
  if (form === 0) return `at(${written()})`
  if (form === 1) return `at ${clock(a * 60 + MINS[b - 100])}, named`
  if (form === 2) return 'all day'
  if (form === 3) return `for ${a} min, named`
  if (form === 4) return `until(${written()})`
  return `for ${tok(a)} ${b ? 'hours' : 'minutes'}`
}
