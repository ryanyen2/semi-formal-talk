// Exact proleptic Gregorian calendar on day numbers (days since 1970-01-01).
// No Date objects and no time zones: a date is an integer.

export type Day = number
export type YMD = { y: number; m: number; d: number }

// Howard Hinnant's days_from_civil / civil_from_days.
export function toDay(y: number, m: number, d: number): Day {
  y -= m <= 2 ? 1 : 0
  const era = Math.floor(y / 400)
  const yoe = y - era * 400
  const doy = Math.floor((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy
  return era * 146097 + doe - 719468
}

export function fromDay(z: Day): YMD {
  z += 719468
  const era = Math.floor(z / 146097)
  const doe = z - era * 146097
  const yoe = Math.floor((doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365)
  const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100))
  const mp = Math.floor((5 * doy + 2) / 153)
  const d = doy - Math.floor((153 * mp + 2) / 5) + 1
  const m = mp + (mp < 10 ? 3 : -9)
  return { y: yoe + era * 400 + (m <= 2 ? 1 : 0), m, d }
}

export const isLeap = (y: number) => (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0
export const daysInMonth = (y: number, m: number) => (m === 2 ? (isLeap(y) ? 29 : 28) : [4, 6, 9, 11].includes(m) ? 30 : 31)
export const valid = (y: number, m: number, d: number) => m >= 1 && m <= 12 && d >= 1 && d <= daysInMonth(y, m)

/** 0 = Monday … 6 = Sunday (ISO). 1970-01-01 was a Thursday. */
export const weekday = (z: Day) => (((z + 3) % 7) + 7) % 7

/** Monday of the ISO week containing z. */
export const weekStart = (z: Day) => z - weekday(z)

export function addMonths(z: Day, n: number): Day {
  const { y, m, d } = fromDay(z)
  const t = y * 12 + (m - 1) + n
  const ny = Math.floor(t / 12), nm = (t % 12 + 12) % 12 + 1
  return toDay(ny, nm, Math.min(d, daysInMonth(ny, nm)))
}

const pad = (n: number, w = 2) => String(n).padStart(w, '0')
export function iso(z: Day): string {
  const { y, m, d } = fromDay(z)
  return `${pad(y, 4)}-${pad(m)}-${pad(d)}`
}
export function parseISO(s: string): Day | null {
  const m = /^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$/.exec(s)
  if (!m || !valid(+m[1], +m[2], +m[3])) return null
  return toDay(+m[1], +m[2], +m[3])
}
