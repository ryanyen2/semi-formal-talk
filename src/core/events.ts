// Seed corpus of event sentences: a title, a date phrase from the date corpus (seed.ts) and a
// time phrase, in any order. The date follows the table's conventions; the time conventions
// are fixed (see `bare` and the defaults below). The model sees only the sentence, today, and
// the day, start and end it means.

import { toDay } from './calendar.ts'
import type { Example } from './model.ts'
import { MAX_TOKENS, lex } from './programs.ts'
import { type Conventions, sample } from './seed.ts'
import { NO_TIME } from './time.ts'

export type EventSeed = Required<Example> & { kind: string; title: string }

export const TITLES = ['movie night', 'dinner', 'lunch', 'breakfast', 'brunch', 'coffee', 'standup', 'team sync', 'design review', 'dentist', 'doctor', 'gym', 'yoga',
  'run', 'flight', 'haircut', 'call', 'meeting', 'interview', 'lecture', 'seminar', 'office hours', 'study group', 'piano lesson', 'birthday party', 'date night',
  'groceries', 'laundry', 'book club', 'game night', 'hike', 'concert', 'reading group', 'lab meeting', 'paper deadline', 'pick up kids', 'therapy', 'drinks',
  'happy hour', 'workshop', 'demo', 'retro', 'planning', 'check in', 'catch up', 'walk the dog', 'swim', 'climbing', 'practice', 'rehearsal', 'talk', 'keynote',
  'réunion', 'déjeuner', 'cena', 'almuerzo', 'treffen', 'abendessen', '晚餐', '午餐', '開會', '看電影', '会議', '飲み会']
const NAMES = ['amy', 'ben', 'chen', 'dana', 'eli', 'fatima', 'george', 'hana', 'ivan', 'jo', 'kai', 'lena', 'mom', 'dad', 'sam', 'the team', 'prof lee', '小明']
/** With no time written, a meal is at its hour; anything else is all day. */
const MEALS: Record<string, number> = { breakfast: 8, brunch: 11, lunch: 12, déjeuner: 12, almuerzo: 12, 午餐: 12, dinner: 19, cena: 19, abendessen: 19, 晚餐: 19 }
/** A bare hour: 1 to 7 is the afternoon, 8 to 12 the morning, as written above that. */
export const bare = (h: number) => (h >= 1 && h <= 7 ? h + 12 : h)
export const DEFAULT_MINUTES = 60
/** The date conventions the app starts from: month first, a year not written is the next occurrence, a bare weekday or "next" one is the coming one. */
export const DEFAULT: Conventions = { order: 0, yrule: 1, weekend: 5, bare: 1, next: 1, last: 0 }

const pad = (n: number) => String(n).padStart(2, '0')

export function sampleEvent(conv: Conventions, rnd: () => number): EventSeed {
  const int = (a: number, b: number) => a + Math.floor(rnd() * (b - a + 1))
  const pick = <X>(xs: readonly X[]) => xs[Math.floor(rnd() * xs.length)]
  const chance = (p: number) => rnd() < p

  /** One time of day, written in a style that means it. */
  const written = (t: number, loc: string): string => {
    const h = Math.floor(t / 60), m = t % 60, h12 = h % 12 || 12, pm = h >= 12
    const mm = (sep: string) => (m ? sep + pad(m) : '')
    const styles: (() => string | null)[] = [
      () => `${h12}${mm(pick([':', ':', '.']))}${pick(['', ' '])}${pick(pm ? ['pm', 'pm', 'p', 'PM', 'p.m.'] : ['am', 'am', 'a', 'AM', 'a.m.'])}`,
      () => `${pick([h, pad(h)])}:${pad(m)}`,
      () => (bare(h12) === h ? `${h12}${mm(':')}` : null),
      () => (h >= 10 || m ? `${pad(h)}${pad(m)}` : null),
      () => (t === 720 ? pick(['noon', 'midday']) : t === 0 ? 'midnight' : null),
      () => (m === 30 ? `half past ${h12}${pm ? 'pm' : 'am'}` : null),
      () => (loc === 'fr' ? `${h}h${m ? pad(m) : ''}` : loc === 'de' ? `${h}${mm(':')} Uhr` : null),
      () => (loc === 'es' && h12 > 1 ? `a las ${h12}${mm(':')} de la ${h < 12 ? 'mañana' : h < 20 ? 'tarde' : 'noche'}` : null),
      () => (loc === 'zh' ? `${h < 6 ? '凌晨' : h < 12 ? pick(['上午', '早上']) : h < 18 ? '下午' : '晚上'}${h12}點${m === 30 ? '半' : m ? m + '分' : ''}` : null),
      () => (loc === 'ja' ? `${pm ? '午後' : '午前'}${h12}時${m === 30 ? '半' : m ? m + '分' : ''}` : null),
    ]
    for (;;) {
      const s = pick(['fr', 'de', 'es', 'zh', 'ja'].includes(loc) && chance(0.6) ? styles.slice(6) : styles.slice(0, 6))()
      if (s !== null) return s
    }
  }
  const length = (d: number, loc: string): string => {
    const hours = d % 60 === 0 && chance(0.8)
    const n = hours ? d / 60 : d
    if (d === 60 && chance(0.2)) return pick(['for an hour', 'one hour'])
    if (d === 30 && chance(0.2)) return 'for half an hour'
    const unit = pick(hours
      ? loc === 'fr' ? ['heures', 'h'] : loc === 'de' ? ['Stunden', 'Std'] : loc === 'es' ? ['horas', 'h'] : loc === 'zh' ? ['小時', '个小时'] : loc === 'ja' ? ['時間'] : ['hours', 'hrs', 'hr', 'h', 'hour']
      : loc === 'fr' || loc === 'es' ? ['minutes', 'min'] : loc === 'de' ? ['Minuten', 'min'] : loc === 'zh' ? ['分鐘', '分钟'] : loc === 'ja' ? ['分'] : ['minutes', 'min', 'mins', 'm'])
    return `${chance(0.5) && loc === 'en' ? 'for ' : ''}${n}${pick(['', ' '])}${unit}`
  }

  for (;;) {
    const loc = chance(0.6) ? 'en' : pick(['en', 'fr', 'es', 'de', 'zh', 'ja'])
    const letters = 'abcdefghijklmnopqrstuvwxyz'
    const word = () => Array.from({ length: int(2, 8) }, () => pick([...letters])).join('')
    let title = chance(0.1) ? '' : chance(0.3) ? Array.from({ length: int(1, 2) }, word).join(' ') : pick(TITLES)
    if (title && chance(0.25)) title += ` ${pick(['with', 'w/', 'and', '+'])} ${chance(0.3) ? word() : pick(NAMES)}`

    // The day.
    const date = chance(0.25) ? null : sample(conv, rnd, [loc === 'en' || chance(0.7) ? loc : 'en'])
    if (date?.kind === 'junk') continue
    const ref = date?.ref ?? toDay(int(2015, 2035), 1, 1) + int(0, 364)
    let day = date ? date.text : ''
    if (date && chance(0.15) && /^[\d\p{L}]/u.test(day) && loc === 'en') day = `on ${day}`

    // The time.
    let start: number = NO_TIME, end = 0, time = '', kind = date?.kind ?? 'today'
    const meal = title.split(' ').map(w => MEALS[w]).find(h => h !== undefined)
    if (chance(0.22)) {
      if (meal !== undefined) { start = meal * 60; end = start + DEFAULT_MINUTES; kind += '+meal' }
      else kind += '+allday'
    } else {
      start = (chance(0.85) ? int(7, 21) : int(0, 23)) * 60 + pick([0, 0, 0, 0, 0, 0, 30, 30, 30, 15, 45, int(0, 11) * 5])
      // 300 to 480 minutes are longer than any named duration: written in hours, only the program
      // that takes the number computes them, so the model learns to read "N hours" from N.
      const d = pick([15, 30, 30, 45, 60, 60, 60, 90, 90, 120, 120, 180, 240, 300, 360, 480, int(1, 36) * 5])
      end = start + d
      if (end > 1440) continue
      const how = rnd()
      if (how < 0.55) { end = start + DEFAULT_MINUTES; if (end > 1440) continue; time = written(start, loc); kind += '+time' }
      else if (how < 0.8) {
        const sep = pick(['-', '-', '–', ' - ', ' to ', ' until ', '~'])
        // "4-5pm": the first hour takes its half of the day from the second.
        const shared = Math.floor(start / 720) === Math.floor(end / 720) && end < 1440 && chance(0.4)
        if (end >= 1440) continue
        const a = shared ? `${(Math.floor(start / 60) % 12) || 12}${start % 60 ? ':' + pad(start % 60) : ''}` : written(start, 'en')
        const b = shared ? `${(Math.floor(end / 60) % 12) || 12}${end % 60 ? ':' + pad(end % 60) : ''}${end >= 720 ? 'pm' : 'am'}` : written(end, 'en')
        time = `${chance(0.2) ? 'from ' : ''}${a}${sep}${b}`
        kind += '+range'
      } else { time = `${written(start, loc)} ${length(d, loc)}`; kind += '+length' }
      if (chance(0.25) && loc === 'en' && !time.startsWith('from')) time = `${pick(['at', 'at', '@'])} ${time}`
    }

    const parts = pick([[title, day, time], [title, day, time], [title, day, time], [title, time, day], [day, time, title], [day, title, time], [time, title, day], [time, day, title]]).filter(Boolean)
    if (!parts.length) continue
    const text = parts.join(chance(0.1) ? ', ' : ' ')
    if (lex(text, MAX_TOKENS + 1).length > MAX_TOKENS) continue
    return { kind, title, text, ref, target: date ? date.target : ref, start, end }
  }
}
