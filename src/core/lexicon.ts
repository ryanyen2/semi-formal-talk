// An unknown word can mean what a known word means. Before any gradient step, a correction
// that does not hold is tried against that hypothesis: for each word of the sentence the seed
// corpus never wrote, is there a corpus word that, put in its place, makes the sentence read
// as corrected? If so the unknown word's embedding is set equal to that word's, which makes
// the two interchangeable in every sentence, not only this one.
//
// A token's embedding is the mean of its n-gram rows, and rows are shared between n-grams, so
// the change Δ is spread over the token's rows by the least total cost under the prior:
// minimise Σ_r precision_r · δ_r² subject to mean_r δ_r = Δ. Rows the seed corpus does not
// rely on take almost all of it.

import { type Config, type Example, type Params, held, ngramStrings, ngrams, rank, score } from './model.ts'
import { lex } from './programs.ts'
import { NO_TIME } from './time.ts'
import { T, resetTape } from './tensor.ts'

export type Alias = { word: string; means: string }

const TRIED = 40 // corpus words tried for spelling, besides those that alone mean the corrected day or time

const overlap = (a: string, b: string) => {
  const x = new Set(ngramStrings(a)), y = new Set(ngramStrings(b))
  let both = 0
  for (const g of x) if (y.has(g)) both++
  return both / (x.size + y.size - both)
}

/**
 * Finds an alias for each correction that does not hold, in the order given, and writes the
 * embedding changes into `theta` (the n-gram table's centre) and P, and stiffens `precision`
 * (the prior's, for that table) on the rows it set. Returns the aliases found.
 * A substitution is accepted when it makes more parts of the correction (day, start, end)
 * hold and none fewer; among those, the word spelled most like the unknown one wins.
 */
export function learnAliases(P: Params, cfg: Config, u: T, theta: Float32Array, precision: Float64Array, words: string[], corrections: Example[]): Alias[] {
  const known = new Set(words), found: Alias[] = []
  const parts = (text: string, ex: Example) => { resetTape(); return held(score(P, cfg, u, text, ex.ref), ex) }
  const count = (h: boolean[]) => h.filter(Boolean).length
  for (const ex of corrections) {
    const before = parts(ex.text, ex)
    if (before.every(Boolean)) continue
    const norm = ex.text.normalize('NFKC').toLowerCase()
    let best: { word: string; means: string; gain: number; like: number } | undefined
    // Trying every corpus word in the sentence would take seconds. Tried are the words that on
    // their own read as the corrected day or start, and the ones spelled most like the unknown word.
    const alone = words.filter(w => {
      resetTape()
      const s = score(P, cfg, u, w, ex.ref)
      return (ex.target !== ex.ref && rank(s.programs.dates, s.scores.d)[0].value === ex.target) ||
        (ex.start !== undefined && ex.start !== NO_TIME && !!s.start && rank(s.start.values, s.startScores!.d)[0].value === ex.start)
    })
    for (const t of lex(ex.text)) {
      if (t.kind !== 'text' || known.has(t.text) || found.some(a => a.word === t.text)) continue
      const alike = [...words].sort((a, b) => overlap(t.text, b) - overlap(t.text, a)).slice(0, TRIED)
      for (const w of new Set([...alone, ...alike])) {
        const after = parts(norm.slice(0, t.at) + w + norm.slice(t.at + t.text.length), ex)
        if (after.some((h, i) => before[i] && !h) || count(after) <= count(before)) continue
        const c = { word: t.text, means: w, gain: count(after), like: overlap(t.text, w) }
        if (!best || c.gain > best.gain || (c.gain === best.gain && c.like > best.like)) best = c
      }
    }
    if (!best) continue
    found.push({ word: best.word, means: best.means })
    const d = cfg.d, table = P.ngram.d
    const mean = (rows: number[], j: number) => rows.reduce((s, r) => s + table[r * d + j], 0) / rows.length
    const from = ngrams(best.word, cfg.buckets), to = ngrams(best.means, cfg.buckets)
    const times = new Map<number, number>() // a row can hold several of the word's n-grams
    for (const r of from) times.set(r, (times.get(r) ?? 0) + 1)
    for (let j = 0; j < d; j++) {
      const delta = mean(to, j) - mean(from, j)
      let z = 0
      for (const [r, m] of times) z += (m * m) / precision[r * d + j]
      for (const [r, m] of times) { const i = r * d + j; table[i] += (from.length * delta * m) / precision[i] / z; theta[i] = table[i] }
      // The word is now read like a corpus word, so its rows are held like that word's: otherwise
      // the rest of the solve would treat them as free and move them again.
      const stiff = to.reduce((s, r) => s + precision[r * d + j], 0) / to.length
      for (const r of times.keys()) precision[r * d + j] = Math.max(precision[r * d + j], stiff)
    }
  }
  return found
}
