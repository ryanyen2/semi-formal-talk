// Teaching. The weights in force minimise one objective over the network weights θ and the
// convention vector u:
//
//   J(θ, u) =   Σ_i w_i · −log P(corrected date_i | text_i, today_i; θ, u)      corrections
//             − log Σ_τ π_τ exp(−|u − U_τ|² / 2s²)                              prior on u
//             + ½ Σ_k (strength · c_k + floor) (θ_k − θ⁰_k)²                    prior on θ
//
// The solve always starts from the prior, so the weights are a function of the set of
// corrections: order does not matter, and taking a correction away is solving without it.
//
// Prior on u: the convention vectors U_τ of the seed tables, each with a Gaussian of width s
// around it. Conventions the network was trained under are where its behaviour is defined; a
// point between two of them is not a convention. The instruction decides how likely each is:
// π_τ = exp(−λ · L_τ). The instruction means one seed table: the one that agrees with its reading
// on the most probes (`meantTable`; the probes are the inputs that tell the tables apart, and
// the instruction's reading of each is chosen by a language model among the dates the tables
// give, server/compile.ts). L_τ is how differently table τ behaves from that one: the share of
// seed rows the two read differently, counted as PSEUDO_ROWS rows of evidence (`tableLoss`). So
// among conventions that fit the corrections, the one that changes the least behaviour is
// preferred. The instruction is evidence about conventions only; it never trains the network.
//
// Prior on θ: a Gaussian centred on the pretrained weights θ⁰. c is the curvature of the
// network's output scores on the seed corpus (scripts/prior.ts), so the exponent is, to second
// order, strength/2 times the expected squared change of those scores: θ may move wherever the
// seed tables, under every convention, would not notice.

import { type Alias, learnAliases } from './lexicon.ts'
import { type Config, type Example, type Params, holds as reads, loss, score } from './model.ts'
import { T, backward, resetTape } from './tensor.ts'

export type Prior = {
  theta: Record<string, Float32Array>
  curvature: Record<string, Float32Array>
  tables: Float32Array      // U: one convention vector per seed table, k per row
  tableLoss: Float32Array   // L: how differently each table behaves from the one the instruction means
  words?: string[]          // the words the seed corpus writes, for lexicon.ts; absent: no aliases are tried
  timeWords?: string[]      // cfg.timeWords as shipped; teach keeps it here because it extends cfg's list with aliases
}
/**
 * grow: what an unmet correction's weight is multiplied by. warm: after that, the descent goes on from
 * where it is instead of starting over. lazy: the descent stops once every correction holds, so a
 * correction that a convention or an alias already explains moves no other weight.
 */
export type Hyper = { strength: number; floor: number; width: number; instructionWeight: number; steps: number; grow: number; warm: boolean; lazy: boolean }
export const DEFAULT_HYPER: Hyper = { strength: 1000, floor: 0.01, width: 0.1, instructionWeight: 1, steps: 80, grow: 2, warm: false, lazy: false }
/** For event sentences, where a solve runs while the user watches: fewer steps, no starting over, and no more change than the corrections need. */
export const EVENT_HYPER: Hyper = { ...DEFAULT_HYPER, steps: 30, grow: 4, warm: true, lazy: true }

/** One optimiser step: the three terms of J, the step size taken, and how many corrections read as corrected. */
export type Step = { step: number; corrections: number; conventions: number; network: number; size: number; held: number }

const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }

/**
 * The seed table the instruction means: the one whose answers agree with the instruction's
 * choices on the most probes or, when the reader gave a log-probability for every option
 * (`scores`), the one whose answers are most probable in total.
 */
export function meantTable(answers: number[][], choices: number[], scores?: number[][]): number {
  const cost = answers.map(a => a.reduce((n, x, i) => n + (scores ? -scores[i][x] : +(x !== choices[i])), 0))
  return cost.indexOf(Math.min(...cost))
}

/** PSEUDO_ROWS × the share of seed rows that table τ reads differently from the table the instruction means. */
export const PSEUDO_ROWS = 20
export function tableLoss(behaviour: number[][], meant: number): Float32Array {
  const m = behaviour[meant]
  return Float32Array.from(behaviour, b => (PSEUDO_ROWS * b.reduce((n, x, i) => n + +(x !== m[i]), 0)) / m.length)
}

/** −log P(target) of each example at the current weights; Infinity where no program computes it. */
export function losses(P: Params, cfg: Config, u: T, examples: Example[]): number[] {
  return examples.map(ex => { resetTape(); return loss(P, cfg, u, ex)?.d[0] ?? Infinity })
}

/**
 * Sets (P, u) to the weights the corrections imply and returns the descent's steps. For an event
 * model it also sets cfg.timeWords: the shipped list and the words aliased to one of its words.
 * Corrections no program can produce are the caller's to refuse beforehand; here they are skipped.
 *
 * The optimiser is L-BFGS whose initial inverse Hessian is the prior's covariance (1/precision
 * for θ, s² for u), with a backtracking line search, so J never increases. Its first step is
 * the minimiser of the linearised loss inside a ball of the prior: a stiff weight moves in
 * proportion to how little the seed corpus relied on it. A step is taken only if it meets the
 * Armijo condition.
 *
 * J is not convex in u (conventions are separate basins), so the descent starts at the seed
 * table that meets the most corrections, the network at θ⁰ (see `start`).
 *
 * A correction is a constraint, not a vote: when the descent has converged and a corrected
 * cell still reads otherwise, that correction's weight doubles and the solve starts over, up to
 * weight 64. Corrections that cannot hold together (the same text meaning two things on two days
 * is one way) stay unmet, and the last step's `held` says how many do hold.
 */
export function teach(P: Params, cfg: Config, u: T, prior: Prior, corrections: Example[], h: Hyper, onStep?: (s: Step) => void, onAlias?: (a: Alias) => void): Step[] {
  // Canonical order: the result depends on which corrections there are, not on when they came.
  corrections = [...corrections].sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : a.ref - b.ref || a.target - b.target || (a.start ?? 0) - (b.start ?? 0) || (a.end ?? 0) - (b.end ?? 0)))
  const names = Object.keys(prior.theta)
  const k = u.d.length, tables = prior.tables.length / k
  // The centre the network is held to: the pretrained weights, moved by the aliases the corrections imply (lexicon.ts).
  const theta = prior.words ? { ...prior.theta, ngram: Float32Array.from(prior.theta.ngram) } : prior.theta
  const nTheta = names.reduce((s, n) => s + prior.theta[n].length, 0), N = nTheta + k
  // One flat vector x = (θ − θ⁰, u).
  const precision = new Float64Array(nTheta)
  { let o = 0; for (const n of names) { const c = prior.curvature[n]; for (let i = 0; i < c.length; i++) precision[o++] = h.strength * c[i] + h.floor } }
  const put = (x: Float64Array) => {
    let o = 0
    for (const n of names) { const d = P[n].d, c = theta[n]; for (let i = 0; i < d.length; i++) d[i] = c[i] + x[o++] }
    for (let i = 0; i < k; i++) u.d[i] = x[nTheta + i]
  }
  const clear = () => { for (const n of names) P[n].g.fill(0); u.g.fill(0) }
  clear()

  const weight = corrections.map(() => 1)
  const holds = (ex: Example) => { resetTape(); return reads(score(P, cfg, u, ex.text, ex.ref), ex) }
  type Value = { corrections: number; conventions: number; network: number }
  const total = (v: Value) => v.corrections + v.conventions + v.network
  /** J at x; fills grad when given. */
  const J = (x: Float64Array, grad?: Float64Array): Value => {
    put(x)
    const v: Value = { corrections: 0, conventions: 0, network: 0 }
    for (const [i, ex] of corrections.entries()) {
      resetTape()
      const l = loss(P, cfg, u, ex, weight[i])
      if (!l) continue
      v.corrections += l.d[0]
      if (grad) backward(l)
    }
    for (let i = 0; i < nTheta; i++) if (x[i] !== 0) v.network += 0.5 * precision[i] * x[i] * x[i]
    // −log Σ_τ π_τ exp(−|u − U_τ|²/2s²) and its gradient Σ_τ r_τ (u − U_τ)/s², r the responsibilities.
    const e = new Float64Array(tables), s2 = h.width * h.width
    let lo = Infinity
    for (let t = 0; t < tables; t++) {
      e[t] = h.instructionWeight * prior.tableLoss[t]
      for (let i = 0; i < k; i++) e[t] += (x[nTheta + i] - prior.tables[t * k + i]) ** 2 / (2 * s2)
      lo = Math.min(lo, e[t])
    }
    let z = 0
    for (let t = 0; t < tables; t++) z += (e[t] = Math.exp(lo - e[t]))
    v.conventions = lo - Math.log(z)
    if (grad) {
      let o = 0
      for (const n of names) { const g = P[n].g; for (let i = 0; i < g.length; i++, o++) grad[o] = g[i] + precision[o] * x[o] }
      for (let i = 0; i < k; i++) {
        let s = 0
        for (let t = 0; t < tables; t++) s += (e[t] / z) * (x[nTheta + i] - prior.tables[t * k + i])
        grad[nTheta + i] = u.g[i] + s / s2
      }
      clear()
    }
    return v
  }
  /** The prior's covariance applied to a vector. */
  const precondition = (q: Float64Array) => {
    const r = new Float64Array(N)
    for (let i = 0; i < nTheta; i++) if (q[i] !== 0) r[i] = q[i] / precision[i]
    for (let i = nTheta; i < N; i++) r[i] = q[i] * h.width * h.width
    return r
  }

  // Starting basin: a seed table, the network at θ⁰. Corrections are constraints, so the table
  // under which the most (weighted) corrections already read as corrected wins; among those,
  // the one that fits the instruction's examples best (precomputed). How improbable an unmet
  // correction is under a table does not count: a word no table knows says nothing about
  // conventions.
  const start = () => {
    let x = new Float64Array(N), best = Infinity
    for (const n of names) P[n].d.set(theta[n])
    const most = prior.tableLoss.reduce((a, b) => Math.max(a, b), 1)
    for (let t = 0; t < tables; t++) {
      u.d.set(prior.tables.subarray(t * k, t * k + k))
      const unmet = corrections.reduce((s, ex, i) => s + (holds(ex) ? 0 : weight[i]), 0)
      const v = unmet + prior.tableLoss[t] / (2 * most) // the second term is below ½: it only breaks ties
      if (v < best) { best = v; x = new Float64Array(N); x.set(prior.tables.subarray(t * k, t * k + k), nTheta) }
    }
    return x
  }
  const shipped = cfg.timeWords && (prior.timeWords ??= cfg.timeWords)
  if (shipped) cfg.timeWords = shipped
  let x = start()
  if (prior.words) {
    put(x)
    let at = 0
    for (const n of names) { if (n === 'ngram') break; at += theta[n].length }
    const found = learnAliases(P, cfg, u, theta.ngram, precision.subarray(at, at + theta.ngram.length), prior.words, corrections)
    found.forEach(a => onAlias?.(a))
    // A word taken to mean a word that says when says when too.
    if (shipped) cfg.timeWords = [...shipped, ...found.filter(a => shipped.includes(a.means)).map(a => a.word)]
    if (found.length) x = start()
  }

  // The first step recorded is where the instruction alone would put the weights, so the
  // trace shows what the corrections cost there.
  const trace: Step[] = []
  if (corrections.length) {
    const origin = new Float64Array(N), t = prior.tableLoss.indexOf(prior.tableLoss.reduce((a, b) => Math.min(a, b)))
    origin.set(prior.tables.subarray(t * k, t * k + k), nTheta)
    const s: Step = { step: 0, ...J(origin), size: 0, held: corrections.filter(holds).length }
    trace.push(s)
    onStep?.(s)
  }
  let memory: { s: Float64Array; y: Float64Array; rho: number }[] = []
  let g = new Float64Array(N), size = 1
  let at = J(x, g)
  for (let left = h.steps; ; left--) { // each solve, and each restart after tightening, has h.steps steps
    const broken = corrections.map(ex => !holds(ex))
    const s: Step = { step: trace.length, ...at, size, held: broken.filter(b => !b).length }
    trace.push(s)
    onStep?.(s)
    if (left === 0 || (h.lazy && s.held === corrections.length)) break
    const value = total(at)

    // Two-loop recursion, with the prior covariance as the initial inverse Hessian.
    const q = Float64Array.from(g), alpha: number[] = []
    for (let i = memory.length - 1; i >= 0; i--) { const m = memory[i]; alpha[i] = m.rho * dot(m.s, q); for (let j = 0; j < N; j++) q[j] -= alpha[i] * m.y[j] }
    const r = precondition(q)
    for (let i = 0; i < memory.length; i++) { const m = memory[i], b = m.rho * dot(m.y, r); for (let j = 0; j < N; j++) r[j] += m.s[j] * (alpha[i] - b) }
    let slope = -dot(g, r)
    if (slope >= 0) { memory = []; r.set(precondition(g)); slope = -dot(g, r) }

    const tighten = () => {
      // Converged for these weights. Tighten the corrections that do not hold yet, or finish.
      if (!broken.some(Boolean) || Math.max(...weight) >= 64) return false
      // The objective changed, so the solve starts over from the prior with the new weights (or, warm, goes on from here).
      broken.forEach((b, i) => { if (b) weight[i] *= h.grow })
      memory = []
      if (!h.warm) x = start()
      at = J(x, g)
      left = h.steps + 1
      return true
    }
    if (-slope < 1e-4 * (1 + value)) { if (tighten()) continue; break }

    // Backtracking (Armijo) line search along −r.
    const next = new Float64Array(N), gNext = new Float64Array(N)
    for (size = 1; ; size /= 2) {
      for (let j = 0; j < N; j++) next[j] = x[j] - size * r[j]
      if (total(J(next)) <= value + 1e-4 * size * slope || size < 1e-9) break
    }
    // No step along this direction lowers J: the descent is done here. Nothing is accepted.
    if (size < 1e-9) { if (tighten()) continue; break }
    const trial = J(next, gNext)
    const sv = new Float64Array(N), yv = new Float64Array(N)
    for (let j = 0; j < N; j++) { sv[j] = next[j] - x[j]; yv[j] = gNext[j] - g[j] }
    const sy = dot(sv, yv)
    if (sy > 1e-12) { memory.push({ s: sv, y: yv, rho: 1 / sy }); if (memory.length > 8) memory.shift() }
    x = next; g = gNext; at = trial
  }
  put(x)
  clear()
  return trace
}
