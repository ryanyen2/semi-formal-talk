// Reverse-mode autodiff over 2-D float tensors. One example at a time, no batching:
// sequences here are a dozen tokens, so a batch is a loop that accumulates gradients.
// Every op records its inputs, so the tape is also the graph the inspector draws.

export class T {
  d: Float32Array
  g: Float32Array
  r: number
  c: number
  op = 'param'
  name = ''
  inputs: T[] = []
  back: (() => void) | null = null
  constructor(r: number, c: number, d?: Float32Array) {
    this.r = r
    this.c = c
    this.d = d ?? new Float32Array(r * c)
    this.g = new Float32Array(r * c)
  }
}

let tape: T[] = []
export const resetTape = () => { tape = [] }
export const getTape = () => tape

function node(op: string, r: number, c: number, inputs: T[]): T {
  const t = new T(r, c)
  t.op = op
  t.inputs = inputs
  tape.push(t)
  return t
}

/** Back-propagates from `from`, seeded with 1 (a scalar loss) or with the given vector. */
export function backward(from: T, seed?: ArrayLike<number>) {
  if (seed) from.g.set(seed)
  else from.g[0] = 1
  for (let i = tape.length - 1; i >= 0; i--) tape[i].back?.()
}

export function matmul(a: T, b: T): T {
  if (a.c !== b.r) throw new Error(`matmul ${a.r}x${a.c} · ${b.r}x${b.c}`)
  const n = a.r, k = a.c, m = b.c
  const o = node('matmul', n, m, [a, b])
  const ad = a.d, bd = b.d, od = o.d
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const av = ad[i * k + p]
      if (av === 0) continue
      const bo = p * m, oo = i * m
      for (let j = 0; j < m; j++) od[oo + j] += av * bd[bo + j]
    }
  o.back = () => {
    const og = o.g, ag = a.g, bg = b.g
    for (let i = 0; i < n; i++)
      for (let p = 0; p < k; p++) {
        const bo = p * m, oo = i * m
        const av = ad[i * k + p]
        let s = 0
        for (let j = 0; j < m; j++) {
          const gv = og[oo + j]
          s += gv * bd[bo + j]
          bg[bo + j] += av * gv
        }
        ag[i * k + p] += s
      }
  }
  return o
}

/** a + b, where b has a's shape or is one row broadcast over a's rows. */
export function add(a: T, b: T): T {
  const row = b.r === 1 && a.r !== 1
  if (a.c !== b.c || (!row && a.r !== b.r)) throw new Error('add shape')
  const o = node('add', a.r, a.c, [a, b])
  const c = a.c
  for (let i = 0; i < o.d.length; i++) o.d[i] = a.d[i] + b.d[row ? i % c : i]
  o.back = () => {
    for (let i = 0; i < o.g.length; i++) {
      a.g[i] += o.g[i]
      b.g[row ? i % c : i] += o.g[i]
    }
  }
  return o
}

export function relu(a: T): T {
  const o = node('relu', a.r, a.c, [a])
  for (let i = 0; i < a.d.length; i++) o.d[i] = a.d[i] > 0 ? a.d[i] : 0
  o.back = () => { for (let i = 0; i < a.d.length; i++) if (a.d[i] > 0) a.g[i] += o.g[i] }
  return o
}

export function layernorm(a: T, gain: T, bias: T): T {
  const n = a.r, c = a.c, eps = 1e-5
  const o = node('layernorm', n, c, [a, gain, bias])
  const xh = new Float32Array(n * c), inv = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let mu = 0
    for (let j = 0; j < c; j++) mu += a.d[i * c + j]
    mu /= c
    let v = 0
    for (let j = 0; j < c; j++) { const e = a.d[i * c + j] - mu; v += e * e }
    inv[i] = 1 / Math.sqrt(v / c + eps)
    for (let j = 0; j < c; j++) {
      xh[i * c + j] = (a.d[i * c + j] - mu) * inv[i]
      o.d[i * c + j] = xh[i * c + j] * gain.d[j] + bias.d[j]
    }
  }
  o.back = () => {
    for (let i = 0; i < n; i++) {
      let m1 = 0, m2 = 0
      for (let j = 0; j < c; j++) {
        const gy = o.g[i * c + j], x = xh[i * c + j]
        gain.g[j] += gy * x
        bias.g[j] += gy
        const gx = gy * gain.d[j]
        m1 += gx
        m2 += gx * x
      }
      m1 /= c; m2 /= c
      for (let j = 0; j < c; j++)
        a.g[i * c + j] += inv[i] * (o.g[i * c + j] * gain.d[j] - m1 - xh[i * c + j] * m2)
    }
  }
  return o
}

export function softmaxRows(a: T): T {
  const n = a.r, c = a.c
  const o = node('softmax', n, c, [a])
  for (let i = 0; i < n; i++) {
    let mx = -Infinity
    for (let j = 0; j < c; j++) mx = Math.max(mx, a.d[i * c + j])
    let s = 0
    for (let j = 0; j < c; j++) s += (o.d[i * c + j] = Math.exp(a.d[i * c + j] - mx))
    for (let j = 0; j < c; j++) o.d[i * c + j] /= s
  }
  o.back = () => {
    for (let i = 0; i < n; i++) {
      let dot = 0
      for (let j = 0; j < c; j++) dot += o.g[i * c + j] * o.d[i * c + j]
      for (let j = 0; j < c; j++) a.g[i * c + j] += o.d[i * c + j] * (o.g[i * c + j] - dot)
    }
  }
  return o
}

export function transpose(a: T): T {
  const o = node('transpose', a.c, a.r, [a])
  for (let i = 0; i < a.r; i++) for (let j = 0; j < a.c; j++) o.d[j * a.r + i] = a.d[i * a.c + j]
  o.back = () => {
    for (let i = 0; i < a.r; i++) for (let j = 0; j < a.c; j++) a.g[i * a.c + j] += o.g[j * a.r + i]
  }
  return o
}

export function scale(a: T, s: number): T {
  const o = node('scale', a.r, a.c, [a])
  for (let i = 0; i < a.d.length; i++) o.d[i] = a.d[i] * s
  o.back = () => { for (let i = 0; i < a.d.length; i++) a.g[i] += o.g[i] * s }
  return o
}

/** Side-by-side concatenation of tensors with the same number of rows. */
export function hcat(ts: T[]): T {
  const n = ts[0].r, c = ts.reduce((s, t) => s + t.c, 0)
  const o = node('hcat', n, c, ts)
  let off = 0
  for (const t of ts) {
    for (let i = 0; i < n; i++) for (let j = 0; j < t.c; j++) o.d[i * c + off + j] = t.d[i * t.c + j]
    off += t.c
  }
  o.back = () => {
    let off = 0
    for (const t of ts) {
      for (let i = 0; i < n; i++) for (let j = 0; j < t.c; j++) t.g[i * t.c + j] += o.g[i * c + off + j]
      off += t.c
    }
  }
  return o
}

/** Stacked concatenation of tensors with the same number of columns. */
export function vcat(ts: T[]): T {
  const c = ts[0].c, n = ts.reduce((s, t) => s + t.r, 0)
  const o = node('vcat', n, c, ts)
  let off = 0
  for (const t of ts) { o.d.set(t.d, off); off += t.d.length }
  o.back = () => {
    let off = 0
    for (const t of ts) {
      for (let i = 0; i < t.d.length; i++) t.g[i] += o.g[off + i]
      off += t.d.length
    }
  }
  return o
}

/** One output row per bag: the mean of the table rows the bag lists. */
export function embedBag(table: T, bags: number[][]): T {
  const c = table.c
  const o = node('embed', bags.length, c, [table])
  bags.forEach((bag, i) => {
    for (const k of bag) for (let j = 0; j < c; j++) o.d[i * c + j] += table.d[k * c + j] / bag.length
  })
  o.back = () => {
    bags.forEach((bag, i) => {
      for (const k of bag) for (let j = 0; j < c; j++) table.g[k * c + j] += o.g[i * c + j] / bag.length
    })
  }
  return o
}

/** A constant (no gradient) tensor placed on the tape so the graph shows it. */
export function constant(r: number, c: number, d: ArrayLike<number>, name = 'input'): T {
  const o = node('const', r, c, [])
  o.d.set(d)
  o.name = name
  return o
}

/**
 * Program scores. Program p's score is the sum of the entries terms[p*k .. p*k+k) of `src`
 * (flat indices, -1 for none): the fixed wiring from head outputs to programs.
 */
export function pickSum(src: T, terms: Int32Array, k: number): T {
  const P = terms.length / k
  const o = node('programs', P, 1, [src])
  for (let p = 0; p < P; p++)
    for (let j = 0; j < k; j++) { const t = terms[p * k + j]; if (t >= 0) o.d[p] += src.d[t] }
  o.back = () => {
    for (let p = 0; p < P; p++)
      for (let j = 0; j < k; j++) { const t = terms[p * k + j]; if (t >= 0) src.g[t] += o.g[p] }
  }
  return o
}

export function softmax1(x: ArrayLike<number>): Float64Array {
  let mx = -Infinity
  for (let i = 0; i < x.length; i++) mx = Math.max(mx, x[i])
  const p = new Float64Array(x.length)
  let s = 0
  for (let i = 0; i < x.length; i++) s += (p[i] = Math.exp(x[i] - mx))
  for (let i = 0; i < x.length; i++) p[i] /= s
  return p
}

/**
 * -log of the probability mass on the marked programs, under a softmax over all scores.
 * Gradient w.r.t. a score: p(z) - p(z | marked). Infinite when nothing is marked.
 */
export function marginalNLL(scores: T, marked: Uint8Array, weight = 1): T {
  const o = node('loss', 1, 1, [scores])
  const p = softmax1(scores.d)
  let mass = 0
  for (let i = 0; i < p.length; i++) if (marked[i]) mass += p[i]
  o.d[0] = -Math.log(mass) * weight
  o.back = () => {
    for (let i = 0; i < p.length; i++)
      scores.g[i] += o.g[0] * weight * (p[i] - (marked[i] ? p[i] / mass : 0))
  }
  return o
}

/** Σ (a_i − target_i)², for regressing onto a fixed vector. */
export function squaredError(a: T, target: ArrayLike<number>): T {
  const o = node('squared error', 1, 1, [a])
  for (let i = 0; i < a.d.length; i++) o.d[0] += (a.d[i] - target[i]) ** 2
  o.back = () => { for (let i = 0; i < a.d.length; i++) a.g[i] += o.g[0] * 2 * (a.d[i] - target[i]) }
  return o
}

/** Adam over a fixed list of parameters. */
export class Adam {
  m: Float32Array[]
  v: Float32Array[]
  t = 0
  params: T[]
  lr: number
  constructor(params: T[], lr: number) {
    this.params = params
    this.lr = lr
    this.m = params.map(p => new Float32Array(p.d.length))
    this.v = params.map(p => new Float32Array(p.d.length))
  }
  step(scaleGrad = 1) {
    this.t++
    const b1 = 0.9, b2 = 0.999
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t)
    this.params.forEach((p, k) => {
      const m = this.m[k], v = this.v[k], g = p.g, d = p.d
      for (let i = 0; i < d.length; i++) {
        const gi = g[i] * scaleGrad
        m[i] = b1 * m[i] + (1 - b1) * gi
        v[i] = b2 * v[i] + (1 - b2) * gi * gi
        d[i] -= this.lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8)
        g[i] = 0
      }
    })
  }
}
