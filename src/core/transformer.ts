// The transformer block shared by the date model and the instruction compiler.
// Parameters are looked up by name under a prefix, so two networks can live in one record.
import type { Params } from './model.ts'
import { T, add, hcat, layernorm, matmul, relu, scale, softmaxRows, transpose } from './tensor.ts'

export type Shape = { d: number; heads: number; layers: number; ff: number }

/** Adds the block's parameters to P. `mat(name, rows, cols, std)` creates one tensor. */
export function initTransformer(mat: (name: string, r: number, c: number, std: number) => T, s: Shape, prefix = '') {
  const { d, heads, ff } = s, dh = d / heads
  for (let l = 0; l < s.layers; l++) {
    const L = `${prefix}L${l}`
    mat(`${L}.ln1.g`, 1, d, 0).d.fill(1)
    mat(`${L}.ln1.b`, 1, d, 0)
    for (let h = 0; h < heads; h++) {
      mat(`${L}.q${h}`, d, dh, 1 / Math.sqrt(d))
      mat(`${L}.k${h}`, d, dh, 1 / Math.sqrt(d))
      mat(`${L}.v${h}`, d, dh, 1 / Math.sqrt(d))
    }
    mat(`${L}.o`, d, d, 0.5 / Math.sqrt(d))
    mat(`${L}.ln2.g`, 1, d, 0).d.fill(1)
    mat(`${L}.ln2.b`, 1, d, 0)
    mat(`${L}.ff1`, d, ff, 1 / Math.sqrt(d))
    mat(`${L}.ff1.b`, 1, ff, 0)
    mat(`${L}.ff2`, ff, d, 0.5 / Math.sqrt(ff))
  }
  mat(`${prefix}out.g`, 1, d, 0).d.fill(1)
  mat(`${prefix}out.b`, 1, d, 0)
}

/** Pre-norm layers over the rows of x, then a final layer norm. */
export function transformer(P: Params, s: Shape, x: T, prefix = ''): T {
  const dh = s.d / s.heads
  for (let l = 0; l < s.layers; l++) {
    const L = `${prefix}L${l}`
    const a = layernorm(x, P[`${L}.ln1.g`], P[`${L}.ln1.b`])
    const outs: T[] = []
    for (let h = 0; h < s.heads; h++) {
      const q = matmul(a, P[`${L}.q${h}`]), k = matmul(a, P[`${L}.k${h}`]), v = matmul(a, P[`${L}.v${h}`])
      const att = softmaxRows(scale(matmul(q, transpose(k)), 1 / Math.sqrt(dh)))
      att.name = `attention ${L} head ${h}`
      outs.push(matmul(att, v))
    }
    x = add(x, matmul(hcat(outs), P[`${L}.o`]))
    const b = layernorm(x, P[`${L}.ln2.g`], P[`${L}.ln2.b`])
    x = add(x, matmul(relu(add(matmul(b, P[`${L}.ff1`]), P[`${L}.ff1.b`])), P[`${L}.ff2`]))
  }
  return layernorm(x, P[`${prefix}out.g`], P[`${prefix}out.b`])
}
