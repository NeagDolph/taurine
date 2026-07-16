import { describe, expect, it } from 'vitest'
import { sourceColor, sourceLabel as graphSourceLabel } from './graph'
import { sourceLabel } from './labels'

describe('taurine source branding', () => {
  it('uses the first-party Taurine label and color everywhere', () => {
    expect(sourceColor('taurine')).toBe('#ffd75e')
    expect(graphSourceLabel('taurine')).toBe('Taurine')
    expect(sourceLabel('taurine')).toBe('Taurine')
  })
})
