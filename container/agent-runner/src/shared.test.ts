import { describe, it, expect } from 'vitest';

import { validateContainerInput } from './shared.js';

describe('validateContainerInput', () => {
  const validBase = {
    prompt: 'hello',
    groupFolder: 'main',
    chatJid: 'group@g.us',
    isMain: true,
  };

  it('accepts valid claude input', () => {
    const input = validateContainerInput({
      ...validBase,
      backend: { type: 'claude' },
    });
    expect(input.backend).toEqual({ type: 'claude' });
    expect(input.prompt).toBe('hello');
  });

  it('accepts valid pi input with provider/model/thinkingLevel', () => {
    const input = validateContainerInput({
      ...validBase,
      backend: { type: 'pi', provider: 'google', model: 'gemini-2.0-flash', thinkingLevel: 'high' },
    });
    expect(input.backend).toEqual({
      type: 'pi',
      provider: 'google',
      model: 'gemini-2.0-flash',
      thinkingLevel: 'high',
    });
  });

  it('defaults missing backend to claude', () => {
    const input = validateContainerInput(validBase);
    expect(input.backend).toEqual({ type: 'claude' });
  });

  it('rejects invalid backend type', () => {
    expect(() =>
      validateContainerInput({ ...validBase, backend: { type: 'openai' } }),
    ).toThrow();
  });

  it('rejects invalid thinkingLevel value', () => {
    expect(() =>
      validateContainerInput({
        ...validBase,
        backend: { type: 'pi', thinkingLevel: 'banana' },
      }),
    ).toThrow();
  });

  it('rejects non-object input', () => {
    expect(() => validateContainerInput('not an object')).toThrow();
    expect(() => validateContainerInput(42)).toThrow();
    expect(() => validateContainerInput(null)).toThrow();
  });

  it('accepts pi backend with no optional fields', () => {
    const input = validateContainerInput({
      ...validBase,
      backend: { type: 'pi' },
    });
    expect(input.backend).toEqual({ type: 'pi' });
  });

  it('preserves optional fields when present', () => {
    const input = validateContainerInput({
      ...validBase,
      sessionId: 'sess-123',
      isScheduledTask: true,
    });
    expect(input.sessionId).toBe('sess-123');
    expect(input.isScheduledTask).toBe(true);
  });
});
