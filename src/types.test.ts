import fs from 'fs';
import path from 'path';

import { describe, it, expect } from 'vitest';
import { validateAgentBackend } from './types.js';

describe('validateAgentBackend', () => {
  it('returns claude backend when agentType is undefined', () => {
    expect(validateAgentBackend()).toEqual({ type: 'claude' });
  });

  it('throws when agentType is unknown string', () => {
    expect(() => validateAgentBackend('openai')).toThrow('Unknown agent type');
  });

  it('returns bare pi backend when agentConfig is undefined', () => {
    expect(validateAgentBackend('pi')).toEqual({ type: 'pi' });
  });

  it('returns full pi backend with all config fields', () => {
    const result = validateAgentBackend('pi', {
      provider: 'google',
      model: 'gemini-2.5-flash',
      thinkingLevel: 'high',
    });
    expect(result).toEqual({
      type: 'pi',
      provider: 'google',
      model: 'gemini-2.5-flash',
      thinkingLevel: 'high',
    });
  });

  it('returns pi backend with partial config', () => {
    const result = validateAgentBackend('pi', { provider: 'openai' });
    expect(result).toEqual({ type: 'pi', provider: 'openai' });
  });

  it('throws for invalid thinkingLevel', () => {
    expect(() =>
      validateAgentBackend('pi', { thinkingLevel: 'banana' }),
    ).toThrow();
  });

  it('strips unknown fields from config', () => {
    const result = validateAgentBackend('pi', {
      provider: 'google',
      unknownField: 'should-be-stripped',
    });
    expect(result).not.toHaveProperty('unknownField');
    expect(result).toEqual({ type: 'pi', provider: 'google' });
  });
});

// ── Cross-validation: catch drift between host and container type definitions ─

describe('shared type sync', () => {
  const hostSource = fs.readFileSync(
    path.join(process.cwd(), 'src/types.ts'),
    'utf-8',
  );
  const containerSource = fs.readFileSync(
    path.join(process.cwd(), 'container/agent-runner/src/shared.ts'),
    'utf-8',
  );

  /** Extract string literals from `export type Foo = 'a' | 'b' | ...;` */
  function extractUnionLiterals(source: string, typeName: string): string[] {
    const pattern = new RegExp(`export\\s+type\\s+${typeName}\\s*=\\s*([^;]+);`, 's');
    const match = source.match(pattern);
    if (!match) throw new Error(`Type ${typeName} not found in source`);
    const literals = match[1].match(/'([^']+)'/g);
    if (!literals) throw new Error(`No string literals in ${typeName}`);
    return literals.map(l => l.slice(1, -1)).sort();
  }

  /** Extract a multi-line type definition, whitespace-normalized. */
  function extractMultilineType(source: string, typeName: string): string {
    const pattern = new RegExp(`export\\s+type\\s+${typeName}\\s*=\\s*([^;]+);`, 's');
    const match = source.match(pattern);
    if (!match) throw new Error(`Type ${typeName} not found in source`);
    return match[1].replace(/\s+/g, ' ').trim();
  }

  /** Extract values from `z.enum([...])` by schema variable name. */
  function extractZodEnumValues(source: string, schemaName: string): string[] {
    const pattern = new RegExp(`${schemaName}\\s*=\\s*z\\.enum\\(\\[([^\\]]+)\\]\\)`);
    const match = source.match(pattern);
    if (!match) throw new Error(`Zod schema ${schemaName} not found in source`);
    const literals = match[1].match(/'([^']+)'/g);
    if (!literals) throw new Error(`No enum values in ${schemaName}`);
    return literals.map(l => l.slice(1, -1)).sort();
  }

  /** Build a diff message showing which values are only in one side. */
  function diffMessage(typeName: string, host: string[], container: string[]): string {
    const hostOnly = host.filter(v => !container.includes(v));
    const containerOnly = container.filter(v => !host.includes(v));
    const parts: string[] = [`${typeName} is out of sync:`];
    if (hostOnly.length) parts.push(`  in src/types.ts only: ${hostOnly.join(', ')}`);
    if (containerOnly.length) parts.push(`  in container/agent-runner/src/shared.ts only: ${containerOnly.join(', ')}`);
    return parts.join('\n');
  }

  it('ThinkingLevel literals match', () => {
    const host = extractUnionLiterals(hostSource, 'ThinkingLevel');
    const container = extractUnionLiterals(containerSource, 'ThinkingLevel');
    expect(host, diffMessage('ThinkingLevel', host, container)).toEqual(container);
  });

  it('KnownProvider literals match', () => {
    const host = extractUnionLiterals(hostSource, 'KnownProvider');
    const container = extractUnionLiterals(containerSource, 'KnownProvider');
    expect(host, diffMessage('KnownProvider', host, container)).toEqual(container);
  });

  it('AgentBackend type definition matches', () => {
    const host = extractMultilineType(hostSource, 'AgentBackend');
    const container = extractMultilineType(containerSource, 'AgentBackend');
    expect(host).toBe(container);
  });

  it('thinkingLevelSchema Zod enum values match', () => {
    const host = extractZodEnumValues(hostSource, 'thinkingLevelSchema');
    const container = extractZodEnumValues(containerSource, 'thinkingLevelSchema');
    expect(host, diffMessage('thinkingLevelSchema', host, container)).toEqual(container);
  });

  // ── ContainerInput drift detection ──────────────────────────────────────

  const containerRunnerSource = fs.readFileSync(
    path.join(process.cwd(), 'src/container-runner.ts'),
    'utf-8',
  );

  /** Extract an interface body, whitespace-normalized. */
  function extractInterface(source: string, name: string): string {
    const pattern = new RegExp(
      `export\\s+interface\\s+${name}\\s*\\{([^}]+)\\}`,
      's',
    );
    const match = source.match(pattern);
    if (!match) throw new Error(`Interface ${name} not found in source`);
    return match[1].replace(/\s+/g, ' ').trim();
  }

  it('ContainerInput interface fields match', () => {
    const host = extractInterface(containerRunnerSource, 'ContainerInput');
    const container = extractInterface(containerSource, 'ContainerInput');
    expect(host).toBe(container);
  });
});
