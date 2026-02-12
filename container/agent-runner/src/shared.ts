/**
 * Shared types, constants, and utilities for NanoClaw agent backends.
 */

import fs from 'fs';
import path from 'path';

import { z } from 'zod';

// ── Types (mirrored from ../../src/types.ts — keep in sync) ─────────────────

export type ThinkingLevel = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh';

export type KnownProvider = 'google' | 'openai' | 'anthropic';
export type Provider = KnownProvider | (string & {});

export type AgentBackend =
  | { type: 'claude' }
  | { type: 'pi'; provider?: Provider; model?: string; thinkingLevel?: ThinkingLevel };

// ── Zod schemas ─────────────────────────────────────────────────────────────

const thinkingLevelSchema = z.enum(['off', 'minimal', 'low', 'medium', 'high', 'xhigh']);

const claudeBackendSchema = z.object({ type: z.literal('claude') });

const piBackendSchema = z.object({
  type: z.literal('pi'),
  provider: z.string().optional(),
  model: z.string().optional(),
  thinkingLevel: thinkingLevelSchema.optional(),
});

const agentBackendSchema = z.discriminatedUnion('type', [claudeBackendSchema, piBackendSchema]);

const containerInputSchema = z.object({
  prompt: z.string(),
  sessionId: z.string().optional(),
  groupFolder: z.string(),
  chatJid: z.string(),
  isMain: z.boolean(),
  isScheduledTask: z.boolean().optional(),
  secrets: z.record(z.string()).optional(),
  backend: agentBackendSchema.default({ type: 'claude' }),
});

// ── Interfaces ──────────────────────────────────────────────────────────────

export interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  secrets?: Record<string, string>;
  backend: AgentBackend;
}

// ── Validation ──────────────────────────────────────────────────────────────

export function validateContainerInput(data: unknown): ContainerInput {
  return containerInputSchema.parse(data);
}

export interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

export interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

export interface SessionsIndex {
  entries: SessionEntry[];
}

export interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

// ── Constants ───────────────────────────────────────────────────────────────

export const IPC_INPUT_DIR = '/workspace/ipc/input';
export const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
export const IPC_POLL_MS = 500;

export const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
export const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

// ── Utility functions ───────────────────────────────────────────────────────

export function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

export function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

export async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

// ── IPC functions ───────────────────────────────────────────────────────────

/**
 * Check for _close sentinel.
 */
export function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

/**
 * Drain all pending IPC input messages.
 * Returns messages found, or empty array.
 */
export function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages as a single string, or null if _close.
 */
export function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}
