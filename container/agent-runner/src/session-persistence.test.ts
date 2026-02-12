/**
 * Integration test for Pi session persistence.
 *
 * Uses the REAL SessionManager (no mocks) to verify that the session file
 * round-trip works correctly: create → write → open → entries preserved.
 * No API keys needed — we're testing the session file lifecycle, not the LLM.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SessionManager } from '@mariozechner/pi-coding-agent';
import type { AssistantMessage, UserMessage } from '@mariozechner/pi-ai';
import { existsSync, mkdtempSync, rmSync } from 'fs';
import path from 'path';
import os from 'os';

// ── Helpers ─────────────────────────────────────────────────────────────────

function makeUserMessage(text: string): UserMessage {
  return {
    role: 'user',
    content: text,
    timestamp: Date.now(),
  };
}

/** Minimal assistant message satisfying the AssistantMessage interface. */
function makeAssistantMessage(text: string): AssistantMessage {
  return {
    role: 'assistant',
    content: [{ type: 'text', text }],
    api: 'test',
    provider: 'test',
    model: 'test-model',
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: Date.now(),
  };
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe('Pi session persistence round-trip', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'pi-session-test-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('create() produces a valid .jsonl session file path', () => {
    const sm = SessionManager.create(tempDir, tempDir);
    const sessionFile = sm.getSessionFile();

    expect(sessionFile).toBeDefined();
    expect(sessionFile).toMatch(/\.jsonl$/);
    expect(path.isAbsolute(sessionFile!)).toBe(true);
  });

  it('JSONL file is written to disk after assistant message', () => {
    const sm = SessionManager.create(tempDir, tempDir);
    const sessionFile = sm.getSessionFile()!;

    // File doesn't exist yet (SDK defers write until assistant message)
    expect(existsSync(sessionFile)).toBe(false);

    sm.appendMessage(makeUserMessage('Hello'));
    // Still no file — only user message
    expect(existsSync(sessionFile)).toBe(false);

    sm.appendMessage(makeAssistantMessage('Hi there!'));
    // NOW the file should exist
    expect(existsSync(sessionFile)).toBe(true);
  });

  it('open() round-trips the session file path from create()', () => {
    const sm1 = SessionManager.create(tempDir, tempDir);
    sm1.appendMessage(makeUserMessage('test'));
    sm1.appendMessage(makeAssistantMessage('response'));
    const sessionFile = sm1.getSessionFile()!;
    const sessionId = sm1.getSessionId();

    const sm2 = SessionManager.open(sessionFile);
    expect(sm2.getSessionId()).toBe(sessionId);
    expect(sm2.getSessionFile()).toBe(sessionFile);
  });

  it('conversation entries survive the create -> open round-trip', () => {
    const sm1 = SessionManager.create(tempDir, tempDir);
    sm1.appendMessage(makeUserMessage('Hello from test'));
    sm1.appendMessage(makeAssistantMessage('Hi there!'));
    const sessionFile = sm1.getSessionFile()!;

    // Reopen and verify entries are preserved
    const sm2 = SessionManager.open(sessionFile);
    const entries = sm2.getEntries();
    const messages = entries.filter((e) => e.type === 'message');
    expect(messages.length).toBe(2); // user + assistant
  });

  it('sessionFile value is compatible with open() (newSessionId round-trip)', () => {
    // Simulates: pi-backend emits session.sessionFile as newSessionId,
    // host stores it in DB, next container run passes it to SessionManager.open()
    const sm1 = SessionManager.create(tempDir, tempDir);
    sm1.appendMessage(makeUserMessage('prompt'));
    sm1.appendMessage(makeAssistantMessage('answer'));

    const newSessionId = sm1.getSessionFile() || sm1.getSessionId();

    const sm2 = SessionManager.open(newSessionId);
    expect(sm2.isPersisted()).toBe(true);
    expect(sm2.getSessionId()).toBe(sm1.getSessionId());
    expect(sm2.getEntries().length).toBeGreaterThan(0);
  });
});
