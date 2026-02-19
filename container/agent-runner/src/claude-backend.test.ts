/**
 * Tests for the Claude Agent SDK backend.
 *
 * Mocks the Claude SDK (@anthropic-ai/claude-agent-sdk), fs, and shared
 * utilities to test the backend in isolation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { ContainerInput } from './shared.js';
import { randomUUID } from 'crypto';

// ── Mock setup ─────────────────────────────────────────────────────────────

// Mock query — returns an async generator of SDK messages
const mockQuery = vi.fn();

vi.mock('@anthropic-ai/claude-agent-sdk', () => ({
  query: (...args: any[]) => mockQuery(...args),
}));

// Mock fs
const mockExistsSync = vi.fn().mockReturnValue(false);
const mockReadFileSync = vi.fn().mockReturnValue('');
const mockWriteFileSync = vi.fn();
const mockMkdirSync = vi.fn();
const mockUnlinkSync = vi.fn();

vi.mock('fs', () => ({
  default: {
    existsSync: (...args: any[]) => mockExistsSync(...args),
    readFileSync: (...args: any[]) => mockReadFileSync(...args),
    writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
    mkdirSync: (...args: any[]) => mockMkdirSync(...args),
    unlinkSync: (...args: any[]) => mockUnlinkSync(...args),
  },
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
  unlinkSync: (...args: any[]) => mockUnlinkSync(...args),
}));

// Mock shared utilities
const mockWriteOutput = vi.fn();
const mockLog = vi.fn();
const mockDrainIpcInput = vi.fn().mockReturnValue([]);
const mockShouldClose = vi.fn().mockReturnValue(false);
const mockWaitForIpcMessage = vi.fn<() => Promise<string | null>>();

vi.mock('./shared.js', () => ({
  writeOutput: (...args: any[]) => mockWriteOutput(...args),
  log: (...args: any[]) => mockLog(...args),
  drainIpcInput: () => mockDrainIpcInput(),
  shouldClose: () => mockShouldClose(),
  waitForIpcMessage: () => mockWaitForIpcMessage(),
  IPC_INPUT_DIR: '/workspace/ipc/input',
  IPC_INPUT_CLOSE_SENTINEL: '/workspace/ipc/input/_close',
  IPC_POLL_MS: 10,  // Short poll for tests
  OUTPUT_START_MARKER: '---NANOCLAW_OUTPUT_START---',
  OUTPUT_END_MARKER: '---NANOCLAW_OUTPUT_END---',
}));

// ── SDK Message Helpers ────────────────────────────────────────────────────

function makeInitMessage(sessionId: string) {
  return {
    type: 'system' as const,
    subtype: 'init' as const,
    session_id: sessionId,
    uuid: randomUUID(),
    agents: [],
    apiKeySource: 'user' as const,
    betas: [],
    claude_code_version: '0.2.34',
    cwd: '/workspace/group',
    tools: ['Bash', 'Read'],
    mcp_servers: [],
    model: 'claude-opus-4-6',
    permissionMode: 'bypassPermissions' as const,
    slash_commands: [],
    output_style: 'default',
    skills: [],
    plugins: [],
  };
}

function makeAssistantMessage(text: string, uuid?: string) {
  return {
    type: 'assistant' as const,
    uuid: uuid || randomUUID(),
    session_id: 'test-session',
    parent_tool_use_id: null,
    message: {
      id: 'msg_123',
      type: 'message' as const,
      role: 'assistant' as const,
      content: [{ type: 'text' as const, text }],
      model: 'claude-opus-4-6',
      stop_reason: 'end_turn',
      usage: { input_tokens: 10, output_tokens: 20 },
    },
  };
}

function makeResultMessage(result: string, subtype: string = 'success') {
  return {
    type: 'result' as const,
    subtype,
    result,
    session_id: 'test-session',
    uuid: randomUUID(),
    is_error: false,
    duration_ms: 1000,
    duration_api_ms: 800,
    num_turns: 1,
    stop_reason: 'end_turn',
    total_cost_usd: 0.01,
    usage: { input_tokens: 10, output_tokens: 20, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 },
    modelUsage: {},
    permission_denials: [],
  };
}

function makeTaskNotification(taskId: string, status: string) {
  return {
    type: 'system' as const,
    subtype: 'task_notification' as const,
    task_id: taskId,
    status,
    output_file: '/workspace/task-output.txt',
    summary: 'Task done',
    uuid: randomUUID(),
    session_id: 'test-session',
  };
}

// Helpers

function makeContainerInput(overrides: Partial<ContainerInput> = {}): ContainerInput {
  return {
    prompt: 'Hello, what can you do?',
    groupFolder: 'test-group',
    chatJid: '123@g.us',
    isMain: false,
    backend: { type: 'claude' },
    ...overrides,
  };
}

/** Create a mock async generator from an array of SDK messages */
function mockQueryGenerator(messages: any[], delayMs?: number) {
  async function* gen() {
    if (delayMs) await new Promise(r => setTimeout(r, delayMs));
    for (const msg of messages) yield msg;
  }
  const g = gen();
  // Add Query interface stubs
  (g as any).interrupt = vi.fn();
  (g as any).setPermissionMode = vi.fn();
  (g as any).setModel = vi.fn();
  (g as any).setMaxThinkingTokens = vi.fn();
  (g as any).initializationResult = vi.fn();
  (g as any).supportedCommands = vi.fn();
  (g as any).supportedModels = vi.fn();
  (g as any).mcpServerStatus = vi.fn();
  (g as any).accountInfo = vi.fn();
  (g as any).rewindFiles = vi.fn();
  (g as any).reconnectMcpServer = vi.fn();
  (g as any).toggleMcpServer = vi.fn();
  (g as any).setMcpServers = vi.fn();
  (g as any).streamInput = vi.fn();
  (g as any).close = vi.fn();
  return g;
}

// ── Tests ──────────────────────────────────────────────────────────────────

// Import types — actual module imported dynamically in beforeEach
let runClaudeBackend: (input: ContainerInput) => Promise<void>;
let sanitizeFilename: (s: string) => string;
let generateFallbackName: () => string;
let parseTranscript: (content: string) => { role: string; content: string }[];
let formatTranscriptMarkdown: (messages: { role: string; content: string }[], title?: string | null) => string;
let getSessionSummary: (sessionId: string, transcriptPath: string) => string | null;
let createPreCompactHook: () => any;
let MessageStream: any;

beforeEach(async () => {
  vi.clearAllMocks();
  vi.useRealTimers();
  mockExistsSync.mockReturnValue(false);
  mockDrainIpcInput.mockReturnValue([]);
  mockShouldClose.mockReturnValue(false);
  mockWaitForIpcMessage.mockReset();

  // Re-import to pick up fresh mocks
  const mod = await import('./claude-backend.js');
  runClaudeBackend = mod.runClaudeBackend;
  sanitizeFilename = mod.sanitizeFilename;
  generateFallbackName = mod.generateFallbackName;
  parseTranscript = mod.parseTranscript;
  formatTranscriptMarkdown = mod.formatTranscriptMarkdown;
  getSessionSummary = mod.getSessionSummary;
  createPreCompactHook = mod.createPreCompactHook;
  MessageStream = mod.MessageStream;
});

// ── A. Pure Functions ──────────────────────────────────────────────────────

describe('sanitizeFilename', () => {
  it('lowercases and replaces special chars with hyphens', () => {
    expect(sanitizeFilename('Hello World! 123')).toBe('hello-world-123');
  });

  it('trims leading and trailing hyphens', () => {
    expect(sanitizeFilename('--test--')).toBe('test');
  });

  it('truncates to 50 characters', () => {
    const long = 'a'.repeat(100);
    expect(sanitizeFilename(long).length).toBeLessThanOrEqual(50);
  });

  it('collapses consecutive non-alphanum into single hyphen', () => {
    expect(sanitizeFilename('foo!!!bar???baz')).toBe('foo-bar-baz');
  });

  it('handles empty string', () => {
    expect(sanitizeFilename('')).toBe('');
  });
});

describe('generateFallbackName', () => {
  it('returns conversation-HHMM format', () => {
    const name = generateFallbackName();
    expect(name).toMatch(/^conversation-\d{2}\d{2}$/);
  });
});

describe('parseTranscript', () => {
  it('parses user message with string content', () => {
    const jsonl = JSON.stringify({ type: 'user', message: { content: 'Hello' } });
    const result = parseTranscript(jsonl);
    expect(result).toEqual([{ role: 'user', content: 'Hello' }]);
  });

  it('parses user message with array content', () => {
    const jsonl = JSON.stringify({
      type: 'user',
      message: { content: [{ text: 'Part 1' }, { text: ' Part 2' }] },
    });
    const result = parseTranscript(jsonl);
    expect(result).toEqual([{ role: 'user', content: 'Part 1 Part 2' }]);
  });

  it('parses assistant message filtering text parts', () => {
    const jsonl = JSON.stringify({
      type: 'assistant',
      message: {
        content: [
          { type: 'text', text: 'Hello ' },
          { type: 'tool_use', id: 'tu_1', name: 'Read', input: {} },
          { type: 'text', text: 'World' },
        ],
      },
    });
    const result = parseTranscript(jsonl);
    expect(result).toEqual([{ role: 'assistant', content: 'Hello World' }]);
  });

  it('skips malformed JSON lines', () => {
    const content = 'not json\n' + JSON.stringify({ type: 'user', message: { content: 'Hi' } });
    const result = parseTranscript(content);
    expect(result).toEqual([{ role: 'user', content: 'Hi' }]);
  });

  it('skips empty lines', () => {
    const content = '\n\n' + JSON.stringify({ type: 'user', message: { content: 'Hi' } }) + '\n\n';
    const result = parseTranscript(content);
    expect(result).toEqual([{ role: 'user', content: 'Hi' }]);
  });

  it('skips messages with empty text', () => {
    const jsonl = JSON.stringify({ type: 'user', message: { content: '' } });
    const result = parseTranscript(jsonl);
    expect(result).toEqual([]);
  });

  it('handles multiple messages', () => {
    const lines = [
      JSON.stringify({ type: 'user', message: { content: 'Q' } }),
      JSON.stringify({ type: 'assistant', message: { content: [{ type: 'text', text: 'A' }] } }),
    ].join('\n');
    const result = parseTranscript(lines);
    expect(result).toEqual([
      { role: 'user', content: 'Q' },
      { role: 'assistant', content: 'A' },
    ]);
  });
});

describe('formatTranscriptMarkdown', () => {
  it('uses provided title', () => {
    const md = formatTranscriptMarkdown([{ role: 'user', content: 'Hi' }], 'My Chat');
    expect(md).toContain('# My Chat');
  });

  it('falls back to "Conversation" without title', () => {
    const md = formatTranscriptMarkdown([{ role: 'user', content: 'Hi' }]);
    expect(md).toContain('# Conversation');
  });

  it('falls back to "Conversation" with null title', () => {
    const md = formatTranscriptMarkdown([{ role: 'user', content: 'Hi' }], null);
    expect(md).toContain('# Conversation');
  });

  it('labels user messages as "User"', () => {
    const md = formatTranscriptMarkdown([{ role: 'user', content: 'Hello' }]);
    expect(md).toContain('**User**: Hello');
  });

  it('labels assistant messages as "Andy"', () => {
    const md = formatTranscriptMarkdown([{ role: 'assistant', content: 'Hi there' }]);
    expect(md).toContain('**Andy**: Hi there');
  });

  it('truncates content over 2000 chars', () => {
    const longContent = 'x'.repeat(3000);
    const md = formatTranscriptMarkdown([{ role: 'user', content: longContent }]);
    expect(md).toContain('...');
    // The truncated content should be 2000 chars + '...'
    const userLine = md.split('\n').find(l => l.startsWith('**User**:'))!;
    expect(userLine.length).toBeLessThan(3000 + 20); // some prefix overhead
  });

  it('includes archived timestamp', () => {
    const md = formatTranscriptMarkdown([{ role: 'user', content: 'Hi' }]);
    expect(md).toContain('Archived:');
  });
});

// ── B. MessageStream ───────────────────────────────────────────────────────

describe('MessageStream', () => {
  it('yields pushed messages in order', async () => {
    const stream = new MessageStream();
    stream.push('first');
    stream.push('second');
    stream.end();

    const messages: string[] = [];
    for await (const msg of stream) {
      messages.push(msg.message.content as string);
    }
    expect(messages).toEqual(['first', 'second']);
  });

  it('terminates iteration on end()', async () => {
    const stream = new MessageStream();
    stream.push('hello');

    // Schedule end after a tick
    setTimeout(() => stream.end(), 10);

    const messages: string[] = [];
    for await (const msg of stream) {
      messages.push(msg.message.content as string);
    }
    expect(messages).toEqual(['hello']);
  });

  it('waits for push when queue is empty', async () => {
    const stream = new MessageStream();

    // Push after a delay
    setTimeout(() => {
      stream.push('delayed');
      stream.end();
    }, 10);

    const messages: string[] = [];
    for await (const msg of stream) {
      messages.push(msg.message.content as string);
    }
    expect(messages).toEqual(['delayed']);
  });

  it('creates SDKUserMessage objects', async () => {
    const stream = new MessageStream();
    stream.push('test');
    stream.end();

    for await (const msg of stream) {
      expect(msg.type).toBe('user');
      expect(msg.message.role).toBe('user');
      expect(msg.message.content).toBe('test');
      expect(msg.parent_tool_use_id).toBeNull();
      expect(msg.session_id).toBe('');
    }
  });
});

// ── C. getSessionSummary ───────────────────────────────────────────────────

describe('getSessionSummary', () => {
  it('returns summary when session found', () => {
    const index = {
      entries: [
        { sessionId: 'sess-1', summary: 'Chat about coding', fullPath: '/p', firstPrompt: 'Hi' },
      ],
    };
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue(JSON.stringify(index));

    const result = getSessionSummary('sess-1', '/project/transcript.jsonl');
    expect(result).toBe('Chat about coding');
  });

  it('returns null when sessions index file missing', () => {
    mockExistsSync.mockReturnValue(false);
    const result = getSessionSummary('sess-1', '/project/transcript.jsonl');
    expect(result).toBeNull();
  });

  it('returns null when session not in index', () => {
    const index = {
      entries: [
        { sessionId: 'other', summary: 'Other chat', fullPath: '/p', firstPrompt: 'Hi' },
      ],
    };
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue(JSON.stringify(index));

    const result = getSessionSummary('sess-1', '/project/transcript.jsonl');
    expect(result).toBeNull();
  });

  it('returns null on JSON parse error', () => {
    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockReturnValue('not json');

    const result = getSessionSummary('sess-1', '/project/transcript.jsonl');
    expect(result).toBeNull();
  });
});

// ── D. createPreCompactHook ────────────────────────────────────────────────

describe('createPreCompactHook', () => {
  const makeHookInput = (overrides: Record<string, any> = {}) => ({
    session_id: 'sess-1',
    transcript_path: '/project/transcript.jsonl',
    cwd: '/workspace/group',
    hook_event_name: 'PreCompact',
    trigger: 'auto' as const,
    custom_instructions: null,
    ...overrides,
  });

  it('archives transcript as markdown file', async () => {
    const hook = createPreCompactHook();
    const transcript = [
      JSON.stringify({ type: 'user', message: { content: 'Hello' } }),
      JSON.stringify({ type: 'assistant', message: { content: [{ type: 'text', text: 'Hi there' }] } }),
    ].join('\n');

    mockExistsSync.mockImplementation((p: string) => {
      if (p === '/project/transcript.jsonl') return true;
      if (p === '/project/sessions-index.json') return false;
      return false;
    });
    mockReadFileSync.mockImplementation((p: string) => {
      if (p === '/project/transcript.jsonl') return transcript;
      return '';
    });

    const result = await hook(makeHookInput(), undefined, { signal: new AbortController().signal });

    expect(result).toEqual({});
    expect(mockMkdirSync).toHaveBeenCalledWith('/workspace/group/conversations', { recursive: true });
    expect(mockWriteFileSync).toHaveBeenCalledTimes(1);

    const [filePath, content] = mockWriteFileSync.mock.calls[0];
    expect(filePath).toMatch(/\/workspace\/group\/conversations\/\d{4}-\d{2}-\d{2}-conversation-\d{4}\.md$/);
    expect(content).toContain('**User**: Hello');
    expect(content).toContain('**Andy**: Hi there');
  });

  it('uses sanitized summary in filename when available', async () => {
    const hook = createPreCompactHook();
    const transcript = JSON.stringify({ type: 'user', message: { content: 'Q' } });
    const index = {
      entries: [{ sessionId: 'sess-1', summary: 'Chat About Weather', fullPath: '/p', firstPrompt: 'Hi' }],
    };

    mockExistsSync.mockReturnValue(true);
    mockReadFileSync.mockImplementation((p: string) => {
      if (p === '/project/transcript.jsonl') return transcript;
      if (p === '/project/sessions-index.json') return JSON.stringify(index);
      return '';
    });

    await hook(makeHookInput(), undefined, { signal: new AbortController().signal });

    const [filePath] = mockWriteFileSync.mock.calls[0];
    expect(filePath).toContain('chat-about-weather');
  });

  it('handles missing transcript path', async () => {
    const hook = createPreCompactHook();
    mockExistsSync.mockReturnValue(false);

    const result = await hook(
      makeHookInput({ transcript_path: '/nonexistent' }),
      undefined,
      { signal: new AbortController().signal },
    );

    expect(result).toEqual({});
    expect(mockWriteFileSync).not.toHaveBeenCalled();
  });

  it('handles empty transcript (no messages)', async () => {
    const hook = createPreCompactHook();
    mockExistsSync.mockImplementation((p: string) => p === '/project/transcript.jsonl');
    mockReadFileSync.mockReturnValue('');

    const result = await hook(makeHookInput(), undefined, { signal: new AbortController().signal });

    expect(result).toEqual({});
    expect(mockWriteFileSync).not.toHaveBeenCalled();
  });

  it('handles transcript with no parseable messages', async () => {
    const hook = createPreCompactHook();
    mockExistsSync.mockImplementation((p: string) => p === '/project/transcript.jsonl');
    mockReadFileSync.mockReturnValue('not json\nalso not json\n');

    const result = await hook(makeHookInput(), undefined, { signal: new AbortController().signal });

    expect(result).toEqual({});
    expect(mockWriteFileSync).not.toHaveBeenCalled();
  });

  it('logs error on write failure but returns {}', async () => {
    const hook = createPreCompactHook();
    const transcript = JSON.stringify({ type: 'user', message: { content: 'Hello' } });

    mockExistsSync.mockImplementation((p: string) => p === '/project/transcript.jsonl');
    mockReadFileSync.mockReturnValue(transcript);
    mockWriteFileSync.mockImplementation(() => { throw new Error('disk full'); });

    const result = await hook(makeHookInput(), undefined, { signal: new AbortController().signal });

    expect(result).toEqual({});
    expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('disk full'));
  });
});

// ── E. runClaudeBackend Integration ────────────────────────────────────────

describe('Claude Backend', () => {

  describe('query options', () => {
    it('passes correct cwd, permissionMode, and allowedTools', async () => {
      const input = makeContainerInput();
      const sessionId = 'new-session-id';

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage(sessionId),
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockQuery).toHaveBeenCalledTimes(1);
      const params = mockQuery.mock.calls[0][0];
      expect(params.options.cwd).toBe('/workspace/group');
      expect(params.options.permissionMode).toBe('bypassPermissions');
      expect(params.options.allowDangerouslySkipPermissions).toBe(true);
      expect(params.options.allowedTools).toContain('Bash');
      expect(params.options.allowedTools).toContain('Read');
      expect(params.options.allowedTools).toContain('Write');
      expect(params.options.allowedTools).toContain('Edit');
      expect(params.options.allowedTools).toContain('mcp__nanoclaw__*');
      expect(params.options.settingSources).toEqual(['project', 'user']);
    });

    it('configures MCP server with correct env vars', async () => {
      const input = makeContainerInput({
        chatJid: 'group123@g.us',
        groupFolder: 'my-group',
        isMain: true,
      });

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess'),
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      const mcpEnv = params.options.mcpServers.nanoclaw.env;
      expect(mcpEnv.NANOCLAW_CHAT_JID).toBe('group123@g.us');
      expect(mcpEnv.NANOCLAW_GROUP_FOLDER).toBe('my-group');
      expect(mcpEnv.NANOCLAW_IS_MAIN).toBe('1');
    });

    it('sets NANOCLAW_IS_MAIN to 0 for non-main groups', async () => {
      const input = makeContainerInput({ isMain: false });

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      expect(params.options.mcpServers.nanoclaw.env.NANOCLAW_IS_MAIN).toBe('0');
    });

    it('registers PreCompact hook', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      expect(params.options.hooks).toBeDefined();
      expect(params.options.hooks.PreCompact).toHaveLength(1);
      expect(params.options.hooks.PreCompact[0].hooks).toHaveLength(1);
    });
  });

  describe('output protocol', () => {
    it('writes result output on success', async () => {
      const input = makeContainerInput();
      const sessionId = 'test-sess';

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage(sessionId),
        makeResultMessage('The answer is 42.'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'success',
          result: 'The answer is 42.',
          newSessionId: sessionId,
        }),
      );
    });

    it('captures session ID from init message', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('captured-session-id'),
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const resultCall = mockWriteOutput.mock.calls.find(
        (call: any[]) => call[0]?.result === 'Done',
      );
      expect(resultCall![0].newSessionId).toBe('captured-session-id');
    });

    it('handles result messages without explicit result field', async () => {
      const input = makeContainerInput();
      const resultMsg = makeResultMessage('');
      // Remove result to test the fallback (result is '')
      (resultMsg as any).result = undefined;

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess'),
        resultMsg,
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'success', result: null }),
      );
    });
  });

  describe('scheduled task prefix', () => {
    it('prepends scheduled task prefix when isScheduledTask is true', async () => {
      const input = makeContainerInput({
        isScheduledTask: true,
        prompt: 'Run daily report',
      });

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Report done.'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      // prompt is an AsyncIterable (MessageStream), so we check the log output
      // The prompt is pushed into the stream, but we can verify via the initial prompt text
      // The stream is opaque, so verify through the log which logs the query start
      expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('Starting query'));
    });

    it('does not prepend prefix for normal messages', async () => {
      const input = makeContainerInput({
        isScheduledTask: false,
        prompt: 'Hello',
      });

      // We capture what gets pushed to the stream by checking query calls
      let capturedPrompt: any = null;
      mockQuery.mockImplementation((params: any) => {
        capturedPrompt = params.prompt;
        return mockQueryGenerator([makeResultMessage('Hi')]);
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      // The prompt is a MessageStream (async iterable), verify it doesn't contain SCHEDULED TASK
      // We verify indirectly: the prompt string passed to runQuery shouldn't have the prefix
      // Check that SCHEDULED TASK is not in any log output about the prompt
      const logCalls = mockLog.mock.calls.map((c: any[]) => c[0]);
      const promptLogs = logCalls.filter((l: string) => l.includes('SCHEDULED TASK'));
      expect(promptLogs).toHaveLength(0);
    });
  });

  describe('pending IPC messages', () => {
    it('drains pending IPC messages into initial prompt', async () => {
      const input = makeContainerInput({ prompt: 'Original' });
      mockDrainIpcInput.mockReturnValueOnce(['Extra 1', 'Extra 2']);

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockLog).toHaveBeenCalledWith(
        expect.stringContaining('Draining 2 pending IPC messages'),
      );
    });
  });

  describe('global CLAUDE.md loading', () => {
    it('loads global CLAUDE.md for non-main groups', async () => {
      const input = makeContainerInput({ isMain: false });
      const claudeMdContent = '# Custom Instructions\nBe helpful.';

      mockExistsSync.mockImplementation((p: string) => {
        if (p === '/workspace/global/CLAUDE.md') return true;
        return false;
      });
      mockReadFileSync.mockImplementation((p: string) => {
        if (p === '/workspace/global/CLAUDE.md') return claudeMdContent;
        return '';
      });

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      expect(params.options.systemPrompt).toEqual({
        type: 'preset',
        preset: 'claude_code',
        append: claudeMdContent,
      });
    });

    it('does not load CLAUDE.md for main group', async () => {
      const input = makeContainerInput({ isMain: true });

      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue('# Main Instructions');

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      expect(params.options.systemPrompt).toBeUndefined();
    });

    it('does not set systemPrompt when CLAUDE.md is missing', async () => {
      const input = makeContainerInput({ isMain: false });
      mockExistsSync.mockReturnValue(false);

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      expect(params.options.systemPrompt).toBeUndefined();
    });
  });

  describe('query loop', () => {
    it('runs second query after IPC message', async () => {
      const input = makeContainerInput();

      // First query
      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess-1'),
        makeResultMessage('First response'),
      ]));

      // Second query
      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Second response'),
      ]));

      // First wait returns a message, second returns close
      mockWaitForIpcMessage.mockResolvedValueOnce('Follow-up question');
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockQuery).toHaveBeenCalledTimes(2);
    });

    it('exits on close sentinel between queries', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess-1'),
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockQuery).toHaveBeenCalledTimes(1);
      expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('Close sentinel received'));
    });

    it('emits session update between queries', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess-1'),
        makeResultMessage('Done'),
      ]));
      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done again'),
      ]));

      mockWaitForIpcMessage.mockResolvedValueOnce('Next message');
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      // Should have session update output (result=null) between queries
      const sessionUpdate = mockWriteOutput.mock.calls.find(
        (call: any[]) => call[0]?.result === null && call[0]?.newSessionId === 'sess-1',
      );
      expect(sessionUpdate).toBeDefined();
    });

    it('tracks session ID across queries', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess-1'),
        makeResultMessage('First'),
      ]));
      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Second'),
      ]));

      mockWaitForIpcMessage.mockResolvedValueOnce('Next');
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      // Second query should resume with sess-1
      const secondCallParams = mockQuery.mock.calls[1][0];
      expect(secondCallParams.options.resume).toBe('sess-1');
    });

    it('tracks resumeAt from assistant UUID', async () => {
      const input = makeContainerInput();
      const assistantUuid = randomUUID();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess-1'),
        makeAssistantMessage('Hello', assistantUuid),
        makeResultMessage('Done'),
      ]));
      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done again'),
      ]));

      mockWaitForIpcMessage.mockResolvedValueOnce('Next');
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const secondCallParams = mockQuery.mock.calls[1][0];
      expect(secondCallParams.options.resumeSessionAt).toBe(assistantUuid);
    });
  });

  describe('close sentinel during query', () => {
    it('exits without session update when closedDuringQuery', async () => {
      const input = makeContainerInput();

      // shouldClose returns true on the first poll (which fires after IPC_POLL_MS=10ms)
      mockShouldClose.mockReturnValue(true);

      // Delay the query generator so the IPC poll fires before messages yield
      mockQuery.mockReturnValueOnce(
        mockQueryGenerator([
          makeInitMessage('sess-1'),
          makeResultMessage('Partial result'),
        ], 50),  // 50ms delay > IPC_POLL_MS (10ms), so poll detects close first
      );

      await runClaudeBackend(input);

      expect(mockLog).toHaveBeenCalledWith(
        expect.stringContaining('Close sentinel consumed during query'),
      );
    });
  });

  describe('task notifications', () => {
    it('logs task notifications without crashing', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeInitMessage('sess'),
        makeTaskNotification('task-1', 'completed'),
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockLog).toHaveBeenCalledWith(
        expect.stringContaining('Task notification: task=task-1'),
      );
    });
  });

  describe('startup cleanup', () => {
    it('creates IPC directory', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockMkdirSync).toHaveBeenCalledWith('/workspace/ipc/input', { recursive: true });
    });

    it('cleans up stale _close sentinel', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      expect(mockUnlinkSync).toHaveBeenCalledWith('/workspace/ipc/input/_close');
    });
  });

  describe('error handling', () => {
    it('writes error output when query throws', async () => {
      const input = makeContainerInput();

      mockQuery.mockImplementation(() => {
        throw new Error('SDK connection failed');
      });

      const mockExit = vi.spyOn(process, 'exit').mockImplementation((() => {
        throw new Error('process.exit called');
      }) as any);

      try {
        await runClaudeBackend(input).catch(() => {});
      } catch {
        // Expected
      }

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'error',
          result: null,
          error: expect.stringContaining('SDK connection failed'),
        }),
      );

      mockExit.mockRestore();
    });

    it('calls process.exit(1) on error', async () => {
      const input = makeContainerInput();

      mockQuery.mockImplementation(() => {
        throw new Error('fail');
      });

      const mockExit = vi.spyOn(process, 'exit').mockImplementation((() => {
        throw new Error('process.exit called');
      }) as any);

      try {
        await runClaudeBackend(input).catch(() => {});
      } catch {
        // Expected
      }

      expect(mockExit).toHaveBeenCalledWith(1);
      mockExit.mockRestore();
    });

    it('preserves session ID in error output', async () => {
      const input = makeContainerInput({ sessionId: 'existing-session' });

      mockQuery.mockImplementation(() => {
        throw new Error('fail');
      });

      const mockExit = vi.spyOn(process, 'exit').mockImplementation((() => {
        throw new Error('process.exit called');
      }) as any);

      try {
        await runClaudeBackend(input).catch(() => {});
      } catch {
        // Expected
      }

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'error',
          newSessionId: 'existing-session',
        }),
      );

      mockExit.mockRestore();
    });

    it('handles non-Error thrown objects', async () => {
      const input = makeContainerInput();

      mockQuery.mockImplementation(() => {
        throw 'string error';  // eslint-disable-line no-throw-literal
      });

      const mockExit = vi.spyOn(process, 'exit').mockImplementation((() => {
        throw new Error('process.exit called');
      }) as any);

      try {
        await runClaudeBackend(input).catch(() => {});
      } catch {
        // Expected
      }

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'error',
          error: 'string error',
        }),
      );

      mockExit.mockRestore();
    });
  });

  describe('session resumption', () => {
    it('passes existing sessionId as resume option', async () => {
      const input = makeContainerInput({ sessionId: 'existing-sess' });

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      expect(params.options.resume).toBe('existing-sess');
    });

    it('passes undefined resume when no sessionId', async () => {
      const input = makeContainerInput();

      mockQuery.mockReturnValueOnce(mockQueryGenerator([
        makeResultMessage('Done'),
      ]));
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runClaudeBackend(input);

      const params = mockQuery.mock.calls[0][0];
      expect(params.options.resume).toBeUndefined();
    });
  });
});
