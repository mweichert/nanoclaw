/**
 * Tests for the Pi coding agent backend.
 *
 * Mocks the Pi SDK (@mariozechner/pi-coding-agent and @mariozechner/pi-ai)
 * and the shared utilities (fs, writeOutput, etc.) to test the backend in isolation.
 */

import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import type { ContainerInput } from './shared.js';

// ── Mock setup ─────────────────────────────────────────────────────────────

// Track events subscribed via session.subscribe()
let capturedEventListener: ((event: any) => void) | null = null;

// Mock session object
const mockSession = {
  prompt: vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined),
  followUp: vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined),
  steer: vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined),
  subscribe: vi.fn((listener: (event: any) => void) => {
    capturedEventListener = listener;
    return () => { capturedEventListener = null; };
  }),
  dispose: vi.fn(),
  sessionFile: '/tmp/pi-sessions/test-session.jsonl',
  sessionId: 'test-session-id',
};

// Mock createAgentSession
const mockCreateAgentSession = vi.fn().mockResolvedValue({
  session: mockSession,
  extensionsResult: { extensions: [], errors: [] },
  modelFallbackMessage: undefined,
});

// Mock model object
const mockModel = {
  id: 'gemini-2.5-flash',
  name: 'Gemini 2.5 Flash',
  api: 'google-generative-ai' as const,
  provider: 'google',
  baseUrl: 'https://generativelanguage.googleapis.com',
  reasoning: true,
  input: ['text' as const, 'image' as const],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1000000,
  maxTokens: 8192,
};

const mockGetModel = vi.fn().mockReturnValue(mockModel);

// Mock individual tools
const mockReadTool = { name: 'read', label: 'Read' };
const mockBashTool = { name: 'bash', label: 'Bash' };
const mockEditTool = { name: 'edit', label: 'Edit' };
const mockWriteTool = { name: 'write', label: 'Write' };
const mockGrepTool = { name: 'grep', label: 'Grep' };
const mockFindTool = { name: 'find', label: 'Find' };
const mockLsTool = { name: 'ls', label: 'Ls' };
const mockAllTools = [mockReadTool, mockBashTool, mockEditTool, mockWriteTool, mockGrepTool, mockFindTool, mockLsTool];

// Mock AuthStorage - must use function (not arrow) so it works with `new`
const mockAuthStorageInstance = {
  setRuntimeApiKey: vi.fn(),
  getApiKey: vi.fn(),
};
const MockAuthStorage = vi.fn(function (this: any) {
  Object.assign(this, mockAuthStorageInstance);
  return this;
});

// Mock ModelRegistry - must use function (not arrow) so it works with `new`
const mockModelRegistryInstance = {
  find: vi.fn(),
  getApiKey: vi.fn(),
  getAvailable: vi.fn().mockReturnValue([]),
};
const MockModelRegistry = vi.fn(function (this: any) {
  Object.assign(this, mockModelRegistryInstance);
  return this;
});

// Mock SessionManager
const mockSessionManagerInstance = {
  isPersisted: vi.fn().mockReturnValue(true),
  getSessionFile: vi.fn().mockReturnValue('/tmp/pi-sessions/test-session.jsonl'),
};
const MockSessionManager = {
  create: vi.fn().mockReturnValue(mockSessionManagerInstance),
  open: vi.fn().mockReturnValue(mockSessionManagerInstance),
  inMemory: vi.fn().mockReturnValue(mockSessionManagerInstance),
};

// Mock DefaultResourceLoader - must use function (not arrow) for `new`
const mockResourceLoaderInstance = {
  reload: vi.fn().mockResolvedValue(undefined),
  getExtensions: vi.fn().mockReturnValue({ extensions: [], errors: [] }),
  getSkills: vi.fn().mockReturnValue({ skills: [], diagnostics: [] }),
  getPrompts: vi.fn().mockReturnValue({ prompts: [], diagnostics: [] }),
  getThemes: vi.fn().mockReturnValue({ themes: [], diagnostics: [] }),
  getAgentsFiles: vi.fn().mockReturnValue({ agentsFiles: [] }),
  getSystemPrompt: vi.fn().mockReturnValue(undefined),
  getAppendSystemPrompt: vi.fn().mockReturnValue([]),
  getPathMetadata: vi.fn().mockReturnValue(new Map()),
  extendResources: vi.fn(),
};
const MockDefaultResourceLoader = vi.fn(function (this: any) {
  Object.assign(this, mockResourceLoaderInstance);
  return this;
});

// Mock SettingsManager
const mockSettingsManagerInstance = {};
const MockSettingsManager = {
  create: vi.fn().mockReturnValue(mockSettingsManagerInstance),
};

// Register mocks before importing the module under test
vi.mock('@mariozechner/pi-coding-agent', () => ({
  createAgentSession: mockCreateAgentSession,
  AuthStorage: MockAuthStorage,
  ModelRegistry: MockModelRegistry,
  SessionManager: MockSessionManager,
  SettingsManager: MockSettingsManager,
  DefaultResourceLoader: MockDefaultResourceLoader,
  readTool: mockReadTool,
  bashTool: mockBashTool,
  editTool: mockEditTool,
  writeTool: mockWriteTool,
  grepTool: mockGrepTool,
  findTool: mockFindTool,
  lsTool: mockLsTool,
  codingTools: mockAllTools.slice(0, 4),
}));

const mockGetModels = vi.fn().mockReturnValue([mockModel]);

vi.mock('@mariozechner/pi-ai', () => ({
  getModel: mockGetModel,
  getModels: (...args: any[]) => mockGetModels(...args),
  getProviders: vi.fn().mockReturnValue(['google', 'openai', 'anthropic']),
}));

// Mock fs for MCPorter config writing and CLAUDE.md reading
const mockWriteFileSync = vi.fn();
const mockExistsSync = vi.fn().mockReturnValue(false);
const mockReadFileSync = vi.fn().mockReturnValue('');
const mockMkdirSync = vi.fn();
const mockUnlinkSync = vi.fn();

vi.mock('fs', () => ({
  default: {
    writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
    existsSync: (...args: any[]) => mockExistsSync(...args),
    readFileSync: (...args: any[]) => mockReadFileSync(...args),
    mkdirSync: (...args: any[]) => mockMkdirSync(...args),
    unlinkSync: (...args: any[]) => mockUnlinkSync(...args),
    readdirSync: vi.fn().mockReturnValue([]),
  },
  writeFileSync: (...args: any[]) => mockWriteFileSync(...args),
  existsSync: (...args: any[]) => mockExistsSync(...args),
  readFileSync: (...args: any[]) => mockReadFileSync(...args),
  mkdirSync: (...args: any[]) => mockMkdirSync(...args),
  unlinkSync: (...args: any[]) => mockUnlinkSync(...args),
  readdirSync: vi.fn().mockReturnValue([]),
}));

// Mock shared utilities — we control writeOutput, drainIpcInput, etc.
const mockWriteOutput = vi.fn();
const mockLog = vi.fn();
const mockDrainIpcInput = vi.fn().mockReturnValue([]);
const mockShouldClose = vi.fn().mockReturnValue(false);
let waitForIpcResolvers: Array<(value: string | null) => void> = [];
const mockWaitForIpcMessage = vi.fn(() => {
  return new Promise<string | null>((resolve) => {
    waitForIpcResolvers.push(resolve);
  });
});

vi.mock('./shared.js', () => ({
  writeOutput: (...args: any[]) => mockWriteOutput(...args),
  log: (...args: any[]) => mockLog(...args),
  drainIpcInput: () => mockDrainIpcInput(),
  shouldClose: () => mockShouldClose(),
  waitForIpcMessage: () => mockWaitForIpcMessage(),
  IPC_INPUT_DIR: '/workspace/ipc/input',
  IPC_INPUT_CLOSE_SENTINEL: '/workspace/ipc/input/_close',
  IPC_POLL_MS: 500,
  OUTPUT_START_MARKER: '---NANOCLAW_OUTPUT_START---',
  OUTPUT_END_MARKER: '---NANOCLAW_OUTPUT_END---',
}));

// ── Helper ─────────────────────────────────────────────────────────────────

function makeContainerInput(overrides: Partial<ContainerInput> = {}): ContainerInput {
  return {
    prompt: 'Hello, what can you do?',
    groupFolder: 'test-group',
    chatJid: '123@g.us',
    isMain: false,
    backend: { type: 'pi' },
    ...overrides,
  };
}

/**
 * Simulate the session lifecycle: fire events through the captured listener,
 * then resolve the prompt promise.
 */
function simulateAgentResponse(text: string): void {
  if (!capturedEventListener) {
    throw new Error('No event listener captured — was session.subscribe() called?');
  }
  const assistantMessage = {
    role: 'assistant' as const,
    content: [{ type: 'text' as const, text }],
    api: 'google-generative-ai',
    provider: 'google',
    model: 'gemini-2.5-flash',
    usage: { input: 10, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 30, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: 'stop' as const,
    timestamp: Date.now(),
  };
  capturedEventListener({ type: 'message_start', message: assistantMessage });
  capturedEventListener({ type: 'message_end', message: assistantMessage });
  capturedEventListener({ type: 'agent_end', messages: [assistantMessage] });
}

// ── Tests ──────────────────────────────────────────────────────────────────

// Import the module under test AFTER mocks are set up
// We'll use dynamic import in each test suite to ensure mocks are applied

let runPiBackend: (input: ContainerInput) => Promise<void>;

beforeEach(async () => {
  vi.clearAllMocks();
  capturedEventListener = null;
  waitForIpcResolvers = [];
  mockExistsSync.mockReturnValue(false);
  mockDrainIpcInput.mockReturnValue([]);
  mockShouldClose.mockReturnValue(false);

  // Reset the session mock to default behavior
  mockSession.prompt.mockReset();
  mockSession.followUp.mockReset();
  mockSession.dispose.mockReset();
  mockSession.subscribe.mockImplementation((listener: (event: any) => void) => {
    capturedEventListener = listener;
    return () => { capturedEventListener = null; };
  });

  // Default: prompt triggers agent response, then we simulate events
  mockSession.prompt.mockImplementation(async () => {
    // Simulate async: the test will fire events after prompt is called
  });
  mockSession.followUp.mockImplementation(async () => {
    // Same pattern
  });

  mockCreateAgentSession.mockResolvedValue({
    session: mockSession,
    extensionsResult: { extensions: [], errors: [] },
    modelFallbackMessage: undefined,
  });

  // Re-import to pick up fresh mocks
  const mod = await import('./pi-backend.js');
  runPiBackend = mod.runPiBackend;
});

describe('Pi Backend', () => {

  // ── MCPorter Config ────────────────────────────────────────────────────

  describe('MCPorter config generation', () => {
    it('writes MCPorter config to /tmp/mcporter-config.json', async () => {
      const input = makeContainerInput();

      // Make prompt resolve immediately and simulate agent_end so the loop
      // proceeds to waitForIpcMessage, then close.
      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hello!');
      });
      // Close on first wait
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      // Find the writeFileSync call for mcporter config
      const configCall = mockWriteFileSync.mock.calls.find(
        (call: any[]) => typeof call[0] === 'string' && call[0].includes('mcporter')
      );
      expect(configCall).toBeDefined();
      expect(configCall![0]).toBe('/tmp/mcporter-config.json');

      const config = JSON.parse(configCall![1] as string);
      expect(config.mcpServers).toBeDefined();
      expect(config.mcpServers.nanoclaw).toBeDefined();
      expect(config.mcpServers.nanoclaw.command).toBe('node');
      expect(config.mcpServers.nanoclaw.args).toEqual(
        expect.arrayContaining([expect.stringContaining('ipc-mcp-stdio.js')])
      );
    });

    it('sets correct env vars in MCPorter config', async () => {
      const input = makeContainerInput({
        chatJid: 'group123@g.us',
        groupFolder: 'my-group',
        isMain: true,
      });

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Done');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const configCall = mockWriteFileSync.mock.calls.find(
        (call: any[]) => typeof call[0] === 'string' && call[0].includes('mcporter')
      );
      const config = JSON.parse(configCall![1] as string);
      const env = config.mcpServers.nanoclaw.env;

      expect(env.NANOCLAW_CHAT_JID).toBe('group123@g.us');
      expect(env.NANOCLAW_GROUP_FOLDER).toBe('my-group');
      expect(env.NANOCLAW_IS_MAIN).toBe('1');
    });

    it('sets MCPORTER_CONFIG env var', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('ok');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(process.env.MCPORTER_CONFIG).toBe('/tmp/mcporter-config.json');
    });
  });

  // ── Model Resolution ───────────────────────────────────────────────────

  describe('model resolution', () => {
    it('uses backend.provider and backend.model when specified', async () => {
      const anthropicModel = { ...mockModel, id: 'claude-opus-4-5', name: 'Claude Opus 4.5', provider: 'anthropic' };
      mockGetModels.mockImplementation((provider: string) => {
        if (provider === 'anthropic') return [anthropicModel];
        return [mockModel];
      });

      const input = makeContainerInput({
        backend: { type: 'pi', provider: 'anthropic', model: 'claude-opus-4-5' },
      });

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(mockGetModels).toHaveBeenCalledWith('anthropic');
      const sessionOptions = mockCreateAgentSession.mock.calls[0][0];
      expect(sessionOptions.model).toBe(anthropicModel);
    });

    it('falls back to default provider/model when backend has no provider/model', async () => {
      const input = makeContainerInput(); // backend: { type: 'pi' } — no provider/model

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      // Should call getModels with default provider 'google'
      expect(mockGetModels).toHaveBeenCalledWith('google');
    });

    it('passes backend.thinkingLevel to createAgentSession', async () => {
      const input = makeContainerInput({
        backend: { type: 'pi', thinkingLevel: 'high' },
      });

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const sessionOptions = mockCreateAgentSession.mock.calls[0][0];
      expect(sessionOptions.thinkingLevel).toBe('high');
    });
  });

  // ── Session Creation ───────────────────────────────────────────────────

  describe('session creation', () => {
    it('passes cwd and agentDir to createAgentSession', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const sessionOptions = mockCreateAgentSession.mock.calls[0][0];
      expect(sessionOptions.cwd).toBe('/workspace/group');
      expect(sessionOptions.agentDir).toBe('/home/node/.pi/agent');
    });

    it('passes allBuiltInTools to createAgentSession', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const sessionOptions = mockCreateAgentSession.mock.calls[0][0];
      expect(sessionOptions.tools).toEqual(mockAllTools);
    });

    it('passes resolved model to createAgentSession', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const sessionOptions = mockCreateAgentSession.mock.calls[0][0];
      expect(sessionOptions.model).toBe(mockModel);
    });

    it('creates new persistent session when no sessionId provided', async () => {
      const input = makeContainerInput(); // no sessionId

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(MockSessionManager.create).toHaveBeenCalledWith('/workspace/group');
      expect(MockSessionManager.open).not.toHaveBeenCalled();
    });

    it('resumes existing session when sessionId is provided', async () => {
      const input = makeContainerInput({
        sessionId: '/home/node/.pi/agent/sessions/test/session.jsonl',
      });

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(MockSessionManager.open).toHaveBeenCalledWith(
        '/home/node/.pi/agent/sessions/test/session.jsonl',
      );
      expect(MockSessionManager.create).not.toHaveBeenCalled();
    });

    it('falls back to create when open throws', async () => {
      const input = makeContainerInput({
        sessionId: '/home/node/.pi/agent/sessions/test/corrupted.jsonl',
      });

      MockSessionManager.open.mockImplementationOnce(() => {
        throw new Error('corrupted session file');
      });

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(MockSessionManager.open).toHaveBeenCalled();
      expect(MockSessionManager.create).toHaveBeenCalledWith('/workspace/group');
    });
  });

  // ── Output Protocol ────────────────────────────────────────────────────

  describe('output protocol', () => {
    it('calls writeOutput with assistant text after agent finishes', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('The answer is 42.');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'success',
          result: 'The answer is 42.',
        })
      );
    });

    it('captures text from multi-block assistant messages', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        if (!capturedEventListener) throw new Error('No listener');
        const msg = {
          role: 'assistant' as const,
          content: [
            { type: 'text' as const, text: 'Part 1. ' },
            { type: 'thinking' as const, thinking: 'hmm...' },
            { type: 'text' as const, text: 'Part 2.' },
          ],
          api: 'google-generative-ai',
          provider: 'google',
          model: 'gemini-2.5-flash',
          usage: { input: 10, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 30, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
          stopReason: 'stop' as const,
          timestamp: Date.now(),
        };
        capturedEventListener({ type: 'message_end', message: msg });
        capturedEventListener({ type: 'agent_end', messages: [msg] });
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'success',
          result: 'Part 1. Part 2.',
        })
      );
    });

    it('returns newSessionId in output', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Done');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      // Should include the session file path or session ID
      const outputCall = mockWriteOutput.mock.calls.find(
        (call: any[]) => call[0]?.status === 'success' && call[0]?.result !== null
      );
      expect(outputCall).toBeDefined();
      expect(outputCall![0].newSessionId).toBeDefined();
    });
  });

  // ── IPC Follow-up Messages ─────────────────────────────────────────────

  describe('IPC follow-up messages', () => {
    it('delivers follow-up messages via session.followUp()', async () => {
      const input = makeContainerInput();

      // First prompt
      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('First response');
      });

      // Follow-up: deliver a message, then close
      let callCount = 0;
      mockWaitForIpcMessage.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve('Follow-up question');
        }
        return Promise.resolve(null); // close
      });

      mockSession.followUp.mockImplementation(async () => {
        simulateAgentResponse('Follow-up response');
      });

      await runPiBackend(input);

      expect(mockSession.followUp).toHaveBeenCalledWith('Follow-up question');
    });

    it('disposes session on _close sentinel', async () => {
      const input = makeContainerInput();

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Done');
      });
      // Return null (close) immediately
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      expect(mockSession.dispose).toHaveBeenCalled();
    });
  });

  // ── Global CLAUDE.md ───────────────────────────────────────────────────

  describe('global CLAUDE.md loading', () => {
    it('loads /workspace/global/CLAUDE.md content for non-main groups', async () => {
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

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      // The CLAUDE.md content should be prepended to the prompt
      const promptText = mockSession.prompt.mock.calls[0][0];
      expect(promptText).toContain(claudeMdContent);
    });

    it('does not load CLAUDE.md for main group', async () => {
      const input = makeContainerInput({ isMain: true });

      mockExistsSync.mockReturnValue(true);
      mockReadFileSync.mockReturnValue('# Main Group Instructions');

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      // The prompt should NOT contain the CLAUDE.md content
      const promptText = mockSession.prompt.mock.calls[0][0];
      expect(promptText).not.toContain('# Main Group Instructions');
    });

    it('includes MCPorter tools documentation in prompt context', async () => {
      const input = makeContainerInput({ isMain: false });

      mockExistsSync.mockReturnValue(false);

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const promptText = mockSession.prompt.mock.calls[0][0];
      expect(promptText).toContain('mcporter');
      expect(promptText).toContain('send_message');
    });
  });

  // ── Scheduled Task Prefix ──────────────────────────────────────────────

  describe('scheduled task prefix', () => {
    it('prepends scheduled task prefix when isScheduledTask is true', async () => {
      const input = makeContainerInput({
        isScheduledTask: true,
        prompt: 'Run daily report',
      });

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Report generated.');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const promptText = mockSession.prompt.mock.calls[0][0];
      expect(promptText).toContain('SCHEDULED TASK');
      expect(promptText).toContain('Run daily report');
    });

    it('does not prepend prefix for normal messages', async () => {
      const input = makeContainerInput({
        isScheduledTask: false,
        prompt: 'Hello',
      });

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Hi');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const promptText = mockSession.prompt.mock.calls[0][0];
      expect(promptText).not.toContain('SCHEDULED TASK');
    });
  });

  // ── Pending IPC Messages in Initial Prompt ─────────────────────────────

  describe('pending IPC messages', () => {
    it('drains pending IPC messages into initial prompt', async () => {
      const input = makeContainerInput({ prompt: 'Original prompt' });
      mockDrainIpcInput.mockReturnValue(['Extra message 1', 'Extra message 2']);

      mockSession.prompt.mockImplementation(async () => {
        simulateAgentResponse('Done');
      });
      mockWaitForIpcMessage.mockResolvedValueOnce(null);

      await runPiBackend(input);

      const promptText = mockSession.prompt.mock.calls[0][0];
      expect(promptText).toContain('Original prompt');
      expect(promptText).toContain('Extra message 1');
      expect(promptText).toContain('Extra message 2');
    });
  });

  // ── Auth Setup ─────────────────────────────────────────────────────────

  describe('auth setup', () => {
    it('sets runtime API keys from env vars', async () => {
      const input = makeContainerInput();

      // Set env vars
      const origGemini = process.env.GEMINI_API_KEY;
      const origOpenai = process.env.OPENAI_API_KEY;
      const origAnthropic = process.env.ANTHROPIC_API_KEY;

      process.env.GEMINI_API_KEY = 'test-gemini-key';
      process.env.OPENAI_API_KEY = 'test-openai-key';
      process.env.ANTHROPIC_API_KEY = 'test-anthropic-key';

      try {
        mockSession.prompt.mockImplementation(async () => {
          simulateAgentResponse('Hi');
        });
        mockWaitForIpcMessage.mockResolvedValueOnce(null);

        await runPiBackend(input);

        expect(mockAuthStorageInstance.setRuntimeApiKey).toHaveBeenCalledWith(
          'google', 'test-gemini-key'
        );
        expect(mockAuthStorageInstance.setRuntimeApiKey).toHaveBeenCalledWith(
          'openai', 'test-openai-key'
        );
        expect(mockAuthStorageInstance.setRuntimeApiKey).toHaveBeenCalledWith(
          'anthropic', 'test-anthropic-key'
        );
      } finally {
        // Restore
        if (origGemini !== undefined) process.env.GEMINI_API_KEY = origGemini;
        else delete process.env.GEMINI_API_KEY;
        if (origOpenai !== undefined) process.env.OPENAI_API_KEY = origOpenai;
        else delete process.env.OPENAI_API_KEY;
        if (origAnthropic !== undefined) process.env.ANTHROPIC_API_KEY = origAnthropic;
        else delete process.env.ANTHROPIC_API_KEY;
      }
    });
  });

  // ── Error Handling ─────────────────────────────────────────────────────

  describe('error handling', () => {
    it('writes error output when createAgentSession fails', async () => {
      const input = makeContainerInput();
      mockCreateAgentSession.mockRejectedValueOnce(new Error('Session creation failed'));

      // Mock process.exit to prevent actual exit
      const mockExit = vi.spyOn(process, 'exit').mockImplementation((() => {
        throw new Error('process.exit called');
      }) as any);

      try {
        await runPiBackend(input).catch(() => {});
      } catch {
        // Expected — process.exit throws
      }

      expect(mockWriteOutput).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'error',
          error: expect.stringContaining('Session creation failed'),
        })
      );

      mockExit.mockRestore();
    });
  });
});
