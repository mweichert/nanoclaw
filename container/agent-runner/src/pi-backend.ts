/**
 * Pi coding agent backend for NanoClaw.
 * Uses the @mariozechner/pi-coding-agent SDK to run an agent session,
 * with MCPorter bridging the nanoclaw MCP server for tool access.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  createAgentSession,
  AuthStorage,
  ModelRegistry,
  SessionManager,
  SettingsManager,
  DefaultResourceLoader,
  readTool,
  bashTool,
  editTool,
  writeTool,
  grepTool,
  findTool,
  lsTool,
} from '@mariozechner/pi-coding-agent';
import type {
  AgentSessionEvent,
  CreateAgentSessionOptions,
} from '@mariozechner/pi-coding-agent';
import { getProviders, getModels } from '@mariozechner/pi-ai';
import type { KnownProvider as PiKnownProvider } from '@mariozechner/pi-ai';
import type { ThinkingLevel } from '@mariozechner/pi-agent-core';

import {
  type ContainerInput,
  writeOutput,
  log,
  IPC_INPUT_DIR,
  IPC_INPUT_CLOSE_SENTINEL,
  IPC_POLL_MS,
  shouldClose,
  drainIpcInput,
  waitForIpcMessage,
} from './shared.js';

// ── Constants ──────────────────────────────────────────────────────────────

const MCPORTER_CONFIG_PATH = '/tmp/mcporter-config.json';
const DEFAULT_PROVIDER = 'google';
const DEFAULT_MODEL = 'gemini-2.5-flash';
const DEFAULT_THINKING_LEVEL: ThinkingLevel = 'medium';

function isPiKnownProvider(s: string): s is PiKnownProvider {
  return (getProviders() as string[]).includes(s);
}

const MCPORTER_TOOLS_DOC = `
## Available NanoClaw Tools (via MCPorter)

You can call nanoclaw tools using the bash tool:
- \`npx mcporter call nanoclaw.send_message text:"message"\` -- Send a message to the group
- \`npx mcporter call nanoclaw.schedule_task prompt:"task" schedule_type:"cron" schedule_value:"0 9 * * *"\` -- Schedule a task
- \`npx mcporter call nanoclaw.list_tasks\` -- List scheduled tasks
- \`npx mcporter call nanoclaw.pause_task task_id:"id"\` -- Pause a task
- \`npx mcporter call nanoclaw.resume_task task_id:"id"\` -- Resume a task
- \`npx mcporter call nanoclaw.cancel_task task_id:"id"\` -- Cancel a task
- \`npx mcporter call nanoclaw.register_group jid:"..." name:"..." folder:"..." trigger:"..."\` -- Register a new group (main only)
`.trim();

// ── MCPorter Configuration ─────────────────────────────────────────────────

function writeMcporterConfig(containerInput: ContainerInput): void {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  const config = {
    mcpServers: {
      nanoclaw: {
        command: 'node',
        args: [mcpServerPath],
        env: {
          NANOCLAW_CHAT_JID: containerInput.chatJid,
          NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
          NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
        },
      },
    },
  };

  fs.writeFileSync(MCPORTER_CONFIG_PATH, JSON.stringify(config, null, 2));
  process.env.MCPORTER_CONFIG = MCPORTER_CONFIG_PATH;
  log(`MCPorter config written to ${MCPORTER_CONFIG_PATH}`);
}

// ── Auth Setup ─────────────────────────────────────────────────────────────

function setupAuth(): AuthStorage {
  const authStorage = new AuthStorage();

  // Set runtime API keys from environment variables
  if (process.env.GEMINI_API_KEY) {
    authStorage.setRuntimeApiKey('google', process.env.GEMINI_API_KEY);
  }
  if (process.env.OPENAI_API_KEY) {
    authStorage.setRuntimeApiKey('openai', process.env.OPENAI_API_KEY);
  }
  if (process.env.ANTHROPIC_API_KEY) {
    authStorage.setRuntimeApiKey('anthropic', process.env.ANTHROPIC_API_KEY);
  }

  return authStorage;
}

// ── Prompt Building ────────────────────────────────────────────────────────

function buildPromptContext(containerInput: ContainerInput): string {
  const parts: string[] = [];

  // Load global CLAUDE.md for non-main groups
  if (!containerInput.isMain) {
    const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
    if (fs.existsSync(globalClaudeMdPath)) {
      const content = fs.readFileSync(globalClaudeMdPath, 'utf-8');
      parts.push(content);
    }
  }

  // Always include MCPorter tools documentation
  parts.push(MCPORTER_TOOLS_DOC);

  return parts.join('\n\n');
}

function buildInitialPrompt(containerInput: ContainerInput): string {
  let prompt = containerInput.prompt;

  // Scheduled task prefix
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }

  // Drain any pending IPC messages
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Prepend context (CLAUDE.md + MCPorter tools)
  const context = buildPromptContext(containerInput);
  if (context) {
    prompt = context + '\n\n---\n\n' + prompt;
  }

  return prompt;
}

// ── Text Extraction ────────────────────────────────────────────────────────

interface ContentBlock {
  type: string;
  text?: string;
}

interface AgentMessage {
  role: string;
  content: ContentBlock[];
}

/**
 * Extract text from an assistant message's content blocks.
 */
function extractAssistantText(message: unknown): string | null {
  const msg = message as Record<string, unknown>;
  if (msg?.role !== 'assistant' || !Array.isArray(msg.content)) {
    return null;
  }
  const textParts = (msg.content as ContentBlock[])
    .filter((block) => block.type === 'text' && block.text)
    .map((block) => block.text!);
  return textParts.length > 0 ? textParts.join('') : null;
}

// ── Main Backend Entry Point ───────────────────────────────────────────────

/**
 * Run the Pi coding agent backend.
 * Receives parsed container input and runs the query loop.
 */
export async function runPiBackend(containerInput: ContainerInput): Promise<void> {
  // 1. Configure MCPorter
  writeMcporterConfig(containerInput);

  // 2. Set up auth
  const authStorage = setupAuth();
  const modelRegistry = new ModelRegistry(authStorage);

  // 3. Resolve model
  const backend = containerInput.backend;
  if (backend.type !== 'pi') {
    throw new Error(`runPiBackend called with backend type: ${backend.type}`);
  }
  const provider = backend.provider || DEFAULT_PROVIDER;
  const modelId = backend.model || DEFAULT_MODEL;
  if (!isPiKnownProvider(provider)) {
    throw new Error(`Unknown provider: ${provider}. Valid: ${getProviders().join(', ')}`);
  }
  const model = getModels(provider).find(m => m.id === modelId);
  if (!model) {
    throw new Error(`Unknown model: ${modelId} for provider ${provider}`);
  }

  const thinkingLevel: ThinkingLevel = backend.thinkingLevel || DEFAULT_THINKING_LEVEL;

  // 4. Create or resume session
  let sessionManager: ReturnType<typeof SessionManager.create>;
  if (containerInput.sessionId) {
    try {
      sessionManager = SessionManager.open(containerInput.sessionId);
      log(`Resuming session: ${containerInput.sessionId}`);
    } catch (err) {
      log(`Failed to open session ${containerInput.sessionId}, creating new: ${err instanceof Error ? err.message : String(err)}`);
      sessionManager = SessionManager.create('/workspace/group');
    }
  } else {
    sessionManager = SessionManager.create('/workspace/group');
    log('Creating new persistent session');
  }
  const settingsManager = SettingsManager.create('/workspace/group', '/home/node/.pi/agent');

  const resourceLoader = new DefaultResourceLoader({
    cwd: '/workspace/group',
    agentDir: '/home/node/.pi/agent',
    settingsManager,
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
  });
  await resourceLoader.reload();

  let session;
  try {
    const sessionOptions: CreateAgentSessionOptions = {
      cwd: '/workspace/group',
      agentDir: '/home/node/.pi/agent',
      authStorage,
      modelRegistry,
      model,
      thinkingLevel,
      tools: [readTool, bashTool, editTool, writeTool, grepTool, findTool, lsTool],
      sessionManager,
      settingsManager,
      resourceLoader,
    };

    const result = await createAgentSession(sessionOptions);
    session = result.session;

    if (result.modelFallbackMessage) {
      log(`Model fallback: ${result.modelFallbackMessage}`);
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Failed to create agent session: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      error: errorMessage,
    });
    process.exit(1);
  }

  // 5. Subscribe to events — capture assistant text
  let lastAssistantText: string | null = null;
  let agentEndResolve: () => void;
  let agentEndPromise = Promise.resolve();

  function resetAgentEnd(): void {
    agentEndPromise = new Promise<void>((r) => { agentEndResolve = r; });
  }

  // Single-waiter: only one caller should await this between each resetAgentEnd() call.
  function waitForAgentEnd(): Promise<void> {
    return agentEndPromise;
  }

  const unsubscribe = session.subscribe((event: AgentSessionEvent) => {
    if (event.type === 'message_end') {
      const text = extractAssistantText(event.message);
      if (text !== null) {
        lastAssistantText = text;
      }
    }

    if (event.type === 'agent_end') {
      agentEndResolve();
    }
  });

  // Prepare IPC directory and clean stale sentinel
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // 6. Build initial prompt
  const initialPrompt = buildInitialPrompt(containerInput);
  const sessionId = session.sessionFile || session.sessionId;

  // 7. Query loop
  let ipcPolling = false;
  let closedDuringQuery = false;

  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query');
      closedDuringQuery = true;
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    for (const text of messages) {
      log(`Steering IPC message into active query (${text.length} chars)`);
      session.steer(text);
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };

  try {
    // First query
    log(`Starting Pi agent (session: ${sessionId})...`);
    resetAgentEnd();
    closedDuringQuery = false;
    ipcPolling = true;
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
    await session.prompt(initialPrompt);
    await waitForAgentEnd();
    ipcPolling = false;

    // Emit result
    writeOutput({
      status: 'success',
      result: lastAssistantText,
      newSessionId: sessionId,
    });

    // If close sentinel arrived during the first query, exit immediately
    // without emitting a session-update marker (matches Claude backend behavior).
    if (closedDuringQuery) {
      log('Close sentinel consumed during query, exiting');
    }

    // Loop: wait for IPC messages
    while (!closedDuringQuery) {
      // Emit session update
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');
      const nextMessage = await waitForIpcMessage();

      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), sending follow-up`);

      // Follow-up query
      lastAssistantText = null;
      resetAgentEnd();
      closedDuringQuery = false;
      ipcPolling = true;
      setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
      await session.followUp(nextMessage);
      await waitForAgentEnd();
      ipcPolling = false;

      // Emit result
      writeOutput({
        status: 'success',
        result: lastAssistantText,
        newSessionId: sessionId,
      });

      // If close sentinel arrived during follow-up, exit immediately.
      if (closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage,
    });
    process.exit(1);
  } finally {
    unsubscribe();
    session.dispose();
  }
}
