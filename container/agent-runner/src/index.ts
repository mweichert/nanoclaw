/**
 * NanoClaw Agent Runner — Dispatcher
 * Runs inside a container, receives config via stdin, routes to the appropriate backend.
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 *   Final marker after loop ends signals completion.
 */

import { readStdin, writeOutput, log, validateContainerInput } from './shared.js';
import { runClaudeBackend } from './claude-backend.js';
import { runPiBackend } from './pi-backend.js';

async function main(): Promise<void> {
  let containerInput;

  try {
    const stdinData = await readStdin();
    const parsed = JSON.parse(stdinData);
    containerInput = validateContainerInput(parsed);
    log(`Received input for group: ${containerInput.groupFolder}, agent: ${containerInput.backend.type}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  if (containerInput.backend.type === 'pi') {
    await runPiBackend(containerInput);
  } else {
    await runClaudeBackend(containerInput);
  }
}

main();
