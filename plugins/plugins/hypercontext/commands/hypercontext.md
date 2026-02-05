---
description: Render session state as ASCII map showing context usage, threads, files, and tools. Supports file output for agent handoffs.
argument-hint: [full|compact|threads|heat] [--out <path>]
allowed-tools: [Read, Write, TaskList, Glob]
---

# Hypercontext Command

Render the current session state as a spatial ASCII map. Optionally write to file for agent-to-agent handoff.

## Arguments

$ARGUMENTS

**Modes:**
- `full` (default): Complete ASCII map with all sections
- `compact`: Dense markdown format for high context usage
- `threads`: Thread dependency map only
- `heat`: Recency heat ranking only

**Output:**
- No flag: render to conversation
- `--out <path>`: write to file (e.g., `--out hypercontexts/parent.md`)

## Handoff Protocol

This skill supports **push-hypercontext** for multi-agent orchestration:

### As Parent (before spawning subagent)
```
/hypercontext --out hypercontexts/parent-001.md
```
Then tell subagent: "Parent context at hypercontexts/parent-001.md"

### As Subagent (on entry)
1. Check for `hypercontexts/` directory
2. Read most recent parent context file
3. Understand: what's done, what's open, what's hot, what decisions were made

### As Subagent (on exit)
```
/hypercontext compact --out hypercontexts/child-001-exit.md
```
Parent can read this to understand what you did without parsing your full transcript.

### Directory Convention
```
hypercontexts/
├── parent-001.md          # parent state at subagent spawn
├── child-001-exit.md      # child state on completion
├── parent-002.md          # parent state at next spawn
└── ...
```

Treat this directory like an append-only mailbox. Old contexts accumulate; readers decide what's relevant. Too many? Read recent ones, skim or skip old ones. This is normal.

## Instructions

0. **Parse arguments**: Check for `--out <path>` flag. If present, output will be written to that file using the Write tool. Create parent directories if needed (e.g., `hypercontexts/`).

1. **Gather session data** by reflecting on:
   - Files read and modified this session
   - Tools used and their frequency
   - Active work threads (check TaskList if available)
   - Decisions made and open questions
   - Approximate turn count

2. **Estimate context usage** using heuristic:
   - Base: ~5000 tokens
   - Per turn: ~2000 tokens
   - Per file/skill read: ~2000 tokens
   - Calculate percentage of ~200k total

3. **Render output** based on mode:

### Full Mode (default)
```
╔══════════════════════════════════════════════════════════════════╗
║                    HYPERCONTEXT SESSION MAP                      ║
╠══════════════════════════════════════════════════════════════════╣
║ Context: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] ~XX% (~XXk/200k)  ║
║ Velocity: ▁▂▃▄▅▆▇█                                               ║
╠══════════════════════════════════════════════════════════════════╣
║ THREADS                                                          ║
║ [thread boxes with status glyphs]                                ║
╠══════════════════════════════════════════════════════════════════╣
║ HEAT                                                             ║
║ [topic recency bars]                                             ║
╠══════════════════════════════════════════════════════════════════╣
║ FILES                                                            ║
║ [◆ modified / ◇ read-only]                                       ║
╠══════════════════════════════════════════════════════════════════╣
║ TOOLS                                                            ║
║ [usage histograms]                                               ║
╠══════════════════════════════════════════════════════════════════╣
║ DECISIONS                                                        ║
║ [concise decision log]                                           ║
╠══════════════════════════════════════════════════════════════════╣
║ OPEN                                                             ║
║ [questions/blockers]                                             ║
╚══════════════════════════════════════════════════════════════════╝
```

### Compact Mode
Use when context > 70% or explicitly requested:
```markdown
## Session State (compact)
**Context:** XX% | **Velocity:** ▅▆▇
**Threads:** name(status), name(status)
**Files:** ◆file(edits) ◇file
**Decisions:** brief list
**Open:** blockers
**Next:** immediate action
```

### Threads Mode
Show only the thread dependency map section.

### Heat Mode
Show only the recency heat ranking section.

## Principles

- **Don't hallucinate**: Every metric must reflect actual session activity
- **Estimate honestly**: Context usage is approximate, not measured
- **Degrade gracefully**: Switch to compact when appropriate
- **File-native handoffs**: When `--out` specified, write silently and confirm path. The file IS the communication.

## Output Behavior

**To conversation (no --out):**
Render the ASCII map directly in your response.

**To file (--out path):**
1. Use Glob to check if parent directory exists
2. Create directory if needed with appropriate structure
3. Write the hypercontext to the specified path
4. Respond briefly: "Wrote hypercontext to `<path>`"

The written file should include a header:
```markdown
<!-- hypercontext: <timestamp> -->
<!-- agent: <parent|child> -->
<!-- session: <brief identifier if known> -->
```

This metadata helps readers (human or model) understand provenance when scanning the mailbox.
