---
name: hypercontext
description: This skill should be used when the user asks to "show session state", "visualize context", "see my threads", "check context usage", "hypercontext", "session map", "context runway", or needs to understand their current session state, work threads, file modifications, or tool usage patterns. Provides ASCII visualization of session metadata.
version: 0.1.0
---

# Hypercontext

> "The agent that can see its own state can reason about its own limitations."

Self-introspection skill for Claude Code sessions. Transforms the invisible context window into observable, navigable information architecture using cartographic ASCII primitives.

## When This Skill Activates

- User asks about session state or context usage
- User wants to see work threads or progress
- User needs to understand what files have been touched
- User asks for "hypercontext" or "session map"
- Context pressure is high and user needs orientation

## Output Format

Generate an ASCII map with the following sections:

### Context Bar
35-character progress indicator showing token usage estimation:
```
[████████████░░░░░░░░░░░░░░░░░░░░░░░] 25% (50k/200k)
```
**Heuristic**: ~5000 base + 2000 per turn + 2000 per file/skill read

### Velocity Sparkline
Interaction momentum across the session using Unicode blocks:
```
▁▂▃▄▅▆▇█
```

### Thread Map
Dependency-aware task visualization:
```
┌─────────────────────────┐
│ Thread: Auth refactor   │
│ Status: ⏳ in-progress  │
│ Blocks: [API migration] │
└─────────────────────────┘
```
**Glyphs**: `✅` done | `⏳` in-progress | `❌` blocked | `💡` idea

### Heat Ranking
Recency-based topic salience (NO fabricated importance scores):
```
auth-system    ████████░░ (3 turns ago)
api-routes     ██████░░░░ (7 turns ago)
test-coverage  ███░░░░░░░ (12 turns ago)
```

### File Tracking
Modified vs read-only with edit counts:
```
◆ src/auth.ts [3 edits]
◆ src/routes.ts [1 edit]
◇ package.json [read]
◇ README.md [read]
```

### Tool Histogram
Usage frequency as scaled bars:
```
Read   ████████████ (24)
Edit   ██████ (12)
Bash   ████ (8)
Grep   ██ (4)
```

### Decisions Log
Concise record of choices made this session:
```
- Chose JWT over session tokens (user preference)
- Using Prisma over raw SQL (existing setup)
```

### Open Questions
Blockers or ambiguities requiring resolution:
```
? Database migration strategy unclear
? Test coverage threshold not specified
```

## Compact Mode

When context usage exceeds ~70%, use dense markdown without ASCII framing:

```markdown
## Session State (compact)
**Context:** 72% | **Velocity:** ▅▆▇
**Threads:** auth-refactor(⏳), api-migration(❌ blocked)
**Files:** ◆auth.ts(3) ◆routes.ts(1) ◇package.json
**Decisions:** JWT auth, Prisma ORM
**Open:** migration strategy, test threshold
**Next:** resolve migration blocker, then continue API work
**Paths:** src/auth.ts:45, src/routes.ts:120
```

## Design Principles

1. **Honesty over estimation**: Every metric reflects actual session activity. No fabricated importance scores or hallucinated state.

2. **Cartographic vocabulary**: Uses spatial metaphors (territory, heat, runway) to make abstract context tangible.

3. **Degradation-aware**: Compact mode preserves essential state when context pressure increases.

## What Can Be Observed

The model CAN observe from conversation history:
- Files read and modified
- Tools invoked and frequency
- Task state (if TaskList/TaskGet were used)
- Turn count and recency

The model CANNOT directly observe:
- Exact token count consumed
- Precise context window remaining
- Internal system state

**Estimates are approximations, not measurements.**

## Agent Handoff Protocol

Hypercontext supports **push-based** state sharing between agents via files.

### Directory Convention
```
hypercontexts/
├── parent-001.md      # parent state before spawning child
├── child-001-exit.md  # child state on completion
└── ...                # append-only mailbox pattern
```

### As Parent Agent
Before spawning a subagent, write your state:
```
/hypercontext --out hypercontexts/parent-001.md
```
Then instruct the subagent: "Parent context at hypercontexts/parent-001.md"

### As Child Agent
1. On entry: read `hypercontexts/` for parent context
2. On exit: write your own state for parent to read

### Why Files?
- No bidirectional protocol needed
- Policy models read text; ASCII art is as machine-readable as JSON to an LLM
- Append-only mailbox scales: too many old contexts? Skip them. This is fine.
- Files survive session boundaries

The theatricality of the ASCII format is load-bearing: it's designed for readers (human or model) who appreciate spatial layout and semantic compression.
