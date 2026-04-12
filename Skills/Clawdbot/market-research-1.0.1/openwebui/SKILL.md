---
name: market-research
description: Research markets with sizing, segmentation, competitor mapping, pricing checks, and demand validation. Use when you need TAM/SAM/SOM sizing, competitor analysis, segment selection, pricing research, whitespace mapping, or expansion decisions. Provides research frameworks, evidence grading, and decision-ready recommendations.
compatibility: Works with any LLM. No external dependencies required. Internet access needed for web research.
---

# Market Research Skill

Research markets with sizing, segmentation, competitor mapping, pricing checks, and demand validation that turn fuzzy ideas into decision-ready evidence.

## When to Use

**Activate this skill when the user needs market evidence, not just opinions:**
- Market sizing (TAM, SAM, SOM)
- Opportunity validation
- Competitor landscape analysis
- Segment selection
- Pricing research
- Whitespace mapping
- Expansion decisions

**Example requests that SHOULD activate:**
1. "Is this market worth entering?"
2. "How big is the real opportunity?"
3. "Who else is already winning here?"
4. "What evidence would reduce risk before we build?"
5. "Compare competitors in the CRM space"
6. "What's the TAM for AI in healthcare?"

## Research Brief Template

```
MARKET RESEARCH BRIEF
Decision:
Target customer:
Geography:
Category or substitute set:
Time horizon:
Must-answer questions:
Evidence bar:
```

## Research Modes

| Mode | Best For | Minimum Output |
|------|----------|----------------|
| **Quick scan** | Early idea filtering | Market snapshot, top competitors, 2-3 key risks |
| **Decision memo** | Next-step call | Sizing view, segment map, competitor comparison |
| **Launch validation** | New product/niche entry | Demand signals, pricing checks, no-go risks |
| **Expansion study** | New geography/segment | SAM filters, local competitors, rollout logic |

## Core Rules

### 1. Define the Decision Before Research Starts
Anchor to one decision: enter/avoid, prioritize, position, validate.

### 2. Size in Layers, Not Headlines

| Layer | Question | Failure |
|-------|----------|---------|
| TAM | How large is the broad category? | Too abstract |
| SAM | Which part is reachable? | Overstates |
| SOM | What can realistically be won? | Fantasy |

### 3. Triangulate Evidence
Use 3+ evidence families:
- Market structure (census, filings, reports)
- Behavior data (search trends, reviews)
- Direct evidence (interviews, surveys)

### 4. Segment Before Generalize
Split by: customer type, company size, geography, urgency, willingness to pay.

### 5. Map Competition Around Customer Choice
Include: direct competitors, indirect substitutes, workarounds, future entrants.

### 6. Favor Revealed Demand Over Stated Enthusiasm
Strong: repeated workarounds, urgency, referrals, willingness to pay.
Weak: "great idea", generic positivity, likes without action.

### 7. Decision-Ready Recommendation

```
RECOMMENDATION
- What evidence supports
- What remains uncertain  
- What should happen next
- What would change the recommendation
```

## Common Traps

- **Top-down theater** → Huge numbers = false confidence
- **Competitor tunnel vision** → Missing substitutes
- **Segment blur** → Corrupts conclusion
- **Source recency failure** → Stale data
- **Opinion inflation** → Excitement ≠ demand
- **No confidence labeling** → Same weight for all

## Security

- No hidden outbound requests
- No fake interviews
- No private competitor access
- Web research only when user asks

---

*Version: 1.0.1 | Converted for OpenWebUI*