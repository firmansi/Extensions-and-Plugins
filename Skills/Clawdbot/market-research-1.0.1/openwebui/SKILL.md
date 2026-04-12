---
name: Market Research
slug: market-research
description: Research markets with sizing, segmentation, competitor mapping, pricing checks, and demand validation that turn fuzzy ideas into decision-ready evidence. Use when you need TAM, SAM, SOM sizing, competitor analysis, segment selection, pricing research, whitespace mapping, or expansion decisions. This version integrates with OpenWebUI and Open Terminal for web research.
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
Anchor to one decision: enter/avoid, prioritize, position, validate. Research without a decision target becomes a document full of facts and no leverage.

### 2. Size in Layers, Not Headlines

| Layer | Question | Failure |
|-------|----------|---------|
| TAM | How large is the broad category? | Too abstract |
| SAM | Which part is reachable? | Overstates |
| SOM | What can realistically be won? | Fantasy |

Never stop at a single big number. Show the formula, assumptions, and confidence level. A smaller defensible number is better than a huge vague one.

### 3. Triangulate Evidence
Use 3+ evidence families:
- Market structure (census, filings, reports)
- Behavior data (search trends, reviews, job posts)
- Direct evidence (interviews, surveys)

**Confidence Ladder:**
| Confidence | Typical Evidence | How to Use It |
|------------|------------------|---------------|
| **High** | Public filings, first-party pricing pages, direct customer behavior, signed pilots, prepayments | Strong enough to support a recommendation |
| **Medium** | Review mining, job posts, trend data, credible analyst reports, repeated interviews | Useful for directional judgment with caveats |
| **Low** | Founder claims, press releases, one-off anecdotes, generic social chatter | Use only as a lead, never as the core conclusion |

### 4. Segment Before Generalize
Do not treat "the market" as one blob. Split by:
- Customer type
- Company size
- Geography
- Urgency of problem
- Willingness to pay
- Existing alternatives

### 5. Map Competition Around Customer Choice

Competitor analysis includes:
- **Direct competitors** - Same solution, same customer
- **Indirect substitutes** - Different solution, same problem
- **Internal workarounds** - Spreadsheets, agencies, manual processes
- **Future entrants** - Companies with clear adjacency

**Where to look for competitor data:**
- **B2B SaaS:** G2, Capterra, TrustRadius
- **Consumer Apps:** App Store, Play Store
- **Physical Products:** Amazon, specialized forums
- **Services:** Yelp, Google Reviews, industry forums

### 6. Favor Revealed Demand Over Stated Enthusiasm

**Strong signals (Worth Building):**
- Customer gives you money (prepayment, LOI)
- Customer spends significant time helping you
- Customer introduces you to others with same problem
- Customer describes workarounds they've built
- Customer articulates the problem in your words

**Weak signals (Keep Digging):**
- "I'd definitely use that" - Polite enthusiasm, not commitment
- "Great idea!" - Compliment, not validation
- Survey says 80% interested - Stated preference ≠ revealed preference
- Lots of social media engagement - Attention ≠ willingness to pay

### 7. Decision-Ready Recommendation

Every deliverable should end with:

```
RECOMMENDATION
- What the evidence supports
- What remains uncertain  
- What should happen next
- What would change the recommendation
```

Good market research reduces uncertainty. Great market research makes the next move obvious.

## Using Open Terminal for Market Research

When connected to OpenWebUI with Open Terminal, use these commands for web research:

### Data Collection Commands
- `curl` - Retrieve web pages and API responses
- `wget` - Download files and documents from the web
- `nslookup` / `dig` - Gather DNS information about competitors
- `grep` - Filter and search through downloaded data
- `jq` - Parse JSON responses from APIs

### Research Workflow
1. **Plan first**: Identify 3-5 angles to investigate
2. **Gather information**: Use curl to fetch content from authoritative sources
3. **Parse data**: Extract key claims and data points
4. **Organize**: Structure into a clear document
5. **Flag conflicts**: Note where sources disagree

### Ethical Research Guidelines
- ✅ Use public filings and press releases
- ✅ Use published pricing pages
- ✅ Use user reviews and forums
- ✅ Access government databases and statistics
- ❌ Do NOT use fake customer inquiries
- ❌ Do NOT access private competitor systems
- ❌ Do NOT scrape behind paywalls
- ❌ Do NOT access internal documents

## Common Traps

- **Top-down theater** → Huge numbers = false confidence
- **Competitor tunnel vision** → Missing substitutes and workarounds
- **Segment blur** → Mixing SMB, enterprise, prosumer corrupts conclusion
- **Source recency failure** → Stale data leads to wrong decisions
- **Opinion inflation** → Excitement ≠ demand
- **No confidence labeling** → Same weight for all evidence
- **Research with no recommendation** → Report without actionable path

## Security & Privacy

This skill does NOT:
- Make unauthorized outbound requests
- Fabricate customer signals or fake interviews
- Access private competitor systems
- Create persistent memory by default
- Store secrets unless the user explicitly asks

Live web research is appropriate when the task requires current market data or the user asks for external evidence. Always respect robots.txt, rate limits, and terms of service.

---

*Version: 1.0.1 | Optimized for OpenWebUI + Open Terminal*