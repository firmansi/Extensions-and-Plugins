# Multi Search Engine

Integration of 16 search engines for web searching without requiring API keys.

## Workflow

1. **Language Evaluation**: Detect the language of the search query. For Chinese queries, prioritize Domestic engines (Baidu, Bing CN, 360, Sogou, WeChat, Shenma). For International queries, use International engines (Google, DuckDuckGo, Yahoo, Brave, etc.).

2. **Controlled Search**: Execute search requests with respect for server load:
   - Use reasonable delays (1-2 seconds) between requests.
   - Batch requests where possible.
   - Use standard browser headers to identify as a legitimate user agent to prevent blocks (403/429).

3. **Result Aggregation**: Consolidate successful results from search engines, organize, and summarize them into a coherent search report.

4. **Terminal/Environment Usage**:
   - **OpenWebUI**: Use the built-in search tools or RAG capabilities for automated result aggregation.
   - **Terminal**: Use `curl` or `wget` with properly formatted URLs.
   - **Python/Scripting**: Use libraries like `duckduckgo-search` or `googlesearch-python` for programmatic access.
   - **Verification**: Always inspect the raw HTML or text output of the search result if the scraper fails.

## Search Engines

### Domestic (7)
- **Baidu**: `https://www.baidu.com/s?wd={keyword}`
- **Bing CN**: `https://cn.bing.com/search?q={keyword}&ensearch=0`
- **Bing INT**: `https://cn.bing.com/search?q={keyword}&ensearch=1`
- **360**: `https://www.so.com/s?q={keyword}`
- **Sogou**: `https://sogou.com/web?query={keyword}`
- **WeChat**: `https://wx.sogou.com/weixin?type=2&query={keyword}`
- **Shenma**: `https://m.sm.cn/s?q={keyword}`

### International (9)
- **Google**: `https://www.google.com/search?q={keyword}`
- **Google HK**: `https://www.google.com.hk/search?q={keyword}`
- **DuckDuckGo**: `https://duckduckgo.com/html/?q={keyword}`
- **Yahoo**: `https://search.yahoo.com/search?p={keyword}`
- **Startpage**: `https://www.startpage.com/sp/search?query={keyword}`
- **Brave**: `https://search.brave.com/search?q={keyword}`
- **Ecosia**: `https://www.ecosia.org/search?q={keyword}`
- **Qwant**: `https://www.qwant.com/?q={keyword}`
- **WolframAlpha**: `https://www.wolframalpha.com/input?i={keyword}`

## Quick Examples (Terminal/cURL)

```bash
# Basic Google search
curl -L -A "Mozilla/5.0" "https://www.google.com/search?q=python+tutorial"

# DuckDuckGo Privacy search
curl -L -A "Mozilla/5.0" "https://duckduckgo.com/html/?q=privacy+tools"

# WolframAlpha (Knowledge Query)
curl -L -A "Mozilla/5.0" "https://www.wolframalpha.com/input?i=100+USD+to+CNY"
```

## Advanced Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `site:` | `site:github.com python` | Search within site |
| `filetype:` | `filetype:pdf report` | Specific file type |
| `""` | `"machine learning"` | Exact match |
| `-` | `python -snake` | Exclude term |
| `OR` | `cat OR dog` | Either term |

## Time Filters

| Parameter | Description |
|-----------|-------------|
| `tbs=qdr:h` | Past hour |
| `tbs=qdr:d` | Past day |
| `tbs=qdr:w` | Past week |
| `tbs=qdr:m` | Past month |
| `tbs=qdr:y` | Past year |

## Privacy Engines

- **DuckDuckGo**: No tracking
- **Startpage**: Google results + privacy
- **Brave**: Independent index
- **Qwant**: EU GDPR compliant

## Bangs Shortcuts (DuckDuckGo)

| Bang | Destination |
|------|-------------|
| `!g` | Google |
| `!gh` | GitHub |
| `!so` | Stack Overflow |
| `!w` | Wikipedia |
| `!yt` | YouTube |

## WolframAlpha Queries

- Math: `integrate x^2 dx`
- Conversion: `100 USD to CNY`
- Stocks: `AAPL stock`
- Weather: `weather in Beijing`

## Documentation

- `references/advanced-search.md` - Domestic search guide
- `references/international-search.md` - International search guide
- `CHANGELOG.md` - Version history

## License

MIT

## Security & Privacy Notice

### Cookie Handling
- **Purpose**: Cookies are used ONLY to maintain search session state when access is denied (403/429 errors)
- **Storage**: Cookies are kept STRICTLY in memory during runtime - NEVER persisted to disk or config files
- **Acquisition**: Cookies are acquired on-demand from search engine homepages only when search requests fail
- **Scope**: Only session cookies from the specific search engine domain are captured
- **Lifecycle**: Cookies are cleared immediately after the search session completes
- **No Pre-configuration**: No cookies are loaded from config.json or any external file at startup
- **No API Keys**: This tool uses standard web search URLs, no authentication required

### Crawling Ethics
- **Rate Limiting**: Implement reasonable delays between requests (recommend 1-2 seconds)
- **Respect robots.txt**: Honor search engine crawling policies
- **Terms of Service**: Users are responsible for complying with search engine ToS
- **Purpose**: Designed for legitimate search aggregation, not mass data scraping

### Data Handling
- **No Personal Data**: Tool does not collect or transmit user personal information
- **Local Execution**: All operations run locally, no external data transmission
- **Session Isolation**: Cookies are session-specific and cleared after use
