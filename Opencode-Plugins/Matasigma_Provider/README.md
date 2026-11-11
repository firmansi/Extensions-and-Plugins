# opencode-model-sync

OpenCode plugin that registers an OpenAI-compatible provider and keeps its model
list in sync with the gateway's `/v1/models` endpoint.

OpenCode only auto-discovers models for Ollama, LM Studio, vLLM and Azure
deployments. A custom `providers` entry in `opencode.json` is static, so new
models at the gateway never show up and removed ones stay listed. This plugin
polls `/v1/models` on a timer and reloads the provider registry, so the catalog
stays current without editing config or restarting OpenCode.

Defaults target [Matasigma](https://models.matasigma.com), but every part of the
provider definition is configurable, so it works with any OpenAI-compatible
gateway, proxy or self-hosted runtime.

## Install

Repository layout:

```text
Opencode-Plugins/Matasigma_Provider/
├── src/index.ts
├── package.json
├── README.md
└── LICENSE
```

### From git (recommended)

This plugin lives in the `Extensions-and-Plugins` repo under
`Opencode-Plugins/Matasigma_Provider`:

```sh
opencode plugin add 'github:firmansi/Extensions-and-Plugins::path:Opencode-Plugins/Matasigma_Provider'
```

`opencode plugin add` writes the entry to your global `opencode.json(c)`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": ["opencode-model-sync"]
}
```

### As a single file

Copy `src/index.ts` into your plugin directory and rename it, for example
`~/.config/opencode/plugins/model-sync.ts` (global) or
`.opencode/plugins/model-sync.ts` (per project). Files in those directories load
automatically.

```sh
mkdir -p ~/.config/opencode/plugins
curl -fsSL -o ~/.config/opencode/plugins/matasigma-models.ts \
  https://raw.githubusercontent.com/firmansi/Extensions-and-Plugins/main/Opencode-Plugins/Matasigma_Provider/src/index.ts
```

## Credentials

The plugin needs the API key to read `/v1/models`. It checks, in order:

1. the environment variable set by `apiKeyEnv` (default `MATASIGMA_API_KEY`)
2. the key file from `keyFile` (default `~/.config/opencode/<providerID>-api-key`)

```sh
printf '%s\n' 'sk-...' > ~/.config/opencode/matasigma-api-key
chmod 600 ~/.config/opencode/matasigma-api-key
```

For the background service, environment variables must be set on the server:

```sh
opencode service set env MATASIGMA_API_KEY sk-...
```

Setting service env restarts the background service.

## Configure

Pass plugin options with the object form:

```jsonc
{
  "plugins": [
    {
      "package": "opencode-model-sync",
      "options": {
        "providerID": "acme",
        "name": "Acme",
        "baseURL": "https://llm.acme.example/v1",
        "apiKeyEnv": "ACME_API_KEY",
        "refreshInterval": 600000
      }
    }
  ]
}
```

| Option | Default | Purpose |
| --- | --- | --- |
| `providerID` | `matasigma` | Provider ID, used as the `provider/model` prefix. |
| `name` | `Matasigma` | Display name. |
| `baseURL` | `https://models.matasigma.com/v1` | OpenAI-compatible base URL, including `/v1`. |
| `apiKeyEnv` | `<PROVIDERID>_API_KEY` | Environment variable holding the key. |
| `keyFile` | `~/.config/opencode/<providerID>-api-key` | Fallback key file, read when the env var is unset. |
| `refreshInterval` | `600000` | Milliseconds between `/v1/models` refreshes. |
| `modes` | `["chat", "responses"]` | Upstream `mode` values to keep. Models without `mode` count as `chat`. |
| `package` | `@opencode/ai/providers/openai-compatible` | Runtime package for the provider. |
| `idPrefix` | provider `name` | Prefix stripped from upstream IDs (`Matasigma/qwen3.8-flash` → `qwen3.8-flash`). Set `false` to keep IDs verbatim. |

## How it behaves

- Models are keyed by the cleaned ID; the raw upstream ID is sent as `modelID`,
  so `matasigma/qwen3.8-flash` calls `Matasigma/qwen3.8-flash` upstream.
- `limit.context` and `limit.output` come from `max_input_tokens` and
  `max_output_tokens` when the gateway reports them, otherwise 200k / 32k.
- Embedding, image generation and evaluation models are skipped: OpenCode only
  offers chat models for sessions.
- A failed refresh keeps the previous inventory; it never empties the list.
- A failed first sync still registers the provider, so the next refresh can fill
  it in.

## Verify

```sh
opencode api get /api/model | grep -o '"providerID":"matasigma"' | wc -l
opencode models | grep '^matasigma/'
opencode run --model matasigma/qwen3.8-flash "Reply with exactly: ok"
```

In the TUI, `/models` lists the synced models.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Provider missing from `/api/provider` | Plugin failed to load; check `~/.local/share/opencode/log/opencode.log` for `loading plugin`. |
| Provider present, no models | `/v1/models` returned 401 or the key was missing at load time. Fix the key, then touch the plugin file or restart the service. |
| Models appear but requests fail | Confirm the `modelID` the gateway expects, and that it accepts `POST /v1/chat/completions`. |
| New models do not show up | Lower `refreshInterval`, or `touch` the plugin file to force a reload. |

## License

MIT