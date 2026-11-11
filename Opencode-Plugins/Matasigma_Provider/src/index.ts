import { readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"
import { Model, Plugin, Provider } from "@opencode/plugin"

// Registers an OpenAI-compatible provider and keeps its model inventory in sync
// with the gateway's /v1/models endpoint, so new models show up and removed ones
// disappear without editing opencode.json by hand.

type Options = {
  providerID: string
  name: string
  baseURL: string
  apiKeyEnv: string
  keyFile: string
  refreshInterval: number
  modes: string[]
  package: string
  idPrefix: string | false
}

type UpstreamModel = {
  id: string
  mode?: string
  max_input_tokens?: number
  max_output_tokens?: number
}

const DEFAULTS = {
  providerID: "matasigma",
  name: "Matasigma",
  baseURL: "https://models.matasigma.com/v1",
  refreshInterval: 10 * 60 * 1000,
  modes: ["chat", "responses"],
  package: "@opencode/ai/providers/openai-compatible",
}

function resolveOptions(raw: unknown): Options {
  const input = (raw ?? {}) as Partial<Options>
  const providerID = input.providerID ?? DEFAULTS.providerID
  const name = input.name ?? (input.providerID ? input.providerID : DEFAULTS.name)

  return {
    providerID,
    name,
    baseURL: input.baseURL ?? DEFAULTS.baseURL,
    apiKeyEnv: input.apiKeyEnv ?? `${providerID.toUpperCase().replace(/[^A-Z0-9]+/g, "_")}_API_KEY`,
    keyFile: input.keyFile ?? join(homedir(), ".config", "opencode", `${providerID}-api-key`),
    refreshInterval: input.refreshInterval ?? DEFAULTS.refreshInterval,
    modes: input.modes ?? DEFAULTS.modes,
    package: input.package ?? DEFAULTS.package,
    idPrefix: input.idPrefix === undefined ? name : input.idPrefix,
  }
}

function readKey(options: Options): string | undefined {
  const fromEnv = process.env[options.apiKeyEnv]
  if (fromEnv) return fromEnv
  try {
    return readFileSync(options.keyFile, "utf8").trim() || undefined
  } catch {
    return undefined
  }
}

function cleanID(id: string, options: Options): string {
  if (options.idPrefix === false) return id
  const prefix = `${options.idPrefix}/`
  return id.toLowerCase().startsWith(prefix.toLowerCase()) ? id.slice(prefix.length) : id
}

export default Plugin.define({
  id: "model-sync.provider",
  async setup(ctx) {
    const options = resolveOptions(ctx.options)
    const providerID = Provider.ID.make(options.providerID)
    const source: { models: Model.Info[] } = { models: [] }

    async function fetchModels(): Promise<Model.Info[]> {
      const key = readKey(options)
      if (!key) {
        throw new Error(`${options.apiKeyEnv} is not set and ${options.keyFile} is unreadable`)
      }

      const response = await fetch(`${options.baseURL}/models`, {
        headers: { authorization: `Bearer ${key}` },
      })
      if (!response.ok) throw new Error(`GET ${options.baseURL}/models returned ${response.status}`)

      const body = (await response.json()) as { data?: UpstreamModel[] }
      const models: Model.Info[] = []

      for (const upstream of body.data ?? []) {
        // Only models that can serve a chat session. Embedding, image and
        // evaluation models are skipped.
        const mode = upstream.mode ?? "chat"
        if (!options.modes.includes(mode)) continue

        const id = cleanID(upstream.id, options)
        models.push({
          ...Model.Info.default(providerID, Model.ID.make(id)),
          // The gateway expects the raw upstream ID.
          modelID: upstream.id,
          name: id,
          limit: {
            context: upstream.max_input_tokens ?? 200_000,
            output: upstream.max_output_tokens ?? 32_000,
          },
        })
      }

      return models
    }

    try {
      source.models = await fetchModels()
    } catch (error) {
      console.error(`${options.providerID}: initial model sync failed`, error)
    }

    await ctx.provider.transform((editor) => {
      editor.add({
        info: {
          ...Provider.Info.empty(providerID),
          name: options.name,
          activation: "enabled",
          package: options.package,
          settings: {
            baseURL: options.baseURL,
            apiKey: readKey(options),
          },
        },
        models: source.models,
      })
    })

    const refresh = async () => {
      source.models = await fetchModels()
      await ctx.provider.reload()
    }

    const timer = setInterval(() => {
      void refresh().catch((error) => console.error(`${options.providerID}: model sync failed`, error))
    }, options.refreshInterval)

    return () => clearInterval(timer)
  },
})