#!/usr/bin/env node
/**
 * FLUX.2-pro image generation via Azure AI Foundry
 *
 * Usage:
 *   node generate.mjs --prompt "a cat" [--width 1024] [--height 1024] [--output out.png]
 *   node generate.mjs --prompt "a cat" --seed 42 [--output out.png]
 *   node generate.mjs --help
 *
 * Environment:
 *   FLUX_ENDPOINT  - Azure AI Foundry FLUX endpoint URL
 *   FLUX_API_KEY   - Bearer token for authentication
 */

import { writeFileSync } from "fs";
import { argv, env, exit } from "process";

const sizePresets = {
  square: { width: 1024, height: 1024 },
  portrait: { width: 896, height: 1280 },
  landscape: { width: 1280, height: 768 },
  "4:3": { width: 1024, height: 768 },
  a4: { width: 896, height: 1244 },
  a3: { width: 1240, height: 1754 },
};

function parseArgs() {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--help" || argv[i] === "-h") {
      args.help = true;
    } else if (argv[i] === "--prompt" && argv[i + 1]) {
      args.prompt = argv[++i];
    } else if (argv[i] === "--width" && argv[i + 1]) {
      args.width = parseInt(argv[++i]);
    } else if (argv[i] === "--height" && argv[i + 1]) {
      args.height = parseInt(argv[++i]);
    } else if (argv[i] === "--output" && argv[i + 1]) {
      args.output = argv[++i];
    } else if (argv[i] === "--seed" && argv[i + 1]) {
      args.seed = parseInt(argv[++i]);
    } else if (argv[i] === "--size" && argv[i + 1]) {
      const preset = sizePresets[argv[++i]];
      if (preset) {
        args.width = preset.width;
        args.height = preset.height;
      }
    }
  }
  return args;
}

function printHelp() {
  console.log(`FLUX.2-pro Image Generator

Usage:
  node generate.mjs --prompt "description" [options]

Options:
  --prompt TEXT     Text prompt for image generation (required)
  --width NUM      Image width (default: 1024)
  --height NUM     Image height (default: 1024)
  --size PRESET    Size preset: square, portrait, landscape, 4:3, a4, a3
  --seed NUM        Seed for reproducible generation
  --output FILE    Output filename (default: generated.png)
  --help, -h       Show this help

Size Presets:
  square    - 1024x1024
  portrait - 896x1280
  landscape- 1280x768
  4:3      - 1024x768
  a4       - 896x1244
  a3       - 1240x1754

Environment:
  FLUX_ENDPOINT   Azure AI Foundry endpoint URL
  FLUX_API_KEY    API key for authentication

Example:
  FLUX_ENDPOINT=https://xxx.services.ai.azure.com/... \\
  FLUX_API_KEY=xxx \\
  node generate.mjs --prompt "a red fox" --size a4 --output fox.png`);
}

async function main() {
  const args = parseArgs();

  if (args.help) {
    printHelp();
    exit(0);
  }

  const endpoint = env.FLUX_ENDPOINT;
  const apiKey = env.FLUX_API_KEY;

  if (!endpoint || !apiKey) {
    console.error("Error: Set FLUX_ENDPOINT and FLUX_API_KEY environment variables");
    exit(1);
  }
  if (!args.prompt) {
    console.error("Error: --prompt is required");
    exit(1);
  }

  const body = {
    prompt: args.prompt,
    width: args.width || 1024,
    height: args.height || 1024,
    n: 1,
    model: "FLUX.2-pro",
  };
  if (args.seed !== undefined) body.seed = args.seed;

  console.error(`Generating: "${args.prompt}" (${body.width}x${body.height})...`);

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });

  if (!res.ok) {
    const text = await res.text();
    console.error(`API error ${res.status}: ${text}`);
    exit(1);
  }

  const data = await res.json();
  const b64 = data?.data?.[0]?.b64_json;
  if (!b64) {
    console.error("No image data in response:", JSON.stringify(data).slice(0, 500));
    exit(1);
  }

  const outPath = args.output || "generated.png";
  writeFileSync(outPath, Buffer.from(b64, "base64"));
  console.log(`Saved: ${outPath} (${Buffer.from(b64, "base64").length} bytes)`);
}

main().catch((e) => {
  console.error(e);
  exit(1);
});