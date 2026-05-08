import re
import requests
import base64
import os
from typing import Optional, Dict, Any, Tuple, List
from pydantic import BaseModel, Field, field_validator


class Pipe:
    RATIO_MAP: dict = {
        "1:1": (1024, 1024),
        "4:3": (1024, 768),
        "3:4": (768, 1024),
        "16:9": (1024, 576),
        "9:16": (576, 1024),
        "21:9": (1024, 448),
        "9:21": (448, 1024),
        "3:2": (1024, 683),
        "2:3": (683, 1024),
        "16:10": (1024, 640),
        "10:16": (640, 1024),
    }

    ALLOWED_SIZES: List[int] = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

    class Valves(BaseModel):
        FLUX_ENDPOINT: str = Field(
            default="",
            description="Azure AI Foundry endpoint URL untuk Flux",
        )
        FLUX_API_KEY: str = Field(
            default="",
            description="API Key untuk Azure AI Foundry",
        )
        RATIO: str = Field(
            default="1:1",
            description="Ratio gambar: 1:1, 4:3, 16:9, 9:16, 21:9, 3:2, 2:3, dll. Otomatis menyesuaikan dimensi.",
        )
        MODEL: str = Field(
            default="FLUX.2-pro",
            description="Model Flux yang digunakan",
        )

        @field_validator("RATIO")
        @classmethod
        def validate_ratio(cls, v: str) -> str:
            v = v.strip().lower()
            if v not in Pipe.RATIO_MAP and not re.match(r"^\d+:\d+$", v):
                return "1:1"
            return v

    def __init__(self):
        self.valves = self.Valves()
        self.endpoint = os.getenv("FLUX_ENDPOINT", "")
        self.api_key = os.getenv("FLUX_API_KEY", "")

    def _calculate_dimensions(self, ratio: str) -> Tuple[int, int]:
        """
        Hitung dimensi berdasarkan ratio.
        """
        ratio = ratio.strip().lower()

        if ratio in self.RATIO_MAP:
            return self.RATIO_MAP[ratio]

        match = re.match(r"(\d+):(\d+)", ratio)
        if match:
            w_r = int(match.group(1))
            h_r = int(match.group(2))
            base = 1024

            if w_r >= h_r:
                width = base
                height = int(base * h_r / w_r)
                height = min(
                    [s for s in self.ALLOWED_SIZES if s >= height], default=768
                )
            else:
                height = base
                width = int(base * w_r / h_r)
                width = min([s for s in self.ALLOWED_SIZES if s >= width], default=768)

            return width, height

        return 1024, 1024

    def _parse_ratio_from_prompt(self, prompt: str) -> Tuple[int, int]:
        """
        Auto-detect ratio dari prompt user.
        Format: 16:9, 4:3, 1:1, 9:16, 21:9, 3:2, 2:3, dll.
        """
        ratio_pattern = r"\b(\d+)\s*:\s*(\d+)\b"
        match = re.search(ratio_pattern, prompt, re.IGNORECASE)

        if not match:
            return self._calculate_dimensions(self.valves.RATIO)

        detected_ratio = f"{match.group(1)}:{match.group(2)}"
        return self._calculate_dimensions(detected_ratio)

    def pipe(
        self, body: Dict[str, Any], __user__: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main pipe function untuk image generation.
        """
        endpoint = (
            self.valves.FLUX_ENDPOINT if self.valves.FLUX_ENDPOINT else self.endpoint
        )
        api_key = self.valves.FLUX_API_KEY if self.valves.FLUX_API_KEY else self.api_key
        model = self.valves.MODEL if self.valves.MODEL else "FLUX.2-pro"

        if not endpoint or not api_key:
            return "Error: FLUX_ENDPOINT atau FLUX_API_KEY belum dikonfigurasi."

        messages = body.get("messages", [])
        if not messages:
            return "Tidak ada prompt provided."

        user_prompt = messages[-1].get("content", "")
        if not user_prompt:
            return "Prompt kosong. Mohon masukkan deskripsi gambar."

        width, height = self._parse_ratio_from_prompt(user_prompt)

        if width not in self.ALLOWED_SIZES:
            return f"Error: Width {width} tidak valid."
        if height not in self.ALLOWED_SIZES:
            return f"Error: Height {height} tidak valid."

        try:
            payload = {
                "prompt": user_prompt,
                "width": width,
                "height": height,
                "num_images": 1,
                "model": model,
                "output_format": "png",
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "api-key": api_key,
            }

            response = requests.post(
                f"{endpoint}&api-version=preview",
                json=payload,
                headers=headers,
                timeout=180,
            )
            response.raise_for_status()

            data = response.json()
            b64_image = data.get("data", [{}])[0].get("b64_json")

            if not b64_image:
                return f"Error: Tidak ada data gambar dari Azure API. Response: {data}"

            return f"![Generated Image ({width}x{height})](data:image/png;base64,{b64_image})"

        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Unexpected Error: {str(e)}"


# Untuk integrasi Open WebUI, class ini akan dipanggil secara otomatis
