"""
title: Qwen Video Generator
description: Generate videos with Qwen API hosted on Alibaba Cloud DashScope
author: dfirmans
version: 1.0 (Based on Image Generator)
"""

from typing import Literal
import os
import json
import requests
import time
from pydantic import BaseModel, Field

# Video size options based on Qwen API documentation
VideoSizeType = Literal[
    # 720p options
    "1280*720",
    "720*1280",
    "960*960",
    "1088*832",
    "832*1088",
    # 1080p options
    "1920*1080",
    "1080*1920",
    "1440*1440",
    "1632*1248",
    "1248*1632",
]

# Qwen video generation models
VideoModelType = Literal["wan2.5-t2v-preview",]


class Tools:
    class Valves(BaseModel):
        DASHSCOPE_API_KEY: str = Field(
            default="",
            description="Your Alibaba Cloud DashScope API key",
        )
        DASHSCOPE_BASE_URL: str = Field(
            default="https://dashscope.aliyuncs.com/api/v1",
            description="DashScope API base URL (China Beijing region)",
        )

    def __init__(self):
        self.valves = self.Valves(
            DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY", ""),
            DASHSCOPE_BASE_URL=os.getenv(
                "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
            ),
        )

    async def generate_video_qwen(
        self,
        video_prompt: str,
        video_size: VideoSizeType = "1280*720",
        duration: int = 10,
        enable_audio: bool = True,
        model: VideoModelType = "wan2.5-t2v-preview",
        __event_emitter__=None,
    ) -> str:
        """
        Generate a video using Qwen API

        Whenever a description of a video is given, use this tool to create the video from a prompt.
        You do not need to ask for permission to generate, just do it!
        DO NOT write out the prompt before OR after generating the videos. The prompt should ONLY ever be written out ONCE, in the `"video_prompt"` field of the request.

        The video_prompt sent to this tool must abide by the following policies:
        1. The prompt should be in English. If the video description was not in English, then translate it.
        2. Always mention the scene type and camera angle at the beginning of the prompt. Include cinematography details like lighting, composition, and movement.
        3. The prompt must intricately describe every part of the video in concrete, objective detail. Include subject actions, environments, and any audio/dialogue elements.
        4. The prompt should be a paragraph of text that is extremely descriptive and detailed. It should be more than 3 sentences long.
        5. If the user requested modifications to a previous video, the prompt should not simply be longer, but rather it should be refactored to integrate the suggestions.

        The video_size should be selected by picking the most suitable one for the video.
        Available sizes based on Qwen API:

        720p options:
        - "1280*720" (16:9) - widescreen landscape
        - "720*1280" (9:16) - vertical mobile format
        - "960*960" (1:1) - square format
        - "1088*832" (4:3) - standard landscape
        - "832*1088" (3:4) - standard vertical

        1080p options:
        - "1920*1080" (16:9) - full HD widescreen
        - "1080*1920" (9:16) - full HD vertical
        - "1440*1440" (1:1) - large square format
        - "1632*1248" (4:3) - high resolution landscape
        - "1248*1632" (3:4) - high resolution vertical

        Duration: Video length in seconds (typically 5-10 seconds)

        :param video_prompt: Text prompt for video generation.
        :param video_size: Size of the generated video.
        :param duration: Duration of the video in seconds.
        :param enable_audio: Whether to generate audio for the video.
        :param model: Qwen model to use for generation.
        """

        try:
            # Emit status events for UI feedback
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Generating video with Qwen...",
                        "done": False,
                    },
                }
            )

            api_key = self.valves.DASHSCOPE_API_KEY
            base_url = self.valves.DASHSCOPE_BASE_URL

            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY is not set")

            video_url = generate_video_with_qwen_api(
                api_key,
                base_url,
                video_prompt,
                video_size,
                duration,
                enable_audio,
                model,
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Generated video successfully:",
                        "done": True,
                    },
                }
            )

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": f"**Video Generated Successfully**  \n**Size:** `{video_size}`  |  **Duration:** `{duration}s`  |  **Audio:** `{'Yes' if enable_audio else 'No'}`  |  **Model:** `{model}`  \n\n[View Video]({video_url})"
                    },
                }
            )

            # Emit completed status
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Video generation completed",
                        "done": True,
                        "status": "completed",
                    },
                }
            )

            return f"""
The video generation completed successfully!

Note that the generated video ALREADY HAS been automatically sent and displayed to the user by the tool.
You don't need to do anything to show the video to the user.

When you answer now - simply tell the user the video was successfully generated.
Answer with text only, NO VIDEO LINK, NO SIZE, NO DURATION, NO MODEL, NO VIDEO PROMPT.
Just tell the user that the video was successfully generated.

At the end of your answer you might ask the user if they want anything changed to the video.
Should they later come back and ask for changes - just generate a new video with potentially modified parameters based on their feedback.

Keep your answer short and concise.
"""

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occurred: {e}", "done": True},
                }
            )

            return f"Tell the user: {e}"


def generate_video_with_qwen_api(
    api_key: str,
    base_url: str,
    prompt: str,
    size: VideoSizeType,
    duration: int,
    enable_audio: bool,
    model: VideoModelType,
) -> str:
    """
    Generate video using Qwen API

    :param api_key: DashScope API key
    :param base_url: DashScope API base URL
    :param prompt: Text prompt for video generation
    :param size: Video size
    :param duration: Duration in seconds
    :param enable_audio: Enable audio generation
    :param model: Model to use for generation
    :return: URL of the generated video
    """

    # Step 1: Create task and get task_id
    # Prepare the request payload for task creation
    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "size": size,
        },
        "parameters": {
            "duration": duration,
            "audio": enable_audio,
            "prompt_extend": True,
        },
    }

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-DashScope-Async": "enable",  # Enable async for task creation
    }

    try:
        # Step 1: Create the video generation task
        create_url = f"{base_url}/services/aigc/video-generation/video-synthesis"

        response = requests.post(
            create_url,
            headers=headers,
            json=payload,
            timeout=60,  # 60 seconds for task creation
        )

        if response.status_code != 200:
            error_msg = f"Task creation failed - HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
                if "code" in error_data:
                    error_msg += f" (Code: {error_data['code']})"
            except:
                error_msg += f": {response.text}"

            raise Exception(error_msg)

        # Get task_id from response
        result = response.json()
        if "output" not in result or "task_id" not in result["output"]:
            raise ValueError(
                f"No task_id found in response: {json.dumps(result, indent=2)}"
            )

        task_id = result["output"]["task_id"]

        # Step 2: Poll for task completion
        query_headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        query_url = f"{base_url}/tasks/{task_id}"
        max_attempts = 180  # 180 attempts for video (longer than image)
        poll_interval = 3  # 3 seconds between polls

        # Total maximum time: (180 × 3) = 540 seconds = 9 minutes
        for attempt in range(max_attempts):
            time.sleep(poll_interval)  # Wait 3 seconds between polls

            query_response = requests.get(
                query_url,
                headers=query_headers,
                timeout=60,  # 60 seconds per polling request
            )

            if query_response.status_code != 200:
                continue  # Continue polling on temporary errors

            query_result = query_response.json()

            if "output" not in query_result:
                continue

            task_status = query_result["output"].get("task_status", "")

            if task_status == "SUCCEEDED":
                # Extract video URL from successful result
                if "video_url" in query_result["output"]:
                    return query_result["output"]["video_url"]

                raise ValueError(
                    f"No video URL found in successful response: {json.dumps(query_result, indent=2)}"
                )

            elif task_status == "FAILED":
                error_msg = "Task failed"
                if "code" in query_result["output"]:
                    error_msg += f" (Code: {query_result['output']['code']})"
                if "message" in query_result["output"]:
                    error_msg += f": {query_result['output']['message']}"

                raise Exception(error_msg)

            elif task_status in ["PENDING", "RUNNING"]:
                continue  # Keep polling

            else:
                # Unknown status, continue polling
                continue

        # If we get here, the task didn't complete within the timeout
        raise Exception(
            "Task timeout - Video generation took too long (>9 minutes). The model inference might be experiencing high load."
        )

    except requests.exceptions.Timeout:
        raise Exception("Request timeout - Network request took too long")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request error: {str(e)}")
    except Exception as e:
        if "Qwen API error" in str(e):
            raise e
        raise Exception(f"Qwen API error: {str(e)}")
