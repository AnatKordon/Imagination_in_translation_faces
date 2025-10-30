import base64
from openai import OpenAI
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import requests
from config import GEN_DIR, params, N_OUT

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

# I am changing the function to work in edit image mode only
from pathlib import Path
from typing import List, Optional
import base64

def send_gpt_edit_request(
    prompt: str,
    input_image_path: Path,
    iteration: int,
    session_num: int,
    user_id: str,
    n_out= N_OUT
) -> Tuple[List[Path], List[bytes]]:
    """
    Edits an existing image using the GPT image model, guided by a text prompt.
    The result(s) are saved locally and returned as file paths.
    """
    #using rest api directly as above worked but moderation forbids distortions
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Open file with context manager so it closes even on error
    with open(str(input_image_path), "rb") as f:
        files = {"image": f}
        data = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": max(1, int(n_out)),
            "size": "1024x1024",
            # REST only extras (not supported by SDK 2.6.1 - possibly, I'm not really sure but this way is supposed to surpass moderation better)
            "quality": "auto",
            "background": "auto",
            "moderation": "low",
            "input_fidelity": "low", # this should be played with
        }

        resp = requests.post(url, headers=headers, files=files, data=data, timeout=200)
        # If not 2xx, raise so caller can handle
        resp.raise_for_status()
        result = resp.json()

    img_data = result.get("data") or []
    if not img_data:
        raise RuntimeError(f"No image data returned: {result}")

    # Generate edited images - works but always moderation forbids distortng
    # img_data = client.images.edit(
    #     model="gpt-image-1",
    #     image=open(input_image_path, "rb"),  # the image to edit
    #     prompt=prompt,  # text prompt to guide the edit
    #     n=n_out,  # number of output images
    #     size="1024x1024",
    #     input_fidelity="high",

    # )

    items = result.get("data") or []
    if not items:
        # helpful dump for debugging
        raise RuntimeError(f"No image data returned: {result}")

    local_paths: List[Path] = []
    image_bytes_list: List[bytes] = []

    for i, item in enumerate(items, start=1):
        # item is a dict from REST, not an object
        b64 = item.get("b64_json")
        if not b64:
            # print the whole item to see what's going on
            raise ValueError(f"No base64 in response item: {item}")
        img_bytes = base64.b64decode(b64)

        filename = f"{user_id}_session{session_num:02d}_iter{iteration:02d}_{i:02d}_gpt_edit.png"
        out_path = GEN_DIR / filename
        out_path.write_bytes(img_bytes)

        local_paths.append(out_path)
        image_bytes_list.append(img_bytes)

    return local_paths, image_bytes_list



