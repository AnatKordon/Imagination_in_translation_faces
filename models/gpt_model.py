import base64
from openai import OpenAI
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from config import GEN_DIR, params, N_OUT

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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
):
    """
    Edits an existing image using the GPT image model, guided by a text prompt.
    The result(s) are saved locally and returned as file paths.
    """

    # Generate edited images
    img_data = client.images.edits(
        model="gpt-image-1",
        image=open(input_image_path, "rb"),  # the image to edit
        prompt=prompt,
        n=n_out,  # number of output images
        size="1024x1024"
    )

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    revised_prompts: List[Optional[str]] = []
    print(f"prompt revision: {revised_prompts}")

    image_bytes = [] # in case there are multiple images this should be better alligned
    
    for i, item in enumerate(img_data.data, start=1):
        img_b64 = getattr(item, "b64_json", None)
        if not img_b64:
            raise ValueError("No base64 image found in response; set response_format='b64_json'.")

        # Decode and save
        bytes = base64.b64decode(img_b64)
        filename = f"{user_id}_session{session_num:02d}_iter{iteration:02d}_{i:02d}_gpt_edit.png"
        path = GEN_DIR / filename
        path.write_bytes(bytes)
        local_paths.append(path)
        image_bytes.append(bytes)
        # # Log revised prompt if returned (not common for edits)
        # rp = getattr(item, "revised_prompt", None)
        # print(f"revised prompt is: {rp}")
        # revised_prompts.append(rp)
       
    return local_paths, image_bytes # revised_prompts



