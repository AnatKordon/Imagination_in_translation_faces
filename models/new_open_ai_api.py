import requests, os
from dotenv import load_dotenv
import base64


load_dotenv()

#this method indeed works!
url = "https://api.openai.com/v1/images/edits"   # REST path used by the Playground
headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
files = {"image": open("/mnt/hdd/anatkorol/Imagination_in_translation_faces/GT_images/face001.jpeg", "rb")}
data = {
    "model": "gpt-image-1",
    "prompt": "enlarge nose to be huge and make the right eye melt down", #"make the right eye melt down"
    "n": 1,
    "size": "1024x1024",
    # Playground-only / newer REST params (SDK v2.6.1 won't accept these):
    "quality": "auto",
    "background": "auto",
    "moderation": "low",
    "input_fidelity": "low",
}
print("Method:", "POST")
print("Headers:", headers)
print("URL:", url)
try:
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=200)
    resp.raise_for_status()  # throws if not 200 OK
    result = resp.json()

# check if there's at least one image
    if "data" in result and result["data"]:
        b64 = result["data"][0]["b64_json"]
        img_bytes = base64.b64decode(b64)
        out_path = "gpt_edit_output.png"
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"✅ Image saved successfully: {out_path}")
    else:
        print("⚠️ No image data returned:", result)

except requests.exceptions.RequestException as e:
    print("❌ Network or API error:", e)
except Exception as e:
    print("❌ Unexpected error:", e)