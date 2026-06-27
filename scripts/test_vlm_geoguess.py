"""VLM 'GeoGuessr': can Qwen3-VL infer the district + read names from dashcam frames?

The agents' #1 untested lever. For each clip we show single frames to Qwen3-VL and ask
for the most likely neighbourhood/district + any readable street/shop/landmark names.
If it reliably names the right district (or geocodable streets), it's an absolute,
shape-independent coarse prior of a different class than VPR.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SP = ("C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
      "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
      "5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad")
CLIPS = [
    ("Ulm 4K (GT: Olgastrasse/centre)", "Ulm, Germany",
     "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"),
    ("London (GT: Bloomsbury)", "London, UK", "data/london_T4wTL3LpLqU/input.mp4"),
    ("Erbach a.d. Donau", "Erbach an der Donau, Germany", f"{SP}/erbach.mp4"),
]
PROMPT = ("This is one dashcam frame from a car driving in {city}. Read any visible "
          "street-name plates, shop/business names, bus stops or landmarks, and infer "
          "the most likely neighbourhood/district. Reply ONLY as two lines:\n"
          "DISTRICT: <name or unknown>\nNAMES: <comma-separated readable names, or none>")


def main():
    print("loading Qwen3-VL-4B...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL)

    def ask(pil, city):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT.format(city=city)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[pil], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=110, do_sample=False)
        return proc.batch_decode(out[:, inp.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0].strip().replace("\n", " | ")

    for name, city, path in CLIPS:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        print(f"\n=== {name} ===", flush=True)
        for t in np.linspace(20, max(40, dur - 20), 6):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            print(f"  t={t:>4.0f}s  {ask(pil, city)}", flush=True)
        cap.release()


if __name__ == "__main__":
    main()
