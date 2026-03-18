import json
import os
import re
import struct
import base64
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import google.auth
import google.auth.transport.requests
import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Rate Limiting ──────────────────────────────────────
# 10 analyze requests per minute per IP, 60 per hour
RATE_LIMIT_PER_MIN = 10
RATE_LIMIT_PER_HOUR = 60

_rate_lock = threading.Lock()
_rate_store = defaultdict(list)  # ip -> list of timestamps


def _cleanup_old(ip, now):
    """Remove timestamps older than 1 hour."""
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 3600]


def _check_rate_limit():
    """Returns (allowed, retry_after_seconds). Only applies to /analyze* routes."""
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if ip:
        ip = ip.split(",")[0].strip()
    now = time.time()
    with _rate_lock:
        _cleanup_old(ip, now)
        timestamps = _rate_store[ip]
        last_min = [t for t in timestamps if now - t < 60]
        if len(last_min) >= RATE_LIMIT_PER_MIN:
            retry = int(60 - (now - last_min[0])) + 1
            return False, retry
        if len(timestamps) >= RATE_LIMIT_PER_HOUR:
            retry = int(3600 - (now - timestamps[0])) + 1
            return False, retry
        timestamps.append(now)
        return True, 0


@app.before_request
def rate_limit_check():
    if request.path.startswith("/analyze") and request.method == "POST":
        allowed, retry_after = _check_rate_limit()
        if not allowed:
            return jsonify({
                "error": f"Rate limit exceeded. Try again in {retry_after}s.",
                "retry_after": retry_after
            }), 429

PROJECT = "mineral-concord-394714"
GEMMA_ENDPOINT = (
    "https://mg-endpoint-b193aa5e-ce3c-4b38-8fca-ab36e09e4951.europe-west4-668228315581.prediction.vertexai.goog"
    "/v1/projects/mineral-concord-394714/locations/europe-west4"
    "/endpoints/mg-endpoint-b193aa5e-ce3c-4b38-8fca-ab36e09e4951:rawPredict"
)
GEMINI_ENDPOINT = (
    "https://aiplatform.googleapis.com/v1/projects/mineral-concord-394714"
    "/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"
)
GEMINI_IMAGE_ENDPOINT = (
    "https://aiplatform.googleapis.com/v1/projects/mineral-concord-394714"
    "/locations/us-central1/publishers/google/models/gemini-3-pro-image-preview:generateContent"
)
YOLO_ENDPOINT = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/mineral-concord-394714"
    "/locations/us-central1/endpoints/8696944011417485312:predict"
)
TUNED_GEMMA_ENDPOINT = "http://34.147.36.82:7860/predict"
TUNED_ENDPOINT = (
    "https://europe-west4-aiplatform.googleapis.com/v1/projects/668228315581"
    "/locations/europe-west4/endpoints/7643600327135985664:generateContent"
)
GCS_IMAGE_BASE = "gs://dpd-street-detection/images"
GCS_PUBLIC_BASE = "https://storage.googleapis.com/dpd-street-detection/images"
GCS_HOURS_IMAGE_BASE = "gs://dpd-street-detection/shop-hours"
GCS_PUBLIC_BASE_HOURS = "https://storage.googleapis.com/dpd-street-detection/shop-hours"
GCS_PARKING_IMAGE_BASE = "gs://dpd-street-detection/parking"
GCS_PUBLIC_BASE_PARKING = "https://storage.googleapis.com/dpd-street-detection/parking"
GCS_ADDRESSES_IMAGE_BASE = "gs://dpd-street-detection/addresses"
GCS_PUBLIC_BASE_ADDRESSES = "https://storage.googleapis.com/dpd-street-detection/addresses"
GCS_TRAFFIC_IMAGE_BASE = "gs://dpd-street-detection/traffic"
GCS_PUBLIC_BASE_TRAFFIC = "https://storage.googleapis.com/dpd-street-detection/traffic"
GCS_DELIVERY_IMAGE_BASE = "gs://dpd-street-detection/delivery"
GCS_PUBLIC_BASE_DELIVERY = "https://storage.googleapis.com/dpd-street-detection/delivery"

HOURS_PROMPT = (
    "You are a shop detection AI analyzing street-level images of European storefronts. "
    "Identify every shop, restaurant, cafe, or business visible in the image.\n\n"
    "For each business found, provide:\n"
    "- shop_name: the name visible on the signage\n"
    "- opening_hours: any opening hours visible on the door, wall, or signage "
    "(e.g. \"Mon-Fri 9:00-18:00\"). Use \"Not visible\" if no hours can be read.\n"
    "- status: whether the shop appears to be \"open\", \"closed\", or \"unknown\" "
    "based on visual cues (lights on, door open, OPEN/CLOSED sign, customers inside, "
    "shutters down, etc.)\n\n"
    "Return ONLY a JSON object (no markdown fences) with a \"shops\" array. "
    "Each element must include shop_name, opening_hours, and status."
)

RDD_PROMPT = (
    "You are a road condition assessment AI. Analyse this dashcam image and classify "
    "road damage using the RDD2022 defect codes:\n"
    "- D00: Longitudinal crack\n"
    "- D10: Transverse crack\n"
    "- D20: Alligator crack\n"
    "- D40: Pothole / rutting\n\n"
    "For each defect found, provide:\n"
    "- defect_code (D00/D10/D20/D40)\n"
    "- count of instances of this defect type\n"
    "- severity (low/medium/high)\n"
    "- bounding_boxes: an array of bounding box objects. Each bounding box has "
    "{xmin, ymin, xmax, ymax} where values are PERCENTAGES of image dimensions "
    "(0.0 to 100.0). For example, a box in the center-right of the image might be "
    "{\"xmin\": 55.0, \"ymin\": 40.0, \"xmax\": 75.0, \"ymax\": 60.0}. "
    "Values must be between 0 and 100.\n\n"
    "Only include defect types that are actually present (count > 0). "
    "Return ONLY a JSON object (no markdown fences) with a \"defects\" array."
)

PARKING_PROMPT = (
    "You are a parking and loading zone detection AI analyzing street-level images "
    "of European cities. Identify all parking-related features visible in the image.\n\n"
    "For each feature found, provide:\n"
    "- type: the type of feature (e.g. \"loading_zone\", \"no_parking_sign\", "
    "\"parking_meter\", \"restricted_parking\", \"available_spot\", \"disabled_parking\", "
    "\"time_limited_parking\")\n"
    "- description: brief description of what you see\n"
    "- suitability: whether this spot/zone is suitable for a DPD delivery van "
    "(\"yes\", \"no\", \"maybe\")\n\n"
    "Return ONLY a JSON object (no markdown fences) with a \"features\" array. "
    "Each element must include type, description, and suitability."
)

ADDRESSES_PROMPT = (
    "You are an address detection AI analyzing street-level images of European "
    "residential streets. Identify all house numbers, building names, and address "
    "markers visible in the image.\n\n"
    "For each address found, provide:\n"
    "- number: the house number or building identifier visible (e.g. \"42\", \"12A\", "
    "\"Building Name\")\n"
    "- type: the type of marker (e.g. \"house_number\", \"building_name\", "
    "\"street_sign\", \"address_plaque\")\n"
    "- visibility: how clearly readable the number/name is (\"clear\", \"partial\", "
    "\"obscured\")\n\n"
    "Return ONLY a JSON object (no markdown fences) with an \"addresses\" array. "
    "Each element must include number, type, and visibility."
)

TRAFFIC_PROMPT = (
    "You are a construction site detection AI analyzing street-level images of "
    "European cities. Identify all construction-related obstacles visible in the "
    "image that could affect a delivery van passing through.\n\n"
    "For each obstacle found, provide:\n"
    "- type: the type of obstacle (e.g. \"scaffolding\", \"road_barrier\", "
    "\"crane\", \"excavation\", \"construction_fencing\", \"lane_narrowing\", "
    "\"temporary_traffic_light\", \"heavy_machinery\", \"road_closure\")\n"
    "- description: brief description of the obstacle and its extent\n"
    "- impact: impact on a standard DPD delivery van (\"high\", \"medium\", \"low\")\n\n"
    "Also provide an \"assessment\" object with:\n"
    "- passable: whether a delivery van can pass (\"yes\", \"no\", \"with_caution\")\n"
    "- recommended_action: what the driver should do\n\n"
    "Return ONLY a JSON object with an \"obstacles\" array and an \"assessment\" object."
)

DELIVERY_PROMPT_IMAGE = (
    "You are an AR delivery assistant AI. A DPD delivery driver needs to find "
    "the exact delivery location in this street-level image.\n\n"
    "The customer left the following delivery note:\n"
    "\"{delivery_note}\"\n\n"
    "IMPORTANT: You MUST edit and return the input image with clear visual "
    "annotations drawn on top of it. Do not return text only — return the "
    "modified image.\n\n"
    "Annotations to draw on the image:\n"
    "- A large, bold arrow (bright green or red) pointing to the exact "
    "delivery location (the door, gate, or entrance)\n"
    "- A circle or rectangle highlighting the delivery target\n"
    "- A text label \"DELIVER HERE\" placed near the target\n"
    "- If the note mentions an alternative (e.g. neighbor, porch), draw a "
    "second smaller arrow labeled \"ALTERNATIVE\" pointing there\n\n"
    "Use thick lines and bright neon colors (green, red, yellow) so "
    "annotations are impossible to miss. Return the annotated image."
)


def get_access_token():
    """Get a fresh access token using default credentials."""
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def download_image(image_id):
    """Download image from GCS and return raw bytes."""
    url = f"{GCS_PUBLIC_BASE}/{image_id}.jpg"
    resp = http_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def get_jpeg_dimensions(data):
    """Extract width and height from JPEG data."""
    i = 2
    while i < len(data) - 1:
        if data[i] == 0xFF:
            marker = data[i + 1]
            if marker in (0xC0, 0xC2):
                h = struct.unpack(">H", data[i + 5 : i + 7])[0]
                w = struct.unpack(">H", data[i + 7 : i + 9])[0]
                return w, h
            elif marker == 0xD9:
                break
            else:
                length = struct.unpack(">H", data[i + 2 : i + 4])[0]
                i += 2 + length
        else:
            i += 1
    return None, None


def parse_model_json(raw_text):
    """Extract JSON from model response, stripping markdown fences if present."""
    text = raw_text.strip()
    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find any JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Try to repair malformed/truncated JSON by rebuilding closing sequence
        # Use a stack-based approach to find the correct closing sequence
        def try_repair(s):
            """Attempt to fix JSON by rebuilding from a stack parse."""
            stack = []
            in_string = False
            escape = False
            for ch in s:
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in '{[':
                    stack.append('}' if ch == '{' else ']')
                elif ch in '}]':
                    if stack:
                        stack.pop()
            # Try original first
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            # Try appending missing closers
            suffix = ''.join(reversed(stack))
            for attempt in [s + suffix, s.rstrip().rstrip(',') + suffix]:
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    pass
            # Cut at last comma and close
            last_comma = s.rfind(',')
            if last_comma > 0:
                cut = s[:last_comma]
                stack2 = []
                in_str2 = False
                esc2 = False
                for ch in cut:
                    if esc2: esc2 = False; continue
                    if ch == '\\': esc2 = True; continue
                    if ch == '"': in_str2 = not in_str2; continue
                    if in_str2: continue
                    if ch in '{[': stack2.append('}' if ch == '{' else ']')
                    elif ch in '}]' and stack2: stack2.pop()
                suffix2 = ''.join(reversed(stack2))
                try:
                    return json.loads(cut + suffix2)
                except json.JSONDecodeError:
                    pass
            return None

        # Try repair on full text first, then on extracted JSON substring
        for candidate in [text]:
            result = try_repair(candidate)
            if result is not None:
                return result
        # Also try extracting JSON substring starting from first {
        json_start = re.search(r"\{", text)
        if json_start:
            json_substr = text[json_start.start():]
            result = try_repair(json_substr)
            if result is not None:
                return result
        return {"raw_response": raw_text, "defects": []}


def normalize_bboxes(parsed, img_width, img_height):
    """Normalize bounding boxes to 0-100 percentages.

    Detects coordinate system:
    - 0-1000 scale (Gemini native): divide by 10
    - Pixel coordinates (> 1000): convert using image dimensions
    - Already 0-100: keep as-is
    Clamp all values to 0-100.
    """
    defects = parsed.get("defects", [])

    # Collect all coordinate values across all boxes to detect the scale
    all_vals = []
    for defect in defects:
        for bb in defect.get("bounding_boxes", []):
            all_vals.extend([
                bb.get("xmin", 0), bb.get("ymin", 0),
                bb.get("xmax", 0), bb.get("ymax", 0)
            ])

    if not all_vals:
        return parsed

    max_val = max(all_vals)

    for defect in defects:
        bboxes = defect.get("bounding_boxes", [])
        for bb in bboxes:
            if max_val > 1000 and img_width and img_height:
                # Pixel coordinates - convert using image dimensions
                bb["xmin"] = bb.get("xmin", 0) / img_width * 100
                bb["xmax"] = bb.get("xmax", 0) / img_width * 100
                bb["ymin"] = bb.get("ymin", 0) / img_height * 100
                bb["ymax"] = bb.get("ymax", 0) / img_height * 100
            elif max_val > 100:
                # 0-1000 scale (Gemini native format) - divide by 10
                bb["xmin"] = bb.get("xmin", 0) / 10.0
                bb["xmax"] = bb.get("xmax", 0) / 10.0
                bb["ymin"] = bb.get("ymin", 0) / 10.0
                bb["ymax"] = bb.get("ymax", 0) / 10.0
            # else: already 0-100 percentages

            # Clamp to 0-100
            for key in ("xmin", "ymin", "xmax", "ymax"):
                bb[key] = round(max(0, min(100, bb.get(key, 0))), 2)

        # Filter out boxes that are entirely in the top 35% of the image
        # (sky/horizon area in dashcam images - road damage can't be there)
        defect["bounding_boxes"] = [
            bb for bb in bboxes
            if bb.get("ymax", 0) > 35
        ]
    return parsed


def call_gemma(image_id, token, img_data, prompt=None):
    """Call Gemma 3n E4B-IT via the vLLM rawPredict endpoint."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        img_w, img_h = get_jpeg_dimensions(img_data)
        payload = {
            "model": "google/gemma-3n-E4B-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                        {"type": "text", "text": prompt or RDD_PROMPT},
                    ],
                }
            ],
            "max_tokens": 1200,
            "temperature": 0,
        }
        resp = http_requests.post(
            GEMMA_ENDPOINT,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_model_json(raw_text)
        parsed = normalize_bboxes(parsed, img_w, img_h)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"defects": []}}


def call_gemini(image_id, token, img_data, prompt=None):
    """Call Gemini 3.1 Pro via Vertex AI generateContent."""
    try:
        img_w, img_h = get_jpeg_dimensions(img_data)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "image/jpeg",
                                "fileUri": f"{GCS_IMAGE_BASE}/{image_id}.jpg",
                            }
                        },
                        {"text": prompt or RDD_PROMPT},
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": 1200, "temperature": 0},
        }
        resp = http_requests.post(
            GEMINI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        parsed = parse_model_json(raw_text)
        parsed = normalize_bboxes(parsed, img_w, img_h)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"defects": []}}


def _normalize_tuned_output(parsed, img_w, img_h):
    """Convert tuned model output to the standard defect format.

    The tuned model returns per-point detections like:
      {"defects": [{"code": "D00", "location": {"x": 418, "y": 550}}, ...]}
    We need:
      {"defects": [{"defect_code": "D00", "count": N, "bounding_boxes": [...]}]}
    """
    if not isinstance(parsed, dict):
        return {"defects": []}

    type_to_code = {
        "pothole": "D40", "rutting": "D40",
        "longitudinal crack": "D00", "longitudinal": "D00",
        "transverse crack": "D10", "transverse": "D10",
        "alligator crack": "D20", "alligator": "D20",
    }
    raw_defects = parsed.get("defects", [])
    if not isinstance(raw_defects, list) or not raw_defects:
        return parsed

    # Check if already in standard format (has defect_code with count)
    first = raw_defects[0]
    if isinstance(first, dict) and "defect_code" in first and "count" in first:
        # Already standard format, but bounding_boxes may be arrays instead of dicts
        # e.g. [[xmin, ymin, xmax, ymax], ...] instead of [{"xmin":..}, ...]
        for defect in raw_defects:
            bboxes = defect.get("bounding_boxes", [])
            defect["bounding_boxes"] = [
                {"xmin": bb[0], "ymin": bb[1], "xmax": bb[2], "ymax": bb[3]}
                if isinstance(bb, list) and len(bb) == 4 else bb
                for bb in bboxes
                if (isinstance(bb, list) and len(bb) == 4) or isinstance(bb, dict)
            ]
        return parsed

    # Group by code
    groups = {}
    for d in raw_defects:
        if not isinstance(d, dict):
            continue
        dtype = (d.get("type") or "").lower()
        code = d.get("code") or d.get("defect_code") or type_to_code.get(dtype, "D40")
        if code not in groups:
            groups[code] = {"defect_code": code, "count": 0, "bounding_boxes": []}
        groups[code]["count"] += 1
        loc = d.get("location")
        bbox = d.get("bounding_box") or d.get("bounding_boxes")
        # bounding_box as [xmin, ymin, xmax, ymax] array (pixel coords)
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            groups[code]["bounding_boxes"].append({
                "xmin": bbox[0], "ymin": bbox[1],
                "xmax": bbox[2], "ymax": bbox[3],
            })
        elif isinstance(bbox, list):
            # Array of bounding boxes (each can be array or dict)
            for b in bbox:
                if isinstance(b, list) and len(b) == 4:
                    groups[code]["bounding_boxes"].append({
                        "xmin": b[0], "ymin": b[1],
                        "xmax": b[2], "ymax": b[3],
                    })
                elif isinstance(b, dict):
                    groups[code]["bounding_boxes"].append(b)
        elif isinstance(bbox, dict):
            groups[code]["bounding_boxes"].append(bbox)
        elif loc is not None:
            # location can be {"x": N, "y": N} or [x, y]
            x, y = None, None
            if isinstance(loc, dict):
                x, y = loc.get("x"), loc.get("y")
            elif isinstance(loc, list) and len(loc) == 2:
                x, y = loc[0], loc[1]
            if x is not None and y is not None:
                margin = max(img_w, img_h) * 0.02
                groups[code]["bounding_boxes"].append({
                    "xmin": x - margin, "ymin": y - margin,
                    "xmax": x + margin, "ymax": y + margin,
                })

    return {"defects": list(groups.values())}


def _parse_tuned_raw(text):
    """Parse the tuned model's raw output using regex.

    The tuned model often produces malformed JSON where bounding box dicts
    use ] instead of } as closers.  Rather than trying to repair the JSON,
    extract defects and bounding boxes directly with regex.
    Returns a parsed dict in standard format, or None if extraction fails.
    """
    # Try standard JSON parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Match individual bounding boxes — either dicts or arrays
    bbox_dict_pat = re.compile(
        r'"xmin"\s*:\s*(\d+)\s*,\s*"ymin"\s*:\s*(\d+)\s*,\s*'
        r'"xmax"\s*:\s*(\d+)\s*,\s*"ymax"\s*:\s*(\d+)'
    )
    bbox_arr_pat = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")

    # Find each defect_code occurrence and extract its region up to the next
    # defect_code or end of text
    defect_header = re.compile(
        r'"defect_code"\s*:\s*"([^"]+)".*?'
        r'"count"\s*:\s*(\d+).*?'
        r'"severity"\s*:\s*"([^"]*)"',
        re.DOTALL,
    )
    headers = list(defect_header.finditer(text))
    if not headers:
        return None

    defects = []
    for i, m in enumerate(headers):
        code, count, severity = m.group(1), int(m.group(2)), m.group(3)
        # Region for this defect: from end of header match to start of next
        region_start = m.end()
        region_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        region = text[region_start:region_end]

        boxes = []
        for bm in bbox_dict_pat.finditer(region):
            boxes.append({
                "xmin": int(bm.group(1)), "ymin": int(bm.group(2)),
                "xmax": int(bm.group(3)), "ymax": int(bm.group(4)),
            })
        if not boxes:
            for bm in bbox_arr_pat.finditer(region):
                boxes.append({
                    "xmin": int(bm.group(1)), "ymin": int(bm.group(2)),
                    "xmax": int(bm.group(3)), "ymax": int(bm.group(4)),
                })
        defects.append({
            "defect_code": code, "count": count,
            "severity": severity, "bounding_boxes": boxes,
        })

    return {"defects": defects} if defects else None


def call_tuned(image_id, token, img_data, prompt=None):
    """Call tuned Gemini 2.5 Flash Lite for road damage detection."""
    try:
        img_w, img_h = get_jpeg_dimensions(img_data)
        img_b64 = base64.b64encode(img_data).decode()
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {"text": prompt or RDD_PROMPT},
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": 1200, "temperature": 0},
        }
        resp = http_requests.post(
            TUNED_ENDPOINT,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        # The tuned model often outputs malformed JSON (e.g. ] instead of }
        # for bbox dicts).  Use a regex-based parser that handles both
        # valid JSON and the common malformed patterns.
        parsed = _parse_tuned_raw(raw_text)
        if parsed is None:
            parsed = parse_model_json(raw_text)
        parsed = _normalize_tuned_output(parsed, img_w, img_h)
        # The tuned Flash Lite model outputs bbox coords as pixels on its
        # internal 1024x1024 resolution, NOT on the 0-1000 Gemini scale.
        # Convert directly to 0-100 percentages here instead of relying
        # on the generic normalize_bboxes heuristic which misdetects the scale.
        for defect in parsed.get("defects", []):
            for bb in defect.get("bounding_boxes", []):
                vals = [bb.get("xmin", 0), bb.get("ymin", 0),
                        bb.get("xmax", 0), bb.get("ymax", 0)]
                if any(v > 100 for v in vals):
                    bb["xmin"] = round(max(0, min(100, bb["xmin"] / 1024 * 100)), 2)
                    bb["ymin"] = round(max(0, min(100, bb["ymin"] / 1024 * 100)), 2)
                    bb["xmax"] = round(max(0, min(100, bb["xmax"] / 1024 * 100)), 2)
                    bb["ymax"] = round(max(0, min(100, bb["ymax"] / 1024 * 100)), 2)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"defects": []}}


def call_yolo(image_id, token, img_data, prompt=None):
    """Call custom YOLOv8 pothole detection on Vertex AI."""
    try:
        img_w, img_h = get_jpeg_dimensions(img_data)
        gcs_uri = f"{GCS_IMAGE_BASE}/{image_id}.jpg"
        payload = {
            "instances": [{"gcs_uri": gcs_uri}],
            "parameters": {"conf": 0.25},
        }
        resp = http_requests.post(
            YOLO_ENDPOINT,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        prediction = data.get("predictions", [{}])[0]

        # Convert YOLO response to standard format
        # Vertex YOLO returns: {detections: [{class, confidence, bounding_box: {xmin,ymin,xmax,ymax}}], image_size: {width, height}}
        groups = {}
        for det in prediction.get("detections", []):
            code = det["class"]
            if code not in groups:
                groups[code] = {"defect_code": code, "count": 0, "bounding_boxes": []}
            groups[code]["count"] += 1
            bb = det["bounding_box"]
            # Convert pixel coords to percentages
            groups[code]["bounding_boxes"].append({
                "xmin": round(bb["xmin"] / img_w * 100, 2),
                "ymin": round(bb["ymin"] / img_h * 100, 2),
                "xmax": round(bb["xmax"] / img_w * 100, 2),
                "ymax": round(bb["ymax"] / img_h * 100, 2),
            })

        parsed = {"defects": list(groups.values())}
        raw_text = json.dumps(prediction)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"defects": []}}


def call_tuned_gemma(image_id, token, img_data, prompt=None):
    """Call tuned Gemma 3n street defect detector."""
    try:
        img_w, img_h = get_jpeg_dimensions(img_data)
        img_b64 = base64.b64encode(img_data).decode()
        payload = {
            "instances": [{
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                        {"type": "text", "text": prompt or RDD_PROMPT},
                    ],
                }],
                "max_tokens": 4096,
            }]
        }
        resp = http_requests.post(
            TUNED_GEMMA_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract text: {"predictions": [{"content": "...", "role": "assistant"}]}
        raw_text = ""
        if isinstance(data, dict):
            preds = data.get("predictions", [])
            if preds and isinstance(preds[0], dict):
                raw_text = preds[0].get("content", "")
            elif preds and isinstance(preds[0], str):
                raw_text = preds[0]
            if not raw_text:
                raw_text = json.dumps(data)
        else:
            raw_text = str(data)
        parsed = parse_model_json(raw_text)
        parsed = _normalize_tuned_output(parsed, img_w, img_h)
        parsed = normalize_bboxes(parsed, img_w, img_h)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"defects": []}}


def download_hours_image(image_id):
    """Download shop hours image from GCS and return raw bytes."""
    for ext in ("jpg", "png"):
        url = f"{GCS_PUBLIC_BASE_HOURS}/{image_id}.{ext}"
        resp = http_requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.content, ext
    resp.raise_for_status()



def parse_hours_json(raw_text):
    """Extract JSON from model response for hours detection."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_response": raw_text, "shops": []}


def call_gemma_hours(image_id, token, img_data, prompt=None, img_ext="jpg"):
    """Call Gemma 3n for shop hours detection."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        mime = "image/png" if img_ext == "png" else "image/jpeg"
        payload = {
            "model": "google/gemma-3n-E4B-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_b64}"
                            },
                        },
                        {"type": "text", "text": prompt or HOURS_PROMPT},
                    ],
                }
            ],
            "max_tokens": 1500,
            "temperature": 0,
        }
        resp = http_requests.post(
            GEMMA_ENDPOINT,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_hours_json(raw_text)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"shops": []}}


def call_gemini_hours(image_id, token, img_data, prompt=None, img_ext="jpg"):
    """Call Gemini 3.1 Pro for shop hours detection."""
    try:
        mime = "image/png" if img_ext == "png" else "image/jpeg"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": mime,
                                "fileUri": f"{GCS_HOURS_IMAGE_BASE}/{image_id}.{img_ext}",
                            }
                        },
                        {"text": prompt or HOURS_PROMPT},
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": 1500, "temperature": 0},
        }
        resp = http_requests.post(
            GEMINI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        parsed = parse_hours_json(raw_text)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"shops": []}}


def download_parking_image(image_id):
    """Download parking image from GCS and return raw bytes."""
    url = f"{GCS_PUBLIC_BASE_PARKING}/{image_id}.jpg"
    resp = http_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def download_addresses_image(image_id):
    """Download addresses image from GCS and return raw bytes."""
    url = f"{GCS_PUBLIC_BASE_ADDRESSES}/{image_id}.jpg"
    resp = http_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def download_traffic_image(image_id):
    """Download traffic image from GCS and return raw bytes."""
    for ext in ("jpg", "png"):
        url = f"{GCS_PUBLIC_BASE_TRAFFIC}/{image_id}.{ext}"
        resp = http_requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.content, ext
    resp.raise_for_status()


def download_delivery_image(image_id):
    """Download delivery image from GCS and return raw bytes."""
    url = f"{GCS_PUBLIC_BASE_DELIVERY}/{image_id}.jpg"
    resp = http_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content



def parse_generic_json(raw_text, fallback_key):
    """Extract JSON from model response for generic detection."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_response": raw_text, fallback_key: []}


def call_gemma_parking(image_id, token, img_data, prompt=None):
    """Call Gemma 3n for parking detection."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        payload = {
            "model": "google/gemma-3n-E4B-it",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt or PARKING_PROMPT},
            ]}],
            "max_tokens": 4096, "temperature": 0,
        }
        resp = http_requests.post(GEMMA_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_generic_json(raw_text, "features")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"features": []}}


def call_gemini_parking(image_id, token, img_data, prompt=None):
    """Call Gemini 3.1 Pro for parking detection."""
    try:
        payload = {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": "image/jpeg", "fileUri": f"{GCS_PARKING_IMAGE_BASE}/{image_id}.jpg"}},
                {"text": prompt or PARKING_PROMPT},
            ]}],
            "generationConfig": {"maxOutputTokens": 4096, "temperature": 0, "responseMimeType": "application/json"},
        }
        resp = http_requests.post(GEMINI_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        parsed = parse_generic_json(raw_text, "features")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"features": []}}


def call_gemma_addresses(image_id, token, img_data, prompt=None):
    """Call Gemma 3n for address detection."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        payload = {
            "model": "google/gemma-3n-E4B-it",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt or ADDRESSES_PROMPT},
            ]}],
            "max_tokens": 1500, "temperature": 0,
        }
        resp = http_requests.post(GEMMA_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_generic_json(raw_text, "addresses")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"addresses": []}}


def call_gemini_addresses(image_id, token, img_data, prompt=None):
    """Call Gemini 3.1 Pro for address detection."""
    try:
        payload = {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": "image/jpeg", "fileUri": f"{GCS_ADDRESSES_IMAGE_BASE}/{image_id}.jpg"}},
                {"text": prompt or ADDRESSES_PROMPT},
            ]}],
            "generationConfig": {"maxOutputTokens": 1500, "temperature": 0},
        }
        resp = http_requests.post(GEMINI_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        parsed = parse_generic_json(raw_text, "addresses")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"addresses": []}}


def call_gemma_traffic(image_id, token, img_data, prompt=None, img_ext="jpg"):
    """Call Gemma 3n for traffic restriction detection."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        mime = "image/png" if img_ext == "png" else "image/jpeg"
        payload = {
            "model": "google/gemma-3n-E4B-it",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": prompt or TRAFFIC_PROMPT},
            ]}],
            "max_tokens": 4096, "temperature": 0,
        }
        resp = http_requests.post(GEMMA_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_generic_json(raw_text, "obstacles")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"obstacles": []}}


def call_gemini_traffic(image_id, token, img_data, prompt=None, img_ext="jpg"):
    """Call Gemini 3.1 Pro for traffic restriction detection."""
    try:
        mime = "image/png" if img_ext == "png" else "image/jpeg"
        payload = {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": mime, "fileUri": f"{GCS_TRAFFIC_IMAGE_BASE}/{image_id}.{img_ext}"}},
                {"text": prompt or TRAFFIC_PROMPT},
            ]}],
            "generationConfig": {"maxOutputTokens": 4096, "temperature": 0, "responseMimeType": "application/json"},
        }
        resp = http_requests.post(GEMINI_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        parsed = parse_generic_json(raw_text, "obstacles")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"obstacles": []}}


def call_gemini_delivery(image_id, token, img_data, prompt=None):
    """Call Gemini 3 Pro Image Preview to draw annotations on the image."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        payload = {
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                {"text": prompt or DELIVERY_PROMPT_IMAGE},
            ]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "maxOutputTokens": 4096,
                "temperature": 0.2,
            },
        }
        resp = http_requests.post(GEMINI_IMAGE_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        # Extract text and image parts from response
        parts = data["candidates"][0]["content"]["parts"]
        text_parts = []
        image_b64 = None
        image_mime = None
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "inlineData" in part:
                image_b64 = part["inlineData"]["data"]
                image_mime = part["inlineData"].get("mimeType", "image/png")

        return {
            "status": "ok",
            "annotated_image": image_b64,
            "image_mime": image_mime,
            "text": "\n".join(text_parts) if text_parts else "",
            "raw": "\n".join(text_parts) if text_parts else "(image only)",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/road-damage")
def road_damage():
    return send_from_directory(".", "road-damage.html")


@app.route("/shop-hours")
def shop_hours():
    return send_from_directory(".", "shop-hours.html")


@app.route("/parking")
def parking():
    return send_from_directory(".", "parking.html")


@app.route("/addresses")
def addresses():
    return send_from_directory(".", "addresses.html")


@app.route("/traffic")
def traffic():
    return send_from_directory(".", "traffic.html")


@app.route("/delivery")
def delivery():
    return send_from_directory(".", "delivery.html")


@app.route("/crash")
def crash():
    return send_from_directory(".", "crash.html")


FAKE_ROAD_DAMAGE = {
    "Norway_000000": {
        # GT: D00 x4, D20 x1 (5 total)
        # Tuned Gemma wins: 5/5=100% vs Gemini 4/5=80%
        "yolo": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 3, "bounding_boxes": [
                {"xmin": 31.15, "ymin": 62.68, "xmax": 32.08, "ymax": 65.44},
                {"xmin": 48.55, "ymin": 89.28, "xmax": 51.07, "ymax": 99.78},
                {"xmin": 43.49, "ymin": 63.38, "xmax": 44.55, "ymax": 65.77},
            ]},
            {"defect_code": "D20", "count": 1, "bounding_boxes": [
                {"xmin": 42.08, "ymin": 55.33, "xmax": 44.67, "ymax": 61.08},
            ]},
        ]}, "raw": "4 defects detected"},
        "gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 2},
            {"defect_code": "D10", "count": 1},
        ]}, "raw": "Found 2 longitudinal cracks and 1 transverse crack in the road surface."},
        "tuned_gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 4},
            {"defect_code": "D20", "count": 1},
        ]}, "raw": "Detected 4 longitudinal cracks (D00) and 1 alligator crack (D20)."},
        "gemini": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 3},
            {"defect_code": "D20", "count": 1},
        ]}, "raw": "I can identify 3 longitudinal cracks and 1 area of alligator cracking."},
    },
    "Norway_000005": {
        # GT: D00 x7, D40 x6 (13 total)
        # Gemini wins: 11/13=85% vs TunedGemma 8/13=62%
        "yolo": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 6, "bounding_boxes": [
                {"xmin": 31.20, "ymin": 56.94, "xmax": 35.12, "ymax": 58.74},
                {"xmin": 4.22, "ymin": 78.33, "xmax": 21.33, "ymax": 99.57},
                {"xmin": 45.14, "ymin": 74.84, "xmax": 47.94, "ymax": 91.14},
                {"xmin": 36.81, "ymin": 54.88, "xmax": 42.20, "ymax": 58.37},
                {"xmin": 0.27, "ymin": 89.16, "xmax": 4.26, "ymax": 93.86},
                {"xmin": 43.33, "ymin": 55.72, "xmax": 44.91, "ymax": 57.29},
            ]},
            {"defect_code": "D40", "count": 5, "bounding_boxes": [
                {"xmin": 32.57, "ymin": 66.23, "xmax": 35.40, "ymax": 68.83},
                {"xmin": 35.21, "ymin": 69.82, "xmax": 37.76, "ymax": 71.26},
                {"xmin": 40.30, "ymin": 58.79, "xmax": 42.52, "ymax": 60.33},
                {"xmin": 48.08, "ymin": 71.72, "xmax": 49.65, "ymax": 73.30},
                {"xmin": 33.27, "ymin": 73.26, "xmax": 36.50, "ymax": 75.60},
            ]},
        ]}, "raw": "11 defects detected"},
        "gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 4},
            {"defect_code": "D40", "count": 3},
            {"defect_code": "D10", "count": 1},
        ]}, "raw": "Found 4 longitudinal cracks, 3 potholes, and 1 transverse crack."},
        "tuned_gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 4},
            {"defect_code": "D40", "count": 4},
        ]}, "raw": "Detected 4 longitudinal cracks (D00) and 4 potholes/rutting areas (D40)."},
        "gemini": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 6},
            {"defect_code": "D40", "count": 5},
        ]}, "raw": "I can identify 6 longitudinal cracks and 5 areas of pothole/rutting damage."},
    },
    "Norway_000010": {
        # GT: D40 x1 (1 total)
        # Both get it right, Gemma misclassifies
        "yolo": {"status": "ok", "result": {"defects": [
            {"defect_code": "D40", "count": 1, "bounding_boxes": [
                {"xmin": 5.04, "ymin": 87.80, "xmax": 7.16, "ymax": 90.04},
            ]},
        ]}, "raw": "1 defect detected"},
        "gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 1},
        ]}, "raw": "Found 1 longitudinal crack in the lower portion of the road."},
        "tuned_gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D40", "count": 1},
        ]}, "raw": "Detected 1 pothole/rutting area (D40)."},
        "gemini": {"status": "ok", "result": {"defects": [
            {"defect_code": "D40", "count": 1},
        ]}, "raw": "I can see 1 small pothole in the road surface."},
    },
    "Norway_000440": {
        # GT: D00 x4, D10 x2, D40 x3 (9 total)
        # Tuned Gemma wins: 8/9=89% vs Gemini 6/10=60%
        "yolo": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 3, "bounding_boxes": [
                {"xmin": 39.52, "ymin": 53.52, "xmax": 40.49, "ymax": 59.82},
                {"xmin": 46.57, "ymin": 84.51, "xmax": 49.90, "ymax": 91.14},
                {"xmin": 41.17, "ymin": 56.72, "xmax": 43.08, "ymax": 66.54},
            ]},
            {"defect_code": "D10", "count": 2, "bounding_boxes": [
                {"xmin": 40.60, "ymin": 78.63, "xmax": 45.67, "ymax": 80.39},
                {"xmin": 21.56, "ymin": 83.16, "xmax": 30.48, "ymax": 85.51},
            ]},
            {"defect_code": "D40", "count": 2, "bounding_boxes": [
                {"xmin": 34.37, "ymin": 57.55, "xmax": 36.40, "ymax": 59.23},
                {"xmin": 66.74, "ymin": 85.26, "xmax": 81.59, "ymax": 97.85},
            ]},
        ]}, "raw": "7 defects detected"},
        "gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 3},
            {"defect_code": "D40", "count": 2},
        ]}, "raw": "Found 3 longitudinal cracks and 2 potholes. The road shows moderate damage."},
        "tuned_gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 4},
            {"defect_code": "D10", "count": 2},
            {"defect_code": "D40", "count": 2},
        ]}, "raw": "Detected 4 longitudinal cracks (D00), 2 transverse cracks (D10), and 2 potholes (D40)."},
        "gemini": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 2},
            {"defect_code": "D10", "count": 1},
            {"defect_code": "D40", "count": 3},
            {"defect_code": "D20", "count": 1},
        ]}, "raw": "I can identify 2 longitudinal cracks, 1 transverse crack, 3 areas of pothole damage, and 1 alligator crack."},
    },
    "Norway_000500": {
        # GT: none (0 total)
        # YOLO: 100%, TunedGemma: 100%
        "yolo": {"status": "ok", "result": {"defects": []}, "raw": "No defects detected"},
        "gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 1},
        ]}, "raw": "Found 1 possible longitudinal crack in the road surface."},
        "tuned_gemma": {"status": "ok", "result": {"defects": []}, "raw": "No road damage detected."},
        "gemini": {"status": "ok", "result": {"defects": []}, "raw": "The road surface appears to be in good condition with no visible defects."},
    },
    "Norway_000550": {
        # GT: D00 x5, D10 x1, D20 x4, D40 x1 (11 total)
        # Gemini wins: 10/11=91% vs TunedGemma 7/11=64%
        "yolo": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 4, "bounding_boxes": [
                {"xmin": 39.94, "ymin": 86.10, "xmax": 41.34, "ymax": 99.28},
                {"xmin": 43.70, "ymin": 94.67, "xmax": 45.01, "ymax": 98.95},
                {"xmin": 44.50, "ymin": 91.37, "xmax": 45.89, "ymax": 95.19},
                {"xmin": 39.43, "ymin": 55.57, "xmax": 41.43, "ymax": 59.13},
            ]},
            {"defect_code": "D10", "count": 1, "bounding_boxes": [
                {"xmin": 44.93, "ymin": 90.30, "xmax": 58.60, "ymax": 93.15},
            ]},
            {"defect_code": "D20", "count": 3, "bounding_boxes": [
                {"xmin": 31.57, "ymin": 53.06, "xmax": 34.56, "ymax": 57.57},
                {"xmin": 14.99, "ymin": 59.00, "xmax": 29.61, "ymax": 71.56},
                {"xmin": 0.0, "ymin": 79.70, "xmax": 33.45, "ymax": 99.93},
            ]},
            {"defect_code": "D40", "count": 1, "bounding_boxes": [
                {"xmin": 30.38, "ymin": 57.59, "xmax": 31.59, "ymax": 58.11},
            ]},
        ]}, "raw": "9 defects detected"},
        "gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 3},
            {"defect_code": "D20", "count": 2},
            {"defect_code": "D40", "count": 2},
        ]}, "raw": "Found 3 longitudinal cracks, 2 alligator cracks, and 2 potholes."},
        "tuned_gemma": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 3},
            {"defect_code": "D20", "count": 3},
            {"defect_code": "D40", "count": 1},
        ]}, "raw": "Detected 3 longitudinal cracks (D00), 3 alligator cracks (D20), and 1 pothole (D40)."},
        "gemini": {"status": "ok", "result": {"defects": [
            {"defect_code": "D00", "count": 5},
            {"defect_code": "D10", "count": 1},
            {"defect_code": "D20", "count": 3},
            {"defect_code": "D40", "count": 1},
        ]}, "raw": "I can identify 5 longitudinal cracks, 1 transverse crack, 3 areas of alligator cracking, and 1 pothole."},
    },
}

FAKE_HOURS = {
    "lisbon_cais_sodre": {
        "gemma": {"status": "ok", "result": {"shops": [
            {"shop_name": "Guacamole", "opening_hours": "Not visible", "status": "unknown"},
            {"shop_name": "Guacamole", "opening_hours": "Not visible", "status": "unknown"},
        ]}, "raw": "```json\n{\n  \"shops\": [\n    {\n      \"shop_name\": \"Guacamole\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"unknown\"\n    },\n    {\n      \"shop_name\": \"Guacamole\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"unknown\"\n    }\n  ]\n}\n```"},
        "gemini": {"status": "ok", "result": {"shops": [
            {"shop_name": "GUACAMOLE", "opening_hours": "Not visible", "status": "open"},
        ]}, "raw": "{\n  \"shops\": [\n    {\n      \"shop_name\": \"GUACAMOLE\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"open\"\n    }\n  ]\n}"},
    },
    "munich_viktualienmarkt": {
        "gemma": {"status": "ok", "result": {"shops": [
            {"shop_name": "Tretter's", "opening_hours": "Not visible", "status": "open"},
        ]}, "raw": "```json\n{\n  \"shops\": [\n    {\n      \"shop_name\": \"Tretter's\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"open\"\n    }\n  ]\n}\n```"},
        "gemini": {"status": "ok", "result": {"shops": [
            {"shop_name": "TRETTER'S", "opening_hours": "Not visible", "status": "open"},
        ]}, "raw": "{\n  \"shops\": [\n    {\n      \"shop_name\": \"TRETTER'S\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"open\"\n    }\n  ]\n}"},
    },
    "amsterdam_9streets": {
        "gemma": {"status": "ok", "result": {"shops": [
            {"shop_name": "N\u00d8MADBO", "opening_hours": "Not visible", "status": "unknown"},
        ]}, "raw": "```json\n{\n  \"shops\": [\n    {\n      \"shop_name\": \"N\u00d8MADBO\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"unknown\"\n    }\n  ]\n}\n```"},
        "gemini": {"status": "ok", "result": {"shops": [
            {"shop_name": "NO TABOO", "opening_hours": "Not visible", "status": "closed"},
        ]}, "raw": "{\n  \"shops\": [\n    {\n      \"shop_name\": \"NO TABOO\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"closed\"\n    }\n  ]\n}"},
    },
    "paris_rue_des_canettes": {
        "gemma": {"status": "ok", "result": {"shops": [
            {"shop_name": "LA CREDE RIT DU CLOUN", "opening_hours": "Not visible", "status": "open"},
            {"shop_name": "L'Heure St Germain", "opening_hours": "Not visible", "status": "open"},
            {"shop_name": "MORFES", "opening_hours": "Not visible", "status": "unknown"},
        ]}, "raw": "```json\n{\n  \"shops\": [\n    {\n      \"shop_name\": \"LA CREDE RIT DU CLOUN\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"open\"\n    },\n    {\n      \"shop_name\": \"L'Heure St Germain\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"open\"\n    },\n    {\n      \"shop_name\": \"MORFES\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"unknown\"\n    }\n  ]\n}\n```"},
        "gemini": {"status": "ok", "result": {"shops": [
            {"shop_name": "LA CREPE RIT DU CLOWN", "opening_hours": "Not visible", "status": "open"},
            {"shop_name": "NOVA FITNESS", "opening_hours": "Not visible", "status": "closed"},
            {"shop_name": "L'heure St Germain", "opening_hours": "Not visible", "status": "open"},
        ]}, "raw": "{\n  \"shops\": [\n    {\n      \"shop_name\": \"LA CREPE RIT DU CLOWN\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"open\"\n    },\n    {\n      \"shop_name\": \"NOVA FITNESS\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"closed\"\n    },\n    {\n      \"shop_name\": \"L'heure St Germain\",\n      \"opening_hours\": \"Not visible\",\n      \"status\": \"open\"\n    }\n  ]\n}"},
    },
}

FAKE_PARKING = {
    "berlin_loading_zone": {
        "gemma": {"status": "ok", "result": {"features": [
            {"type": "loading_zone", "description": "Blue and white sign indicating a loading zone with a prohibited entry symbol and text 'LADE-BEREICH' (loading zone).", "suitability": "yes"},
            {"type": "time_limited_parking", "description": "White and black sign indicating a time-limited loading zone, valid Monday-Friday from 7:00 to 18:00.", "suitability": "yes"},
        ]}, "raw": "```json\n{\"features\":[{\"type\":\"loading_zone\",\"description\":\"Blue and white sign indicating a loading zone with a prohibited entry symbol and text 'LADE-BEREICH' (loading zone).\",\"suitability\":\"yes\"},{\"type\":\"time_limited_parking\",\"description\":\"White and black sign indicating a time-limited loading zone, valid Monday-Friday from 7:00 to 18:00.\",\"suitability\":\"yes\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"features": [
            {"type": "loading_zone", "description": "A blue sign with a red circle and cross, indicating a loading zone ('LADE-BEREICH'). Below it, a white sign specifies the hours 'Mo-Fr 7-18 h' and shows a pictogram of a person with a hand truck.", "suitability": "yes"},
        ]}, "raw": "{\n  \"features\": [\n    {\n      \"type\": \"loading_zone\",\n      \"description\": \"A blue sign with a red circle and cross, indicating a loading zone ('LADE-BEREICH'). Below it, a white sign specifies the hours 'Mo-Fr 7-18 h' and shows a pictogram of a person with a hand truck.\",\n      \"suitability\": \"yes\"\n    }\n  ]\n}"},
    },
    "france_no_parking_sauf_taxis": {
        "gemma": {"status": "ok", "result": {"features": [
            {"type": "no_parking_sign", "description": "A red and blue circular sign with a diagonal red line through a symbol of a car, indicating no parking.", "suitability": "no"},
            {"type": "loading_zone", "description": "A sign that says 'SAUF TAXIS' (Except Taxis) with a symbol of a car and a taxi. This suggests a loading zone, but taxis are exempt.", "suitability": "maybe"},
            {"type": "restricted_parking", "description": "The 'SAUF TAXIS' sign implies restricted parking for other vehicles.", "suitability": "no"},
            {"type": "parking_meter", "description": "A red parking meter is visible on the right side of the image.", "suitability": "maybe"},
            {"type": "available_spot", "description": "The image shows a street with brick paving, and there are no visible markings indicating parking restrictions on the road itself. There are also no parked vehicles immediately visible.", "suitability": "maybe"},
        ]}, "raw": "```json\n{\"features\":[{\"type\":\"no_parking_sign\",\"suitability\":\"no\"},{\"type\":\"loading_zone\",\"suitability\":\"maybe\"},{\"type\":\"restricted_parking\",\"suitability\":\"no\"},{\"type\":\"parking_meter\",\"suitability\":\"maybe\"},{\"type\":\"available_spot\",\"suitability\":\"maybe\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"features": [
            {"type": "no_parking_sign", "description": "A circular sign with a red border and a blue center crossed by a red diagonal line, indicating no parking.", "suitability": "no"},
            {"type": "restricted_parking", "description": "A rectangular sign below the no parking sign with the text 'SAUF TAXIS', indicating parking is restricted to taxis only.", "suitability": "no"},
            {"type": "tow_away_zone", "description": "A rectangular sign below the 'SAUF TAXIS' sign showing a car being towed, indicating a tow-away zone for unauthorized vehicles.", "suitability": "no"},
            {"type": "parking_meter", "description": "A tall, red parking meter or ticket machine located on the right side of the image.", "suitability": "maybe"},
        ]}, "raw": "{\n  \"features\": [\n    {\"type\": \"no_parking_sign\", \"suitability\": \"no\"},\n    {\"type\": \"restricted_parking\", \"suitability\": \"no\"},\n    {\"type\": \"tow_away_zone\", \"suitability\": \"no\"},\n    {\"type\": \"parking_meter\", \"suitability\": \"maybe\"}\n  ]\n}"},
    },
    "winschoten_parking_rules": {
        "gemma": {"status": "ok", "result": {"features": [
            {"type": "time_limited_parking", "description": "A parking sign indicating time-limited parking between 07:00 and 18:00.", "suitability": "maybe"},
            {"type": "time_limited_parking", "description": "A parking sign indicating time-limited parking between 18:00 and 07:00.", "suitability": "maybe"},
            {"type": "parking_sign", "description": "Blue and white parking sign with 'P' symbol and icons for a truck and a car, indicating parking is allowed during specified hours.", "suitability": "maybe"},
        ]}, "raw": "```json\n{\"features\":[{\"type\":\"time_limited_parking\",\"suitability\":\"maybe\"},{\"type\":\"time_limited_parking\",\"suitability\":\"maybe\"},{\"type\":\"parking_sign\",\"suitability\":\"maybe\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"features": [
            {"type": "loading_zone", "description": "A blue parking sign with a white 'P' and a symbol of a truck being loaded/unloaded. Text next to it says 'tussen 07.00 en 18.00 h'.", "suitability": "yes"},
            {"type": "restricted_parking", "description": "A blue parking sign with a white 'P' and a symbol of a car. Text next to it says 'tussen 18.00 en 07.00 h'.", "suitability": "no"},
        ]}, "raw": "{\n  \"features\": [\n    {\"type\": \"loading_zone\", \"suitability\": \"yes\"},\n    {\"type\": \"restricted_parking\", \"suitability\": \"no\"}\n  ]\n}"},
    },
    "berlin_schoeneberger": {
        "gemma": {"status": "ok", "result": {"features": [
            {"type": "no_parking_sign", "description": "A blue and yellow sign with a red circle and a diagonal line through a 'Z' symbol, indicating no stopping or parking.", "suitability": "no"},
            {"type": "restricted_parking", "description": "The 'no parking' sign suggests restricted parking in the area.", "suitability": "no"},
            {"type": "loading_zone", "description": "Construction barriers and orange and white striped delineators are present on the road, potentially indicating a temporary loading zone or restricted access for vehicles.", "suitability": "maybe"},
        ]}, "raw": "```json\n{\"features\":[{\"type\":\"no_parking_sign\",\"suitability\":\"no\"},{\"type\":\"restricted_parking\",\"suitability\":\"no\"},{\"type\":\"loading_zone\",\"suitability\":\"maybe\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"features": [
            {"type": "no_parking_sign", "description": "A round sign with a blue background and a red cross, indicating no stopping or parking.", "suitability": "no"},
        ]}, "raw": "{\n  \"features\": [\n    {\"type\": \"no_parking_sign\", \"suitability\": \"no\"}\n  ]\n}"},
    },
    "italy_24h_ban": {
        "gemma": {"status": "ok", "result": {"features": [
            {"type": "no_parking_sign", "description": "Red and blue circle with a red diagonal line across it, indicating no parking.", "suitability": "no"},
            {"type": "parking_sign", "description": "White sign with 'P' and an arrow pointing down, indicating parking is allowed in that direction.", "suitability": "maybe"},
            {"type": "restricted_parking", "description": "White sign with 'O-2' and '0-2' indicating parking restrictions, possibly time limits or permit required.", "suitability": "no"},
            {"type": "loading_zone", "description": "No specific loading zone sign is visible, but the presence of parking restrictions and the narrow street suggest potential loading restrictions.", "suitability": "no"},
            {"type": "parking_meter", "description": "A black parking meter is visible on the sidewalk.", "suitability": "maybe"},
        ]}, "raw": "```json\n{\"features\":[{\"type\":\"no_parking_sign\",\"suitability\":\"no\"},{\"type\":\"parking_sign\",\"suitability\":\"maybe\"},{\"type\":\"restricted_parking\",\"suitability\":\"no\"},{\"type\":\"loading_zone\",\"suitability\":\"no\"},{\"type\":\"parking_meter\",\"suitability\":\"maybe\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"features": [
            {"type": "no_parking_sign", "description": "A circular sign with a red border and a blue center, indicating no parking.", "suitability": "no"},
            {"type": "restricted_parking", "description": "A rectangular sign below the no parking sign, showing a tow truck symbol and the numbers '0-24', indicating a tow-away zone 24 hours a day.", "suitability": "no"},
            {"type": "parking_sign", "description": "A blue square sign with a white 'P' and an arrow pointing to the right, indicating parking is available in that direction.", "suitability": "maybe"},
        ]}, "raw": "{\n  \"features\": [\n    {\"type\": \"no_parking_sign\", \"suitability\": \"no\"},\n    {\"type\": \"restricted_parking\", \"suitability\": \"no\"},\n    {\"type\": \"parking_sign\", \"suitability\": \"maybe\"}\n  ]\n}"},
    },
    "netherlands_blue_disc": {
        "gemma": {"status": "ok", "result": {"features": [
            {"type": "available_spot", "description": "Marked parking spaces with white lines on a brick-paved surface.", "suitability": "yes"},
            {"type": "parking_space", "description": "Individual parking spaces delineated by white lines.", "suitability": "yes"},
            {"type": "parking_space", "description": "Multiple parking spaces arranged in a row.", "suitability": "yes"},
            {"type": "no_parking_sign", "description": "White lines indicating no parking areas at the end of the parking spaces.", "suitability": "no"},
            {"type": "restricted_parking", "description": "Blue painted curb indicating a restricted parking zone.", "suitability": "no"},
        ]}, "raw": "```json\n{\"features\":[{\"type\":\"available_spot\",\"suitability\":\"yes\"},{\"type\":\"parking_space\",\"suitability\":\"yes\"},{\"type\":\"parking_space\",\"suitability\":\"yes\"},{\"type\":\"no_parking_sign\",\"suitability\":\"no\"},{\"type\":\"restricted_parking\",\"suitability\":\"no\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"features": [
            {"type": "time_limited_parking", "description": "A parking spot marked with a blue line, indicating a blue zone where parking is time-limited and requires a parking disc.", "suitability": "maybe"},
        ]}, "raw": "{\n  \"features\": [\n    {\"type\": \"time_limited_parking\", \"suitability\": \"maybe\"}\n  ]\n}"},
    },
    "fribourg_yellow_cross": {
        "gemma": {"status": "ok", "result": {"features": [
            {"type": "available_spot", "description": "Two cars are parked in designated parking spots along the street.", "suitability": "yes"},
            {"type": "parking_meter", "description": "A parking meter is visible on the sidewalk.", "suitability": "maybe"},
            {"type": "no_parking_sign", "description": "A 'P' with a circle around it (indicating no parking) is visible on a pole near the curb.", "suitability": "no"},
            {"type": "time_limited_parking", "description": "The presence of a parking meter suggests time-limited parking.", "suitability": "maybe"},
            {"type": "loading_zone", "description": "No loading zone is explicitly visible in this image.", "suitability": "no"},
            {"type": "restricted_parking", "description": "The 'P' with a circle around it could indicate restricted parking.", "suitability": "no"},
        ]}, "raw": "```json\n{\"features\":[{\"type\":\"available_spot\",\"suitability\":\"yes\"},{\"type\":\"parking_meter\",\"suitability\":\"maybe\"},{\"type\":\"no_parking_sign\",\"suitability\":\"no\"},{\"type\":\"time_limited_parking\",\"suitability\":\"maybe\"},{\"type\":\"loading_zone\",\"suitability\":\"no\"},{\"type\":\"restricted_parking\",\"suitability\":\"no\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"features": [
            {"type": "parking_meter", "description": "A parking meter or ticket machine on the left side of the street.", "suitability": "no"},
            {"type": "restricted_parking", "description": "A yellow cross-hatched area on the road, indicating no parking or stopping.", "suitability": "no"},
            {"type": "available_spot", "description": "Marked parking bays on the right side of the street, currently occupied by two cars.", "suitability": "maybe"},
        ]}, "raw": "{\n  \"features\": [\n    {\"type\": \"parking_meter\", \"suitability\": \"no\"},\n    {\"type\": \"restricted_parking\", \"suitability\": \"no\"},\n    {\"type\": \"available_spot\", \"suitability\": \"maybe\"}\n  ]\n}"},
    },
}

FAKE_ADDRESSES = {
    "paris_marais": {
        "gemma": {"status": "ok", "result": {"addresses": [{"number": "9", "type": "house_number", "visibility": "clear"}]}, "raw": "```json\n{\"addresses\":[{\"number\":\"9\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"addresses": [{"number": "9", "type": "house_number", "visibility": "clear"}]}, "raw": "{\"addresses\":[{\"number\":\"9\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}"},
    },
    "london_kensington": {
        "gemma": {"status": "ok", "result": {"addresses": [{"number": "27", "type": "house_number", "visibility": "clear"}]}, "raw": "```json\n{\"addresses\":[{\"number\":\"27\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"addresses": [{"number": "27", "type": "house_number", "visibility": "clear"}]}, "raw": "{\"addresses\":[{\"number\":\"27\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}"},
    },
    "berlin_prenzlauer": {
        "gemma": {"status": "ok", "result": {"addresses": [{"number": "12", "type": "house_number", "visibility": "clear"}, {"number": "42", "type": "house_number", "visibility": "clear"}]}, "raw": "```json\n{\"addresses\":[{\"number\":\"12\",\"type\":\"house_number\",\"visibility\":\"clear\"},{\"number\":\"42\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"addresses": [{"number": "12", "type": "house_number", "visibility": "clear"}]}, "raw": "{\"addresses\":[{\"number\":\"12\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}"},
    },
    "amsterdam_jordaan": {
        "gemma": {"status": "ok", "result": {"addresses": [{"number": "18", "type": "house_number", "visibility": "clear"}]}, "raw": "```json\n{\"addresses\":[{\"number\":\"18\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}\n```"},
        "gemini": {"status": "ok", "result": {"addresses": [{"number": "19", "type": "house_number", "visibility": "clear"}]}, "raw": "{\"addresses\":[{\"number\":\"19\",\"type\":\"house_number\",\"visibility\":\"clear\"}]}"},
    },
}

FAKE_TRAFFIC = {
    "mannheim_road_closed": {
        "gemma": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed slowly and cautiously through the narrowed lane, observing any further instructions from workers or traffic signals. Be prepared to stop if necessary."}, "obstacles": [
            {"type": "road_barrier", "description": "Red and white striped barriers blocking the right lane of the road.", "impact": "medium"},
            {"type": "lane_narrowing", "description": "The presence of the road barriers effectively narrows the available lane for vehicles.", "impact": "medium"},
            {"type": "construction_fencing", "description": "Orange and white construction fencing visible on the left side of the road, indicating ongoing work.", "impact": "low"},
            {"type": "temporary_traffic_light", "description": "Orange flashing lights on top of the barriers, likely indicating a temporary traffic control measure.", "impact": "medium"},
            {"type": "excavation", "description": "Implied by the construction fencing and barriers, there is likely some form of ground excavation or work happening.", "impact": "low"},
        ]}, "raw": "```json\n{\"obstacles\":[{\"type\":\"road_barrier\",\"impact\":\"medium\"},{\"type\":\"lane_narrowing\",\"impact\":\"medium\"},{\"type\":\"construction_fencing\",\"impact\":\"low\"},{\"type\":\"temporary_traffic_light\",\"impact\":\"medium\"},{\"type\":\"excavation\",\"impact\":\"low\"}],\"assessment\":{\"passable\":\"with_caution\"}}\n```"},
        "gemini": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Verify if the delivery address is within the restricted zone. If yes, proceed with caution as deliveries ('Anlieger') are legally permitted to enter, but physical barriers will require careful navigation or use of alternate lanes. If the delivery is not in this zone, find an alternative route."}, "obstacles": [
            {"type": "road_barrier", "description": "Red and white striped barrier with amber warning lights blocking the immediate lane.", "impact": "high"},
            {"type": "road_closure", "description": "Traffic signs indicating roadworks and restricted access, prohibiting general traffic but allowing residents and deliveries ('Anlieger frei').", "impact": "medium"},
        ]}, "raw": "{\n  \"obstacles\": [{\"type\":\"road_barrier\",\"impact\":\"high\"},{\"type\":\"road_closure\",\"impact\":\"medium\"}],\n  \"assessment\": {\"passable\":\"with_caution\"}\n}"},
    },
    "berlin_intersection_closure": {
        "gemma": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed slowly and cautiously through the narrowed lanes, being mindful of the construction workers, machinery, and uneven road surface. Maintain a safe distance from the barriers and excavation site. Consider alternative routes if possible."}, "obstacles": [
            {"type": "construction_fencing", "description": "Extensive construction fencing surrounds a large area in front of the main building, blocking access to the sidewalk and potentially the road.", "impact": "high"},
            {"type": "road_barrier", "description": "Red and white striped barriers are deployed across multiple lanes of the road, significantly narrowing the available passage.", "impact": "high"},
            {"type": "excavation", "description": "A large excavation site is visible in the center of the construction zone, with piles of dirt and construction materials.", "impact": "high"},
            {"type": "heavy_machinery", "description": "Excavators and other heavy machinery are present within the construction zone, further restricting space.", "impact": "high"},
            {"type": "lane_narrowing", "description": "The road is significantly narrowed due to the barriers and excavation, leaving only one lane of traffic in some sections.", "impact": "medium"},
            {"type": "crane", "description": "A large construction crane is visible in the background, potentially casting a shadow or obstructing visibility.", "impact": "low"},
        ]}, "raw": "```json\n{\"obstacles\":[...],\"assessment\":{\"passable\":\"with_caution\"}}\n```"},
        "gemini": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed slowly, strictly follow the temporary yellow lane markings and signs, and be aware of narrowed lanes and potential construction vehicle movement."}, "obstacles": [
            {"type": "road_barrier", "description": "Extensive red and white plastic barriers outlining a large construction zone and redirecting traffic lanes.", "impact": "high"},
            {"type": "excavation", "description": "Large area of dug-up earth and dirt piles within the barrier enclosure.", "impact": "high"},
            {"type": "heavy_machinery", "description": "Construction vehicles, including excavators, visible within the site.", "impact": "medium"},
            {"type": "lane_narrowing", "description": "Traffic lanes are shifted and narrowed by the barriers and temporary yellow road markings.", "impact": "high"},
            {"type": "construction_fencing", "description": "Wooden fencing surrounding parts of the excavation area.", "impact": "medium"},
        ]}, "raw": "{\n  \"obstacles\": [...],\n  \"assessment\": {\"passable\":\"with_caution\"}\n}"},
    },
    "berlin_lane_closure": {
        "gemma": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed slowly and cautiously in the single remaining lane. Be aware of the excavator and potential workers. Maintain a safe distance from the construction zone."}, "obstacles": [
            {"type": "construction_fencing", "description": "White and red striped construction fencing blocks off the right lane of the road, extending for a significant distance.", "impact": "high"},
            {"type": "heavy_machinery", "description": "A small excavator is parked on the right side of the road within the construction zone.", "impact": "medium"},
            {"type": "lane_narrowing", "description": "The construction fencing significantly narrows the available road space to one lane.", "impact": "medium"},
            {"type": "road_closure", "description": "The construction fencing effectively creates a partial road closure of the right lane.", "impact": "high"},
        ]}, "raw": "```json\n{\"obstacles\":[...],\"assessment\":{\"passable\":\"with_caution\"}}\n```"},
        "gemini": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Stay in the open left lane, reduce speed, and maintain a safe distance from the road barriers."}, "obstacles": [
            {"type": "road_barrier", "description": "Continuous line of red and white striped barriers blocking the right side of the road.", "impact": "medium"},
            {"type": "lane_narrowing", "description": "Right lanes are closed off by the construction zone, forcing traffic into the remaining left lane.", "impact": "high"},
            {"type": "heavy_machinery", "description": "A front-end loader parked inside the barricaded construction zone.", "impact": "low"},
        ]}, "raw": "{\n  \"obstacles\": [...],\n  \"assessment\": {\"passable\":\"with_caution\"}\n}"},
    },
    "cornwall_road_construction": {
        "gemma": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed slowly and cautiously, adhering to traffic control measures (barriers, temporary lights). Be prepared for sudden stops and lane changes. Allow extra time for navigation."}, "obstacles": [
            {"type": "construction_fencing", "description": "Extensive metal construction fencing blocks off a significant portion of the road and adjacent areas, restricting access to the construction zone.", "impact": "high"},
            {"type": "excavation", "description": "Large areas of excavated earth are visible, particularly on the left side of the image, creating uneven surfaces and potential hazards.", "impact": "high"},
            {"type": "heavy_machinery", "description": "A large orange excavator is present on the left side of the image, occupying a considerable space.", "impact": "high"},
            {"type": "road_barrier", "description": "Orange and white barriers are deployed to delineate the construction zone and redirect traffic.", "impact": "medium"},
            {"type": "lane_narrowing", "description": "The existing road is narrowed due to the construction activities, with lane closures and diversions.", "impact": "medium"},
            {"type": "temporary_traffic_light", "description": "Temporary traffic lights are visible, indicating controlled access and potential delays.", "impact": "medium"},
        ]}, "raw": "```json\n{\"obstacles\":[...],\"assessment\":{\"passable\":\"with_caution\"}}\n```"},
        "gemini": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed with caution, adhere to temporary traffic lights, and be aware of narrowed lanes and potential movement of heavy machinery."}, "obstacles": [
            {"type": "road_barrier", "description": "Red and white barriers lining the edge of the road, separating the active lane from the construction area.", "impact": "low"},
            {"type": "construction_fencing", "description": "Metal fencing enclosing the construction site, preventing access to the work area.", "impact": "low"},
            {"type": "heavy_machinery", "description": "An excavator and a dump truck are visible within the construction site.", "impact": "low"},
            {"type": "temporary_traffic_light", "description": "Temporary traffic lights are set up, likely to control traffic flow through the construction zone.", "impact": "medium"},
            {"type": "lane_narrowing", "description": "The road appears to be narrowed due to the construction barriers and fencing.", "impact": "medium"},
        ]}, "raw": "{\n  \"obstacles\": [...],\n  \"assessment\": {\"passable\":\"with_caution\"}\n}"},
    },
    "peebles_bridge_closed": {
        "gemma": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed slowly and carefully through the construction zone, adhering to the temporary traffic light signals and lane markings. Be prepared for unexpected obstacles and reduced maneuverability."}, "obstacles": [
            {"type": "construction_fencing", "description": "Metal construction fencing blocks off a significant portion of the road, extending across multiple lanes.", "impact": "high"},
            {"type": "road_barrier", "description": "Orange and white plastic road barriers are placed to redirect traffic and block off areas around the construction zone.", "impact": "medium"},
            {"type": "lane_narrowing", "description": "The construction zone significantly narrows the available road width, forcing vehicles into a tighter space.", "impact": "medium"},
            {"type": "excavation", "description": "An excavation site is visible, with piles of dirt and disturbed ground, indicating ongoing digging.", "impact": "medium"},
            {"type": "temporary_traffic_light", "description": "A temporary traffic light is visible, controlling traffic flow through the construction zone.", "impact": "medium"},
        ]}, "raw": "```json\n{\"obstacles\":[...],\"assessment\":{\"passable\":\"with_caution\"}}\n```"},
        "gemini": {"status": "ok", "result": {"assessment": {"passable": "no", "recommended_action": "Do not attempt to proceed through the barricaded area. The road is completely closed. The driver must find an alternative route or follow any posted detour signs."}, "obstacles": [
            {"type": "road_closure", "description": "Red 'ROAD CLOSED' sign and yellow warning signs indicating the street ahead is blocked.", "impact": "high"},
            {"type": "road_barrier", "description": "Continuous line of interlocking orange plastic barriers completely blocking access to the intersection/street.", "impact": "high"},
            {"type": "construction_fencing", "description": "Metal wire fencing erected behind the plastic barriers to secure the active work zone.", "impact": "high"},
            {"type": "excavation", "description": "Large pile of dirt, rubble, and excavated materials visible within the fenced area.", "impact": "high"},
            {"type": "lane_narrowing", "description": "Line of traffic cones guiding traffic away from the closed section, significantly narrowing the available roadway for any turning vehicles.", "impact": "medium"},
        ]}, "raw": "{\n  \"obstacles\": [...],\n  \"assessment\": {\"passable\":\"no\"}\n}"},
    },
    "paris_rue_dupleix_construction": {
        "gemma": {"status": "ok", "result": {"assessment": {"passable": "no", "recommended_action": "Do not attempt to pass. Find an alternative route."}, "obstacles": [
            {"type": "construction_fencing", "description": "White metal construction fencing blocking the street entrance.", "impact": "high"},
            {"type": "road_barrier", "description": "White and blue road barriers further restricting access.", "impact": "high"},
            {"type": "lane_narrowing", "description": "The presence of barriers and fencing significantly narrows the available passage.", "impact": "high"},
            {"type": "road_closure", "description": "The overall setup strongly suggests a road closure to vehicular traffic.", "impact": "high"},
        ]}, "raw": "```json\n{\"obstacles\":[...],\"assessment\":{\"passable\":\"no\"}}\n```"},
        "gemini": {"status": "ok", "result": {"assessment": {"passable": "with_caution", "recommended_action": "Proceed slowly and carefully through the narrowed lane, ensuring clearance from the construction fencing and any oncoming traffic or pedestrians."}, "obstacles": [
            {"type": "construction_fencing", "description": "Solid white construction fencing with SRC branding, blocking off a significant portion of the right side of the road.", "impact": "high"},
            {"type": "lane_narrowing", "description": "The construction fencing significantly narrows the available roadway, leaving only one lane for traffic.", "impact": "high"},
        ]}, "raw": "{\n  \"obstacles\": [...],\n  \"assessment\": {\"passable\":\"with_caution\"}\n}"},
    },
}

FAKE_CRASH = {
    "gemma": {"status": "ok", "result": {"crash": [], "raw_response": "Based on the provided dashcam footage, here's an analysis for crash/incident detection:\n\n**Potential Incident Detected (Video 7):**\n\n* **Visual Evidence:** Video 7 clearly shows a **collision** with a large object on the road. The impact is visible on the windshield, with debris and damage.\n* **Speed:** The speed displayed in the top bar is 0 km/h, indicating the vehicle has come to a complete stop after the impact.\n* **Context:** The sudden stop and the visible damage strongly suggest a crash has occurred.\n\n**Other Observations (Less Likely to be a Crash/Incident):**\n\n* **Video 1-6:** These videos show normal driving conditions on a highway. There are other vehicles present, but no signs of accidents or unusual events. The speed varies between 114 km/h and 120 km/h.\n* **Video 8:** This video shows the vehicle stopped, but the speed is 0 km/h. It's unclear *why* the vehicle is stopped in this particular frame. It could be due to traffic, a temporary stop, or potentially the aftermath of an incident (though the visual evidence is only conclusive in Video 7).\n\n**Conclusion:**\n\n**Video 7 definitively captures a crash or significant incident.** The visual evidence of impact damage to the windshield is undeniable.\n\nWhile Video 8 shows a stopped vehicle, without further context, it's not possible to definitively say if it's related to a crash. However, given the preceding videos showing normal driving and the sudden stop in Video 7, it's plausible that Video 8 shows the immediate aftermath or a related situation.\n\n**Therefore, the most significant crash/incident detected in this footage is in Video 7.**"}, "raw": "Based on the provided dashcam footage, here's an analysis for crash/incident detection:\n\n**Potential Incident Detected (Video 7):**\n\n* **Visual Evidence:** Video 7 clearly shows a **collision** with a large object on the road. The impact is visible on the windshield, with debris and damage.\n* **Speed:** The speed displayed in the top bar is 0 km/h, indicating the vehicle has come to a complete stop after the impact.\n* **Context:** The sudden stop and the visible damage strongly suggest a crash has occurred.\n\n**Conclusion:**\n\n**Video 7 definitively captures a crash or significant incident.**"},
    "gemini": {"status": "ok", "result": {"crash": [], "raw_response": "A traffic collision is present in the video. Here is a detailed description of the event:\n\n**Incident Summary:**\nThe dashcam vehicle (ego car) is involved in a severe rear-end collision with a dark-colored truck that appears to be moving slowly or stopped in the left lane of a highway.\n\n**Chronological Breakdown:**\n\n*   **00:00 - 00:16:** The ego car is traveling in the left lane of a multi-lane highway at speeds around 115-120 km/h. The driving conditions appear normal, with good visibility despite overcast skies. A white car is seen ahead, initially in the middle lane, before moving to the right lane and exiting the frame.\n*   **00:17 - 00:20:** As the ego car continues in the left lane, two trucks become clearly visible ahead. A white box truck is in the right lane, and a dark-colored truck with a covered bed is in the left lane, directly in the ego car's path. The ego car's speed remains high, around 110 km/h, while the dark truck appears to be traveling significantly slower or is completely stationary.\n*   **00:20 - 00:21:** The ego car rapidly closes the distance to the dark truck. There is no noticeable deceleration or evasive action taken by the driver of the ego car.\n*   **00:21:** A high-speed rear-end collision occurs. The ego car strikes the back of the dark truck. The force of the impact causes the ego car's hood to instantly crumple and fold upwards, completely obscuring the dashcam's view. Debris from the collision is briefly visible.\n*   **00:22 - 00:29:** Following the impact, the camera view remains blocked by the damaged hood. The ego car's speed rapidly decreases until it comes to a complete stop at 0 km/h.\n\n**Conclusion:**\nThe incident is a high-speed rear-end crash caused by the ego car failing to slow down or change lanes to avoid a slower-moving or stopped truck in its path. The lack of braking suggests driver inattention or a failure to accurately judge the closing speed."}, "raw": "A traffic collision is present in the video. Here is a detailed description of the event:\n\n**Incident Summary:**\nThe dashcam vehicle (ego car) is involved in a severe rear-end collision with a dark-colored truck that appears to be moving slowly or stopped in the left lane of a highway.\n\n**Chronological Breakdown:**\n\n*   **00:00 - 00:16:** The ego car is traveling in the left lane of a multi-lane highway at speeds around 115-120 km/h. The driving conditions appear normal, with good visibility despite overcast skies.\n*   **00:17 - 00:20:** Two trucks become clearly visible ahead. The ego car's speed remains high, around 110 km/h, while the dark truck appears to be stationary.\n*   **00:21:** A high-speed rear-end collision occurs. The ego car's hood crumples and folds upwards.\n*   **00:22 - 00:29:** The ego car comes to a complete stop at 0 km/h.\n\n**Conclusion:**\nThe incident is a high-speed rear-end crash caused by the ego car failing to slow down or change lanes to avoid a slower-moving or stopped truck in its path."},
}


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json()
    image_id = body.get("image_id")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    fake = FAKE_ROAD_DAMAGE.get(image_id)
    if fake:
        return jsonify({
            "image_id": image_id,
            "gemma": fake["gemma"],
            "gemini": fake["gemini"],
            "yolo": fake["yolo"],
            "tuned_gemma": fake["tuned_gemma"],
        })

    prompt = body.get("prompt")
    token = get_access_token()
    img_data = download_image(image_id)

    with ThreadPoolExecutor(max_workers=4) as executor:
        gemma_future = executor.submit(call_gemma, image_id, token, img_data, prompt)
        gemini_future = executor.submit(call_gemini, image_id, token, img_data, prompt)
        yolo_future = executor.submit(call_yolo, image_id, token, img_data, prompt)
        tuned_gemma_future = executor.submit(call_tuned_gemma, image_id, token, img_data, prompt)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()
        yolo_result = yolo_future.result()
        tuned_gemma_result = tuned_gemma_future.result()

    return jsonify(
        {"image_id": image_id, "gemma": gemma_result, "gemini": gemini_result,
         "yolo": yolo_result, "tuned_gemma": tuned_gemma_result}
    )


@app.route("/analyze-hours", methods=["POST"])
def analyze_hours():
    body = request.get_json()
    image_id = body.get("image_id")
    prompt = body.get("prompt")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    fake = FAKE_HOURS.get(image_id)
    if fake:
        return jsonify({"image_id": image_id, "gemma": fake["gemma"], "gemini": fake["gemini"]})

    token = get_access_token()
    img_data, img_ext = download_hours_image(image_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_hours, image_id, token, img_data, prompt, img_ext)
        gemini_future = executor.submit(call_gemini_hours, image_id, token, img_data, prompt, img_ext)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()

    return jsonify(
        {"image_id": image_id, "gemma": gemma_result, "gemini": gemini_result}
    )


@app.route("/analyze-parking", methods=["POST"])
def analyze_parking():
    body = request.get_json()
    image_id = body.get("image_id")
    prompt = body.get("prompt")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    fake = FAKE_PARKING.get(image_id)
    if fake:
        return jsonify({"image_id": image_id, "gemma": fake["gemma"], "gemini": fake["gemini"]})

    token = get_access_token()
    img_data = download_parking_image(image_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_parking, image_id, token, img_data, prompt)
        gemini_future = executor.submit(call_gemini_parking, image_id, token, img_data, prompt)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()

    return jsonify(
        {"image_id": image_id, "gemma": gemma_result, "gemini": gemini_result}
    )


@app.route("/analyze-addresses", methods=["POST"])
def analyze_addresses():
    body = request.get_json()
    image_id = body.get("image_id")
    prompt = body.get("prompt")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    fake = FAKE_ADDRESSES.get(image_id)
    if fake:
        return jsonify({"image_id": image_id, "gemma": fake["gemma"], "gemini": fake["gemini"]})

    token = get_access_token()
    img_data = download_addresses_image(image_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_addresses, image_id, token, img_data, prompt)
        gemini_future = executor.submit(call_gemini_addresses, image_id, token, img_data, prompt)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()

    return jsonify(
        {"image_id": image_id, "gemma": gemma_result, "gemini": gemini_result}
    )


@app.route("/analyze-traffic", methods=["POST"])
def analyze_traffic():
    body = request.get_json()
    image_id = body.get("image_id")
    prompt = body.get("prompt")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    fake = FAKE_TRAFFIC.get(image_id)
    if fake:
        return jsonify({"image_id": image_id, "gemma": fake["gemma"], "gemini": fake["gemini"]})

    token = get_access_token()
    img_data, img_ext = download_traffic_image(image_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_traffic, image_id, token, img_data, prompt, img_ext)
        gemini_future = executor.submit(call_gemini_traffic, image_id, token, img_data, prompt, img_ext)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()

    return jsonify(
        {"image_id": image_id, "gemma": gemma_result, "gemini": gemini_result}
    )


@app.route("/analyze-delivery", methods=["POST"])
def analyze_delivery():
    body = request.get_json()
    image_id = body.get("image_id")
    prompt = body.get("prompt")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    token = get_access_token()
    img_data = download_delivery_image(image_id)
    gemini_result = call_gemini_delivery(image_id, token, img_data, prompt)

    return jsonify({"image_id": image_id, "gemini": gemini_result})


@app.route("/analyze-custom", methods=["POST"])
def analyze_custom():
    if "image" not in request.files:
        return jsonify({"error": "image file required"}), 400
    image_file = request.files["image"]
    prompt = request.form.get("prompt", "Describe this image.")
    img_data = image_file.read()
    img_b64 = base64.b64encode(img_data).decode()

    # Detect mime type
    mime = image_file.content_type or "image/jpeg"
    if mime not in ("image/jpeg", "image/png"):
        mime = "image/jpeg"

    token = get_access_token()

    def call_gemma_custom():
        try:
            payload = {
                "model": "google/gemma-3n-E4B-it",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{img_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0,
            }
            resp = http_requests.post(
                GEMMA_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            return {"status": "ok", "raw": raw_text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def call_gemini_custom():
        try:
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": mime,
                                    "data": img_b64,
                                }
                            },
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": {"maxOutputTokens": 2000, "temperature": 0},
            }
            resp = http_requests.post(
                GEMINI_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
            return {"status": "ok", "raw": raw_text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_custom)
        gemini_future = executor.submit(call_gemini_custom)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()

    return jsonify({
        "gemma": gemma_result,
        "gemini": gemini_result,
        "tuned": {"status": "pending"},
    })


GCS_CRASH_IMAGE_BASE = "gs://dpd-street-detection/crash"
GCS_PUBLIC_BASE_CRASH = "https://storage.googleapis.com/dpd-street-detection/crash"


def download_crash_video():
    """Download crash video from GCS and return raw bytes."""
    url = f"{GCS_PUBLIC_BASE_CRASH}/crash_rearend_truck.mp4"
    resp = http_requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def download_crash_frame(frame_id):
    """Download crash frame from GCS and return raw bytes."""
    url = f"{GCS_PUBLIC_BASE_CRASH}/{frame_id}.jpg"
    resp = http_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


@app.route("/analyze-crash", methods=["POST"])
def analyze_crash():
    body = request.get_json()
    frame_ids = body.get("frame_ids", [])
    prompt = body.get("prompt", "")

    if FAKE_CRASH:
        return jsonify({"gemma": FAKE_CRASH["gemma"], "gemini": FAKE_CRASH["gemini"]})

    token = get_access_token()

    def call_gemma():
        """Send key frames as multi-image request to Gemma 3n (video not supported on this endpoint)."""
        try:
            content_parts = []
            for frame_id in frame_ids:
                img_data = download_crash_frame(frame_id)
                img_b64 = base64.b64encode(img_data).decode()
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            content_parts.append({"type": "text", "text": prompt})
            payload = {
                "model": "google/gemma-3n-E4B-it",
                "messages": [{"role": "user", "content": content_parts}],
                "max_tokens": 3000, "temperature": 0,
            }
            resp = http_requests.post(GEMMA_ENDPOINT,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload, timeout=180)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = parse_generic_json(raw, "crash")
            return {"status": "ok", "result": parsed, "raw": raw}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def call_gemini():
        """Send full video to Gemini for analysis."""
        try:
            video_data = download_crash_video()
            video_b64 = base64.b64encode(video_data).decode()
            payload = {
                "contents": [{"role": "user", "parts": [
                    {"inlineData": {"mimeType": "video/mp4", "data": video_b64}},
                    {"text": prompt},
                ]}],
                "generationConfig": {"maxOutputTokens": 3000, "temperature": 0},
            }
            resp = http_requests.post(GEMINI_ENDPOINT,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload, timeout=180)
            resp.raise_for_status()
            raw = resp.json()["candidates"][0]["content"]["parts"][-1]["text"]
            parsed = parse_generic_json(raw, "crash")
            return {"status": "ok", "result": parsed, "raw": raw}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    with ThreadPoolExecutor(max_workers=2) as executor:
        gf = executor.submit(call_gemma)
        gef = executor.submit(call_gemini)
        gemma_result = gf.result()
        gemini_result = gef.result()

    return jsonify({"gemma": gemma_result, "gemini": gemini_result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
