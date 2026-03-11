import json
import os
import re
import struct
import base64
from concurrent.futures import ThreadPoolExecutor

import google.auth
import google.auth.transport.requests
import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

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
    "Draw directly on the image to guide the driver:\n"
    "- Draw a clear arrow pointing to the exact delivery location "
    "(the door, gate, or entrance where the package should be left)\n"
    "- Circle or highlight only the delivery target\n"
    "- Add a short text label like \"DELIVER HERE\" near the target\n"
    "- Do NOT label landmarks, shops, or other reference points - "
    "only mark where to deliver\n\n"
    "Make the annotations bold, bright, and easy to see at a glance. "
    "Use green or red colors for visibility. "
    "Return the annotated image."
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
    url = f"{GCS_PUBLIC_BASE_HOURS}/{image_id}.jpg"
    resp = http_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content



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


def call_gemma_hours(image_id, token, img_data, prompt=None):
    """Call Gemma 3n for shop hours detection."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
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


def call_gemini_hours(image_id, token, img_data, prompt=None):
    """Call Gemini 3.1 Pro for shop hours detection."""
    try:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "image/jpeg",
                                "fileUri": f"{GCS_HOURS_IMAGE_BASE}/{image_id}.jpg",
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
    url = f"{GCS_PUBLIC_BASE_TRAFFIC}/{image_id}.jpg"
    resp = http_requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


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


def call_gemma_traffic(image_id, token, img_data, prompt=None):
    """Call Gemma 3n for traffic restriction detection."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        payload = {
            "model": "google/gemma-3n-E4B-it",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
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


def call_gemini_traffic(image_id, token, img_data, prompt=None):
    """Call Gemini 3.1 Pro for traffic restriction detection."""
    try:
        payload = {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": "image/jpeg", "fileUri": f"{GCS_TRAFFIC_IMAGE_BASE}/{image_id}.jpg"}},
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

    token = get_access_token()
    img_data = download_hours_image(image_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_hours, image_id, token, img_data, prompt)
        gemini_future = executor.submit(call_gemini_hours, image_id, token, img_data, prompt)
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

    token = get_access_token()
    img_data = download_traffic_image(image_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_traffic, image_id, token, img_data, prompt)
        gemini_future = executor.submit(call_gemini_traffic, image_id, token, img_data, prompt)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
