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
    "https://1901510802238603264.europe-west4-668228315581.prediction.vertexai.goog"
    "/v1/projects/mineral-concord-394714/locations/europe-west4"
    "/endpoints/1901510802238603264:rawPredict"
)
GEMINI_ENDPOINT = (
    "https://aiplatform.googleapis.com/v1/projects/mineral-concord-394714"
    "/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"
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
    "You are a traffic restriction detection AI analyzing street-level images of "
    "European cities. Identify all traffic signs, restrictions, and regulations "
    "visible in the image that could affect a delivery van.\n\n"
    "For each restriction found, provide:\n"
    "- type: the type of restriction (e.g. \"weight_limit\", \"height_limit\", "
    "\"no_entry\", \"one_way\", \"pedestrian_zone\", \"ZTL\", \"time_restriction\", "
    "\"speed_limit\", \"no_trucks\", \"low_emission_zone\")\n"
    "- description: brief description of what the sign/restriction indicates\n"
    "- impact: impact on a standard DPD delivery van (\"high\", \"medium\", \"low\", "
    "\"none\")\n\n"
    "Return ONLY a JSON object (no markdown fences) with a \"restrictions\" array. "
    "Each element must include type, description, and impact."
)

DELIVERY_PROMPT = (
    "You are an AR delivery assistant AI analyzing a street-level image to help "
    "a DPD delivery driver find the exact delivery location.\n\n"
    "The customer left the following delivery note:\n"
    "\"{delivery_note}\"\n\n"
    "Analyze the image and:\n"
    "1. Identify the exact location described in the delivery note\n"
    "2. Provide step-by-step visual instructions for the driver\n"
    "3. Draw a bounding box around the delivery target area\n\n"
    "For each instruction step, provide:\n"
    "- action: what the driver should do\n"
    "- detail: additional context or landmarks\n"
    "- confidence: how confident you are this matches the note "
    "(\"high\", \"medium\", \"low\")\n"
    "- bounding_box: {{xmin, ymin, xmax, ymax}} as percentages (0-100) of image "
    "dimensions marking the relevant area. Only include for the primary delivery target.\n\n"
    "Return ONLY a JSON object (no markdown fences) with an \"instructions\" array. "
    "Each element must include action, detail, and confidence. "
    "Include bounding_box only on the step that marks the delivery location."
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
            "max_tokens": 1500, "temperature": 0,
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
            "generationConfig": {"maxOutputTokens": 1500, "temperature": 0},
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
            "max_tokens": 1500, "temperature": 0,
        }
        resp = http_requests.post(GEMMA_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_generic_json(raw_text, "restrictions")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"restrictions": []}}


def call_gemini_traffic(image_id, token, img_data, prompt=None):
    """Call Gemini 3.1 Pro for traffic restriction detection."""
    try:
        payload = {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": "image/jpeg", "fileUri": f"{GCS_TRAFFIC_IMAGE_BASE}/{image_id}.jpg"}},
                {"text": prompt or TRAFFIC_PROMPT},
            ]}],
            "generationConfig": {"maxOutputTokens": 1500, "temperature": 0},
        }
        resp = http_requests.post(GEMINI_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        parsed = parse_generic_json(raw_text, "restrictions")
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"restrictions": []}}


def normalize_delivery_bboxes(parsed, img_width, img_height):
    """Normalize bounding boxes in delivery instructions to 0-100 percentages."""
    instructions = parsed.get("instructions", [])
    all_vals = []
    for inst in instructions:
        bb = inst.get("bounding_box")
        if bb:
            all_vals.extend([
                bb.get("xmin", 0), bb.get("ymin", 0),
                bb.get("xmax", 0), bb.get("ymax", 0)
            ])
    if not all_vals:
        return parsed
    max_val = max(all_vals)
    for inst in instructions:
        bb = inst.get("bounding_box")
        if not bb:
            continue
        if max_val > 1000 and img_width and img_height:
            bb["xmin"] = bb.get("xmin", 0) / img_width * 100
            bb["xmax"] = bb.get("xmax", 0) / img_width * 100
            bb["ymin"] = bb.get("ymin", 0) / img_height * 100
            bb["ymax"] = bb.get("ymax", 0) / img_height * 100
        elif max_val > 100:
            bb["xmin"] = bb.get("xmin", 0) / 10.0
            bb["xmax"] = bb.get("xmax", 0) / 10.0
            bb["ymin"] = bb.get("ymin", 0) / 10.0
            bb["ymax"] = bb.get("ymax", 0) / 10.0
        for key in ("xmin", "ymin", "xmax", "ymax"):
            bb[key] = round(max(0, min(100, bb.get(key, 0))), 2)
    return parsed


def call_gemma_delivery(image_id, token, img_data, prompt=None):
    """Call Gemma 3n for delivery instruction detection."""
    try:
        img_b64 = base64.b64encode(img_data).decode()
        img_w, img_h = get_jpeg_dimensions(img_data)
        payload = {
            "model": "google/gemma-3n-E4B-it",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt or DELIVERY_PROMPT},
            ]}],
            "max_tokens": 2000, "temperature": 0,
        }
        resp = http_requests.post(GEMMA_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_generic_json(raw_text, "instructions")
        parsed = normalize_delivery_bboxes(parsed, img_w, img_h)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"instructions": []}}


def call_gemini_delivery(image_id, token, img_data, prompt=None):
    """Call Gemini 3.1 Pro for delivery instruction detection."""
    try:
        img_w, img_h = get_jpeg_dimensions(img_data)
        payload = {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": "image/jpeg", "fileUri": f"{GCS_DELIVERY_IMAGE_BASE}/{image_id}.jpg"}},
                {"text": prompt or DELIVERY_PROMPT},
            ]}],
            "generationConfig": {"maxOutputTokens": 2000, "temperature": 0},
        }
        resp = http_requests.post(GEMINI_ENDPOINT, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][-1]["text"]
        parsed = parse_generic_json(raw_text, "instructions")
        parsed = normalize_delivery_bboxes(parsed, img_w, img_h)
        return {"status": "ok", "result": parsed, "raw": raw_text}
    except Exception as e:
        return {"status": "error", "error": str(e), "result": {"instructions": []}}


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


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json()
    image_id = body.get("image_id")
    prompt = body.get("prompt")
    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    token = get_access_token()
    img_data = download_image(image_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma, image_id, token, img_data, prompt)
        gemini_future = executor.submit(call_gemini, image_id, token, img_data, prompt)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()

    return jsonify(
        {"image_id": image_id, "gemma": gemma_result, "gemini": gemini_result}
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

    with ThreadPoolExecutor(max_workers=2) as executor:
        gemma_future = executor.submit(call_gemma_delivery, image_id, token, img_data, prompt)
        gemini_future = executor.submit(call_gemini_delivery, image_id, token, img_data, prompt)
        gemma_result = gemma_future.result()
        gemini_result = gemini_future.result()

    return jsonify(
        {"image_id": image_id, "gemma": gemma_result, "gemini": gemini_result}
    )


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
