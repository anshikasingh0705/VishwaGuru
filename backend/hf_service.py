import os
import io
import httpx
from PIL import Image
import asyncio

# HF_TOKEN is optional for public models but recommended for higher limits
token = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}"} if token else {}
API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

async def query_hf_api(image_bytes, labels, client=None):
    """
    Queries Hugging Face API using a shared or new HTTP client.
    """
    if client:
        return await _make_request(client, image_bytes, labels)

    async with httpx.AsyncClient() as new_client:
        return await _make_request(new_client, image_bytes, labels)

async def _make_request(client, image_bytes, labels):
    import base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        "inputs": image_base64,
        "parameters": {
            "candidate_labels": labels
        }
    }

    try:
        response = await client.post(API_URL, headers=headers, json=payload, timeout=20.0)
        if response.status_code != 200:
            print(f"HF API Error: {response.status_code} - {response.text}")
            return []
        return response.json()
    except Exception as e:
        print(f"HF API Request Exception: {e}")
        return []

async def detect_vandalism_clip(image: Image.Image, client: httpx.AsyncClient = None):
    """
    Detects vandalism/graffiti using Zero-Shot Image Classification with CLIP (Async).
    """
    try:
        labels = ["graffiti", "vandalism", "spray paint", "street art", "clean wall", "public property", "normal street"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        # Results format: [{'label': 'graffiti', 'score': 0.9}, ...]
        if not isinstance(results, list):
             return []

        vandalism_labels = ["graffiti", "vandalism", "spray paint"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in vandalism_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []

async def detect_infrastructure_clip(image: Image.Image, client: httpx.AsyncClient = None):
    try:
        labels = ["broken streetlight", "damaged traffic sign", "fallen tree", "damaged fence", "pothole", "clean street", "normal infrastructure"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        if not isinstance(results, list):
             return []

        damage_labels = ["broken streetlight", "damaged traffic sign", "fallen tree", "damaged fence"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in damage_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []

async def detect_flooding_clip(image: Image.Image, client: httpx.AsyncClient = None):
    try:
        labels = ["flooded street", "waterlogging", "blocked drain", "heavy rain", "dry street", "normal road"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        if not isinstance(results, list):
             return []

        flooding_labels = ["flooded street", "waterlogging", "blocked drain", "heavy rain"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in flooding_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []

async def detect_illegal_parking_clip(image: Image.Image, client: httpx.AsyncClient = None):
    try:
        labels = ["illegally parked car", "car blocking driveway", "car on sidewalk", "double parking", "parked car", "empty street", "traffic jam"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        if not isinstance(results, list):
             return []

        parking_labels = ["illegally parked car", "car blocking driveway", "car on sidewalk", "double parking"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in parking_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []

async def detect_street_light_clip(image: Image.Image, client: httpx.AsyncClient = None):
    try:
        labels = ["broken streetlight", "dark street", "street light off", "working streetlight", "daytime street"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        if not isinstance(results, list):
             return []

        light_labels = ["broken streetlight", "dark street", "street light off"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in light_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []

async def detect_fire_clip(image: Image.Image, client: httpx.AsyncClient = None):
    try:
        labels = ["fire", "smoke", "flames", "forest fire", "building fire", "normal street", "clear sky"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        if not isinstance(results, list):
             return []

        fire_labels = ["fire", "smoke", "flames", "forest fire", "building fire"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in fire_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []

async def detect_stray_animal_clip(image: Image.Image, client: httpx.AsyncClient = None):
    try:
        labels = ["stray dog", "stray cow", "stray cattle", "wild animal", "pet dog", "empty street"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        if not isinstance(results, list):
             return []

        animal_labels = ["stray dog", "stray cow", "stray cattle", "wild animal"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in animal_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []

async def detect_blocked_road_clip(image: Image.Image, client: httpx.AsyncClient = None):
    try:
        labels = ["fallen tree", "construction work", "road barrier", "traffic accident", "landslide", "clear road", "normal traffic"]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        results = await query_hf_api(img_bytes, labels, client=client)

        if not isinstance(results, list):
             return []

        block_labels = ["fallen tree", "construction work", "road barrier", "traffic accident", "landslide"]
        detected = []

        for res in results:
            if isinstance(res, dict) and res.get('label') in block_labels and res.get('score', 0) > 0.4:
                 detected.append({
                     "label": res['label'],
                     "confidence": res['score'],
                     "box": []
                 })
        return detected
    except Exception as e:
        print(f"HF Detection Error: {e}")
        return []
