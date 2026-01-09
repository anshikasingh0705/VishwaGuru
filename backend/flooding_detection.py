from PIL import Image
from local_ml_service import detect_flooding_local

def detect_flooding(image: Image.Image):
    """
    Detects flooding in an image.
    Delegates to the Local ML Service.
    """
    return detect_flooding_local(image)
