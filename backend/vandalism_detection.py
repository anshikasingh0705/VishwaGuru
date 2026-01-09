from local_ml_service import detect_vandalism_local
from PIL import Image

def detect_vandalism(image: Image.Image):
    """
    Wrapper for vandalism detection using Local ML Service.
    """
    return detect_vandalism_local(image)
