from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def image_embeddings(image_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    return model.get_image_features(**inputs).detach().numpy()