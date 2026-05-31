from io import BytesIO

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval()

preprocess = weights.transforms() # resize, center crop, convert image to tensor, normalize with specific mean/std
categories = weights.meta["categories"]


def predict_image(image_bytes: bytes, top_k: int = 3):
    image = Image.open(BytesIO(image_bytes)).convert("RGB") # input standardization
    input_tensor = preprocess(image).unsqueeze(0) # Add 1 dimension for batch size at 0th position

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = [
        {
            "label": categories[index.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, index in zip(top_probs, top_indices)
    ]

    return {
        "predicted_label": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
    }