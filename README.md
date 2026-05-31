# AI Image Classifier

## Project Overview

AI Image Classifier is a full-stack web application that classifies uploaded images using a pretrained MobileNetV2 model.

The goal of this project is not to train a model from scratch. Instead, the goal is to practice how a pretrained model can be integrated into a product-like workflow:

1. A user uploads an image from the browser.
2. The frontend sends the image to the backend API.
3. The backend validates and preprocesses the image.
4. A pretrained MobileNetV2 model runs inference.
5. The backend returns the predicted label, confidence score, and top predictions.
6. The frontend displays the result.

This project is designed as a small AI productization demo, focusing on model input/output contracts, preprocessing consistency, and practical integration between frontend and backend.

## Repository Description

Full-stack image classifier demo using React, FastAPI, PyTorch, Pillow, and pretrained MobileNetV2 inference.

## Tech Stack

- Frontend: React, TypeScript, Vite
- Backend: Python, FastAPI
- Image preprocessing: PIL / Pillow
- Model inference: PyTorch + torchvision pretrained model
- Model: MobileNetV2, ImageNet class labels
- Deployment demo: Docker optional

## Architecture

```text
React + TypeScript frontend
        |
        | multipart/form-data image upload
        v
FastAPI backend
        |
        | Pillow image loading + RGB conversion
        v
torchvision preprocessing
        |
        | PyTorch tensor
        v
Pretrained MobileNetV2 model
        |
        | logits
        v
Softmax + top-k prediction
        |
        | JSON response
        v
Frontend result display

```

## API Contract

### Health Check

`GET /health`

### Predict Image

`POST /predict`

## Model Input / Output

### Model Input

The model does not receive a raw JPG or PNG file directly.

The uploaded image must first be converted into a PyTorch tensor that matches the format expected by the pretrained MobileNetV2 model.

Expected model input shape: `[batch, channel, height, width]`. For one RGB image, the tensor shape is similar to: `[1, 3, 224, 224]`. This means:

```
1 image
3 color channels: red, green, blue
224 pixels height
224 pixels width
```

### Model Output

The model outputs logits.

Logits are raw model scores for each class. For ImageNet classification, MobileNetV2 outputs scores for 1000 classes.

The backend then converts logits into confidence-like scores using softmax.

The output flow is:

```
logits
→ softmax probabilities
→ top-k class indices
→ ImageNet label names
→ JSON response
```

### API Output

The API returns:

```
predicted_label: the label with the highest confidence score
confidence: the confidence score of the top prediction
top_predictions: the top-k predicted labels and scores
```

## Preprocessing Steps

The backend preprocessing pipeline includes:

1. Read uploaded image bytes from the request.
2. Open the image with Pillow.
3. Convert the image to RGB.
4. Apply the preprocessing transform associated with the pretrained MobileNetV2 weights.
5. Convert the image into a PyTorch tensor.
6. Normalize the tensor using the pretrained weight settings.
7. Add a batch dimension.
8. Send the tensor into the model for inference.

Key preprocessing code:

```python
image = Image.open(BytesIO(image_bytes)).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)
```

- The pretrained MobileNetV2 model expects 3-channel RGB input. So I use `.convert("RGB")` to standardize uploaded images into the 3-channel RGB format expected by the pretrained model.
- `weights.transforms()` provides the preprocessing pipeline associated with the selected pretrained weights. This helps ensure that inference-time preprocessing is consistent with the model's training setup.
  - The transform may include: resize, center crop, conversion to tensor, normalization
  - Incorrect preprocessing may not crash the program, but it can make predictions inaccurate.
- After preprocessing, a single image tensor may have this shape: `[3, 224, 224]`. But the model expects a batch of images: `[batch, channel, height, width]`.
  - `unsqueeze(0)` adds the batch dimension:
    ```
    [3, 224, 224] → [1, 3, 224, 224]
    ```

## Inference Flow

The model inference code follows this flow:

```python
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
```

### model.eval()

This tells the model that it is being used for inference, not training.

### torch.no_grad()

During training, PyTorch tracks operations so it can calculate gradients and update model parameters.

During inference, the model is only making predictions. It does not need to update its parameters.

`torch.no_grad()` disables gradient tracking during inference.

### Logits

The output of the model is called logits. Logits are raw scores, not probabilities.
Example:

```
class A: 8.2
class B: 6.4
class C: -1.3
```

### Softmax

Softmax converts logits into confidence-like scores.

Example:

```
class A: 0.82
class B: 0.15
class C: 0.03
```

### Top-k Predictions

The backend returns the top predictions instead of only one result.

This is useful because it shows whether the model is confident or uncertain.

Example:

```
Top prediction: 0.91
Second prediction: 0.04
```

This suggests the model is relatively confident.

Example:

```
Top prediction: 0.34
Second prediction: 0.31
```

This suggests the model is less certain.

## Known Failure Cases

### Invalid File Type

If the uploaded file is not an image, the backend should reject it. For examples, PDF, TXT, ZIP, or CSV.

The frontend also restricts file selection with:

```tsx
accept = "image/\*";
```

However, frontend validation is only for user experience. Backend validation is still required.

### Corrupted Image

A file may have an image extension but contain invalid or corrupted data.

Pillow may fail to open the file.

### RGB Conversion May Lose Information

This demo converts uploaded images to RGB because the pretrained MobileNetV2 model expects 3-channel RGB input.

However, RGB conversion may lose information in some cases:

- RGBA images lose alpha transparency
- Grayscale images are expanded into RGB
- CMYK images may shift colors during conversion
- Specialized images may contain channels that should not be discarded

For this ImageNet-based demo, RGB conversion is reasonable. For production use, the model's expected input format should always be confirmed.

### Incorrect Preprocessing

If resize, crop, tensor conversion, or normalization does not match the pretrained model's expected preprocessing, predictions may become inaccurate even if the API does not crash.

### Wrong Tensor Shape

PyTorch vision models usually expect:

```
[batch, channel, height, width]
```

Common mistakes include passing:

```
[height, width, channel]
```

or forgetting the batch dimension.

### RGB vs. BGR Mismatch

Some image libraries use different channel orders.

Pillow usually works with RGB.

OpenCV commonly uses BGR.

If the model expects RGB but receives BGR, predictions can become incorrect even though the image shape looks valid.

### Label Mapping Errors

The model returns class indices, not human-readable labels.

If the index-to-label mapping is wrong, the displayed prediction will be wrong even if the model output is correct.

### Confidence Misinterpretation

Confidence scores are softmax scores over the model's known classes.

A high confidence score does not guarantee the prediction is correct.

For example, if the uploaded image is outside the model's training distribution, the model may still choose one of the known ImageNet classes with high confidence.

### Out-of-Distribution Inputs

The pretrained model was trained on ImageNet-style natural images.

It may not work well on:

- medical images
- satellite images
- diagrams
- screenshots
- abstract images
- domain-specific product images

The model will still return a prediction, but the result may not be meaningful.

### Large Image Files

Very large uploaded images may increase memory usage or processing time.

A production system should set file size limits and handle large inputs safely.

## Future Improvements

- Add Docker support for backend deployment.
- Add backend tests for valid images, invalid files, and corrupted files.
- Add file size limits.
- Add model warm-up on backend startup.
- Add better frontend result visualization.
- Add support for switching between MobileNetV2 and ResNet18.
- Add logging for inference time and errors.
- Add confidence threshold handling for uncertain predictions.
- Add production-ready CORS and environment-based API configuration.

## Note

### Start Python venv:

```bash
cd ~/AI-image-classifier/backend/
source .venv/bin/activate
```

### Start backend service

```bash
uvicorn main:app --reload
```

### Start frontend

```bash
cd ~/AI-image-classifier/frontend/
npm run dev
```
