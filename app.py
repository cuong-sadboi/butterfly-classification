import os
import numpy as np
import torch
import gradio as gr
from PIL import Image
from matplotlib import cm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import build_model


def find_weights():
    candidates = [
        "butterfly_model.pth",
        os.path.join("models", "resnet18_butterfly.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find model weights. Expected butterfly_model.pth or models/resnet18_butterfly.pth"
    )


def get_class_names(test_dir="dataset/test"):
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing folder: {test_dir}")
    names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    if not names:
        raise RuntimeError(f"No class folders found in {test_dir}")
    return names


def load_model(model_name="resnet18", test_dir="dataset/test"):
    class_names = get_class_names(test_dir)
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    weights_path = find_weights()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return model, class_names, transform, device


model, class_names, transform, device = load_model()


def get_cam_target_layer(model):
    # Support the common timm backbones used for image classification.
    if hasattr(model, "layer4") and len(model.layer4) > 0:
        return model.layer4[-1]
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        return model.blocks[-1]
    if hasattr(model, "features"):
        features = model.features
        if isinstance(features, torch.nn.Sequential) and len(features) > 0:
            return features[-1]

    raise RuntimeError("Could not infer a target layer for Grad-CAM.")


cam_target_layer = get_cam_target_layer(model)


def build_cam_overlay(image: Image.Image, cam_map: np.ndarray, alpha=0.45):
    resized = image.resize((224, 224)).convert("RGB")
    img_np = np.asarray(resized).astype(np.float32) / 255.0

    cam_map = np.clip(cam_map, 0.0, 1.0)
    heatmap = cm.get_cmap("jet")(cam_map)[..., :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)

    return Image.fromarray((overlay * 255).astype(np.uint8))


def predict_butterfly(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().tolist()

    pred_idx = int(torch.tensor(probs).argmax().item())
    targets = [ClassifierOutputTarget(pred_idx)]

    with GradCAM(model=model, target_layers=[cam_target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    cam_image = build_cam_overlay(image, grayscale_cam)

    pred_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return pred_dict, cam_image


interface = gr.Interface(
    fn=predict_butterfly,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediction"),
        gr.Image(type="pil", label="Grad-CAM"),
    ],
    title="Butterfly Classification",
    description="Upload a butterfly image to predict its species.",
)


if __name__ == "__main__":
    interface.launch()
