import torch
from torch import nn
from torchvision import models, transforms
import os
import cv2
from PIL import Image

def create_transform_pipeline(is_training=True):
    """
    Generate a sequence of image transformations based on training/testing mode.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def setup_resnet_model(output_classes, dropout_prob=0.5):
    """
    Configure a ResNet18 model with a custom classification head.
    """
    base_model = models.resnet18(pretrained=True)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Dropout(dropout_prob),
        nn.Linear(num_features, output_classes)
    )
    return base_model

def annotate_image(image_path, score_values, total_value):
    """
    Annotates an image with rectangles and associated values.

    Args:
        image_path (str): path to target image.
        rectangle_values (list): numeric values corresponding to rectangles.
        total_label (str): text to annotate total value.
    """
    score_coords = [
        (138, 70, 184, 104),
        (185, 70, 231, 104),
        (232, 70, 278, 104),
        (279, 70, 325, 104),
        (326, 70, 372, 104),
        (373, 70, 419, 104)
    ]

    point_coords = [
        (138, 40, 184, 74),
        (185, 40, 231, 74),
        (232, 40, 278, 74),
        (279, 40, 325, 74),
        (326, 40, 372, 74),
        (373, 40, 419, 74)
    ]

    points = [4.0, 6.0, 10.0, 5.0, 10.0, 10.0]

    question_coords = [
        (138, 10, 184, 44),
        (185, 10, 231, 44),
        (232, 10, 278, 44),
        (279, 10, 325, 44),
        (326, 10, 372, 44),
        (373, 10, 419, 44)
    ]

    question_no = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    score_colors = [
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0)
    ]

    point_colors = [
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255)
    ]

    question_colors = [
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0)
    ]

    img = cv2.imread(image_path)

    if img is None:
        print(f"Image not found at: {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, (x1, y1, x2, y2) in enumerate(question_coords):
        color = question_colors[idx % len(question_colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
        cv2.putText(img, str(question_no[idx]), (x1 + 20, y1 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=2)

    for idx, (x1, y1, x2, y2) in enumerate(point_coords):
        color = point_colors[idx % len(point_colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
        cv2.putText(img, str(points[idx]), (x1 + 20, y1 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 100, 0), thickness=2)
        
    for idx, (x1, y1, x2, y2) in enumerate(score_coords):
        color = score_colors[idx % len(score_colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(img, str(score_values[idx]), (x1 + 20, y1 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=2)
        
    x_last, y_last, _, _ = score_coords[-1]
    cv2.putText(img, f"{total_value}", (x_last + 65, y_last + 20),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=2)

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, img)
    print(f"Annotated image saved at: {save_path}")

def predict_single_crop(model, cropped_img, preprocess, device):
    """
    Predict the class label for a cropped image.

    Returns:
        Predicted class index.
    """
    cropped_tensor = preprocess(cropped_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(cropped_tensor)
        _, prediction = torch.max(output, 1)
    return prediction.item()

def evaluate_image_scorecard(model, img_path, box_coords, preprocess, device):
    """
    Evaluate and predict values from predefined regions of a scorecard image.

    Args:
        model: Trained PyTorch model.
        img_path: Path to the input scorecard image.
        box_coords: List of rectangle coordinates (x1, y1, x2, y2).
        preprocess: Image preprocessing function.
        device: Execution device (CPU/GPU).

    Returns:
        Total sum of predicted values and an array of predicted values.
    """
    img = Image.open(img_path).convert("RGB")
    predictions = []
    total = 0

    for coord in box_coords:
        cropped = img.crop(coord)
        predicted_value = predict_single_crop(model, cropped, preprocess, device) / 2
        predictions.append(predicted_value)
        total += predicted_value

    return total, predictions

device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transform = create_transform_pipeline(is_training=False)

classifier = setup_resnet_model(output_classes=21)
classifier.load_state_dict(torch.load("./resenet_model.pth", weights_only=True))
classifier.to(device_type)

for index in range(1, 57):
    image_file = os.path.join("../score_tables", f"image{index}.png")
    score_coords = [
        (138, 70, 184, 104),
        (185, 70, 231, 104),
        (232, 70, 278, 104),
        (279, 70, 325, 104),
        (326, 70, 372, 104),
        (373, 70, 419, 104)
    ]

    total_prediction, predictions_array = evaluate_image_scorecard(
        classifier, image_file, score_coords, test_transform, device_type
    )

    annotate_image(image_file, predictions_array, total_prediction)
    print(f"image{index}.png -> Total: {total_prediction}")
