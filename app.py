from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import torch.nn as nn
import torch.hub
from transformers import AutoModel
import pandas as pd

app = Flask(__name__)
CORS(app)

df = pd.read_csv("Dataset/Split Dataset/Training_meme_dataset.csv")

TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"

IMAGE_FOLDER = "Dataset/Labelled Images" 

df["image_path"] = df["image_name"].apply(lambda x: os.path.join(IMAGE_FOLDER, x))
IMAGE_COLUMN = "image_path"

class MultiModalModel(nn.Module):
    def __init__(self, text_model_name, image_model_name, output_classes):
        super().__init__()
        
        # Select appropriate text model
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Select appropriate image model
        self.image_encoder = torch.hub.load("pytorch/vision", image_model_name, pretrained=True)
        self.image_encoder.fc = nn.Identity()

        # Final classifier
        self.fc_combined = nn.Linear(512 + 768 + 128, output_classes)

    def forward(self, text_tokens, image_tensor):
        text_features = self.text_encoder(**text_tokens).last_hidden_state[:, 0, :]
        image_features = self.image_encoder(image_tensor)
        
        combined = torch.cat((text_features, image_features), dim=1)
        return self.fc_combined(combined)

# 加载训练好的模型
model = MultiModalModel(
    text_model_name="bert-base-uncased",
    image_model_name="resnet50",
    output_classes=len(df[LABEL_COLUMN].unique()) 
)

if os.path.exists("multi_modal_model.pth"):
    model.load_state_dict(torch.load("multi_modal_model.pth", map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully")
else:
    print("Model file not found. Train and save the model first")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # 处理文本输入
    text_tokens = tokenizer(data["sentence"], return_tensors="pt", padding=True, truncation=True)

    # 处理图像输入
    if os.path.exists(data["image_path"]):
        image = Image.open(data["image_path"]).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
    else:
        return jsonify({"error": "Image file not found"}), 400

    # 进行预测
    with torch.no_grad():
        prediction = model(text_tokens, image_tensor)

    response = {"prediction": torch.argmax(prediction, dim=1).item()}
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)

