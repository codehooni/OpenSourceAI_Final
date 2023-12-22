# 오픈소스AI 응용 - 2023년 2학기

## Title: 반려동물 피부질환 데이터를 활용한 피부병 진단 모델

### Dataset

AI-hub '반려동물 피부질환 데이터'

https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=561

### Tool

- Google collab

### 요구사항

---

1. **AI-hub의 데이터 다운로드**

   a) [Dataset Link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=561) 클릭 후 회원가입 진행

   b) 라벨링 데이터 다운로드

   <img src='https://lh3.googleusercontent.com/fife/AGXqzDlXKYeduQdOFBs6XSfIOX-W32YZZI-q24LBTOskou2akekv1J3G8YIFv3JR9MTigGW4A4nL6DPjRihSBEJ8-SJDpYUuU-bUszDEd-tB_h8zljoWX5Eq9-fcu2f1ifXDVHd5k8-WVi9mBwhCVbdVorotW-fDpax8Euwtkn7ykEthVJfp6yVd7dvU9vOQCVd2YxdmekOTnh7jl271RsJE2XpH3EXTCTgiMC7OhfVUJqPf2206CytcgYnFYlaAkCsPI3gWKWJxYGyRKmJxcTVF9IfMWkzuOBA--hUFm-qttLAW16wWq0n4BUgASZC3WUWCU317kOWCi_S5xazpLOOVfBjFDwCdEpn0Vl-ZLPahutI-ak1jisFke7P-bEa8WYT_ZaDxndpDJ_QxIiTgz07axh4jNtt5mUpYksv3WHopWL_hFnI1Qwz6tC3jl1-JpfFeg5egIF1jxGdlbJOpYktc1xzDI_4ptTrne65JopS04rhNarwThuAdmU5GJpQYdVxSaq2at-0gtpPrE81nNVVszGLsN1mYYxTF-B5WMseugZLzFVYcS3dNdtu5LHsxCfuHbCz8o7YJgjZLUD9V_ubCuzNf6-HpseU_0eJFzvYpaiAY7AKsRNhN5oBNSIjsMJBaqo3riJlC8b2_BjJS3VKVn_DB0gjZusCe1RnscN3esRLyFwwgtAK0hNn0lgu2b3G60Xn8QeMa_2EcW_2mHk0ubY0R7owxAztoNOGGgxlD7g3zLVjEmlS1ace0R-_ujdThCZ23R3PriVgv6rJyEGfKmTHESQOakJWu0o641yieg7NZTKHwFxgImwqAxP07NaJgU4l5W0Hg6TwugHvI0zhOaYXApqUEsR6CKIY1cp24rSq7uERzN_cGY1X4jUuZQoNftFgEeZbEnD-gpAef3wcLrFuU6SlN9vcNagg_1SPqUAhKJWaFC9APqXPHAZ_z92k7grAcCyFmlqv0el_pzUa8PPYPfsbx6ztA0_Kmt-SJ61NPwfOa7u-K2o6mWihCCIaNZ9xfxDmvscBpaJjN3qwaBgbuIfBTjGD2BsAI73DVg4gQ7CFTmkFwWjtTJxlfVX-51M104f24DaTHLmSvJSf9ZQ6Rb3PxwgmJvAsRb3C1I4tZHPVa9x0zML6TA-Qn1z0B1ByGTXr_rJF6xcb81bJEEXDcf25kBaKDWaJuks6Ijo98KeEoHjADLtOsnQyV6LYntQWZoYinQwqFdi06RXZ8Tycx75MgtpbzwcTpb56XtyNbNjTgxihjjlU68XAZ_rgsJ57m8iRqkhO4hubx10qKzN2NL3GRmA1Izkl_je2xUyh1AEgpDIgViKz6Lb2MDwjs65zyCrccyfFOm6OcCHDQklSgwDpwZ7Lz-VUttXOzW4qVh5gMluTzqzjjFiw8B_UL-s_NKA83GedsJaN-1GtTBoniKWrxdGefPRMLFOIvn74z5jEHbYgkL1h0rZAeszM8RHlkO6jU2yLkANsmn7MtZlcKKV-qF1go5n2JSEuuw5GoVTcP5VRDiWPvuw=w2560-h1279' />

---

2. **Google collab에 .ipynb 파일 업로드**

---

### 실행 방법

1. Import Libraries
<pre> <code>
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
from tqdm import tqdm  # Import tqdm for progress bar

</code></pre>

2. Data Pretreatment
<pre> <code>

# 데이터 생성

'''
from PIL import Image
import os

# Set the input and output folders
input_folder = "/content/gdrive/MyDrive/openAI/data/유증상_검증/유증상_라벨"
output_folder = "유증상_검증"

# Desired resolution
new_resolution = (480, 360)

# Iterate through subfolders and resize images
for root, dirs, files in os.walk(input_folder):
    for file in files:
        # Check if the file is an image (you can add more image extensions if needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Create input and output paths
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))

            # Create output folder if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Open image, resize, and save to the output folder
            with Image.open(input_path) as img:
                img = img.resize(new_resolution, Image.ANTIALIAS)
                img.save(output_path)

print("Resizing complete.")

'''

</code></pre>
4. Data Load

<pre> <code>

# 이미지 데이터 불러오기
input_folder = "/content/drive/MyDrive/openAI/data/유증상_학습"
validation_folder = "/content/drive/MyDrive/openAI/data/유증상_검증"
</code></pre>

5. Define Model
<pre> <code>
class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        vgg_model = vgg16(pretrained=True)
        self.features = vgg_model.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

</code></pre>

<pre> <code>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 앞서 정의한 모델을 장치로 올림, 모델 정의
model = model.to(device)
</code></pre>
<pre> <code>

6. Model Train
<pre><code>
#학습
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': train_loss / total, 'accuracy': correct / total})
            pbar.update()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy * 100:.2f}%")
</code></pre>
