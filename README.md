# 오픈소스AI 응용 - 2023년 2학기

## Title: 반려동물 피부질환 데이터를 활용한 피부병 진단 모델

### Dataset

AI-hub '반려동물 피부질환 데이터'

https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=561

### Tool

- Google collab
- wandb

### 요구사항

---

1. **AI-hub의 데이터 다운로드**

   a) [Dataset Link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=561) 클릭 후 회원가입 진행

   b) 라벨링 데이터 다운로드

   <img src='https://lh3.googleusercontent.com/fife/AGXqzDlXKYeduQdOFBs6XSfIOX-W32YZZI-q24LBTOskou2akekv1J3G8YIFv3JR9MTigGW4A4nL6DPjRihSBEJ8-SJDpYUuU-bUszDEd-tB_h8zljoWX5Eq9-fcu2f1ifXDVHd5k8-WVi9mBwhCVbdVorotW-fDpax8Euwtkn7ykEthVJfp6yVd7dvU9vOQCVd2YxdmekOTnh7jl271RsJE2XpH3EXTCTgiMC7OhfVUJqPf2206CytcgYnFYlaAkCsPI3gWKWJxYGyRKmJxcTVF9IfMWkzuOBA--hUFm-qttLAW16wWq0n4BUgASZC3WUWCU317kOWCi_S5xazpLOOVfBjFDwCdEpn0Vl-ZLPahutI-ak1jisFke7P-bEa8WYT_ZaDxndpDJ_QxIiTgz07axh4jNtt5mUpYksv3WHopWL_hFnI1Qwz6tC3jl1-JpfFeg5egIF1jxGdlbJOpYktc1xzDI_4ptTrne65JopS04rhNarwThuAdmU5GJpQYdVxSaq2at-0gtpPrE81nNVVszGLsN1mYYxTF-B5WMseugZLzFVYcS3dNdtu5LHsxCfuHbCz8o7YJgjZLUD9V_ubCuzNf6-HpseU_0eJFzvYpaiAY7AKsRNhN5oBNSIjsMJBaqo3riJlC8b2_BjJS3VKVn_DB0gjZusCe1RnscN3esRLyFwwgtAK0hNn0lgu2b3G60Xn8QeMa_2EcW_2mHk0ubY0R7owxAztoNOGGgxlD7g3zLVjEmlS1ace0R-_ujdThCZ23R3PriVgv6rJyEGfKmTHESQOakJWu0o641yieg7NZTKHwFxgImwqAxP07NaJgU4l5W0Hg6TwugHvI0zhOaYXApqUEsR6CKIY1cp24rSq7uERzN_cGY1X4jUuZQoNftFgEeZbEnD-gpAef3wcLrFuU6SlN9vcNagg_1SPqUAhKJWaFC9APqXPHAZ_z92k7grAcCyFmlqv0el_pzUa8PPYPfsbx6ztA0_Kmt-SJ61NPwfOa7u-K2o6mWihCCIaNZ9xfxDmvscBpaJjN3qwaBgbuIfBTjGD2BsAI73DVg4gQ7CFTmkFwWjtTJxlfVX-51M104f24DaTHLmSvJSf9ZQ6Rb3PxwgmJvAsRb3C1I4tZHPVa9x0zML6TA-Qn1z0B1ByGTXr_rJF6xcb81bJEEXDcf25kBaKDWaJuks6Ijo98KeEoHjADLtOsnQyV6LYntQWZoYinQwqFdi06RXZ8Tycx75MgtpbzwcTpb56XtyNbNjTgxihjjlU68XAZ_rgsJ57m8iRqkhO4hubx10qKzN2NL3GRmA1Izkl_je2xUyh1AEgpDIgViKz6Lb2MDwjs65zyCrccyfFOm6OcCHDQklSgwDpwZ7Lz-VUttXOzW4qVh5gMluTzqzjjFiw8B_UL-s_NKA83GedsJaN-1GtTBoniKWrxdGefPRMLFOIvn74z5jEHbYgkL1h0rZAeszM8RHlkO6jU2yLkANsmn7MtZlcKKV-qF1go5n2JSEuuw5GoVTcP5VRDiWPvuw=w2560-h1279' />

---

2. **wandb 홈페이지 회원가입**

---

3. **Google collab에 .ipynb 파일 업로드**

---

### 실행 방법

1. Import wandb
<pre> <code>
!pip install wandb
import wandb
!wandb login
</code></pre>

2. Import Libraries
<pre> <code>
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import os
import random
from PIL import Image, UnidentifiedImageError,ImageFile

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import json

from torchvision import transforms

</code></pre>

3. Data Pretreatment
<pre> <code>

# 압축 풀기 코드

'''
import zipfile
zipfile.ZipFile('drive/My Drive/openAI/data/data.zip').extractall('drive/My Drive/openAI/data')
'''

</code></pre>

<pre> <code>
# 원본 이미지 검증
'''
def validate_image(filepath):
    try:
        img = Image.open(filepath).convert('RGB')
        img.load()
    except UnidentifiedImageError:
        print(f'Corrupted Image is found at: {filepath}')
        return False
    except (IOError, OSError):
        print(f'Truncated Image is found at: {filepath}')
        return False
    else:
        return True
'''
</code></pre>
<pre> <code>
# 이미지 검증
'''
root = 'drive/My Drive/openAI/data/유증상_라벨_검증'

dirs = os.listdir(root)

for dir* in dirs:
folder_path = os.path.join(root, dir*)
files = os.listdir(folder_path)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # 파일 확장자 확인
        _, file_extension = os.path.splitext(file_name)

        if file_extension.lower() == '.jpg':
            valid = validate_image(file_path)
            if not valid:
                # 오류가 있는 이미지 파일 삭제
                os.remove(file_path)
        elif file_extension.lower() == '.json':
            # JSON 파일은 삭제하지 않음
            pass

'''

</code></pre>

4. Data Load

<pre> <code>

# json 파일에서 뽑아 사용하는 경우
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
class MyDataLoader(Dataset):
    def __init__(self, root_folder, batch_size, transform):
        self.root_folder = root_folder
        self.transform = transform
        self.batch_size = batch_size
        self.data = self.load_data()

    def load_data(self):
        data = []

        for root, dirs, files in os.walk(self.root_folder):
            for filename in files:
                if filename.endswith('.jpg'):
                    img_path = os.path.join(root, filename)

                    json_filename = os.path.splitext(filename)[0] + '.json'
                    json_file_path = os.path.join(root, json_filename)

                    if not os.path.exists(json_file_path):
                        continue

                    with open(json_file_path, 'r') as json_file:
                        metadata = json.load(json_file)

                    label = metadata['metaData']['lesions']
                    data.append((img_path, label))

        return data

    def __len__(self):
        return len(self.data)

    def label_convert(self,label):
        label_mapping = {"A2": 0, "A3": 1, "A4": 2, "A5": 3, "A6": 4}
        return label_mapping.get(label,-1)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label=self.label_convert(label)

        return image, label

</code></pre>

<pre> <code>

#이미지 데이터 가져오기
from torchvision import transforms
'''
#데이터 위치 지정
train_data = torchvision.datasets.ImageFolder(
    root = 'drive/My Drive/openAI/data/유증상_라벨_학습',
    transform= transforms.Compose([
      # 사이즈가 너무 커서 세션 다운되므로 줄임
      transforms.Resize([28,28]),
      # dataloader가 PIL 이미지를 수신하기 위해 이미지의 변형이 필요
      transforms.ToTensor()
    ])
)
test_data= torchvision.datasets.ImageFolder(
    root = 'drive/My Drive/openAI/data/유증상_라벨_검증',
    transform= transforms.Compose([
      transforms.Resize([28,28]),
      transforms.ToTensor()
    ])
)
print(train_data)
print(test_data)

#데이터를 배치 사이즈에 맞게 가져오기
#배치 사이즈=100

train_loader=MyDataLoader(train_data,batch_size=100)
test_loader=MyDataLoader(test_data,batch_size=100)
'''

train_root='drive/My Drive/openAI/data/유증상_라벨_학습'
test_root= 'drive/My Drive/openAI/data/유증상_라벨_검증'

transform=transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor()
])

train_loader = MyDataLoader(train_root, batch_size=100, transform=transform)

test_loader = MyDataLoader(test_root, batch_size=100, transform=transform)

</code></pre>

5. Define Model
<pre> <code>
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    #convolution
    self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),padding=1)
    self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1)
    #pooling
    self.pool=nn.MaxPool2d(kernel_size=(2,2))

    self.fc1=nn.Linear(64*7*7,128)
    self.fc2=nn.Linear(128,10)

def forward(self,x): # 각자 합성곱, 활성함수, pooling 적용
x=self.pool(F.relu(self.conv1(x)))
x=self.pool(F.relu(self.conv2(x)))
x=x.view(-1,64*7*7)
x=F.relu(self.fc1(x))
x=self.fc2(x)

    return F.softmax(x,dim=1)

</code></pre>

<pre> <code>
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 앞서 정의한 모델을 장치로 올림, 모델 정의
model = CNN().to(device)
</code></pre>
<pre> <code>
# 모델 학습을 위한 손실 함수와 최적화 함수
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
</code></pre>

6. Model Train
<pre><code>
#학습
for epoch in range(5):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0
  for i, data in enumerate(train_loader, 0):
      # Get the inputs; data is a list of [inputs, labels]
      #label 데이터 추출 후 담기
      inputs, labels= data

      if isinstance(labels, str):
        labels = torch.tensor([int(labels)]).to(device)

      inputs, labels= inputs.to(device), labels.to(device)

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)

      # Compute the loss
      loss = criterion(outputs, labels)

      # Backward pass and optimize
      loss.backward()
      optimizer.step()

      # Accuracy
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # log loss
      wandb.log({'train/loss':loss.item(), 'train/acc':correct/total})

      # Print statistics
      running_loss += loss.item()
      if i % 100 == 99:    # Print every 100 mini-batches
          print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
          running_loss = 0.0

model.eval()
with torch.no_grad():
running_loss = 0.0
correct = 0
total = 0
for i, data in enumerate(test_loader, 0):
#label 데이터 추출 후 담기
inputs, labels= data

      if isinstance(labels, str):
        labels = torch.tensor([int(labels)]).to(device)

      inputs, labels= inputs.to(device), labels.to(device)

      # Forward pass
      outputs = model(inputs)

      # Compute the loss
      loss = criterion(outputs, labels)

      # Accuracy
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      running_loss += loss.item()

      # log loss
      wandb.log({'test/loss':loss.item(), 'test/acc':correct/total})

      # Print statistics
      running_loss += loss.item()
      if i % 100 == 99:    # Print every 100 mini-batches
          print(f'Test Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
          running_loss = 0.0

print('Finished Training')
wandb.finish()
</code></pre>
