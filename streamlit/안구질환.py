#라이브러리 import
#필요한 경우 install
import streamlit as st
import os
from streamlit_cropper import st_cropper
from PIL import Image
import pandas as pd
import numpy as np
from io import StringIO
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as img
import statsmodels.api as sm
import time
from streamlit_option_menu import option_menu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR
import sklearn
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

#########################################중요###########################################
# cd C:/Users/sook7/Desktop/CUAI 프로젝트/반려동물 안구질환 탐지/streamlit
# 터미널에서 명령어(streamlit run 안구질환.py)를 실행 시켜주어야 로컬에서 스트림릿이 작동함

## 가상환경 설정 필요
# 활성화 : venv\Scripts\activate
# 비활성화 : venv\Scripts\deactivate

#######################################################################################
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=2, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

# 모델 클래스 정의
class SingleImageEfficientNet(nn.Module):
    def __init__(self, output_dim):
        super(SingleImageEfficientNet, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()  # 마지막 classifier 레이어 제거 (특징만 추출)
        self.dropout = nn.Dropout(0.2)  # 드롭아웃 비율 0.2 설정
        self.batch_norm = nn.BatchNorm1d(1280 + 8)  # Batch Normalization layer 추가, 원-핫 인코딩 벡터 차원만큼 증가
        self.fc = nn.Linear(1280 + 8, output_dim)  # 모델의 출력을 FC layer 통과, 원-핫 인코딩 벡터 차원만큼 증가

    def forward(self, x, pos_vector):
        x = self.model(x)  # 모델에 이미지를 전달하여 특징 추출
        x = x.view(x.size(0), -1)
        x = torch.cat((x, pos_vector), dim=1)  # 특징과 원-핫 인코딩 벡터를 연결
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.batch_norm(x)  # Batch Normalization 적용
        x = self.fc(x)  # 특징을 FC layer 통과
        return x

# Depthwise Separable Convolution
class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Basic Conv2d
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# MobileNetV1
class MobileNet(nn.Module):
    def __init__(self, width_multiplier, num_classes=2, init_weights=True):
        super().__init__()
        self.init_weights=init_weights
        alpha = width_multiplier

        self.conv1 = BasicConv2d(3, int(32*alpha), 3, stride=2, padding=1)
        self.conv2 = Depthwise(int(32*alpha), int(64*alpha), stride=1)
        # down sample
        self.conv3 = nn.Sequential(
            Depthwise(int(64*alpha), int(128*alpha), stride=2),
            Depthwise(int(128*alpha), int(128*alpha), stride=1)
        )
        # down sample
        self.conv4 = nn.Sequential(
            Depthwise(int(128*alpha), int(256*alpha), stride=2),
            Depthwise(int(256*alpha), int(256*alpha), stride=1)
        )
        # down sample
        self.conv5 = nn.Sequential(
            Depthwise(int(256*alpha), int(512*alpha), stride=2),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
            Depthwise(int(512*alpha), int(512*alpha), stride=1),
        )
        # down sample
        self.conv6 = nn.Sequential(
            Depthwise(int(512*alpha), int(1024*alpha), stride=2)
        )
        # down sample
        self.conv7 = nn.Sequential(
            Depthwise(int(1024*alpha), int(1024*alpha), stride=2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(int(1024*alpha), num_classes)

        # weights initialization
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    # weights initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def mobilenet(alpha=1, num_classes = 2):
    return MobileNet(alpha, num_classes)


@st.cache_resource
def load_model(path, load_full_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #root_dir = 'binineni/cuai-cv-project/main/models/'
    root_dir = 'streamlit/models/'

    if load_full_model:
        model = torch.load(root_dir + path, map_location=torch.device('cpu'))
    else:
        # If only model parameters are saved
        model = mobilenet(alpha=1).to(device)
        model.load_state_dict(torch.load(root_dir + path, map_location=device))

    return model

def get_prediction(model, original_image_path, cropped_image_path, use_cropper=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # Load the original image or cropped image based on the condition
    image_path = cropped_image_path if use_cropper else original_image_path

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((76, 76)),  # 이미지 사이즈 통일
        transforms.ToTensor(),  # 이미지를 PyTorch tensor로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 업로드된 이미지를 PIL 이미지로 변환
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    pred = output.argmax(dim=1, keepdim=True)
    prob = F.softmax(output, dim=1)[0]

    return pred, prob


# main
# layout = wide : 화면 설정 디폴트값을 와이드로
st.set_page_config(page_title="안구질환")

st.title("안구질환 예측 프로토타입")
st.divider()

st.sidebar.title("이미지 파일 업로드")

model_paths = [
    'pinkeye_MobileNet_model_full.pth', #결막염
    'ker_MobileNet_model_full.pth', #색소침착성각막질환
    'ent_MobileNet_model_full.pth', #안검내반증
    'ble_MobileNet_model.pth', #안검염
    'tumor_MobileNet_model_full.pth', #안검종양
    'epi_MobileNet_model_full.pth', #유루증
    'ns_MobileNet_model_full.pth' #핵경화
]

# Load models using st.cache
pinkeye_model = load_model(model_paths[0])
ker_model = load_model(model_paths[1], load_full_model=False)
ent_model = load_model(model_paths[2], load_full_model=False)
ble_model = load_model(model_paths[3])
tumor_model = load_model(model_paths[4])
epi_model = load_model(model_paths[5])
ns_model = load_model(model_paths[6])

#모델 리스트로 관리
models = [pinkeye_model, ker_model, ent_model,
          ble_model, tumor_model, epi_model, ns_model
]

# Define disease names corresponding to each model
disease_names = [
    "각막염",
    "색소침착성각막염",
    "안검내반증",
    "안검염",
    "안검종양",
    "유루증",
    "핵경화"
]

#Constant
cropped_image_path = "data/cropped_image.jpg"
img_path = "data/upload.jpg"

def main():
    # Upload an image and set some options for demo purposes
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])

    st.sidebar.title("Cropper")
    use_cropper = st.sidebar.checkbox(label="Use Cropper", value=False)

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # 모델 예측에 사용하기 위해 image 저장
        image.save(img_path, format='JPEG')

        if use_cropper:
            st.divider()
            st.info("검사하고 싶은 눈 부위를 확대해 주세요.")
            box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
            aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
            aspect_dict = {
                "1:1": (1, 1),
                "16:9": (16, 9),
                "4:3": (4, 3),
                "2:3": (2, 3),
                "Free": None
            }
            aspect_ratio = aspect_dict[aspect_choice]

            # Get a cropped image from the frontend
            cropped_image = st_cropper(image, realtime_update=True, box_color=box_color,
                                       aspect_ratio=aspect_ratio)
            st.write("Preview")
            _ = cropped_image.thumbnail((150, 150))
            st.image(cropped_image)

            # cropped image 저장
            cropped_image.save(cropped_image_path, format='JPEG')

        else:
            st.caption("이미지를 Crop하시려면, Cropper 체크박스를 클릭해 주세요.")

    # 예측 실행 버튼
    st.sidebar.title("Predict")
    pred_button = st.sidebar.button("Predict!")
    pred_list = []
    prob_list = []  # 라벨이 0일 확률(병 없을 확률)

    if pred_button:
        try:
            if 'cropped_image' in locals() and use_cropper:  # Check if cropped_image exists and cropper is used
                for model in models:
                    pred, prob = get_prediction(model, img_path, cropped_image_path, use_cropper=True)
                    pred_list.append(pred)
                    prob_list.append(prob)
            else:
                for model in models:
                    pred, prob = get_prediction(model, img_path, cropped_image_path, use_cropper=False)
                    pred_list.append(pred)
                    prob_list.append(prob)
        except Exception as e:
            print("error : ", e)

        #예측 라벨이 1인 질병 저장
        suspicious_diseases = [disease_names[i] for i, pred in enumerate(pred_list) if pred.item() == 1]

        if suspicious_diseases:
            st.divider()
            st.error("다음과 같은 안구질환이 의심됩니다.", icon="🚨")
            st.title("")
            # Display a list of suspicious diseases with improved formatting
            for i, disease in enumerate(suspicious_diseases):
                st.markdown(f"""<span style="font-size: 25px; color: #333333;"> - **{disease}**</span>""", unsafe_allow_html=True)
            st.title("")
            # You can add additional information or styling as needed
            st.info("이 결과는 의학적인 조언이나 진단을 대신하지 않습니다. 전문의와 상담해 주세요.")
        else:
            st.divider()
            st.info("건강한 상태입니다.")


if __name__ == "__main__":
    main()

