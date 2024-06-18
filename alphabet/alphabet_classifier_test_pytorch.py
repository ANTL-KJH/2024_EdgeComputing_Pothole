import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib

matplotlib.use('TkAgg')
# 클래스 이름 리스트
class_names = ['alpha_class_a', 'alpha_class_b', 'alpha_class_c', 'alpha_class_d', 'alpha_class_e',
               'alpha_class_f', 'alpha_class_g', 'alpha_class_h', 'alpha_class_i', 'alpha_class_j',
               'alpha_class_k', 'alpha_class_l', 'alpha_class_m', 'alpha_class_n', 'alpha_class_o',
               'alpha_class_p', 'alpha_class_q', 'alpha_class_r', 'alpha_class_s', 'alpha_class_t',
               'alpha_class_u', 'alpha_class_v', 'alpha_class_w', 'alpha_class_x', 'alpha_class_y', 'alpha_class_z']

# 모델 초기화 및 상태 사전 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()  # 모델을 평가 모드로 설정

def load_and_prep_image(img, crop_coords=None, img_size=224):
    """이미지를 로드하고 전처리 (크롭 포함)"""
    if crop_coords:
        img = img.crop(crop_coords)
    img = img.resize((img_size, img_size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def predict_image(model, img, crop_coords=None):
    """이미지 경로를 받아서 예측 (크롭 포함)"""
    img_tensor = load_and_prep_image(img, crop_coords)
    with torch.no_grad():
        output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]
    return predicted_class

# 이미지 회전 함수
def rotate_image(img, angle):
    """이미지를 특정 각도로 회전"""
    return img.rotate(angle, expand=True)

# 테스트할 이미지 경로 설정
test_image_path = 'test/test_c.jpg'  # 테스트할 이미지 파일 경로로 변경

# 크롭할 이미지 로드
img = Image.open(test_image_path)
current_angle = 0  # 현재 각도 초기화

fig, ax = plt.subplots()
image_display = ax.imshow(img)

# 크롭 좌표 초기화
crop_coords = None

# RectangleSelector로 마우스로 크롭 영역 선택
def onselect(eclick, erelease):
    global crop_coords
    crop_coords = (int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))
    print(f"Crop coordinates: {crop_coords}")
    rect_selector.set_active(False)
    plt.close()

# 이미지 클릭 시 회전시키는 함수
def onclick(event):
    global img, current_angle, image_display, ax, rect_selector
    if event.button == 1:  # 마우스 왼쪽 버튼 클릭 시
        current_angle = (current_angle + 90) % 360
        img = rotate_image(img, 90)
        ax.clear()
        image_display = ax.imshow(img)
        rect_selector = RectangleSelector(ax, onselect,
                                          useblit=True,
                                          button=[3],  # 마우스 오른쪽 버튼으로 크롭 영역 선택
                                          minspanx=5,
                                          minspany=5,
                                          spancoords='pixels',
                                          props=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))
        fig.canvas.draw()

# RectangleSelector 설정
rect_selector = RectangleSelector(ax, onselect,
                                  useblit=True,
                                  button=[3],  # 마우스 오른쪽 버튼으로 크롭 영역 선택
                                  minspanx=5,
                                  minspany=5,
                                  spancoords='pixels',
                                  props=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))

fig.canvas.mpl_connect('button_press_event', onclick)  # 마우스 클릭 이벤트 연결

plt.show()

# 선택된 크롭 좌표 출력 및 예측 수행
if crop_coords:
    print(f"Selected crop coordinates: {crop_coords}")
    predicted_class = predict_image(model, img, crop_coords)
    print(f"The predicted class is: {predicted_class}")

    # 크롭된 이미지 시각화
    img_cropped = img.crop(crop_coords)
    plt.imshow(img_cropped)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()
else:
    print("No crop coordinates selected.")
