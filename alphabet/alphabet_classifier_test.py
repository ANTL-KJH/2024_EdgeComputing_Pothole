import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

# 저장된 모델 로드
model = load_model('image_classifier_model.h5')

# 클래스 이름 리스트
class_names = ['alpha_class_a', 'alpha_class_b', 'alpha_class_c', 'alpha_class_d', 'alpha_class_e','alpha_class_f','alpha_class_g','alpha_class_h','alpha_class_i','alpha_class_j','alpha_class_k','alpha_class_l','alpha_class_m','alpha_class_n','alpha_class_o','alpha_class_p','alpha_class_q','alpha_class_r','alpha_class_s','alpha_class_t','alpha_class_u','alpha_class_v','alpha_class_w','alpha_class_x','alpha_class_y','alpha_class_z']

def load_and_prep_image(img, crop_coords=None, img_size=150):
    """이미지를 로드하고 전처리 (크롭 포함)"""
    if crop_coords:
        img = img.crop(crop_coords)
        
    img = img.resize((img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # 스케일링 
    return img_array

def predict_image(model, img, crop_coords=None):
    """이미지 경로를 받아서 예측 (크롭 포함)"""
    img_array = load_and_prep_image(img, crop_coords)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# 이미지 회전 함수
def rotate_image(img, angle):
    """이미지를 특정 각도로 회전"""
    return img.rotate(angle, expand=True)

# 테스트할 이미지 경로 설정
test_image_path = 'C:\\Users\\mch2d\\Desktop\\GitHub\\2024_EdgeComputing_Pothole\\alphabet\\test_m.jpg'  # 테스트할 이미지 파일 경로로 변경

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
