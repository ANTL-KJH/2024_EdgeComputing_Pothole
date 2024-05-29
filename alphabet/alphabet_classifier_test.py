import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 저장된 모델 로드
model = load_model('image_classifier_model.h5')

# 클래스 이름 리스트
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

def load_and_prep_image(img_path, img_size=150):
    """이미지를 로드하고 전처리"""
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # 스케일링
    return img_array

def predict_image(model, img_path):
    """이미지 경로를 받아서 예측"""
    img_array = load_and_prep_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# 테스트할 이미지 경로 설정
test_image_path = 'C:\\Users\\mch2d\Desktop\\GitHub\\2024_EdgeComputing_Pothole\\alphabet\\dog.jpg'  # 테스트할 이미지 파일 경로로 변경

# 예측 수행
predicted_class = predict_image(model, test_image_path)
print(f"The predicted class is: {predicted_class}")

# 이미지 시각화
img = image.load_img(test_image_path)
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()
