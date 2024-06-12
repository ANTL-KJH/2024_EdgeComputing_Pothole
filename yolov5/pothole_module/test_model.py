import torch

# 모델 로딩 테스트
def load_model(model_path):
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print("모델 로딩 성공")
        return model
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None

# 예측 테스트
def test_model(model, input_tensor):
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        print("예측 성공")
        return output
    except Exception as e:
        print(f"예측 실패: {e}")
        return None

# 메인 테스트 코드
if __name__ == "__main__":
    # 모델 경로
    yolov5s_model_path = "yolov5s.pt"
    best_model_path = "best.pt"

    # 임의의 입력 텐서 생성 (batch_size=1, channels=3, height=640, width=640)
    input_tensor = torch.randn(1, 3, 640, 640)

    # yolov5s.pt 모델 테스트
    yolov5s_model = load_model(yolov5s_model_path)
    if yolov5s_model:
        test_model(yolov5s_model, input_tensor)

    # best.pt 모델 테스트
    best_model = load_model(best_model_path)
    if best_model:
        test_model(best_model, input_tensor)
