import cv2

# 이미지 파일 경로
image_path = 'test_c.jpg'

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지가 있는지 확인
if image is None:
    print("이미지를 읽을 수 없습니다.")
else:
    # 이미지 출력
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # 아무 키나 누를 때까지 대기
    cv2.destroyAllWindows()  # 창 닫기
