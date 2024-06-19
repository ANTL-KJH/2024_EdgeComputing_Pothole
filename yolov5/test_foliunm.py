import folium
import base64

# 이미지 파일을 base64로 인코딩
image_path = 'C:\\Users\\ANTL\\Desktop\\GitHub\\2024_EdgeComputing_Pothole\\yolov5\\img\\received_image_20240619-180718.jpg'
with open(image_path, 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# HTML로 변환된 base64 이미지 태그 생성
image_html = f'<img src="data:image/jpeg;base64,{encoded_image}" width="300">'

# 지도 생성 (중심 좌표와 줌 레벨 설정)
m = folium.Map(location=[37.5665, 126.9780], zoom_start=12)

# 마커 추가
folium.Marker(
    location=[37.5665, 126.9780],  # 마커를 추가할 위치 (위도, 경도)
    popup=folium.Popup(image_html, max_width=300),  # 팝업에 이미지 추가
    tooltip='클릭하면 이미지가 나타납니다.'
).add_to(m)

# 지도 저장 (HTML 파일로 저장)
m.save('map_with_image_marker.html')

# 지도 출력 (주피터 노트북에서 실행 시)
m
