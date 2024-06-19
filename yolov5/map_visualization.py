import folium
import webbrowser
import os
import socket
import base64
class pothole_visualization:
    def __init__(self):
        # 지도 생성 (위치: 서울, 줌 레벨: 12)
        self.map = folium.Map(location=[35.830615, 128.754465], zoom_start=14)  # 영남대학교 IT 관 기준
        # UDP 서버 설정
        self.UDP_IP = "165.229.185.185"  # 수신 IP 주소
        self.UDP_PORT = 8080  # 수신 포트 번호

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))

        # 이미지 저장할 경로 설정
        self.image_save_path = 'C:\\Users\\ANTL\\Desktop\\GitHub\\2024_EdgeComputing_Pothole\\yolov5\\img\\'

    # HTML 파일 저장 함수
    def save_map(self):
        self.map.save("map.html")
        print("map.html 파일이 생성되었습니다.")

        # HTML 파일에 새로고침 meta 태그 추가
        with open("map.html", "r") as file:
            html_content = file.read()

        refresh_meta_tag = '<meta http-equiv="refresh" content="10">'
        html_content = html_content.replace("<head>", f"<head>{refresh_meta_tag}")

        with open("map.html", "w") as file:
            file.write(html_content)

    # 지도에 마커 추가 함수
    def add_marker(self, location, popup, image):
        with open(image, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        # HTML로 변환된 base64 이미지 태그 생성
        image_html = f'<img src="data:image/jpeg;base64,{encoded_image}" width="300"><br>{popup}'

        # 마커 추가
        folium.Marker(
            location=location,  # 마커를 추가할 위치 (위도, 경도)
            popup=folium.Popup(image_html, max_width=300),  # 팝업에 이미지 추가
            tooltip='클릭하면 이미지가 나타납니다.'
        ).add_to(self.map)

    # UDP 수신 함수
    def receive_data(self):
        count = 0
        while count <=5:
            data, addr = self.sock.recvfrom(65536)
            message = data.decode("utf-8")
            latitude, longitude, popup = message.split(',')
            print(f"Latitude:{latitude}, Longitude:{longitude}, Time:{popup}")

            # 이미지 데이터 수신
            img_data, addr = self.sock.recvfrom(65536)

            # 이미지 데이터를 파일로 저장
            image_filename = f"received_image_{popup.strip()}.jpg"  # 예시로 jpg 확장자 사용
            save_path = os.path.join(self.image_save_path, image_filename)

            with open(save_path, 'wb') as f:
                f.write(img_data)

            # 마커 추가
            location = [float(latitude), float(longitude)]
            self.add_marker(location, popup, save_path)
            count +=1

def main():
    pv = pothole_visualization()
    pv.receive_data()
    pv.map.save("map.html")
    webbrowser.open("map.html")  # 프로그램 시작 시 한 번만 호출하여 지도를 열도록 변경

if __name__ == "__main__":
    main()
