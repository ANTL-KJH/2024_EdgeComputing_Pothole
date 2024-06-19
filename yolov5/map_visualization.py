import folium
import webbrowser
import os
import socket
import threading

class pothole_visualization:
    def __init__(self):
        # 지도 생성 (위치: 서울, 줌 레벨: 12)
        self.map = folium.Map(location=[35.830615, 128.754465], zoom_start=14)  # 영남대학교 IT 관 기준
        # UDP 서버 설정
        self.UDP_IP = "165.229.185.185"  # 수신 IP 주소
        self.UDP_PORT = 8080  # 수신 포트 번호

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))

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

        # HTML 파일 열기
        webbrowser.open("map.html")

    # 지도에 마커 추가 함수
    def add_marker(self, location, popup, image):
        popup_content = f'<img src="{image}" alt="{popup}" width="150" height="100"><br>{popup}'
        popup = folium.Popup(popup_content, max_width=200)
        folium.Marker(
            location=location,
            popup=popup,
        ).add_to(map)
        self.save_map()


    # UDP 수신 함수
    def receive_data(self):
        while True:
            data, addr = self.sock.recvfrom(65536)
            message = data.decode("utf-8")
            first_comma_idx = message.find(',')
            if first_comma_idx != -1:
                latitude = message[:first_comma_idx]
                remaining_message = message[first_comma_idx + 1:]

                # 두번째 쉼표 위치 찾기
                second_comma_idx = remaining_message.find(',')
                if second_comma_idx != -1:
                    longitude = remaining_message[:second_comma_idx]
                    remaining_message = remaining_message[second_comma_idx + 1:]

                    # 세번째 쉼표 위치 찾기
                    third_comma_idx = remaining_message.find(',')
                    if third_comma_idx != -1:
                        popup = remaining_message[:third_comma_idx]
                        recv_img = remaining_message[third_comma_idx + 1:]
                    else:
                        # 예외 처리: 세번째 쉼표를 찾을 수 없는 경우
                        print("Error: Third comma not found in message.")
                        latitude, longitude, popup, recv_img = None, None, None, None
                else:
                    # 예외 처리: 두번째 쉼표를 찾을 수 없는 경우
                    print("Error: Second comma not found in message.")
                    latitude, longitude, popup, recv_img = None, None, None, None
            else:
                # 예외 처리: 첫번째 쉼표를 찾을 수 없는 경우
                print("Error: First comma not found in message.")
                latitude, longitude, popup, recv_img = None, None, None, None
            location = [float(latitude), float(longitude)]
            image_filename = f"received_image_{popup}.jpg"  # 예시로 jpg 확장자 사용
            image_path = os.path.join("C:\\Users\\ANTL\\Desktop\\GitHub\\2024_EdgeComputing_Pothole\\yolov5\\img", image_filename)  # 저장할 디렉토리 경로 설정

            with open(image_path, 'wb') as f:
                f.write(recv_img)  # recv_img를 파일에 쓰기

            self.add_marker(location, popup, image_path)





def main():
    pv = pothole_visualization()
    thread = threading.Thread(target=pv.receive_data)
    thread.start()
    pv.save_map()


if __name__=="__main__":
    main()