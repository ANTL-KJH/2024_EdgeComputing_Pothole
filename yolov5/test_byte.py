string_data = "\xff\xd8\xff\xe0\x00\x10JFIF"  # 예시로 주어진 문자열
byte_data = string_data.encode('latin1')  # 문자열을 latin1 인코딩으로 바이트로 변환

print(byte_data)
