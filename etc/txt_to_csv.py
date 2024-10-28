import os
import pandas as pd

# TXT 파일이 있는 폴더 경로
folder_path = '/root/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/etc/predictions'

# 파일명과 내용을 저장할 리스트 초기화
data = []

# 폴더 내의 모든 txt 파일을 순회
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as file:
            # 파일 내용을 공백으로 줄바꿈 대체 후 저장
            content = file.read().replace('\n', ' ').strip()
            data.append({'test_img': filename, 'prediction': content})

# DataFrame으로 변환 후 CSV 저장
df = pd.DataFrame(data)
df.to_csv('/root/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/etc/output.csv', index=False)
