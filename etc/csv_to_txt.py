import pandas as pd
import os

# CSV 파일 경로
csv_file = '/root/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/etc/output.csv'
# TXT 파일을 저장할 폴더 경로
output_folder = '/root/Gyeonggi-Autonomous-Driving-Center-Data-Utilization-Competition/etc/txt_outputs'

# 출력 폴더가 없는 경우 생성
os.makedirs(output_folder, exist_ok=True)

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 각 행을 순회하며 txt 파일로 저장
for _, row in df.iterrows():
    filename = row['test_img'] + '.txt'  # 'test_img' 값을 파일 이름으로 사용
    content = row['prediction']
    
    # content를 공백 기준으로 분리한 후 6개씩 줄바꿈 추가
    values = content.split()
    formatted_content = '\n'.join([' '.join(values[i:i+6]) for i in range(0, len(values), 6)])
    
    # 파일 저장
    with open(os.path.join(output_folder, filename), 'w') as file:
        file.write(formatted_content)
