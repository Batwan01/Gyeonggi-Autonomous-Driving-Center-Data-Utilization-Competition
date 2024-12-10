import os
import cv2
import albumentations as A
from tqdm import tqdm

# 입력 및 출력 디렉토리 설정
input_dir = "data/test"
output_dir = "data/rain_test"

# 출력 디렉토리가 없다면 생성
os.makedirs(output_dir, exist_ok=True)

# RandomRain 변환 정의
transform = A.Compose([
    A.RandomRain(brightness_coefficient=0.75, drop_width=1, blur_value=3, p=1),
])

# 입력 디렉토리의 모든 이미지 파일 처리
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in tqdm(image_files, desc="이미지 처리 중"):
    # 이미지 읽기
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 비 효과 적용
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    
    # RGB에서 BGR로 다시 변환
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    
    # 결과 이미지 저장
    output_path = os.path.join(output_dir, f"rain_{image_file}")
    cv2.imwrite(output_path, transformed_image)

print("모든 이미지 처리가 완료되었습니다.")