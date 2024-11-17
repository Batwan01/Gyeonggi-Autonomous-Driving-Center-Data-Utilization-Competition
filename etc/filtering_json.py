import json

# "special_vehicles" 항목을 삭제하는 함수
def delete_special_vehicles(item, img_id):
    for ann in item["annotations"]:
        if ann.get("lbl_nm") == "special_vehicles":
            print(f"[delete - img_id: {img_id}, lbl_nm: {ann.get('lbl_nm')}]")
    item["annotations"] = [
        ann for ann in item["annotations"]
        if ann.get("lbl_nm") != "special_vehicles"
    ]

# "truck"을 "special_vehicles"로 변경하는 함수
def modify_truck_to_special_vehicles(item, img_id):
    for ann in item["annotations"]:
        if ann.get("lbl_nm") == "truck":
            print(f"[modify - img_id: {img_id}, lbl_nm: truck -> special_vehicles]")
            ann["lbl_nm"] = "special_vehicles"

# "human" 항목을 삭제하는 함수
def delete_human(item, img_id):
    for ann in item["annotations"]:
        if ann.get("lbl_nm") == "human":
            print(f"[delete - img_id: {img_id}, lbl_nm: {ann.get('lbl_nm')}]")
    item["annotations"] = [
        ann for ann in item["annotations"]
        if ann.get("lbl_nm") != "human"
    ]

# 메인 처리 함수
def process_annotations(data):
    for item in data:
        # 각 item이 dict 형태인지 확인하고, images와 annotations 키가 있는지 확인
        if isinstance(item, dict) and "images" in item and "annotations" in item:
            img_id = int(item["images"].get("img_id", 0))

            # img_id 조건에 맞춰 작업 수행
            if 698 <= img_id <= 749:
                delete_special_vehicles(item, img_id)

            if 5043 <= img_id <= 5092:
                modify_truck_to_special_vehicles(item, img_id)

            if 11235 <= img_id <= 12048:
                delete_human(item, img_id)

    return data

# 파일을 읽고 처리 후 저장하는 함수
def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # annotations 처리
    data = process_annotations(data)

    # 결과를 새로운 JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"필터링 및 수정 완료. {output_file} 파일이 생성되었습니다.")

# 실행
process_file("train.json", "filtered_train.json")
