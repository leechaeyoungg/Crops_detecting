from ultralytics import YOLO
import cv2
import os

def split_image(image, tile_size):
    """이미지를 tile_size로 타일로 나누는 함수"""
    tiles = []
    h, w, _ = image.shape
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append((x, y, tile))
    return tiles

def merge_tiles(image, results_list, tile_size):
    """타일별 예측 결과를 원본 이미지에 다시 합치는 함수"""
    counts = {}  # 작물 종류와 개수 저장
    for (x, y, results) in results_list:
        for result in results:
            # 바운딩 박스와 라벨 정보 가져오기
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                x1 += x
                y1 += y
                x2 += x
                y2 += y
                class_id = box.cls[0].int().item()  # 탐지된 클래스 ID
                label = result.names[class_id]  # 클래스 이름

                # 바운딩 박스 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 작물 종류별 개수 카운팅
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1

    return image, counts

def draw_counts(image, counts):
    """탐지된 작물 종류와 개수를 이미지에 출력"""
    text = "\n".join([f"{label}: {count}" for label, count in counts.items()])
    
    # 텍스트를 이미지의 오른쪽 상단에 표시 (크기와 두께 조정)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # 텍스트 크기 증가
    color = (0, 0, 255)  # 빨간색
    thickness = 3  # 두께 증가
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    x = image.shape[1] - text_size[0] - 40  # 오른쪽 상단에 여유를 더 줌
    y = 50  # 텍스트 시작 위치

    # 텍스트 출력 (여러 줄일 경우 줄마다 표시)
    for i, line in enumerate(text.split("\n")):
        y_pos = y + i * (text_size[1] + 20)  # 줄 간격을 넓게 설정
        cv2.putText(image, line, (x, y_pos), font, font_scale, color, thickness)

    return image

def run_inference():
    # 모델 로드
    model = YOLO(r"D:\lettuce_cabbage_v2.pt")
    
    # 고해상도 이미지 로드
    image_path = r"C:\Users\dromii\Downloads\j_237.tif"
    image = cv2.imread(image_path)
    
    # 타일 크기 설정 640,240,124..
    tile_size = 240
    
    # 이미지를 타일로 분할
    tiles = split_image(image, tile_size)
    
    results_list = []

    # 타일별로 예측 수행
    for (x, y, tile) in tiles:
        results = model.predict(tile, conf=0.5)
        results_list.append((x, y, results))

    # 타일별 결과를 원본 이미지로 병합
    final_image, counts = merge_tiles(image, results_list, tile_size)
    
    # 탐지된 작물 종류와 개수를 오른쪽 상단에 표시
    final_image = draw_counts(final_image, counts)
    
    # 저장 경로 지정
    save_dir = r"D:\results2"
    
    # 경로가 없다면 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 결과 이미지 TIF 형식으로 저장
    save_path = os.path.join(save_dir, 'final_image_with_counts2.tif')
    cv2.imwrite(save_path, final_image)
    
    # 저장된 경로 및 탐지된 작물 출력
    print(f"결과 이미지가 TIF 형식으로 저장되었습니다: {save_path}")
    print("탐지된 작물 종류와 개수:")
    for label, count in counts.items():
        print(f"{label}: {count}")
    
    # 결과 이미지 표시
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()
