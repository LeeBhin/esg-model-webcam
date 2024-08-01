import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import pygame
import numpy as np

# 안전한 전역 변수로 DetectionModel 추가
torch.serialization.add_safe_globals([DetectionModel])

# Pygame 초기화
pygame.init()

# 모델 파일 경로
model_path = 'best.pt'

# YOLO 모델 로드
model = YOLO(model_path)

# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 화면 크기 설정
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Real-time Object Detection")

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 객체 탐지
    results = model(frame)

    # 결과 시각화
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 클래스 이름과 신뢰도 표시
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # OpenCV BGR에서 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # NumPy 배열을 Pygame surface로 변환
    pygame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    
    # 화면에 표시
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()

    clock.tick(300)  # FPS 제한

# 정리
cap.release()
pygame.quit()

# 메모리에서 모델 제거
del model
torch.cuda.empty_cache()

print("프로그램이 종료되었습니다.")