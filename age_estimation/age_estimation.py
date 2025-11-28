#!/usr/bin/env python3
"""
라즈베리파이 카메라를 사용한 실시간 얼굴 나이 추정 프로젝트
OpenCV DNN 모듈을 사용하여 얼굴 감지 및 나이 추정을 수행합니다.
(FPS 최적화 버전 - 나이 추정 로직 원본 유지)
"""

import cv2
import numpy as np
import os
import sys
import time
from threading import Thread, Lock

# 모델 파일 경로
MODEL_DIR = "models"
FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")

# 나이 그룹 레이블 (원본 유지)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# 모델 입력 크기 (원본 유지)
FACE_INPUT_SIZE = (300, 300)
AGE_INPUT_SIZE = (227, 227)

# 평균값 (BGR) (원본 유지)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 얼굴 감지 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.7


class CameraThread:
    """비동기 카메라 캡처 스레드"""
    
    def __init__(self, camera, camera_type):
        self.camera = camera
        self.camera_type = camera_type
        self.frame = None
        self.stopped = False
        self.lock = Lock()
        
    def start(self):
        Thread(target=self._update, daemon=True).start()
        return self
    
    def _update(self):
        while not self.stopped:
            if self.camera_type == "picamera2":
                frame = self.camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.camera.read()
                if not ret:
                    continue
            
            with self.lock:
                self.frame = frame
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True


class AgeEstimator:
    """얼굴 나이 추정 클래스 (원본 로직 유지)"""
    
    def __init__(self):
        """모델 초기화"""
        print("모델 로딩 중...")
        
        # 모델 파일 존재 확인
        self._check_model_files()
        
        # 얼굴 감지 모델 로드
        self.face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        print("얼굴 감지 모델 로드 완료")
        
        # 나이 추정 모델 로드
        self.age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        print("나이 추정 모델 로드 완료")
        
        # 라즈베리파이에서는 CPU 사용 권장
        self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        print("모델 초기화 완료!")
    
    def _check_model_files(self):
        """모델 파일 존재 확인"""
        required_files = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("다음 모델 파일이 없습니다:")
            for f in missing_files:
                print(f"  - {f}")
            print("\n먼저 download_models.py를 실행하세요:")
            print("  python3 download_models.py")
            sys.exit(1)
    
    def detect_faces(self, frame):
        """프레임에서 얼굴 감지 (원본 유지)"""
        h, w = frame.shape[:2]
        
        # 얼굴 감지를 위한 블롭 생성
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, FACE_INPUT_SIZE, 
            (104.0, 177.0, 123.0), 
            swapRB=False, crop=False
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > CONFIDENCE_THRESHOLD:
                # 얼굴 경계 상자 좌표 계산
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                # 경계 상자가 프레임 내에 있도록 조정
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    faces.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': float(confidence)
                    })
        
        return faces
    
    def estimate_age(self, frame, face_box):
        """얼굴 영역에서 나이 추정 (원본 유지)"""
        x1, y1, x2, y2 = face_box
        
        # 얼굴 영역 추출
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None, 0.0
        
        # 나이 추정을 위한 블롭 생성 (원본 그대로)
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, AGE_INPUT_SIZE,
            MODEL_MEAN_VALUES, swapRB=False
        )
        
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        
        # 가장 높은 확률의 나이 그룹 선택
        age_idx = age_preds[0].argmax()
        age = AGE_BUCKETS[age_idx]
        confidence = age_preds[0][age_idx]
        
        return age, confidence
    
    def process_frame(self, frame):
        """프레임 처리 및 결과 시각화 (원본 유지)"""
        # 얼굴 감지
        faces = self.detect_faces(frame)
        
        results = []
        for face in faces:
            x1, y1, x2, y2 = face['box']
            
            # 나이 추정
            age, age_conf = self.estimate_age(frame, face['box'])
            
            if age:
                results.append({
                    'box': face['box'],
                    'face_confidence': face['confidence'],
                    'age': age,
                    'age_confidence': age_conf
                })
                
                # 얼굴 경계 상자 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 나이 정보 표시
                label = f"Age: {age}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # 레이블 배경
                cv2.rectangle(frame, 
                             (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0] + 10, y1),
                             (0, 255, 0), -1)
                
                # 레이블 텍스트
                cv2.putText(frame, label,
                           (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 0, 0), 2)
        
        return frame, results


def init_camera():
    """카메라 초기화 (라즈베리파이 카메라 또는 USB 카메라)"""
    camera = None
    
    # 방법 1: picamera2 사용 (라즈베리파이 카메라 모듈)
    try:
        from picamera2 import Picamera2
        print("Picamera2를 사용하여 라즈베리파이 카메라 초기화...")
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)  # 카메라 안정화 대기
        print("라즈베리파이 카메라 초기화 완료!")
        return camera, "picamera2"
    except ImportError:
        print("Picamera2를 사용할 수 없습니다. OpenCV VideoCapture를 시도합니다...")
    except Exception as e:
        print(f"Picamera2 초기화 실패: {e}")
    
    # 방법 2: OpenCV VideoCapture 사용 (USB 카메라 또는 일반 카메라)
    print("OpenCV VideoCapture를 사용하여 카메라 초기화...")
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if not camera.isOpened():
        camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    
    if camera.isOpened():
        # MJPG 포맷으로 더 빠른 캡처
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
        print("USB/일반 카메라 초기화 완료!")
        return camera, "opencv"
    
    print("카메라를 찾을 수 없습니다!")
    return None, None


def release_camera(camera, camera_type):
    """카메라 자원 해제"""
    if camera_type == "picamera2":
        camera.stop()
    else:
        camera.release()


def main():
    """메인 함수"""
    print("=" * 50)
    print("실시간 얼굴 나이 추정 시스템")
    print("라즈베리파이 + OpenCV (FPS 최적화)")
    print("=" * 50)
    print()
    
    # 나이 추정기 초기화
    estimator = AgeEstimator()
    
    # 카메라 초기화
    camera, camera_type = init_camera()
    if camera is None:
        print("카메라 초기화 실패!")
        sys.exit(1)
    
    # 비동기 카메라 스레드 시작
    cam_thread = CameraThread(camera, camera_type).start()
    time.sleep(1)  # 스레드 안정화 대기
    
    print()
    print("시스템 시작됨!")
    print("'q' 키를 누르면 종료됩니다.")
    print("'s' 키를 누르면 현재 프레임을 저장합니다.")
    print()
    
    # FPS 계산용 변수
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # 캐시 변수 (프레임 스킵용)
    cached_results = []
    frame_count = 0
    PROCESS_EVERY_N = 2  # 2프레임마다 추론
    
    # 저장 디렉토리 생성
    save_dir = "captured"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        while True:
            # 비동기로 프레임 읽기
            frame = cam_thread.read()
            
            if frame is None:
                continue
            
            frame_count += 1
            
            # N 프레임마다만 추론 수행
            if frame_count % PROCESS_EVERY_N == 0:
                processed_frame, results = estimator.process_frame(frame.copy())
                cached_results = results
            else:
                # 캐시된 결과로 그리기만 수행
                processed_frame = frame.copy()
                for r in cached_results:
                    x1, y1, x2, y2 = r['box']
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"Age: {r['age']}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(processed_frame, 
                                 (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0] + 10, y1),
                                 (0, 255, 0), -1)
                    cv2.putText(processed_frame, label,
                               (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 0, 0), 2)
                results = cached_results
            
            # FPS 계산
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1:
                fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # FPS 표시
            cv2.putText(processed_frame, f"FPS: {fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
            
            # 감지된 얼굴 수 표시
            cv2.putText(processed_frame, f"Faces: {len(results)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
            
            # 결과 화면 표시
            cv2.imshow("Age Estimation", processed_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("종료합니다...")
                break
            elif key == ord('s'):
                # 현재 프레임 저장
                timestamp = int(time.time())
                filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, processed_frame)
                print(f"이미지 저장됨: {filename}")
    
    except KeyboardInterrupt:
        print("\n키보드 인터럽트 감지. 종료합니다...")
    
    finally:
        # 자원 해제
        cam_thread.stop()
        release_camera(camera, camera_type)
        cv2.destroyAllWindows()
        print("프로그램 종료.")


if __name__ == "__main__":
    main()