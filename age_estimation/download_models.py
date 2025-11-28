#!/usr/bin/env python3
"""
얼굴 감지 및 나이 추정을 위한 사전 훈련된 모델 다운로드 스크립트
"""

import urllib.request
import os

# 모델 저장 디렉토리
MODEL_DIR = "models"

# 모델 파일 URL
MODELS = {
    # 얼굴 감지 모델 (OpenCV DNN)
    "face_detector": {
        "prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    },
    # 나이 추정 모델
    "age_net": {
        "prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
        "caffemodel": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel"
    }
}

def download_file(url, filepath):
    """URL에서 파일 다운로드"""
    print(f"다운로드 중: {url}")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"완료: {filepath}")
        return True
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return False

def main():
    # 모델 디렉토리 생성
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"디렉토리 생성: {MODEL_DIR}")

    # 얼굴 감지 모델 다운로드
    print("\n=== 얼굴 감지 모델 다운로드 ===")
    download_file(
        MODELS["face_detector"]["prototxt"],
        os.path.join(MODEL_DIR, "deploy.prototxt")
    )
    download_file(
        MODELS["face_detector"]["caffemodel"],
        os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    )

    # 나이 추정 모델 다운로드
    print("\n=== 나이 추정 모델 다운로드 ===")
    download_file(
        MODELS["age_net"]["prototxt"],
        os.path.join(MODEL_DIR, "age_deploy.prototxt")
    )
    
    # caffemodel은 큰 파일이라 직접 다운로드 안내
    caffemodel_path = os.path.join(MODEL_DIR, "age_net.caffemodel")
    if not os.path.exists(caffemodel_path):
        print(f"\n※ age_net.caffemodel 파일을 수동으로 다운로드해야 합니다.")
        print("다음 링크에서 다운로드하세요:")
        print("https://github.com/GilLevi/AgeGenderDeepLearning")
        print("또는 다음 명령어를 실행하세요:")
        print(f"wget https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel -O {caffemodel_path}")

    print("\n모델 다운로드 완료!")

if __name__ == "__main__":
    main()
