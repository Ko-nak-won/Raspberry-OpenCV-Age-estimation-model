# 라즈베리파이 실시간 얼굴 나이 추정 시스템

OpenCV DNN 모듈을 사용하여 라즈베리파이 카메라에서 실시간으로 얼굴을 감지하고 나이를 추정하는 프로젝트입니다.

## 📋 기능

- 실시간 얼굴 감지 (SSD 기반)
- 나이 추정 (8개 연령대 분류)
- FPS 표시
- 이미지 캡처 저장
- 라즈베리파이 카메라 모듈 및 USB 카메라 지원

## 🔧 요구사항

### 하드웨어
- 라즈베리파이 (3B+ 이상 권장)
- 라즈베리파이 카메라 모듈 또는 USB 카메라
- 모니터 (HDMI 또는 VNC 연결)

### 소프트웨어
- Python 3.7+
- OpenCV 4.5+
- NumPy
- Picamera2 (라즈베리파이 카메라 사용 시)

## 📦 설치 방법

### 1. 시스템 패키지 설치

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-pip
```

### 2. 라즈베리파이 카메라 사용 시 (선택사항)

```bash
sudo apt install -y python3-picamera2
```

### 3. Python 패키지 설치

```bash
pip3 install -r requirements.txt
```

### 4. 모델 다운로드

```bash
python3 download_models.py
```

그 후 age_net.caffemodel 파일을 수동으로 다운로드해야 합니다:

```bash
cd models
wget https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
```

또는 다음 링크에서 직접 다운로드:
- https://github.com/GilLevi/AgeGenderDeepLearning

## 🚀 실행 방법

```bash
python3 qqq.py
```

## ⌨️ 키보드 단축키

| 키 | 기능 |
|---|------|
| `q` | 프로그램 종료 |
| `s` | 현재 프레임 저장 |

## 📁 프로젝트 구조

```
age_estimation/
├── age_estimation.py      # 메인 프로그램
├── download_models.py     # 모델 다운로드 스크립트
├── age_deploy.prototxt    # 나이 추정 모델 정의
├── requirements.txt       # Python 의존성
├── README.md             # 이 파일
├── models/               # 모델 파일 디렉토리
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── age_deploy.prototxt
│   └── age_net.caffemodel
└── captured/             # 캡처된 이미지 저장
```

## 📊 나이 추정 범위

| 클래스 | 연령대 |
|--------|--------|
| 0 | 0-2세 |
| 1 | 4-6세 |
| 2 | 8-12세 |
| 3 | 15-20세 |
| 4 | 25-32세 |
| 5 | 38-43세 |
| 6 | 48-53세 |
| 7 | 60-100세 |

## ⚠️ 문제 해결

### 카메라가 인식되지 않을 때

1. 라즈베리파이 카메라 활성화 확인:
```bash
sudo raspi-config
# Interface Options > Camera > Enable
```

2. 카메라 연결 확인:
```bash
vcgencmd get_camera
# supported=1 detected=1 이면 정상
```

3. 권한 문제:
```bash
sudo usermod -aG video $USER
# 로그아웃 후 다시 로그인
```

### 모델 파일 오류

모델 파일이 손상되었거나 없을 경우 `download_models.py`를 다시 실행하세요.

### 낮은 FPS

라즈베리파이에서는 1-5 FPS 정도가 정상입니다. 성능 개선을 위해:
- 프레임 크기 축소 (640x480 → 320x240)
- 프레임 스킵 적용

## 📝 라이선스

MIT License

## 🙏 참고 자료

- [OpenCV DNN 모듈](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
- [Age and Gender Classification](https://github.com/GilLevi/AgeGenderDeepLearning)
- [Picamera2 문서](https://github.com/raspberrypi/picamera2)