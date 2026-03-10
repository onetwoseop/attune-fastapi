"""
파이프라인 구조 검증 테스트 (WebSocket / 모델 다운로드 불필요)
Dummy 모델로 전체 흐름이 연결되는지 로그로 확인합니다.

실행:
    cd counseling_server
    python -m tests.test_pipeline
"""

import logging
import os
import sys

# 프로젝트 루트(counseling_server/)를 모듈 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "=" * 60


# ------------------------------------------------------------------
# 더미 데이터 생성 헬퍼
# ------------------------------------------------------------------

def make_dummy_jpeg() -> bytes:
    """유효한 최소 JPEG 바이너리 (1×1 흰색 픽셀)"""
    return (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
        b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e\x1b'
        b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
        b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
        b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd5\xff\xd9'
    )


def make_dummy_audio(size_kb: int = 50) -> bytes:
    """더미 오디오 바이너리 (size_kb KB 크기의 0으로 채워진 데이터)"""
    return bytes(size_kb * 1024)


def split_chunks(data: bytes, chunk_size: int = 4096):
    """바이트 데이터를 chunk_size 단위로 분할"""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


# ------------------------------------------------------------------
# 테스트 실행
# ------------------------------------------------------------------

def run():
    from app.core.container import ai_container
    from app.services.pipeline import CounselingPipeline

    SESSION = "test-session-001"

    # ── STEP 0: 모델 로딩 ──────────────────────────────────────────
    logger.info(SEP)
    logger.info("STEP 0  AI 모델 로딩 (Dummy)")
    logger.info(SEP)
    ai_container.load_models()

    pipe = CounselingPipeline(ai_container)
    pipe.init_session(SESSION)

    # ── STEP 1: 이미지 → 얼굴 감정 ───────────────────────────────
    logger.info("")
    logger.info("STEP 1  이미지 프레임 3개 → 얼굴 감정 분석")
    dummy_image = make_dummy_jpeg()
    for i in range(3):
        result = pipe.process_face_frame(SESSION, dummy_image)
        logger.info(f"         Frame {i + 1} 결과: {result.primary_emotion}")

    # ── STEP 2: 오디오 청크 수신 ─────────────────────────────────
    logger.info("")
    logger.info("STEP 2  오디오 청크 수신 (발화 중 시뮬레이션)")
    audio_data = make_dummy_audio(50)
    chunks = split_chunks(audio_data)
    logger.info(f"         {len(audio_data)}B → {len(chunks)}개 청크로 분할하여 전송")
    for chunk in chunks:
        pipe.append_audio_chunk(SESSION, chunk)

    # ── STEP 3: 발화 종료 → STT + 음성 감정 ──────────────────────
    logger.info("")
    logger.info("STEP 3  END_OF_SPEECH → STT + 음성 감정 추출")
    stt1 = pipe.on_speech_end(SESSION)
    if stt1:
        logger.info(f"         [1차 발화] STT: '{stt1.text}'")

    # 두 번째 발화 시뮬레이션
    logger.info("")
    logger.info("         두 번째 발화 시뮬레이션")
    for chunk in split_chunks(make_dummy_audio(30)):
        pipe.append_audio_chunk(SESSION, chunk)
    stt2 = pipe.on_speech_end(SESSION)
    if stt2:
        logger.info(f"         [2차 발화] STT: '{stt2.text}'")

    # ── STEP 4-5: 세션 종료 → 감정 종합 → LLM ───────────────────
    logger.info("")
    logger.info("STEP 4-5  END_OF_SESSION → 감정 종합 → LLM 응답 생성")
    response = pipe.generate_response(SESSION)
    if response:
        logger.info(f"          LLM 응답  : '{response.reply_text}'")
        if response.suggested_action:
            logger.info(f"          추천 행동 : '{response.suggested_action}'")

    # ── 정리 ──────────────────────────────────────────────────────
    pipe.cleanup_session(SESSION)

    logger.info("")
    logger.info(SEP)
    logger.info("전체 파이프라인 구조 검증 완료")
    logger.info(SEP)


if __name__ == "__main__":
    run()
