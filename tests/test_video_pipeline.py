"""
실제 비디오 파일로 파이프라인 데이터 전달 테스트
- 영상 프레임 → JPEG bytes → process_face_frame
- 음성 → 2초 청크 bytes → append_audio_chunk → on_speech_end
- END_OF_SESSION → generate_response

실행:
    cd counseling_server
    PYTHONIOENCODING=utf-8 python -m tests.test_video_pipeline
"""

import io
import logging
import os
import sys

import cv2
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_files", "test_video1.mp4")
SEP = "=" * 60

# 몇 프레임마다 한 번 감정 분석할지 (너무 많으면 느려짐)
FRAME_SAMPLE_INTERVAL = 30  # 30프레임마다 1장


def extract_frames_as_jpeg(video_path: str) -> list[bytes]:
    """cv2로 영상에서 프레임을 추출해 JPEG bytes 리스트로 반환"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"[Video] 총 {total_frames}프레임, {fps:.1f}fps → {total_frames/fps:.1f}초")

    jpeg_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                jpeg_frames.append(buf.tobytes())
        frame_idx += 1

    cap.release()
    logger.info(f"[Video] {frame_idx}프레임 중 {len(jpeg_frames)}장 샘플 추출 완료")
    return jpeg_frames


def extract_audio_chunks(video_path: str, chunk_sec: float = 2.0) -> list[bytes]:
    """torchaudio로 음성을 읽어 chunk_sec 단위 WAV bytes 청크 리스트로 반환"""
    waveform, sample_rate = torchaudio.load(video_path)
    total_sec = waveform.shape[1] / sample_rate
    logger.info(f"[Audio] {total_sec:.1f}초, {sample_rate}Hz, {waveform.shape[0]}ch")

    chunk_samples = int(sample_rate * chunk_sec)
    chunks = []
    for start in range(0, waveform.shape[1], chunk_samples):
        segment = waveform[:, start:start + chunk_samples]
        buf = io.BytesIO()
        torchaudio.save(buf, segment, sample_rate, format="wav")
        chunks.append(buf.getvalue())

    logger.info(f"[Audio] {chunk_sec}초 단위 → {len(chunks)}개 청크")
    return chunks


def run():
    from app.core.container import ai_container
    from app.services.pipeline import CounselingPipeline

    SESSION = "test-video-session"

    logger.info(SEP)
    logger.info(f"테스트 영상: {VIDEO_PATH}")
    logger.info(SEP)

    # ── STEP 0: 모델 로딩 ─────────────────────────────────────────
    logger.info("\nSTEP 0  AI 모델 로딩 (Dummy)")
    ai_container.load_models()
    pipe = CounselingPipeline(ai_container)
    pipe.init_session(SESSION)

    # ── STEP 1: 프레임 추출 → face emotion ───────────────────────
    logger.info(f"\nSTEP 1  영상 프레임 → 얼굴 감정 분석 (매 {FRAME_SAMPLE_INTERVAL}프레임마다 1장)")
    jpeg_frames = extract_frames_as_jpeg(VIDEO_PATH)
    for i, jpeg_bytes in enumerate(jpeg_frames):
        result = pipe.process_face_frame(SESSION, jpeg_bytes)
        logger.info(f"         Frame {i+1}/{len(jpeg_frames)}: {len(jpeg_bytes)}B → {result.primary_emotion}")

    # ── STEP 2-3: 오디오 청크 → END_OF_SPEECH ────────────────────
    logger.info("\nSTEP 2-3  음성 청크 수신 → END_OF_SPEECH")
    audio_chunks = extract_audio_chunks(VIDEO_PATH, chunk_sec=2.0)
    for i, chunk in enumerate(audio_chunks):
        pipe.append_audio_chunk(SESSION, chunk)
        logger.info(f"         청크 {i+1}/{len(audio_chunks)}: {len(chunk)}B 전달")

    logger.info("\n         END_OF_SPEECH 신호 → STT + 음성 감정")
    stt_result = pipe.on_speech_end(SESSION)
    if stt_result:
        logger.info(f"         STT 결과: '{stt_result.text}'")

    # ── STEP 4: END_OF_SESSION → LLM ─────────────────────────────
    logger.info("\nSTEP 4  END_OF_SESSION → 감정 종합 → LLM 응답")
    response = pipe.generate_response(SESSION)
    if response:
        logger.info(f"         LLM 응답: '{response.reply_text}'")
        if response.suggested_action:
            logger.info(f"         추천 행동: '{response.suggested_action}'")

    pipe.cleanup_session(SESSION)

    logger.info(f"\n{SEP}")
    logger.info("실제 영상 데이터 전달 테스트 완료")
    logger.info(SEP)


if __name__ == "__main__":
    run()
