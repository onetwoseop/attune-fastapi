import logging
from typing import Dict, List, Optional

from ai_modules.schemas import (
    EmotionResult, FaceInput, LLMContext, LLMResponse, STTInput, STTOutput
)
from app.core.container import AIContainer

logger = logging.getLogger(__name__)

# CounselingPipeline
# 역할: WebSocket에서 수신한 데이터를 버퍼링하고, AI 모델을 순서대로 호출하는 처리 파이프라인.

class CounselingPipeline:

    def __init__(self, container: AIContainer):
        self.container = container
        # 세션별 버퍼
        self._audio_buffers: Dict[str, bytearray] = {}
        self._face_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._voice_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._stt_text_buffer: Dict[str, List[str]] = {}

    # 세션 수명 주기

    def init_session(self, session_id: str) -> None:
        self._audio_buffers[session_id] = bytearray()
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []
        self._stt_text_buffer[session_id] = []
        logger.info(f"세션 초기화: {session_id}")

    def cleanup_session(self, session_id: str) -> None:
        for buf in (
            self._audio_buffers,
            self._face_emotion_buffer,
            self._voice_emotion_buffer,
            self._stt_text_buffer,
        ):
            buf.pop(session_id, None)
        logger.info(f"세션 정리: {session_id}")


    # 오디오 청크 누적

    def append_audio_chunk(self, session_id: str, chunk: bytes) -> None:
        """클라이언트에서 실시간으로 들어오는 오디오 청크를 버퍼에 누적."""
        self._audio_buffers[session_id].extend(chunk)
        total = len(self._audio_buffers[session_id])
        logger.info(f"[Audio] {session_id}: +{len(chunk)}B (누적: {total}B)")

    # 이미지 프레임 → 얼굴 감정 분석

    def process_face_frame(self, session_id: str, image_bytes: bytes) -> EmotionResult:
        """JPEG bytes를 받아 얼굴 감정을 추출하고 버퍼에 저장."""
        face_input = FaceInput(video_frame=image_bytes)
        result = self.container.face_emotion.analyze(face_input)
        self._face_emotion_buffer[session_id].append(result)
        logger.info(
            f"[Face] {session_id}: "
            f"{result.primary_emotion} {result.probabilities}"
        )
        return result

    # 발화 종료 → STT + 음성 감정

    def on_speech_end(self, session_id: str) -> Optional[STTOutput]:
        """
        END_OF_SPEECH 신호 수신 시 호출.
        버퍼된 오디오 전체를 STT와 음성 감정 모델에 넘김.
        """
        audio_data = bytes(self._audio_buffers[session_id])
        if not audio_data:
            logger.warning(f"[SpeechEnd] {session_id}: 오디오 버퍼 비어있음, 건너뜀")
            return None

        logger.info(f"[SpeechEnd] {session_id}: 버퍼 {len(audio_data)}B → 처리 시작")
        stt_input = STTInput(audio_data=audio_data)

        # STT
        stt_result = self.container.stt.transcribe(stt_input)
        self._stt_text_buffer[session_id].append(stt_result.text)
        logger.info(f"[STT] {session_id}: '{stt_result.text}'")

        # 음성 감정
        voice_emotion = self.container.audio_emotion.analyze(stt_input)
        self._voice_emotion_buffer[session_id].append(voice_emotion)
        logger.info(
            f"[VoiceEmotion] {session_id}: "
            f"{voice_emotion.primary_emotion} {voice_emotion.probabilities}"
        )

        self._audio_buffers[session_id].clear()
        return stt_result

    # 세션 종료 → 감정 종합 → LLM 응답

    def generate_response(self, session_id: str) -> Optional[LLMResponse]:
        """
        상담 싱글턴 종료 신호 수신 시 호출.
        누적된 텍스트와 감정 목록을 그대로 AI에 전달.
        """
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            logger.warning(f"[Generate] {session_id}: 누적 텍스트 없음, 건너뜀")
            return None

        face_emotions = self._face_emotion_buffer[session_id]
        voice_emotions = self._voice_emotion_buffer[session_id]

        logger.info(
            f"[Generate] {session_id}: "
            f"face {len(face_emotions)}건, voice {len(voice_emotions)}건 → AI 전달"
        )
        logger.info(f"[Generate] {session_id}: 누적 텍스트='{accumulated_text}'")

        llm_context = LLMContext(
            user_text=accumulated_text,
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
        )
        response = self.container.llm.generate_response(llm_context)
        logger.info(f"[LLM] {session_id}: '{response.reply_text}'")
        if response.suggested_action:
            logger.info(f"[LLM] 추천 행동: '{response.suggested_action}'")

        return response


# 전역 인스턴스 (session_manager에서 import해서 사용)
from app.core.container import ai_container
pipeline = CounselingPipeline(ai_container)
