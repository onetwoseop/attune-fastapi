import abc
import time
import random
from typing import Any
from .schemas import (
    VADInput, VADOutput, STTInput, STTOutput,
    EmotionResult, LLMContext, LLMResponse, FaceInput
)
# 인터페이스 클래스(현재는 더미 데이터로 구성)

# VAD(음성감지)
class BaseVADModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def process(self, input_data: VADInput) -> VADOutput: pass

class DummyVADModel(BaseVADModel):
    def load_model(self):
        print("[VAD] Dummy Model Loaded (Silero VAD 흉내)")
        
    def process(self, input_data: VADInput) -> VADOutput:
        # 가짜 로직
        is_talking = random.choice([True, False])
        return VADOutput(is_speech=is_talking, confidence=0.9 if is_talking else 0.1)


# STT(받아쓰기)
class BaseSTTModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def transcribe(self, input_data: STTInput) -> STTOutput: pass

class DummySTTModel(BaseSTTModel):
    def load_model(self):
        print("[STT] Dummy Whisper Model Loaded")
        
    def transcribe(self, input_data: STTInput) -> STTOutput:
        return STTOutput(text="아, 정말 힘드셨겠어요. (테스트 텍스트)", language="ko")
    
    
class BaseEmotionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def analyze(self, input_data: Any) -> EmotionResult: pass

    
# Emotion Analysis (음성 감정 분석)
class DummyAudioEmotionModel(BaseEmotionModel):
    def load_model(self):
        print("[Audio Emo] Dummy SpeechBrain Loaded")
        
    def analyze(self, input_data: STTInput) -> EmotionResult:
        return EmotionResult(
            primary_emotion="fear",  # 목소리가 떨린다고 가정
            probabilities={"fear": 0.7, "sad": 0.3}
        )


# Emotion Analysis (안면 감정 분석)
class DummyFaceEmotionModel(BaseEmotionModel):
    def load_model(self):
        print("[Face] Dummy DeepFace Loaded")
        
    def analyze(self, input_data: FaceInput) -> EmotionResult:
        return EmotionResult(
            primary_emotion="sad",
            probabilities={"sad": 0.8, "neutral": 0.2}
        )


# 4. LLM
class BaseLLMModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def generate_response(self, context: LLMContext) -> LLMResponse: pass

class DummyLLMModel(BaseLLMModel):
    """테스트용 가짜 구현. 실제 모델 연결 시 이 클래스를 교체하면 됩니다."""

    def load_model(self):
        print("[LLM] Dummy Qwen2 Model Loaded")

    def generate_response(self, context: LLMContext) -> LLMResponse:
        # context에서 사용 가능한 데이터:
        #   context.user_text      : 이번 발화 전체 텍스트 (STT 결과 합산)
        #   context.face_emotions  : List[EmotionResult] - 프레임별 얼굴 감정
        #   context.voice_emotions : List[EmotionResult] - 발화별 음성 감정
        face_summary = context.face_emotions[0].primary_emotion if context.face_emotions else "N/A"
        voice_summary = context.voice_emotions[0].primary_emotion if context.voice_emotions else "N/A"
        return LLMResponse(
            reply_text=(
                f"사용자님, '{context.user_text}'라고 하셨군요. "
                f"(얼굴: {face_summary}, 목소리: {voice_summary}) 많이 속상하셨겠습니다."
            ),
            suggested_action="심호흡 하기"
        )