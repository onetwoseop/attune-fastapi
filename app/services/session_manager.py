import json
import base64
import logging
from fastapi import WebSocket
from typing import Dict

from app.schemas import InputTest, ServerResponse
from app.services.pipeline import pipeline

logger = logging.getLogger(__name__)

# 접속자를 관리하고 데이터를 전달하는 역할, 비동기 처리

class ConnectionManager:
    def __init__(self):
        # 활성화된 상담 세션들을 저장하는 장부
        self.active_connections: Dict[str, WebSocket] = {}

    # [초기 상담 생성] 웹소캣 연결 수락
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        pipeline.init_session(client_id)
        # 연결 성공 로깅으로 바꿈
        logger.info(f"--- [Session] {client_id} 연결 (현재 접속자: {len(self.active_connections)}명)")

        # 연결 성공 메시지 전송
        await self.send_personal_message(
            ServerResponse(status="connected", message="상담실에 입장하였습니다.").model_dump(),
            client_id
        )

    # 연결 해제 처리
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            pipeline.cleanup_session(client_id)
            logger.info(f"[Session] {client_id} 연결 해제")

    # 특정 사용자에게 JSON 변환 메시지 전송
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            ws = self.active_connections[client_id]
            await ws.send_text(json.dumps(message, ensure_ascii=False)) # 파이썬 딕셔너리를 문자열로 변환

    # [데이터 처리 파이프라인] 들어온 데이터 분류 및 처리
    async def process_data(self, client_id: str, raw_data: str):
        try:
            # JSON 데이터 파싱
            data_dict = json.loads(raw_data)
            input_obj = InputTest(**data_dict)

            # 타입별 처리 분기

            # [음성 데이터 처리]
            if input_obj.type == "audio":
                try:
                    base64_data = str(input_obj.data)
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    pipeline.append_audio_chunk(client_id, base64.b64decode(base64_data))
                except Exception as e:
                    logger.error(f"[Error] 오디오 처리 실패: {e}")
                # VAD 등 음성 처리 로직 추가

            # [이미지 데이터 처리]
            elif input_obj.type == "video":
                # 표정기반 감정추출 로직 추가 필요
                try:
                    base64_data = str(input_obj.data)
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    pipeline.process_face_frame(client_id, image_bytes)
                except Exception as e:
                    logger.error(f"[Error] 이미지 변환 실패: {e}")

            # [발화 신호 처리]
            elif input_obj.type == "control":
                if input_obj.data == "END_OF_SPEECH":
                    logger.info(f"[Control] {client_id}의 발화 종료, 처리 시작 ...")
                    pipeline.on_speech_end(client_id)
                    # 말하기 종료 및 STT, LLM 로직 추가
                    await self.send_personal_message(
                        {"status": "processing", "message": "답변 생성 중..."},
                        client_id
                    )

                elif input_obj.data == "END_OF_SESSION":
                    logger.info(f"[Session] {client_id} 세션 종료 신호 수신")
                    # 세션 종료 처리 로직 추가 예정

            else:
                logger.warning(f"[Session] 알 수 없는 타입: {input_obj.type}")

            await self.send_personal_message(
                ServerResponse(status="received", message=f"{input_obj.type} 수신 완료").model_dump(),
                client_id
            )

        except json.JSONDecodeError:
            logger.error(f"[Session] JSON 파싱 실패: {raw_data[:100]}")
        except Exception as e:
            logger.error(f"[Session] 처리 중 오류: {e}")


manager = ConnectionManager()
