import json
import base64
import logging
from fastapi import WebSocket
from typing import Dict

from app.schemas import InputTest, ServerResponse
from app.services.pipeline import pipeline

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket 연결 관리 + 수신 데이터를 pipeline으로 전달. AI 처리 로직은 pipeline.py가 담당."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        pipeline.init_session(client_id)
        logger.info(f"[Session] {client_id} 연결 (현재 접속자: {len(self.active_connections)}명)")
        await self.send(
            ServerResponse(status="connected", message="상담실에 입장하였습니다.").model_dump(),
            client_id
        )

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            pipeline.cleanup_session(client_id)
            logger.info(f"[Session] {client_id} 연결 해제")

    async def send(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            ws = self.active_connections[client_id]
            await ws.send_text(json.dumps(message, ensure_ascii=False))

    async def process_data(self, client_id: str, raw_data: str):
        try:
            data_dict = json.loads(raw_data)
            input_obj = InputTest(**data_dict)

            if input_obj.type == "audio":
                base64_data = str(input_obj.data)
                if "," in base64_data:
                    base64_data = base64_data.split(",")[1]
                pipeline.append_audio_chunk(client_id, base64.b64decode(base64_data))

            elif input_obj.type == "video":
                base64_data = str(input_obj.data)
                if "," in base64_data:
                    base64_data = base64_data.split(",")[1]
                pipeline.process_face_frame(client_id, base64.b64decode(base64_data))

            elif input_obj.type == "control":
                if input_obj.data == "END_OF_SPEECH":
                    logger.info(f"[Session] {client_id} 발화 종료 신호 수신")
                    pipeline.on_speech_end(client_id)
                    await self.send({"status": "processing", "message": "답변 생성 중..."}, client_id)

                elif input_obj.data == "END_OF_SESSION":
                    logger.info(f"[Session] {client_id} 세션 종료 신호 수신")
                    response = pipeline.generate_response(client_id)
                    if response:
                        await self.send(
                            ServerResponse(
                                status="reply",
                                message=response.reply_text,
                                next_action=response.suggested_action
                            ).model_dump(),
                            client_id
                        )

            else:
                logger.warning(f"[Session] 알 수 없는 타입: {input_obj.type}")

            await self.send(
                ServerResponse(status="received", message=f"{input_obj.type} 수신 완료").model_dump(),
                client_id
            )

        except json.JSONDecodeError:
            logger.error(f"[Session] JSON 파싱 실패: {raw_data[:100]}")
        except Exception as e:
            logger.error(f"[Session] 처리 중 오류: {e}")


manager = ConnectionManager()
