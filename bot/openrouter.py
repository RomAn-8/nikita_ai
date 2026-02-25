import requests
import logging
from .config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUDIO_URL = "https://openrouter.ai/api/v1/audio/transcriptions"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def chat_completion_raw(
    messages,
    timeout: int = 60,
    temperature: float = 0.7,
    model: str | None = None,
) -> dict:
    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
        "temperature": float(temperature),
    }

    try:
        r = requests.post(OPENROUTER_CHAT_URL, headers=_headers(), json=payload, timeout=timeout)
        
        # Логируем детали ошибки перед raise_for_status
        if r.status_code != 200:
            error_detail = ""
            try:
                error_detail = r.json()
            except:
                error_detail = r.text[:500]
            
            logger.error(f"OpenRouter API error {r.status_code}: {error_detail}")
            logger.error(f"Request payload: model={payload.get('model')}, messages_count={len(payload.get('messages', []))}")
            
            # Пытаемся извлечь более понятное сообщение об ошибке
            if isinstance(error_detail, dict):
                error_msg = error_detail.get("error", {}).get("message", "") if isinstance(error_detail.get("error"), dict) else str(error_detail)
                if error_msg:
                    raise requests.exceptions.HTTPError(f"OpenRouter API error: {error_msg}")
        
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        # Пробрасываем HTTPError дальше с улучшенным сообщением
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in chat_completion_raw: {e}")
        raise


def chat_completion(
    messages,
    timeout: int = 60,
    temperature: float = 0.7,
    model: str | None = None,
) -> str:
    data = chat_completion_raw(messages, timeout=timeout, temperature=temperature, model=model)
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def transcribe_audio(
    audio_bytes: bytes,
    model: str | None = None,
    mime_type: str | None = None,
    timeout: int = 120,
) -> str:
    """Распознает речь из аудио через OpenRouter API с поддержкой аудио."""
    from .config import VOICE_MODEL
    import base64
    import io
    import tempfile
    import os
    import pydub
    from pydub import AudioSegment
    
    # Устанавливаем явный путь к ffmpeg для pydub
    pydub.AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
    pydub.AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"
    
    # Используем переданную модель или VOICE_MODEL по умолчанию
    if model is None:
        model = VOICE_MODEL
    
    try:
        logger.debug(f"Transcribing audio using OpenRouter model: {model}, mime_type: {mime_type}")
        
        # Определяем формат входного файла
        if mime_type and ("mp4" in mime_type or "m4a" in mime_type):
            input_format = "mp4"
        elif mime_type and "ogg" in mime_type:
            input_format = "ogg"
        else:
            input_format = None
        
        logger.info(f"Input audio: {len(audio_bytes)} bytes, mime_type={mime_type}, input_format={input_format}")
        
        # Сохраняем во временный файл
        suffix = ".mp4" if input_format == "mp4" else ".ogg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            if input_format:
                audio = AudioSegment.from_file(tmp_path, format=input_format)
            else:
                audio = AudioSegment.from_file(tmp_path)
            logger.info(f"AudioSegment loaded: duration={len(audio)}ms")
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
        finally:
            os.unlink(tmp_path)  # удаляем временный файл
        
        # Сжимаем
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Экспортируем в WAV (не mp3 — wav надёжнее)
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_bytes = wav_buffer.getvalue()
        audio_base64 = base64.b64encode(wav_bytes).decode()
        audio_format = "wav"
        
        logger.info(f"Converted to WAV: {len(wav_bytes)} bytes")
        
        # Используем формат input_audio для Voxtral с строгим system_prompt
        messages = [
            {
                "role": "system",
                "content": "You are a strict transcription engine. Your only task is to convert audio to text. Do not answer questions, do not provide information, and do not engage in conversation. Output ONLY the words spoken in the audio."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe the speech from this audio. Return ONLY the transcribed text, nothing else. No answers, no comments."
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": audio_format
                        }
                    }
                ]
            }
        ]
        
        data = chat_completion_raw(
            messages=messages,
            timeout=timeout,
            temperature=0.0,
            model=model
        )
        
        transcribed = _get_content_from_raw(data)
        if not transcribed:
            # Пробуем альтернативный формат с более строгим системным промптом
            logger.debug("Trying alternative format with stricter system prompt")
            messages_alt = [
                {
                    "role": "system",
                    "content": "You are a strict transcription engine. Your only task is to convert audio to text. Do not answer questions, do not provide information, and do not engage in conversation. Output ONLY the words spoken in the audio."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": audio_format
                            }
                        }
                    ]
                }
            ]
            data = chat_completion_raw(
                messages=messages_alt,
                timeout=timeout,
                temperature=0.0,
                model=model
            )
            transcribed = _get_content_from_raw(data)
        
        # Логируем результат транскрипции для отладки
        if transcribed:
            logger.info(f"Transcription result: {transcribed[:200]}")
        else:
            logger.warning("Transcription returned empty result")
        
        return transcribed.strip() if transcribed else ""
            
    except Exception as e:
        logger.exception(f"Error transcribing audio: {e}")
        raise ValueError(f"Ошибка при распознавании речи: {e}")


def _get_content_from_raw(data: dict) -> str:
    """Извлекает текстовый контент из ответа OpenRouter."""
    try:
        return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""
