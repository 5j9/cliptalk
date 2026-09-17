from asyncio import to_thread

import pythoncom
import win32com.client as wincl

from cliptalk import AudioQ, logger
from cliptalk.config import SAPI_VOICE_NAME, SAPI_VOICE_RATE
from cliptalk.engines import create_wav_header, split_text

# -----------------------------------------------------------------------------
# SAPI Constants
# -----------------------------------------------------------------------------

SAFT16kHz16BitMono = 18


# -----------------------------------------------------------------------------
# Voice Selection
# -----------------------------------------------------------------------------


def _select_voice(voice_obj):
    try:
        voices = voice_obj.GetVoices()

        for voice in voices:
            desc = voice.GetDescription()

            if SAPI_VOICE_NAME in desc:
                voice_obj.Voice = voice
                logger.debug(f'SAPI voice selected: {desc}')
                return

    except Exception as e:
        logger.exception('Failed to select SAPI voice: %r', e)


# -----------------------------------------------------------------------------
# WAV Header
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# SAPI Synthesis
# -----------------------------------------------------------------------------


def _synthesize_chunk(text: str) -> bytes:
    """
    Convert a single text chunk to WAV bytes.
    """
    pythoncom.CoInitialize()

    try:
        voice = wincl.Dispatch('SAPI.SpVoice')
        voice.Rate = SAPI_VOICE_RATE
        voice.Volume = 100

        _select_voice(voice)

        stream = wincl.Dispatch('SAPI.SpMemoryStream')
        fmt = wincl.Dispatch('SAPI.SpAudioFormat')

        fmt.Type = SAFT16kHz16BitMono
        stream.Format = fmt

        voice.AudioOutputStream = stream

        voice.Speak(text)

        raw_pcm = stream.GetData()

        return create_wav_header(16000) + raw_pcm

    except Exception as e:
        logger.exception('SAPI synthesis failed: %r', e)
        return b''

    finally:
        pythoncom.CoUninitialize()


# -----------------------------------------------------------------------------
# Legacy Helper (optional)
# -----------------------------------------------------------------------------


async def convert_to_wave(text: str) -> bytes:
    """
    Maintained for compatibility.

    Converts the entire text into a single WAV.
    """
    return await to_thread(_synthesize_chunk, text)


# -----------------------------------------------------------------------------
# Streaming Producer
# -----------------------------------------------------------------------------


async def prefetch_audio(
    text: str,
    lang: str,
    audio_q: AudioQ,
):
    chunks = split_text(text)

    first_chunk = True

    for i, chunk in enumerate(chunks, start=1):
        if getattr(audio_q, 'is_shutdown', False):
            logger.debug('TTS aborted')
            return

        wav_bytes = await to_thread(
            _synthesize_chunk,
            chunk,
        )

        if not wav_bytes:
            continue

        if getattr(audio_q, 'is_shutdown', False):
            logger.debug('TTS aborted')
            return

        if first_chunk:
            # send WAV header once
            await audio_q.put(wav_bytes)

            first_chunk = False

        else:
            # remove WAV header, keep only PCM
            await audio_q.put(wav_bytes[44:])

        logger.debug('audio chunk %d sent (%d bytes)', i, len(wav_bytes))
