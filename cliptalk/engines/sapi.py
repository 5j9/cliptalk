import re
import struct
from asyncio import to_thread

import pythoncom
import win32com.client as wincl

from cliptalk import AudioQ, logger
from cliptalk.config import SAPI_VOICE_NAME, SAPI_VOICE_RATE

# -----------------------------------------------------------------------------
# SAPI Constants
# -----------------------------------------------------------------------------

SAFT16kHz16BitMono = 18

# -----------------------------------------------------------------------------
# Audio Specifications
# -----------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BLOCK_ALIGN = CHANNELS * (BITS_PER_SAMPLE // 8)
BYTE_RATE = SAMPLE_RATE * BLOCK_ALIGN

# -----------------------------------------------------------------------------
# Text Chunking
# -----------------------------------------------------------------------------


def split_text(text: str, target_chars: int = 250) -> list[str]:
    """
    Split text into reasonably sized chunks.

    We split on sentence boundaries and then combine nearby sentences until
    approximately target_chars is reached.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if current and current_len + len(sentence) > target_chars:
            chunks.append(' '.join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        chunks.append(' '.join(current))

    return chunks


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


def _create_wav_header(data_size: int) -> bytes:
    file_size = 36 + data_size

    header = struct.pack(
        '<4sI4s',
        b'RIFF',
        file_size,
        b'WAVE',
    )

    header += struct.pack(
        '<4sIHHIIHH',
        b'fmt ',
        16,
        1,  # PCM
        CHANNELS,
        SAMPLE_RATE,
        BYTE_RATE,
        BLOCK_ALIGN,
        BITS_PER_SAMPLE,
    )

    header += struct.pack(
        '<4sI',
        b'data',
        data_size,
    )

    return header


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

        return _create_wav_header(len(raw_pcm)) + raw_pcm

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
    """
    Synthesizes text in chunks and pushes each WAV chunk onto the queue
    immediately after it is ready.

    This greatly reduces startup latency and allows shutdown to interrupt
    processing between chunks.
    """
    chunks = split_text(text)

    logger.debug(
        'TTS split into %d chunks',
        len(chunks),
    )

    for i, chunk in enumerate(chunks, start=1):
        if getattr(audio_q, 'is_shutdown', False):
            logger.debug(
                'TTS aborted before chunk %d/%d',
                i,
                len(chunks),
            )
            return

        wave_bytes = await to_thread(
            _synthesize_chunk,
            chunk,
        )

        if not wave_bytes:
            continue

        await audio_q.put(wave_bytes)

        logger.debug(
            'Queued chunk %d/%d (%d bytes)',
            i,
            len(chunks),
            len(wave_bytes),
        )
