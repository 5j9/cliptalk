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


def _create_wav_header() -> bytes:
    """
    Creates a streaming WAV header.

    The sizes are set to 0xFFFFFFFF because the final audio length
    is not known when streaming begins.
    """

    unknown_size = 0xFFFFFFFF

    header = struct.pack(
        '<4sI4s',
        b'RIFF',
        unknown_size,
        b'WAVE',
    )

    header += struct.pack(
        '<4sIHHIIHH',
        b'fmt ',
        16,  # PCM fmt chunk size
        1,  # PCM format
        CHANNELS,
        SAMPLE_RATE,
        BYTE_RATE,
        BLOCK_ALIGN,
        BITS_PER_SAMPLE,
    )

    header += struct.pack(
        '<4sI',
        b'data',
        unknown_size,
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

        return _create_wav_header() + raw_pcm

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

    logger.debug(
        'TTS split into %d chunks',
        len(chunks),
    )

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

        logger.debug(
            'audio chunk %d/%d sent (%d bytes)',
            i,
            len(chunks),
            len(wav_bytes),
        )
