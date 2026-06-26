import wave
from asyncio import sleep, to_thread
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path

from piper import AudioChunk, PiperVoice, SynthesisConfig

from cliptalk import AudioQ, logger

THIS_DIR = Path(__file__).parent


class VoiceConfig(dict[str, tuple[PiperVoice, SynthesisConfig]]):
    def __missing__(self, lang: str):
        if lang == 'fa':
            return (
                PiperVoice.load(THIS_DIR / 'voices/fa_IR-gyro-medium.onnx'),
                SynthesisConfig(length_scale=1.0),
            )

        return (
            PiperVoice.load(THIS_DIR / 'voices/en_US-hfc_male-medium.onnx'),
            SynthesisConfig(),
        )


voice_config = VoiceConfig()


def _create_wav_header(chunk: AudioChunk) -> bytes:
    """
    Create a streaming WAV header with unknown data size.
    """
    mock_wav_file = BytesIO()

    with wave.open(mock_wav_file, 'wb') as w:
        w.setframerate(chunk.sample_rate)
        w.setsampwidth(chunk.sample_width)
        w.setnchannels(chunk.sample_channels)

    mock_wav_file.seek(0)
    return mock_wav_file.read()


def _get_next(iterator: Iterator[AudioChunk]):
    return next(iterator, None)


async def stream_audio_to_q(
    audio_generator: Iterable[AudioChunk],
    audio_q: AudioQ,
):
    iterator = iter(audio_generator)

    first_chunk = True

    while True:
        chunk = await to_thread(_get_next, iterator)

        if chunk is None:
            break

        if first_chunk:
            await audio_q.put(_create_wav_header(chunk))
            first_chunk = False

        await audio_q.put(chunk.audio_int16_bytes)
        await sleep(0)


async def prefetch_audio(
    text: str,
    lang: str,
    audio_q: AudioQ,
):
    voice, syn_config = voice_config[lang]

    await stream_audio_to_q(
        voice.synthesize(text, syn_config),
        audio_q,
    )

    logger.debug(f'Audio cached for {text[:20] + "..."!r}')
