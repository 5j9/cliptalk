from asyncio import sleep, to_thread
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

from piper import AudioChunk, PiperVoice, SynthesisConfig

from cliptalk import AudioQ, logger
from cliptalk.engines import create_wav_header

THIS_DIR = Path(__file__).parent


@cache
def get_voice_config(lang: str) -> tuple[PiperVoice, SynthesisConfig]:
    logger.info(f'Loading Piper voice for {lang!r}')
    if lang == 'fa':
        return (
            PiperVoice.load(THIS_DIR / 'voices/fa_IR-gyro-medium.onnx'),
            SynthesisConfig(length_scale=1.0),
        )

    return (
        PiperVoice.load(THIS_DIR / 'voices/en_US-hfc_male-medium.onnx'),
        SynthesisConfig(),
    )


def _get_next(iterator: Iterator[AudioChunk]):
    return next(iterator, None)


async def stream_audio_to_q(
    audio_generator: Iterable[AudioChunk],
    audio_q: AudioQ,
    sample_rate: int,
):
    iterator = iter(audio_generator)
    await audio_q.put(create_wav_header(sample_rate=sample_rate))

    while True:
        chunk = await to_thread(_get_next, iterator)
        if chunk is None:
            break
        await audio_q.put(chunk.audio_int16_bytes)
        await sleep(0)


async def prefetch_audio(
    text: str,
    lang: str,
    audio_q: AudioQ,
):
    voice, syn_config = get_voice_config(lang)

    await stream_audio_to_q(
        voice.synthesize(text, syn_config),
        audio_q,
        voice.config.sample_rate,
    )
    logger.debug(f'Audio cached for {text[:20] + "..."!r}')
