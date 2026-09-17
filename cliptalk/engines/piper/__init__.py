from asyncio import sleep, to_thread
from collections.abc import AsyncGenerator, Iterable
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


async def chunks_generator(
    audio_generator: Iterable[AudioChunk], sample_rate: int
) -> AsyncGenerator[bytes]:
    yield create_wav_header(sample_rate=sample_rate)
    audio_iterator = iter(audio_generator)
    next_chunk = lambda: next(audio_iterator, None)
    while True:
        chunk = await to_thread(next_chunk)
        if chunk is None:
            break
        yield chunk.audio_int16_bytes


async def prefetch_audio(text: str, lang: str, audio_q: AudioQ):
    voice, syn_config = get_voice_config(lang)
    async for data in chunks_generator(
        voice.synthesize(text, syn_config),
        voice.config.sample_rate,
    ):
        await audio_q.put(data)
        await sleep(0.1)
    logger.debug(f'Audio cached for {text[:20] + "..."!r}')
