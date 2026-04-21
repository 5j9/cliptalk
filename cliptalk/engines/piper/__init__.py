import wave
from asyncio import sleep
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

from piper import AudioChunk, PiperVoice, SynthesisConfig

from cliptalk import AudioQ, logger

THIS_DIR = Path(__file__).parent


class VoiceConfig(dict[str, tuple[PiperVoice, SynthesisConfig]]):
    def __missing__(self, lang: str):
        if lang == 'fa':
            # https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
            return PiperVoice.load(
                THIS_DIR / 'voices/fa_IR-gyro-medium.onnx'
            ), SynthesisConfig(length_scale=1.0)
        return PiperVoice.load(
            THIS_DIR / 'voices/en_US-hfc_male-medium.onnx'
        ), SynthesisConfig()


voice_config = VoiceConfig()


async def stream_audio_to_q(
    audio_generator: Iterable[AudioChunk], audio_q: AudioQ
):
    first_chunk = True

    for chunk in audio_generator:
        if first_chunk:
            # Create a mock WAV file in memory to get the header
            mock_wav_file = BytesIO()
            with wave.open(mock_wav_file, 'wb') as w:
                w.setframerate(chunk.sample_rate)
                w.setsampwidth(chunk.sample_width)
                w.setnchannels(chunk.sample_channels)
                # The wave library writes a header with a data chunk of size 0
                # when you close the file without writing frames.
                # It will automatically be updated with the actual size later.
                w.close()

            # Seek to the beginning and read the header bytes
            mock_wav_file.seek(0)
            wav_header = mock_wav_file.read()

            # The header written by the wave library has a placeholder size.
            # We need to send this placeholder first. The receiver should be
            # designed to update the size or handle a stream of unknown size.
            # For a true live stream, this is the best we can do.
            await audio_q.put(wav_header)

            first_chunk = False

        await audio_q.put(chunk.audio_int16_bytes)
        # `await audio_q.put` does not yield control unless audio_q is full
        await sleep(0.3)
        logger.debug('audio_q chunk sent')


async def prefetch_audio(text: str, lang: str, audio_q: AudioQ):
    voice, syn_config = voice_config[lang]
    short_text = repr(text[:20] + '...')
    await stream_audio_to_q(voice.synthesize(text, syn_config), audio_q)
    logger.debug(f'Audio cached for {short_text}')
