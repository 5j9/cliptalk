from asyncio import to_thread
from collections.abc import Iterable
from functools import cache
from multiprocessing import Process
from multiprocessing.connection import Connection, PipeConnection
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


def stream_audio(
    audio_generator: Iterable[AudioChunk],
    sub_process_conn: Connection,
    sample_rate: int,
):
    sub_process_conn.send_bytes(create_wav_header(sample_rate=sample_rate))
    for chunk in audio_generator:
        sub_process_conn.send_bytes(chunk.audio_int16_bytes)
    sub_process_conn.send_bytes(b'')


def worker(sub_process_conn: Connection):
    while True:
        text, lang = sub_process_conn.recv()
        voice, syn_config = get_voice_config(lang)
        stream_audio(
            voice.synthesize(text, syn_config),
            sub_process_conn,
            voice.config.sample_rate,
        )
        logger.debug(f'Audio cached for {text[:20] + "..."!r}')


main_process_conn: PipeConnection


def start_sub_process(
    sub_process_conn: PipeConnection, main_process_conn_: PipeConnection
):
    global main_process_conn
    main_process_conn = main_process_conn_
    process = Process(target=worker, args=(sub_process_conn,), daemon=True)
    process.start()


async def prefetch_audio(text: str, lang: str, audio_q: AudioQ):
    main_process_conn.send((text, lang))
    while True:
        data = await to_thread(main_process_conn.recv_bytes)
        if not data:
            break
        await audio_q.put(data)
