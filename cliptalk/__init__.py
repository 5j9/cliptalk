from asyncio import CancelledError, Queue, QueueShutDown, Task, new_event_loop
from collections.abc import Awaitable, Callable
from functools import partial
from re import compile as rc

from aiohttp import ClientConnectionResetError
from applog import logger

from cliptalk import config

loop = new_event_loop()
create_task = loop.create_task
background_tasks = set[Task]()


class SizeUpdatingQ[T](Queue):
    def __init__(self, action: str, current_ws_container, maxsize=0):
        self.action = action
        super().__init__(maxsize)
        self.current_ws_container = current_ws_container

    async def put(self, item: T):
        await super().put(item)
        self.update_front_end_status()

    async def _update_front_end_status(self):
        current_ws = self.current_ws_container.get('current_ws')
        if current_ws is not None:
            try:
                await current_ws.send_json(
                    {
                        'action': self.action,
                        'value': f'{self.qsize()}/{self.maxsize}',
                    }
                )
            except ClientConnectionResetError:
                logger.warning('Could not update front-end status')

    def update_front_end_status(self):
        update_task = create_task(self._update_front_end_status())
        background_tasks.add(update_task)
        update_task.add_done_callback(background_tasks.discard)

    def task_done(self):
        super().task_done()
        self.update_front_end_status()


AudioQ = Queue[bytes]
# Queue to store incoming clipboard texts
InputQ = SizeUpdatingQ[str]
# Queue to store pre-generated audio data
OutputQ = SizeUpdatingQ[tuple[str, bool, AudioQ]]


in_q = InputQ(
    maxsize=500, action='input-queue-size', current_ws_container=globals()
)
out_q = OutputQ(
    maxsize=25, action='output-queue-size', current_ws_container=globals()
)


type Prefetchers = dict[str, Callable[[str, str, AudioQ], Awaitable]]


def load_engines() -> Prefetchers:
    """
    piper engine uses a lot more memory, but is usually more responsive.
    edge engine uses the Microsoft Edge tts servers.
    sapi uses Microsoft Speech API (SAPI). It has limited features,
        but is usually the most responsive one.
    """
    audio_prefetchers: Prefetchers = {}

    for lang, engine in config.ENGINES.items():
        match engine:
            case 'edge':
                from cliptalk.engines.edge import prefetch_audio

                audio_prefetchers[lang] = prefetch_audio

            case 'sapi':
                from cliptalk.engines.sapi import prefetch_audio

                audio_prefetchers[lang] = prefetch_audio

            case 'piper':
                from cliptalk.engines.piper import prefetch_audio

                audio_prefetchers[lang] = prefetch_audio
            case _:
                raise ValueError('unknown engine')

    return audio_prefetchers


remove_urls = partial(rc(r'https?://\S+').sub, 'URL')


async def prefetch_audio_loop(
    in_q: InputQ,
    out_q: OutputQ,
):
    """Prefetch audio for all texts in the queue."""
    audio_prefetchers = load_engines()
    from cliptalk.engines import detect_lang

    try:
        while True:
            text = await in_q.get()

            lang = detect_lang(text)
            text = remove_urls(text)
            short_text = text[:20] + '...'
            audio_q = AudioQ()
            await out_q.put((text, lang == 'fa', audio_q))
            fetcher = (
                audio_prefetchers.get(lang) or audio_prefetchers['default']
            )
            try:
                for _ in range(3):
                    try:
                        await fetcher(text, lang, audio_q)
                    except Exception as e:
                        logger.debug(f'Retrying {e!r}.')
                        continue
                    logger.info(f'Audio cached for: {short_text}')
                    break
            except QueueShutDown:
                logger.debug(f'audio_q QueueShutDown for {short_text}')
            except Exception as e:
                logger.error(
                    f'Error prefetching audio for {short_text}: {e!r}'
                )
            finally:
                logger.debug('calling audio_q.shutdown()')
                audio_q.shutdown()
                in_q.task_done()
    except Exception:
        logger.critical('Fatal Error')


prefetch_audio_task = create_task(prefetch_audio_loop(in_q, out_q))


async def reload_engines() -> None:
    global prefetch_audio_task
    prefetch_audio_task.cancel()
    try:
        await prefetch_audio_task
    except CancelledError:
        pass

    prefetch_audio_task = create_task(prefetch_audio_loop(in_q, out_q))
