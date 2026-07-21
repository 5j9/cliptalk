from asyncio import Queue, Task, create_task

from aiohttp import ClientConnectionResetError
from applog import logger

__version__ = '2025.09.19'

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
        raise NotImplementedError('use `atask_done` instead')

    async def atask_done(self):
        super().task_done()
        self.update_front_end_status()


AudioQ = Queue[bytes]
# Queue to store incoming clipboard texts
InputQ = SizeUpdatingQ[str]
# Queue to store pre-generated audio data
OutputQ = SizeUpdatingQ[tuple[str, bool, AudioQ]]
