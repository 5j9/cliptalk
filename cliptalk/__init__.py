from asyncio import Queue

from aiohttp import ClientConnectionResetError
from applog import logger

__version__ = '2025.09.19'


class SizeUpdatingQ[T](Queue):
    def __init__(self, action: str, current_ws_container, maxsize=0):
        self.action = action
        super().__init__(maxsize)
        self.current_ws_container = current_ws_container

    async def put(self, item: T):
        await super().put(item)
        await self.update_front_end_status()

    async def update_front_end_status(self):
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

    def task_done(self):
        raise NotImplementedError('use `atask_done` instead')

    async def atask_done(self):
        super().task_done()
        await self.update_front_end_status()


AudioQ = Queue[bytes]
# Queue to store incoming clipboard texts
InputQ = SizeUpdatingQ[str]
# Queue to store pre-generated audio data
OutputQ = SizeUpdatingQ[tuple[str, bool, AudioQ]]
