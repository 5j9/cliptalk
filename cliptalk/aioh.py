import webbrowser
from asyncio import (
    CancelledError,
    Event,
    QueueShutDown,
    Task,
    current_task,
    sleep,
)
from pathlib import Path

from aiohttp.web import (
    Request,
    Response,
    RouteTableDef,
    StreamResponse,
    WebSocketResponse,
)
from applog import logger

from cliptalk import AudioQ, create_task, out_q

next_request = Event()
routes = RouteTableDef()


@routes.get('/next')
async def _(request: Request) -> Response:
    logger.debug('recieved /next request')
    current_audio_q.shutdown(immediate=True)
    next_request.set()
    return Response()


current_ws: WebSocketResponse | None = None
current_audio_q: AudioQ
current_ws_task: Task | None = None


@routes.get('/ws')
async def ws_handler(request):
    global current_audio_q, current_ws, current_ws_task

    logger.info('new websocket connection')

    ws = WebSocketResponse()
    await ws.prepare(request)

    old_task = current_ws_task
    old_ws = current_ws

    if old_task is not None and old_task is not current_task():
        if old_ws is not None and not old_ws.closed:
            await old_ws.close()

        old_task.cancel()
        try:
            await old_task
        except CancelledError:
            pass

    current_ws = ws
    current_ws_task = current_task()

    try:
        while True:
            item = await out_q.get()
            try:
                text, is_fa, audio_q = item
                current_audio_q = audio_q
                next_request.clear()

                await ws.send_json(
                    {
                        'action': 'new-text',
                        'text': text,
                        'is_fa': is_fa,
                    }
                )
            finally:
                out_q.task_done()

            await next_request.wait()

    except CancelledError:
        logger.debug('WebSocket handler cancelled')
        raise

    except Exception:
        logger.exception('WebSocket error')

    finally:
        if current_ws is ws:
            current_ws = None
            current_ws_task = None

        if not ws.closed:
            await ws.close()

    return ws


this_dir = Path(__file__).parent


@routes.get('/cliptalk.html')
async def cliptalk_html(_):
    return Response(
        text=(this_dir / 'cliptalk.html')
        .read_text('utf8')
        .format(config_ui=(this_dir / 'config/ui.html').read_text('utf8')),
        content_type='text/html',
    )


@routes.get('/cliptalk.js')
async def cliptalk_js(_):
    return Response(
        text=(this_dir / 'cliptalk.js').read_text('utf8'),
        content_type='application/javascript',
    )


@routes.get('/cliptalk.css')
async def cliptalk_css(_):
    return Response(
        text=(this_dir / 'cliptalk.css').read_text('utf8'),
        content_type='text/css',
    )


audio_headers = {
    'Access-Control-Allow-Origin': '*',
    'Content-Type': 'audio/wav',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
}


@routes.get('/audio')
async def audio_handler(request: Request) -> StreamResponse:
    audio_q = current_audio_q
    logger.info('Serving audio started.')
    response = StreamResponse(status=200, reason='OK', headers=audio_headers)
    await response.prepare(request)
    try:
        while True:
            data = await audio_q.get()
            await response.write(data)
            audio_q.task_done()
    except QueueShutDown:
        logger.debug('/audio reached QueueShutDown')
    except Exception as e:
        logger.error(f'unexpected error: {e!r}')
    return response


async def open_tab_if_no_conn():
    await sleep(5.0)
    if current_ws is None:
        webbrowser.open('http://127.0.0.1:3775/cliptalk.html')


open_tab_task = create_task(open_tab_if_no_conn())
