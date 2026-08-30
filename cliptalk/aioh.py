import webbrowser
from asyncio import Event, QueueShutDown, sleep
from pathlib import Path

from aiohttp.web import (
    Request,
    Response,
    RouteTableDef,
    StreamResponse,
    WebSocketResponse,
)
from applog import logger

from cliptalk import create_task, out_q

next_request = Event()
routes = RouteTableDef()


@routes.get('/next')
async def _(request: Request) -> Response:
    logger.debug('recieved /next request')
    current_audio_q.shutdown(immediate=True)
    next_request.set()
    return Response()


current_ws: WebSocketResponse | None = None


@routes.get('/ws')
async def ws_handler(request):
    global current_audio_q, current_ws
    logger.info('new websocket connection')

    ws = WebSocketResponse()
    await ws.prepare(request)
    if current_ws and not current_ws.closed:
        logger.debug('closing current_ws before using new one')
        await current_ws.close()

    current_ws = ws

    while True:
        text, is_fa, audio_q = await out_q.get()
        logger.info('Sending new clipboard text to front-end.')
        # Store audio_q in request.app for /audio endpoint
        current_audio_q = audio_q
        next_request.clear()
        try:
            await ws.send_json(
                {'action': 'new-text', 'text': text, 'is_fa': is_fa}
            )
        except Exception as e:
            logger.exception(f'WebSocket error: {e}')
            await ws.close()
            return ws
        finally:
            out_q.task_done()
        logger.debug('awaiting next_request')
        await next_request.wait()


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
