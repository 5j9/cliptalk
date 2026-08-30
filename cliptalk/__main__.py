__version__ = '0.1.dev0'

from asyncio import (
    Task,
    sleep,
    to_thread,
)
from multiprocessing import Pipe, Process

from aiohttp.web import (
    Application,
    Request,
    Response,
    run_app,
)
from config.ui import routes as ui_routes

from cliptalk import (
    aioh,
    config,
    create_task,
    in_q,
    logger,
    loop,
)
from cliptalk.aioh import routes
from cliptalk.qt_server import run_qt_app


@routes.put('/monitoring')
async def _(request: Request) -> Response:
    global monitoring
    logger.debug('/monitoring recieved request')
    monitoring = new_state = await request.json()
    conn.send(new_state)
    logger.info(f'monitoring state: {new_state}')
    return Response()


@routes.options('/q')
async def q_preflight_handler(request: Request) -> Response:
    """Handle CORS preflight for /q endpoint."""
    logger.debug('recieved preflight request')
    return Response(
        status=200,
        headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400',  # 24 hours
        },
    )


@routes.post('/q')
async def add_to_in_q(request: Request) -> Response:
    """Receive text data via POST request and add to processing queue."""
    body = await request.read()
    text = body.decode()
    await in_q.put(text.strip())
    logger.info(f'Added text to queue via /q endpoint: {text[:20]}...')
    return Response()


monitoring: bool = False
temp_monitor_task: Task | None = None


temp_monitor_task: Task | None = None


async def _send_false_later() -> None:
    await sleep(2.0)
    conn.send(False)


@routes.get('/temp_monitor')
async def temp_monitor(request: Request) -> Response:
    global temp_monitor_task

    conn.send(True)

    if temp_monitor_task is not None:
        temp_monitor_task.cancel()

    temp_monitor_task = create_task(_send_false_later())

    return Response()


async def listen_to_qt():
    """Monitor clipboard and add texts to queue."""
    global monitoring
    while True:
        try:
            data = await to_thread(conn.recv)

            if type(data) is bool:
                monitoring = data
                logger.debug(f'qt toggled monitoring: {data}')
                if aioh.current_ws is None:
                    continue
                await aioh.current_ws.send_json(
                    {'action': 'toggle-monitoring', 'state': data}
                )
                continue

            if type(data) is str:
                data = data.strip()
                await in_q.put(data)
                continue

            logger.error(f'Unexpected data type recieved from conn: {data=}')
        except Exception as e:
            logger.error(f'listen_to_qt loop failed with {e!r}')


if __name__ == '__main__':
    app = Application()
    app.add_routes(routes)
    app.add_routes(ui_routes)

    # loop.create_task(set_voice_names())

    qt_conn, conn = Pipe(True)
    conn.send((config.MIN_SPACE_RATIO, config.MIN_TEXT_LENGTH))
    qt_process = Process(target=run_qt_app, args=(qt_conn,))
    qt_process.start()
    listen_to_qt_task = create_task(listen_to_qt())

    try:
        run_app(app, host='127.0.0.1', port=3775, loop=loop)
    except KeyboardInterrupt:
        pass
    finally:
        qt_process.terminate()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
