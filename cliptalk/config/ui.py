from pathlib import Path

from aiohttp.web import RouteTableDef, json_response
from aiohttp.web_request import Request
from aiohttp.web_response import Response
from applog import logger

from cliptalk import config, reload_engines

routes = RouteTableDef()

AVAILABLE_ENGINES = {'sapi', 'edge', 'piper'}


@routes.get('/config')
async def get_config(request: Request) -> Response:
    """Get current TTS configuration."""
    return json_response(
        {
            'engines': config.ENGINES,
        }
    )


@routes.post('/config')
async def update_config(request: Request) -> Response:
    """Update TTS configuration."""
    try:
        data = await request.json()
        engines = data.get('engines')

        if not isinstance(engines, dict):
            raise TypeError('engines must be an object')

        for key, engine in engines.items():
            if not isinstance(key, str):
                raise TypeError('engine keys must be strings')
            if engine not in AVAILABLE_ENGINES:
                raise ValueError(f'unknown TTS engine: {engine!r}')

        config.ENGINES |= engines
        await reload_engines()

        return json_response({'status': 'success'})

    except Exception as e:
        logger.exception(e)
        return json_response(
            {'status': 'error', 'message': str(e)},
            status=400,
        )


THIS_DIR = Path(__file__).parent


@routes.get('/config/ui.js')
async def ui_js(request: Request) -> Response:
    return Response(
        text=(THIS_DIR / 'ui.js').read_text('utf8'),
        content_type='application/javascript',
    )


@routes.get('/config/ui.css')
async def ui_css(request: Request) -> Response:
    return Response(
        text=(THIS_DIR / 'ui.css').read_text('utf8'),
        content_type='text/css',
    )
