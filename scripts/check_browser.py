"""Local headless Edge UI check. Requires websocket-client in the test environment.

Start Edge with --headless=new --remote-debugging-port=8767 and a separate profile.
The application server must already be running at 127.0.0.1:8765.
"""
import base64
import json
import time
from pathlib import Path
from urllib.request import urlopen
import websocket

ROOT = Path(__file__).resolve().parents[1]
with urlopen('http://127.0.0.1:8767/json') as response:
    pages = json.load(response)
page = next(p for p in pages if p['type'] == 'page')
ws = websocket.create_connection(page['webSocketDebuggerUrl'], suppress_origin=True, timeout=30)
sequence = 0
def call(method, params=None):
    global sequence
    sequence += 1
    ws.send(json.dumps(dict(id=sequence, method=method, params=params or {})))
    while True:
        msg = json.loads(ws.recv())
        if msg.get('id') == sequence:
            if 'error' in msg:
                raise RuntimeError(msg['error'])
            return msg['result']

def evaluate(expression):
    result = call('Runtime.evaluate', dict(expression=expression, returnByValue=True, awaitPromise=True))
    if 'exceptionDetails' in result:
        raise RuntimeError(result['exceptionDetails'])
    return result['result'].get('value')

def wait_ready():
    for _ in range(100):
        if evaluate("!!document.querySelector('.card') && !document.querySelector('#run').disabled && document.querySelector('#status').textContent.startsWith('用户')"):
            return
        time.sleep(.15)
    raise TimeoutError('Page did not finish rendering')

try:
    call('Page.enable')
    call('Emulation.setDeviceMetricsOverride', dict(width=1440, height=1100, deviceScaleFactor=1, mobile=False))
    call('Page.navigate', dict(url='http://127.0.0.1:8765'))
    wait_ready()
    evaluate("document.querySelector('details').open=true")
    assert evaluate("document.querySelector('details').open")
    for method in ('selected','itemcf','popularity','semantic'):
        evaluate(f"document.querySelector('#method').value='{method}'; document.querySelector('#run').click()")
        wait_ready()
    evaluate("document.querySelector('#method').value='selected'; document.querySelector('#game').checked=true; document.querySelector('#rank').checked=false; document.querySelector('#run').click()")
    wait_ready()
    assert evaluate("Array.from(document.querySelectorAll('.card-head')).every(el=>el.textContent.includes('游戏'))")
    evaluate("document.querySelector('#game').checked=false; document.querySelector('#rank').checked=true; document.querySelector('#run').click()")
    wait_ready()
    for width, height, name in [(1440,1100,'desktop'),(390,1400,'mobile')]:
        call('Emulation.setDeviceMetricsOverride', dict(width=width, height=height, deviceScaleFactor=1, mobile=width < 500))
        time.sleep(.2)
        sizes = evaluate('({viewport:innerWidth,scroll:document.documentElement.scrollWidth})')
        assert sizes['scroll'] <= width, sizes
        data = call('Page.captureScreenshot', dict(format='png', captureBeyondViewport=False))
        (ROOT / f'reports/release/demo-{name}.png').write_bytes(base64.b64decode(data['data']))
    print('UI passed: 4 routes, score details, game filter, rank toggle, 1440px and 390px without horizontal overflow.')
finally:
    call('Browser.close')
    ws.close()
