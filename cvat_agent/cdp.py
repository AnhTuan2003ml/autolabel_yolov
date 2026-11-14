import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import websocket


ROOT = Path(__file__).resolve().parent.parent
CDP_ENDPOINT = 'http://127.0.0.1:9222/json'
ORIGIN = os.environ.get('CVAT_CDP_ORIGIN', '').rstrip('/')


def set_origin(url):
    global ORIGIN
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme and parsed.netloc:
        ORIGIN = f'{parsed.scheme}://{parsed.netloc}'
        os.environ['CVAT_CDP_ORIGIN'] = ORIGIN
    return ORIGIN


def origin():
    return ORIGIN


def origin_match(target, wanted):
    if not wanted:
        return False
    target_url = urllib.parse.urlsplit(target.get('url', ''))
    origin_url = urllib.parse.urlsplit(wanted)
    return target_url.scheme == origin_url.scheme and target_url.netloc == origin_url.netloc


def chrome_path():
    candidates = [
        Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),
        Path('C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'),
        Path.home() / 'AppData/Local/Google/Chrome/Application/chrome.exe',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError('Google Chrome was not found. Install Chrome or start Chrome with CDP port 9222.')


def cdp_targets():
    with urllib.request.urlopen(CDP_ENDPOINT, timeout=2) as response:
        return json.load(response)


def open_target(url):
    request = urllib.request.Request(f'{CDP_ENDPOINT}/new?{urllib.parse.quote(url, safe="")}', method='PUT')
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)


def ensure_browser(url):
    set_origin(url)
    try:
        targets = cdp_targets()
    except (urllib.error.URLError, TimeoutError, OSError):
        profile = ROOT / 'profile'
        profile.mkdir(exist_ok=True)
        subprocess.Popen([str(chrome_path()), '--remote-debugging-port=9222', '--remote-allow-origins=*', f'--user-data-dir={profile}', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                cdp_targets()
                return
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.5)
        raise RuntimeError('Chrome CDP is not ready at http://127.0.0.1:9222')
    if not any(target.get('type') == 'page' and origin_match(target, ORIGIN) for target in targets):
        try:
            open_target(url)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass


def page_target(url_part=None):
    try:
        targets = cdp_targets()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError('Cannot connect to Chrome CDP on port 9222') from error
    pages = [target for target in targets if target.get('type') == 'page']
    if ORIGIN:
        for target in pages:
            if origin_match(target, ORIGIN):
                return target
        raise RuntimeError(f'No Chrome page target for origin {ORIGIN}')
    if url_part:
        for target in pages:
            if url_part in target.get('url', ''):
                return target
    for target in pages:
        if '127.0.0.1:8787' in target.get('url', ''):
            return target
    if pages:
        return pages[0]
    raise RuntimeError('Browser page target was not found')


def evaluate(expression, url_part=None):
    target = page_target(url_part)
    socket = websocket.create_connection(target['webSocketDebuggerUrl'], timeout=20)
    try:
        socket.send(json.dumps({'id': 1, 'method': 'Runtime.evaluate', 'params': {'expression': expression, 'returnByValue': True, 'awaitPromise': True}}))
        while True:
            message = json.loads(socket.recv())
            if message.get('id') == 1:
                if 'exceptionDetails' in message.get('result', {}):
                    raise RuntimeError(json.dumps(message['result']['exceptionDetails']))
                return message['result']['result'].get('value')
    finally:
        socket.close()


if __name__ == '__main__':
    url_part = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(evaluate(sys.argv[1], url_part), ensure_ascii=False, indent=2))
