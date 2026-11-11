import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import websocket


ROOT = Path(__file__).resolve().parent.parent
CDP_ENDPOINT = 'http://127.0.0.1:9222/json'


def chrome_path():
    candidates = [
        Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),
        Path('C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'),
        Path.home() / 'AppData/Local/Google/Chrome/Application/chrome.exe',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError('Không tìm thấy Google Chrome. Hãy cài Chrome hoặc mở Chrome thủ công với CDP 9222.')


def cdp_targets():
    with urllib.request.urlopen(CDP_ENDPOINT, timeout=2) as response:
        return json.load(response)


def ensure_browser(url):
    try:
        cdp_targets()
        return
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
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
    raise RuntimeError('Chrome CDP chưa sẵn sàng tại http://127.0.0.1:9222. Hãy kiểm tra Chrome và thử lại.')


def page_target(url_part='cvat.edufun.ai'):
    try:
        targets = cdp_targets()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError('Không kết nối được Chrome CDP 9222. Agent đã thử tự mở Chrome nhưng không thành công.') from error
    for target in targets:
        if target.get('type') == 'page' and url_part in target.get('url', ''):
            return target
    raise RuntimeError('CVAT page target was not found')


def evaluate(expression, url_part='cvat.edufun.ai'):
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
    url_part = sys.argv[2] if len(sys.argv) > 2 else 'cvat.edufun.ai'
    print(json.dumps(evaluate(sys.argv[1], url_part), ensure_ascii=False, indent=2))
