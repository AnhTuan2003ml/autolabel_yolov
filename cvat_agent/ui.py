import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from .cdp import ensure_browser, evaluate
    from .label_agent import LabelDescriptionAgent
    from .shape_rules import allowed_labels
except ImportError:
    from cdp import ensure_browser, evaluate
    from label_agent import LabelDescriptionAgent
    from shape_rules import allowed_labels

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
AGENT_HOST = os.environ.get('CVAT_AGENT_HOST', '127.0.0.1')
AGENT_PORT = int(os.environ.get('CVAT_AGENT_PORT', '8787'))
AGENT_TOKEN = os.environ.get('CVAT_AGENT_TOKEN', '')
JOBS = {}
PHASES = ['Kết nối CVAT', 'Metadata', 'Labels', 'Prompt', 'Frames', 'Inference', 'Kiểm tra', 'Upload']

PAGE = '''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CVAT Agent</title><style>
:root{--bg:#212121;--panel:#171717;--card:#2b2b2b;--line:#454545;--text:#ececec;--muted:#a5a5a5;--green:#10a37f;--blue:#5b9cff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui}.app{display:grid;grid-template-columns:280px 1fr;min-height:100vh}.side{background:var(--panel);border-right:1px solid #303030;padding:14px;display:flex;flex-direction:column;gap:14px}.brand{display:flex;gap:10px;align-items:center;font-weight:700;font-size:16px;padding:9px}.mark{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;background:linear-gradient(135deg,var(--green),var(--blue))}.side button,.side input,.side select{width:100%;border-radius:8px;padding:10px;background:#222;color:var(--text);border:1px solid #4b4b4b}.side button{cursor:pointer;text-align:left}.side button:hover{background:#333}.config{border-top:1px solid #333;padding-top:14px}.config h4{font-size:11px;color:var(--muted);font-weight:500;margin:0 0 12px}.field{margin:11px 0}.field label{display:block;font-size:12px;color:#bbb;margin-bottom:6px}.side .bottom{margin-top:auto}.main{min-width:0;display:flex;flex-direction:column}.top{height:58px;border-bottom:1px solid #333;display:flex;align-items:center;justify-content:space-between;padding:0 28px}.status{color:#9de4d1}.content{width:min(1000px,100%);margin:auto;padding:30px 24px 190px;flex:1}.welcome{text-align:center;padding:28px 0}.welcome h1{font-size:29px;margin:0 0 8px}.welcome p{color:var(--muted);margin:0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:22px 0}.stat,.card{background:var(--card);border:1px solid #3b3b3b;border-radius:12px}.stat{padding:12px}.stat span{display:block;color:#999;font-size:10px}.stat strong{font-size:19px;display:block;margin-top:6px}.card{padding:18px}.current{font-size:18px;font-weight:600}.progress{height:7px;background:#171717;border-radius:7px;overflow:hidden;margin:14px 0 20px}.progress i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--green),var(--blue));transition:.35s}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.step{min-height:68px;border:1px solid #444;background:#222;border-radius:8px;padding:10px;color:#777}.step b{font-size:11px}.step small{display:block;margin-top:7px;color:#999}.step.done{color:#b9f2df;border-color:#18785f;background:#14382f}.step.active{color:#e8f2ff;border-color:var(--blue);background:#1d3551;animation:pulse 1.5s infinite}.activity{margin-top:14px}.activity summary{color:#aaa;cursor:pointer}.console{height:180px;overflow:auto;white-space:pre-wrap;background:#111;border:1px solid #333;border-radius:8px;margin-top:10px;padding:10px;color:#b8e8c7;font:12px/1.5 Consolas}.composer-wrap{position:fixed;left:280px;right:0;bottom:0;padding:34px 24px 18px;background:linear-gradient(transparent,var(--bg) 25%)}.composer{width:min(1000px,100%);margin:auto;background:#303030;border:1px solid #565656;border-radius:15px;padding:10px;box-shadow:0 10px 35px #0009}.messages{max-height:210px;overflow:auto}.msg{max-width:85%;margin:6px 0;padding:9px 11px;border-radius:10px;line-height:1.4}.msg.user{margin-left:auto;background:#3b3b3b}.msg.agent{background:#17382f;color:#c9f5e8}.row{display:flex;gap:9px;align-items:flex-end}.row textarea{flex:1;min-height:62px;max-height:180px;resize:vertical;background:#151515;color:var(--text);border:1px solid #4c4c4c;border-radius:9px;padding:10px;outline:0}.row textarea:focus{border-color:var(--green)}.send,.reset{border:0;border-radius:8px;padding:11px 14px;color:#fff;font-weight:600}.send{background:var(--green)}.reset{background:#555}.hint{font-size:11px;color:#888;padding:5px}.send:disabled{opacity:.5}@keyframes pulse{50%{box-shadow:0 0 0 4px #5b9cff22}}@media(max-width:850px){.app{grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid #333}.composer-wrap{left:0}.steps{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.content{padding:16px 12px 210px}.stats{grid-template-columns:repeat(2,1fr)}.top{padding:0 14px}.composer-wrap{padding:24px 10px 10px}.row{flex-direction:column}.row button{width:100%}}
</style></head><body><div class="app"><aside class="side"><div class="brand"><span class="mark">A</span>CVAT Agent</div><button id="profile">＋ Mở profile đăng nhập</button><div class="config"><h4>CẤU HÌNH NHIỆM VỤ</h4><div class="field"><label>CVAT job URL</label><input id="url" value="https://cvat.edufun.ai/tasks/56/jobs/130"></div><div class="field"><label>Organization</label><input id="org" value="cohort-5a"></div><div class="field"><label>Kiểu annotation</label><select id="mode"><option value="segmentation">Segmentation polygon</option><option value="bbox">Bounding box</option></select></div><div class="field"><label>Điểm polygon tối đa</label><input id="points" type="number" min="0" max="60" value="60"></div></div><div class="bottom"><button id="resetSide">↻ Bắt đầu lại</button></div></aside><main class="main"><header class="top"><span>Agent workspace</span><span class="status" id="state">Sẵn sàng</span></header><section class="content"><div class="welcome"><h1>Agent có thể giúp gì cho bạn?</h1><p>Dán yêu cầu để Agent chọn label, tạo prompt và chạy CVAT.</p></div><div class="stats"><div class="stat"><span>TASK</span><strong id="task">—</strong></div><div class="stat"><span>JOB</span><strong id="job">—</strong></div><div class="stat"><span>LABEL</span><strong id="labels">—</strong></div><div class="stat"><span>FRAME</span><strong id="frames">—</strong></div></div><section class="card"><div class="current" id="current">Chưa có nhiệm vụ</div><div class="progress"><i id="bar"></i></div><div class="steps" id="steps"></div><details class="activity"><summary>Activity log</summary><pre class="console" id="console">Sẵn sàng.</pre></details></section></section></main></div><div class="composer-wrap"><div class="composer"><div class="messages" id="messages"></div><div class="row"><textarea id="prompt" placeholder="Nhập yêu cầu... Ví dụ: chỉ segment car, traffic light và road"></textarea><button class="send" id="send">Gửi</button><button class="reset" id="reset">↻</button></div><div class="hint">Enter để gửi · Shift+Enter để xuống dòng</div></div></div><script>
const phases=['Kết nối CVAT','Metadata','Labels','Prompt','Frames','Inference','Kiểm tra','Upload'];const $=id=>document.getElementById(id);let runId;function message(text,role='agent'){const el=document.createElement('div');el.className='msg '+role;el.textContent=text;$('messages').appendChild(el);$('messages').scrollTop=$('messages').scrollHeight}function steps(active,detail){$('steps').innerHTML=phases.map((name,index)=>`<div class="step ${index<active-1?'done':''} ${index===active-1?'active':''}"><b>${String(index+1).padStart(2,'0')} · ${name}</b><small>${index===active-1?(detail||'Đang xử lý...'):index<active-1?'Hoàn tất':''}</small></div>`).join('')}steps(0,'Chờ yêu cầu');async function poll(){const result=await fetch('/api/status?id='+runId).then(r=>r.json());$('console').textContent=result.output||result.error||'Đang khởi động';$('state').textContent=result.state==='running'?'Đang chạy':result.state==='done'?'Hoàn tất':result.state==='error'?'Lỗi':'Sẵn sàng';$('current').textContent=result.phase||'Đang khởi động';$('bar').style.width=(result.progress||0)+'%';steps(result.phase_index||0,result.phase);if(result.task_id)$('task').textContent=result.task_id;if(result.job_id)$('job').textContent=result.job_id;if(result.labels!=null)$('labels').textContent=result.labels;if(result.frame_current!=null)$('frames').textContent=result.frame_current+'/'+(result.frame_total||'?');if(result.state==='running')setTimeout(poll,600)}async function labels(){const response=await fetch('/api/inspect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Không lấy được label');return result.descriptions}async function submit(){const text=$('prompt').value.trim();if(!text){message('Hãy nhập yêu cầu trước khi gửi.');return}$('send').disabled=true;try{message(text,'user');message('Đang đọc label và tạo mô tả...');const descriptions=await labels();const response=await fetch('/api/chat-filter',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,descriptions})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Không lọc được label');const selected=result.selected.map(index=>descriptions[index]);if(!selected.length)throw new Error('Agent không tìm thấy label phù hợp');message(`Đã chọn ${selected.length}/${descriptions.length} label: ${selected.map(item=>item.label).join(', ')}`);const run=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value,organization:$('org').value,model:'sam3',mode:$('mode').value,max_points:$('points').value,selected_prompts:selected})});const data=await run.json();if(!run.ok)throw new Error(data.error||'Không khởi động được job');runId=data.id;poll()}catch(error){message(error.message);$('state').textContent='Lỗi'}finally{$('send').disabled=false}}function reset(){runId=null;$('messages').innerHTML='';$('console').textContent='Sẵn sàng.';$('state').textContent='Sẵn sàng';$('current').textContent='Chưa có nhiệm vụ';$('bar').style.width='0%';$('task').textContent='—';$('job').textContent='—';$('labels').textContent='—';$('frames').textContent='—';steps(0,'Chờ yêu cầu')}$('send').onclick=submit;$('prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submit()}};$('reset').onclick=reset;$('resetSide').onclick=reset;$('profile').onclick=async()=>{const r=await fetch('/api/open-profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value})});const data=await r.json();$('state').textContent=data.error||'Đã mở profile'};
</script></body></html>'''

PAGE = PAGE.replace('</details></section></section></main></div><div class="composer-wrap">', '</details></section><section class="chat-row"><div class="messages" id="messages"></div></section></section></main></div><div class="composer-wrap">')
PAGE = PAGE.replace('<div class="messages" id="messages"></div><div class="row">', '<div class="row">')
PAGE = PAGE.replace('.content{width:min(1000px,100%);margin:auto;padding:30px 24px 190px;flex:1}', '.content{width:min(1000px,100%);margin:auto;padding:30px 24px 190px;flex:1;display:flex;flex-direction:column;min-height:calc(100vh - 58px)}')
PAGE = PAGE.replace('.messages{max-height:210px;overflow:auto}', '.chat-row{flex:1;min-height:180px;background:#191919;border:1px solid #383838;border-radius:12px;padding:12px;margin-top:14px;overflow:hidden}.chat-row .messages{height:100%;max-height:none;overflow:auto}.messages{max-height:210px;overflow:auto}')


def inspect_job(url):
    match = re.search(r'/jobs/(\d+)', url)
    if not match:
        raise ValueError('URL must contain /jobs/<job>')
    job_id = int(match.group(1))
    ensure_browser(url)
    evaluate(f'location.href={json.dumps(url)}')
    time.sleep(1)
    labels = []
    page = 1
    while True:
        data = evaluate(f"fetch('/api/labels?job_id={job_id}&page={page}').then(r=>r.json())")
        labels.extend(item['name'] for item in data.get('results', []))
        if not data.get('next'):
            break
        page += 1
    review_id = uuid.uuid4().hex
    path = RUNS / f'review_{review_id}.json'
    return review_id, LabelDescriptionAgent().build(allowed_labels(labels), path)


def filter_labels(message, descriptions):
    text = message.strip().lower().replace('ligt', 'light')
    groups = {'người': {'person', 'pedestrian', 'rider'}, 'xe': {'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'rider'}, 'light': {'traffic light', 'traffic_light'}, 'cây': {'vegetation', 'tree', 'trees'}, 'thiên nhiên': {'vegetation', 'terrain', 'sky'}, 'đường': {'road', 'sidewalk', 'lane/crosswalk', 'lane/double white', 'lane/double yellow', 'lane/road curb', 'lane/single other', 'lane/single white', 'lane/single yellow'}, 'nhà': {'building', 'wall', 'fence'}}
    wanted = set()
    for key, values in groups.items():
        if key in text:
            wanted.update(values)
    for item in descriptions:
        label = item['label'].lower()
        if label in text or label.replace('_', ' ') in text:
            wanted.add(label)
    return [index for index, item in enumerate(descriptions) if item['label'].lower() in wanted or item['label'].lower().replace('_', ' ') in wanted]


def read_json_body(handler):
    body = handler.rfile.read(int(handler.headers.get('Content-Length', '0')))
    for encoding in ('utf-8', 'utf-8-sig', 'cp1258', 'cp1252'):
        try:
            return json.loads(body.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError('Invalid JSON request body')


def update(run_id, line):
    job = JOBS[run_id]
    match = re.search(r'\[agent (\d+)/(\d+)\] (.*)', line)
    if match:
        index = int(match.group(1))
        job['phase_index'] = index
        job['progress'] = round(index / int(match.group(2)) * 100)
        job['phase'] = match.group(3)
        ids = re.search(r'task=(\d+), job=(\d+)', line)
        if ids:
            job.update(task_id=int(ids.group(1)), job_id=int(ids.group(2)))
    frame = re.search(r'\[frame (\d+)/(\d+)\]', line)
    if frame:
        job['frame_current'] = int(frame.group(1))
        job['frame_total'] = int(frame.group(2))
        job['phase'] = f'Inference: frame {frame.group(1)}/{frame.group(2)}'


def start(payload, run_id):
    JOBS[run_id] = {'state': 'running', 'output': '', 'phase': 'Đang khởi động', 'phase_index': 0, 'progress': 0}
    command = [sys.executable, '-m', 'cvat_agent.job_runner', payload['url'], payload.get('organization', ''), payload.get('model', 'sam3'), payload.get('mode', 'segmentation'), str(RUNS), str(payload.get('max_points', 60))]
    if payload.get('prompt_file'):
        command.append(payload['prompt_file'])
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    lines = []
    for line in process.stdout:
        value = line.rstrip()
        lines.append(value)
        update(run_id, value)
        JOBS[run_id]['output'] = '\n'.join(lines[-160:])
    code = process.wait()
    JOBS[run_id]['state'] = 'done' if code == 0 else 'error'
    JOBS[run_id]['progress'] = 100 if code == 0 else JOBS[run_id]['progress']
    if code == 0:
        JOBS[run_id]['phase_index'] = len(PHASES) + 1
        JOBS[run_id]['phase'] = 'Hoàn tất'
    if code:
        JOBS[run_id]['error'] = JOBS[run_id]['output']


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/health':
            self.send_json({'state': 'ready'})
        elif self.path.startswith('/api/status'):
            self.send_json(JOBS.get(self.path.split('id=', 1)[-1], {'state': 'missing'}))
        elif self.path == '/api/agent/schema':
            self.send_json({'endpoint': '/api/agent/chat', 'method': 'POST'})
        else:
            self.send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path == '/api/chat-filter':
            try:
                payload = read_json_body(self)
                self.send_json({'selected': filter_labels(payload.get('message', ''), payload.get('descriptions', []))})
            except Exception as error:
                self.send_json({'error': str(error)}, 500)
            return
        if self.path == '/api/inspect':
            try:
                payload = read_json_body(self)
                review_id, descriptions = inspect_job(payload.get('url', ''))
                self.send_json({'id': review_id, 'descriptions': descriptions})
            except Exception as error:
                self.send_json({'error': str(error)}, 500)
            return
        if self.path == '/api/open-profile':
            try:
                payload = read_json_body(self)
                ensure_browser(payload.get('url', 'https://cvat.edufun.ai'))
                evaluate(f"location.href={json.dumps(payload.get('url', 'https://cvat.edufun.ai'))}")
                self.send_json({'state': 'ready'})
            except Exception as error:
                self.send_json({'error': str(error)}, 500)
            return
        if self.path == '/api/agent/chat':
            if AGENT_TOKEN and self.headers.get('X-Agent-Token', '') != AGENT_TOKEN:
                self.send_json({'error': 'Unauthorized'}, 401)
                return
            try:
                payload = read_json_body(self)
                descriptions = inspect_job(payload.get('url', ''))[1]
                selected = [descriptions[index] for index in filter_labels(payload.get('message', ''), descriptions)]
                if not selected:
                    self.send_json({'error': 'Agent không tìm thấy label phù hợp'}, 400)
                    return
                run_id = uuid.uuid4().hex
                prompt_file = RUNS / f'prompt_{run_id}.json'
                prompt_file.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
                payload['prompt_file'] = str(prompt_file)
                threading.Thread(target=start, args=(payload, run_id), daemon=True).start()
                self.send_json({'id': run_id, 'state': 'running', 'selected': selected})
            except Exception as error:
                self.send_json({'error': str(error)}, 500)
            return
        if self.path != '/api/run':
            self.send_json({'error': 'Not found'}, 404)
            return
        payload = read_json_body(self)
        if not payload.get('url'):
            self.send_json({'error': 'Job URL is required'}, 400)
            return
        points = int(payload.get('max_points', 60))
        if points < 0 or points > 60:
            self.send_json({'error': 'Số điểm phải từ 0 đến 60'}, 400)
            return
        payload['max_points'] = points
        run_id = uuid.uuid4().hex
        if payload.get('selected_prompts'):
            prompt_file = RUNS / f'prompt_{run_id}.json'
            prompt_file.write_text(json.dumps(payload['selected_prompts'], ensure_ascii=False, indent=2), encoding='utf-8')
            payload['prompt_file'] = str(prompt_file)
        threading.Thread(target=start, args=(payload, run_id), daemon=True).start()
        self.send_json({'id': run_id})

    def log_message(self, format, *args):
        return


if __name__ == '__main__':
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((AGENT_HOST, AGENT_PORT), Handler)
    threading.Timer(0.8, lambda: webbrowser.open(f'http://127.0.0.1:{AGENT_PORT}')).start()
    server.serve_forever()
