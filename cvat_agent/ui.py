import json
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
except ImportError:
    from cdp import ensure_browser, evaluate
    from label_agent import LabelDescriptionAgent

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
JOBS = {}
REVIEWS = {}
PHASES = ['Kết nối CVAT', 'Đọc metadata', 'Lấy label', 'Sinh mô tả', 'Tải frame', 'Chạy SAM3', 'Hậu kiểm', 'Upload']

PAGE = '''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CVAT Agent</title><style>
*{box-sizing:border-box}body{margin:0;background:#07111f;color:#e6edf5;font:14px system-ui;min-height:100vh}.wrap{max-width:1240px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}.eyebrow{color:#62c9ff;text-transform:uppercase;letter-spacing:2px;font-size:11px}.title{font-size:28px;font-weight:800;margin:5px 0}.muted{color:#8da2b8}.status{border:1px solid #245477;border-radius:99px;padding:8px 14px;color:#8fe1ff;background:#0b2539}.grid{display:grid;grid-template-columns:360px 1fr;gap:18px}.panel,.card{background:#0c1b2e;border:1px solid #1d3b59;border-radius:16px;padding:18px}.panel h2{font-size:15px;margin:0 0 16px}.field{margin:14px 0}.field label{display:block;color:#a9bad0;margin-bottom:6px;font-size:12px}input,select,button{width:100%;border:1px solid #2c4c6a;border-radius:9px;background:#081625;color:#f3f8ff;padding:11px;font-size:14px}button{background:#1285da;border-color:#39b8f3;font-weight:800;cursor:pointer;margin-top:12px}button:disabled{opacity:.5}.hint{color:#7891aa;font-size:12px;line-height:1.5;margin-top:14px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:18px}.stat{background:#0b1c30;border:1px solid #1b3a58;border-radius:11px;padding:12px}.stat span{display:block;color:#7f97af;font-size:10px}.stat strong{font-size:20px}.current{font-size:18px;font-weight:800}.progress{height:9px;background:#071321;border-radius:99px;margin:13px 0 18px;overflow:hidden}.progress i{display:block;height:100%;background:linear-gradient(90deg,#25d6d1,#3985ff);width:0;transition:width .3s}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:20px}.node{position:relative;min-height:78px;padding:12px;border:1px solid #264561;border-radius:10px;color:#7189a3;background:#0b1a2c}.node:after{content:'';position:absolute;right:-8px;top:38px;width:8px;height:2px;background:#264561;z-index:2}.node:nth-child(4):after,.node:nth-child(8):after{display:none}.node.done{color:#98efbf;border-color:#25865c;background:#0b2b2a}.node.done:after{background:#27c987}.node.active{color:#d9f7ff;border-color:#3cc8fa;background:#103452;box-shadow:0 0 0 2px #3cc8fa33}.node.active:before{content:'';position:absolute;right:10px;top:10px;width:10px;height:10px;border:2px solid #65ddff;border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite}.node b{font-size:11px}.node small{display:block;color:#91a9bf;font-size:11px;line-height:1.35;margin-top:7px}.node.active small{color:#c9f4ff}.console{background:#040a12;border:1px solid #1b354e;border-radius:10px;padding:13px;min-height:220px;max-height:380px;overflow:auto;white-space:pre-wrap;color:#baf4d1;font:12px/1.5 Consolas,monospace}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:850px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:repeat(2,1fr)}.node:nth-child(2n):after{display:none}}
</style></head><body><div class="wrap"><div class="top"><div><div class="eyebrow">Agent orchestration</div><div class="title">CVAT Agent Control Center</div><div class="muted">Theo dõi nhiệm vụ theo từng bước xử lý</div></div><div class="status" id="state">Sẵn sàng</div></div><div class="grid"><section class="panel"><h2>Tạo nhiệm vụ</h2><div class="field"><label>CVAT job URL</label><input id="url" value="https://cvat.edufun.ai/tasks/56/jobs/130"></div><div class="field"><label>Organization</label><input id="org" value="cohort-5a"></div><div class="field"><label>Kiểu annotation</label><select id="mode"><option value="segmentation">Segmentation polygon</option><option value="bbox">Bounding box</option></select></div><div class="field" id="pointsBox"><label>Số điểm tối đa mỗi polygon</label><input id="points" type="number" min="0" max="1000" value="80"></div><div class="hint">Bounding box không dùng point. Segmentation có thể giới hạn số điểm polygon.</div><button id="run">Chạy toàn bộ frame</button></section><main><div class="stats"><div class="stat"><span>TASK</span><strong id="task">—</strong></div><div class="stat"><span>JOB</span><strong id="job">—</strong></div><div class="stat"><span>LABEL</span><strong id="labels">—</strong></div><div class="stat"><span>FRAME</span><strong id="frames">—</strong></div></div><section class="card"><div class="current" id="current">Chưa có nhiệm vụ</div><div class="progress"><i id="bar"></i></div><div class="flow" id="flow"></div><details><summary>Console kỹ thuật</summary><pre class="console" id="console">Sẵn sàng.</pre></details></section></main></div></div><script>
const phases=['Kết nối CVAT','Đọc metadata','Lấy label','Sinh mô tả','Tải frame','Chạy SAM3','Hậu kiểm','Upload'];const $=id=>document.getElementById(id);let runId;function flow(active,detail){$('flow').innerHTML=phases.map((name,index)=>`<div class="node ${index<active-1?'done':''} ${index===active-1?'active':''}"><b>${String(index+1).padStart(2,'0')} · ${name}</b><small>${index===active-1?(detail||'Đang xử lý...'):index<active-1?'Hoàn tất':''}</small></div>`).join('')}flow(0,'Chờ chạy');$('mode').onchange=()=>{$('pointsBox').style.display=$('mode').value==='bbox'?'none':'block'};async function poll(){const result=await fetch('/api/status?id='+runId).then(response=>response.json());$('console').textContent=result.output||result.error||'Đang khởi động';$('state').textContent=result.state==='running'?'Đang chạy':result.state==='done'?'Hoàn tất':'Lỗi';$('current').textContent=result.phase||'Đang khởi động';$('bar').style.width=(result.progress||0)+'%';flow(result.phase_index||0,result.phase);if(result.task_id)$('task').textContent=result.task_id;if(result.job_id)$('job').textContent=result.job_id;if(result.labels!=null)$('labels').textContent=result.labels;if(result.frame_current!=null)$('frames').textContent=result.frame_current+'/'+(result.frame_total||'?');if(result.state==='running')setTimeout(poll,500);else $('run').disabled=false}$('run').onclick=async()=>{$('run').disabled=true;const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value,organization:$('org').value,model:'sam3',mode:$('mode').value,max_points:$('points').value})});const result=await response.json();if(!response.ok){$('console').textContent=result.error;$('run').disabled=false;return}runId=result.id;poll()};
</script></body></html>'''

PAGE = PAGE.replace('</style>', '.top-actions{display:flex;align-items:center;gap:10px}.profile-button{width:auto;margin:0;padding:9px 14px}.chat-panel{display:flex;flex-direction:column;min-height:560px}.chat-title{display:flex;justify-content:space-between;align-items:center}.chat-log{flex:1;min-height:320px;background:#081625;border:1px solid #1b3a58;border-radius:10px;padding:12px;overflow:auto;color:#b9cce0;line-height:1.55}.chat-message{padding:9px 11px;border-radius:10px;margin-bottom:9px;background:#102b42}.chat-message.agent{background:#0b2b2a;color:#baf4d1}.chat-input{min-height:92px;resize:vertical;margin-top:12px}.chat-button{margin-top:10px}')
PAGE = PAGE.replace('.grid{display:grid;grid-template-columns:360px 1fr;gap:18px}', '.grid{display:grid;grid-template-columns:320px minmax(0,1fr) 350px;gap:18px}')
PAGE = PAGE.replace('<div class="status" id="state">Sẵn sàng</div>', '<div class="top-actions"><button id="profile" class="profile-button">Mở profile đăng nhập</button><div class="status" id="state">Sẵn sàng</div></div>')
PAGE = PAGE.replace('</main></div></div><script>', '<aside class="panel chat-panel"><div class="chat-title"><h2>Agent chat</h2><span class="muted">Tự lọc label</span></div><div class="chat-log" id="chatLog"><div class="chat-message agent">Nhập yêu cầu cho nhiệm vụ, ví dụ: chỉ segment car và traffic light.</div></div><textarea id="agentChat" class="chat-input" placeholder="Ví dụ: chỉ seg car, light và road"></textarea><button id="agentRun" class="chat-button">Bắt đầu theo yêu cầu</button></aside></main></div></div><script>')
PAGE = PAGE.replace('</script></body></html>', r'''$('profile').onclick=async()=>{const response=await fetch('/api/open-profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value})});const result=await response.json();$('state').textContent=result.error||'Đã mở profile'};function chat(text,agent=false){$('chatLog').insertAdjacentHTML('beforeend',`<div class="chat-message ${agent?'agent':''}">${text}</div>`);$('chatLog').scrollTop=$('chatLog').scrollHeight}async function inspectLabels(){const response=await fetch('/api/inspect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Không lấy được label');return result.descriptions}$('agentRun').onclick=async()=>{const message=$('agentChat').value.trim();if(!message){chat('Hãy nhập yêu cầu label trước khi bắt đầu.',true);return}$('agentRun').disabled=true;$('run').disabled=true;try{chat(message);chat('Đang lấy label và tạo mô tả...',true);const descriptions=await inspectLabels();const filterResponse=await fetch('/api/chat-filter',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,descriptions})});const filterResult=await filterResponse.json();if(!filterResponse.ok)throw new Error(filterResult.error||'Không lọc được label');const selected=filterResult.selected.map(index=>descriptions[index]);if(!selected.length)throw new Error('Agent không tìm thấy label phù hợp');chat(`Đã chọn ${selected.length}/${descriptions.length} label: ${selected.map(item=>item.label).join(', ')}`,true);const runResponse=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value,organization:$('org').value,model:'sam3',mode:$('mode').value,max_points:$('points').value,selected_prompts:selected})});const runResult=await runResponse.json();if(!runResponse.ok)throw new Error(runResult.error||'Không khởi động được nhiệm vụ');runId=runResult.id;poll()}catch(error){chat(error.message,true);$('state').textContent='Lỗi';$('agentRun').disabled=false;$('run').disabled=false}};$('run').style.display='none';</script></body></html>''')


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
    descriptions = LabelDescriptionAgent().build(list(dict.fromkeys(labels)), path)
    REVIEWS[review_id] = path
    return review_id, descriptions


def filter_labels(message, descriptions):
    text = message.strip().lower()
    if not text:
        return list(range(len(descriptions)))
    groups = {
        'người': {'person', 'pedestrian', 'rider'},
        'xe': {'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'rider'},
        'giao thông': {'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic_light', 'traffic sign', 'traffic_sign'},
        'cây': {'vegetation', 'tree', 'trees'},
        'thiên nhiên': {'vegetation', 'terrain', 'sky'},
        'đường': {'road', 'sidewalk', 'lane/crosswalk', 'lane/double white', 'lane/double yellow', 'lane/road curb', 'lane/single other', 'lane/single white', 'lane/single yellow'},
        'nhà': {'building', 'wall', 'fence'},
    }
    terms = {word for word in re.split(r'[,;\s]+', text) if word}
    wanted = set()
    for key, values in groups.items():
        if key in text:
            wanted.update(values)
    for item in descriptions:
        normalized = item['label'].lower().replace('_', ' ')
        if normalized in terms or item['label'].lower() in text or normalized in text:
            wanted.add(item['label'].lower())
    return [index for index, item in enumerate(descriptions) if item['label'].lower() in wanted or item['label'].lower().replace('_', ' ') in wanted]


def read_json_body(handler):
    length = int(handler.headers.get('Content-Length', '0'))
    body = handler.rfile.read(length)
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
        labels = re.search(r'Tạo (\d+) mô tả', line)
        if ids:
            job.update(task_id=int(ids.group(1)), job_id=int(ids.group(2)))
        if labels:
            job['labels'] = int(labels.group(1))
    frame = re.search(r'\[frame (\d+)/(\d+)\]', line)
    if frame:
        job['frame_current'] = int(frame.group(1))
        job['frame_total'] = int(frame.group(2))
        job['phase'] = f'Chạy SAM3: frame {frame.group(1)}/{frame.group(2)}'


def start(payload, run_id):
    JOBS[run_id] = {'state': 'running', 'output': '', 'phase': 'Đang khởi động', 'phase_index': 0, 'progress': 0}
    command = [sys.executable, '-m', 'cvat_agent.job_runner', payload['url'], payload.get('organization', ''), payload.get('model', 'sam3'), payload.get('mode', 'segmentation'), str(RUNS), str(payload.get('max_points', 80))]
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
        else:
            self.send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path == '/api/chat-filter':
            try:
                payload = read_json_body(self)
                selected = filter_labels(payload.get('message', ''), payload.get('descriptions', []))
                self.send_json({'selected': selected})
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
        if self.path != '/api/run':
            self.send_json({'error': 'Not found'}, 404)
            return
        payload = read_json_body(self)
        if not payload.get('url'):
            self.send_json({'error': 'Job URL is required'}, 400)
            return
        if payload.get('mode') not in {'segmentation', 'bbox'}:
            self.send_json({'error': 'Mode must be segmentation or bbox'}, 400)
            return
        try:
            points = int(payload.get('max_points', 80))
        except (TypeError, ValueError):
            self.send_json({'error': 'Số điểm không hợp lệ'}, 400)
            return
        if points < 0 or points > 1000:
            self.send_json({'error': 'Số điểm phải từ 0 đến 1000'}, 400)
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
    server = ThreadingHTTPServer(('127.0.0.1', 8787), Handler)
    threading.Timer(0.8, lambda: webbrowser.open('http://127.0.0.1:8787')).start()
    server.serve_forever()
