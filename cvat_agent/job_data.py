import base64
import json
import sys
import zipfile
from pathlib import Path

try:
    from .cdp import evaluate
except ImportError:
    from cdp import evaluate


def fetch_chunk(job_id, chunk_number=0):
    expression = f"fetch('/api/jobs/{job_id}/data?type=chunk&number={chunk_number}').then(async r=>{{if(!r.ok) throw new Error('HTTP '+r.status); const b=await r.arrayBuffer(); let s=''; const a=new Uint8Array(b); for(let i=0;i<a.length;i+=0x8000) s+=String.fromCharCode(...a.subarray(i,i+0x8000)); return btoa(s)}})"
    return base64.b64decode(evaluate(expression))


def save_chunk(job_id, output_dir, chunk_number=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f'job_{job_id}_chunk_{chunk_number}.zip'
    archive.write_bytes(fetch_chunk(job_id, chunk_number))
    with zipfile.ZipFile(archive) as source:
        source.extractall(output_dir / f'job_{job_id}_chunk_{chunk_number}')
    return archive


if __name__ == '__main__':
    job_id = int(sys.argv[1])
    output_dir = Path(sys.argv[2])
    chunk_number = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print(save_chunk(job_id, output_dir, chunk_number))
