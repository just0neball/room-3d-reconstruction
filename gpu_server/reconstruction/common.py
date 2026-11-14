import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def project_root(value):
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def send_progress(args, stage, progress, status='running', error=None):
    event = {
        'stage': stage,
        'progress': int(progress),
        'status': status,
        'error': error,
        'timestamp_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    }
    root = project_root(args.project_dir)
    with (root / 'progress_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    base = getattr(args, 'backend_url', None)
    task_id = getattr(args, 'task_id', None)
    if base and task_id:
        url = f"{base.rstrip('/')}/tasks/{task_id}/progress"
        headers = {'Content-Type': 'application/json'}
        token = os.environ.get('PERSON_A_BACKEND_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
        request = urllib.request.Request(url, data=json.dumps(event).encode('utf-8'), headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status >= 400:
                    raise RuntimeError(f'HTTP {response.status}')
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f'Progress notification failed; retained locally: {exc}', flush=True)
    print(f'[{stage}] {progress}% {status}', flush=True)


def add_service_args(parser):
    parser.add_argument('--project_dir', required=True)
    parser.add_argument('--backend_url', default=None)
    parser.add_argument('--task_id', default=None)
    return parser


def elapsed(start):
    return round(time.perf_counter() - start, 3)
