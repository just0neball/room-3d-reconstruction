// Thin client for docs/API.md. Same-origin by default; localStorage "room3d.apiBase" overrides for dev.
const K_TOKEN = 'room3d.token';
const K_BASE = 'room3d.apiBase';

const ls = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* private mode */ } },
};

export const getBase = () => (ls.get(K_BASE) || '').replace(/\/$/, '');
export const getToken = () => ls.get(K_TOKEN);
export const setToken = (t) => ls.set(K_TOKEN, t);
export const setBase = (b) => ls.set(K_BASE, b || null);

/** Turn a relative API url (already carrying ?t=) into something a browser can fetch. */
export const url = (rel) => (!rel ? null : /^(https?:|data:|blob:)/.test(rel) ? rel : getBase() + rel);

export class ApiError extends Error {
  constructor(status, detail) { super(detail || `HTTP ${status}`); this.status = status; this.detail = detail; }
}

function authHeaders(extra = {}) {
  const t = getToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

async function parse(res) {
  let data = null;
  const text = await res.text();
  if (text) { try { data = JSON.parse(text); } catch { data = text; } }
  if (!res.ok) {
    let detail = data && typeof data === 'object' ? data.detail : data;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    if (res.status === 401) window.dispatchEvent(new CustomEvent('room3d:unauthorized'));
    throw new ApiError(res.status, typeof detail === 'string' && detail ? detail : `Request failed (${res.status})`);
  }
  return data;
}

async function req(method, path, body, opts = {}) {
  const init = { method, headers: authHeaders(), signal: opts.signal };
  if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
  let res;
  try { res = await fetch(getBase() + path, init); }
  catch (e) { if (e.name === 'AbortError') throw e; throw new ApiError(0, 'Cannot reach the Room3D server. Is the backend running?'); }
  return parse(res);
}

const get = (p, o) => req('GET', p, undefined, o);
const post = (p, b) => req('POST', p, b ?? {});
const put = (p, b) => req('PUT', p, b);
const patch = (p, b) => req('PATCH', p, b);
const del = (p) => req('DELETE', p);

/** Multipart upload with progress (fetch has no upload progress events). */
function upload(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', getBase() + path);
    const t = getToken();
    if (t) xhr.setRequestHeader('Authorization', `Bearer ${t}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError(0, 'Upload failed: the connection dropped.'));
    xhr.onload = () => {
      let data = null; try { data = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
      let d = data?.detail; if (Array.isArray(d)) d = d.map((x) => x.msg).join('; ');
      reject(new ApiError(xhr.status, d || `Upload failed (${xhr.status})`));
    };
    const fd = new FormData(); fd.append('file', file);
    xhr.send(fd);
  });
}

export const api = {
  health: () => get('/health'),
  login: (email, password) => post('/auth/login', { email, password }),
  signup: (email, password) => post('/auth/signup', { email, password }),
  gpuStatus: (o) => get('/gpu/status', o),
  gpuConfig: (u, tok) => put('/gpu/config', tok ? { url: u, token: tok } : { url: u }),

  projects: () => get('/projects'),
  project: (id) => get(`/projects/${id}`),
  createProject: (name) => post('/projects', { name }),
  uploadVideo: (id, file, onProgress) => upload(`/projects/${id}/video`, file, onProgress),
  reconstruct: (id, body) => post(`/projects/${id}/reconstruct`, body),
  scene: (id) => get(`/projects/${id}/scene`),
  frames: (id) => get(`/projects/${id}/frames`),
  cameras: (id) => get(`/projects/${id}/cameras`),

  task: (id) => get(`/tasks/${id}`),
  cancelTask: (id) => post(`/tasks/${id}/cancel`),

  edits: (pid) => get(`/projects/${pid}/edits`),
  createEdit: (pid, body) => post(`/projects/${pid}/edits`, body),
  edit: (id) => get(`/edits/${id}`),
  patchEdit: (id, body) => patch(`/edits/${id}`, body),
  deleteEdit: (id) => del(`/edits/${id}`),
  pick3d: (id, point) => post(`/edits/${id}/pick3d`, { point }),
  segment: (id, body) => post(`/edits/${id}/segment`, body),
  track: (id, body) => post(`/edits/${id}/track`, body),
  patchFrames: (id, body) => patch(`/edits/${id}/frames`, body),
  editReference: (id, body) => post(`/edits/${id}/edit/reference`, body),
  acceptSeed: (id, seed) => post(`/edits/${id}/edit/accept`, { seed }),
  editRun: (id) => post(`/edits/${id}/edit/run`, {}),
  rebuild: (id, body) => post(`/edits/${id}/rebuild`, body),
};
