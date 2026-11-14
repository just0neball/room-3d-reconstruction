import { api, getToken, setToken } from './api.js';
import { store, on } from './state.js';
import { h, clear, icon, button, toast, errMsg } from './ui.js';
import { gpuHeaderPill, gpuBanner, startGpuPolling } from './components/gpu.js';

const PAGES = {
  projects: () => import('./pages/projects.js'),
  project: () => import('./pages/project.js'),
  viewer: () => import('./pages/viewer.js'),
};
const ROUTES = [
  [/^\/?$/, 'projects', () => ({})],
  [/^\/p\/(\d+)\/?$/, 'project', (m) => ({ pid: +m[1] })],
  [/^\/p\/(\d+)\/view(?:\/(\d+))?\/?$/, 'viewer', (m) => ({ pid: +m[1], editId: m[2] ? +m[2] : null, wizard: false })],
  [/^\/p\/(\d+)\/edit\/(\d+)\/?$/, 'viewer', (m) => ({ pid: +m[1], editId: +m[2], wizard: true })],
];

let current = null; // { name, key, instance }
let main;
let navToken = 0;

export function navigate(path) { location.hash = '#' + path; }

function match() {
  const path = (location.hash.replace(/^#/, '') || '/').split('?')[0];
  for (const [re, name, params] of ROUTES) { const m = path.match(re); if (m) return { name, params: params(m) }; }
  return { name: 'projects', params: {} };
}

async function route() {
  const token = ++navToken;
  if (store.auth.required && !getToken()) return showLogin();
  const { name, params } = match();
  const key = name === 'viewer' ? `viewer:${params.pid}` : name === 'project' ? `project:${params.pid}` : name;
  // Same project viewer (wizard open/close, switching edits): update in place, keep the WebGL context.
  if (current && current.key === key && current.instance.update) { current.instance.update(params); return; }
  current?.instance.destroy?.();
  clear(main);
  current = null;
  main.append(h('div', { class: 'page-loading' }, h('span', { class: 'spinner' })));
  try {
    const mod = await PAGES[name]();
    if (token !== navToken) return;
    clear(main);
    const root = h('div', { class: 'page-root' });
    main.append(root);
    const instance = await mod.mount(root, params);
    if (token !== navToken) { instance?.destroy?.(); return; }
    current = { name, key, instance: instance || {} };
    document.title = 'Room3D';
  } catch (e) {
    if (token !== navToken) return;
    clear(main);
    main.append(h('div', { class: 'page narrow' }, h('div', { class: 'empty' }, icon('warn', 28), h('h2', null, 'This page did not load'), h('p', null, errMsg(e)),
      button('Back to projects', { onClick: () => navigate('/') }))));
    console.error(e);
  }
}

function showLogin() {
  current?.instance.destroy?.(); current = null;
  import('./pages/login.js').then((m) => { clear(main); m.mount(main, { onDone: () => { buildChrome(); route(); } }); });
}

let chromeBuilt = false;
let stopGpu;
function buildChrome() {
  const app = document.getElementById('app');
  clear(app);
  const pill = gpuHeaderPill();
  const header = h('header', { class: 'topbar' },
    h('a', { class: 'brand', href: '#/', 'aria-label': 'Room3D home' }, brandMark(), h('span', null, 'Room3D')),
    h('nav', { class: 'topnav', 'aria-label': 'Main' }, h('a', { href: '#/' }, 'Projects')),
    h('span', { class: 'spacer' }),
    pill.el,
    store.auth.required && getToken() ? button('Sign out', { kind: 'ghost sm', onClick: () => { setToken(null); location.hash = '#/'; buildChrome(); route(); } }) : null);
  main = h('main', { class: 'main', id: 'main' });
  app.append(header, gpuBanner(), main);
  if (!chromeBuilt) { stopGpu = startGpuPolling(); chromeBuilt = true; }
  pill.render();
}

function brandMark() {
  const s = h('span', { class: 'brand-mark', 'aria-hidden': 'true' });
  s.innerHTML = '<svg viewBox="0 0 32 32" width="26" height="26"><path d="M16 5l10 5.5v11L16 27 6 21.5v-11z" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/><path d="M6 10.5l10 5.5 10-5.5M16 16v11" fill="none" stroke="currentColor" stroke-width="1.4" opacity=".55"/><circle cx="16" cy="16" r="2.6" fill="var(--mask)"/></svg>';
  return s;
}

async function boot() {
  const app = document.getElementById('app');
  app.append(h('div', { class: 'page-loading' }, h('span', { class: 'spinner' })));
  try { const hl = await api.health(); store.auth.required = !!hl.auth_required; }
  catch (e) {
    clear(app);
    app.append(h('div', { class: 'page narrow' }, h('div', { class: 'empty' }, icon('warn', 28), h('h2', null, 'Cannot reach the Room3D server'),
      h('p', null, 'Start the backend (uvicorn app.main:app) and reload this page.'), button('Try again', { kind: 'primary', onClick: () => location.reload() }))));
    return;
  }
  on('toast', (m) => toast(m));
  window.addEventListener('room3d:unauthorized', () => { if (store.auth.required) { setToken(null); route(); } });
  window.addEventListener('hashchange', route);
  buildChrome();
  route();
}
boot();
