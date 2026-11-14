// GPU server status pill, banner and "Connect" dialog.
import { api } from '../api.js';
import { store, emit, poller } from '../state.js';
import { h, icon, openDialog, toast, errMsg, clear } from '../ui.js';

export function gpuHeaderPill() {
  const pill = h('button', { class: 'pill', type: 'button', title: 'GPU server connection', onClick: () => openConnectDialog() });
  const render = () => {
    const g = store.gpu;
    clear(pill);
    let kind = 'warn'; let label = 'Checking GPU…';
    if (g) {
      if (!g.configured) { kind = 'err'; label = 'Connect GPU server'; }
      else if (!g.reachable) { kind = 'err'; label = 'GPU offline'; }
      else { kind = 'ok'; label = g.mock ? 'GPU demo mode' : (g.devices?.[0]?.name ? `GPU · ${g.devices.map((d) => d.name).join(' + ')}` : 'GPU online'); }
    }
    pill.className = `pill ${kind}`;
    pill.append(h('i', { class: 'dot' }), icon('gpu', 15), h('span', null, label));
  };
  render();
  const un = window.addEventListener('room3d:gpu', render);
  return { el: pill, render };
}

export function gpuBanner() {
  const el = h('div', { class: 'banner hidden', role: 'status' });
  const render = () => {
    const g = store.gpu;
    clear(el);
    if (!g) { el.classList.add('hidden'); return; }
    if (!g.configured || !g.reachable) {
      el.className = 'banner warn';
      el.append(icon('warn', 16), h('span', null, !g.configured
        ? 'No GPU server connected. Start the Kaggle notebook and paste its public URL to reconstruct or edit.'
        : `GPU server unreachable${g.message ? ` (${g.message})` : ''}. Jobs wait in the queue and start when it comes back.`),
      h('button', { class: 'btn sm', type: 'button', onClick: () => openConnectDialog() }, 'Connect'));
    } else if (g.mock) {
      el.className = 'banner demo';
      el.append(icon('info', 16), h('span', null, 'DEMO MODE — synthetic models. The scene, masks and edits are generated stand-ins, not real SAM or diffusion output.'));
    } else el.className = 'banner hidden';
  };
  window.addEventListener('room3d:gpu', render);
  render();
  return el;
}

export async function refreshGpu() {
  try { store.gpu = await api.gpuStatus(); }
  catch (e) { store.gpu = { configured: store.gpu?.configured ?? false, reachable: false, message: e.status === 0 ? 'backend not reachable' : errMsg(e) }; }
  window.dispatchEvent(new CustomEvent('room3d:gpu'));
  emit('gpu', store.gpu);
}
export const startGpuPolling = () => { refreshGpu(); return poller(async () => { await refreshGpu(); }, 10000); };

export function openConnectDialog() {
  const urlInput = h('input', { class: 'input', type: 'url', placeholder: 'https://something-random.trycloudflare.com', value: store.gpu?.url || '', autocomplete: 'off', spellcheck: false, 'aria-label': 'GPU server URL' });
  const tokenInput = h('input', { class: 'input', type: 'password', placeholder: 'Leave empty to keep the current token', autocomplete: 'off', 'aria-label': 'GPU server token' });
  const status = h('p', { class: 'hint', role: 'status' });
  const g = store.gpu;
  if (g?.configured) status.textContent = g.reachable ? 'Connected.' : `Not reachable${g.message ? `: ${g.message}` : ''}`;
  const body = h('div', { class: 'stack' },
    h('p', null, 'The GPU server is the Kaggle notebook running ', h('b', null, 'notebooks/gpu_server.ipynb'), '. When it starts it prints a public ', h('b', null, 'trycloudflare.com'), ' URL. Paste it here.'),
    h('p', { class: 'hint' }, 'A Kaggle session gets a new URL every time it restarts, so you will need to paste it again after a restart.'),
    h('label', { class: 'field' }, h('span', null, 'Server URL'), urlInput),
    h('label', { class: 'field' }, h('span', null, 'Access token (optional)'), tokenInput),
    status);
  const dlg = openDialog({
    title: 'Connect GPU server', body,
    actions: [{ label: 'Cancel', onClick: ({ close }) => close() }, {
      label: 'Save and test', primary: true,
      onClick: async ({ close }) => {
        status.textContent = 'Testing connection…';
        try {
          store.gpu = await api.gpuConfig(urlInput.value.trim(), tokenInput.value.trim() || undefined);
          window.dispatchEvent(new CustomEvent('room3d:gpu'));
          if (store.gpu.reachable) { toast('GPU server connected', 'ok'); close(); }
          else status.textContent = `Saved, but the server did not answer${store.gpu.message ? `: ${store.gpu.message}` : ''}. Check that the notebook is still running.`;
        } catch (e) { status.textContent = errMsg(e); }
      },
    }],
  });
  setTimeout(() => urlInput.focus(), 30);
  return dlg;
}
