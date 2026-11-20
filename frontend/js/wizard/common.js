import { api } from '../api.js';
import { h, icon, button, progressBar, toast, errMsg, isActiveTask } from '../ui.js';

/** Progress card for the edit's active task. */
export function taskPanel(w, { title, hint, cancelable = true }) {
  const t = w.edit.active_task;
  const pending = t?.status === 'pending';
  return h('div', { class: 'task-panel' },
    h('div', { class: 'task-top' }, h('h3', null, title), h('span', { class: 'big-num' }, `${t?.progress ?? 0}%`)),
    progressBar(t?.progress ?? 0, { indeterminate: !t, label: title }),
    h('p', { class: 'meta task-detail' }, pending ? (t.detail || 'Queued. Waiting for the GPU server.') : [t?.stage_label, t?.detail].filter(Boolean).join(' • ') || hint || 'Working…'),
    hint && t ? h('p', { class: 'hint' }, hint) : null,
    cancelable && t && isActiveTask(t) ? button('Cancel', { kind: 'ghost sm', onClick: async () => { try { await api.cancelTask(t.id); await w.refresh(); } catch (e) { toast(errMsg(e), 'error'); } } }) : null);
}

/** Shows edit.error (when the edit failed) with a retry for the current step. */
export function errorBlock(w, retry, label = 'Retry') {
  if (w.edit.state !== 'failed' && !w.edit.error) return null;
  return h('div', { class: 'callout err' }, icon('warn', 18), h('div', { class: 'stack tight' },
    h('strong', null, 'This step failed'), h('pre', { class: 'log' }, (w.edit.error || 'No details were returned.').slice(-900)),
    retry ? button(label, { icon: 'refresh', onClick: retry }) : null));
}

export const frameUrl = (w, name) => w.frameMap.get(name)?.url || null;
export const frameThumb = (w, name) => w.frameMap.get(name)?.thumb_url || null;
export const pct = (a) => (a == null ? '' : `${Math.round(a * 100)}%`);

/** Run an async action, surface errors as toasts, then refresh the edit. */
export async function act(w, fn) {
  try { await fn(); } catch (e) { toast(errMsg(e), 'error', 8000); }
  try { await w.refresh(); } catch { /* polling will recover */ }
}
