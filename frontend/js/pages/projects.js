import { api } from '../api.js';
import { poller } from '../state.js';
import { h, clear, icon, button, openDialog, toast, errMsg, timeAgo, fmtDuration, projectStatus, progressBar } from '../ui.js';
import { navigate } from '../main.js';

export async function mount(root) {
  const list = h('div', { class: 'proj-grid' });
  const head = h('div', { class: 'page-head' },
    h('div', null, h('h1', null, 'Projects'), h('p', { class: 'lede' }, 'Each project is one room: a video in, a 3D scene you can walk through and edit.')),
    button('New project', { kind: 'primary', icon: 'plus', onClick: newProject }));
  root.append(h('div', { class: 'page' }, head, list));

  let projects = null;
  const render = () => {
    clear(list);
    if (!projects) { list.append(h('div', { class: 'skeleton-row' }), h('div', { class: 'skeleton-row' })); return; }
    if (!projects.length) {
      list.append(h('div', { class: 'empty wide' }, icon('video', 30), h('h2', null, 'No rooms yet'),
        h('p', null, 'Create a project, then upload a 2–5 minute video of a single room filmed slowly in good light.'),
        button('Create your first project', { kind: 'primary', icon: 'plus', onClick: newProject })));
      return;
    }
    for (const p of projects) {
      const st = projectStatus(p);
      const busy = st.kind === 'busy';
      list.append(h('a', { class: 'proj-card', href: `#/p/${p.id}` },
        h('div', { class: 'proj-art', 'aria-hidden': 'true' }, icon(p.scene_ready ? 'cube' : 'video', 28)),
        h('div', { class: 'proj-info' },
          h('h3', null, p.name),
          h('p', { class: 'meta' }, [p.video_duration ? fmtDuration(p.video_duration) : null, timeAgo(p.created_at)].filter(Boolean).join(' • ')),
          h('span', { class: `chip ${st.kind}` }, busy ? h('i', { class: 'pulse' }) : null, st.label),
          busy ? progressBar(p.latest_task.progress) : null)));
    }
  };

  async function load() {
    try { projects = await api.projects(); render(); }
    catch (e) { clear(list); list.append(h('div', { class: 'empty wide' }, icon('warn', 28), h('h2', null, 'Could not load projects'), h('p', null, errMsg(e)), button('Retry', { onClick: load }))); }
    return false;
  }
  function newProject() {
    const input = h('input', { class: 'input', placeholder: 'Living room', maxLength: 200, 'aria-label': 'Project name' });
    const go = async ({ close }) => {
      const name = input.value.trim(); if (!name) { input.focus(); return; }
      try { const p = await api.createProject(name); close(); navigate(`/p/${p.id}`); } catch (e) { toast(errMsg(e), 'error'); }
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go({ close: () => document.querySelector('dialog')?.close() }); });
    openDialog({ title: 'New project', body: h('label', { class: 'field' }, h('span', null, 'Name'), input), actions: [{ label: 'Cancel', onClick: ({ close }) => close() }, { label: 'Create project', primary: true, onClick: go }] });
    setTimeout(() => input.focus(), 30);
  }
  render(); await load();
  const stop = poller(async () => { if (projects?.some((p) => projectStatus(p).kind === 'busy')) await load(); }, 4000);
  return { destroy: stop };
}
