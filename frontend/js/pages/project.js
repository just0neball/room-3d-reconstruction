import { api, url } from '../api.js';
import { store, poller } from '../state.js';
import { h, clear, icon, button, toast, errMsg, fmtBytes, fmtDuration, timeAgo, progressBar, stateChip, openDialog, segmented, isActiveTask } from '../ui.js';
import { navigate } from '../main.js';

const STAGES = [
  ['extract', 'Extract frames', 'Sampling the video and dropping blurry frames'],
  ['poses', 'Estimate cameras', 'Solving where the camera was for every frame'],
  ['train', 'Train splat', 'Fitting the 3D Gaussians to your frames'],
  ['export', 'Export scene', 'Packaging the scene for the viewer'],
];

export async function mount(root, { pid }) {
  let project = null; let task = null; let edits = []; let sceneInfo = null;
  let file = null; let uploading = null; let stopTask = null; let dead = false;
  const opts = { frame_fps: 1, steps: 7000 };

  const body = h('div', { class: 'page' });
  root.append(body);

  async function loadProject() {
    project = await api.project(pid);
    const t = project.latest_task;
    task = t && t.type === 'reconstruction' ? t : task && task.project_id === pid ? task : null;
    if (task && isActiveTask(task)) watchTask(task.id);
    if (project.scene_ready) {
      try { edits = await api.edits(pid); } catch { edits = []; }
      if (!sceneInfo) api.scene(pid).then((s) => { sceneInfo = s; render(); }).catch(() => {});
    }
  }

  function watchTask(id) {
    if (stopTask) return;
    stopTask = poller(async () => {
      const t = await api.task(id);
      task = t;
      if (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled') {
        stopTask = null; sceneInfo = null; await loadProject(); render(); return true;
      }
      render(); return false;
    }, 1500, { immediate: true });
  }

  async function start() {
    try {
      if (file) {
        uploading = 0; render();
        const r = await api.uploadVideo(pid, file, (f) => { uploading = f; updateUpload(); });
        (r.warnings || []).forEach((w) => toast(w, 'info', 8000));
        file = null; uploading = null;
      }
      task = await api.reconstruct(pid, { steps: opts.steps, frame_fps: opts.frame_fps });
      await loadProject(); watchTask(task.id); render();
    } catch (e) { uploading = null; toast(errMsg(e), 'error', 8000); await loadProject().catch(() => {}); render(); }
  }

  let uploadBar;
  function updateUpload() { if (uploadBar) { uploadBar.firstChild.style.width = `${Math.round((uploading || 0) * 100)}%`; uploadBar.setAttribute('aria-valuenow', Math.round((uploading || 0) * 100)); } }

  function dropzone() {
    const input = h('input', { type: 'file', accept: 'video/*,.mkv', class: 'visually-hidden', id: 'vid', onChange: () => pick(input.files[0]) });
    const pick = (f) => { if (!f) return; file = f; render(); };
    const zone = h('label', { class: 'dropzone', for: 'vid', tabindex: 0,
      onDragover: (e) => { e.preventDefault(); zone.classList.add('over'); }, onDragleave: () => zone.classList.remove('over'),
      onDrop: (e) => { e.preventDefault(); zone.classList.remove('over'); pick(e.dataTransfer.files[0]); },
      onKeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } } },
    icon('upload', 30),
    file ? h('div', null, h('strong', null, file.name), h('p', { class: 'meta' }, fmtBytes(file.size), ' • click to choose another'))
      : h('div', null, h('strong', null, 'Drop a room video here'), h('p', { class: 'meta' }, 'or click to browse. MP4, MOV, WebM or MKV, 2–5 minutes, slow camera motion.')),
    input);
    return zone;
  }

  function options() {
    const steps = h('input', { class: 'input num', type: 'number', min: 1000, max: 30000, step: 1000, value: opts.steps, 'aria-label': 'Training steps', onInput: (e) => { opts.steps = Math.max(1000, Math.min(30000, +e.target.value || 7000)); } });
    return h('div', { class: 'opts' },
      h('div', { class: 'field' }, h('span', null, 'Frame rate'), segmented([{ value: 1, label: '1 fps', title: 'Fewer frames, faster' }, { value: 2, label: '2 fps', title: 'More frames, slower pose solving' }], opts.frame_fps, (v) => { opts.frame_fps = v; }, 'Frame rate')),
      h('label', { class: 'field' }, h('span', null, 'Training steps'), steps));
  }

  function stepper(t) {
    const failed = t?.status === 'failed';
    const group = t?.stage_group;
    const idx = STAGES.findIndex((s) => s[0] === group);
    const wrap = h('ol', { class: 'stepper' });
    STAGES.forEach(([key, label, desc], i) => {
      const done = t?.status === 'completed' || (idx > -1 && i < idx);
      const active = !done && i === idx && !failed;
      const bad = failed && i === idx;
      wrap.append(h('li', { class: `step ${done ? 'done' : ''} ${active ? 'active' : ''} ${bad ? 'bad' : ''}`, 'aria-current': active ? 'step' : null },
        h('span', { class: 'step-dot' }, done ? icon('check', 14) : bad ? icon('x', 14) : String(i + 1)),
        h('div', null, h('strong', null, label), h('p', { class: 'meta' }, active || bad ? desc : ''))));
    });
    return wrap;
  }

  function reconstructionCard() {
    const active = isActiveTask(task);
    const failed = task?.status === 'failed';
    const hasVideo = project.has_video;
    if (active) {
      return h('section', { class: 'card stack' },
        h('div', { class: 'card-head' }, h('h2', null, 'Reconstructing your room'), h('span', { class: 'big-num' }, `${task.progress}%`)),
        progressBar(task.progress, { label: 'Reconstruction progress' }),
        h('p', { class: 'meta' }, task.status === 'pending' ? (task.detail || 'Queued. Waiting for the GPU server.') : `${task.stage_label || 'Working'}${task.detail ? ` • ${task.detail}` : ''}`),
        stepper(task),
        h('div', { class: 'row' }, button('Cancel', { kind: 'ghost sm', onClick: async () => { try { await api.cancelTask(task.id); } catch (e) { toast(errMsg(e), 'error'); } } })));
    }
    if (uploading != null) {
      uploadBar = progressBar((uploading || 0) * 100, { label: 'Upload progress' });
      return h('section', { class: 'card stack' }, h('h2', null, 'Uploading video'), uploadBar, h('p', { class: 'meta' }, file ? `${file.name} • ${fmtBytes(file.size)}` : ''));
    }
    const rows = [];
    if (failed) rows.push(h('div', { class: 'callout err' }, icon('warn', 18), h('div', null, h('strong', null, `Failed during: ${task.stage_label || task.stage || 'reconstruction'}`), h('pre', { class: 'log' }, (task.error || 'No error message.').slice(-1200)))));
    if (project.scene_ready) {
      rows.push(h('div', { class: 'card-head' }, h('h2', null, 'Reconstruct again'), h('span', { class: 'meta' }, 'Replaces the current scene and all edits')));
    } else {
      rows.push(h('div', { class: 'card-head' }, h('h2', null, hasVideo ? 'Ready to reconstruct' : 'Upload a video')));
    }
    if (hasVideo && !file) rows.push(h('p', { class: 'meta' }, `${project.video_name || 'video'} • ${fmtBytes(project.video_size)}${project.video_duration ? ` • ${fmtDuration(project.video_duration)}` : ''}`));
    rows.push(dropzone(), options());
    rows.push(h('div', { class: 'row' }, button(file ? 'Upload and reconstruct' : hasVideo ? 'Start reconstruction' : 'Choose a video first', { kind: 'primary', disabled: !file && !hasVideo, onClick: start })));
    return h('section', { class: 'card stack' }, rows);
  }

  function sceneCard() {
    const meta = sceneInfo?.metadata || {};
    const gs = meta.scene?.gaussian_count ?? meta.training?.gaussians;
    return h('section', { class: 'card scene-card' },
      h('div', { class: 'scene-art', 'aria-hidden': 'true' }, icon('cube', 44)),
      h('div', { class: 'stack' },
        h('h2', null, 'Your 3D room is ready'),
        h('p', { class: 'meta' }, [gs ? `${Number(gs).toLocaleString()} Gaussians` : null, project.video_duration ? `from ${fmtDuration(project.video_duration)} of video` : null].filter(Boolean).join(' • ') || 'Walk through it with orbit or free-fly controls.'),
        h('div', { class: 'row' }, button('Open viewer', { kind: 'primary', icon: 'cube', onClick: () => navigate(`/p/${pid}/view`) }), button('Edit an object', { icon: 'target', onClick: newEdit }))));
  }

  async function newEdit() {
    try { const e = await api.createEdit(pid, { name: `Edit ${edits.length + 1}` }); navigate(`/p/${pid}/edit/${e.id}`); }
    catch (e) { toast(errMsg(e), 'error'); }
  }

  function editsSection() {
    const cards = edits.map((e) => {
      const thumb = e.ref_overlay_url || e.reference_url || e.candidates?.[0]?.thumb_url;
      const dest = e.state === 'done' ? `/p/${pid}/view/${e.id}` : `/p/${pid}/edit/${e.id}`;
      return h('article', { class: 'edit-card' },
        h('a', { class: 'edit-thumb', href: `#${dest}`, 'aria-label': `Open ${e.name}` }, thumb ? h('img', { src: url(thumb), alt: '', loading: 'lazy' }) : icon('target', 26)),
        h('div', { class: 'edit-body' },
          h('h3', null, e.name || `Edit ${e.id}`),
          h('p', { class: 'prompt' }, e.prompt ? `“${e.prompt}”` : 'No prompt yet'),
          h('div', { class: 'row between' }, stateChip(e.state), h('span', { class: 'meta' }, timeAgo(e.updated_at || e.created_at))),
          h('div', { class: 'row' },
            h('a', { class: 'btn sm', href: `#${dest}` }, e.state === 'done' ? 'View result' : 'Continue'),
            h('button', { class: 'btn ghost sm', type: 'button', 'aria-label': `Delete ${e.name}`, onClick: () => confirmDelete(e) }, icon('trash', 15)))));
    });
    return h('section', { class: 'stack' },
      h('div', { class: 'section-head' }, h('h2', null, 'Edit history'), button('New edit', { icon: 'plus', onClick: newEdit })),
      edits.length ? h('div', { class: 'edit-grid' }, cards)
        : h('div', { class: 'empty small' }, icon('target', 24), h('p', null, 'No edits yet. Pick an object in a frame, describe the change, and the whole room updates.')));
  }
  function confirmDelete(e) {
    openDialog({ title: 'Delete this edit?', body: h('p', null, `“${e.name}” and its edited frames will be removed. The original scene is untouched.`),
      actions: [{ label: 'Keep it', onClick: ({ close }) => close() }, { label: 'Delete edit', kind: 'danger', onClick: async ({ close }) => { try { await api.deleteEdit(e.id); close(); await loadProject(); render(); } catch (x) { toast(errMsg(x), 'error'); } } }] });
  }

  function render() {
    if (dead || !project) return;
    clear(body);
    body.append(
      h('a', { class: 'back', href: '#/' }, icon('back', 16), 'Projects'),
      h('div', { class: 'page-head' }, h('div', null, h('h1', null, project.name), h('p', { class: 'meta' }, `Created ${timeAgo(project.created_at)}`))),
      project.scene_ready && !isActiveTask(task) ? sceneCard() : null,
      reconstructionCard(),
      project.scene_ready ? editsSection() : null);
  }

  try { await loadProject(); render(); }
  catch (e) { body.append(h('div', { class: 'empty' }, icon('warn', 28), h('h2', null, 'Project not found'), h('p', null, errMsg(e)), button('Back to projects', { onClick: () => navigate('/') }))); }
  return { destroy() { dead = true; stopTask?.(); } };
}
