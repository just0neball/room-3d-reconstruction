import { api, url } from '../api.js';
import { h, clear, icon, button, segmented, toast, errMsg, spinner, stateChip } from '../ui.js';
import { navigate } from '../main.js';
import { SplatViewer } from '../viewer/viewer.js';
import { Wizard } from '../wizard/wizard.js';

export async function mount(root, params) {
  const pid = params.pid;
  let dead = false; let viewer; let wizard = null; let cams = null; let camIndex = 0; let edits = []; let project; let framesInfo = null;
  let current = 'orig';

  const stage = h('div', { class: 'viewer-stage' });
  const hudTop = h('div', { class: 'hud hud-top' });
  const hudBottom = h('div', { class: 'hud hud-bottom' });
  const loading = h('div', { class: 'viewer-loading' }, spinner(), h('span', null, 'Loading scene…'));
  const wizHost = h('div', { class: 'wiz-host hidden' });
  const pickBar = h('div', { class: 'pick-bar hidden', role: 'status' });
  const page = h('div', { class: 'viewer-page' }, stage, hudTop, hudBottom, loading, wizHost, pickBar);
  root.append(page);

  async function init() {
  try {
    const [scene, cameras, proj] = await Promise.all([api.scene(pid), api.cameras(pid).catch(() => null), api.project(pid).catch(() => null)]);
    project = proj; cams = cameras;
    viewer = new SplatViewer(stage);
    viewer.setUp(cameras?.up || [0, -1, 0]);
    const pts = (cameras?.cameras || []).map((c) => c.position);
    if (pts.length) {
      const min = [Infinity, Infinity, Infinity]; const max = [-Infinity, -Infinity, -Infinity];
      pts.forEach((p) => p.forEach((v, i) => { min[i] = Math.min(min[i], v); max[i] = Math.max(max[i], v); }));
      viewer.setSceneScale(Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 4);
    } else viewer.setSceneScale(4);
    await viewer.load('orig', url(scene.splat_url));
    if (dead) return;
    if (cams?.cameras?.length) viewer.setCameraPose(cams.cameras[0], { animate: false });
    else viewer.camera.position.set(0, -1, -4);
    window.room3dViewer = viewer; // handy for the console and for Week 2 experiments
    edits = await api.edits(pid).catch(() => []);
    loading.classList.add('hidden');
    buildHud(scene);
    update(params);
  } catch (e) {
    loading.classList.add('hidden');
    page.append(h('div', { class: 'empty viewer-error' }, icon('warn', 30), h('h2', null, 'The scene could not be opened'), h('p', null, errMsg(e)),
      h('div', { class: 'row' }, button('Back to project', { onClick: () => navigate(`/p/${pid}`) }), button('Try again', { kind: 'primary', onClick: () => location.reload() }))));
    console.error(e);
  }
  }

  // ---------- HUD ----------
  let modeSeg; let camLabel; let versionBox; let editBtn;
  function buildHud() {
    clear(hudTop); clear(hudBottom);
    editBtn = button('Edit object', { kind: 'primary', icon: 'target', onClick: newEdit });
    hudTop.append(
      h('div', { class: 'hud-left hud-card' }, h('a', { class: 'btn ghost icon', href: `#/p/${pid}`, 'aria-label': 'Back to project', title: 'Back to project' }, icon('back', 18)), h('strong', null, project?.name || 'Scene')),
      h('span', { class: 'spacer' }), versionBox = h('div', { class: 'hud-card versions' }), editBtn);
    modeSeg = segmented([{ value: 'orbit', label: 'Orbit', icon: 'orbit' }, { value: 'fly', label: 'Fly', icon: 'fly' }], viewer.mode, (v) => viewer.setMode(v), 'Camera mode');
    camLabel = h('span', { class: 'cam-label' });
    const nCam = cams?.cameras?.length || 0;
    const snap = (d) => { if (!nCam) return; camIndex = (camIndex + d + nCam) % nCam; viewer.setCameraPose(cams.cameras[camIndex]); updateCam(); };
    window.room3dSnapTo = (i) => { camIndex = i; viewer.setCameraPose(cams.cameras[i]); updateCam(); };
    const updateCam = () => { camLabel.textContent = nCam ? `Camera ${camIndex + 1} / ${nCam}` : 'No cameras'; };
    updateCam();
    hudBottom.append(
      h('div', { class: 'hud-card toolbar' }, modeSeg,
        h('span', { class: 'sep' }),
        h('button', { class: 'btn ghost icon', type: 'button', 'aria-label': 'Previous training camera', disabled: !nCam, onClick: () => snap(-1) }, icon('back', 16)),
        h('span', { class: 'cam-wrap' }, icon('camera', 15), camLabel),
        h('button', { class: 'btn ghost icon', type: 'button', 'aria-label': 'Next training camera', disabled: !nCam, onClick: () => snap(1) }, h('span', { class: 'flip' }, icon('back', 16)))),
      h('p', { class: 'hud-help' }, h('span', { class: 'kbd' }, 'Drag'), ' look ', h('span', { class: 'kbd' }, 'WASD'), ' move ', h('span', { class: 'kbd' }, 'Q'), h('span', { class: 'kbd' }, 'E'), ' down/up ', h('span', { class: 'kbd' }, 'Shift'), ' faster ', h('span', { class: 'kbd' }, 'Wheel'), ' speed'));
    viewer.addEventListener('mode', (e) => { modeSeg.querySelectorAll('button').forEach((b, i) => { const on = ['orbit', 'fly'][i] === e.detail.mode; b.classList.toggle('on', on); b.setAttribute('aria-checked', String(on)); }); });
    renderVersions();
  }

  // Versions: original + every finished edit. Switching keeps the camera.
  function versionList() {
    const done = edits.filter((e) => e.result?.splat_url).sort((a, b) => a.id - b.id);
    return [{ key: 'orig', label: 'Original', url: null }, ...done.map((e) => ({ key: `edit:${e.id}`, label: e.name || `Edit ${e.id}`, url: e.result.splat_url, edit: e }))];
  }
  function renderVersions() {
    if (!versionBox) return;
    clear(versionBox);
    const list = versionList();
    if (list.length === 1) { versionBox.classList.add('hidden'); return; }
    versionBox.classList.remove('hidden');
    const idx = list.findIndex((v) => v.key === current);
    const select = h('select', { class: 'select tight', 'aria-label': 'Scene version', onChange: (ev) => showVersion(ev.target.value) },
      list.map((v) => h('option', { value: v.key, selected: v.key === current }, v.label)));
    const original = current === 'orig';
    versionBox.append(icon('compare', 16),
      h('div', { class: 'ba-toggle', role: 'group', 'aria-label': 'Before and after' },
        h('button', { type: 'button', class: original ? 'on' : '', 'aria-pressed': String(original), onClick: () => showVersion('orig') }, 'Before'),
        h('button', { type: 'button', class: original ? '' : 'on', 'aria-pressed': String(!original), onClick: () => showVersion(original ? list[list.length - 1].key : current) }, 'After')),
      select,
      h('button', { class: 'btn ghost icon', type: 'button', title: 'Undo: go back to the previous version', 'aria-label': 'Previous version', disabled: idx <= 0, onClick: () => showVersion(list[idx - 1].key) }, icon('undo', 16)));
  }
  async function showVersion(key) {
    const v = versionList().find((x) => x.key === key); if (!v) return;
    try {
      if (v.url && !viewer.has(key)) { loading.classList.remove('hidden'); await viewer.load(key, url(v.url)); }
      viewer.show(key); current = key;
    } catch (e) { toast(`Could not load ${v.label}: ${errMsg(e)}`, 'error', 7000); }
    finally { loading.classList.add('hidden'); renderVersions(); }
  }

  async function newEdit() {
    editBtn.disabled = true;
    try { const e = await api.createEdit(pid, { name: `Edit ${edits.length + 1}` }); edits.unshift(e); navigate(`/p/${pid}/edit/${e.id}`); }
    catch (e) { toast(errMsg(e), 'error'); } finally { editBtn.disabled = false; }
  }

  // ---------- 3D pick hook for the wizard ----------
  function pick3d() {
    return new Promise((resolve) => {
      wizHost.classList.add('collapsed'); viewer.paused = false; viewer.setMode('orbit'); viewer.setPickMode(true);
      clear(pickBar); pickBar.classList.remove('hidden');
      const finish = (p) => { viewer.setPickMode(false); viewer.removeEventListener('pick', onPick); window.removeEventListener('keydown', onKey); pickBar.classList.add('hidden'); wizHost.classList.remove('collapsed'); viewer.paused = !!wizard; resolve(p); };
      const onPick = (ev) => { const { point } = ev.detail; if (!point) { toast('Nothing was hit there. Click directly on the object.', 'info'); return; } finish(point); };
      const onKey = (ev) => { if (ev.key === 'Escape') finish(null); };
      pickBar.append(icon('target', 18), h('span', null, 'Click the object you want to edit'), button('Cancel', { kind: 'sm', onClick: () => finish(null) }));
      viewer.addEventListener('pick', onPick); window.addEventListener('keydown', onKey);
    });
  }

  // ---------- wizard open/close, driven by the route ----------
  async function openWizard(editId) {
    if (wizard?.edit.id === editId) return;
    closeWizard(false);
    wizHost.classList.remove('hidden'); clear(wizHost); wizHost.append(h('div', { class: 'viewer-loading inline' }, spinner(), h('span', null, 'Opening editor…')));
    try {
      const [edit, fr] = await Promise.all([api.edit(editId), framesInfo ? Promise.resolve(framesInfo) : api.frames(pid)]);
      framesInfo = fr;
      if (dead) return;
      wizard = new Wizard({ pid, edit, framesInfo, hooks: { pick3d, close: () => navigate(wizard?.edit.state === 'done' ? `/p/${pid}/view/${editId}` : `/p/${pid}/view`), viewResult: (e) => viewResult(e) },
        onEditChange: (e) => { const i = edits.findIndex((x) => x.id === e.id); if (i >= 0) edits[i] = e; else edits.unshift(e); } });
      clear(wizHost); wizHost.append(wizard.el);
      editBtn.classList.add('hidden'); viewer.paused = true;
    } catch (e) {
      clear(wizHost); wizHost.append(h('div', { class: 'empty' }, icon('warn', 26), h('h2', null, 'Could not open this edit'), h('p', null, errMsg(e)), button('Close', { onClick: () => navigate(`/p/${pid}/view`) })));
    }
  }
  function closeWizard(hide = true) {
    wizard?.destroy(); wizard = null; if (viewer) viewer.paused = false;
    if (hide) { wizHost.classList.add('hidden'); clear(wizHost); }
    editBtn?.classList.remove('hidden');
  }
  async function viewResult(e) {
    edits = await api.edits(pid).catch(() => edits);
    navigate(`/p/${pid}/view/${e.id}`);
  }

  async function update(p) {
    if (dead || !viewer) return;
    if (p.wizard && p.editId) { await openWizard(p.editId); return; }
    const hadWizard = !!wizard; closeWizard(true);
    if (hadWizard) edits = await api.edits(pid).catch(() => edits);
    renderVersions();
    if (p.editId) { const key = `edit:${p.editId}`; if (versionList().some((v) => v.key === key)) await showVersion(key); }
  }

  init();
  return { update, destroy() { dead = true; wizard?.destroy(); viewer?.dispose(); delete window.room3dViewer; delete window.room3dSnapTo; } };
}
