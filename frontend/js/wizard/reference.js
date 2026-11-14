import { api, url } from '../api.js';
import { h, icon, button, toast, errMsg } from '../ui.js';
import { taskPanel, errorBlock, act, frameUrl } from './common.js';

export function renderReference(w) {
  const e = w.edit;
  const R = (w.ref ||= { seeds: e.settings?.seeds || 3, frame: 'largest' });
  if (w.busy && e.state === 'editing_reference') {
    return h('div', { class: 'step-pane narrow-pane' }, taskPanel(w, { title: 'Generating first-frame options', hint: 'The diffusion model loads once, then each seed is one image.' }));
  }
  const frames = e.tracking?.frames || [];
  const best = frames.filter((f) => f.included && f.mask_url && f.area != null).sort((a, b) => b.area - a.area)[0];
  const frameName = () => (R.frame === 'largest' && best ? best.name : e.ref_frame);
  const generate = (label, kind = 'primary') => button(label, { kind, icon: 'edit', onClick: () => act(w, async () => {
    const name = frameName(); R.used = name; w.refFrameUsed = name;
    await api.editReference(e.id, { prompt: e.prompt || undefined, seeds: R.seeds, frame: name });
  }) });

  const hasCands = (e.candidates || []).length > 0;
  const root = h('div', { class: 'step-pane stack' });
  root.append(h('div', null, h('h3', null, 'Edit the first frame'), h('p', { class: 'meta' }, `“${e.prompt || 'No prompt'}” — choose the result you like best. Every other frame is edited to match it.`)));
  root.append(errorBlock(w, null));

  const seeds = h('input', { type: 'range', min: 1, max: 6, step: 1, value: R.seeds, 'aria-label': 'Number of options', onInput: (ev) => { R.seeds = +ev.target.value; out.textContent = R.seeds; } });
  const out = h('output', null, String(R.seeds));
  const frameSel = h('select', { class: 'select', 'aria-label': 'Frame to edit first', onChange: (ev) => { R.frame = ev.target.value; } },
    h('option', { value: 'largest', selected: R.frame === 'largest' }, best ? `Frame with the largest object (${((best.seconds ?? 0)).toFixed(1)}s)` : 'Reference frame'),
    e.ref_frame ? h('option', { value: 'ref', selected: R.frame === 'ref' }, 'The frame I clicked on') : null);
  root.append(h('div', { class: 'row wrap controls-row' },
    h('label', { class: 'field inline' }, h('span', null, 'Options to generate'), h('span', { class: 'range' }, seeds, out)),
    h('label', { class: 'field inline' }, h('span', null, 'Start from'), frameSel),
    hasCands ? generate('Regenerate with new seeds', '') : generate('Generate options')));

  if (hasCands) {
    const orig = frameUrl(w, w.refFrameUsed || R.used || e.ref_frame);
    const grid = h('div', { class: 'cand-grid' },
      orig ? h('figure', { class: 'cand original' }, h('img', { src: url(orig), alt: 'Original frame' }), h('figcaption', null, 'Original')) : null,
      e.candidates.map((c) => h('figure', { class: `cand ${e.chosen_seed === c.seed ? 'chosen' : ''}` },
        h('button', { type: 'button', class: 'cand-btn', 'aria-pressed': e.chosen_seed === c.seed ? 'true' : 'false', onClick: () => act(w, () => api.acceptSeed(e.id, c.seed)) }, h('img', { src: url(c.url || c.thumb_url), alt: `Option ${c.seed}` })),
        h('figcaption', null, e.chosen_seed === c.seed ? h('span', { class: 'chosen-tag' }, icon('check', 14), `Option ${c.seed} chosen`) : `Option ${c.seed}`))));
    root.append(grid);
    const total = frames.filter((f) => f.included).length;
    root.append(h('div', { class: 'row' }, button(`Apply to all ${total || ''} frames`.replace('  ', ' '), { kind: 'primary', icon: 'arrow', disabled: e.chosen_seed == null,
      title: e.chosen_seed == null ? 'Choose an option first' : '', onClick: () => act(w, () => api.editRun(e.id)) }),
    e.chosen_seed == null ? h('span', { class: 'hint' }, 'Click an option to choose it.') : null));
  }
  return root;
}
