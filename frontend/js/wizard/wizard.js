import { api } from '../api.js';
import { poller } from '../state.js';
import { h, clear, icon, stateChip, isActiveTask, toast, errMsg } from '../ui.js';
import { renderPick } from './pick.js';
import { renderDescribe } from './describe.js';
import { renderReview } from './review.js';
import { renderReference } from './reference.js';
import { renderApply } from './apply.js';
import { renderRebuild } from './rebuild.js';

const STEPS = [
  { n: 1, label: 'Pick object', render: renderPick },
  { n: 2, label: 'Describe', render: renderDescribe },
  { n: 3, label: 'Review frames', render: renderReview },
  { n: 4, label: 'First edit', render: renderReference },
  { n: 5, label: 'Edit all frames', render: renderApply },
  { n: 6, label: 'Rebuild', render: renderRebuild },
];
const AUTO = { draft: 1, masked: 2, tracking: 2, tracked: 3, editing_reference: 4, reference_ready: 4, editing_frames: 5, edited: 6, rebuilding: 6, done: 6 };
const ACTIVE_STATES = new Set(['tracking', 'editing_reference', 'editing_frames', 'rebuilding']);

export function autoStep(e) {
  if (e.state !== 'failed') return AUTO[e.state] || 1;
  if (!e.ref_frame) return 1;
  if (!e.tracking) return 2;
  if (!e.candidates?.length && !e.chosen_seed) return e.edit_progress?.total ? 5 : 4;
  if (e.edit_progress?.total && e.edit_progress.done < e.edit_progress.total) return 5;
  if (e.chosen_seed && !e.edit_progress?.total) return 5;
  return 6;
}

export class Wizard {
  /** hooks: { pick3d(): Promise<[x,y,z]|null>, close(), viewResult(edit) } */
  constructor({ pid, edit, framesInfo, hooks, onEditChange }) {
    this.pid = pid; this.edit = edit; this.framesInfo = framesInfo; this.hooks = hooks; this.onEditChange = onEditChange;
    this.frameMap = new Map((framesInfo?.frames || []).map((f) => [f.name, f]));
    this.view = null; this.stop = null; this.sig = '';
    this.el = h('section', { class: 'wiz', 'aria-label': 'Edit object' });
    this.head = h('header', { class: 'wiz-head' }); this.body = h('div', { class: 'wiz-body' });
    this.el.append(this.head, this.body);
    this.render(true); this.syncPolling();
  }
  get step() { return this.view ?? autoStep(this.edit); }
  get busy() { return ACTIVE_STATES.has(this.edit.state) || isActiveTask(this.edit.active_task); }
  goStep(n) { this.view = n; this.render(true); }

  setEdit(e) {
    const prev = this.edit; this.edit = e;
    if (prev.state !== e.state) this.view = null;
    this.onEditChange?.(e);
    const sig = [e.state, e.error, e.active_task?.progress, e.active_task?.detail, e.active_task?.status, e.tracking?.included_count, e.candidates?.length, e.chosen_seed, e.edit_progress?.done, e.result?.splat_url, e.name].join('|');
    const changed = sig !== this.sig; this.sig = sig;
    if (changed) this.render(prev.state !== e.state || !this._keep);
    this.syncPolling();
  }
  async refresh() { this.setEdit(await api.edit(this.edit.id)); }
  syncPolling() {
    if (this.busy && !this.stop) this.stop = poller(async () => { await this.refresh(); if (!this.busy) { this.stop = null; return true; } return false; }, 1500);
    if (!this.busy && this.stop) { this.stop(); this.stop = null; }
  }

  render(full = true) {
    const step = this.step;
    this._keep = false;
    // Header: title, stepper, close
    clear(this.head);
    const reach = this.view != null ? Math.max(autoStep(this.edit), this.view) : autoStep(this.edit);
    const nav = h('ol', { class: 'wiz-steps' }, STEPS.map((s) => {
      const clickable = !this.busy && s.n <= reach && s.n !== step;
      const done = s.n < reach || (s.n === 6 && this.edit.state === 'done');
      return h('li', { class: `${s.n === step ? 'active' : ''} ${done ? 'done' : ''}`, 'aria-current': s.n === step ? 'step' : null },
        h('button', { type: 'button', disabled: !clickable, onClick: () => this.goStep(s.n) }, h('span', { class: 'step-dot' }, done && s.n !== step ? icon('check', 13) : String(s.n)), h('span', { class: 'step-label' }, s.label)));
    }));
    this.head.append(h('div', { class: 'wiz-title' }, h('h2', null, this.edit.name || 'Edit object'), stateChip(this.edit.state)), nav,
      h('button', { class: 'btn ghost icon', type: 'button', 'aria-label': 'Close editor', onClick: () => this.hooks.close() }, icon('x', 18)));
    clear(this.body);
    try { this.body.append(STEPS[step - 1].render(this)); }
    catch (e) { console.error(e); this.body.append(h('div', { class: 'callout err' }, icon('warn', 18), h('div', null, h('strong', null, 'This step could not be displayed'), h('p', null, errMsg(e))))); }
  }
  destroy() { this.stop?.(); this.stop = null; }
}
