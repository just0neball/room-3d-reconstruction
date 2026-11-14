import * as THREE from 'three';

/** WASD + mouse-look controls that respect an arbitrary world "up" (COLMAP scenes are Y-down).
 *  Drag with the left or right button to look, W/A/S/D move, Q/E down/up, Shift = fast, wheel = speed. */
export class FlyControls {
  constructor(camera, dom, { baseSpeed = 1 } = {}) {
    this.camera = camera; this.dom = dom; this.enabled = false;
    this.baseSpeed = baseSpeed; this.speedMul = 1; this.keys = new Set();
    this.dragging = false; this.last = { x: 0, y: 0 };
    this._f = new THREE.Vector3(); this._r = new THREE.Vector3(); this._q = new THREE.Quaternion(); this._v = new THREE.Vector3();
    this.onChange = () => {};
    this._down = (e) => { if (!this.enabled) return; this.dragging = true; this.last = { x: e.clientX, y: e.clientY }; dom.setPointerCapture?.(e.pointerId); dom.focus?.(); };
    this._move = (e) => {
      if (!this.enabled || !this.dragging) return;
      this.look(e.clientX - this.last.x, e.clientY - this.last.y); this.last = { x: e.clientX, y: e.clientY };
    };
    this._up = (e) => { this.dragging = false; dom.releasePointerCapture?.(e.pointerId); };
    this._wheel = (e) => { if (!this.enabled) return; e.preventDefault(); this.speedMul = Math.min(20, Math.max(0.05, this.speedMul * Math.exp(-e.deltaY * 0.0015))); this.onSpeed?.(this.speedMul); };
    this._kd = (e) => { if (!this.enabled || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.ctrlKey || e.metaKey) return; this.keys.add(e.code); if (/^(KeyW|KeyA|KeyS|KeyD|KeyQ|KeyE|Space)$/.test(e.code)) e.preventDefault(); };
    this._ku = (e) => this.keys.delete(e.code);
    this._blur = () => this.keys.clear();
    dom.addEventListener('pointerdown', this._down); window.addEventListener('pointermove', this._move); window.addEventListener('pointerup', this._up);
    dom.addEventListener('wheel', this._wheel, { passive: false });
    window.addEventListener('keydown', this._kd); window.addEventListener('keyup', this._ku); window.addEventListener('blur', this._blur);
  }
  look(dx, dy) {
    const up = this.camera.up; const sens = 0.0042;
    this.camera.getWorldDirection(this._f);
    this._f.applyQuaternion(this._q.setFromAxisAngle(up, -dx * sens));
    this._r.crossVectors(this._f, up).normalize();
    const nf = this._v.copy(this._f).applyQuaternion(this._q.setFromAxisAngle(this._r, -dy * sens));
    if (Math.abs(nf.dot(up)) < 0.985) this._f.copy(nf);
    this.camera.lookAt(this._v.copy(this.camera.position).add(this._f));
    this.onChange();
  }
  update(dt) {
    if (!this.enabled || !this.keys.size) return;
    const up = this.camera.up; const k = this.keys;
    this.camera.getWorldDirection(this._f); this._r.crossVectors(this._f, up).normalize();
    const mv = new THREE.Vector3();
    if (k.has('KeyW')) mv.add(this._f); if (k.has('KeyS')) mv.sub(this._f);
    if (k.has('KeyD')) mv.add(this._r); if (k.has('KeyA')) mv.sub(this._r);
    if (k.has('KeyE') || k.has('Space')) mv.add(up); if (k.has('KeyQ')) mv.sub(up);
    if (!mv.lengthSq()) return;
    const fast = k.has('ShiftLeft') || k.has('ShiftRight') ? 3.5 : 1;
    this.camera.position.addScaledVector(mv.normalize(), this.baseSpeed * this.speedMul * fast * dt);
    this.onChange();
  }
  dispose() {
    this.dom.removeEventListener('pointerdown', this._down); window.removeEventListener('pointermove', this._move); window.removeEventListener('pointerup', this._up);
    this.dom.removeEventListener('wheel', this._wheel); window.removeEventListener('keydown', this._kd); window.removeEventListener('keyup', this._ku); window.removeEventListener('blur', this._blur);
  }
}
