import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as GS from 'gaussian-splats-3d';
import { FlyControls } from './fly.js';

/**
 * Splat viewer. We own the three.js scene/camera and mount each splat file as a DropInViewer group,
 * so several versions (original, edit 1, edit 2...) can stay loaded and be swapped without moving the camera.
 *
 * Events (EventTarget): 'click' {x,y,clientX,clientY}   screen position of every click that was not a drag (Week-2 hook)
 *                       'pick'  {x,y,point:[x,y,z]|null} same, but only while pickMode is on; point = nearest splat hit
 *                       'camera' camera moved
 *                       'mode'  {mode}
 */
export class SplatViewer extends EventTarget {
  constructor(container) {
    super();
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e14);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 2000);
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.dom = this.renderer.domElement;
    this.dom.tabIndex = 0; this.dom.className = 'viewer-canvas'; this.dom.setAttribute('aria-label', '3D scene. Drag to look, WASD to move.');
    container.append(this.dom);

    this.versions = new Map(); // key -> { group, status }
    this.activeKey = null; this.mode = 'orbit'; this.pickMode = false; this.paused = false; this.sceneScale = 1;

    this.orbit = new OrbitControls(this.camera, this.dom);
    this.orbit.enableDamping = true; this.orbit.dampingFactor = 0.08; this.orbit.screenSpacePanning = true;
    this.fly = new FlyControls(this.camera, this.dom);
    this.fly.onChange = () => this.dispatchEvent(new Event('camera'));
    this.orbit.addEventListener('change', () => this.dispatchEvent(new Event('camera')));
    this._bindClicks();

    this._ro = new ResizeObserver(() => this.resize()); this._ro.observe(container);
    this.resize();
    this._clock = new THREE.Clock(); this._raf = 0; this._dead = false;
    this._loop = () => {
      if (this._dead) return;
      this._raf = requestAnimationFrame(this._loop);
      if (this.paused) { this._clock.getDelta(); return; } // editor open: don't burn the GPU on a hidden scene
      const dt = Math.min(this._clock.getDelta(), 0.1);
      if (this.mode === 'fly') this.fly.update(dt); else this.orbit.update();
      this.renderer.render(this.scene, this.camera);
    };
    this._loop();
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth); const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false); this.dom.style.width = '100%'; this.dom.style.height = '100%';
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  /** up: [x,y,z] world up (COLMAP: [0,-1,0]). Must be set before the first camera pose. */
  setUp(up = [0, -1, 0]) { this.camera.up.set(...up).normalize(); this.orbit.update(); }
  setSceneScale(s) { this.sceneScale = Math.max(0.05, s); this.fly.baseSpeed = this.sceneScale * 0.25; this.camera.near = this.sceneScale * 0.004; this.camera.far = this.sceneScale * 60; this.camera.updateProjectionMatrix(); this.orbit.maxDistance = this.sceneScale * 20; }

  /** cam: {position:[x,y,z], rotation_c2w:[9 row-major, COLMAP axes], fx,fy,cx,cy} from GET /cameras. */
  setCameraPose(cam, { animate = true } = {}) {
    const p = new THREE.Vector3(...cam.position); const R = cam.rotation_c2w;
    const fwd = new THREE.Vector3(R[2], R[5], R[8]).normalize(); // third column = +Z (forward) in world
    const target = p.clone().addScaledVector(fwd, Math.max(this.sceneScale * 0.2, 0.5));
    if (cam.fy && cam.cy) { this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan((cam.cy) / cam.fy)); this.camera.updateProjectionMatrix(); }
    const apply = (pos, tgt) => { this.camera.position.copy(pos); this.orbit.target.copy(tgt); this.camera.lookAt(tgt); this.orbit.update(); };
    if (!animate) return apply(p, target);
    const p0 = this.camera.position.clone(); const t0 = this.orbit.target.clone(); const t1 = performance.now(); const dur = 650;
    const step = (now) => {
      const k = Math.min(1, (now - t1) / dur); const e = 1 - Math.pow(1 - k, 3);
      apply(p0.clone().lerp(p, e), t0.clone().lerp(target, e));
      if (k < 1 && !this._dead) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  getCameraState() { return { position: this.camera.position.toArray(), target: this.orbit.target.toArray(), quaternion: this.camera.quaternion.toArray(), fov: this.camera.fov }; }
  setCameraState(s) { if (!s) return; this.camera.position.fromArray(s.position); this.orbit.target.fromArray(s.target); this.camera.lookAt(this.orbit.target); this.orbit.update(); }

  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    if (mode === 'fly') { this.orbit.enabled = false; this.fly.enabled = true; this.dom.focus(); }
    else {
      const f = new THREE.Vector3(); this.camera.getWorldDirection(f);
      this.orbit.target.copy(this.camera.position).addScaledVector(f, Math.max(this.sceneScale * 0.25, 0.5));
      this.fly.enabled = false; this.orbit.enabled = true; this.orbit.update();
    }
    this.dispatchEvent(new CustomEvent('mode', { detail: { mode } }));
  }

  /** Load a splat file under `key`. Resolves when it is ready; the first loaded version becomes visible. */
  async load(key, splatUrl, onProgress) {
    if (this.versions.has(key)) return this.versions.get(key);
    const group = new GS.DropInViewer({ gpuAcceleratedSort: false, sharedMemoryForWorkers: false, selfDrivenMode: false, antialiased: false, dynamicScene: false });
    const entry = { group, status: 'loading', url: splatUrl };
    this.versions.set(key, entry);
    group.visible = false; this.scene.add(group);
    try {
      await group.addSplatScenes([{ path: splatUrl, format: GS.SceneFormat.Splat, splatAlphaRemovalThreshold: 5, progressiveLoad: false, onProgress }], false);
      entry.status = 'ready';
    } catch (e) { entry.status = 'error'; this.scene.remove(group); this.versions.delete(key); throw e; }
    if (!this.activeKey) this.show(key);
    return entry;
  }
  show(key) {
    for (const [k, v] of this.versions) v.group.visible = k === key;
    this.activeKey = key;
    this.dispatchEvent(new CustomEvent('version', { detail: { key } }));
  }
  has(key) { return this.versions.get(key)?.status === 'ready'; }
  drop(key) { const v = this.versions.get(key); if (v) { this.scene.remove(v.group); v.group.dispose?.(); this.versions.delete(key); if (this.activeKey === key) this.activeKey = null; } }

  // ---- clicking / picking ----
  setPickMode(on) { this.pickMode = on; this.dom.classList.toggle('picking', on); }
  _bindClicks() {
    let down = null;
    this.dom.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
    this.dom.addEventListener('pointerup', (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y); const dt = performance.now() - down.t; down = null;
      if (moved > 5 || dt > 600 || e.button !== 0) return;
      const r = this.dom.getBoundingClientRect(); const x = e.clientX - r.left; const y = e.clientY - r.top;
      this.dispatchEvent(new CustomEvent('click', { detail: { x, y, clientX: e.clientX, clientY: e.clientY } }));
      if (this.pickMode) this.dispatchEvent(new CustomEvent('pick', { detail: { x, y, point: this.pickPoint(x, y) } }));
    });
  }

  /** Nearest splat along the ray through screen px (x,y) relative to the canvas. Returns [x,y,z] or null. */
  pickPoint(x, y) {
    const entry = this.versions.get(this.activeKey); if (!entry) return null;
    const mesh = entry.group.viewer?.splatMesh ?? entry.group.splatMesh;
    const r = this.dom.getBoundingClientRect();
    if (mesh && GS.Raycaster) {
      try {
        const rc = new GS.Raycaster(); const hits = [];
        rc.setFromCameraAndScreenPosition(this.camera, new THREE.Vector2(x, y), new THREE.Vector2(r.width, r.height));
        rc.intersectSplatMesh(mesh, hits);
        if (hits.length) { hits.sort((a, b) => a.origin.distanceToSquared(this.camera.position) - b.origin.distanceToSquared(this.camera.position)); return hits[0].origin.toArray(); }
      } catch (e) { console.warn('Library raycaster failed, using fallback', e); }
    }
    return this._fallbackPick(mesh, x, y, r);
  }
  _fallbackPick(mesh, x, y, r) {
    if (!mesh?.getSplatCount) return null;
    const ndc = new THREE.Vector2((x / r.width) * 2 - 1, -(y / r.height) * 2 + 1);
    const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, this.camera);
    const ro = ray.ray.origin; const rd = ray.ray.direction;
    const n = mesh.getSplatCount(); const stride = Math.max(1, Math.floor(n / 250000));
    const c = new THREE.Vector3(); const v = new THREE.Vector3();
    let best = null; let bestT = Infinity; let near = null; let nearScore = Infinity;
    for (let i = 0; i < n; i += stride) {
      mesh.getSplatCenter(i, c, true);
      v.subVectors(c, ro); const t = v.dot(rd); if (t < 0.02) continue;
      const d = v.sub(rd.clone().multiplyScalar(t)).length();
      if (d < t * 0.025) { if (t < bestT) { bestT = t; best = c.clone(); } }
      else if (d / t < nearScore) { nearScore = d / t; near = c.clone(); }
    }
    return (best || near)?.toArray() ?? null;
  }

  dispose() {
    this._dead = true; cancelAnimationFrame(this._raf); this._ro.disconnect();
    this.fly.dispose(); this.orbit.dispose();
    for (const [, v] of this.versions) { try { v.group.dispose?.(); } catch { /* ignore */ } }
    this.renderer.dispose(); this.renderer.forceContextLoss?.(); this.dom.remove();
  }
}
