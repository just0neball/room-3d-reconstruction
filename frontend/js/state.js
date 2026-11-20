// Tiny pub/sub + polling helper. No framework.
export const store = { auth: { required: false }, gpu: null };

const bus = new EventTarget();
export const on = (name, fn) => {
  const h = (e) => fn(e.detail);
  bus.addEventListener(name, h);
  return () => bus.removeEventListener(name, h);
};
export const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }));

/** Run fn every `ms` until it returns true or the returned stop() is called. Never overlaps itself. */
export function poller(fn, ms, { immediate = false } = {}) {
  let stopped = false; let timer;
  const tick = async () => {
    if (stopped) return;
    let done = false;
    try { done = (await fn()) === true; } catch { /* keep polling; transient network errors are normal on tunnels */ }
    if (!stopped && !done) timer = setTimeout(tick, ms);
  };
  timer = setTimeout(tick, immediate ? 0 : ms);
  return () => { stopped = true; clearTimeout(timer); };
}
