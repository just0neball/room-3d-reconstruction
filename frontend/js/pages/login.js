import { api, setToken } from '../api.js';
import { h, button, errMsg } from '../ui.js';

export function mount(root, { onDone }) {
  let mode = 'login';
  const err = h('p', { class: 'form-error', role: 'alert' });
  const email = h('input', { class: 'input', type: 'email', required: true, autocomplete: 'username', id: 'em' });
  const pass = h('input', { class: 'input', type: 'password', required: true, minLength: 8, autocomplete: 'current-password', id: 'pw' });
  const title = h('h1', null, 'Sign in');
  const submit = button('Sign in', { kind: 'primary', type: 'submit' });
  const swap = h('button', { type: 'button', class: 'linkish' }, 'Create an account');
  const form = h('form', { class: 'card auth-card stack', onsubmit: async (e) => {
    e.preventDefault(); err.textContent = ''; submit.disabled = true;
    try { const r = mode === 'login' ? await api.login(email.value, pass.value) : await api.signup(email.value, pass.value); setToken(r.access_token); onDone(); }
    catch (x) { err.textContent = errMsg(x); } finally { submit.disabled = false; }
  } },
  title, h('label', { class: 'field', for: 'em' }, h('span', null, 'Email'), email), h('label', { class: 'field', for: 'pw' }, h('span', null, 'Password'), pass,
    h('small', { class: 'hint' }, 'At least 8 characters.')), err, submit, swap);
  swap.onclick = () => {
    mode = mode === 'login' ? 'signup' : 'login';
    title.textContent = mode === 'login' ? 'Sign in' : 'Create account';
    submit.lastChild.textContent = mode === 'login' ? 'Sign in' : 'Create account';
    swap.textContent = mode === 'login' ? 'Create an account' : 'I already have an account';
    pass.autocomplete = mode === 'login' ? 'current-password' : 'new-password';
  };
  root.append(h('div', { class: 'page narrow' }, form));
  email.focus();
}
