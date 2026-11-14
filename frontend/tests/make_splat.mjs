// Generates a small synthetic room as a .splat (32-byte records) for UI smoke tests.
// COLMAP-style axes: +Y is DOWN, so the floor is at y=+1.2 and the ceiling at y=-1.6.
import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const dir = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
mkdirSync(dir, { recursive: true });
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

function build(sofaColor) {
  const pts = [];
  const add = (x, y, z, c, s = 0.05) => pts.push([x, y, z, s, c]);
  const noise = (c, a = 18) => c.map((v) => Math.max(0, Math.min(255, v + (rnd() - 0.5) * a)));
  for (let i = 0; i < 4200; i++) { const x = (rnd() - 0.5) * 6; const z = rnd() * 6 - 1; const checker = (Math.floor(x * 1.5) + Math.floor(z * 1.5)) & 1; add(x, 1.2, z, noise(checker ? [150, 118, 84] : [128, 98, 68])); }
  for (let i = 0; i < 2600; i++) { const x = (rnd() - 0.5) * 6; const y = rnd() * 2.8 - 1.6; add(x, y, 5, noise([212, 208, 198], 10), 0.07); }
  for (let i = 0; i < 1500; i++) { const z = rnd() * 6 - 1; const y = rnd() * 2.8 - 1.6; add(-3, y, z, noise([190, 200, 214], 10), 0.07); add(3, y, z, noise([214, 198, 190], 10), 0.07); }
  for (let i = 0; i < 800; i++) add((rnd() - 0.5) * 6, -1.6, rnd() * 6 - 1, noise([232, 232, 230], 6), 0.08);
  for (let i = 0; i < 2600; i++) { const x = -1.6 + rnd() * 2.2; const z = 2.6 + rnd() * 0.9; const y = 1.2 - rnd() * 0.55; add(x, y, z, noise(sofaColor, 16), 0.04); }
  for (let i = 0; i < 900; i++) { const x = -1.6 + rnd() * 2.2; const y = 1.2 - 0.55 - rnd() * 0.5; add(x, y, 3.5, noise(sofaColor, 16), 0.04); }
  for (let i = 0; i < 1000; i++) { const a = rnd() * Math.PI * 2; const r = rnd() * 0.45; add(1.7 + Math.cos(a) * r, 0.6, 2.4 + Math.sin(a) * r, noise([86, 60, 40], 12), 0.04); }
  for (let i = 0; i < 300; i++) { add(1.7 + (rnd() - 0.5) * 0.06, 0.6 + rnd() * 0.6, 2.4, noise([60, 40, 28], 8), 0.03); }
  for (let i = 0; i < 400; i++) { const a = rnd() * Math.PI * 2; const b = rnd() * Math.PI; add(2.2 + Math.sin(b) * Math.cos(a) * 0.25, -0.2 + Math.cos(b) * 0.25, 3.8 + Math.sin(b) * Math.sin(a) * 0.25, noise([255, 224, 140], 10), 0.04); }
  const buf = Buffer.alloc(pts.length * 32);
  pts.forEach(([x, y, z, s, c], i) => {
    const o = i * 32;
    buf.writeFloatLE(x, o); buf.writeFloatLE(y, o + 4); buf.writeFloatLE(z, o + 8);
    buf.writeFloatLE(s, o + 12); buf.writeFloatLE(s, o + 16); buf.writeFloatLE(s, o + 20);
    buf[o + 24] = c[0]; buf[o + 25] = c[1]; buf[o + 26] = c[2]; buf[o + 27] = 235;
    buf[o + 28] = 255; buf[o + 29] = 128; buf[o + 30] = 128; buf[o + 31] = 128;
  });
  return buf;
}
writeFileSync(join(dir, 'room.splat'), build([168, 92, 60]));
writeFileSync(join(dir, 'room_edit.splat'), build([30, 96, 58]));
console.log('wrote splats');
