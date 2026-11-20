# Frontend

Static single-page app, **no build step**: ES modules + an import map, three.js and `@mkkellogg/gaussian-splats-3d` vendored under `vendor/`.
The backend serves this folder at `/`; open the backend URL. Hash routes: `#/` projects, `#/p/:id` project, `#/p/:id/view[/:editId]` viewer (optionally showing edit result), `#/p/:id/edit/:editId` edit wizard (opens over the viewer).

```
js/api.js            fetch wrapper (+ signed-URL helper)      js/state.js     polling helper, session
js/ui.js             tiny DOM helpers, buttons, dialogs        js/components/  GPU status badge / banner
js/pages/            login, projects, project, viewer          js/viewer/      SplatViewer (three + splats), fly controls
js/wizard/           pick · describe · review · reference · apply · rebuild  (one file per wizard step)
```
Design notes: dark UI, one accent colour (mask pink) for everything that is "the object"; the viewer pauses rendering while the wizard covers it; all long steps show a task panel (stage, %, `detail` such as “frame 12 of 80”) fed by `GET /tasks/{id}`.
Splats are loaded as DropInViewer groups so original and edited scenes stay in memory and swap without moving the camera.
`tests/make_splat.mjs` writes a small fixture splat for viewer tests.
