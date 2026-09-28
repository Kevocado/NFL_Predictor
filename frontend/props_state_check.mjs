/**
 * props_state_check.mjs -- the frontend half of the player-props empty-state bug.
 *
 * The reader-visible symptom lives here, not in Python: `PlayerPropsPage` called
 * `api.playerProps(...).then(setProps)` with no `.catch`, so a backend 503 left
 * `props` at its initial `[]` and rendered the same markup as a genuinely empty
 * week. The three states a reader must be able to tell apart -- still loading,
 * loaded and empty, and failed -- were one.
 *
 * There is no JS test runner in this project (no vitest/jest, and CI gates on
 * pytest + ruff + tsc), so this is a standalone node script: it renders the real
 * component through vite's SSR pipeline with `fetch` stubbed, and asserts on the
 * markup. Run it with `node props_state_check.mjs` from `frontend/`.
 *
 * M9 in ../tests/mutation_check.py points here: deleting the error branch leaves
 * every assertion below green, because tsc only typechecks. This script is what
 * actually pins the behaviour.
 */
import { createServer } from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';

const EMPTY_STATE_SENTENCE = 'No player projection props available for this specific game yet.';

let failures = 0;
let currentTest = '';

function assert(cond, msg) {
  if (cond) {
    console.log(`  ok   ${currentTest}: ${msg}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${currentTest}: ${msg}`);
  }
}

const server = await createServer({
  configFile: false,
  logLevel: 'error',
  plugins: [react()],
  server: { middlewareMode: true },
  appType: 'custom',
});

/**
 * Render PlayerPropsPage with `fetch` replaced by `respond`.
 *
 * `respond` returning a promise is passed straight through, which is how the
 * in-flight case produces a fetch that never settles. Returning a never-settling
 * value from a non-async stub would NOT work: the client awaits `res.json()`
 * either way, so it would settle and the case would be indistinguishable from a
 * normal error.
 */
async function render(respond) {
  const realFetch = globalThis.fetch;
  globalThis.fetch = (url) => {
    const outcome = respond(String(url));
    if (outcome && typeof outcome.then === 'function') return outcome;
    const { status, body } = outcome;
    return Promise.resolve({
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? 'OK' : 'Service Unavailable',
      json: async () => body,
    });
  };
  try {
    // State freshness comes from a fresh React root per case (below), not from a
    // re-import: vite's SSR transform chokes on a cache-busting query on a .tsx
    // module, and a stale module would carry the previous case's useState over
    // and let a later case pass for the wrong reason.
    const mod = await server.ssrLoadModule('/src/pages/PlayerPropsPage.tsx');
    const html = await renderWithEffects(mod.PlayerPropsPage, { season: 2026, week: 3 });
    return html;
  } finally {
    globalThis.fetch = realFetch;
  }
}

/**
 * `renderToString` skips effects entirely, so the page would always render in its
 * loading state and every assertion above would be trivially true. This mounts
 * the real component with react-dom/client into a jsdom document and lets the
 * effect's promise settle before reading the markup.
 */
async function renderWithEffects(Component, props) {
  const { JSDOM } = await import('jsdom');
  const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    url: 'http://localhost/',
  });
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  // node 24 defines `navigator` as a getter-only global, so a plain assignment
  // throws. defineProperty is the supported override.
  Object.defineProperty(globalThis, 'navigator', {
    value: dom.window.navigator, configurable: true, writable: true,
  });
  globalThis.HTMLElement = dom.window.HTMLElement;
  globalThis.Node = dom.window.Node;
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;

  const { createRoot } = await import('react-dom/client');
  const container = dom.window.document.getElementById('root');
  const root = createRoot(container);
  const { act } = await import('react');

  await act(async () => {
    root.render(React.createElement(Component, props));
  });
  // Two flushes: one for the fetch promise to reject and set state, one for the
  // re-render that state change schedules. A single flush renders the loading
  // state and every "is it an error" assertion below would pass for free.
  //
  // Bounded, not open-ended: a stub that never settles (the in-flight case) would
  // make `act` wait forever, so the flushes race a cap. A case needing longer than
  // the cap surfaces as a failed assertion on the loading markup, not a pass.
  await Promise.race([
    (async () => {
      await act(async () => {});
      await act(async () => {});
    })(),
    new Promise((resolve) => setTimeout(resolve, 1500)),
  ]);

  const html = container.innerHTML;
  await act(async () => root.unmount());
  dom.window.close();
  return html;
}

try {
  // --- 1. a genuine empty week says the empty-state sentence ---------------
  currentTest = 'empty week';
  let html = await render(() => ({ status: 200, body: [] }));
  assert(
    html.includes(EMPTY_STATE_SENTENCE),
    `a 200 [] renders the empty-state sentence (got: ${JSON.stringify(html.slice(0, 160))})`,
  );
  assert(!html.includes('role="alert"'), 'an honest empty week is not styled as an error');

  // --- 2. a 503 does NOT render the empty-state sentence -------------------
  currentTest = 'failed fetch';
  html = await render(() => ({
    status: 503,
    body: { detail: 'player props for season 2026 week 3 could not be loaded: timed out' },
  }));
  assert(
    !html.includes(EMPTY_STATE_SENTENCE),
    `a failed fetch must not claim the week has no props (got: ${JSON.stringify(html.slice(0, 200))})`,
  );
  assert(html.includes('role="alert"'), 'a failed fetch is announced to the reader');
  assert(
    html.includes('timed out'),
    `the reader is told why it failed (got: ${JSON.stringify(html.slice(0, 200))})`,
  );

  // --- 3. a 200 with rows shows the props, not the empty state -------------
  currentTest = 'populated week';
  html = await render(() => ({
    status: 200,
    body: [{ player_id: 'p1', player_name: 'A. Back', anytime_td_prob: 0.42, passing_yards: 275 }],
  }));
  assert(html.includes('A. Back'), 'a populated week renders the player');
  assert(!html.includes(EMPTY_STATE_SENTENCE), 'a populated week does not render the empty state');
  assert(!html.includes('role="alert"'), 'a populated week is not an error');

  // --- 4. the failure is not silently swallowed into a blank table ---------
  currentTest = 'failure is not a blank table';
  html = await render(() => ({ status: 503, body: { detail: 'upstream data gap' } }));
  assert(!html.includes('<tbody></tbody>'), 'a failed fetch is not rendered as a zero-row table');

  // --- 5. a request still in flight says so, and claims nothing ------------
  // Without a loading state the page renders the empty-state sentence for a
  // fetch that has not come back yet, which is a third way for the same
  // sentence to mean something other than what it says. The stub here never
  // settles, so this is the only state this render can be in.
  currentTest = 'in-flight request';
  html = await render(() => new Promise(() => {})); // never settles
  assert(
    html.includes('Loading'),
    `a pending fetch announces itself (got: ${JSON.stringify(html.slice(0, 200))})`,
  );
  assert(
    !html.includes(EMPTY_STATE_SENTENCE),
    'a pending fetch does not claim the week has no props',
  );
} finally {
  await server.close();
}

console.log(failures === 0 ? '\nall frontend state assertions passed' : `\n${failures} frontend assertion(s) failed`);
process.exit(failures === 0 ? 0 : 1);
