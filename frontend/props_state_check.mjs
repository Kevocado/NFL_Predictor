/**
 * props_state_check.mjs -- the frontend half of the player-props empty-state bug.
 *
 * The reader-visible symptom lives here, not in Python: `PlayerPropsPage` called
 * `api.playerProps(...).then(setProps)` with no `.catch`, so a backend 503 left
 * `props` at its initial `[]` and rendered the same markup as a genuinely empty
 * week. "Still loading", "loaded and empty" and "failed" were one state, and all
 * three rendered a zero-row grid of column headers besides it.
 *
 * There is no JS test runner in this project (no vitest/jest, and CI gates on
 * pytest + ruff + tsc), so this is a standalone node script: it renders the real
 * component through vite's SSR pipeline with `fetch` stubbed, and asserts on the
 * markup. Run it with `node props_state_check.mjs` from `frontend/`.
 *
 * This file is on loan. It is scheduled to be replaced by a pure state function
 * plus node's built-in `node --test`, with no new dependencies (jsdom only for
 * wiring the effect, and only until the state is extracted). That rewrite is a
 * separate PR; it must not be started inside a bug fix.
 *
 * tsc cannot stand in for this: it only typechecks, so deleting the error branch
 * or the loading branch still compiles clean. M9 and M11 in
 * ../tests/mutation_check.py are exactly those two cases, and both were caught
 * here rather than by the compiler.
 *
 * Lint coverage: this file is deliberately outside `tsconfig.app.json`'s
 * `include`, because it is a Node script rather than app code and
 * `tsconfig.app.json` has no `allowJs`. `checkJs` was tried and is not the
 * answer: it demands JSDoc annotations on every parameter and a `@types/jsdom`
 * dependency to get past `import('jsdom')`, which buys nothing for a script the
 * replacement PR deletes. What actually covers this file is `oxlint`, which does
 * lint it -- verified by injecting an unused binding and watching it warn, not by
 * reading the config. So: linted, not typechecked.
 */
import { createServer } from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';

const EMPTY_STATE_SENTENCE = 'No player projection props available for this week yet.';
const STALE_NOTICE = 'earlier build';

let failures = 0;
let currentTest = '';

// An unhandled rejection is a FAILURE, not a crash.
//
// This line is why M12 is a real catch. Deleting the component's `.catch`
// restores the original swallowed rejection, node then exits non-zero on the
// unhandled promise, and the previous harness scored that non-zero exit as
// "caught" -- for the wrong reason, and only by accident. Recording it as a
// failure means the mutation is caught by an assertion that looked at the markup
// rather than by the process falling over.
const unhandled = [];
process.on('unhandledRejection', (reason) => {
  currentTest = currentTest || 'unhandled rejection';
  unhandled.push(String(reason));
  assert(false, `an unhandled rejection escaped the component: ${String(reason)}`);
});

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
 * `respond` returns `{status, body, headers}`. A never-settling promise is passed
 * through untouched, which is how the in-flight case produces a request that
 * never comes back. A non-async stub could not do that: the client awaits
 * `res.json()` either way, so it would settle and the case would be
 * indistinguishable from a normal error.
 */
async function render(respond) {
  const realFetch = globalThis.fetch;
  globalThis.fetch = (url) => {
    const outcome = respond(String(url));
    if (outcome && typeof outcome.then === 'function') return outcome;
    const { status = 200, body = null, headers = {} } = outcome;
    return Promise.resolve({
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? 'OK' : 'Service Unavailable',
      headers: { get: (name) => headers[name] ?? null },
      json: async () => body,
    });
  };
  try {
    // State freshness comes from a fresh React root per case (below), not from a
    // re-import: vite's SSR transform chokes on a cache-busting query on a .tsx
    // module, and a stale module would carry the previous case's useState over
    // and let a later case pass for the wrong reason.
    const mod = await server.ssrLoadModule('/src/pages/PlayerPropsPage.tsx');
    return await renderWithEffects(mod.PlayerPropsPage, { season: 2026, week: 3 });
  } finally {
    globalThis.fetch = realFetch;
  }
}

/**
 * `renderToString` skips effects entirely, so the page would always render in its
 * loading state and most assertions below would be trivially true. This mounts
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
  // Two flushes: one for the fetch promise to settle and set state, one for the
  // re-render that state change schedules. A single flush renders the loading
  // state and most of the assertions below would pass for free.
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

const PROP_ROW = (id, name) => ({
  player_id: id, player_name: name, anytime_td_prob: 0.42, passing_yards: 275,
});

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
    body: { detail: 'Player props are temporarily unavailable for this game.' },
  }));
  assert(
    !html.includes(EMPTY_STATE_SENTENCE),
    `a failed fetch must not claim the week has no props (got: ${JSON.stringify(html.slice(0, 200))})`,
  );
  assert(html.includes('role="alert"'), 'a failed fetch is announced to the reader');
  assert(
    html.includes('temporarily unavailable'),
    `the reader is told why it failed (got: ${JSON.stringify(html.slice(0, 200))})`,
  );

  // --- 3. a 200 with rows shows the props, not the empty state -------------
  currentTest = 'populated week';
  html = await render(() => ({ status: 200, body: [PROP_ROW('p1', 'A. Back')] }));
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
  // sentence to mean something other than what it says. The stub never settles,
  // so this is the only state this render can be in.
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

  // --- 6. no zero-row grid of headers while still loading -------------------
  // "Loading…" above a table of column headers with nothing under them reads as
  // an empty result, not a pending request.
  currentTest = 'in-flight request';
  assert(!html.includes('<th>'), 'a pending fetch does not render an empty table skeleton');

  // --- 7. a stale week is labelled, not passed off as this build's ---------
  // The route sets X-Player-Props-Stale when the rows were carried forward from
  // an earlier build. A key nothing reads is the defect, so the notice has to be
  // on screen.
  currentTest = 'stale week';
  html = await render(() => ({
    status: 200,
    body: [PROP_ROW('p1', 'A. Back')],
    headers: { 'X-Player-Props-Stale': 'true' },
  }));
  assert(
    html.includes(STALE_NOTICE),
    `a stale week says so (got: ${JSON.stringify(html.slice(0, 240))})`,
  );
  assert(html.includes('A. Back'), 'a stale week still shows its rows');
  assert(!html.includes(EMPTY_STATE_SENTENCE), 'a stale week with rows is not an empty week');

  // --- 8. a stale week with no rows does not claim the empty state ---------
  // Stale-and-empty is the case where both messages want to render. Only the
  // stale notice is true, and only one of the two may appear.
  currentTest = 'stale, empty week';
  html = await render(() => ({
    status: 200, body: [], headers: { 'X-Player-Props-Stale': 'true' },
  }));
  assert(
    !html.includes(EMPTY_STATE_SENTENCE),
    `a stale empty week does not also claim the week has no props (got: ${JSON.stringify(html.slice(0, 240))})`,
  );
  assert(html.includes(STALE_NOTICE), 'a stale empty week still says the rows are stale');

  // --- 9. a fresh week carries no stale notice -----------------------------
  currentTest = 'fresh week';
  html = await render(() => ({ status: 200, body: [PROP_ROW('p1', 'A. Back')] }));
  assert(
    !html.includes(STALE_NOTICE),
    'a week rebuilt this run is not labelled stale',
  );
} finally {
  await server.close();
}

// Two machine-readable lines, for tests/mutation_check.py.
//
// The exit code alone cannot be trusted here and never is: a runner that cannot
// import jsdom exits non-zero exactly like a runner that caught a real assertion,
// and the first version of the harness read the two as the same thing. So the
// harness requires the literal success sentence before believing a pass, and a
// `FAIL` marker before believing a failure -- and treats anything else as
// INCONCLUSIVE rather than as evidence either way.
if (failures === 0) {
  console.log('all frontend state assertions passed');
  console.log('FRONTEND RESULT: PASS');
} else {
  console.log(`${failures} frontend assertion(s) failed`);
  console.log('FRONTEND RESULT: FAIL');
}
process.exit(failures === 0 ? 0 : 1);
