import { test, expect, Page } from "@playwright/test";
import { AxeBuilder } from "@axe-core/playwright";
import { stubBackend } from "./stub";

// Wayfinder view e2e evidence (cond-0394), at 1280×800 and 390×844 (both
// configured projects). The §10.5 stub serves the fleet; this spec adds the
// tracker routes itself (registered after stubBackend, so they win) with a
// small in-memory tracker: one wayfinder map, four tickets in four different
// states, and two external link endpoints (one actual blocker, one context
// neighbour with relates/duplicates edges).

const SHOTS = "e2e/__screenshots__/wayfinder";

const VOCAB = {
  statuses: ["open", "triage", "in-progress", "blocked", "resolved", "closed", "wontfix", "duplicate"],
  terminal_statuses: ["closed", "duplicate", "wontfix"],
  statuses_by_kind: { issue: ["open", "triage", "in-progress", "blocked", "resolved", "closed", "wontfix", "duplicate"], feature: ["open", "triage", "in-progress", "blocked", "resolved", "closed", "wontfix", "duplicate"] },
  terminal_statuses_by_kind: { issue: ["closed", "duplicate", "wontfix"], feature: ["closed", "duplicate", "wontfix"] },
  item_kinds: ["issue", "feature"],
  severities: ["P0", "P1", "P2", "P3", "P4", "unset"],
  scope_kinds: ["path", "session", "git_remote", "project_id"],
  link_kinds: ["blocks", "relates", "duplicates", "caused-by", "part-of"],
  project_statuses: ["active", "archived"],
};

function issue(over: Record<string, unknown>) {
  return {
    key: "",
    title: "",
    project_id: "cao-system",
    kind: "issue",
    body: "",
    status: "open",
    severity: "unset",
    component: null,
    reporter: null,
    assignee: null,
    labels: [],
    failing_command: null,
    reproduction_steps: null,
    evidence: null,
    resolution: null,
    session_name: null,
    terminal_id: null,
    source_path: null,
    duplicate_of: null,
    origin: "api",
    created_at: "2026-08-10T00:00:00Z",
    updated_at: "2026-08-10T00:00:00Z",
    closed_at: null,
    comments: [],
    events: [],
    links: [],
    ...over,
  };
}

const MAP = issue({
  key: "cond-0001",
  title: "Find the deploy path",
  labels: ["wayfinder:map", "effort:deploy"],
  body: "## Destination\n\nDecide the cutover strategy.\n\n## Notes\n\nAsk the operator before billing.",
});
const T_RESEARCH = issue({ key: "cond-0002", title: "research providers", labels: ["wayfinder:research"] });
const T_GRILL = issue({ key: "cond-0003", title: "grill the operator", assignee: "terra", labels: ["wayfinder:grilling"] });
const T_PROTO = issue({ key: "cond-0004", title: "prototype the UI", status: "closed", closed_at: "2026-08-12T00:00:00Z" });
const T_MIGRATE = issue({ key: "cond-0005", title: "migrate the store" });
const EXT = issue({ key: "cond-0009", title: "quota repair" });
const EXT2 = issue({ key: "cond-0010", title: "prior art survey" });
const ALL = [MAP, T_RESEARCH, T_GRILL, T_PROTO, T_MIGRATE, EXT, EXT2];

const PROJECTION = {
  map: MAP,
  children: [
    { ...T_RESEARCH, blocked_by: [], frontier: true },
    { ...T_GRILL, blocked_by: [], frontier: false },
    { ...T_PROTO, blocked_by: [], frontier: false },
    { ...T_MIGRATE, blocked_by: ["cond-0009"], frontier: false },
  ],
  frontier: ["cond-0002"],
  links: [
    { id: 1, kind: "part-of", from_key: "cond-0002", to_key: "cond-0001" },
    { id: 2, kind: "part-of", from_key: "cond-0003", to_key: "cond-0001" },
    { id: 3, kind: "part-of", from_key: "cond-0004", to_key: "cond-0001" },
    { id: 4, kind: "part-of", from_key: "cond-0005", to_key: "cond-0001" },
    { id: 5, kind: "blocks", from_key: "cond-0009", to_key: "cond-0005" },
    { id: 6, kind: "relates", from_key: "cond-0002", to_key: "cond-0003" },
    { id: 7, kind: "relates", from_key: "cond-0002", to_key: "cond-0010" },
    { id: 8, kind: "duplicates", from_key: "cond-0010", to_key: "cond-0004" },
  ],
  // Every non-member endpoint of the links above, each carrying the member
  // children it actually benches — cond-0010 is context, not a blocker.
  external: [
    { ...EXT, blocking: ["cond-0005"] },
    { ...EXT2, blocking: [] },
  ],
  progress: { total: 4, open: 3, terminal: 1, resolved: 0, claimed: 1, frontier: 1 },
};

const FACETS = {
  project_id: "cao-system",
  labels: [
    { label: "effort:deploy", total: 5, open: 4 },
    { label: "wayfinder:map", total: 1, open: 1 },
    { label: "wayfinder:research", total: 1, open: 1 },
  ],
  unlabeled: 2,
  unlabeled_open: 2,
};

const PROJECT = {
  id: "cao-system",
  name: "CAO System",
  description: "Conductor and its fork",
  status: "active",
  issue_prefix: "cond",
  next_issue_number: 10,
  created_at: "2026-08-01T00:00:00Z",
  updated_at: "2026-08-01T00:00:00Z",
  counts: {
    total: 5,
    open: 4,
    by_status: { open: 4, closed: 1 },
    by_kind: { issue: { total: 5, open: 4 }, feature: { total: 0, open: 0 } },
    all_total: 5,
    all_open: 4,
  },
  scopes: [],
};

interface TrackerStubOptions {
  /** When true, the first map-body PATCH 409s (a concurrent edit landed). */
  staleFirstPatch?: boolean;
  /** When true, claiming cond-0002 conflicts with observed owner terra. */
  claimConflict?: boolean;
  /** When true, the project has no maps at all. */
  noMaps?: boolean;
}

async function stubTracker(page: Page, options?: TrackerStubOptions) {
  let patchCount = 0;
  const calls: Array<{ method: string; url: string; body: unknown }> = [];
  await page.route("**/tracker/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();
    const json = (data: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(data) });

    if (path === "/tracker/vocabulary") return json(VOCAB);
    if (path === "/tracker/projects" && method === "GET") return json([PROJECT]);
    if (path === `/tracker/projects/${PROJECT.id}/labels`) return json(FACETS);
    if (path === `/tracker/projects/${PROJECT.id}/options`) {
      const field = url.searchParams.get("field") ?? "label";
      const values = field === "label"
        ? FACETS.labels.map((item) => ({ value: item.label, total: item.total, open: item.open }))
        : [];
      return json({ project_id: PROJECT.id, field, query: url.searchParams.get("q") ?? "", matching_total: values.length, options: values });
    }
    if (path === `/tracker/projects/${PROJECT.id}`) return json(PROJECT);
    if (path === "/tracker/issues/cond-0001/map") {
      return json(options?.noMaps ? { detail: "no such issue" } : PROJECTION, options?.noMaps ? 404 : 200);
    }
    if (path === "/tracker/issues" && method === "GET") {
      let rows = options?.noMaps ? [] : ALL;
      if (url.searchParams.get("label") === "wayfinder:map") rows = options?.noMaps ? [] : [MAP];
      if (url.searchParams.get("unlabeled") === "true") rows = [T_MIGRATE, EXT];
      return json({ total: rows.length, limit: 50, offset: 0, issues: rows });
    }
    if (path === "/tracker/issues/cond-0002/claim" && method === "POST") {
      if (options?.claimConflict) {
        return json(
          { detail: { message: "cond-0002 could not be claimed", code: "conflict", observed_assignee: "terra" } },
          409,
        );
      }
      return json({ ...T_RESEARCH, assignee: "dashboard", claimed: true, already_claimed: false });
    }
    const unclaimMatch = /^\/tracker\/issues\/([^/]+)\/unclaim$/.exec(path);
    if (unclaimMatch && method === "POST") {
      const key = unclaimMatch[1];
      const found = ALL.find((i) => i.key === key) ?? T_GRILL;
      return json({ ...found, assignee: null, unclaimed: true, was_claimed: true });
    }
    if (path === "/tracker/issues/cond-0001" && method === "PATCH") {
      const body = request.postDataJSON() as Record<string, unknown>;
      calls.push({ method, url: path, body });
      patchCount += 1;
      if (options?.staleFirstPatch && patchCount === 1) {
        return json({
          detail: { message: "cond-0001 has changed", code: "conflict", current_updated_at: "2026-08-11T09:30:00Z" },
        }, 409);
      }
      return json({ ...MAP, body: String(body.body ?? MAP.body), updated_at: "2026-08-11T09:30:00Z" });
    }
    const detailMatch = /^\/tracker\/issues\/([^/]+)$/.exec(path);
    if (detailMatch && method === "GET") {
      const found = ALL.find((i) => i.key === detailMatch[1]);
      if (found) {
        if (found.key === "cond-0001" && options?.staleFirstPatch) {
          // The re-read after a stale write carries the newer version.
          return json({ ...MAP, updated_at: "2026-08-11T09:30:00Z" });
        }
        return json(found);
      }
      return json({ detail: "no such issue" }, 404);
    }
    return json({ detail: `unstubbed ${method} ${path}` }, 404);
  });
  return { calls };
}

async function openWayfinder(page: Page) {
  await gotoWayfinderTab(page);
  await page.getByRole("listbox", { name: "Wayfinder maps" }).waitFor();
}

/** The navigation half of openWayfinder, without waiting for a successful
 * map list (error-state tests never render one). Center-scrolling before the
 * click keeps the tab from moving mid-click when the panel below reloads. */
async function gotoWayfinderTab(page: Page) {
  await page.goto("/");
  await page.getByRole("tab", { name: "Projects" }).click();
  await page.getByText("CAO System").first().waitFor();
  const tab = page.getByRole("tab", { name: "Wayfinder" });
  await tab.evaluate((el) => el.scrollIntoView({ block: "center" }));
  await tab.click();
}

test("map browser → map view: destination, progress, frontier, graph, legend, list", async ({ page }, testInfo) => {
  await stubBackend(page);
  await stubTracker(page);
  await openWayfinder(page);
  await page.screenshot({ path: `${SHOTS}/${testInfo.project.name}-map-list.png`, fullPage: false });

  await page.getByRole("option", { name: /Find the deploy path/ }).click();
  const mapView = page.getByTestId("map-view");
  await mapView.waitFor();
  await expect(mapView.getByText("Decide the cutover strategy.")).toBeVisible();
  await expect(mapView.getByText("1/4 done · 1 claimed · 1 on the frontier")).toBeVisible();
  // The frontier is ordered and takeable; the blocked/claimed tickets are not in it.
  const frontier = mapView.getByLabel("Frontier");
  await expect(frontier.getByText("research providers")).toBeVisible();
  await expect(frontier.getByText("migrate the store")).not.toBeVisible();
  // The graph canvas, its legend, and the accessible list all render.
  await expect(page.getByTestId("map-graph-canvas")).toBeVisible();
  const legend = page.getByTestId("map-graph-legend");
  await expect(legend.getByText("blocks →")).toBeVisible();
  await expect(legend.getByText("part-of →")).toBeVisible();
  await expect(legend.getByText("external")).toBeVisible();
  const children = page.getByTestId("map-children");
  await expect(children.getByText("claimed by terra")).toBeVisible();
  await expect(children.getByText("blocked by cond-0009")).toBeVisible();
  // Every external link endpoint renders with its relationship and direction
  // as text; the benching blocks phrase is the explicit blocker marker.
  const external = page.getByTestId("map-external");
  await expect(page.getByText("External links (2)")).toBeVisible();
  await expect(external.getByText("quota repair")).toBeVisible();
  await expect(external.getByText("blocks cond-0005")).toBeVisible();
  await expect(external.getByText("prior art survey")).toBeVisible();
  await expect(external.getByText("relates to cond-0002")).toBeVisible();
  await expect(external.getByText("duplicates cond-0004")).toBeVisible();
  // No horizontal overflow at either viewport.
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: `${SHOTS}/${testInfo.project.name}-map-view.png`, fullPage: true });
});

test("advanced label search drives the list without rendering the whole vocabulary", async ({ page }) => {
  await stubBackend(page);
  await stubTracker(page);
  await page.goto("/");
  await page.getByRole("tab", { name: "Projects" }).click();
  await page.getByText("CAO System").first().waitFor();
  await expect(page.getByTestId("label-filter-bar")).toHaveCount(0);
  const advanced = page.getByRole("button", { name: /Advanced filters/ });
  await advanced.scrollIntoViewIfNeeded();
  await advanced.focus();
  await page.keyboard.press("Enter");
  const labelSearch = page.getByRole("combobox", { name: "Filter labels" });
  await labelSearch.fill("effort");
  await page.getByRole("option", { name: /effort:deploy.*4 open.*5 total/ }).click();
  await expect(page).toHaveURL(/label=effort%3Adeploy/);
  const unlabeled = page.getByLabel("Only items without labels");
  await unlabeled.focus();
  await page.keyboard.press("Space");
  await expect(page).toHaveURL(/unlabeled=1/);
  await expect(page.getByText("migrate the store")).toBeVisible();
  await expect(page.getByText("research providers")).not.toBeVisible();
});

test("claim conflict surfaces the observed owner; unclaim is the exit", async ({ page }, testInfo) => {
  await stubBackend(page);
  await stubTracker(page, { claimConflict: true });
  await openWayfinder(page);
  await page.getByRole("option", { name: /Find the deploy path/ }).click();
  const frontier = page.getByLabel("Frontier");
  await frontier.getByRole("button", { name: "Claim cond-0002" }).click();
  await expect(page.getByText("cond-0002 is already claimed by terra")).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/${testInfo.project.name}-claim-conflict.png`, fullPage: false });

  // The claimed ticket offers Unclaim and it releases.
  const children = page.getByTestId("map-children");
  await children.getByRole("button", { name: /Unclaim cond-0003/ }).click();
  await expect(page.getByText("cond-0003 released")).toBeVisible();
});

test("stale map-body edit: draft preserved, re-read & retry applies it", async ({ page }, testInfo) => {
  await stubBackend(page);
  const { calls } = await stubTracker(page, { staleFirstPatch: true });
  await openWayfinder(page);
  await page.getByRole("option", { name: /Find the deploy path/ }).click();
  const mapView = page.getByTestId("map-view");
  await mapView.waitFor();
  await mapView.getByRole("button", { name: "Edit" }).click();
  const body = page.getByRole("textbox", { name: "Map body" });
  await body.fill("## Destination\n\nDrafted against the old version");
  await mapView.getByRole("button", { name: "Save" }).click();

  const banner = page.getByTestId("map-body-conflict");
  await banner.waitFor();
  await expect(banner).toContainText("2026-08-11T09:30:00Z");
  await expect(body).toHaveValue("## Destination\n\nDrafted against the old version");
  await page.screenshot({ path: `${SHOTS}/${testInfo.project.name}-stale-body-conflict.png`, fullPage: false });

  await banner.getByRole("button", { name: /Re-read & retry/ }).click();
  await expect(page.getByText("cond-0001 updated")).toBeVisible();
  const patches = calls.filter((c) => c.method === "PATCH");
  expect(patches.length).toBe(2);
  expect(patches[1].body).toMatchObject({
    body: "## Destination\n\nDrafted against the old version",
    expected_updated_at: "2026-08-11T09:30:00Z",
  });
});

test("empty state names the convention", async ({ page }) => {
  await stubBackend(page);
  await stubTracker(page, { noMaps: true });
  await gotoWayfinderTab(page);
  await expect(page.getByTestId("wayfinder-empty")).toContainText("wayfinder:map");
});

test("a failed map-list load shows the error and a retry", async ({ page }) => {
  await stubBackend(page);
  await stubTracker(page);
  await page.route("**/tracker/issues**", (route) =>
    route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({ detail: "boom" }),
    }),
  );
  await gotoWayfinderTab(page);
  await expect(page.getByText("boom", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
});

test("keyboard and axe basics on the map view", async ({ page }) => {
  await stubBackend(page);
  await stubTracker(page);
  await openWayfinder(page);
  await page.getByRole("option", { name: /Find the deploy path/ }).click();
  await page.getByTestId("map-view").waitFor();
  // The accessible list is keyboard-operable: focus a ticket row, Enter selects it.
  const row = page.getByTestId("map-children").getByRole("button", { name: /migrate the store/ });
  await row.focus();
  await page.keyboard.press("Enter");
  const detail = page.getByTestId("wayfinder-detail");
  await expect(detail.getByLabel("Issue title")).toHaveValue("migrate the store");
  const scan = await new AxeBuilder({ page }).include('[data-testid="map-view"]').analyze();
  expect(scan.violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);
});
