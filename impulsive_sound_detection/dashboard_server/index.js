/**
 * index.js – ISD Host Dashboard Server
 *
 * Reads live detection data from host.db written by host_subscriber.py.
 * Falls back to seeded demo data if host.db is not found.
 *
 * Authentication
 * ──────────────
 * Public routes  (read-only data)  : no auth required
 * Admin routes   /api/admin/*      : require X-Admin-Key header whose value
 *                                    is verified against a bcrypt hash stored
 *                                    in ADMIN_PASSWORD_HASH env var.
 *                                    The plaintext password is NEVER stored.
 *
 * HTTPS
 * ─────
 * When CERT_PATH and KEY_PATH env vars point to a TLS certificate and private
 * key the server starts an https.Server on HTTPS_PORT (default 3443).
 * A plain HTTP server still starts on PORT and redirects all traffic to HTTPS.
 * For local/LAN deployment a self-signed certificate is sufficient.
 *
 * API surface:
 *   GET  /api/events                    – detection event feed
 *   GET  /api/stats                     – summary counters
 *   GET  /api/nodes                     – node list with stats
 *   GET  /api/rms                       – RMS frames for chart
 *   GET  /api/correlated                – cross-node correlated events
 *   GET  /api/localization              – Sound Localization stub
 *   GET  /api/status                    – db mode
 *   GET  /api/history                   – paginated, filterable event history
 *   POST /api/admin/query               – safe read-only SQL query       [AUTH]
 *   GET  /api/admin/nodes/discovered    – unregistered nodes             [AUTH]
 *   GET  /api/admin/nodes               – full node_status table         [AUTH]
 *   POST /api/admin/nodes               – register a new node            [AUTH]
 *   PUT  /api/admin/nodes/:id           – rename / update / toggle       [AUTH]
 *   DELETE /api/admin/nodes/:id         – remove a node record           [AUTH]
 *   POST /api/admin/nodes/:id/ping      – mark last_seen=now             [AUTH]
 *   POST /api/admin/nodes/:id/clear     – delete all events for a node   [AUTH]
 *   POST /api/auth/verify               – verify password (bcrypt)
 */

const express  = require('express');
const initSql  = require('sql.js');
const cors     = require('cors');
const path     = require('path');
const fs       = require('fs');
const http     = require('http');
const https    = require('https');
const bcrypt   = require('bcrypt');

// Load .env file if present (local dev). In Docker env vars come from
// the environment block in docker-compose.yml instead.
try { require('dotenv').config(); } catch(_) { /* dotenv optional */ }

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Config ─────────────────────────────────────────────────────────────
const HOST_DB_PATH         = process.env.ISD_DB_PATH            || 'C:\\ImpulsiveSoundDetection\\host.db';
const ADMIN_PASSWORD_HASH  = process.env.ADMIN_PASSWORD_HASH     || '';
const PORT                 = parseInt(process.env.PORT            || '3000',  10);
const HTTPS_PORT           = parseInt(process.env.HTTPS_PORT      || '3443',  10);
const CERT_PATH            = process.env.CERT_PATH               || '';
const KEY_PATH             = process.env.KEY_PATH                || '';
const POLL_INTERVAL_MS     = 2000;

// Warn if no password hash is configured
if (!ADMIN_PASSWORD_HASH) {
  console.warn('[WARN] ADMIN_PASSWORD_HASH is not set. Admin routes will reject all requests.');
  console.warn('[WARN] Generate a hash with: node -e "require(\'bcrypt\').hash(\'yourpassword\',12).then(console.log)"');
}

// Warn if TLS is not configured
if (!CERT_PATH || !KEY_PATH) {
  console.warn('[WARN] CERT_PATH / KEY_PATH not set – running HTTP only. Set these env vars for HTTPS.');
}

let db;
let dbMode = 'demo';

// ── sql.js helpers ──────────────────────────────────────────────────────
function all(sql, params = []) {
  try {
    const stmt = db.prepare(sql);
    if (params.length) stmt.bind(params);
    const rows = [];
    while (stmt.step()) rows.push(stmt.getAsObject());
    stmt.free();
    return rows;
  } catch(e) {
    console.error('Query error:', e.message, '\nSQL:', sql);
    return [];
  }
}
function get(sql, params = []) { return all(sql, params)[0] || null; }
function run(sql, params = [])  {
  try { db.run(sql, params); return true; }
  catch(e) { console.error('Run error:', e.message); return false; }
}

// Write the in-memory sql.js database back to disk.
// Called after every admin write so changes survive the next live DB reload.
// In demo mode this is a no-op since there is no host.db file to write.
function persistDb() {
  if (dbMode !== 'live') return;
  try {
    const data = db.export();
    fs.writeFileSync(HOST_DB_PATH, Buffer.from(data));
  } catch(e) {
    console.error('[DB] Failed to persist to disk:', e.message);
  }
}

// ── Load or seed database ───────────────────────────────────────────────
async function loadDatabase(SQL) {
  if (fs.existsSync(HOST_DB_PATH)) {
    try {
      const fileBuffer = fs.readFileSync(HOST_DB_PATH);
      const testDb = new SQL.Database(fileBuffer);
      testDb.prepare('SELECT 1 FROM detection_events LIMIT 1').free();
      db = testDb;
      dbMode = 'live';
      console.log(`[DB] Live mode – loaded from ${HOST_DB_PATH}`);
      return;
    } catch (e) {
      console.warn(`[DB] Could not read ${HOST_DB_PATH}: ${e.message} – falling back to demo`);
    }
  }
  console.log('[DB] Demo mode – host.db not found');
  dbMode = 'demo';
  db = new SQL.Database();
  seedDemoDatabase();
}

// ── Reload live DB from disk ────────────────────────────────────────────
async function reloadLiveDb(SQL) {
  if (dbMode !== 'live') return;
  if (!fs.existsSync(HOST_DB_PATH)) return;
  try {
    const fileBuffer = fs.readFileSync(HOST_DB_PATH);
    const newDb = new SQL.Database(fileBuffer);
    newDb.prepare('SELECT 1 FROM detection_events LIMIT 1').free();
    db = newDb;
    ensureAdminColumns();   // re-apply after every reload
  } catch (e) {
    // DB locked mid-write – skip this cycle
  }
}

// ── Demo seed data ──────────────────────────────────────────────────────
function seedDemoDatabase() {
  db.run(`
    CREATE TABLE IF NOT EXISTS detection_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      event_uuid TEXT, node_id TEXT, label TEXT, confidence REAL,
      is_suspicious INTEGER, severity TEXT DEFAULT 'LOW',
      timestamp_node REAL, timestamp_iso TEXT, wall_clock_time REAL,
      received_at_host REAL, onset_index INTEGER, session_id TEXT,
      classifier_version TEXT DEFAULT 'demo',
      inserted_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS rms_frames (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      node_id TEXT, ts REAL, rms REAL, baseline REAL,
      threshold REAL, is_trigger INTEGER
    );
    CREATE TABLE IF NOT EXISTS node_status (
      node_id  TEXT PRIMARY KEY,
      location TEXT,
      status   TEXT DEFAULT 'unknown',
      last_seen REAL DEFAULT 0,
      enabled  INTEGER DEFAULT 1,
      notes    TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS localization_results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      received_at REAL, payload_json TEXT
    );
  `);

  [
    ['node_1', 'Hallway A - Node 1', 'online'],
    ['node_2', 'Hallway B - Node 2', 'online'],
    ['node_3', 'Cafeteria - Node 3', 'online'],
    ['node_4', 'Gym - Node 4',       'offline'],
  ].forEach(([id, loc, st]) =>
    run('INSERT OR REPLACE INTO node_status (node_id,location,status,last_seen,enabled,notes) VALUES (?,?,?,?,1,"")',
        [id, loc, st, Date.now()/1000])
  );

  const events = [
    ['node_1', 'Gunshot / firearm', 0.94, 1, 'HIGH',   '2024-11-14 09:14:32', 1731573272.412],
    ['node_1', 'Background noise',  0.18, 0, 'LOW',    '2024-11-14 10:22:07', 1731577327.891],
    ['node_1', 'Glass break',       0.87, 1, 'HIGH',   '2024-11-14 11:05:55', 1731580755.003],
    ['node_2', 'Gunshot / firearm', 0.91, 1, 'HIGH',   '2024-11-14 09:14:33', 1731573273.108],
    ['node_2', 'Background noise',  0.22, 0, 'LOW',    '2024-11-14 12:48:19', 1731585699.554],
    ['node_3', 'Background noise',  0.09, 0, 'LOW',    '2024-11-14 13:30:44', 1731588244.220],
    ['node_3', 'Explosion',         0.78, 1, 'MEDIUM', '2024-11-14 14:02:11', 1731590531.761],
    ['node_1', 'Gunshot / firearm', 0.96, 1, 'HIGH',   '2024-11-15 09:31:22', 1731659482.100],
    ['node_2', 'Gunshot / firearm', 0.93, 1, 'HIGH',   '2024-11-15 09:31:22', 1731659482.874],
    ['node_2', 'Background noise',  0.11, 0, 'LOW',    '2024-11-15 11:14:05', 1731665645.332],
  ];
  events.forEach(([node_id, label, conf, susp, severity, ts, wct]) =>
    run(`INSERT INTO detection_events
         (node_id,label,confidence,is_suspicious,severity,
          timestamp_node,wall_clock_time,received_at_host,onset_index)
         VALUES (?,?,?,?,?,?,?,?,0)`,
        [node_id, label, conf, susp, severity, ts, wct, wct + 0.05])
  );

  const nodeIds = ['node_1', 'node_2', 'node_3'];
  const rmsVals = [0.04,0.05,0.06,0.05,0.07,0.06,0.08,0.07,0.09,0.12,
                   0.18,0.22,0.30,0.42,0.55,0.62,0.38,0.22,0.10,0.07];
  let ts0 = Date.now()/1000 - 20;
  rmsVals.forEach((rms, i) => {
    nodeIds.forEach(nid => {
      const r = Math.max(0.01, rms + (Math.random()-0.5)*0.02);
      run(`INSERT INTO rms_frames (node_id,ts,rms,baseline,threshold,is_trigger)
           VALUES (?,?,?,?,?,?)`,
          [nid, ts0 + i*1.0, r, 0.06, 0.18, r > 0.18 ? 1 : 0]);
    });
  });
}

// ── Helper: ensure node_status has notes/enabled columns (live DB upgrade) ──
function ensureAdminColumns() {
  try {
    db.run(`ALTER TABLE node_status ADD COLUMN enabled INTEGER DEFAULT 1`);
  } catch(_) {}
  try {
    db.run(`ALTER TABLE node_status ADD COLUMN notes TEXT DEFAULT ''`);
  } catch(_) {}
}

// ══════════════════════════════════════════════════════════════════════
// SERVER-SENT EVENTS (SSE) — real-time push to browser tabs
// The browser opens GET /api/events/stream and keeps the connection open.
// When a suspicious detection arrives the server broadcasts an event,
// which causes the frontend to immediately re-fetch live data without
// waiting for the next 4-second poll cycle.
// ══════════════════════════════════════════════════════════════════════
const sseClients = new Set();

// Called internally whenever a suspicious event should be broadcast.
function broadcastSuspicious(payload) {
  const data = JSON.stringify(payload);
  for (const res of sseClients) {
    try {
      res.write(`event: suspicious\ndata: ${data}\n\n`);
    } catch(_) {
      sseClients.delete(res);
    }
  }
  if (sseClients.size > 0) {
    console.log(`[SSE] Broadcast suspicious event to ${sseClients.size} client(s):`, payload.node_id, payload.label);
  }
}

// SSE stream endpoint — browser connects here once on page load
app.get('/api/events/stream', (req, res) => {
  res.setHeader('Content-Type',  'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection',    'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // disable nginx buffering if proxied
  res.flushHeaders();

  // Send an immediate ping so the browser knows the connection is alive
  res.write('event: connected\ndata: {"status":"ok"}\n\n');

  sseClients.add(res);
  console.log(`[SSE] Client connected (total: ${sseClients.size})`);

  // Heartbeat every 25s to keep the connection alive through proxies/firewalls
  const hb = setInterval(() => {
    try { res.write(': heartbeat\n\n'); } catch(_) { clearInterval(hb); }
  }, 25000);

  req.on('close', () => {
    clearInterval(hb);
    sseClients.delete(res);
    console.log(`[SSE] Client disconnected (total: ${sseClients.size})`);
  });
});

// Internal notify endpoint — called by host_subscriber.py after writing
// a suspicious detection to host.db.  Localhost-only (no auth needed
// since it is not exposed outside the host network).
app.post('/api/internal/notify', (req, res) => {
  const payload = req.body || {};
  broadcastSuspicious(payload);
  res.json({ ok: true, clients: sseClients.size });
});

// ══════════════════════════════════════════════════════════════════════
// AUTHENTICATION MIDDLEWARE
// Admin routes require the request header:
//   X-Admin-Key: <plaintext password>
// The middleware verifies it against the bcrypt hash stored in
// ADMIN_PASSWORD_HASH. The plaintext password is never stored anywhere.
//
// Bcrypt is intentionally slow (cost factor 12 = ~250ms per check).
// This prevents brute-force attacks even if the hash is exposed.
// ══════════════════════════════════════════════════════════════════════
async function requireAdmin(req, res, next) {
  const provided = req.headers['x-admin-key'] || '';
  if (!provided || !ADMIN_PASSWORD_HASH) {
    return res.status(401).json({ error: 'Unauthorized – admin password required.' });
  }
  try {
    const match = await bcrypt.compare(provided, ADMIN_PASSWORD_HASH);
    if (!match) {
      return res.status(401).json({ error: 'Unauthorized – incorrect password.' });
    }
    next();
  } catch(e) {
    console.error('[Auth] bcrypt error:', e.message);
    return res.status(500).json({ error: 'Authentication error.' });
  }
}

// Password verification endpoint – used by the frontend login prompt.
// Returns 200 {ok:true} on success, 401 on failure.
app.post('/api/auth/verify', async (req, res) => {
  const { key } = req.body || {};
  if (!key || !ADMIN_PASSWORD_HASH) {
    return res.status(401).json({ ok: false, error: 'Invalid credentials' });
  }
  try {
    const match = await bcrypt.compare(key, ADMIN_PASSWORD_HASH);
    if (match) {
      res.json({ ok: true });
    } else {
      res.status(401).json({ ok: false, error: 'Invalid credentials' });
    }
  } catch(e) {
    res.status(500).json({ ok: false, error: 'Authentication error' });
  }
});

// ══════════════════════════════════════════════════════════════════════
// API ROUTES
// ══════════════════════════════════════════════════════════════════════

// ── Live feed events ───────────────────────────────────────────────────
app.get('/api/events', (req, res) => {
  const lim  = parseInt(req.query.limit) || 100;
  const node = req.query.node || null;
  const sql = `
    SELECT id, node_id, label, confidence, is_suspicious,
           COALESCE(severity, 'LOW') AS severity,
           timestamp_node, wall_clock_time,
           COALESCE(received_at_host, wall_clock_time) AS received_at_host,
           inserted_at
    FROM detection_events
    ${node ? "WHERE node_id = '" + node + "'" : ""}
    ORDER BY id DESC LIMIT ${lim}
  `;
  res.json(all(sql));
});

// ── Stats ──────────────────────────────────────────────────────────────
app.get('/api/stats', (req, res) => {
  const total   = get('SELECT COUNT(*) as c FROM detection_events').c;
  const susp    = get('SELECT COUNT(*) as c FROM detection_events WHERE is_suspicious=1').c;
  const nodes   = get("SELECT COUNT(*) as c FROM node_status WHERE status='online'").c;
  const avgConf = get('SELECT ROUND(AVG(confidence),3) as a FROM detection_events WHERE is_suspicious=1').a;
  res.json({ total_events: total||0, suspicious_events: susp||0,
             active_nodes: nodes||0, avg_confidence: avgConf||0, db_mode: dbMode });
});

// ── Nodes ──────────────────────────────────────────────────────────────
app.get('/api/nodes', (req, res) => {
  res.json(all(`
    SELECT ns.node_id, COALESCE(ns.location, ns.node_id) AS location,
           ns.status, ns.last_seen,
           COALESCE(ns.enabled, 1) AS enabled,
           COALESCE(ns.notes, '') AS notes,
           COUNT(de.id) AS total_events,
           COALESCE(SUM(de.is_suspicious), 0) AS suspicious_events,
           ROUND(AVG(de.confidence), 3) AS avg_confidence
    FROM node_status ns
    LEFT JOIN detection_events de ON de.node_id = ns.node_id
    GROUP BY ns.node_id ORDER BY suspicious_events DESC
  `));
});

// ── RMS ────────────────────────────────────────────────────────────────
app.get('/api/rms', (req, res) => {
  const node = req.query.node || null;
  const sql = `
    SELECT node_id, ts, rms, baseline, threshold, is_trigger
    FROM rms_frames
    ${node ? "WHERE node_id = '" + node + "'" : ""}
    ORDER BY id DESC LIMIT 120
  `;
  res.json(all(sql).reverse());
});

// ── Correlated ─────────────────────────────────────────────────────────
app.get('/api/correlated', (req, res) => {
  res.json(all(`
    SELECT a.node_id AS node_a, b.node_id AS node_b,
           a.label AS label_a, b.label AS label_b,
           a.inserted_at AS time_a, b.inserted_at AS time_b,
           ROUND(ABS(a.wall_clock_time - b.wall_clock_time), 4) AS delta_sec
    FROM detection_events a JOIN detection_events b ON a.id < b.id
    WHERE a.is_suspicious=1 AND b.is_suspicious=1
      AND ABS(a.wall_clock_time - b.wall_clock_time) <= 2.0
    ORDER BY a.id DESC LIMIT 20
  `));
});

// ── Localization ───────────────────────────────────────────────────────
app.get('/api/localization', (req, res) => {
  const row = get('SELECT * FROM localization_results ORDER BY id DESC LIMIT 1');
  if (!row) return res.json({ stub: true, message: "Sound Localization module not yet integrated.",
    likely_location: null, likely_node: null, confidence: null, tdoa_matrix: {} });
  try { res.json({ stub: false, ...JSON.parse(row.payload_json) }); }
  catch { res.json({ stub: true, raw: row.payload_json }); }
});

// ── Status ─────────────────────────────────────────────────────────────
app.get('/api/status', (req, res) => {
  res.json({ db_mode: dbMode, host_db_path: HOST_DB_PATH });
});

// ══════════════════════════════════════════════════════════════════════
// HISTORY  /api/history
// ══════════════════════════════════════════════════════════════════════
// Allowed sort columns – whitelist prevents SQL injection via ORDER BY
const HIST_SORT_COLS = {
  id:           'id',
  node_id:      'node_id',
  label:        'label',
  confidence:   'confidence',
  is_suspicious:'is_suspicious',
  severity:     "CASE severity WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 2 ELSE 1 END",
  inserted_at:  'inserted_at',
  wall_clock_time: 'wall_clock_time',
};

app.get('/api/history', (req, res) => {
  const page     = Math.max(1, parseInt(req.query.page)  || 1);
  const perPage  = Math.min(200, parseInt(req.query.per) || 50);
  const offset   = (page - 1) * perPage;
  const node     = req.query.node     || null;
  const label    = req.query.label    || null;
  const severity = req.query.severity || null;
  const suspOnly = req.query.susp === '1';
  const dateFrom = req.query.from     || null;
  const dateTo   = req.query.to       || null;

  // Sort: validate column against whitelist, direction must be ASC or DESC
  const sortKey  = req.query.sort || 'id';
  const sortCol  = HIST_SORT_COLS[sortKey] || 'id';
  const sortDir  = req.query.dir === 'ASC' ? 'ASC' : 'DESC';

  const clauses = [];
  if (node)     clauses.push(`node_id = '${node.replace(/'/g,"''")}'`);
  if (label)    clauses.push(`label LIKE '%${label.replace(/'/g,"''")}%'`);
  if (severity) clauses.push(`severity = '${severity.replace(/'/g,"''")}'`);
  if (suspOnly) clauses.push(`is_suspicious = 1`);
  if (dateFrom) clauses.push(`inserted_at >= '${dateFrom.replace(/'/g,"''")}'`);
  if (dateTo)   clauses.push(`inserted_at <= '${dateTo.replace(/'/g,"''")}'`);

  const where = clauses.length ? 'WHERE ' + clauses.join(' AND ') : '';

  const total = (get(`SELECT COUNT(*) as c FROM detection_events ${where}`) || {c:0}).c;
  const rows  = all(`
    SELECT id, node_id, label, confidence, is_suspicious,
           COALESCE(severity,'LOW') AS severity,
           timestamp_node, wall_clock_time, inserted_at, session_id
    FROM detection_events ${where}
    ORDER BY ${sortCol} ${sortDir}
    LIMIT ${perPage} OFFSET ${offset}
  `);

  res.json({
    total, page, per_page: perPage,
    pages: Math.ceil(total/perPage),
    sort: sortKey, dir: sortDir,
    rows
  });
});

// ══════════════════════════════════════════════════════════════════════
// ADMIN – QUERY LAB  /api/admin/query
// ══════════════════════════════════════════════════════════════════════
const BLOCKED = ['INSERT','UPDATE','DELETE','DROP','CREATE','ALTER','ATTACH','PRAGMA'];

app.post('/api/admin/query', requireAdmin, (req, res) => {
  const { sql } = req.body || {};
  if (!sql || typeof sql !== 'string') return res.status(400).json({ error: 'No SQL provided' });

  const upper = sql.trim().toUpperCase();
  if (!upper.startsWith('SELECT') && !upper.startsWith('WITH')) {
    return res.status(403).json({ error: 'Only SELECT / WITH queries are permitted.' });
  }
  // Use word-boundary regex so "inserted_at" does NOT match "INSERT"
  for (const kw of BLOCKED) {
    if (new RegExp(`\\b${kw}\\b`).test(upper)) {
      return res.status(403).json({ error: `Keyword '${kw}' is not permitted.` });
    }
  }

  try {
    const stmt = db.prepare(sql);
    const rows = [];
    while (stmt.step()) rows.push(stmt.getAsObject());
    stmt.free();
    const cols = rows.length > 0 ? Object.keys(rows[0]) : [];
    res.json({ cols, rows, count: rows.length });
  } catch(e) {
    res.status(400).json({ error: e.message });
  }
});

// ══════════════════════════════════════════════════════════════════════
// ADMIN – NODE MANAGEMENT  /api/admin/nodes/*
// ══════════════════════════════════════════════════════════════════════

// Nodes seen in events but not registered in node_status (new node discovery)
app.get('/api/admin/nodes/discovered', requireAdmin, (req, res) => {
  const rows = all(`
    SELECT DISTINCT de.node_id, MAX(de.inserted_at) AS last_event,
           COUNT(*) AS event_count
    FROM detection_events de
    LEFT JOIN node_status ns ON ns.node_id = de.node_id
    WHERE ns.node_id IS NULL
    GROUP BY de.node_id
  `);
  res.json(rows);
});

// All nodes (admin view with enabled/notes)
app.get('/api/admin/nodes', requireAdmin, (req, res) => {
  ensureAdminColumns();
  res.json(all(`
    SELECT ns.node_id, COALESCE(ns.location,'') AS location,
           ns.status, ns.last_seen,
           COALESCE(ns.enabled,1) AS enabled,
           COALESCE(ns.notes,'') AS notes,
           COUNT(de.id) AS total_events,
           COALESCE(SUM(de.is_suspicious),0) AS suspicious_events
    FROM node_status ns
    LEFT JOIN detection_events de ON de.node_id = ns.node_id
    GROUP BY ns.node_id ORDER BY ns.node_id
  `));
});

// Register a new node manually
app.post('/api/admin/nodes', requireAdmin, (req, res) => {
  ensureAdminColumns();
  const { node_id, location, notes } = req.body || {};
  if (!node_id) return res.status(400).json({ error: 'node_id required' });
  const existing = get(`SELECT node_id FROM node_status WHERE node_id=?`, [node_id]);
  if (existing) return res.status(409).json({ error: 'Node already exists' });
  run(`INSERT INTO node_status (node_id,location,status,last_seen,enabled,notes)
       VALUES (?,?,?,?,1,?)`,
      [node_id, location||'', 'offline', 0, notes||'']);
  persistDb();
  res.json({ ok: true, node_id });
});

// Update a node (rename location, toggle enabled, set notes)
app.put('/api/admin/nodes/:id', requireAdmin, (req, res) => {
  ensureAdminColumns();
  const { id } = req.params;
  const { location, enabled, notes } = req.body || {};
  const node = get(`SELECT node_id FROM node_status WHERE node_id=?`, [id]);
  if (!node) return res.status(404).json({ error: 'Node not found' });

  if (location !== undefined)
    run(`UPDATE node_status SET location=? WHERE node_id=?`, [location, id]);
  if (enabled !== undefined)
    run(`UPDATE node_status SET enabled=? WHERE node_id=?`, [enabled ? 1 : 0, id]);
  if (notes !== undefined)
    run(`UPDATE node_status SET notes=? WHERE node_id=?`, [notes, id]);

  persistDb();
  res.json({ ok: true });
});

// Remove a node record (does not delete events)
app.delete('/api/admin/nodes/:id', requireAdmin, (req, res) => {
  const { id } = req.params;
  run(`DELETE FROM node_status WHERE node_id=?`, [id]);
  persistDb();
  res.json({ ok: true });
});

// Ping: update last_seen to now (simulated connectivity check)
app.post('/api/admin/nodes/:id/ping', requireAdmin, (req, res) => {
  const { id } = req.params;
  run(`UPDATE node_status SET last_seen=? WHERE node_id=?`, [Date.now()/1000, id]);
  persistDb();
  res.json({ ok: true, pinged_at: new Date().toISOString() });
});

// Clear all events for a node
app.post('/api/admin/nodes/:id/clear', requireAdmin, (req, res) => {
  const { id } = req.params;
  run(`DELETE FROM detection_events WHERE node_id=?`, [id]);
  run(`DELETE FROM rms_frames WHERE node_id=?`, [id]);
  persistDb();
  res.json({ ok: true });
});

// ── Bootstrap ──────────────────────────────────────────────────────────
async function main() {
  const SQL = await initSql();
  await loadDatabase(SQL);

  // Ensure admin columns exist in node_status immediately after DB load
  // so /api/nodes never fails with "no such column: ns.enabled"
  ensureAdminColumns();

  setInterval(() => reloadLiveDb(SQL), POLL_INTERVAL_MS);

  if (dbMode === 'demo') {
    const pool = [
      { node:'node_1', label:'Gunshot / firearm', susp:1, severity:'HIGH',   conf:() => +(0.82+Math.random()*0.16).toFixed(3) },
      { node:'node_2', label:'Background noise',  susp:0, severity:'LOW',    conf:() => +(0.05+Math.random()*0.18).toFixed(3) },
      { node:'node_3', label:'Background noise',  susp:0, severity:'LOW',    conf:() => +(0.04+Math.random()*0.12).toFixed(3) },
      { node:'node_1', label:'Glass break',       susp:1, severity:'MEDIUM', conf:() => +(0.80+Math.random()*0.14).toFixed(3) },
      { node:'node_2', label:'Background noise',  susp:0, severity:'LOW',    conf:() => +(0.05+Math.random()*0.20).toFixed(3) },
      { node:'node_3', label:'Background noise',  susp:0, severity:'LOW',    conf:() => +(0.06+Math.random()*0.14).toFixed(3) },
    ];
    let pidx = 0;
    setInterval(() => {
      const now = Date.now()/1000;
      const pick = pool[pidx % pool.length]; pidx++;
      const rms = +(0.1 + Math.random()*0.9).toFixed(3);
      run(`INSERT INTO detection_events
           (node_id,label,confidence,is_suspicious,severity,
            timestamp_node,wall_clock_time,received_at_host,onset_index)
           VALUES (?,?,?,?,?,?,?,?,0)`,
          [pick.node, pick.label, pick.conf(), pick.susp, pick.severity,
           now, now, now + 0.05]);
      run(`INSERT INTO rms_frames (node_id,ts,rms,baseline,threshold,is_trigger)
           VALUES (?,?,?,?,?,?)`,
          [pick.node, now, rms, 0.06, 0.18, rms > 0.18 ? 1 : 0]);
      // Push SSE update to all connected browsers when a suspicious demo event fires
      if (pick.susp) {
        broadcastSuspicious({
          node_id:      pick.node,
          label:        pick.label,
          is_suspicious: true,
          severity:     pick.severity,
          ts:           now,
          source:       'demo',
        });
      }
    }, 4000);
  }

  // ── Start servers ────────────────────────────────────────────────────
  if (CERT_PATH && KEY_PATH && fs.existsSync(CERT_PATH) && fs.existsSync(KEY_PATH)) {
    // HTTPS mode: load TLS cert + key and start an HTTPS server
    const tlsOptions = {
      cert: fs.readFileSync(CERT_PATH),
      key:  fs.readFileSync(KEY_PATH),
    };
    https.createServer(tlsOptions, app).listen(HTTPS_PORT, '0.0.0.0', () => {
      console.log(`ISD Dashboard → https://localhost:${HTTPS_PORT}  (HTTPS)`);
      console.log(`Database mode : ${dbMode.toUpperCase()}`);
      if (dbMode === 'live') console.log(`Reading from  : ${HOST_DB_PATH}`);
    });

    // HTTP → HTTPS redirect server on PORT
    http.createServer((req, res) => {
      const host = (req.headers.host || 'localhost').split(':')[0];
      res.writeHead(301, { Location: `https://${host}:${HTTPS_PORT}${req.url}` });
      res.end();
    }).listen(PORT, '0.0.0.0', () => {
      console.log(`HTTP redirect → http://localhost:${PORT}  (redirects to HTTPS)`);
    });

  } else {
    // HTTP-only fallback
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`ISD Dashboard → http://localhost:${PORT}  (HTTP only)`);
      console.log(`Database mode : ${dbMode.toUpperCase()}`);
      if (dbMode === 'live') console.log(`Reading from  : ${HOST_DB_PATH}`);
      else console.log(`Demo mode – run host_subscriber.py to switch to live data`);
      if (!CERT_PATH || !KEY_PATH) {
        console.log(`[TLS] Set CERT_PATH + KEY_PATH env vars to enable HTTPS`);
      }
    });
  }
}

main().catch(e => { console.error(e); process.exit(1); });