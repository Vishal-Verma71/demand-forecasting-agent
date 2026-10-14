"""Lightweight SQLite persistence for inventory, PO drafts, alerts."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory (
    store TEXT NOT NULL,
    dept TEXT NOT NULL,
    sku TEXT,
    on_hand REAL NOT NULL,
    safety_stock REAL,
    lead_time_days INTEGER,
    supplier TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (store, dept)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    store TEXT, dept TEXT, level TEXT, message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS purchase_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    store TEXT, dept TEXT, supplier TEXT,
    qty REAL, lead_time_days INTEGER, status TEXT DEFAULT 'DRAFT',
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as c:
        c.executescript(SCHEMA)
        c.commit()


@contextmanager
def conn(path: Path = DB_PATH):
    init_db(path)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def upsert_inventory(rows):
    with conn() as c:
        c.executemany(
            """INSERT INTO inventory(store,dept,sku,on_hand,safety_stock,lead_time_days,supplier)
               VALUES(:store,:dept,:sku,:on_hand,:safety_stock,:lead_time_days,:supplier)
               ON CONFLICT(store,dept) DO UPDATE SET
                 on_hand=excluded.on_hand, safety_stock=excluded.safety_stock,
                 lead_time_days=excluded.lead_time_days, supplier=excluded.supplier,
                 updated_at=datetime('now')""",
            rows,
        )


def get_inventory(store=None, dept=None):
    with conn() as c:
        q = "SELECT * FROM inventory WHERE 1=1"
        p = []
        if store:
            q += " AND store=?"; p.append(store)
        if dept:
            q += " AND dept=?"; p.append(dept)
        return [dict(r) for r in c.execute(q, p).fetchall()]


def add_alert(store, dept, level, message):
    with conn() as c:
        c.execute("INSERT INTO alerts(store,dept,level,message) VALUES(?,?,?,?)",
                  (store, dept, level, message))


def get_alerts(limit=50):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def add_po(store, dept, supplier, qty, lead_time_days):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO purchase_orders(store,dept,supplier,qty,lead_time_days) VALUES(?,?,?,?,?)",
            (store, dept, supplier, qty, lead_time_days))
        return cur.lastrowid


def get_pos(limit=50):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM purchase_orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
