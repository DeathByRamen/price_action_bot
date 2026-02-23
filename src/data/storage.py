"""
SQLite storage backend for OHLCV candles, predictions, accuracy logs,
and sample weights.

Tables
------
ohlcv                  : (symbol, ts, open, high, low, close, volume)
predictions            : predictions + outcome columns for scoring
accuracy_log           : daily aggregate accuracy metrics
sample_weights         : per-symbol training weights from the adaptive tuner
order_book_snapshots   : periodic order book depth snapshots for future feature extraction
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ohlcv.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    interval   TEXT    NOT NULL DEFAULT '60',
    open       REAL    NOT NULL,
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    volume     REAL    NOT NULL DEFAULT 0,
    UNIQUE(symbol, ts, interval)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_ts_int ON ohlcv(symbol, ts, interval);

CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol           TEXT    NOT NULL,
    ts               TEXT    NOT NULL,
    interval         TEXT    NOT NULL DEFAULT '60',
    direction        TEXT    NOT NULL,
    prob_up          REAL,
    prob_flat        REAL,
    prob_down        REAL,
    magnitude        REAL,
    signal_score     REAL,
    actual_direction TEXT,
    actual_magnitude REAL,
    was_correct      INTEGER,
    scored_at        TEXT,
    created_at       TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts, interval)
);

CREATE INDEX IF NOT EXISTS idx_pred_sym_ts_int ON predictions(symbol, ts, interval);
CREATE INDEX IF NOT EXISTS idx_pred_unscored ON predictions(scored_at) WHERE scored_at IS NULL;

CREATE TABLE IF NOT EXISTS accuracy_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date            TEXT    NOT NULL,
    total_preds         INTEGER,
    direction_accuracy  REAL,
    magnitude_mae       REAL,
    up_precision        REAL,
    up_recall           REAL,
    down_precision      REAL,
    down_recall         REAL,
    flat_precision      REAL,
    flat_recall         REAL,
    flat_threshold_used REAL,
    created_at          TEXT    DEFAULT (datetime('now')),
    UNIQUE(run_date)
);

CREATE TABLE IF NOT EXISTS sample_weights (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    weight     REAL    NOT NULL DEFAULT 1.0,
    error_rate REAL,
    n_preds    INTEGER,
    updated_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol)
);

CREATE TABLE IF NOT EXISTS feature_importance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date      TEXT    NOT NULL,
    interval      TEXT    NOT NULL DEFAULT '60',
    feature_name  TEXT    NOT NULL,
    importance    REAL    NOT NULL DEFAULT 0.0,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE(run_date, interval, feature_name)
);

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    bid_prices TEXT    NOT NULL,
    bid_vols   TEXT    NOT NULL,
    ask_prices TEXT    NOT NULL,
    ask_vols   TEXT    NOT NULL,
    spread     REAL,
    mid_price  REAL,
    imbalance  REAL,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_ob_sym_ts ON order_book_snapshots(symbol, ts);

CREATE TABLE IF NOT EXISTS funding_rate_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    ts              TEXT    NOT NULL,
    funding_rate    REAL    NOT NULL,
    mark_price      REAL,
    last_price      REAL,
    next_funding_ts TEXT,
    funding_interval_hours INTEGER DEFAULT 8,
    created_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_fr_sym_ts ON funding_rate_snapshots(symbol, ts);

CREATE TABLE IF NOT EXISTS coinalyze_oi (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    oi_open    REAL,
    oi_high    REAL,
    oi_low     REAL,
    oi_close   REAL,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_ca_oi_sym_ts ON coinalyze_oi(symbol, ts);

CREATE TABLE IF NOT EXISTS coinalyze_liquidations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    long_vol   REAL    NOT NULL DEFAULT 0,
    short_vol  REAL    NOT NULL DEFAULT 0,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_ca_liq_sym_ts ON coinalyze_liquidations(symbol, ts);

CREATE TABLE IF NOT EXISTS coinalyze_long_short (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    ratio      REAL,
    long_pct   REAL,
    short_pct  REAL,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_ca_ls_sym_ts ON coinalyze_long_short(symbol, ts);

CREATE TABLE IF NOT EXISTS fear_greed_index (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT    NOT NULL,
    value      REAL    NOT NULL,
    label      TEXT,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(ts)
);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    positive   INTEGER NOT NULL DEFAULT 0,
    negative   INTEGER NOT NULL DEFAULT 0,
    neutral    INTEGER NOT NULL DEFAULT 0,
    total      INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE TABLE IF NOT EXISTS binance_funding_rate (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    ts            TEXT    NOT NULL,
    funding_rate  REAL    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE TABLE IF NOT EXISTS binance_oi (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    oi_value   REAL    NOT NULL,
    created_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);

CREATE TABLE IF NOT EXISTS model_sharpes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT    NOT NULL,
    interval   TEXT    NOT NULL DEFAULT '60',
    sharpe     REAL    NOT NULL DEFAULT 0.0,
    updated_at TEXT    DEFAULT (datetime('now')),
    UNIQUE(model_name, interval)
);

CREATE TABLE IF NOT EXISTS feature_retirement (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name    TEXT    NOT NULL,
    interval        TEXT    NOT NULL DEFAULT '60',
    below_count     INTEGER NOT NULL DEFAULT 0,
    is_retired      INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(feature_name, interval)
);
"""

MIGRATION_SQL = """
-- Add outcome columns to predictions if they don't exist yet
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we catch errors
"""


class Storage:
    """Async SQLite storage for OHLCV data and prediction logs."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = os.path.abspath(db_path or DEFAULT_DB_PATH)
        self._db: Optional[aiosqlite.Connection] = None

    async def open(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(SCHEMA_SQL)
        await self._migrate()
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.commit()
        logger.info("Database opened at %s", self._db_path)

    async def _migrate(self) -> None:
        """Add columns that may not exist in older databases."""
        assert self._db is not None

        # Check if ohlcv table needs interval column + constraint migration
        cursor = await self._db.execute("PRAGMA table_info(ohlcv)")
        ohlcv_cols = [row[1] for row in await cursor.fetchall()]

        if "interval" not in ohlcv_cols:
            logger.info("Migrating ohlcv table to add interval column...")
            await self._db.executescript("""
                ALTER TABLE ohlcv RENAME TO _ohlcv_old;

                CREATE TABLE ohlcv (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol   TEXT NOT NULL,
                    ts       TEXT NOT NULL,
                    interval TEXT NOT NULL DEFAULT '60',
                    open     REAL NOT NULL,
                    high     REAL NOT NULL,
                    low      REAL NOT NULL,
                    close    REAL NOT NULL,
                    volume   REAL NOT NULL DEFAULT 0,
                    UNIQUE(symbol, ts, interval)
                );

                INSERT OR IGNORE INTO ohlcv
                    (symbol, ts, interval, open, high, low, close, volume)
                SELECT symbol, ts, '60', open, high, low, close, volume
                FROM _ohlcv_old;

                DROP TABLE _ohlcv_old;

                CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_ts_int
                    ON ohlcv(symbol, ts, interval);
            """)
            logger.info("ohlcv migration complete")

        # Check if predictions table needs interval column
        cursor = await self._db.execute("PRAGMA table_info(predictions)")
        pred_cols = [row[1] for row in await cursor.fetchall()]

        if "interval" not in pred_cols:
            logger.info("Migrating predictions table to add interval column...")
            await self._db.executescript("""
                ALTER TABLE predictions RENAME TO _predictions_old;

                CREATE TABLE predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol           TEXT NOT NULL,
                    ts               TEXT NOT NULL,
                    interval         TEXT NOT NULL DEFAULT '60',
                    direction        TEXT NOT NULL,
                    prob_up          REAL,
                    prob_flat        REAL,
                    prob_down        REAL,
                    magnitude        REAL,
                    signal_score     REAL,
                    actual_direction TEXT,
                    actual_magnitude REAL,
                    was_correct      INTEGER,
                    scored_at        TEXT,
                    created_at       TEXT DEFAULT (datetime('now')),
                    UNIQUE(symbol, ts, interval)
                );

                INSERT OR IGNORE INTO predictions
                    (symbol, ts, interval, direction, prob_up, prob_flat,
                     prob_down, magnitude, signal_score, actual_direction,
                     actual_magnitude, was_correct, scored_at, created_at)
                SELECT symbol, ts, '60', direction, prob_up, prob_flat,
                       prob_down, magnitude, signal_score, actual_direction,
                       actual_magnitude, was_correct, scored_at, created_at
                FROM _predictions_old;

                DROP TABLE _predictions_old;

                CREATE INDEX IF NOT EXISTS idx_pred_sym_ts_int
                    ON predictions(symbol, ts, interval);
                CREATE INDEX IF NOT EXISTS idx_pred_unscored
                    ON predictions(scored_at) WHERE scored_at IS NULL;
            """)
            logger.info("predictions migration complete")

        # Legacy column migrations for very old databases
        for table, col, col_type in [
            ("predictions", "actual_direction", "TEXT"),
            ("predictions", "actual_magnitude", "REAL"),
            ("predictions", "was_correct", "INTEGER"),
            ("predictions", "scored_at", "TEXT"),
        ]:
            try:
                await self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                )
            except Exception:
                pass
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> "Storage":
        await self.open()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # OHLCV operations
    # ------------------------------------------------------------------
    async def insert_candles(self, rows: List[Tuple], interval: str = "60") -> int:
        """
        Bulk-insert candles.
        Each row is (symbol, ts, open, high, low, close, volume).
        The interval is appended automatically.
        Duplicates on (symbol, ts, interval) are silently ignored.
        Returns the number of rows actually inserted.
        """
        assert self._db is not None
        sql = """
            INSERT OR IGNORE INTO ohlcv (symbol, ts, interval, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows_with_interval = [
            (r[0], r[1], interval, r[2], r[3], r[4], r[5], r[6]) for r in rows
        ]
        cursor = await self._db.executemany(sql, rows_with_interval)
        await self._db.commit()
        return cursor.rowcount

    async def get_candles(
        self,
        symbol: str,
        limit: int = 500,
        before_ts: Optional[str] = None,
        interval: str = "60",
    ) -> pd.DataFrame:
        """Return candles for *symbol* at *interval* as a DataFrame, ordered by ts ascending."""
        assert self._db is not None
        if before_ts:
            sql = """
                SELECT symbol, ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND interval = ? AND ts < ?
                ORDER BY ts ASC
                LIMIT ?
            """
            params: Tuple = (symbol, interval, before_ts, limit)
        else:
            sql = """
                SELECT symbol, ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND interval = ?
                ORDER BY ts DESC
                LIMIT ?
            """
            params = (symbol, interval, limit)

        rows = await self._db.execute_fetchall(sql, params)
        df = pd.DataFrame(
            rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"]
        )
        if not before_ts:
            df = df.iloc[::-1].reset_index(drop=True)
        return df

    async def get_latest_ts(self, symbol: str, interval: str = "60") -> Optional[str]:
        """Return the most recent candle timestamp for *symbol* at *interval*, or None."""
        assert self._db is not None
        row = await self._db.execute_fetchall(
            "SELECT MAX(ts) FROM ohlcv WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        )
        if row and row[0][0]:
            return row[0][0]
        return None

    async def get_all_symbols(self) -> List[str]:
        """Return distinct symbols stored in ohlcv table."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
        )
        return [r[0] for r in rows]

    async def candle_count(self, symbol: Optional[str] = None) -> int:
        """Return number of candle rows, optionally filtered by symbol."""
        assert self._db is not None
        if symbol:
            rows = await self._db.execute_fetchall(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol = ?", (symbol,)
            )
        else:
            rows = await self._db.execute_fetchall("SELECT COUNT(*) FROM ohlcv")
        return rows[0][0]

    async def candle_count_for_symbol(
        self, symbol: str, interval: str = "60"
    ) -> int:
        """Return number of candle rows for a symbol at a specific interval."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        )
        return rows[0][0]

    # ------------------------------------------------------------------
    # Prediction operations
    # ------------------------------------------------------------------
    async def insert_predictions(self, rows: List[Tuple], interval: str = "60") -> int:
        """
        Bulk-insert predictions.
        Each row: (symbol, ts, direction, prob_up, prob_flat, prob_down, magnitude, signal_score)
        The interval is appended automatically.
        """
        assert self._db is not None
        sql = """
            INSERT OR REPLACE INTO predictions
                (symbol, ts, interval, direction, prob_up, prob_flat, prob_down, magnitude, signal_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows_with_interval = [
            (r[0], r[1], interval, r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows
        ]
        cursor = await self._db.executemany(sql, rows_with_interval)
        await self._db.commit()
        return cursor.rowcount

    async def get_predictions(
        self, limit: int = 50, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Return recent predictions as a DataFrame."""
        assert self._db is not None
        if symbol:
            sql = """
                SELECT symbol, ts, direction, prob_up, prob_flat, prob_down,
                       magnitude, signal_score, actual_direction, actual_magnitude,
                       was_correct, scored_at, created_at
                FROM predictions
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            rows = await self._db.execute_fetchall(sql, (symbol, limit))
        else:
            sql = """
                SELECT symbol, ts, direction, prob_up, prob_flat, prob_down,
                       magnitude, signal_score, actual_direction, actual_magnitude,
                       was_correct, scored_at, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT ?
            """
            rows = await self._db.execute_fetchall(sql, (limit,))

        return pd.DataFrame(
            rows,
            columns=[
                "symbol", "ts", "direction", "prob_up", "prob_flat", "prob_down",
                "magnitude", "signal_score", "actual_direction", "actual_magnitude",
                "was_correct", "scored_at", "created_at",
            ],
        )

    # ------------------------------------------------------------------
    # Scoring operations
    # ------------------------------------------------------------------
    async def get_unscored_predictions(self) -> pd.DataFrame:
        """Return predictions that have not yet been scored against actuals."""
        assert self._db is not None
        sql = """
            SELECT id, symbol, ts, interval, direction, prob_up, prob_flat,
                   prob_down, magnitude, signal_score, created_at
            FROM predictions
            WHERE scored_at IS NULL AND ts != ''
            ORDER BY ts ASC
        """
        rows = await self._db.execute_fetchall(sql)
        return pd.DataFrame(
            rows,
            columns=[
                "id", "symbol", "ts", "interval", "direction", "prob_up",
                "prob_flat", "prob_down", "magnitude", "signal_score",
                "created_at",
            ],
        )

    async def update_prediction_outcome(
        self,
        pred_id: int,
        actual_direction: str,
        actual_magnitude: float,
        was_correct: bool,
        scored_at: str,
    ) -> None:
        """Update a single prediction row with its actual outcome."""
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE predictions
            SET actual_direction = ?, actual_magnitude = ?,
                was_correct = ?, scored_at = ?
            WHERE id = ?
            """,
            (actual_direction, actual_magnitude, int(was_correct), scored_at, pred_id),
        )

    async def commit(self) -> None:
        """Explicit commit (useful after batch updates)."""
        assert self._db is not None
        await self._db.commit()

    async def get_scored_predictions(
        self,
        days: int = 7,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return scored predictions from the last N days.

        Parameters
        ----------
        interval : str | None
            If set, only return predictions for this candle interval.
            If None, returns all intervals (backward-compatible).
        """
        assert self._db is not None

        conditions = ["scored_at IS NOT NULL", "scored_at >= datetime('now', ?)"]
        params: list = [f"-{days} days"]

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if interval:
            conditions.append("interval = ?")
            params.append(interval)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT symbol, ts, interval, direction, prob_up, prob_flat, prob_down,
                   magnitude, actual_direction, actual_magnitude, was_correct,
                   scored_at
            FROM predictions
            WHERE {where}
            ORDER BY ts DESC
        """
        rows = await self._db.execute_fetchall(sql, tuple(params))

        return pd.DataFrame(
            rows,
            columns=[
                "symbol", "ts", "interval", "direction", "prob_up", "prob_flat",
                "prob_down", "magnitude", "actual_direction", "actual_magnitude",
                "was_correct", "scored_at",
            ],
        )

    async def get_close_at_ts(
        self, symbol: str, ts: str, interval: str = "60"
    ) -> Optional[float]:
        """Return the close price for a symbol at a specific timestamp and interval."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT close FROM ohlcv WHERE symbol = ? AND ts = ? AND interval = ? LIMIT 1",
            (symbol, ts, interval),
        )
        if rows:
            return float(rows[0][0])
        return None

    async def get_next_candle_close(
        self, symbol: str, ts: str, interval: str = "60"
    ) -> Optional[Tuple[str, float]]:
        """Return (ts, close) for the first candle strictly after *ts* at the given *interval*."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            """
            SELECT ts, close FROM ohlcv
            WHERE symbol = ? AND interval = ? AND ts > ?
            ORDER BY ts ASC
            LIMIT 1
            """,
            (symbol, interval, ts),
        )
        if rows:
            return (rows[0][0], float(rows[0][1]))
        return None

    # ------------------------------------------------------------------
    # Accuracy log operations
    # ------------------------------------------------------------------
    async def insert_accuracy_log(self, row: Tuple) -> None:
        """
        Insert or replace a daily accuracy summary.
        Row: (run_date, total_preds, direction_accuracy, magnitude_mae,
              up_precision, up_recall, down_precision, down_recall,
              flat_precision, flat_recall, flat_threshold_used)
        """
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO accuracy_log
                (run_date, total_preds, direction_accuracy, magnitude_mae,
                 up_precision, up_recall, down_precision, down_recall,
                 flat_precision, flat_recall, flat_threshold_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        await self._db.commit()

    async def get_recent_interval_accuracy(
        self, interval: str, days: int = 7
    ) -> Optional[float]:
        """
        Return the average direction accuracy for predictions at the given
        interval over the last *days*. Returns None if no scored data.
        """
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            """
            SELECT AVG(CASE WHEN was_correct = 1 THEN 1.0 ELSE 0.0 END) as acc
            FROM predictions
            WHERE scored_at IS NOT NULL
              AND interval = ?
              AND scored_at >= datetime('now', ?)
            """,
            (interval, f"-{days} days"),
        )
        if rows and rows[0][0] is not None:
            return float(rows[0][0])
        return None

    async def get_accuracy_history(self, days: int = 30) -> pd.DataFrame:
        """Return accuracy log entries from the last N days."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            """
            SELECT run_date, total_preds, direction_accuracy, magnitude_mae,
                   up_precision, up_recall, down_precision, down_recall,
                   flat_precision, flat_recall, flat_threshold_used
            FROM accuracy_log
            WHERE run_date >= date('now', ?)
            ORDER BY run_date DESC
            """,
            (f"-{days} days",),
        )
        return pd.DataFrame(
            rows,
            columns=[
                "run_date", "total_preds", "direction_accuracy", "magnitude_mae",
                "up_precision", "up_recall", "down_precision", "down_recall",
                "flat_precision", "flat_recall", "flat_threshold_used",
            ],
        )

    # ------------------------------------------------------------------
    # Sample weights operations
    # ------------------------------------------------------------------
    async def upsert_sample_weights(self, rows: List[Tuple]) -> None:
        """
        Insert or update per-symbol sample weights.
        Each row: (symbol, weight, error_rate, n_preds)
        """
        assert self._db is not None
        await self._db.executemany(
            """
            INSERT OR REPLACE INTO sample_weights (symbol, weight, error_rate, n_preds)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        await self._db.commit()

    async def get_sample_weights(self) -> Dict[str, float]:
        """Return {symbol: weight} dict from the sample_weights table."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT symbol, weight FROM sample_weights"
        )
        return {r[0]: float(r[1]) for r in rows}

    # ------------------------------------------------------------------
    # Feature importance operations
    # ------------------------------------------------------------------
    async def insert_feature_importance(
        self, run_date: str, interval: str, importances: Dict[str, float]
    ) -> None:
        """
        Insert or replace feature importance scores for a given run date + interval.
        importances: {feature_name: importance_score}
        """
        assert self._db is not None
        rows = [
            (run_date, interval, name, score)
            for name, score in importances.items()
        ]
        await self._db.executemany(
            """
            INSERT OR REPLACE INTO feature_importance
                (run_date, interval, feature_name, importance)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        await self._db.commit()

    async def get_low_importance_features(
        self, interval: str = "60", last_n_runs: int = 3, threshold: float = 0.001
    ) -> List[Tuple[str, float]]:
        """
        Return features that scored below *threshold* for the last *last_n_runs*
        consecutive daily runs at the given interval.

        Returns list of (feature_name, avg_importance).
        """
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            """
            SELECT feature_name, AVG(importance) as avg_imp, COUNT(*) as n_runs
            FROM feature_importance
            WHERE interval = ?
              AND run_date IN (
                  SELECT DISTINCT run_date FROM feature_importance
                  WHERE interval = ?
                  ORDER BY run_date DESC
                  LIMIT ?
              )
            GROUP BY feature_name
            HAVING n_runs >= ? AND avg_imp < ?
            ORDER BY avg_imp ASC
            """,
            (interval, interval, last_n_runs, last_n_runs, threshold),
        )
        return [(r[0], float(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # Order book snapshot operations
    # ------------------------------------------------------------------
    async def insert_order_book_snapshots(self, rows: List[Tuple]) -> int:
        """
        Bulk-insert order book snapshots.
        Each row: (symbol, ts, bid_prices, bid_vols, ask_prices, ask_vols,
                   spread, mid_price, imbalance)
        Duplicates on (symbol, ts) are silently ignored.
        Returns the number of rows inserted.
        """
        assert self._db is not None
        sql = """
            INSERT OR IGNORE INTO order_book_snapshots
                (symbol, ts, bid_prices, bid_vols, ask_prices, ask_vols,
                 spread, mid_price, imbalance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_order_book_snapshots(
        self,
        symbol: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Return order book snapshots for a symbol within a time range."""
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]

        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("ts <= ?")
            params.append(end_ts)

        where = " AND ".join(conditions)
        params.append(limit)
        sql = f"""
            SELECT symbol, ts, bid_prices, bid_vols, ask_prices, ask_vols,
                   spread, mid_price, imbalance, created_at
            FROM order_book_snapshots
            WHERE {where}
            ORDER BY ts DESC
            LIMIT ?
        """
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(
            rows,
            columns=[
                "symbol", "ts", "bid_prices", "bid_vols", "ask_prices",
                "ask_vols", "spread", "mid_price", "imbalance", "created_at",
            ],
        )

    async def order_book_snapshot_count(
        self, symbol: Optional[str] = None
    ) -> int:
        """Return number of order book snapshots, optionally filtered by symbol."""
        assert self._db is not None
        if symbol:
            rows = await self._db.execute_fetchall(
                "SELECT COUNT(*) FROM order_book_snapshots WHERE symbol = ?",
                (symbol,),
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT COUNT(*) FROM order_book_snapshots"
            )
        return rows[0][0]

    # ------------------------------------------------------------------
    # Funding rate snapshot operations
    # ------------------------------------------------------------------
    async def insert_funding_rate_snapshots(self, rows: List[Tuple]) -> int:
        """
        Bulk-insert funding rate snapshots.
        Each row: (symbol, ts, funding_rate, mark_price, last_price,
                   next_funding_ts, funding_interval_hours)
        Duplicates on (symbol, ts) are silently ignored.
        Returns the number of rows inserted.
        """
        assert self._db is not None
        sql = """
            INSERT OR IGNORE INTO funding_rate_snapshots
                (symbol, ts, funding_rate, mark_price, last_price,
                 next_funding_ts, funding_interval_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_funding_rate_snapshots(
        self,
        symbol: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Return funding rate snapshots for a symbol within a time range."""
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]

        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("ts <= ?")
            params.append(end_ts)

        where = " AND ".join(conditions)
        params.append(limit)
        sql = f"""
            SELECT symbol, ts, funding_rate, mark_price, last_price,
                   next_funding_ts, funding_interval_hours, created_at
            FROM funding_rate_snapshots
            WHERE {where}
            ORDER BY ts DESC
            LIMIT ?
        """
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(
            rows,
            columns=[
                "symbol", "ts", "funding_rate", "mark_price", "last_price",
                "next_funding_ts", "funding_interval_hours", "created_at",
            ],
        )

    # ------------------------------------------------------------------
    # Coinalyze: open interest
    # ------------------------------------------------------------------
    async def insert_coinalyze_oi(self, rows: List[Tuple]) -> int:
        """
        Bulk-insert open interest OHLC rows.
        Each row: (symbol, ts, oi_open, oi_high, oi_low, oi_close)
        """
        assert self._db is not None
        sql = """
            INSERT OR IGNORE INTO coinalyze_oi
                (symbol, ts, oi_open, oi_high, oi_low, oi_close)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_coinalyze_oi(
        self,
        symbol: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("ts <= ?")
            params.append(end_ts)
        where = " AND ".join(conditions)
        params.append(limit)
        sql = f"""
            SELECT symbol, ts, oi_open, oi_high, oi_low, oi_close
            FROM coinalyze_oi WHERE {where} ORDER BY ts ASC LIMIT ?
        """
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(
            rows,
            columns=["symbol", "ts", "oi_open", "oi_high", "oi_low", "oi_close"],
        )

    # ------------------------------------------------------------------
    # Coinalyze: liquidations
    # ------------------------------------------------------------------
    async def insert_coinalyze_liquidations(self, rows: List[Tuple]) -> int:
        """
        Bulk-insert liquidation rows.
        Each row: (symbol, ts, long_vol, short_vol)
        """
        assert self._db is not None
        sql = """
            INSERT OR IGNORE INTO coinalyze_liquidations
                (symbol, ts, long_vol, short_vol)
            VALUES (?, ?, ?, ?)
        """
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_coinalyze_liquidations(
        self,
        symbol: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("ts <= ?")
            params.append(end_ts)
        where = " AND ".join(conditions)
        params.append(limit)
        sql = f"""
            SELECT symbol, ts, long_vol, short_vol
            FROM coinalyze_liquidations WHERE {where} ORDER BY ts ASC LIMIT ?
        """
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(
            rows, columns=["symbol", "ts", "long_vol", "short_vol"],
        )

    # ------------------------------------------------------------------
    # Coinalyze: long/short ratio
    # ------------------------------------------------------------------
    async def insert_coinalyze_long_short(self, rows: List[Tuple]) -> int:
        """
        Bulk-insert long/short ratio rows.
        Each row: (symbol, ts, ratio, long_pct, short_pct)
        """
        assert self._db is not None
        sql = """
            INSERT OR IGNORE INTO coinalyze_long_short
                (symbol, ts, ratio, long_pct, short_pct)
            VALUES (?, ?, ?, ?, ?)
        """
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_coinalyze_long_short(
        self,
        symbol: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("ts <= ?")
            params.append(end_ts)
        where = " AND ".join(conditions)
        params.append(limit)
        sql = f"""
            SELECT symbol, ts, ratio, long_pct, short_pct
            FROM coinalyze_long_short WHERE {where} ORDER BY ts ASC LIMIT ?
        """
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(
            rows, columns=["symbol", "ts", "ratio", "long_pct", "short_pct"],
        )

    # ------------------------------------------------------------------
    # Fear & Greed Index
    # ------------------------------------------------------------------
    async def insert_fear_greed(self, rows: List[Tuple]) -> int:
        assert self._db is not None
        sql = "INSERT OR IGNORE INTO fear_greed_index (ts, value, label) VALUES (?, ?, ?)"
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_fear_greed(
        self, start_ts: Optional[str] = None, limit: int = 500
    ) -> pd.DataFrame:
        assert self._db is not None
        if start_ts:
            sql = "SELECT ts, value, label FROM fear_greed_index WHERE ts >= ? ORDER BY ts ASC LIMIT ?"
            rows = await self._db.execute_fetchall(sql, (start_ts, limit))
        else:
            sql = "SELECT ts, value, label FROM fear_greed_index ORDER BY ts DESC LIMIT ?"
            rows = await self._db.execute_fetchall(sql, (limit,))
        return pd.DataFrame(rows, columns=["ts", "value", "label"])

    # ------------------------------------------------------------------
    # News Sentiment
    # ------------------------------------------------------------------
    async def insert_news_sentiment(self, rows: List[Tuple]) -> int:
        assert self._db is not None
        sql = """
            INSERT OR IGNORE INTO news_sentiment
                (symbol, ts, positive, negative, neutral, total)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_news_sentiment(
        self, symbol: str, start_ts: Optional[str] = None, limit: int = 500
    ) -> pd.DataFrame:
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        params.append(limit)
        where = " AND ".join(conditions)
        sql = f"SELECT symbol, ts, positive, negative, neutral, total FROM news_sentiment WHERE {where} ORDER BY ts ASC LIMIT ?"
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(rows, columns=["symbol", "ts", "positive", "negative", "neutral", "total"])

    # ------------------------------------------------------------------
    # Binance cross-exchange data
    # ------------------------------------------------------------------
    async def insert_binance_funding_rate(self, rows: List[Tuple]) -> int:
        assert self._db is not None
        sql = "INSERT OR IGNORE INTO binance_funding_rate (symbol, ts, funding_rate) VALUES (?, ?, ?)"
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_binance_funding_rate(
        self, symbol: str, start_ts: Optional[str] = None, limit: int = 500
    ) -> pd.DataFrame:
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        params.append(limit)
        where = " AND ".join(conditions)
        sql = f"SELECT symbol, ts, funding_rate FROM binance_funding_rate WHERE {where} ORDER BY ts ASC LIMIT ?"
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(rows, columns=["symbol", "ts", "funding_rate"])

    async def insert_binance_oi(self, rows: List[Tuple]) -> int:
        assert self._db is not None
        sql = "INSERT OR IGNORE INTO binance_oi (symbol, ts, oi_value) VALUES (?, ?, ?)"
        cursor = await self._db.executemany(sql, rows)
        await self._db.commit()
        return cursor.rowcount

    async def get_binance_oi(
        self, symbol: str, start_ts: Optional[str] = None, limit: int = 500
    ) -> pd.DataFrame:
        assert self._db is not None
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start_ts:
            conditions.append("ts >= ?")
            params.append(start_ts)
        params.append(limit)
        where = " AND ".join(conditions)
        sql = f"SELECT symbol, ts, oi_value FROM binance_oi WHERE {where} ORDER BY ts ASC LIMIT ?"
        rows = await self._db.execute_fetchall(sql, tuple(params))
        return pd.DataFrame(rows, columns=["symbol", "ts", "oi_value"])

    # ------------------------------------------------------------------
    # Model Sharpes (for ensemble weighting)
    # ------------------------------------------------------------------
    async def upsert_model_sharpe(self, model_name: str, interval: str, sharpe: float) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO model_sharpes (model_name, interval, sharpe, updated_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(model_name, interval) DO UPDATE SET sharpe=excluded.sharpe, updated_at=excluded.updated_at",
            (model_name, interval, sharpe),
        )
        await self._db.commit()

    async def get_model_sharpes(self, interval: str = "60") -> Dict[str, float]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT model_name, sharpe FROM model_sharpes WHERE interval = ?", (interval,)
        )
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Feature Retirement tracking
    # ------------------------------------------------------------------
    async def get_feature_retirement_status(self, interval: str = "60") -> Dict[str, bool]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT feature_name, is_retired FROM feature_retirement WHERE interval = ?",
            (interval,),
        )
        return {r[0]: bool(r[1]) for r in rows}

    async def update_feature_retirement(
        self, feature_name: str, interval: str, below_count: int, is_retired: bool
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO feature_retirement (feature_name, interval, below_count, is_retired, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(feature_name, interval) DO UPDATE SET "
            "below_count=excluded.below_count, is_retired=excluded.is_retired, updated_at=excluded.updated_at",
            (feature_name, interval, below_count, int(is_retired)),
        )
        await self._db.commit()
