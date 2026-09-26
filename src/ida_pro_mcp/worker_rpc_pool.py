"""Bounded, exclusive HTTP/1.1 leases for one worker process.

No request is retried here: a disconnected tool call may already have mutated
the database. Closing the pool rejects new leases, closes idle connections and
discards in-flight connections when their owners release them.
"""

from __future__ import annotations

import http.client
import select
from threading import Condition
import time


MAX_CONNECTIONS = 2
ACQUIRE_TIMEOUT_SECONDS = 10.0
# The worker HTTP handler closes idle connections after 30 seconds.
IDLE_TIMEOUT_SECONDS = 15.0


class WorkerRpcPool:
    def __init__(
        self,
        *,
        max_connections: int = MAX_CONNECTIONS,
        acquire_timeout: float = ACQUIRE_TIMEOUT_SECONDS,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
    ) -> None:
        if max_connections < 1 or acquire_timeout <= 0 or idle_timeout < 0:
            raise ValueError("Invalid worker RPC connection pool limits")
        self._limit = max_connections
        self._acquire_timeout = acquire_timeout
        self._idle_timeout = idle_timeout
        self._condition = Condition()
        self._endpoint: tuple[str, int] | None = None
        self._idle: list[tuple[http.client.HTTPConnection, float]] = []
        self._leased: set[http.client.HTTPConnection] = set()
        self._closed = False

    @staticmethod
    def _usable(conn: http.client.HTTPConnection) -> bool:
        if conn.sock is None:
            return False
        try:
            # A fully drained idle HTTP connection has nothing to read. EOF,
            # unsolicited bytes or a socket error require a fresh connection.
            readable, _, exceptional = select.select([conn.sock], [], [conn.sock], 0)
            return not readable and not exceptional
        except (OSError, ValueError):
            return False

    def acquire(self, host: str, port: int, *, timeout: float | None) -> http.client.HTTPConnection:
        started = time.monotonic()
        wait_budget = self._acquire_timeout if timeout is None else min(timeout, self._acquire_timeout)
        deadline = started + wait_budget
        with self._condition:
            if self._endpoint is None:
                self._endpoint = (host, port)
            elif self._endpoint != (host, port):
                raise RuntimeError("Worker RPC connection pool endpoint changed")
            while True:
                if self._closed:
                    raise RuntimeError("Worker RPC connection pool is closed")
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError("Timed out waiting for worker RPC connection pool")
                conn = None
                while self._idle:
                    candidate, returned_at = self._idle.pop()
                    if now - returned_at < self._idle_timeout and self._usable(candidate):
                        conn = candidate
                        break
                    candidate.close()
                if conn is None and len(self._leased) < self._limit:
                    conn = http.client.HTTPConnection(host, port)
                if conn is not None:
                    # Reused sockets must not retain a prior ping's short timeout
                    # or a prior analysis request's long/unlimited timeout.
                    remaining = None if timeout is None else max(0.0, timeout - (now - started))
                    try:
                        conn.timeout = remaining
                        if conn.sock is not None:
                            conn.sock.settimeout(remaining)
                    except BaseException:
                        conn.close()
                        self._condition.notify()
                        raise
                    self._leased.add(conn)
                    return conn
                self._condition.wait(deadline - now)

    def release(self, conn: http.client.HTTPConnection, *, reusable: bool) -> None:
        with self._condition:
            if conn not in self._leased:
                raise RuntimeError("Connection does not belong to an active worker RPC lease")
            self._leased.remove(conn)
            if reusable and not self._closed and conn.sock is not None:
                self._idle.append((conn, time.monotonic()))
            else:
                conn.close()
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            for conn, _ in self._idle:
                conn.close()
            self._idle.clear()
            self._condition.notify_all()
