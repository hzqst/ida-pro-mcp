"""Real HTTP/1.1 regression tests; no IDA installation required."""

from concurrent.futures import ThreadPoolExecutor
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import select
import socket
import threading
import time

import pytest

from ida_pro_mcp import idalib_supervisor as supmod
from ida_pro_mcp.worker_rpc_pool import WorkerRpcPool


@pytest.fixture
def endpoint():
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        accepted = 0
        requests = 0
        connections = []
        gate = threading.Event()
        entered = threading.Event()

        def get_request(self):
            result = super().get_request()
            self.accepted += 1
            self.connections.append(result[0])
            return result

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests += 1
            if payload.get("block"):
                self.server.entered.set()
                self.server.gate.wait(3)
            if payload.get("drop"):
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            data = json.dumps({"id": payload["id"], "result": payload.get("value")}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            if payload.get("close"):
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    supervisor = supmod.IdalibSupervisor(supmod.McpServer("pool-test"))
    worker = supmod.WorkerSession("test", "", "", port=server.server_port, owned=False)
    try:
        yield server, supervisor, worker
    finally:
        worker.rpc_pool.close()
        server.gate.set()
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_sequential_rpcs_reuse_one_tcp_connection(endpoint):
    server, supervisor, worker = endpoint
    for index in range(30):
        assert {"id": index, "result": index} == supervisor._worker_rpc(
            worker, {"id": index, "value": index}, timeout=2
        )
    assert 1 == server.accepted
    assert 30 == server.requests


def test_parallel_rpcs_are_bounded_and_responses_do_not_cross(endpoint):
    server, supervisor, worker = endpoint
    def call(index):
        return supervisor._worker_rpc(worker, {"id": index, "value": index}, timeout=3)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(call, range(40)))
    assert [{"id": i, "result": i} for i in range(40)] == results
    assert 1 <= server.accepted <= 2


def test_pool_wait_is_bounded_and_close_wakes_waiter():
    pool = WorkerRpcPool(max_connections=1, acquire_timeout=0.05)
    conn = pool.acquire("127.0.0.1", 1, timeout=None)
    with pytest.raises(TimeoutError, match="connection pool"):
        pool.acquire("127.0.0.1", 1, timeout=None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(pool.acquire, "127.0.0.1", 1, timeout=None)
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            waiting.result(timeout=1)
    pool.release(conn, reusable=True)
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire("127.0.0.1", 1, timeout=None)


def test_server_close_reconnects_on_next_call(endpoint):
    server, supervisor, worker = endpoint
    supervisor._worker_rpc(worker, {"id": 1, "close": True}, timeout=2)
    supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
    assert 2 == server.accepted


def test_side_effect_request_is_never_replayed_on_disconnect(endpoint):
    server, supervisor, worker = endpoint
    with pytest.raises((OSError, http.client.HTTPException)):
        supervisor._worker_rpc(worker, {"id": 1, "drop": True}, timeout=2)
    assert 1 == server.requests
    assert {"id": 2, "result": None} == supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
    assert 2 == server.requests
    assert 2 == server.accepted


def test_reused_socket_uses_current_timeout(endpoint):
    server, supervisor, worker = endpoint
    supervisor._worker_rpc(worker, {"id": 1}, timeout=2)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        supervisor._worker_rpc(worker, {"id": 2, "block": True}, timeout=0.1)
    assert time.monotonic() - started < 1
    assert 2 == server.requests
    server.gate.set()
    supervisor._worker_rpc(worker, {"id": 3}, timeout=2)
    assert 2 == server.accepted


def test_idle_connection_expires_before_reuse(endpoint):
    server, supervisor, worker = endpoint
    worker.rpc_pool = WorkerRpcPool(idle_timeout=0)
    supervisor._worker_rpc(worker, {"id": 1}, timeout=2)
    supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
    assert 2 == server.accepted


def test_detach_closes_pool_without_killing_adopted_worker(endpoint):
    server, supervisor, worker = endpoint
    supervisor._worker_rpc(worker, {"id": 1}, timeout=2)
    supervisor._terminate_worker(worker)
    with pytest.raises(RuntimeError, match="closed"):
        supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
    assert 1 == server.requests


def test_shutdown_closes_persistent_session_connections(endpoint):
    server, supervisor, worker = endpoint
    supervisor.sessions[worker.session_id] = worker
    supervisor._worker_rpc(worker, {"id": 1}, timeout=2)
    supervisor.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
    assert 1 == server.requests


def test_close_during_request_discards_returned_connection(endpoint):
    server, supervisor, worker = endpoint
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(supervisor._worker_rpc, worker, {"id": 1, "block": True}, timeout=2)
        assert server.entered.wait(1)
        worker.rpc_pool.close()
        server.gate.set()
        assert {"id": 1, "result": None} == active.result(timeout=2)
    with pytest.raises(RuntimeError, match="closed"):
        supervisor._worker_rpc(worker, {"id": 2}, timeout=2)


def test_worker_replacement_cannot_reuse_previous_pool(endpoint):
    server, supervisor, worker = endpoint
    supervisor.sessions[worker.session_id] = worker
    supervisor._worker_rpc(worker, {"id": 1}, timeout=2)
    replacement = supmod.WorkerSession("test", "", "", port=server.server_port, owned=False)
    try:
        supervisor._register_session_locked(replacement, "sample.bin")
        with pytest.raises(RuntimeError, match="closed"):
            supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
        supervisor._worker_rpc(replacement, {"id": 3}, timeout=2)
        assert 2 == server.accepted
    finally:
        replacement.rpc_pool.close()


def test_pool_does_not_cross_worker_endpoints():
    pool = WorkerRpcPool()
    conn = pool.acquire("127.0.0.1", 1, timeout=1)
    try:
        with pytest.raises(RuntimeError, match="endpoint changed"):
            pool.acquire("127.0.0.1", 2, timeout=1)
    finally:
        pool.release(conn, reusable=False)
        pool.close()


def test_saturated_pool_does_not_open_overflow_connections(endpoint):
    server, supervisor, worker = endpoint
    worker.rpc_pool = WorkerRpcPool(max_connections=1, acquire_timeout=0.05)
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(supervisor._worker_rpc, worker, {"id": 1, "block": True}, timeout=2)
        assert server.entered.wait(1)
        try:
            with pytest.raises(TimeoutError, match="connection pool"):
                supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
            assert 1 == server.accepted
            assert 1 == server.requests
        finally:
            server.gate.set()
        active.result(timeout=2)


def test_idle_peer_disconnect_is_discarded_before_next_request(endpoint):
    server, supervisor, worker = endpoint
    supervisor._worker_rpc(worker, {"id": 1}, timeout=2)
    # Observe EOF on the client before the next lease; a later race is still
    # allowed to fail, but must never replay a request.
    conn = worker.rpc_pool.acquire(worker.host, worker.port, timeout=2)
    server.connections[0].shutdown(socket.SHUT_RDWR)
    assert select.select([conn.sock], [], [], 1)[0]
    worker.rpc_pool.release(conn, reusable=True)
    supervisor._worker_rpc(worker, {"id": 2}, timeout=2)
    assert 2 == server.accepted
    assert 2 == server.requests
