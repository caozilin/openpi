import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    / 使用 WebSocket 协议提供策略服务。参见 websocket_client_policy.py 了解客户端实现。

    Currently only implements the `load` and `infer` methods.
    目前仅实现了 `load` 和 `infer` 方法。
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize WebSocket policy server.
        / 初始化 WebSocket 策略服务器。

        Args:
            policy: The policy to serve. / 要提供的策略。
            host: Server host address. / 服务器主机地址。
            port: Server port number. / 服务器端口号。
            metadata: Optional metadata to send to clients. / 可选的元数据，将发送给客户端。
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Start the server and run forever. / 启动服务器并永久运行。"""
        asyncio.run(self.run())

    async def run(self):
        """Run the async server. / 运行异步服务器。"""
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """Handle WebSocket connections and process inference requests.
        / 处理 WebSocket 连接并处理推理请求。

        Args:
            websocket: The WebSocket connection. / WebSocket 连接对象。
        """
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send metadata to client upon connection. / 连接时向客户端发送元数据。
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                # Receive and deserialize observation from client. / 从客户端接收并反序列化观察数据。
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # Perform inference. / 执行推理。
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # Add server timing information. / 添加服务器计时信息。
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    # 我们只能记录上一次的总时间，因为我们还想包含发送时间。
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # Send serialized action back to client. / 将序列化的动作发送回客户端。
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # Send error traceback to client. / 向客户端发送错误堆栈跟踪。
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """Health check endpoint handler. / 健康检查端点处理器。

    Args:
        connection: The server connection. / 服务器连接对象。
        request: The HTTP request. / HTTP 请求对象。

    Returns:
        Health check response if path is "/healthz", None otherwise.
        / 如果路径是 "/healthz" 则返回健康检查响应，否则返回 None。
    """
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling. / 继续正常的请求处理。
    return None
