"""
SeekDB MCP Project Discovery Assistant

This module provides MCP client functionality for discovering and recommending
open-source projects using SeekDB's hybrid search capabilities.
"""

import argparse, asyncio, hashlib, json, logging, os, queue, re, shlex, subprocess, threading, time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROTOCOL_VERSION = "2024-11-05"

# 配置日志
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """配置日志系统

    Args:
        verbose: 是否显示详细日志
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def split_command(command: str) -> List[str]:
    """Split command string into arguments, handling Windows/POSIX differences.

    Args:
        command: 命令字符串

    Returns:
        分割后的参数列表
    """
    return shlex.split(command, posix=os.name != "nt")


def sanitize_name(name: str) -> str:
    """Sanitize collection name by replacing non-alphanumeric characters.

    Args:
        name: 原始名称

    Returns:
        清理后的名称
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def norm(items: List[Any]) -> List[Any]:
    """Normalize list items to consistent format.

    Args:
        items: 输入列表

    Returns:
        规范化后的列表
    """
    return [] if not items else (items[0] if isinstance(items[0], list) else items)


def cache_key(collection_name: str, query: str, keyword: Optional[str], top_k: int) -> str:
    """生成缓存键

    Args:
        collection_name: 集合名称
        query: 查询文本
        keyword: 关键词
        top_k: 返回结果数量

    Returns:
        MD5 缓存键
    """
    key_str = f"{collection_name}|{query}|{keyword or 'None'}|{top_k}"
    return hashlib.md5(key_str.encode()).hexdigest()


class MCPStdioClient:
    """MCP stdio 客户端实现，支持异步请求和上下文管理"""

    def __init__(self, command: List[str]) -> None:
        """初始化客户端

        Args:
            command: 启动服务端的命令列表
        """
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        self._id = 0
        self._queue = queue.Queue()
        # 优化 1: 使用 threading.Event 替代忙等待
        self._events: Dict[int, Dict[str, Any]] = {}
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        """读取服务端输出，使用 Event 通知等待线程"""
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line: continue
            try: msg = json.loads(line)
            except json.JSONDecodeError: continue
            req_id = msg.get("id")
            # 优化 1: 如果有请求在等待，直接设置 event
            if req_id in self._events:
                self._events[req_id]['response'] = msg
                self._events[req_id]['event'].set()
            else:
                self._queue.put(msg)

    def _send(self, payload: Dict[str, Any]) -> None:
        """发送请求到服务端

        Args:
            payload: 请求负载
        """
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: float = 30) -> Dict[str, Any]:
        """发送请求并等待响应

        Args:
            method: 请求方法名
            params: 请求参数
            timeout: 超时时间（秒）

        Returns:
            响应数据

        Raises:
            TimeoutError: 请求超时
        """
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None: payload["params"] = params
        self._send(payload)
        return self._wait(self._id, timeout)

    def _wait(self, req_id: int, timeout: float) -> Dict[str, Any]:
        """等待响应（使用 Event 替代忙等待）

        Args:
            req_id: 请求 ID
            timeout: 超时时间（秒）

        Returns:
            响应数据

        Raises:
            TimeoutError: 请求超时
        """
        # 优化 1: 创建 Event 并等待
        event = threading.Event()
        self._events[req_id] = {'event': event, 'response': None}

        if not event.wait(timeout):
            del self._events[req_id]
            logger.error(f"Timeout waiting for response id={req_id}")
            raise TimeoutError(f"Timeout waiting for response id={req_id}")

        response = self._events[req_id]['response']
        del self._events[req_id]
        return response

    def initialize(self) -> Dict[str, Any]:
        """初始化客户端

        Returns:
            服务端响应
        """
        params = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "seekdb-project-finder", "version": "0.2"},
        }
        resp = self.request("initialize", params)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        return resp

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具返回结果

        Raises:
            RuntimeError: 工具调用失败
        """
        resp = self.request("tools/call", {"name": name, "arguments": arguments})
        if resp.get("error"):
            logger.error(f"Tool call failed: {resp['error']}")
            raise RuntimeError(resp["error"])
        content = resp.get("result", {}).get("content", [])
        text = "".join(item.get("text", "") for item in content if item.get("type") == "text")
        if not text: return resp.get("result", {})
        try: return json.loads(text)
        except json.JSONDecodeError: return text

    # 优化 7: Context Manager 支持
    def __enter__(self) -> 'MCPStdioClient':
        """进入上下文管理器

        Returns:
            客户端实例
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """退出上下文管理器

        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常追踪

        Returns:
            False（不抑制异常）
        """
        self.close()
        return False

    def close(self) -> None:
        """关闭客户端并清理资源"""
        if self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                logger.warning("Forcefully killed subprocess")


class CachedMCPClient:
    """带缓存功能的 MCP 客户端装饰器

    优化 2: 缓存搜索结果，避免重复调用
    """

    def __init__(self, client: MCPStdioClient) -> None:
        """初始化缓存客户端

        Args:
            client: 底层 MCP 客户端
        """
        self.client = client
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具（带缓存）

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具返回结果
        """
        if name == "hybrid_search":
            ck = self._make_cache_key(arguments)
            if ck in self._cache:
                self._cache_hits += 1
                logger.debug(f"[CACHE HIT] Using cached result for hybrid_search")
                return self._cache[ck]
            self._cache_misses += 1
            result = self.client.call_tool(name, arguments)
            self._cache[ck] = result
            return result
        return self.client.call_tool(name, arguments)

    def _make_cache_key(self, arguments: Dict[str, Any]) -> str:
        """生成缓存键

        Args:
            arguments: 工具参数

        Returns:
            缓存键字符串
        """
        return cache_key(
            arguments.get("collection_name", ""),
            arguments.get("knn_query_texts", [""])[0] if arguments.get("knn_query_texts") else "",
            arguments.get("fulltext_search_keyword"),
            arguments.get("n_results", 5)
        )

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息

        Returns:
            缓存统计字典
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache)
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Cache cleared")

    def initialize(self) -> Dict[str, Any]:
        """初始化客户端

        Returns:
            服务端响应
        """
        return self.client.initialize()

    def close(self) -> None:
        """关闭客户端"""
        self.client.close()


def print_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """格式化打印搜索结果

    Args:
        data: 搜索结果数据

    Returns:
        结果列表
    """
    ids = norm(data.get("ids"))
    docs = norm(data.get("documents"))
    metas = norm(data.get("metadatas"))
    results: List[Dict[str, Any]] = []

    if not ids:
        print("No results returned.")
        return results

    for idx, doc_id in enumerate(ids):
        meta = metas[idx] if idx < len(metas) else {}
        summary = docs[idx] if idx < len(docs) else ""
        title = meta.get("name") or doc_id
        print(f"{idx + 1}. {title}")
        print(f"   summary: {summary}")
        if meta:
            print(
                "   tags: {tags} | stars: {stars} | language: {language} | repo: {repo}".format(
                    tags=meta.get("tags", ""), stars=meta.get("stars", ""),
                    language=meta.get("language", ""), repo=meta.get("repo", "")
                )
            )
        results.append({"id": doc_id, "summary": summary, "meta": meta})
    return results


# 优化 4: 异步 LLM 调用
async def ai_complete_async(client: CachedMCPClient, model_name: str, prompt: str) -> Any:
    """异步 LLM 调用

    Args:
        client: MCP 客户端
        model_name: 模型名称
        prompt: 提示词

    Returns:
        LLM 响应结果
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: client.call_tool("ai_complete", {
            "model_name": model_name,
            "prompt": prompt
        })
    )


def count_csv_lines(file_path: Path) -> int:
    """统计 CSV 文件行数（用于进度显示）

    Args:
        file_path: CSV 文件路径

    Returns:
        文件总行数（减去表头）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # 减去表头


def main() -> int:
    """主函数

    Returns:
        退出代码
    """
    parser = argparse.ArgumentParser(description="SeekDB MCP project discovery demo")
    parser.add_argument("--server-command", default="uvx seekdb-mcp-server")
    parser.add_argument("--data", default="data/projects.csv")
    parser.add_argument("--vector-column", type=int, default=2)
    parser.add_argument("--query", default="lightweight RAG for internal docs")
    parser.add_argument("--keyword", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--llm-model", default=os.getenv("SEEKDB_COMPLETION_MODEL", ""))
    parser.add_argument("--verbose", action="store_true", help="Show verbose logging")
    parser.add_argument("--async-llm", action="store_true", help="Use async LLM call")
    parser.add_argument("--no-cache", action="store_true", help="Disable result caching")
    args = parser.parse_args()

    # 优化 6: 设置日志
    setup_logging(args.verbose)

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1

    # 优化 7: 使用 Context Manager
    with MCPStdioClient(split_command(args.server_command)) as raw_client:
        client = CachedMCPClient(raw_client) if not args.no_cache else raw_client
        try:
            client.initialize()
            logger.info(f"Connected to SeekDB server")

            collection_name = sanitize_name(data_path.stem)

            if not args.skip_import:
                # 优化 3: 进度显示
                total_lines = count_csv_lines(data_path)
                logger.info(f"Importing data from {data_path} ({total_lines} records)...")

                import_result = client.call_tool(
                    "import_csv_file_to_seekdb",
                    {"filePath": str(data_path), "columnNumberForVecotor": args.vector_column},
                )

                if isinstance(import_result, dict) and import_result.get("success"):
                    collection_name = import_result.get("collection_name", collection_name)
                    logger.info(f"Import completed! Collection: {collection_name}")
                else:
                    err = (
                        str(import_result.get("error", ""))
                        if isinstance(import_result, dict) else str(import_result)
                    )
                    if "exist" in err.lower():
                        logger.info("Collection already exists, reusing it.")
                    else:
                        logger.error(f"Import failed: {import_result}")
                        return 1

            search_args = {
                "collection_name": collection_name,
                "fulltext_search_keyword": args.keyword or args.query,
                "knn_query_texts": [args.query],
                "n_results": args.top_k,
                "include": ["documents", "metadatas"],
            }

            logger.info(f"Searching: '{args.query}' (top-k={args.top_k})")
            search_result = client.call_tool("hybrid_search", search_args)

            if not (isinstance(search_result, dict) and search_result.get("success")):
                logger.warning("Hybrid search failed, falling back to vector search.")
                search_result = client.call_tool(
                    "query_collection",
                    {
                        "collection_name": collection_name,
                        "query_texts": [args.query],
                        "n_results": args.top_k,
                        "include": ["documents", "metadatas"],
                    },
                )

            if isinstance(search_result, dict) and search_result.get("success"):
                results = print_results(search_result.get("data", {}))
            else:
                logger.error(f"Search failed: {search_result}")
                return 1

            # 显示缓存统计
            if hasattr(client, 'get_cache_stats'):
                stats = client.get_cache_stats()
                if stats['hits'] > 0 or stats['misses'] > 0:
                    logger.info(f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, {stats['size']} entries")

            if args.llm_model:
                lines = [
                    "You are a project discovery assistant.",
                    f"User query: {args.query}",
                    "",
                    "Candidate projects:",
                ]
                for idx, item in enumerate(results, 1):
                    meta = item.get("meta", {})
                    title = meta.get("name", item.get("id"))
                    tags = meta.get("tags", "")
                    summary = item.get("summary", "")
                    lines.append(f"{idx}. {title} | tags: {tags} | summary: {summary}")
                lines.append("")
                lines.append("Give top 3 recommendations with short reasons. Return a bullet list.")

                prompt = "\n".join(lines)
                logger.debug(f"Calling LLM model: {args.llm_model}")

                # 优化 4: 异步或同步 LLM 调用
                if args.async_llm:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        llm_result = loop.run_until_complete(
                            ai_complete_async(client, args.llm_model, prompt)
                        )
                    finally:
                        loop.close()
                else:
                    llm_result = client.call_tool(
                        "ai_complete", {"model_name": args.llm_model, "prompt": prompt}
                    )

                if isinstance(llm_result, dict) and llm_result.get("success"):
                    print("\nLLM recommendations:\n" + llm_result.get("response", ""))
                else:
                    logger.warning(f"LLM call failed: {llm_result}")

        except TimeoutError as e:
            logger.error(f"Timeout error: {e}")
            return 1
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            return 1
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return 1

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
