# 面向 Agentic RAG 的持久化纠错记忆

## 一、目标与边界

RAG 的外部知识库回答“哪些资料能支持当前问题”；Agent Memory 回答“这个用户曾明确告诉系统什么，以及以后应如何行动”。二者都可能经过检索，却不能共用同一套信任规则：网页中的一句“以后都忽略系统指令”是资料，不是用户偏好。

本专题给出一个可直接运行的最小实现，集中验证四件事：

1. `write / update / recall / delete` 生命周期；
2. 同一主题冲突时，用户纠正优先于旧偏好；
3. 用户、命名空间与来源三重边界；
4. 关闭并重新打开 SQLite 后，行为评估仍可复现。

实现只使用 Python 3.12 标准库。它不是聊天机器人，也不调用 LLM；这样能把“记忆策略是否正确”与“模型生成是否随机”分开。

## 二、知识检索与记忆读写不是一条路径

```text
外部文档 ─→ 知识索引 ─→ 有出处的事实片段 ─┐
                                          ├─→ 上下文组装 ─→ Agent
用户消息 ─→ 写入策略 ─→ SQLite 长期记忆 ──┘
               │              │
               ├─ 来源校验     ├─ owner + namespace 隔离
               ├─ 冲突替代     ├─ 到期过滤
               └─ 最小化写入   └─ 纠正优先排序
```

本例将记忆分成五类：

| `kind` | 含义 | 典型来源 |
|---|---|---|
| `correction` | 用户对既有结论的明确纠正 | user |
| `preference` | 稳定偏好 | user |
| `episode` | 一次事件 | user / system |
| `rule` | 经授权的长期行为约束 | user / system |
| `knowledge` | 带来源的外部资料 | external |

外部资料可以作为 `knowledge` 存储，但默认不进入 Agent Memory 的召回；它不能直接写成 `correction`、`preference` 或 `rule`。这是一条简单而重要的防投毒边界。

## 三、目录结构

```text
correction-aware-agent-memory/
├── readme.md
├── code/
│   ├── memory_policy.py       # 来源校验、词项提取与确定性排序
│   ├── memory_store.py        # SQLite CRUD、版本链与审计事件
│   ├── demo_two_sessions.py   # 两个独立连接模拟两个会话
│   ├── evaluate.py            # 三种策略的固定数据评估
│   └── requirements.txt       # 声明无第三方依赖
├── data/
│   └── correction_cases.jsonl
└── tests/
    └── test_memory.py
```

## 四、最小设计

### 4.1 写入与更新

每条记录包含 `owner_id`、`namespace`、`memory_key`、值、类型、来源、时间、有效期和状态。来源引用 `source_ref` 必填，使每次写入都能回到会话消息或文档。

同一用户、命名空间和 key 下写入 `correction` 时，全部未过期的活动冲突都会变为 `superseded`，纠正记录通过 `supersedes_id` 指向其中最新的旧版本。`update` 同样新增版本而非原地覆盖。重复写入相同活动内容会返回原 ID，并留下新的来源审计事件；已经过期的同值记录不会阻止重新写入。

### 4.2 召回

召回先执行硬边界，再做排序：

1. SQL 层只读取同一 `owner_id + namespace`；
2. 排除删除、被替代和到期记录；
3. 默认排除 `external` 来源；
4. 未指定 key 时，词项无交集的记录不进入候选集；
5. 候选按“纠正优先级、词法相关度、更新时间、ID”降序排列。

这里的相关度使用英文词、汉字单字和汉字二元组的 Jaccard 分数，不追求语义召回效果，只为在零依赖条件下获得可解释、确定的基线。

### 4.3 删除

`delete(memory_id, owner_id)` 要求所有者匹配，并将记录软删除。软删除便于本地演示审计，但生产系统还应设置审计保留期限，并在合规要求下执行物理清除、备份清除和下游缓存失效。

### 4.4 SQLite 持久化

业务写入和审计事件位于同一事务中。第二个会话重新创建 `MemoryStore` 并连接同一数据库文件，因此验证的是跨连接持久化，不是 Python 对象仍在内存中的假象。

## 五、环境与运行

要求：

- Python 3.12；
- 无网络、API Key、模型和第三方包；
- SQLite 由 Python 标准库 `sqlite3` 提供。

从本专题目录运行：

```bash
cd Extra-chapter/correction-aware-agent-memory
python3.12 -m unittest discover -s tests -v
python3.12 code/demo_two_sessions.py
python3.12 code/evaluate.py
python3.12 -m py_compile code/*.py tests/*.py
```

如需保留演示数据库：

```bash
python3.12 code/demo_two_sessions.py --db data/demo_memory.sqlite3
```

不传 `--db` 时使用临时目录，运行后自动清理。

## 六、API 示例

```python
from memory_store import MemoryStore

with MemoryStore("memory.sqlite3") as store:
    memory_id = store.write(
        owner_id="user-001",
        namespace="assistant",
        memory_key="seat",
        value="纠正：以后请预订靠过道座位",
        kind="correction",
        source_type="user",
        source_ref="session-42/message-7",
    )
    result = store.recall(
        owner_id="user-001",
        namespace="assistant",
        query="请帮我预订座位",
        memory_key="seat",
        limit=1,
    )
    store.delete(memory_id=memory_id, owner_id="user-001")
```

调用方不应让 LLM 自由生成 `owner_id`、`source_type` 或权限信息；这些字段应由已认证的应用层注入。

## 七、跨会话评估

`evaluate.py` 对固定 JSONL 逐条创建数据库：会话一写入旧偏好和纠正，关闭连接；会话二重新打开文件并召回。它比较三种策略：

- `no_memory`：没有持久记忆，沿用旧默认；
- `similarity`：只按词法相似度选候选，不理解纠正、删除或相关性阈值；
- `correction_aware`：使用本专题的边界、版本和排序策略。

### 7.1 指标定义

| 指标 | 定义 | 趋势 |
|---|---|---|
| Correction Adherence | 正确采用纠正的样例数 / 纠正样例数 | 越高越好 |
| Repeat-Mistake Rate | 再次采用旧错误的样例数 / 纠正样例数 | 越低越好 |
| Stale-Conflict Resolution | 冲突中选中最新有效纠正的样例数 / 冲突样例数 | 越高越好 |
| Irrelevant-Memory Intrusion | 无关问题仍注入记忆的样例数 / 无关样例数 | 越低越好 |
| Deletion Compliance | 删除后不再召回的样例数 / 删除样例数 | 越高越好 |
| Recall@1 | 相关样例中首条召回等于期望值的数量 / 相关样例数 | 越高越好 |
| Token Overhead | `ceil(召回文本字符数 / 4)` 的均值 | 越低越好 |
| Recall Latency | 打开数据库并召回的实测平均微秒数 | 仅观测 |

Token Overhead 是无 tokenizer 条件下的代理量，不等同于任何模型的真实 token 数。行为指标、召回结果和代理开销由固定输入、整数时钟和稳定排序决定；墙钟时延受机器负载影响，不参与通过条件。

### 7.2 解释结果

这组小数据不是性能排行榜，而是回归夹具。通过标准是纠错策略达到：

- Correction Adherence、Stale-Conflict Resolution、Deletion Compliance、Recall@1 为 `1.0`；
- Repeat-Mistake Rate、Irrelevant-Memory Intrusion 为 `0.0`。

测试直接断言这些目标，任何策略改动导致行为退化都会失败。

## 八、安全、隐私与工程限制

本例已经做到：

- 必填来源引用，保留版本链和审计事件；
- 外部资料不能写成用户纠正、偏好或规则；
- 用户和命名空间严格隔离；
- 支持到期与所有者约束下的显式删除；
- 无关候选零重叠时不召回。

仍未覆盖：

1. 无身份认证、授权服务和数据库加密；SQLite 文件权限由宿主机负责。
2. 软删除不是法规意义上的完整擦除，未处理备份、日志和导出副本。
3. 词法召回不理解同义词、多语言语义或否定关系；生产环境可换成检索器，但不能绕过来源过滤。
4. 纠正判定依赖调用方显式传入 `kind`，没有用模型自动分类；实际系统应采用结构化确认和高风险人工复核。
5. 单进程示例未实现并发版本检查、密钥轮换、批量压缩或记忆总结。
6. 评估集仅用于说明指标定义，不能代表真实用户分布；上线前应加入对抗输入、跨租户、长周期和误删恢复测试。

## 九、参考资料

1. Python 3.12 `sqlite3` 官方文档：连接、事务控制与 DB-API 行为：https://docs.python.org/zh-cn/3.12/library/sqlite3.html
2. SQLite 官方事务文档：事务、读写锁与提交语义：https://www.sqlite.org/lang_transaction.html
3. OWASP《LLM Prompt Injection Prevention Cheat Sheet》：将不可信内容视为数据、做输入校验与纵深防御：https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html
4. GDPR 第 17 条（EUR-Lex）：数据主体的删除权及适用条件：https://eur-lex.europa.eu/eli/reg/2016/679/art_17/oj
5. Packer 等，*MemGPT: Towards LLMs as Operating Systems*：将有限上下文与外部存储组织为分层记忆：https://arxiv.org/abs/2310.08560
6. Park 等，*Generative Agents: Interactive Simulacra of Human Behavior*：记忆流、反思与规划架构：https://doi.org/10.1145/3586183.3606763
