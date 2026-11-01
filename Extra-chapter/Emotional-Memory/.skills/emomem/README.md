# EmoMem

跨会话长期记忆命令行工具 + Claude Code skill（**完全自包含版**）。

基于 Reverie 的 `src/memory` 模块能力封装，但所有用到的类/函数都已作为源码副本
vendored 到本应用的 `src/` 目录下：**运行时不 import 任何 Reverie 仓库代码，
也不依赖当前工作目录**，可放在任意位置直接运行。

- 数据默认存在 `~/.emomem/`（SQLite 事实库 `metadata.db`），可用 `--data-dir` 覆盖。
- 语义检索依赖本机缓存的 `sentence-transformers` 模型；缺失时自动降级为词法召回。
- skill 定义见 `SKILL.md`。

## 依赖

```bash
pip install -r requirements.txt
```

必需：`numpy`、`pydantic>=2.13`、`sqlite-vec`。可选：

- `sqlcipher3`：数据库加密（缺省时用明文 SQLite，功能不受影响）；
- `sentence-transformers`：语义召回（缺省时降级为词法召回）；
- `lancedb`：旧版 LanceDB 数据的一次性迁移（可选迁移工具）。

## 快速开始

```bash
python d:/src/skills/emomem/bin/emomem.py --help
python d:/src/skills/emomem/bin/emomem.py add "用户喜欢手冲咖啡"
python d:/src/skills/emomem/bin/emomem.py search 咖啡
python d:/src/skills/emomem/bin/emomem.py remember "用户生日是3月2日"
python d:/src/skills/emomem/bin/emomem.py list
python d:/src/skills/emomem/bin/emomem.py export ~/memory-backup.json
python d:/src/skills/emomem/bin/emomem.py import ~/memory-backup.json --replace
```

所有命令支持 `--json` 输出，便于脚本与 Claude Code 解析。常用命令清单见
`SKILL.md`。

## 目录结构

```
emomem/
├── SKILL.md          # Claude Code skill 定义
├── README.md
├── requirements.txt
├── bin/emomem.py     # CLI 入口
├── tests/            # pytest 冒烟测试
└── src/              # ★ vendored 记忆引擎副本（本应用自有的 src 包）
    ├── memory/       # 记忆核心（manager/catalog/layers/…）
    ├── persona/      # 身份护栏 + Persona 卡
    ├── config/       # 裁剪版 MemorySettings/FeatureSettings
    ├── storage/      # SQLite（SQLCipher 可选）连接
    ├── web/          # 文本净化（提示注入防护）
    └── backup.py     # 增量 JSON 读取（restore 用）
```

## 数据与隐私

- 数据明文存放在你指定的目录（默认 `~/.emomem`），与应用/仓库目录隔离。
- 身份锚点（`My name is EmoMem …`）是 CLI 自身的中性身份，不是用户记忆。
- 备份文件是标准 JSON 数组，用 `export` 生成、`import` 恢复。

## 测试

```bash
python -m pytest d:/src/skills/emomem/tests -q
```

测试使用临时 `--data-dir`，不触碰 `~/.emomem`。

## 副本来源

`src/` 下的 vendored 文件复制自项目 Reverie 的 `src/memory` / `src/persona` /
`src/config` / `src/storage` / `src/web`（经裁剪：settings 只剩记忆相关字段、
backup 只保留增量读取器）。如需与上游同步，对比对应文件并手动合并即可。