# MemeVault（表情包多维语义检索）

MemeVault 将表情包元数据拆成独立检索通道，再融合各通道排名：

- `character`：人物、角色或外观，默认权重 `3`
- `usage`：表情包用途或聊天场景，默认权重 `3`
- `content`：描述、标签、情绪、梗背景和文件名，默认权重 `1`

每个通道分别进行余弦检索，最终使用加权 RRF（Reciprocal Rank Fusion）输出 Top N。普通自然语言查询会先由聊天模型拆成三个通道；显式结构化查询不会调用聊天模型。

## 快速开始

```bash
pip install -e .

# Linux / macOS
export SILICONFLOW_API_KEY="sk-xxx"

# PowerShell
$env:SILICONFLOW_API_KEY="sk-xxx"

# 分析图片并写进 memes/（图片会复制进库，原图保留）
meme-vault parse -r ./ResourceImages

# 构建 character / usage / content 三通道索引
meme-vault build

# 自然语言查询：先自动拆解，再分别检索
meme-vault search "找一张初音未来吐槽群友的无语表情" --top-n 5

# 结构化查询：不调用查询拆解模型
meme-vault search --character "初音未来" --usage "吐槽群友" --content "无语"

# 指定库目录（默认 ./data）
meme-vault --data-dir /path/to/vault status
```

## Python API

```python
import asyncio

from meme_vault import MemeVault, SearchQuery


async def main():
    async with MemeVault(data_dir="./data") as vault:
        await vault.build()

        # 自动拆解自然语言
        results = await vault.search(
            "找一张初音未来吐槽群友的无语表情",
            top_n=5,
        )

        # 或直接指定各维度，权重作为检索参数传入
        results = await vault.search(
            SearchQuery(
                character="初音未来",
                usage="吐槽群友",
                content="无语",
            ),
            top_n=5,
            weights={"character": 3, "usage": 3, "content": 1},
        )

        for result in results:
            print(f"[{result.score:.4f}] {result.metadata.text}")
            for dimension, detail in result.dimensions.items():
                print(
                    dimension,
                    detail.similarity,
                    detail.rank,
                    detail.weight,
                    detail.contribution,
                )


asyncio.run(main())
```

`SearchResult` 包含：

- `metadata`：命中的 `Metadata`
- `score`：归一化后的加权 RRF 总分
- `dimensions`：实际参与该结果的各通道 `DimensionScore`

`DimensionScore` 包含 `similarity`、`rank`、`weight` 和 `contribution`，可用于解释排序与调整权重。

### 服务方配置

嵌入、视觉（解析图片）、对话（查询拆解）、重排可以分别指向不同的 OpenAI 兼容服务：

```python
MemeVault(
    data_dir="./data",
    api_key="...",                        # 默认 key，被下面没单独指定的角色复用
    embedding_model="Qwen/Qwen3-VL-Embedding-8B",
    vision_model="Qwen/Qwen3-VL-32B-Instruct",   # 加图时的图片解析
    vision_api_key="...", vision_base_url="http://your-relay/v1",
    chat_model="Qwen/Qwen3-VL-32B-Instruct",   # 每条消息的查询拆解
    chat_api_key="...", chat_base_url="https://api.siliconflow.cn/v1",
    rerank_model="Qwen/Qwen3-VL-Reranker-8B",
)
```

`chat_*`（查询拆解）和 `embedding_*`、`rerank_*` 不填时分别回落到默认地址 / 默认 key；`chat_*` 不填时跟随 `vision_*`。每条回复的查询拆解建议放在响应快的服务上，图片解析可以放在质量更好的模型上。

### 注入自定义客户端

四个角色都可以直接注入现成的客户端对象（比如宿主机器人框架自己的模型接口），而不是让库去创建 HTTP 客户端：

```python
MemeVault(
    data_dir="./data",
    chat_client=my_chat_client,        # 需要 .model / .chat / .parse_image / .close
    vision_client=my_vision_client,
    embed_client=..., rerank_client=...,
)
```

- 客户端要实现 `AIClient` 的接口：`.model` 字符串、`async chat(messages, temperature, response_format) -> str`、
  `async parse_image(path, filename="") -> dict`、`async embed(inputs)`、`async rerank(query, documents, top_n)`、`async close()`
- 模型名不显式配置时取客户端的 `.model`（索引清单、`analyzed_by` 都用它）
- **注入的客户端归调用方所有**，`MemeVault.close()` 只会关掉自己创建的客户端
- 没有 `response_format` 能力的客户端可以只在提示词里要求 JSON，再用 `meme_vault.client.parse_json_object(text)`
  解析回复——它会容忍 ```json 围栏和前后废话，取第一个 JSON 对象，失败抛 `ValueError`

## CLI

| 命令 | 说明 |
|---|---|
| `build [--force]` | 构建或强制重建三通道索引 |
| `search <text> [--top-n N] [--rerank]` | 自动拆解自然语言并搜索；`--rerank` 用重排模型精排 |
| `search --character ... --usage ... --content ...` | 使用显式维度搜索 |
| `parse <path>` | 分析单张图片并更新元数据 |
| `parse -r [-f] <dir>` | 递归分析目录；`-f` 强制重新分析 |
| `prune` | 清理图片文件已丢失的记录 |
| `status` | 显示元数据、索引和图片目录状态 |

结构化搜索还可通过 `--character-weight`、`--usage-weight` 和 `--content-weight` 覆盖默认权重。

全局选项 `--data-dir`（放在子命令之前）用于指定库目录，默认 `./data`。

## 大批量导入

数千张图片建议分批导入：`tools/import_batches.py` 把目录按文件名排序切成 N 批，一次跑一批，
中途限流、网络抖动、重启都不影响进度——**同一批重复跑是幂等的**：已用同一视觉模型解析过的图会自动跳过。

```bash
# 先看这一批有多少张要跑（不调用 API）
python tools/import_batches.py --data-dir ./data --source D:/memes --batch 2 --dry-run

# 跑第 2 批（共 4 批），用插件配置里的视觉服务
python tools/import_batches.py --data-dir ./data --source D:/memes \
    --batch 2 --batches 4 --config <astrbot>/data/config/memeManager_config.json \
    --build --summary batch2.json
```

- `--config` 指向插件配置 JSON（读其中的 `sub_config`），不给就用库默认服务 + `SILICONFLOW_API_KEY`。
- `--build` 在该批解析完后立刻重建索引（只重嵌变化的条目）；不加的话插件会在下次检索时自动增量重建，或手动 `build`。
- `--concurrency` 覆盖 `config.PARSE_CONCURRENCY`（服务商限流时调小）；`--summary` 落一份 JSON 报告（含失败文件清单）。
- 扩展名与实际格式不符的图（如 AVIF 改名成 `.jpg`）会被视觉 API 拒绝，出现在 `failures` 里；
  用 ffmpeg 转成真 JPEG 再导入即可。

## 数据与索引

```text
data/
  memes/    一条表情包一个 JSON（<图片名>.<id 前 8 位>.json），可直接复制/编辑
  images/   图片副本，条目里用相对路径（images/xxx.jpg）引用
  search_index.npz        可重建的三通道向量缓存
  search_index_meta.json  索引版本、模型、维度和元数据指纹
```

```text
memes/*.json
    |
    +-- character text -- embedding -- character matrix
    +-- usage text ----- embedding -- usage matrix
    +-- content text --- embedding -- content matrix
                                      |
                               weighted RRF
                                      |
                                    Top N
```

- `memes/` 是源数据，每条包含 `text`、`tags`、`character`、`emotion`、`usage` 和 `background`；`images/` 是图片副本，整个数据目录可直接拷贝到别处使用。
- 旧的单文件 `metadata.json` 会在首次读取时自动迁移到 `memes/` + `images/`，原文件改名为 `metadata.json.bak` 保留。
- 元数据变化后搜索会自动重建索引，且**只重新嵌入变化的行**（换嵌入模型时才会全量重嵌）。
- `MemeVault` 实例在内存里缓存条目和索引，磁盘文件没变就不重读——适合长驻进程（机器人、服务）。
- 解析图片时会把原始文件名（去掉库内 id 后缀）作为线索一并交给视觉模型，让它补上画面看不出的语境；提示词同时要求不得据文件名推断角色/出处，画面认不出就按外观描述，冲突时以画面为准。
- `parse -r` 并发解析（`config.PARSE_CONCURRENCY`，默认 4），已解析过的图片自动跳过，可断点续跑。
- 网络抖动会自动重试：连接错误与超时在内的网络异常、以及 429/5xx 状态码，都按 `config.RETRY_COUNT` 次重试（优先尊重服务端 `Retry-After`，否则退避 2/4/8 秒），不会一次抖动就中断批量任务。
- 嵌入客户端超时 300 秒（慢服务商单批 64 条可能要 90 秒以上），对话/视觉/重排仍是 120 秒。
- 当前版本不再提供图片向量检索模式。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用确定性假客户端，不调用外部 API。仓库旁存在 `ResourceImages` 时，还会验证该图片集可被状态扫描识别。
