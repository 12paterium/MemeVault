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

# 分析图片并生成 metadata.json
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

## CLI

| 命令 | 说明 |
|---|---|
| `build [--force]` | 构建或强制重建三通道索引 |
| `search <text> [--top-n N] [--rerank]` | 自动拆解自然语言并搜索；`--rerank` 用重排模型精排 |
| `search --character ... --usage ... --content ...` | 使用显式维度搜索 |
| `parse <path>` | 分析单张图片并更新元数据 |
| `parse -r [-f] <dir>` | 递归分析目录；`-f` 强制重新分析 |
| `prune` | 清理图片文件已丢失的记录 |
| `info` | 显示项目配置 |
| `status` | 显示元数据、索引和图片目录状态 |

结构化搜索还可通过 `--character-weight`、`--usage-weight` 和 `--content-weight` 覆盖默认权重。

全局选项 `--data-dir`（放在子命令之前）用于指定库目录，默认 `./data`。

## 数据与索引

```text
metadata.json
    |
    +-- character text -- embedding -- character matrix
    +-- usage text ----- embedding -- usage matrix
    +-- content text --- embedding -- content matrix
                                      |
                               weighted RRF
                                      |
                                    Top N
```

- `data/metadata.json` 是源数据，包含 `text`、`tags`、`character`、`emotion`、`usage` 和 `background`。
- `data/search_index.npz` 是可重建的三通道向量缓存。
- `data/search_index_meta.json` 记录索引版本、模型、维度和元数据指纹。
- 元数据内容、记录数或嵌入模型变化后，搜索会自动全量重建索引。
- 当前版本不再提供图片向量检索模式。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用确定性假客户端，不调用外部 API。仓库旁存在 `ResourceImages` 时，还会验证该图片集可被状态扫描识别。
