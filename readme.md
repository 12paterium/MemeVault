# MemeVault（表情包语义检索系统）

输入自然语言，找到最匹配的表情包。基于向量检索的轻量级本地系统。

```bash
pip install -e .
export SILICONFLOW_API_KEY="sk-xxx"
meme-vault status
meme-vault search 无语
```

---

## 快速开始

```bash
# 安装
pip install -e .

# 设置 API 密钥（使用 SiliconFlow）
# Windows: set SILICONFLOW_API_KEY=sk-xxx
export SILICONFLOW_API_KEY="sk-xxx"

# 查看工作目录状态
meme-vault status

# 导入表情包
meme-vault parse path/to/image.jpg          # 单张
meme-vault parse -r ./images                # 批量

# 构建索引
meme-vault build                            # 文本嵌入
meme-vault build-image                      # 图片嵌入（跨模态）

# 搜索
meme-vault search 无语
meme-vault search "群友又开始逆天发言了"
```

## CLI 命令

| 命令 | 说明 |
|---|---|
| `--version` | 显示版本号 |
| `status` | 查看工作目录状态（元数据、嵌入向量、图片文件统计） |
| `build` / `build-text` | 构建文本嵌入向量（默认） |
| `build-image` | 构建图片嵌入向量（跨模态） |
| `search <text>` | 语义搜索表情包（自动检测嵌入模式，自动重建） |
| `parse <path>` | 分析单张图片并入库 |
| `parse -r <dir>` | 递归导入目录下所有图片 |
| `list` | 列出所有表情包 |
| `info` | 查看项目信息 |

## Python API

```python
from meme_vault import MemeVault
import asyncio

async def main():
    vault = MemeVault(data_dir="./data")

    # 导入
    await vault.parse("path/to/image.jpg")
    await vault.parse_dir("./images")

    # 构建嵌入
    await vault.build_text()       # 字段分别嵌入 → 平均池化

    # 搜索
    results = await vault.search("无语", top_k=5)
    for meta, score in results:
        print(f"[{score:.3f}] {meta.text}")

    # 状态
    info = vault.status()
    print(f"版本: {info['version']}, 表情包: {info['metadata']['total']} 张")

    await vault.close()

asyncio.run(main())
```

## 嵌入策略

- **文本模式**：将 text、tags、emotion、usage、character、background 等字段分别输入嵌入模型，然后平均池化。短字段（标签、情绪）不会被长描述淹没。
- **图片模式**：直接对图片进行跨模态嵌入。
- 每次构建自动记录模式，搜索时自动检测并使用。

## 系统架构

```
                      ┌──────────────────┐
                      │   metadata.json   │  ← 数据层（源数据）
                      └────────┬─────────┘
                               │
                      ┌────────▼─────────┐
                      │  embeddings.npy   │  ← 特征层（可重建缓存）
                      │  embeddings_meta  │
                      └────────┬─────────┘
                               │
                      ┌────────▼─────────┐
                      │  cosine search    │  ← 检索层（numpy）
                      └──────────────────┘
```

- **数据 ≠ 特征 ≠ 索引**，三层解耦
- embedding 是可重建的缓存，不属资产
- 文件 ID 基于内容 MD5 哈希，重命名不影响去重

## 配置

| 项目 | 说明 |
|---|---|
| API | SiliconFlow (`SILICONFLOW_API_KEY`) |
| 嵌入模型 | `Qwen/Qwen3-VL-Embedding-8B` |
| 视觉模型 | `Qwen/Qwen3-VL-32B-Instruct` |

默认模型在 `meme_vault/config.py` 中配置，可在 `MemeVault()` 初始化时覆盖。

## 目录结构

```
meme_vault/
  vault.py       — 核心 API
  client.py      — SiliconFlow API 客户端
  metadata.py    — 元数据模型
  embedding.py   — 嵌入构建与检索
  cli.py         — 命令行入口
  config.py      — 默认配置
data/             — metadata.json, embeddings.npy（已 gitignore）
images/           — 表情包图片文件
```
