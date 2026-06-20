# MemeVault（表情包语义检索系统）

一个基于语义向量检索的本地表情包管理系统，用于解决“表情包难找、翻图库效率低”的问题。

核心能力：  
> 输入一句自然语言 → 返回最相关的表情包图片

---

# 1. 项目目标

## 目标
构建一个轻量级、本地可运行的表情包检索系统，实现：

- 语义检索（而非关键词匹配）
- 秒级返回最相关表情包
- 支持后续扩展为推荐系统 / 图谱系统 / Bot系统

## 非目标（MVP阶段不做）
- 不做复杂数据库系统
- 不做多模态识别
- 不做在线训练或模型微调
- 不做分布式架构

---

# 2. 使用场景

典型输入：

- “群友又开始逆天发言了”
- “我人麻了”
- “不知道怎么吐槽”
- “震惊但又不想说话”

输出：

- 最匹配的 3~5 张表情包图片

---

# 3. 系统架构（MVP）


用户输入
↓
Embedding模型
↓
向量相似度计算
↓
TopK候选（如20）
↓
返回TopN（如5）图片


---

# 4. 系统分层设计

## 4.1 数据层（Data Layer）

存储所有原始数据：


images/
metadata.json


### metadata结构

meta.json
{
  "id": "546bf5345d5a3b......",
  "path": "images/芙莉莲 无语 凝视.jpg",

  "text": "芙莉莲 无语 嫌弃 看傻子",

  "tags": ["无语", "嫌弃", "看傻子"],

  "character": ["芙莉莲"],

  "emotion": ["无语", "嫌弃"],

  "usage": ["群友发病", "不想说话"]
}
特点
不参与计算
不存embedding
不依赖模型
是唯一“真实数据源”
4.2 特征层（Feature Layer）

将文本转换为向量表示：

text → embedding vector

输出：

embeddings.npy
说明
embedding是缓存，不是资产
可随模型更新重建
与图片文件解耦
4.3 检索层（Retrieval Layer）

核心计算逻辑：

query → embedding
↓
与所有向量计算相似度
↓
返回TopK

算法：

cosine similarity
numpy / faiss（2000规模可直接numpy）
4.4 应用层（Application Layer）

提供使用方式：

CLI
Bot（QQ/Telegram）
Web API（FastAPI，可选）
5. 数据结构设计
5.1 图片文件
images/
  xxx.jpg

不要求命名规范，不做强约束

5.2 metadata.json
meta.json
{
  "id": "546bf5345d5a3b......",
  "path": "images/芙莉莲 无语 凝视.jpg",

  "text": "芙莉莲 无语 嫌弃 看傻子",

  "tags": ["无语", "嫌弃", "看傻子"],

  "character": ["芙莉莲"],

  "emotion": ["无语", "嫌弃"],

  "usage": ["群友发病", "不想说话"]
}
设计原则
ID 独立于文件名
text = 所有语义拼接（不做结构拆分）
后续可扩展为结构化字段
5.3 embeddings.npy

结构：

index → vector

要求：

顺序与 metadata 对齐
可随时重建
不作为长期存储真相
1. 核心流程
6.1 构建流程（build）
读取 metadata
↓
拼接 text
↓
调用 embedding模型
↓
生成向量
↓
保存 embeddings.npy
6.2 检索流程（search）
输入 query
↓
query embedding
↓
计算 cosine similarity
↓
排序
↓
返回 TopK 图片
1. 技术选型（推荐）
Embedding模型
BAAI/bge-large-zh-v1.5
或 Qwen Embedding
或 SiliconFlow API
相似度计算
numpy（推荐MVP）
faiss（规模扩大后）
后端（可选）
FastAPI
或纯 Python CLI
1. MVP实现范围
必须实现
metadata.json
embedding构建脚本
cosine检索
TopK输出
暂不实现
向量数据库
rerank模型
图结构推荐
多模态识别
UUID文件系统
1. 可扩展性设计（关键）

该系统设计预留以下升级路径：

9.1 标签结构化
text → emotion / usage / character
9.2 推荐系统
基于 embedding 相似度 + tag overlap
9.3 图谱系统
图片 → 节点
相似度 → 边
9.4 LLM语义增强
用户输入 → LLM → tags → embedding
9.5 多模态扩展
image → CLIP embedding → 直接搜索图片
10. 设计原则（非常重要）
10.1 数据与计算分离
数据 ≠ 特征 ≠ 索引
10.2 embedding可重建

模型可以随时替换

10.3 ID不依赖文件系统

文件只是存储形式，不是逻辑结构

10.4 metadata是核心资产

所有未来扩展依赖 metadata

11. 性能预期

在 2000 张规模下：

检索速度：< 50ms
embedding构建：< 1s/张（API）
内存占用：极低
无需数据库
12. 项目价值

该系统本质是：

一个“语义驱动的表情包操作系统”

将传统：

手动翻图

升级为：

语义检索 → 自动匹配表达
13. 最终系统形态
输入：
“群友又开始发病”

输出：
- 芙莉莲无语.jpg
- 猫猫震惊.jpg
- 熊猫头沉默.jpg
14. 一句话总结

这是一个基于向量检索的轻量语义表情包系统，核心是 metadata + embedding + cosine search 的三层解耦架构。


---