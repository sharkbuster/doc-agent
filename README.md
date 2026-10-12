# 拾光优选 · 售后客服智能体

[![CI](https://github.com/sharkbuster/doc-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sharkbuster/doc-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)

一个电商平台的售后客服智能体：收到问题后先去知识库里检索平台政策条款，需要订单数据时调用工具去查，
最后基于**条款原文和工具算出来的数字**生成回答，并标出依据的是哪一条。

**克隆下来不配任何密钥就能跑。** 模型层是可插拔的，默认是一个离线的确定性桩模型，
单元测试、端到端自检、指标评测全都不需要网络和 API key。

---

## 快速开始

```bash
git clone https://github.com/sharkbuster/doc-agent.git
cd doc-agent

python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest -q               # 单元测试
python -m scripts.selfcheck       # 端到端自检
python -m scripts.seed            # 生成演示订单数据
python -m uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000> 就是聊天页面。

想换成真实模型：把 `.env.example` 复制成 `.env`，改成

```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxxx
```

DeepSeek 是 OpenAI 兼容接口，不需要额外写适配层。任何 OpenAI 兼容的网关
（官方 API、硅基流动、本地 vLLM）填 `OPENAI_*` 三项也能直接用。

---

## 它能做什么

| 问题类型 | 例子 | 系统怎么处理 |
|---|---|---|
| 政策咨询 | 「7 天无理由退货的期限是几天？」 | 检索条款，带引用回答 |
| 订单数据 | 「SO20261001001 无理由退货能退多少钱？」 | 调 `calc_refund` 算，金额不由模型估 |
| 混合 | 「这个冰箱签收 8 天了还有质量问题，能退吗？」 | 先查订单签收时间，再套大家电 30 天规则 |
| 知识库外 | 「怎么申请提高信用卡额度？」 | 拒答，建议转人工 |

**它不做什么**：不承诺政策之外的东西，不在没有依据时硬编答案，不把别的用户的订单信息说出去。

---

## 架构

```
用户提问
   │
   ├─► 检索       中文感知切分 → 字符 n-gram 哈希向量 + BM25 → RRF 融合 → Top-5 条款
   │
   ├─► 组装提示   条款正文 + 引用编号规则 + 拒答规则（命不中时不塞条款）
   │
   ├─► 模型循环   最多 4 轮
   │      ├─ 模型要求调用工具 → 执行 → 结果回灌 → 继续
   │      └─ 模型给出回答 → 结束
   │
   └─► 引用校验   越界编号直接从回答里抹掉 → 返回回答 + 引用 + 工具轨迹
```

前端把每一步都展示出来：检索到哪几条、相似度多少、调了什么工具、参数是什么、
工具返回了什么。这一层是这个项目刻意做的「白箱」——客服系统出错的时候，
你需要知道它到底依据了什么。

---

## 目录结构

```
app/
  kb/          知识库：切分、分词、向量化、索引、检索
    chunker.py   中文感知切分，保留标题路径做引用
    text.py      分词（全角归一 + 汉字单字 + 英文词 + n-gram）
    embedder.py  字符 n-gram 哈希向量（默认）/ API embeddings（可选）
    index.py     FAISS 索引，装不上时退回 numpy 暴力检索
    retriever.py BM25 + 稠密向量 + RRF 融合 + 拒答判据
  agent/       智能体
    llm.py       模型层：离线桩 / DeepSeek / 任意 OpenAI 兼容网关
    tools.py     六个工具，返回值是 JSON，业务错误不抛异常
    loop.py      多轮循环、步数上限、工具异常回灌、引用校验
    prompts.py   提示词模板
  policy.py    售后规则计算（日期窗口、优惠券分摊、退款金额、赔付）
  store/       订单/物流/库存（SQLite）
  api/         HTTP 接口（含 SSE 流式）
  static/      前端（原生 HTML/CSS/JS，无构建步骤）
data/
  corpus/      知识库语料：8 篇虚构的售后政策文档
  eval/        评测集与评测结果
scripts/
  seed.py      播种演示订单数据
  evaluate.py  跑评测集，输出检索 / 工具选择 / 拒答指标
  selfcheck.py 端到端自检，一条命令确认项目是活的
```

---

## 评测结果

<!--METRICS-->

---

## 几个刻意的设计取舍

**能算的绝不让模型算。** 退款金额、剩余天数、赔付额度全部由 `policy.py` 算完再交给模型转述。
让大模型做算术是这类系统最稳定的错误来源：它算错了不会报错，会一本正经地给出一个错误数字。

**不用 `sentence-transformers`。** 上 bge 之类的模型会让克隆者拖进 2GB 级别的 torch 依赖，
CI 从一分钟变十分钟，大多数人到这一步就放弃了。所以内置了一个字符 n-gram 哈希向量
（纯 numpy，零下载），并保留 `EMBEDDING_BACKEND=api` 给需要更好效果的人。
代价是召回率有上限，这个上限在评测集里是量化过的。

**不用 LangChain 的 `RecursiveCharacterTextSplitter`。** 它的默认分隔符是
`["\n\n", "\n", " ", ""]`，中文没有词间空格，长段落会一路退化到逐字切分。
更关键的是它不保留标题路径，而「依据的是哪一条政策的哪一小节」正是引用溯源要用的信息，
切完再回头补是补不回来的。

**接口都写成同步 `def`。** 模型调用和 SQLite 查询都是阻塞操作，放在 `async def` 里会卡住事件循环。
交给 FastAPI 自动丢线程池更简单，也省掉了 `pytest-asyncio` 这层依赖。
代价是 SQLite 连接必须允许跨线程使用（见「踩过的坑」）。

**拒答判据用稠密余弦的绝对值，排序用 RRF 融合分。** 融合分是名次的函数，
检索质量整体很差时它照样能给出很高的值；余弦才是有绝对意义的相似度，才配用来判断
「知识库里到底有没有」。

**索引不落盘。** 几十个片段重建一次是毫秒级的事，而持久化要额外处理
「语料改了但索引没重建」的一致性问题，收益不抵麻烦。

---

## 踩过的坑

这些都是开发过程中真实出现并被测试固定住的问题，不是假想的。

| 问题 | 现象 | 原因与修法 |
|---|---|---|
| 重叠静默失效 | 相邻片段完全不重叠 | 超长段落先「按 max_chars 预分组」，每组都贴着上限，重叠尾巴永远塞不进下一块。改成句级单元后打包器才有余量留重叠 |
| 空查询返回一堆来源 | 输入纯标点，返回 5 条「相关条款」 | 全零查询向量和每个片段的余弦都是 0，排序退化成按下标。现在查不出词就直接返回空，且余弦 ≤ 0 的片段不算候选 |
| SQLite 跨线程崩 | 接口一调工具就 `ProgrammingError` | 同步 `def` 由线程池执行，连接却是主线程建的。改为允许跨线程（本应用数据库只读，写库时需另想办法） |
| 日期窗口差一天 | 「签收第 7 天下午」被判超期 | 用精确 24 小时相减，而政策写的是「自然日」。改成取日期差 |
| 过期订单照样算退款 | 已超期的订单返回了退款金额 | 模型很可能照抄这个数字。现在窗口外、未签收、未发货一律只给解释不给金额 |
| 假模型抄了提示词 | 回答里引用 `[2]`，指向的却是提示词里的示例 | 兜底逻辑只认 `[数字]`，先命中了「例如 [1][2]」那句示例。改成必须匹配「来源：」标记 |
| 拒答误判 | 正常回答被判成拒答 | 拒答判据原本是关键词匹配，而提示词自己就写着「查不到」三个字。改成结构性判据：没检索到 + 没调工具 |
| 优惠券分摊对不上账 | 退款明细加起来 ≠ 订单实付 | 四舍五入的余数没人吃。改成最后一件商品吃掉余数，保证分摊之和精确等于券面金额 |
| GBK 控制台崩溃 | 自检全部通过，退出码却是 1 | 输出里的 `✓` 在中文 Windows 的 GBK 控制台上编码不了，`UnicodeEncodeError` 把整个脚本带崩 |
| 工具报错就 500 | 订单号不存在时整个请求失败 | 业务错误当异常抛了。现在订单不存在、参数不合法都作为正常返回值交回模型，让它自己组织语言 |

---

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `stub` | `stub`（离线桩）/ `deepseek` / `openai` |
| `DEEPSEEK_API_KEY` | 空 | 用 DeepSeek 时必填 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | |
| `DEEPSEEK_MODEL` | `deepseek-chat` | |
| `OPENAI_API_KEY` / `_BASE_URL` / `_MODEL` | 空 | 任意 OpenAI 兼容网关 |
| `EMBEDDING_BACKEND` | `hashing` | `hashing`（内置，零下载）/ `api` |
| `EMBEDDING_DIM` | `1024` | 哈希向量的维度 |
| `RETRIEVAL_TOP_K` | `5` | 送进提示词的条款条数 |
| `REFUSAL_THRESHOLD` | `0.35` | 最高余弦分低于它就判定知识库里没有 |
| `DB_PATH` | `data/demo.db` | 订单库位置 |

> DeepSeek 没有 embeddings 接口。`EMBEDDING_BACKEND=api` 需要另找一家提供
> `/embeddings` 的服务，别把 DeepSeek 的 key 填在 `EMBEDDING_API_KEY` 里。

---

## 待办

- [ ] 用真实模型跑一遍评测，把工具选择准确率和回答质量纳入门禁（当前 CI 门禁只覆盖检索与拒答）
- [ ] 检索换成真嵌入模型，量化一下比字符 n-gram 好在哪、值不值那个依赖
- [ ] 多轮对话里的指代消解：「那能退吗」目前要用户自己带上订单号
- [ ] 语料扩充到多平台多品类，现在只有 8 篇文档
- [ ] 工具调用加超时和重试预算，目前只靠模型层的 60 秒超时兜着

## License

MIT
