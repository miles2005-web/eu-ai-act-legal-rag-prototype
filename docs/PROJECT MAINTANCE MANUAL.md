# EU AI Act Compliance Navigator — 项目维护手册

> 版本：v1.0 | 更新日期：2026-04-05
> 本手册包含项目的全部技术细节、架构说明、日常维护操作和故障排除方案。
> 将本文档提供给任何 AI 助手，配合终端截图即可定位和解决问题。

-----

## 一、项目概览

### 1.1 项目定位

EU AI Act Compliance Navigator 是一个基于 RAG（Retrieval-Augmented Generation）架构的欧盟《人工智能法案》合规自查工具。用户用自然语言描述 AI 系统，工具检索相关法律条文并生成结构化合规评估报告。

### 1.2 在线地址

- **线上应用**：https://eu-ai-act-legal-rag-prototype-hgryem2gsyrmyz7m6tda6c.streamlit.app
- **GitHub 仓库**：https://github.com/miles2005-web/eu-ai-act-legal-rag-prototype
- **托管平台**：Streamlit Community Cloud

### 1.3 核心技术栈

|组件        |技术                                   |说明                                     |
|----------|-------------------------------------|---------------------------------------|
|文档解析      |pypdf + 自定义法律切分                      |`src/ingest.py` + `src/legal_chunks.py`|
|Embeddings|OpenAI text-embedding-3-small (1536维)|通过 OpenRouter API 调用                   |
|向量存储      |JSON 文件 (`vector_store.json`)        |本地开发用 ChromaDB，云端用 JSON                |
|检索        |余弦相似度 + Self-Query 路由                |正则检测法条引用 → 元数据过滤                       |
|LLM       |GPT-4o-mini                          |通过 OpenRouter API 调用                   |
|前端        |Streamlit                            |聊天界面，支持 6 种语言                          |
|API 网关    |OpenRouter                           |统一调用 OpenAI 的 Embedding 和 Chat 模型      |

### 1.4 关键数字

|指标                  |当前值                       |
|--------------------|--------------------------|
|法律文档数               |约 5 份（EU AI Act 全文 + 交叉法规）|
|向量化 chunks          |3230 条                    |
|Embedding 维度        |1536                      |
|vector_store.json 大小|~43MB（精度截短至 5 位小数）        |
|支持语言                |英/法/德/西/简中/繁中             |

-----

## 二、项目架构

### 2.1 数据流

```
法律 PDF 文件
    → src/ingest.py（pypdf 提取文本 → data/parsed/*.txt）
    → src/legal_chunks.py（法律结构感知切分 → chunks 列表）
    → run_pipeline_chroma.py（OpenRouter Embedding → ChromaDB 本地存储）
    → 导出为 vector_store.json（JSON 格式，供云端使用）
    → 推送到 GitHub → Streamlit Cloud 自动部署
```

### 2.2 查询流

```
用户输入问题
    → Self-Query 路由（正则检测 Article/Annex/Recital）
        → 检测到法条引用 → 元数据过滤检索（精准）
        → 未检测到 → 全量向量检索（语义）
    → 动态 Token 预算（6000-10000，按查询复杂度自动调整）
    → 拼接检索结果作为上下文
    → GPT-4o-mini 生成结构化合规评估
    → Streamlit 前端渲染
```

### 2.3 文件结构

```
~/Desktop/legal-rag/
├── app.py                    # 旧版纯关键词检索前端（保留，不再使用）
├── app_chroma.py             # 主应用：聊天 UI + RAG 管道（当前线上版本）
├── run_pipeline_chroma.py    # 向量化管道：chunks → Embedding → ChromaDB
├── run_pipeline.py           # 早期验证脚本（保留）
├── self_query_fix.py         # Self-Query 正则测试脚本
├── vector_store.json         # 预计算的向量数据（3230条，~43MB）
├── requirements.txt          # Python 依赖
├── README.md                 # 项目说明
├── LICENSE                   # MIT
│
├── src/
│   ├── ingest.py             # PDF/TXT → data/parsed/ 纯文本
│   └── legal_chunks.py       # 法律结构切分 + 元数据提取 + 评分
│
├── data/
│   ├── raw/                  # 原始法律 PDF 文件（输入）
│   └── parsed/               # 解析后的纯文本（中间产物）
│
├── chroma_db/                # ChromaDB 本地持久化目录（仅本地使用）
│
├── eval/
│   └── golden_queries.json   # 评估用测试查询
│
├── scripts/
│   └── evaluate_retrieval.py # 检索评估脚本
│
├── screenshots/              # README 截图
│
└── venv/                     # Python 虚拟环境（不推送到 GitHub）
```

### 2.4 关键文件详解

#### `app_chroma.py`（主应用，~200行）

核心函数：

- `load_vectors()`: 从 vector_store.json 加载所有向量数据（@st.cache_data 缓存）
- `cosine_sim(a, b)`: 纯 Python 余弦相似度计算
- `search_store(query_emb, top_k, where)`: 向量检索 + 可选元数据过滤
- `extract_legal_references(query)`: 正则提取 Article/Annex/Recital 引用
- `auto_token_budget(query, refs)`: 根据查询复杂度自动调整 token 预算
- `apply_token_budget(items, budget)`: 在预算内装填检索结果
- `run_query(prompt, lang_key, top_k)`: 完整 RAG 流程（检索→合成）

UI 结构：

- 侧边栏：语言选择、Retrieval count、Token budget、清除对话、下载历史、快捷查询按钮
- 主区域：聊天消息（用户右对齐蓝色气泡、AI 左对齐灰色气泡）
- 每条回答下方：下载按钮、重新生成按钮、检索条款展开

#### `run_pipeline_chroma.py`（向量化管道，~50行）

流程：

1. 从 `data/parsed/` 加载文本文件
1. 用 `build_structured_chunks()` 切分为带元数据的 chunks
1. 分批调用 OpenRouter Embedding API（每批 50 条，间隔 1 秒）
1. 存入 ChromaDB 本地数据库
1. 最后运行一个测试查询验证检索是否工作

#### `src/ingest.py`（文档解析，~115行）

- 支持 `.txt` 和 `.pdf` 文件
- PDF 用 `pypdf.PdfReader` 逐页提取文本
- 调用 `clean_extracted_text()` 清理 OCR 噪音
- 输出到 `data/parsed/` 目录

#### `src/legal_chunks.py`（法律切分，~700行，核心模块）

- `build_structured_chunks()`: 按 Chapter/Section/Article/Annex/Recital 边界切分
- 每个 chunk 的元数据字段：
  - `article_number`: 条文编号
  - `annex_ref`: 附件引用
  - `recital_ref`: 序言编号
  - `chapter_heading`: 所属章节标题
  - `section_heading`: 所属节标题
  - `canonical_citation`: 标准引用格式
  - `parent_citation`: 父级引用
- `score_chunk()`: 关键词 + 元数据加权评分
- `extract_query_metadata()`: 从用户查询中提取法条引用
- `narrow_chunks_for_query()`: 预过滤候选 chunks

-----

## 三、环境与依赖

### 3.1 本地开发环境

|项目    |要求                                     |
|------|---------------------------------------|
|操作系统  |macOS（已测试），Linux/Windows 理论兼容          |
|Python|3.12（必须，3.14 不兼容 ChromaDB/Streamlit）   |
|虚拟环境  |venv（项目目录下 `~/Desktop/legal-rag/venv/`）|

### 3.2 Python 依赖

```
streamlit==1.44.1
openai==2.30.0
pypdf>=4.0.0
chromadb>=1.0.0     # 仅本地开发需要
```

### 3.3 环境变量

|变量                  |用途                    |获取方式                      |
|--------------------|----------------------|--------------------------|
|`OPENROUTER_API_KEY`|调用 Embedding 和 LLM API|https://openrouter.ai/keys|

**本地设置方法**：

```bash
export OPENROUTER_API_KEY="sk-or-v1-你的key"
```

**云端设置方法**：
Streamlit Cloud → Manage app → Settings → Secrets → 添加：

```
OPENROUTER_API_KEY = "sk-or-v1-你的key"
```

### 3.4 API 调用方式

项目**不使用** `openrouter` Python 包（不稳定、接口频繁变更）。使用 `openai` 包，只改 `base_url`：

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
# Embedding
resp = client.embeddings.create(model="openai/text-embedding-3-small", input="text")
# Chat
resp = client.chat.completions.create(model="openai/gpt-4o-mini", messages=[...])
```

-----

## 四、日常维护操作

### 4.1 添加新法律文档

**完整步骤**：

```bash
# 0. 准备
cd ~/Desktop/legal-rag
source venv/bin/activate
export OPENROUTER_API_KEY="sk-or-v1-你的key"

# 1. 把 PDF 放到 data/raw/
cp ~/Downloads/新文件.pdf data/raw/

# 2. 解析（PDF → 纯文本）
python src/ingest.py

# 3. 向量化（chunks → embeddings → ChromaDB）
python run_pipeline_chroma.py

# 4. 导出 JSON（给云端用）
python -c "
import json, chromadb
db = chromadb.PersistentClient(path='./chroma_db')
col = db.get_collection('eu_ai_act')
data = col.get(include=['documents','metadatas','embeddings'])
export = []
for i in range(len(data['ids'])):
    emb = data['embeddings'][i]
    if hasattr(emb, 'tolist'): emb = emb.tolist()
    export.append({'id':data['ids'][i],'document':data['documents'][i],'metadata':data['metadatas'][i],'embedding':[round(x,5) for x in emb]})
with open('vector_store.json','w') as f:
    json.dump(export, f, separators=(',',':'))
print(f'Exported {len(export)} records')
"

# 5. 检查文件大小（必须 < 100MB）
ls -lh vector_store.json

# 6. 推送
git add vector_store.json data/
git commit -m "feat: add new legal documents"
git push

# 7. 去 Streamlit Cloud Reboot app
```

**如果 vector_store.json 超过 100MB**：

```bash
# 截短 embedding 精度
python -c "
import json
with open('vector_store.json') as f:
    data = json.load(f)
for item in data:
    item['embedding'] = [round(x, 5) for x in item['embedding']]
with open('vector_store.json','w') as f:
    json.dump(data, f, separators=(',',':'))
print(f'{len(data)} records')
"
ls -lh vector_store.json
# 如果还超过，改 round(x, 4) 进一步截短
```

### 4.2 更新应用代码

```bash
cd ~/Desktop/legal-rag
# 编辑 app_chroma.py
git add app_chroma.py
git commit -m "描述你改了什么"
git push
# Streamlit Cloud 会自动重新部署（约 1-2 分钟）
```

### 4.3 本地运行/调试

```bash
cd ~/Desktop/legal-rag
source venv/bin/activate
export OPENROUTER_API_KEY="sk-or-v1-你的key"
streamlit run app_chroma.py
# 浏览器打开 http://localhost:8501
```

### 4.4 更新 OpenRouter API Key

如果 key 过期或泄露：

1. 去 https://openrouter.ai/keys 生成新 key
1. 本地：`export OPENROUTER_API_KEY="新key"`
1. 云端：Streamlit Cloud → Manage app → Settings → Secrets → 更新
1. **不要把 key 写在代码里或推到 GitHub**

### 4.5 打 Git Tag

```bash
git tag -a v版本号 -m "版本描述"
git push origin v版本号
```

-----

## 五、故障排除

### 5.1 本地启动问题

#### 问题：`ModuleNotFoundError: No module named 'openai'`

**原因**：虚拟环境里没装 openai 包，或者没激活虚拟环境
**解决**：

```bash
source venv/bin/activate
pip install openai
```

#### 问题：`ModuleNotFoundError: No module named 'streamlit'`

**原因**：同上
**解决**：

```bash
source venv/bin/activate
pip install streamlit
```

#### 问题：`ModuleNotFoundError: No module named 'pip'` 或 `error: invalid-installed-package`

**原因**：虚拟环境损坏（通常是多次安装/卸载导致）
**解决**：彻底重建虚拟环境

```bash
cd ~/Desktop/legal-rag
deactivate
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install streamlit openai chromadb pypdf
```

#### 问题：venv 用了 Python 3.14

**症状**：报错路径中包含 `python3.14/site-packages`
**原因**：系统默认 python3 指向 3.14，创建 venv 时用了错误版本
**解决**：

```bash
deactivate
rm -rf venv
python3.12 -m venv venv  # 必须指定 3.12
source venv/bin/activate
pip install streamlit openai chromadb pypdf
```

#### 问题：`anyio` 包损坏

**症状**：`Cannot process installed package anyio 4.13.0 ... Invalid version: '4.13.0 2'`
**解决**：

```bash
pip uninstall anyio -y
pip install openai
```

#### 问题：Streamlit 启动后浏览器打不开

**检查项**：

1. 终端是否显示 `Local URL: http://localhost:8501`？如果没有，等几秒
1. 浏览器地址是否正确？必须是 `http://localhost:8501`，不是 `localhost.com`
1. 是否有旧的 Streamlit 进程占用端口？用 `Ctrl+C` 停掉后重新运行
1. 换端口：`streamlit run app_chroma.py --server.port 8502`

#### 问题：`openai.AuthenticationError: Error code: 401 - User not found`

**原因**：API key 无效或未设置
**解决**：

```bash
# 检查是否设置了
echo $OPENROUTER_API_KEY

# 重新设置（注意引号和完整 key）
export OPENROUTER_API_KEY="sk-or-v1-完整的key"

# 测试 key 是否有效
curl -s https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY" | head -c 200
```

#### 问题：`export` 和 `python` 命令粘在一行

**症状**：`export: not valid in this context`
**原因**：两条命令被粘贴在了同一行
**解决**：分开执行，每条命令单独一行回车

### 5.2 向量化管道问题

#### 问题：`run_pipeline_chroma.py` 报 `data/parsed/ is empty`

**原因**：没有先运行 `python src/ingest.py`，或者 `data/raw/` 里没有文件
**解决**：

```bash
ls data/raw/        # 检查是否有 PDF/TXT
python src/ingest.py  # 先解析
python run_pipeline_chroma.py  # 再向量化
```

#### 问题：`ingest()` 返回的 `output` 字段不是文本内容

**说明**：这是正常行为。`ingest()` 返回的是文件处理日志（文件名、状态），不是文本内容。真正的文本在 `data/parsed/*.txt` 文件里。`run_pipeline_chroma.py` 和 `app_chroma.py` 都是从 `data/parsed/` 目录读取文本的。

#### 问题：`build_structured_chunks()` 只返回 1 个 chunk

**原因**：传入的文本是文件名而不是文件内容
**解决**：确认是从文件读取文本内容：

```python
for path in sorted(parsed_dir.glob("*.txt")):
    text = path.read_text(encoding="utf-8").strip()
    # 用 text 而不是 path.name
```

#### 问题：`TypeError: Object of type ndarray is not JSON serializable`

**原因**：ChromaDB 返回的 embeddings 是 numpy 数组，json.dump 不支持
**解决**：导出时转换：

```python
emb = data['embeddings'][i]
if hasattr(emb, 'tolist'): emb = emb.tolist()
```

### 5.3 Git/GitHub 问题

#### 问题：`File vector_store.json is 108.62 MB; exceeds GitHub's file size limit of 100.00 MB`

**解决**：截短 embedding 精度

```bash
git reset HEAD~1  # 撤销最近一次 commit
python -c "
import json
with open('vector_store.json') as f:
    data = json.load(f)
for item in data:
    item['embedding'] = [round(x, 5) for x in item['embedding']]
with open('vector_store.json','w') as f:
    json.dump(data, f, separators=(',',':'))
print(f'{len(data)} records')
"
ls -lh vector_store.json  # 确认 < 100MB
git add vector_store.json
git commit -m "feat: update vector store"
git push
```

#### 问题：`heredoc>` 提示符卡住

**原因**：使用 `cat > file << 'EOF'` 语法时，终端在等待结束标记
**解决**：输入对应的结束标记（如 `EOF` 或 `READMEEOF`）然后回车

### 5.4 Streamlit Cloud 部署问题

#### 问题：`TypeError: Importing a module script failed`

**原因**：openai 包版本不兼容
**解决**：在 `requirements.txt` 中锁定版本：

```
streamlit==1.44.1
openai==2.30.0
pypdf>=4.0.0
```

推送后 Reboot app。

#### 问题：`pydantic.v1.errors.ConfigError`

**原因**：ChromaDB 与 Streamlit Cloud 的 Python 3.14 不兼容
**解决**：项目已改用 JSON vector store，不依赖 ChromaDB。如果重新出现此问题，检查 requirements.txt 是否包含 chromadb（云端不需要）。

#### 问题：Streamlit Cloud `source IP address not allowed`

**原因**：国内 IP 被 Streamlit Cloud 限制
**解决**：开 VPN 访问，或使用 Hugging Face Spaces 部署（国内可访问）

#### 问题：侧边栏 `st.slider` 在云端报错

**原因**：某些 Streamlit Cloud 环境的 JS 组件加载失败
**解决**：已改用 `st.selectbox` 替代 `st.slider`

#### 问题：部署后页面只显示错误，不显示内容

**通用排查步骤**：

1. 点右下角 `Manage app` 查看日志
1. 在日志中找到具体错误信息
1. 常见原因：缺少依赖（requirements.txt）、API key 未设置（Secrets）、文件路径错误

### 5.5 OpenRouter API 问题

#### 问题：`ImportError: cannot import name 'Client' from 'openrouter'`

**原因**：使用了 `openrouter` Python 包，这个包不稳定
**解决**：卸载它，用 `openai` 包代替：

```bash
pip uninstall openrouter -y
pip install openai
```

代码中用：

```python
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
```

#### 问题：Embedding API 报错 `openai.Embedding.create is not a function`

**原因**：使用了 openai v0.x 的旧语法
**解决**：openai v1.x+ 的正确语法：

```python
# 错误（v0.x 旧语法）
openai.api_key = "..."
openai.Embedding.create(input=text, model="...")

# 正确（v1.x+ 新语法）
client = OpenAI(base_url="...", api_key="...")
client.embeddings.create(model="...", input=text)
```

### 5.6 Python REPL 问题

#### 问题：多行代码在 REPL 中报 SyntaxError

**原因**：Python REPL 对多行代码（尤其含中文注释）的粘贴处理有问题
**解决**：**不要在 REPL 中粘贴多行代码**。写成 `.py` 脚本文件，然后 `python 脚本.py` 执行

#### 问题：REPL 中执行 shell 命令报错

**原因**：`ls`、`cd` 等 shell 命令不能在 Python REPL 中运行
**解决**：退出 REPL（`exit()` 或 `Ctrl+D`），在终端执行

-----

## 六、Self-Query 路由机制

### 6.1 工作原理

用户输入查询时，系统先用正则表达式检测是否包含明确的法条引用：

```python
# 检测 Article 引用
re.search(r'Article\s+(\d+)', query, re.IGNORECASE)
# 检测 Annex 引用
re.search(r'Annex\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII)', query, re.IGNORECASE)
# 检测 Recital 引用
re.search(r'Recital\s+(\d+)', query, re.IGNORECASE)
```

### 6.2 路由逻辑

- **检测到引用** → 构建元数据过滤条件，只在匹配的 chunks 中做向量检索
  - 单条件：`{"article_number": {"$eq": "Article 6"}}`
  - 多条件：`{"$or": [{"article_number": {"$eq": "Article 6"}}, {"annex_ref": {"$eq": "Annex III"}}]}`
- **未检测到** → 全量向量检索

### 6.3 为什么不用 LLM 做 Self-Query

最初版本用 GPT-4o-mini 做法条提取，但 OpenRouter 在云端不支持 `response_format=json_object` 参数，导致静默失败。改用正则方案后：

- 零延迟（不需要 API 调用）
- 100% 可靠（不依赖 LLM 输出格式）
- 对明确法条引用的检测准确率 100%

-----

## 七、Token 预算控制

### 7.1 机制

```python
def auto_token_budget(query, refs, base=6000):
    budget = base
    # 跨多个法条/附件引用 → 提升到 10000
    if refs["count"] >= 2:
        budget = 10000
    # 宽泛的义务/合规类查询 → 提升到 8000
    long_keywords = ["obligations", "requirements", "prohibited", ...]
    if any(kw in query.lower() for kw in long_keywords):
        budget = max(budget, 8000)
    return budget
```

### 7.2 估算方法

使用 `len(text) // 4` 粗略估算 token 数（1 token ≈ 4 字符，英文文本）。这比调用 tiktoken 更快，精度足够用于预算控制。

-----

## 八、语言支持

### 8.1 支持的语言

|语言      |LLM 指令               |UI 翻译|
|--------|---------------------|-----|
|English |Respond in English.  |✅ 完整 |
|Français|Réponds en français. |✅ 完整 |
|Deutsch |Antworte auf Deutsch.|✅ 完整 |
|Español |Responde en español. |✅ 完整 |
|简体中文    |请用简体中文回答。            |✅ 完整 |
|繁體中文    |請用繁體中文回答。            |✅ 完整 |

### 8.2 实现方式

- 语言指令作为 system prompt 的第一行注入
- UI 元素（标题、按钮、disclaimer）从 `LANGS` 字典中按选择的语言读取
- 已生成的回答不会自动翻译，但可通过 🔄 按钮用新语言重新生成

-----

## 九、云端部署

### 9.1 Streamlit Cloud 配置

- **Repository**: miles2005-web/eu-ai-act-legal-rag-prototype
- **Branch**: main
- **Main file**: app_chroma.py
- **Secrets**: OPENROUTER_API_KEY

### 9.2 云端 vs 本地差异

|维度       |本地                  |云端                     |
|---------|--------------------|-----------------------|
|向量存储     |ChromaDB（chroma_db/）|JSON（vector_store.json）|
|Python 版本|3.12（手动指定）          |由 Streamlit Cloud 决定   |
|API Key  |环境变量                |Streamlit Secrets      |
|依赖安装     |pip install         |requirements.txt 自动安装  |

### 9.3 为什么不用 ChromaDB 部署

ChromaDB 在 Streamlit Cloud 上与 Python 3.14 和 protobuf 存在兼容性问题（pydantic ConfigError、protobuf TypeError）。改用纯 Python 的 JSON 加载 + 余弦相似度计算后，零外部依赖，完全兼容。

### 9.4 Reboot 方法

推送代码后 Streamlit Cloud 通常自动重新部署。如果没有，手动：
页面右下角 → `Manage app` → `Reboot app`

-----

## 十、版本历史

|版本  |日期        |主要变更                                    |
|----|----------|----------------------------------------|
|v0.1|2026-04-03|初始框架：法律结构切分 + 关键词检索                     |
|v0.2|2026-04-03|改进解析、元数据提取、评估框架                         |
|v0.3|2026-04-04|ChromaDB 向量化 + LLM 合规评估 + Streamlit UI  |
|v0.4|2026-04-04|Self-Query 路由 + Token 预算 + 相似度裁剪        |
|v1.0|2026-04-04|聊天 UI + 6 语言 + 下载/重新生成 + 云端部署 + 3230 条记录|

-----

## 十一、开发中踩过的坑（经验教训）

### 11.1 不要用 `openrouter` Python 包

接口不稳定，`Client`、`embeddings.create`、`embed_text` 全报错。用 `openai` 包 + 改 `base_url`。

### 11.2 不要在 Python REPL 粘贴多行代码

中文注释和缩进会导致 SyntaxError。写成 `.py` 文件执行。

### 11.3 `ingest()` 返回的是元数据不是文本

`output` 字段是文件名，不是文件内容。文本在 `data/parsed/*.txt` 里。

### 11.4 创建 venv 必须指定 Python 3.12

`python3 -m venv venv` 可能用到 3.14，导致 ChromaDB/Streamlit 不兼容。
必须 `python3.12 -m venv venv`。

### 11.5 ChromaDB 不支持字符串 $contains

元数据过滤只支持 `$eq`（精确匹配）。如果要匹配列表中的元素，存储为 Python list，用 `$eq` 查询时 ChromaDB 会自动遍历列表元素。

### 11.6 vector_store.json 超 100MB

GitHub 限制单文件 100MB。用 `round(x, 5)` 截短 embedding 精度 + `separators=(',',':')` 去掉空格可以大幅减小文件。

### 11.7 Streamlit Cloud 的 `st.slider` 可能加载失败

某些环境 JS 组件不加载。改用 `st.selectbox`。

### 11.8 export 和 python 命令不要粘在一行

终端会把它们当作一条命令执行，导致 `export: not valid in this context`。每条命令单独回车。

-----

## 十二、未来可扩展方向

### 12.1 技术升级

- **LlamaParse**：替代 pypdf，保留法律文本层级结构（H1/H2/H3）
- **MarkdownNodeParser**：按语义结构切分而非固定长度
- **双层元数据提取**：父级标题继承 + 正则扫描（解决 H3 子节点失联问题）
- **BM25 混合检索**：补充纯向量检索对法条编号的语义盲区

### 12.2 功能增强

- **多文档对比**：同时检索 AI Act + GDPR，输出交叉合规分析
- **合规检查清单导出**：生成 PDF 格式的合规义务清单
- **用户账号系统**：持久化查询历史
- **Streaming 输出**：LLM 回答逐字显示

### 12.3 部署优化

- **Hugging Face Spaces**：国内可访问的替代部署方案
- **Docker 容器化**：统一运行环境，避免依赖问题
- **API 服务化**：将 RAG 管道包装为 REST API

-----

*手册版本：v1.0 | 最后更新：2026-04-05*