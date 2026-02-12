# Quant Python AI

量化投資研究 AI Agent — 透過 CLI 互動介面，自動搜尋財經新聞、分析市場情緒、產生風險評估報告。

## 功能

- **新聞研究** — 透過 Tavily API 即時搜尋財經新聞與財務報告
- **LLM 分析** — 將研究資料餵入 LLM（預設 GPT-4o-mini），產出財務摘要與市場情緒判斷
- **風險審查** — LLM 扮演風險管理專家，評估潛在風險並提供建議
- **多模型切換** — 支援 OpenAI / Anthropic / Google / DeepSeek 等模型，CLI 中即時切換
- **Rich CLI** — 美觀的終端介面，含 spinner、表格、Markdown 報告

## 專案結構

```
quant-python-ai/
├── main.py                          # CLI 入口
├── agent/
│   ├── quant_python_agent.py        # 主 Agent（Pipeline 調度）
│   ├── llm.py                       # LLM Client（OpenAI wrapper）
│   ├── tools.py                     # 工具（Tavily, TwStock, Backtrader）
│   ├── scratchpad.py                # Agent 記憶暫存區
│   └── base_agent.py               # Agent 基底類別
├── pyproject.toml
├── env.example
└── .env                             # 你的 API Keys（不進版控）
```

## 安裝

### 前置需求

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)（推薦）或 pip

### 步驟

```bash
# 1. Clone 專案
git clone <repo-url>
cd quant-python-ai

# 2. 建立環境並安裝依賴
uv sync

# 3. 設定 API Keys
cp env.example .env
# 編輯 .env，填入你的 API Keys（至少需要以下兩組）：
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...
```

如果不用 uv，也可以：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 取得 API Keys

| 服務 | 用途 | 申請連結 |
|------|------|----------|
| OpenAI | LLM 分析（必要） | https://platform.openai.com/api-keys |
| Tavily | 新聞搜尋（必要） | https://tavily.com |

## 使用方式

```bash
# 啟動 CLI
uv run python main.py
```

進入互動介面後，直接輸入研究任務：

```
>> 給我台積電一月的財務報告 並判斷市場情緒
>> 比較 台積電 和 聯發科 的財務指標 誰更有可能具備更多的漲勢
```

### CLI 指令

| 指令 | 說明 |
|------|------|
| `/models` | 列出所有可用模型 |
| `/model <id>` | 切換 LLM 模型（例如 `/model gpt-4o`） |
| `/help` | 顯示幫助資訊 |
| `/quit` | 離開程式 |

### Pipeline 流程

```
使用者輸入
   ↓
1. 規劃 — 拆解研究步驟
   ↓
2. 研究 — Tavily 搜尋即時新聞與財報
   ↓
3. 分析 — LLM 彙整資料，產出財務摘要 + 市場情緒
   ↓
4. 審查 — LLM 風險管理專家審閱，輸出風險評估
   ↓
Rich 格式化報告輸出
```

## 授權

MIT License
