# 專案重構計畫：Quant Python AI → Agent Native 架構

> 日期：2026-02-19
> 目標：將現有硬編碼 pipeline 重構為 **Agent Native** 架構，
> 並整合 **FinMind API** 作為台灣市場資料工具。

---

## 一、現有架構問題分析

| 問題 | 說明 |
|------|------|
| **固定 Pipeline** | `run_mission` 是硬編碼 4 階段流程（Plan→Research→Analyze→Review），LLM 無法自行決定要呼叫哪些工具 |
| **Agent 基底未實作** | `base_agent.py` 的所有方法均為 `NotImplementedError`，`Researcher`/`Risk_Manager` 實例從未被當作真正的 Agent 使用 |
| **無 Function Calling** | `llm.py` 僅做 `chat(system, user)` 純文字對話，未使用 OpenAI 的 `tools` / `function_calling` 機制 |
| **工具無法動態調用** | Tavily 搜尋是直接 hardcode 呼叫，不是由 LLM 判斷後觸發 |
| **TwStock / Backtrader 空殼** | 兩個類別均為 `pass`，無任何功能 |

---

## 二、Agent Native 重構方案

### 核心概念
讓 LLM 成為真正的 **決策者**：根據使用者的問題，自主決定要呼叫哪些工具、呼叫幾次、以什麼順序執行，並根據回傳結果決定下一步。

### 2.1 工具定義層（Tools as Functions）

將每個工具定義為 OpenAI function calling 的 JSON schema，讓 LLM 可以自主選擇：

| 工具名稱 | 用途 | 來源 |
|----------|------|------|
| `search_news` | 搜尋近期財經新聞 | Tavily API |
| `search_finance` | 搜尋財經資訊 | Tavily API |
| `get_stock_price` | 取得台灣個股歷史股價 | FinMind `TaiwanStockPrice` |
| `get_institutional_investors` | 取得三大法人買賣超 | FinMind `TaiwanStockInstitutionalInvestorsBuySell` |
| `get_margin_trading` | 取得融資融券資料 | FinMind `TaiwanStockMarginPurchaseShortSale` |
| `get_revenue` | 取得月營收資料 | FinMind `TaiwanStockMonthRevenue` |
| `get_financial_statements` | 取得財報（損益表/資產負債表） | FinMind |
| `get_pe_pb_ratio` | 取得本益比/股價淨值比 | FinMind `TaiwanStockPER` |

### 2.2 Agent 迴圈（Agent Loop）

```
使用者提問
    ↓
LLM 規劃（決定需要哪些資料）
    ↓
┌─→ LLM 發出 function_call（選擇工具 + 參數）
│     ↓
│   執行工具，回傳結果
│     ↓
│   LLM 判斷：資料足夠？
│     ├── 否 → 再呼叫更多工具 ─┐
│     └── 是 → 生成分析報告    │
│                               │
└───────────────────────────────┘
    ↓
風險審查 Agent（可選的第二輪 Agent Loop）
    ↓
輸出最終報告 + 來源引用
```

### 2.3 檔案變更清單

#### [MODIFY] `agent/tools.py`
- 重構 `Tavily`，新增 function schema 描述
- 新增 `FinMindTool` 類別，實作上述 FinMind 端點
- 移除空殼 `TwStock`
- 每個工具方法需提供對應的 OpenAI function JSON schema

#### [MODIFY] `agent/llm.py`
- 擴充 `chat()` 方法支援 `tools` 參數（OpenAI function calling）
- 新增 `chat_with_tools()` 方法，處理 tool_calls 迴圈

#### [MODIFY] `agent/base_agent.py`
- 重新設計為真正的 Agent 基底：包含 tool registry、agent loop、max iterations 控制

#### [MODIFY] `agent/quant_python_agent.py`
- 將固定 pipeline 改為 agent loop
- Researcher Agent：擁有 Tavily + FinMind 工具，自主決定搜尋策略
- Risk Manager Agent：審閱研究結果，可選擇性呼叫工具補充風險資料
- 保留 CLI 輸出格式（`rich` Panel/Table）

#### [MODIFY] `agent/scratchpad.py`
- 擴充為完整的對話記憶，支援 tool call/result 的歷史追蹤

#### [MODIFY] `pyproject.toml`
- 新增 `FinMind` 依賴

#### [MODIFY] `env.example`
- 新增 `FINMIND_API_KEY` 欄位

---

## 三、待辦事項 (Todo Items)

- [ ] **Phase 1：工具層**
    - [ ] 安裝 FinMind 依賴
    - [ ] 在 `tools.py` 中實作 `FinMindTool` 類別
    - [ ] 為每個工具定義 OpenAI function schema
- [ ] **Phase 2：Agent 核心**
    - [ ] 重構 `llm.py`，加入 `chat_with_tools()` 支援 function calling
    - [ ] 重設計 `base_agent.py`，實現 agent loop
    - [ ] 重構 `scratchpad.py`，支援 tool call 歷史
- [ ] **Phase 3：整合調度**
    - [ ] 重構 `quant_python_agent.py`，用 agent loop 取代固定 pipeline
    - [ ] 保持 CLI 體驗（進度顯示、報告格式）
- [ ] **Phase 4：測試與驗證**
    - [ ] 測試單一股票查詢
    - [ ] 測試多股票比較
    - [ ] 驗證工具自主選擇行為

---

## 四、Pull Request 操作指引

1. **Fork**：在 GitHub 點擊原專案 `Fork` 按鈕
2. **Clone**：`git clone` 您 Fork 後的版本
3. **Branch**：`git checkout -b feat/agent-native-finmind`
4. **Commit**：完成後 `git commit -m "feat: refactor to agent-native architecture with FinMind integration"`
5. **Push**：`git push origin feat/agent-native-finmind`
6. **Create PR**：回原專案點擊 `Compare & pull request`

---

## 審查 (Review)
### 變更說明
1. **Agent Native 全面重構**：將硬編碼 Pipeline 改為基於 OpenAI Function Calling 的自主 Agent 迴圈。
2. **FinMind 專業數據庫整合**：提供包含量價、籌碼、月營收、估值（PER/PBR）、股利政策及三大財務報表的完整工具集。
3. **記憶與推理強化**：實作了帶有「歷史記憶」的 Agent 迴圈，並針對 **GPT-5.2** 與 **o3-mini** 實作了自動參數適配（如 `max_completion_tokens`）。
4. **驗證方式**：透過 `test_agent_deep_fix.py` 成功驗證多輪對話與工具聯動邏輯，確認系統已具備深度基本面分析能力。

### 相關資訊
- **環境變數**：請在 `.env` 中設定 `FINMIND_API_KEY` 以利資料獲取。
- **PR 指引**：本件改動已準備好按 GitHub 標準進行 Fork/Pull Request。
