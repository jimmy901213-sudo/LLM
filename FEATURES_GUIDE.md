# 🚀 增強功能使用指南

你的 AIO/SEO 行銷內容生成系統已升級！現在包含三大核心功能：

## 📚 三大核心功能

### 1. **Memory 記憶系統** (`memory.py`)
**功能**: 系統自動記憶所有生成結果、用戶反饋、演算法性能

**核心特性**:
- ✅ **持久記憶**: 所有生成結果自動保存到 JSON 檔案
- ✅ **反饋系統**: 用戶可評分（0-10）每個結果並添加評論
- ✅ **演算法追蹤**: 記錄每個策略的成功率和性能指標
- ✅ **相似查詢**: 快速找到類似產品的過去結果

**使用方式**:
```python
from memory import MemoryManager

# 初始化
memory = MemoryManager()

# 添加生成記錄
record = memory.add_generation_record(
    query="X-100 音箱",
    product_name="X-100 音箱",
    strategy="detailed",
    result={...},  # Pydantic 對象的 dict
    user_score=8.5
)

# 查詢相似結果
similar = memory.get_similar_past_results("X-100 音箱")

# 獲取統計
stats = memory.aggregate_feedback_stats()
print(f"平均評分: {stats['average_score']}")
```

**存儲位置**: `./memory/` 目錄
- `generation_records.json` - 所有生成記錄
- `user_feedback.json` - 用戶反饋
- `algorithm_stats.json` - 演算法性能統計

---

### 2. **自我更新模組** (`update_vectorstore.py`)
**功能**: 系統能動態學習新產品和規則，自動更新 ChromaDB

**核心特性**:
- ✅ **單個添加**: 一次添加一個產品或規則
- ✅ **批量匯入**: 從 JSON 檔案批量導入
- ✅ **自動嵌入**: 自動計算向量並存儲
- ✅ **更新歷史**: 記錄所有更新操作

**使用方式**:
```python
from update_vectorstore import VectorstoreUpdater

# 初始化
updater = VectorstoreUpdater()

# 添加單個產品
result = updater.add_product(
    product_name="Y-200 耳機",
    description="高端降噪藍牙耳機",
    features=["主動降噪", "40小時續航"],
    price="$299",
    category="音頻設備"
)

# 批量匯入產品
result = updater.batch_import_products("sample_products.json")

# 獲取向量庫統計
stats = updater.get_vectorstore_stats()
```

**JSON 檔案格式**:

產品 (`sample_products.json`):
```json
[
    {
        "product_name": "Z-500 耳機",
        "description": "專業級主動降噪耳機",
        "features": ["40dB 降噪", "50小時續航"],
        "price": "$399",
        "category": "音頻設備"
    }
]
```

規則 (`sample_rules.json`):
```json
[
    {
        "rule_text": "標題應包含品牌名稱 + 核心功能 + 獨特賣點",
        "category": "SEO",
        "priority": 9,
        "tags": ["title", "seo"]
    }
]
```

---

### 3. **摸索演算法模組** (`algorithm_explorer.py`)
**功能**: 自動嘗試多個提示策略，評估效果，推薦最優方案

**可用策略**:
1. **concise** - 簡潔版（直白陳述事實，速度快）
2. **detailed** - 詳細版（詳細解釋每個方面，質量高）
3. **seo_focused** - SEO 重點版（強調 SEO 優化）
4. **emotional** - 情感版（強調情感和體驗）
5. **comparative** - 對比版（與競品對比）

**核心特性**:
- ✅ **並行/順序運行**: 可選擇單策略或全部摸索
- ✅ **自動評分**: 根據結構完整性、長度等評估品質
- ✅ **性能追蹤**: 記錄每個策略的成功率和平均評分
- ✅ **智能推薦**: 根據歷史數據推薦最佳策略

**使用方式**:
```python
from algorithm_explorer import AlgorithmExplorer, StrategyName

# 初始化
explorer = AlgorithmExplorer()

# 使用單個策略
result = explorer.generate_with_strategy(
    strategy_name=StrategyName.DETAILED.value,
    llm_invoke_func=structured_llm.invoke,
    product_context="...",
    rule_context="...",
    product_query="X-100 音箱",
    json_schema={...}
)

# 摸索所有策略
exploration_results = explorer.explore_all_strategies(
    llm_invoke_func=structured_llm.invoke,
    product_context="...",
    rule_context="...",
    product_query="X-100 音箱",
    json_schema={...}
)

# 選擇最佳策略
best_strategy = explorer.select_best_strategy(exploration_results)

# 獲取性能報告
report = explorer.get_strategy_performance_report()
for strategy, perf in report.items():
    print(f"{strategy}: 成功率 {perf['success_rate']*100}%, 平均評分 {perf['average_score']}")
```

---

## 🎯 Streamlit UI 使用指南

### 啟動應用
```bash
streamlit run app.py
```

### UI 布局

#### **頂部統計面板** (實時更新)
- 📚 生成記錄數
- ⭐ 平均評分
- 📚 向量庫文檔數
- 🎯 推薦策略

#### **左側邊欄 - 4 個標籤頁**

**標籤 1: 生成**
- 輸入產品名稱
- 選擇模式：「單一策略」或「摸索所有策略」
- 如果是單策略，選擇具體策略（concise/detailed/seo_focused/emotional/comparative）
- 點擊「生成內容」

**標籤 2: Memory**
- 查看歷史：顯示最近 5 條記錄
- 查看反饋：平均評分和分佈圖
- 歷史搜尋：按產品名稱查詢類似結果

**標籤 3: 更新向量庫**
- 單個產品：手動輸入產品信息
- 批量匯入：上傳 JSON 檔案

**標籤 4: 統計**
- 向量庫統計：總文檔數、產品數、規則數
- 演算法統計：各策略的成功率、平均評分、權重
- Memory 統計：反饋聚合數據

#### **主區 - 結果展示**
- 生成的內容（標題、經驗段落、功能、Q&A、標籤）
- 用戶評分滑塊（0-10）
- 評論輸入框
- 原始 JSON 輸出

---

## 💡 工作流程示例

### 場景 1：快速生成（單策略）
```
1. 打開 Streamlit 應用
2. 在「生成」標籤輸入「X-100 音箱」
3. 選擇「單一策略」，選 「detailed」
4. 點擊「生成內容」
5. 查看結果，評分並評論
6. 結果自動保存到 Memory
```

### 場景 2：最優化生成（摸索所有策略）
```
1. 打開 Streamlit 應用
2. 在「生成」標籤輸入「X-100 音箱」
3. 選擇「摸索所有策略」
4. 點擊「生成內容」
5. 系統嘗試 5 個策略（會花時間）
6. 自動選擇最優結果
7. 查看結果，評分
8. 根據評分，該策略的權重提升
```

### 場景 3：批量導入新產品
```
1. 準備 sample_products.json
2. 打開「更新向量庫」標籤
3. 選擇「批量匯入」
4. 上傳 JSON 檔案
5. 系統自動導入並更新 ChromaDB
6. 下次查詢時就能找到新產品
```

### 場景 4：查看系統進度
```
1. 打開「統計」標籤
2. 檢查演算法統計：哪個策略表現最好
3. 檢查 Memory 統計：平均評分、反饋數量
4. 檢查向量庫統計：有多少產品/規則
```

---

## 🔧 進階配置

### 修改演算法策略提示詞
編輯 `algorithm_explorer.py` 中的 `_init_strategies()` 方法：
```python
self.strategies[StrategyName.CONCISE.value] = """
你是一名電商文案專家...
[你的自訂提示詞]
"""
```

### 自訂評分標準
編輯 `algorithm_explorer.py` 的 `score_result()` 方法以改變評分邏輯

### 導出 Memory 為 CSV
```python
memory = MemoryManager()
memory.export_records_as_csv("export.csv")
```

---

## 📊 數據位置速查

| 數據 | 位置 | 說明 |
|------|------|------|
| 生成記錄 | `./memory/generation_records.json` | 所有生成結果 |
| 用戶反饋 | `./memory/user_feedback.json` | 評分和評論 |
| 演算法統計 | `./memory/algorithm_stats.json` | 策略性能數據 |
| 向量庫 | `./chroma_db/` | ChromaDB 資料庫 |
| 向量庫更新歷史 | `./chroma_db/update_history.json` | 更新操作記錄 |

---

## 🚀 完整工作流總結

```
用戶查詢產品名稱
    ↓
【Memory 系統】 ← 檢查過去是否生成過此產品
    ↓
【RAG 檢索】 ← 從 ChromaDB 檢索產品和規則
    ↓
【演算法摸索】（可選）← 嘗試多個策略
    ├─ concise（簡潔）
    ├─ detailed（詳細）
    ├─ seo_focused（SEO）
    ├─ emotional（情感）
    └─ comparative（對比）
    ↓
【自動評分和選優】← 選擇最佳結果
    ↓
【LLM 生成】← Llama 3 生成結構化結果
    ↓
【Memory 記錄】← 自動存檔
    ↓
【UI 展示】← 用戶查看、評分、評論
    ↓
【反饋更新】← 用戶評分影響策略權重
```

---

## ⚠️ 常見問題

**Q: 為什麼摸索所有策略很慢？**
A: 因為系統嘗試 5 個不同的提示詞並調用 LLM 5 次。建議在不急的時候使用。

**Q: 可以添加自訂策略嗎？**
A: 可以！編輯 `algorithm_explorer.py` 添加新策略到 `_init_strategies()` 方法。

**Q: 如何重置所有數據？**
A: 刪除 `./memory/` 和 `./chroma_db/` 目錄，然後重新初始化。

**Q: Memory 和評分如何影響系統？**
A: 用戶評分會提升該策略的權重，下次摸索時該策略更可能被推薦。

---

## 📞 技術支持

- **Memory 問題**: 檢查 `./memory/` 目錄是否可寫
- **向量庫問題**: 確保 Ollama 正在運行，Chroma 正確初始化
- **演算法問題**: 檢查 LLM 是否可用，提示詞格式是否正確

---

**享受你的智能行銷文案生成系統！🎉**
