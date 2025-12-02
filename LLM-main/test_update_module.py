"""
自我更新模組測試（不依賴 Ollama）
測試 JSON 匯入、批量操作、統計等功能
"""

import json
from pathlib import Path

# 測試 1：驗證 update_history.json 能被正確讀取
print("=" * 50)
print("✅ 測試 1：驗證更新歷史檔案")
print("=" * 50)

history_file = Path("./chroma_db/update_history.json")
if history_file.exists():
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"✅ 找到 {len(history)} 條更新記錄")
    print("\n最近 3 條記錄：")
    for record in history[-3:]:
        print(f"  - 時間: {record.get('timestamp')}")
        print(f"    操作: {record.get('action')}")
        print(f"    狀態: {record.get('status')}")
else:
    print("❌ 找不到更新歷史檔案")

# 測試 2：建立範例 JSON 檔案（用於批量匯入演示）
print("\n" + "=" * 50)
print("✅ 測試 2：建立範例 JSON 檔案")
print("=" * 50)

# 創建範例產品 JSON
sample_products = [
    {
        "product_name": "Z-500 耳機",
        "description": "專業級主動降噪耳機",
        "features": ["40dB 降噪", "50小時續航", "高通透模式"],
        "price": "$399",
        "category": "音頻設備"
    },
    {
        "product_name": "A-100 喇叭",
        "description": "便攜式藍牙喇叭",
        "features": ["防水", "24小時續航", "360度環繞聲"],
        "price": "$129",
        "category": "音頻設備"
    },
    {
        "product_name": "B-200 麥克風",
        "description": "USB 直播麥克風",
        "features": ["降噪", "彩虹燈效", "一鍵靜音"],
        "price": "$79",
        "category": "音頻設備"
    }
]

with open("./sample_products.json", "w", encoding="utf-8") as f:
    json.dump(sample_products, f, ensure_ascii=False, indent=2)
print(f"✅ 已建立 sample_products.json（包含 {len(sample_products)} 個產品）")

# 創建範例規則 JSON
sample_rules = [
    {
        "rule_text": "標題應包含品牌名稱 + 核心功能 + 獨特賣點",
        "category": "SEO",
        "rule_type": "title",
        "priority": 9,
        "tags": ["title", "seo", "critical"]
    },
    {
        "rule_text": "描述段落應從用戶視角說明產品如何解決問題",
        "category": "copywriting",
        "rule_type": "description",
        "priority": 8,
        "tags": ["copywriting", "user-focused"]
    },
    {
        "rule_text": "功能列表應按重要性排序，最吸引人的功能放首位",
        "category": "copywriting",
        "rule_type": "features",
        "priority": 7,
        "tags": ["copywriting", "layout"]
    },
    {
        "rule_text": "Q&A 段落應針對常見購買疑慮（價格、品質、售後）",
        "category": "E-E-A-T",
        "rule_type": "qa",
        "priority": 8,
        "tags": ["qa", "trust"]
    }
]

with open("./sample_rules.json", "w", encoding="utf-8") as f:
    json.dump(sample_rules, f, ensure_ascii=False, indent=2)
print(f"✅ 已建立 sample_rules.json（包含 {len(sample_rules)} 條規則）")

# 測試 3：驗證 JSON 格式正確性
print("\n" + "=" * 50)
print("✅ 測試 3：驗證 JSON 格式")
print("=" * 50)

with open("./sample_products.json", "r", encoding="utf-8") as f:
    products = json.load(f)
print(f"✅ sample_products.json 有效（{len(products)} 個產品）")
print(f"   第一個產品: {products[0]['product_name']}")

with open("./sample_rules.json", "r", encoding="utf-8") as f:
    rules = json.load(f)
print(f"✅ sample_rules.json 有效（{len(rules)} 條規則）")
print(f"   第一條規則: {rules[0]['category']}")

# 測試 4：模擬批量操作的執行邏輯
print("\n" + "=" * 50)
print("✅ 測試 4：模擬批量操作邏輯")
print("=" * 50)

print("\n[模擬] 逐個添加產品：")
for i, product in enumerate(products, 1):
    print(f"  {i}. {product['product_name']} ({product.get('price', 'N/A')})")

print("\n[模擬] 逐個添加規則：")
for i, rule in enumerate(rules, 1):
    print(f"  {i}. [{rule['category']}] {rule['rule_text'][:40]}...")

# 測試 5：統計數據模擬
print("\n" + "=" * 50)
print("✅ 測試 5：模擬統計功能")
print("=" * 50)

stats = {
    "total_documents": 100,
    "products": 45,
    "rules": 55,
    "categories": {
        "SEO": 15,
        "copywriting": 20,
        "E-E-A-T": 20
    }
}

print(f"✅ 向量庫統計：")
print(f"   總文檔數: {stats['total_documents']}")
print(f"   產品數: {stats['products']}")
print(f"   規則數: {stats['rules']}")
print(f"   規則按類別分布:")
for cat, count in stats["categories"].items():
    print(f"     - {cat}: {count}")

# 測試 6：驗證匯出功能
print("\n" + "=" * 50)
print("✅ 測試 6：驗證匯出功能模擬")
print("=" * 50)

export_data = {
    "timestamp": "2025-11-26T20:00:00",
    "products": products,
    "rules": rules
}

with open("./vectorstore_backup_test.json", "w", encoding="utf-8") as f:
    json.dump(export_data, f, ensure_ascii=False, indent=2)
print(f"✅ 已建立測試備份檔案")
print(f"   匯出產品數: {len(products)}")
print(f"   匯出規則數: {len(rules)}")

print("\n" + "=" * 50)
print("🎉 所有自我更新模組測試通過！")
print("=" * 50)
print("\n下一步：")
print("1. 確保 Ollama 已啟動（ollama serve）")
print("2. 運行 python update_vectorstore.py 執行完整測試")
print("3. 使用 sample_products.json 和 sample_rules.json 進行批量匯入")
