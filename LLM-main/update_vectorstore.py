"""
自我更新模組：允許系統動態學習新的產品和規則
支持從 JSON 檔案、API、Streamlit UI 輸入新增內容到 ChromaDB
"""

import json
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class VectorstoreUpdater:
    """
    向量庫更新管理器
    - 支持新增單個產品/規則
    - 支持批量匯入 JSON 檔案
    - 自動計算嵌入向量
    - 記錄更新歷史
    """

    def __init__(self, db_path: str = "./chroma_db", embedding_model: str = "nomic-embed-text"):
        """
        初始化更新器
        
        Args:
            db_path: ChromaDB 存儲路徑
            embedding_model: 使用的嵌入模型
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        
        # 初始化嵌入函數
        try:
            self.embeddings = OllamaEmbeddings(model=embedding_model)
        except Exception as e:
            raise RuntimeError(
                f"無法初始化嵌入模型 '{embedding_model}'。"
                f"請確保 Ollama 正在運行並已拉取該模型。\n錯誤: {e}"
            )
        
        # 加載或創建向量庫
        if os.path.exists(db_path):
            self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        else:
            os.makedirs(db_path, exist_ok=True)
            self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        
        # 記錄更新歷史
        self.update_history_file = Path(db_path) / "update_history.json"
        self._load_update_history()

    def _load_update_history(self):
        """加載更新歷史"""
        if self.update_history_file.exists():
            with open(self.update_history_file, "r", encoding="utf-8") as f:
                self.update_history = json.load(f)
        else:
            self.update_history = []

    def _save_update_history(self):
        """保存更新歷史"""
        with open(self.update_history_file, "w", encoding="utf-8") as f:
            json.dump(self.update_history, f, ensure_ascii=False, indent=2)

    def add_product(
        self,
        product_name: str,
        description: str,
        features: List[str],
        price: Optional[str] = None,
        category: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        新增單個產品到向量庫
        
        Args:
            product_name: 產品名稱
            description: 產品描述
            features: 功能列表
            price: 價格（可選）
            category: 產品類別（可選）
            additional_info: 額外信息（可選字典）
        
        Returns:
            操作結果
        """
        try:
            # 構建產品內容
            features_str = "、".join(features)
            content = f"""
產品名稱: {product_name}
描述: {description}
功能: {features_str}
"""
            if price:
                content += f"價格: {price}\n"
            if category:
                content += f"類別: {category}\n"
            
            # 創建文檔並添加到向量庫
            doc = Document(
                page_content=content,
                metadata={
                    "source": "product_db",
                    "product_name": product_name,
                    "category": category or "未分類",
                    "price": price or "未定價",
                    "added_at": datetime.now().isoformat(),
                    **(additional_info or {})
                }
            )
            
            # 添加到 ChromaDB
            doc_ids = self.vectorstore.add_documents([doc])
            
            # 記錄更新歷史
            record = {
                "timestamp": datetime.now().isoformat(),
                "action": "add_product",
                "product_name": product_name,
                "doc_id": doc_ids[0] if doc_ids else None,
                "status": "success"
            }
            self.update_history.append(record)
            self._save_update_history()
            
            return {
                "success": True,
                "message": f"✅ 產品 '{product_name}' 已成功添加到向量庫",
                "doc_id": doc_ids[0] if doc_ids else None,
                "product_name": product_name
            }
        
        except Exception as e:
            record = {
                "timestamp": datetime.now().isoformat(),
                "action": "add_product",
                "product_name": product_name,
                "status": "failed",
                "error": str(e)
            }
            self.update_history.append(record)
            self._save_update_history()
            
            return {
                "success": False,
                "message": f"❌ 添加產品失敗: {e}",
                "product_name": product_name
            }

    def add_rule(
        self,
        rule_text: str,
        category: str,
        rule_type: str = "general",
        priority: int = 5,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        新增單個行銷規則到向量庫
        
        Args:
            rule_text: 規則內容
            category: 規則類別（如 'SEO', 'E-E-A-T', 'copywriting'）
            rule_type: 規則類型
            priority: 優先級（1-10，10 最高）
            tags: 標籤列表
        
        Returns:
            操作結果
        """
        try:
            # 構建規則內容
            content = f"""
規則類別: {category}
優先級: {priority}/10
內容: {rule_text}
"""
            if tags:
                content += f"標籤: {', '.join(tags)}\n"
            
            # 創建文檔
            doc = Document(
                page_content=content,
                metadata={
                    "source": "aio_rules",
                    "category": category,
                    "rule_type": rule_type,
                    "priority": priority,
                    "tags": tags or [],
                    "added_at": datetime.now().isoformat()
                }
            )
            
            # 添加到 ChromaDB
            doc_ids = self.vectorstore.add_documents([doc])
            
            # 記錄更新歷史
            record = {
                "timestamp": datetime.now().isoformat(),
                "action": "add_rule",
                "category": category,
                "doc_id": doc_ids[0] if doc_ids else None,
                "status": "success"
            }
            self.update_history.append(record)
            self._save_update_history()
            
            return {
                "success": True,
                "message": f"✅ 規則已添加到類別 '{category}'",
                "doc_id": doc_ids[0] if doc_ids else None,
                "category": category
            }
        
        except Exception as e:
            record = {
                "timestamp": datetime.now().isoformat(),
                "action": "add_rule",
                "category": category,
                "status": "failed",
                "error": str(e)
            }
            self.update_history.append(record)
            self._save_update_history()
            
            return {
                "success": False,
                "message": f"❌ 添加規則失敗: {e}",
                "category": category
            }

    def batch_import_products(self, json_file: str) -> Dict[str, Any]:
        """
        批量匯入產品（從 JSON 檔案）
        
        JSON 格式範例:
        [
            {
                "product_name": "X-100 音箱",
                "description": "高品質無線音箱",
                "features": ["藍牙 5.0", "30小時續航", "防水"],
                "price": "$199",
                "category": "音頻設備"
            },
            ...
        ]
        
        Args:
            json_file: JSON 檔案路徑
        
        Returns:
            批量操作結果
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                products = json.load(f)
            
            if not isinstance(products, list):
                return {
                    "success": False,
                    "message": "❌ JSON 檔案應包含產品列表 (list)",
                    "total": 0,
                    "imported": 0
                }
            
            results = []
            imported_count = 0
            
            for product in products:
                result = self.add_product(
                    product_name=product.get("product_name", "未知"),
                    description=product.get("description", ""),
                    features=product.get("features", []),
                    price=product.get("price"),
                    category=product.get("category"),
                    additional_info=product.get("additional_info")
                )
                results.append(result)
                if result["success"]:
                    imported_count += 1
            
            return {
                "success": True,
                "message": f"✅ 批量匯入完成: {imported_count}/{len(products)} 成功",
                "total": len(products),
                "imported": imported_count,
                "details": results
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 批量匯入失敗: {e}",
                "total": 0,
                "imported": 0
            }

    def batch_import_rules(self, json_file: str) -> Dict[str, Any]:
        """
        批量匯入規則（從 JSON 檔案）
        
        JSON 格式範例:
        [
            {
                "rule_text": "產品標題應包含主要功能和品牌",
                "category": "SEO",
                "rule_type": "title",
                "priority": 9,
                "tags": ["title", "seo"]
            },
            ...
        ]
        
        Args:
            json_file: JSON 檔案路徑
        
        Returns:
            批量操作結果
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                rules = json.load(f)
            
            if not isinstance(rules, list):
                return {
                    "success": False,
                    "message": "❌ JSON 檔案應包含規則列表 (list)",
                    "total": 0,
                    "imported": 0
                }
            
            results = []
            imported_count = 0
            
            for rule in rules:
                result = self.add_rule(
                    rule_text=rule.get("rule_text", ""),
                    category=rule.get("category", "general"),
                    rule_type=rule.get("rule_type", "general"),
                    priority=rule.get("priority", 5),
                    tags=rule.get("tags")
                )
                results.append(result)
                if result["success"]:
                    imported_count += 1
            
            return {
                "success": True,
                "message": f"✅ 批量匯入完成: {imported_count}/{len(rules)} 成功",
                "total": len(rules),
                "imported": imported_count,
                "details": results
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 批量匯入失敗: {e}",
                "total": 0,
                "imported": 0
            }

    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """
        獲取向量庫統計信息
        
        Returns:
            統計數據
        """
        try:
            # 查詢不同來源的文檔數量
            products = self.vectorstore.get(where={"source": "product_db"})
            rules = self.vectorstore.get(where={"source": "aio_rules"})
            
            product_count = len(products.get("ids", [])) if products else 0
            rule_count = len(rules.get("ids", [])) if rules else 0
            
            return {
                "total_documents": product_count + rule_count,
                "products": product_count,
                "rules": rule_count,
                "db_path": self.db_path,
                "total_updates": len(self.update_history),
                "last_update": self.update_history[-1].get("timestamp") if self.update_history else None
            }
        except Exception as e:
            return {
                "error": str(e),
                "message": "無法獲取統計信息"
            }

    def search_products(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        基於關鍵字（部分匹配）搜索產品元數據。
        支持匹配 product_name、category、price 以及 metadata 中的其他欄位（不區分大小寫）。

        Args:
            keyword: 搜索關鍵字（部分或完整）
            limit: 最多返回結果數

        Returns:
            匹配的產品元數據列表（每項包含 product_name, category, price, doc_id, snippet）
        """
        # First attempt: simple substring match (fast)
        if not keyword:
            return []

        keyword_lower = keyword.lower()
        all_docs = self.vectorstore.get()
        matches: List[Dict[str, Any]] = []

        metadatas = all_docs.get("metadatas") if all_docs else []
        ids = all_docs.get("ids") if all_docs else []

        if not metadatas:
            return []

        seen = set()
        for idx, metadata in enumerate(metadatas):
            if len(matches) >= limit:
                break

            # 聚合要檢查的欄位為字串
            candidate_fields = []
            # 優先使用 product_name，若沒有則嘗試其他欄位（product_id, name）作為展示名稱
            product_name = (
                metadata.get("product_name")
                or metadata.get("product")
                or metadata.get("product_id")
                or metadata.get("name")
                or ""
            )
            category = metadata.get("category", "")
            price = str(metadata.get("price", ""))
            candidate_fields.extend([product_name, category, price])

            # 也檢查 metadata 中的其他值
            for k, v in metadata.items():
                if isinstance(v, str):
                    candidate_fields.append(v)

            # 檢查是否部分匹配
            joined = " ".join(candidate_fields).lower()
            if keyword_lower in joined:
                display_name = product_name or metadata.get("product_name") or metadata.get("product_id") or metadata.get("name") or "未知"
                key = f"{display_name}||{(category or metadata.get('source','未分類'))}"
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "product_name": display_name,
                    "category": category or metadata.get("source", "未分類") or "未分類",
                    "price": price or "",
                    "doc_id": ids[idx] if ids and idx < len(ids) else None,
                    "snippet": (metadata.get("description") or metadata.get("content") or "")
                })

        # If substring match returned results, return them (fast path)
        if matches:
            return matches[:limit]

        # Fallback: use RapidFuzz fuzzy matching if available
        try:
            from rapidfuzz import fuzz
            scored = []
            for idx, metadata in enumerate(metadatas):
                # build a candidate string to compare
                candidate = " ".join(
                    [
                        str(metadata.get("product_name") or metadata.get("product_id") or ""),
                        str(metadata.get("category") or metadata.get("source") or ""),
                    ]
                )
                score = fuzz.partial_ratio(keyword_lower, candidate.lower())
                scored.append((score, idx, metadata))

            # sort by score desc and return top-k above a threshold
            scored.sort(reverse=True, key=lambda x: x[0])
            threshold = 50  # 可調：最低相似度得分
            results = []
            seen = set()
            for score, idx, metadata in scored:
                if len(results) >= limit:
                    break
                if score < threshold:
                    break
                product_name = (
                    metadata.get("product_name")
                    or metadata.get("product")
                    or metadata.get("product_id")
                    or metadata.get("name")
                    or "未知"
                )
                category = metadata.get("category", "") or metadata.get("source", "未分類")
                key = f"{product_name}||{category}"
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "product_name": product_name,
                    "category": category,
                    "price": str(metadata.get("price", "")),
                    "doc_id": ids[idx] if ids and idx < len(ids) else None,
                    "snippet": (metadata.get("description") or metadata.get("content") or ""),
                    "score": score
                })

            return results
        except Exception:
            # rapidfuzz not available or other error, return empty
            return []

    def get_update_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        獲取更新歷史
        
        Args:
            limit: 返回數量限制
        
        Returns:
            更新歷史列表
        """
        return self.update_history[-limit:]

    def export_vectorstore(self, output_file: str = "vectorstore_backup.json") -> Dict[str, Any]:
        """
        匯出向量庫內容（用於備份）
        
        Args:
            output_file: 輸出檔案路徑
        
        Returns:
            匯出結果
        """
        try:
            # 獲取所有文檔
            all_docs = self.vectorstore.get()
            
            # 準備匯出數據
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "products": [],
                "rules": []
            }
            
            if all_docs and all_docs.get("metadatas"):
                for metadata in all_docs["metadatas"]:
                    if metadata.get("source") == "product_db":
                        export_data["products"].append(metadata)
                    elif metadata.get("source") == "aio_rules":
                        export_data["rules"].append(metadata)
            
            # 保存到檔案
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "message": f"✅ 向量庫已匯出到 {output_file}",
                "products_exported": len(export_data["products"]),
                "rules_exported": len(export_data["rules"])
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 匯出失敗: {e}"
            }


# ============ 快速使用示例 ============

if __name__ == "__main__":
    print("初始化向量庫更新器...")
    updater = VectorstoreUpdater()
    
    # 示例 1：添加單個產品
    print("\n[示例 1] 添加單個產品...")
    result = updater.add_product(
        product_name="Y-200 耳機",
        description="高端降噪藍牙耳機",
        features=["主動降噪", "40小時續航", "舒適佩戴"],
        price="$299",
        category="音頻設備"
    )
    print(result["message"])
    
    # 示例 2：添加規則
    print("\n[示例 2] 添加行銷規則...")
    result = updater.add_rule(
        rule_text="強調產品的獨特賣點 (USP)，而不只是列舉特徵",
        category="copywriting",
        priority=8,
        tags=["copywriting", "persuasion"]
    )
    print(result["message"])
    
    # 示例 3：查看統計
    print("\n[示例 3] 向量庫統計...")
    stats = updater.get_vectorstore_stats()
    print(f"✅ 總文檔數: {stats.get('total_documents')}")
    print(f"   產品: {stats.get('products')}")
    print(f"   規則: {stats.get('rules')}")
    
    # 示例 4：查看更新歷史
    print("\n[示例 4] 最近的更新歷史...")
    history = updater.get_update_history(limit=5)
    for record in history:
        print(f"  - {record.get('timestamp')}: {record.get('action')} ({record.get('status')})")
    
    print("\n✅ 自我更新模組演示完成！")
