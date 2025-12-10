"""
Memory System for AIO/SEO Marketing Content Generator
支持三層記憶：會話記憶、持久記憶、反饋記憶
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import hashlib


class MemoryManager:
    """
    本地記憶管理系統
    - 會話記憶：當前會話中的歷史（內存中）
    - 持久記憶：保存到本地 JSON 檔案
    - 反饋記憶：用戶評分、評論
    """

    def __init__(self, memory_dir: str = "./memory"):
        """
        初始化記憶管理器
        
        Args:
            memory_dir: 記憶檔案存儲目錄
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # 會話記憶（內存中）
        self.session_records: List[Dict[str, Any]] = []
        
        # 記憶檔案路徑
        self.records_file = self.memory_dir / "generation_records.json"
        self.feedback_file = self.memory_dir / "user_feedback.json"
        self.algorithm_stats_file = self.memory_dir / "algorithm_stats.json"
        
        # 初始化或加載持久記憶
        self._load_from_disk()

    def _load_from_disk(self):
        """從磁盤加載持久記憶"""
        if self.records_file.exists():
            with open(self.records_file, "r", encoding="utf-8") as f:
                self.persistent_records = json.load(f)
        else:
            self.persistent_records = []
        
        if self.feedback_file.exists():
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                self.feedback_data = json.load(f)
        else:
            self.feedback_data = {}
        
        if self.algorithm_stats_file.exists():
            with open(self.algorithm_stats_file, "r", encoding="utf-8") as f:
                self.algorithm_stats = json.load(f)
        else:
            self.algorithm_stats = {}

    def _save_to_disk(self):
        """保存持久記憶到磁盤"""
        with open(self.records_file, "w", encoding="utf-8") as f:
            json.dump(self.persistent_records, f, ensure_ascii=False, indent=2)
        
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        
        with open(self.algorithm_stats_file, "w", encoding="utf-8") as f:
            json.dump(self.algorithm_stats, f, ensure_ascii=False, indent=2)

    def add_generation_record(
        self,
        query: str,
        product_name: str,
        strategy: str,
        result: Dict[str, Any],
        user_score: Optional[float] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        記錄一次生成結果
        
        Args:
            query: 用戶查詢
            product_name: 產品名稱
            strategy: 使用的策略（模板名稱）
            result: 生成的結果（MarketingContent 的 dict）
            user_score: 用戶評分（0-10，可選）
            notes: 用戶備註
        
        Returns:
            記錄對象
        """
        record = {
            "id": self._generate_id(),
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "product_name": product_name,
            "strategy": strategy,
            "result": result,
            "user_score": user_score,
            "notes": notes
        }
        
        # 加入會話記憶和持久記憶
        self.session_records.append(record)
        self.persistent_records.append(record)
        self._save_to_disk()
        
        return record

    def add_feedback(
        self,
        record_id: str,
        score: float,
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        為某個生成結果添加反饋
        
        Args:
            record_id: 記錄 ID
            score: 用戶評分（0-10）
            comment: 評論
        
        Returns:
            反饋對象
        """
        feedback = {
            "record_id": record_id,
            "score": max(0, min(10, score)),  # 確保評分在 0-10 之間
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        if record_id not in self.feedback_data:
            self.feedback_data[record_id] = []
        
        self.feedback_data[record_id].append(feedback)
        
        # 更新記錄中的評分
        for record in self.persistent_records:
            if record.get("id") == record_id:
                record["user_score"] = score
                break
        
        self._save_to_disk()
        return feedback

    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """獲取特定 ID 的記錄"""
        for record in self.persistent_records:
            if record.get("id") == record_id:
                return record
        return None

    def get_similar_past_results(
        self,
        query: str,
        product_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        獲取類似的過去結果（基於查詢和產品）
        
        Args:
            query: 查詢字符串
            product_name: 產品名稱（可選）
            limit: 返回數量限制
        
        Returns:
            相似的歷史記錄列表
        """
        results = []
        query_lower = query.lower()
        
        for record in reversed(self.persistent_records):  # 最新的優先
            if product_name and record.get("product_name") != product_name:
                continue
            
            # 簡單的文本相似度檢查
            if query_lower in record.get("query", "").lower() or \
               record.get("query", "").lower() in query_lower:
                results.append(record)
            
            if len(results) >= limit:
                break
        
        return results

    def get_best_strategy_for_product(self, product_category: str) -> Optional[str]:
        """
        根據歷史數據，獲取對某類產品最優的策略
        
        Args:
            product_category: 產品類別
        
        Returns:
            最優策略名稱
        """
        strategy_scores = {}
        
        for record in self.persistent_records:
            if record.get("product_name", "").startswith(product_category):
                strategy = record.get("strategy")
                score = record.get("user_score", 5)  # 默認評分 5
                
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = {"total": 0, "count": 0}
                
                strategy_scores[strategy]["total"] += score
                strategy_scores[strategy]["count"] += 1
        
        if not strategy_scores:
            return None
        
        # 計算平均評分
        best_strategy = max(
            strategy_scores.items(),
            key=lambda x: x[1]["total"] / x[1]["count"]
        )
        
        return best_strategy[0]

    def update_algorithm_stats(
        self,
        strategy: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        更新演算法統計數據
        
        Args:
            strategy: 策略名稱
            success: 是否成功
            metrics: 其他性能指標（如生成時間、質量評分等）
        """
        if strategy not in self.algorithm_stats:
            self.algorithm_stats[strategy] = {
                "total_runs": 0,
                "successful_runs": 0,
                "metrics": {}
            }
        
        self.algorithm_stats[strategy]["total_runs"] += 1
        if success:
            self.algorithm_stats[strategy]["successful_runs"] += 1
        
        if metrics:
            for key, value in metrics.items():
                if key not in self.algorithm_stats[strategy]["metrics"]:
                    self.algorithm_stats[strategy]["metrics"][key] = []
                self.algorithm_stats[strategy]["metrics"][key].append(value)
        
        self._save_to_disk()

    def get_algorithm_stats(self) -> Dict[str, Any]:
        """獲取所有演算法統計數據"""
        return self.algorithm_stats

    def get_algorithm_success_rate(self, strategy: str) -> float:
        """
        獲取特定策略的成功率
        
        Args:
            strategy: 策略名稱
        
        Returns:
            成功率（0-1）
        """
        if strategy not in self.algorithm_stats:
            return 0.0
        
        stats = self.algorithm_stats[strategy]
        if stats["total_runs"] == 0:
            return 0.0
        
        return stats["successful_runs"] / stats["total_runs"]

    def get_session_history(self) -> List[Dict[str, Any]]:
        """獲取當前會話的歷史"""
        return self.session_records.copy()

    def get_all_records(self) -> List[Dict[str, Any]]:
        """獲取所有持久記憶"""
        return self.persistent_records.copy()

    def get_feedback_for_record(self, record_id: str) -> List[Dict[str, Any]]:
        """獲取特定記錄的所有反饋"""
        return self.feedback_data.get(record_id, [])

    def aggregate_feedback_stats(self) -> Dict[str, Any]:
        """
        聚合所有反饋統計
        
        Returns:
            包含平均評分、評分分佈等的統計數據
        """
        all_scores = []
        feedback_count = 0
        
        for record_id, feedback_list in self.feedback_data.items():
            for feedback in feedback_list:
                all_scores.append(feedback.get("score", 5))
                feedback_count += 1
        
        if not all_scores:
            return {
                "average_score": None,
                "total_feedback": 0,
                "distribution": {}
            }
        
        avg_score = sum(all_scores) / len(all_scores)
        
        # 評分分佈
        distribution = {}
        for score in all_scores:
            distribution[score] = distribution.get(score, 0) + 1
        
        return {
            "average_score": round(avg_score, 2),
            "total_feedback": feedback_count,
            "distribution": distribution,
            "highest_score": max(all_scores),
            "lowest_score": min(all_scores)
        }

    def clear_session_memory(self):
        """清空會話記憶（但保留持久記憶）"""
        self.session_records = []

    def export_records_as_csv(self, output_file: str = "memory_export.csv"):
        """
        匯出記錄為 CSV 檔案
        
        Args:
            output_file: 輸出檔案路徑
        """
        import csv
        
        if not self.persistent_records:
            print("沒有記錄可匯出")
            return
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # 寫入表頭
            writer.writerow([
                "ID", "時間", "查詢", "產品", "策略", "用戶評分", "備註"
            ])
            
            # 寫入數據
            for record in self.persistent_records:
                writer.writerow([
                    record.get("id"),
                    record.get("timestamp"),
                    record.get("query"),
                    record.get("product_name"),
                    record.get("strategy"),
                    record.get("user_score"),
                    record.get("notes")
                ])
        
        print(f"記錄已匯出到 {output_file}")

    def _generate_id(self) -> str:
        """生成唯一 ID"""
        timestamp = datetime.now().isoformat()
        hash_obj = hashlib.md5(timestamp.encode())
        return hash_obj.hexdigest()[:8]


# ============ 快速使用示例 ============

if __name__ == "__main__":
    # 初始化記憶管理器
    memory = MemoryManager()
    
    # 添加一個生成記錄
    result = {
        "product_name": "X-100 音箱",
        "catchy_title": "沉浸式音樂體驗",
        "experience_paragraph": "使用 X-100 後，我發現...",
        "features_bullets": ["高保真音質", "便攜設計", "10小時續航"],
        "semantic_tags": ["音樂", "戶外", "科技"],
        "qa_pairs": [{"q": "電池續航多久？", "a": "10小時"}]
    }
    
    record = memory.add_generation_record(
        query="X-100 音箱",
        product_name="X-100 音箱",
        strategy="detailed",
        result=result,
        user_score=8.5,
        notes="很好的一次生成"
    )
    print(f"✅ 記錄已保存：{record['id']}")
    
    # 添加反饋
    feedback = memory.add_feedback(
        record_id=record["id"],
        score=9,
        comment="文案很專業，改進建議：增加價格對比"
    )
    print(f"✅ 反饋已記錄：{feedback}")
    
    # 獲取統計
    stats = memory.aggregate_feedback_stats()
    print(f"\n📊 反饋統計：{stats}")
    
    # 獲取算法統計
    memory.update_algorithm_stats("detailed", success=True, metrics={"quality": 8.5})
    print(f"\n📈 算法統計：{memory.get_algorithm_stats()}")
    
    # 導出記錄
    memory.export_records_as_csv("./memory_export.csv")
