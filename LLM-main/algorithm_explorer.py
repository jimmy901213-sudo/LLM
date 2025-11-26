"""
摸索演算法模組：自動嘗試多種提示策略，評估並選擇最優方案
支持並行運行不同策略、記錄性能指標、自動權重調整
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path


class StrategyName(str, Enum):
    """提示策略枚舉"""
    CONCISE = "concise"           # 簡潔版：直白陳述事實
    DETAILED = "detailed"          # 詳細版：詳細解釋每個方面
    SEO_FOCUSED = "seo_focused"    # SEO 重點版：強調 SEO 優化
    EMOTIONAL = "emotional"        # 情感版：強調情感和體驗
    COMPARATIVE = "comparative"    # 對比版：與競品對比


class AlgorithmExplorer:
    """
    演算法摸索管理器
    - 定義多個提示策略
    - 並行或順序執行不同策略
    - 評估結果品質
    - 自動選擇或推薦最優策略
    """

    def __init__(self, memory_manager=None):
        """
        初始化演算法摸索器
        
        Args:
            memory_manager: MemoryManager 實例（用於記錄結果）
        """
        self.memory_manager = memory_manager
        
        # 定義提示策略模板
        self.strategies = self._init_strategies()
        
        # 策略統計（成功率、平均評分等）
        self.strategy_stats = {}
        self._load_strategy_stats()

    def _init_strategies(self) -> Dict[str, str]:
        """初始化提示策略模板"""
        return {
            StrategyName.CONCISE.value: """
你是一名電商文案專家。請為以下產品生成簡潔而有力的行銷內容。
只提供必要信息，每部分言簡意賅。

【產品事實】
{product_context}

【行銷規則】
{rule_context}

【查詢】
為 {product_query} 生成簡潔版行銷內容

輸出必須嚴格遵循此 JSON Schema: {json_schema}
""",
            
            StrategyName.DETAILED.value: """
你是一名資深電商行銷顧問，精通 AIO 和 E-E-A-T 規則。
請為以下產品生成詳細、全面的行銷內容，涵蓋所有重要面向。

【產品事實】
{product_context}

【行銷規則】
{rule_context}

【查詢】
為 {product_query} 生成詳細版行銷內容

你應該：
1. 詳細說明產品的每個功能和優勢
2. 解釋為什麼這些功能對用戶有價值
3. 提供具體的使用場景和實際案例

輸出必須嚴格遵循此 JSON Schema: {json_schema}
""",
            
            StrategyName.SEO_FOCUSED.value: """
你是一名 SEO 專家。請為以下產品生成高度優化的行銷內容。
重點放在 SEO 關鍵字密度、搜索意圖匹配和排名潛力上。

【產品事實】
{product_context}

【行銷規則】
{rule_context}

【查詢】
為 {product_query} 生成 SEO 優化版行銷內容

優化重點：
1. 標題應包含主要關鍵字和品牌名稱
2. 功能描述應自然融入長尾關鍵字
3. 語義標籤應涵蓋相關搜索詞

輸出必須嚴格遵循此 JSON Schema: {json_schema}
""",
            
            StrategyName.EMOTIONAL.value: """
你是一名創意文案撰寫者。請為以下產品生成情感驅動的行銷內容。
重點放在用戶的情感需求、生活方式和品牌故事上。

【產品事實】
{product_context}

【行銷規則】
{rule_context}

【查詢】
為 {product_query} 生成情感驅動版行銷內容

請突出：
1. 產品如何改善用戶的生活品質
2. 品牌背後的故事和價值觀
3. 用戶使用該產品時的情感體驗

輸出必須嚴格遵循此 JSON Schema: {json_schema}
""",
            
            StrategyName.COMPARATIVE.value: """
你是一名產品對比分析師。請為以下產品生成對比性的行銷內容。
突出該產品相比競品的優勢。

【產品事實】
{product_context}

【行銷規則】
{rule_context}

【查詢】
為 {product_query} 生成對比版行銷內容

請強調：
1. 該產品的獨特優勢
2. 相比常見替代品的優越性
3. 價格-性能比的優勢

輸出必須嚴格遵循此 JSON Schema: {json_schema}
"""
        }

    def get_strategy_prompt(
        self,
        strategy_name: str,
        product_context: str,
        rule_context: str,
        product_query: str,
        json_schema: Dict[str, Any]
    ) -> str:
        """
        獲取特定策略的完整提示詞
        
        Args:
            strategy_name: 策略名稱
            product_context: 產品背景信息
            rule_context: 行銷規則
            product_query: 產品查詢
            json_schema: 輸出格式 schema
        
        Returns:
            完整的提示詞
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"未知策略: {strategy_name}")
        
        template = self.strategies[strategy_name]
        return template.format(
            product_context=product_context,
            rule_context=rule_context,
            product_query=product_query,
            json_schema=json_schema
        )

    def generate_with_strategy(
        self,
        strategy_name: str,
        llm_invoke_func: Callable,  # 接收 (prompt) -> result 的函數
        product_context: str,
        rule_context: str,
        product_query: str,
        json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用特定策略進行生成
        
        Args:
            strategy_name: 策略名稱
            llm_invoke_func: LLM 調用函數
            product_context: 產品背景信息
            rule_context: 行銷規則
            product_query: 產品查詢
            json_schema: 輸出格式 schema
        
        Returns:
            生成結果
        """
        try:
            # 獲取提示詞
            prompt = self.get_strategy_prompt(
                strategy_name,
                product_context,
                rule_context,
                product_query,
                json_schema
            )
            
            # 記錄開始時間
            start_time = time.time()
            
            # 調用 LLM
            result = llm_invoke_func(prompt)
            
            # 計算執行時間
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "strategy": strategy_name,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "success": False,
                "strategy": strategy_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def explore_all_strategies(
        self,
        llm_invoke_func: Callable,
        product_context: str,
        rule_context: str,
        product_query: str,
        json_schema: Dict[str, Any],
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        嘗試所有策略
        
        Args:
            llm_invoke_func: LLM 調用函數
            product_context: 產品背景信息
            rule_context: 行銷規則
            product_query: 產品查詢
            json_schema: 輸出格式 schema
            parallel: 是否並行執行（目前順序執行）
        
        Returns:
            所有策略的結果
        """
        results = {}
        
        print("🔍 開始摸索演算法...")
        for i, strategy_name in enumerate(self.strategies.keys(), 1):
            print(f"\n  [{i}/{len(self.strategies)}] 嘗試策略: {strategy_name}")
            
            result = self.generate_with_strategy(
                strategy_name,
                llm_invoke_func,
                product_context,
                rule_context,
                product_query,
                json_schema
            )
            
            results[strategy_name] = result
            
            if result["success"]:
                print(f"  ✅ {strategy_name} 成功 ({result['execution_time']:.2f}s)")
            else:
                print(f"  ❌ {strategy_name} 失敗: {result.get('error', '未知錯誤')}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "query": product_query,
            "results": results,
            "successful_strategies": sum(1 for r in results.values() if r["success"]),
            "total_strategies": len(results)
        }

    def score_result(
        self,
        result: Dict[str, Any],
        criteria: Optional[Dict[str, float]] = None
    ) -> float:
        """
        評分生成結果（0-10）
        
        評分標準：
        - 文本長度：過短或過長扣分
        - 結構完整性：是否包含所有必要字段
        - 內容相關性：是否針對查詢產品
        
        Args:
            result: 生成結果對象
            criteria: 自訂評分標準
        
        Returns:
            評分（0-10）
        """
        score = 10.0
        
        # 檢查必要字段
        required_fields = ["catchy_title", "experience_paragraph", "features_bullets", "qa_pairs"]
        missing_fields = sum(1 for field in required_fields if not result.get(field))
        score -= missing_fields * 1.5
        
        # 檢查內容長度
        title_len = len(result.get("catchy_title", ""))
        if title_len < 10 or title_len > 70:
            score -= 1
        
        features_count = len(result.get("features_bullets", []))
        if features_count < 3 or features_count > 6:
            score -= 0.5
        
        qa_count = len(result.get("qa_pairs", []))
        if qa_count < 2 or qa_count > 4:
            score -= 0.5
        
        # 套用自訂標準
        if criteria:
            for criterion, weight in criteria.items():
                # 這裡可以添加更複雜的評分邏輯
                pass
        
        return max(0, min(10, score))

    def select_best_strategy(
        self,
        exploration_results: Dict[str, Any]
    ) -> Optional[str]:
        """
        根據探索結果選擇最佳策略
        
        Args:
            exploration_results: 探索結果（來自 explore_all_strategies）
        
        Returns:
            最佳策略名稱
        """
        results = exploration_results.get("results", {})
        
        # 計算每個策略的評分
        strategy_scores = {}
        for strategy_name, result in results.items():
            if result["success"]:
                score = self.score_result(result.get("result", {}))
                # 考慮執行時間（越快越好）
                execution_time = result.get("execution_time", 0)
                # 加入時間因素（最多減 1 分）
                time_penalty = min(1.0, execution_time / 10)
                final_score = score - time_penalty
                strategy_scores[strategy_name] = final_score
            else:
                strategy_scores[strategy_name] = 0
        
        if not strategy_scores:
            return None
        
        # 選擇評分最高的策略
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        return best_strategy[0]

    def update_strategy_weights(
        self,
        strategy_name: str,
        performance_score: float,
        success: bool = True
    ):
        """
        更新策略權重（基於實際性能）
        
        Args:
            strategy_name: 策略名稱
            performance_score: 性能評分（0-10）
            success: 是否成功
        """
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                "total_runs": 0,
                "successful_runs": 0,
                "scores": [],
                "weight": 1.0
            }
        
        stats = self.strategy_stats[strategy_name]
        stats["total_runs"] += 1
        if success:
            stats["successful_runs"] += 1
        stats["scores"].append(performance_score)
        
        # 根據平均評分調整權重
        if stats["scores"]:
            avg_score = sum(stats["scores"]) / len(stats["scores"])
            stats["weight"] = avg_score / 10.0 * 2.0  # 權重範圍 0-2
        
        self._save_strategy_stats()

    def update_algorithm_stats(
        self,
        strategy: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        兼容 MemoryManager 的接口：更新算法統計數據

        這個方法會：
        - 將 metrics 中的 quality 指標映射為 performance_score，並更新本地策略權重
        - 如果存在 memory_manager，則把統計信息也寫入 MemoryManager
        """
        # 從 metrics 中提取 quality 作為 performance_score（如果沒有則使用 0）
        perf = 0.0
        if metrics and isinstance(metrics, dict):
            perf = float(metrics.get("quality", 0.0))

        # 更新本地策略權重統計
        try:
            self.update_strategy_weights(strategy, perf, success=success)
        except Exception:
            # 保持穩健，不讓統計更新影響主流程
            pass

        # 如果有 memory_manager，則也更新其算法統計（保持雙向同步）
        try:
            if self.memory_manager and hasattr(self.memory_manager, "update_algorithm_stats"):
                self.memory_manager.update_algorithm_stats(strategy, success=success, metrics=metrics)
        except Exception:
            pass

    def _load_strategy_stats(self):
        """加載策略統計"""
        stats_file = Path("./memory/strategy_stats.json")
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                self.strategy_stats = json.load(f)
        else:
            self.strategy_stats = {}

    def _save_strategy_stats(self):
        """保存策略統計"""
        stats_file = Path("./memory/strategy_stats.json")
        stats_file.parent.mkdir(exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.strategy_stats, f, ensure_ascii=False, indent=2)

    def get_recommended_strategy(self) -> Optional[str]:
        """
        根據歷史性能推薦最佳策略
        
        Returns:
            推薦策略名稱
        """
        if not self.strategy_stats:
            return None
        
        # 根據平均評分和成功率推薦
        best_strategy = max(
            self.strategy_stats.items(),
            key=lambda x: x[1].get("weight", 1.0)
        )
        return best_strategy[0]

    def get_strategy_performance_report(self) -> Dict[str, Any]:
        """獲取策略性能報告"""
        report = {}
        
        for strategy_name, stats in self.strategy_stats.items():
            total = stats.get("total_runs", 0)
            successful = stats.get("successful_runs", 0)
            scores = stats.get("scores", [])
            
            report[strategy_name] = {
                "total_runs": total,
                "success_rate": successful / total if total > 0 else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "weight": stats.get("weight", 1.0)
            }
        
        return report


# ============ 快速使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 摸索演算法模組演示")
    print("=" * 60)
    
    # 初始化探索器
    explorer = AlgorithmExplorer()
    
    # 示例 1：查看所有策略
    print("\n✅ 示例 1：可用策略")
    print("-" * 60)
    for i, strategy_name in enumerate(explorer.strategies.keys(), 1):
        print(f"{i}. {strategy_name}")
    
    # 示例 2：獲取特定策略的提示詞
    print("\n✅ 示例 2：SEO 優化策略的提示詞片段")
    print("-" * 60)
    prompt = explorer.get_strategy_prompt(
        strategy_name=StrategyName.SEO_FOCUSED.value,
        product_context="X-100 音箱：高保真音質、30小時續航",
        rule_context="SEO 規則：標題應包含關鍵字",
        product_query="X-100 音箱",
        json_schema={"type": "object"}
    )
    print(prompt[:300] + "...")
    
    # 示例 3：評分示例結果
    print("\n✅ 示例 3：評分結果")
    print("-" * 60)
    sample_result = {
        "catchy_title": "高保真音樂體驗",
        "experience_paragraph": "這個音箱改變了我的聽音方式",
        "features_bullets": ["高保真", "30小時續航", "防水設計"],
        "qa_pairs": [{"q": "續航時間？", "a": "30小時"}]
    }
    score = explorer.score_result(sample_result)
    print(f"結果評分: {score:.1f}/10")
    
    # 示例 4：策略性能報告
    print("\n✅ 示例 4：策略性能報告")
    print("-" * 60)
    
    # 模擬一些策略性能數據
    explorer.update_strategy_weights("concise", 8.5, success=True)
    explorer.update_strategy_weights("detailed", 9.2, success=True)
    explorer.update_strategy_weights("seo_focused", 8.8, success=True)
    
    report = explorer.get_strategy_performance_report()
    for strategy, perf in report.items():
        print(f"{strategy}:")
        print(f"  - 成功率: {perf['success_rate']*100:.0f}%")
        print(f"  - 平均評分: {perf['average_score']:.1f}")
        print(f"  - 權重: {perf['weight']:.2f}")
    
    # 示例 5：推薦策略
    print("\n✅ 示例 5：推薦的最佳策略")
    print("-" * 60)
    recommended = explorer.get_recommended_strategy()
    print(f"推薦使用: {recommended}")
    
    print("\n" + "=" * 60)
    print("🎉 摸索演算法模組演示完成！")
    print("=" * 60)
