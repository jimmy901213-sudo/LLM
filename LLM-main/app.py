import streamlit as st
import os
import json
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# 匯入新模組
from memory import MemoryManager
from algorithm_explorer import AlgorithmExplorer, StrategyName
from update_vectorstore import VectorstoreUpdater

# --- 1. 定義 Pydantic Schema (我們的結構化輸出) ---
# 這個 Schema 將強制 Llama 3 以我們想要的 JSON 格式回應 [3, 4]
class MarketingContent(BaseModel):
    product_name: str = Field(description="產品的官方全名")
    catchy_title: str = Field(description="優化的 AIO/SEO 標題，不超過 60 個字符")
    experience_paragraph: str = Field(description="E-E-A-T 化的第一人稱使用經驗段落，需結合一個真實場景")
    features_bullets: List[str] = Field(description="從產品事實中提取的 3-5 個核心功能列表")
    semantic_tags: List[str] = Field(description="相關的語義關鍵字和實體 (例如 '戶外', '派對')")
    qa_pairs: List[Dict[str, str]] = Field(description="2-3 個 Q&A 對，格式為 [{'q': '...', 'a': '...'}]")

# --- 2. 系統初始化 (緩存以提高效能) ---
@st.cache_resource
def load_system():
    # 檢查 ChromaDB 是否存在
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        return None, None, None, "ChromaDB 目錄未找到。請先運行 'create_vectorstore.py'。", None, None, None

    try:
        # 1. 初始化 LLM (結構化輸出)
        llm = ChatOllama(model="llama3:8b", temperature=0.1)
        llm_structured = llm.with_structured_output(MarketingContent)
        
        # 2. 初始化嵌入模型
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 3. 加載向量數據庫
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # 4. 創建兩個檢索器
        retriever_products = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"source": "product_db"}, "k": 1}
        )
        retriever_rules = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"source": "aio_rules"}, "k": 5}
        )
        
        # 5. 初始化新模組：Memory、AlgorithmExplorer、VectorstoreUpdater
        memory_manager = MemoryManager()
        algorithm_explorer = AlgorithmExplorer(memory_manager)
        vectorstore_updater = VectorstoreUpdater(db_path)

        return llm_structured, retriever_products, retriever_rules, None, memory_manager, algorithm_explorer, vectorstore_updater
    except Exception as e:
        return None, None, None, f"系統初始化失敗：{e}。請確保 Ollama 正在運行，並且您已拉取 'llama3:8b' 和 'nomic-embed-text' 模型。", None, None, None

# --- 3. Streamlit 界面 ---
st.set_page_config(layout="wide")
st.title("🚀 AIO/SEO 行銷內容生成器 (本地版增強版)")
st.caption("整合 Memory、演算法摸索、自我更新的智能行銷文案生成系統")

# 加載系統
structured_llm, prod_retriever, rule_retriever, error_msg, memory_manager, algorithm_explorer, vectorstore_updater = load_system()

if error_msg:
    st.error(error_msg)
else:
    # 頂部統計面板
    col1, col2, col3, col4 = st.columns(4)
    
    if memory_manager:
        stats = memory_manager.aggregate_feedback_stats()
        vs_stats = vectorstore_updater.get_vectorstore_stats()
        algo_stats = algorithm_explorer.get_strategy_performance_report()
        
        with col1:
            st.metric("📚 生成記錄", len(memory_manager.get_all_records()), "條")
        with col2:
            st.metric("⭐ 平均評分", stats.get("average_score", 0), "/10")
        with col3:
            st.metric("📚 向量庫文檔", vs_stats.get("total_documents", 0), "個")
        with col4:
            recommended_strategy = algorithm_explorer.get_recommended_strategy()
            st.metric("🎯 推薦策略", recommended_strategy or "待評估", "")
    
    st.divider()
    
    # 左側邊欄設置
    st.sidebar.header("⚙️ 設置面板")
    
    # Tab 1: 主要生成
    tab1, tab2, tab3, tab4 = st.sidebar.tabs(["生成", "Memory", "更新向量庫", "統計"])
    
    with tab1:
        st.subheader("🎯 內容生成")
        # 支援部分關鍵字匹配：使用者可輸入部分名稱或類別，點擊「🔎 搜索產品」來列出匹配項
        product_query_input = st.text_input("輸入產品名稱或關鍵字", placeholder="例如: X-100 或 音箱 或 X-100 音箱")

        # 搜索按鈕：僅在按下時執行向量庫檢索，結果保存在 session_state
        if st.button("🔎 搜索產品"):
            if product_query_input and vectorstore_updater:
                st.session_state.search_matches = vectorstore_updater.search_products(product_query_input, limit=50)
            else:
                st.session_state.search_matches = []

        # 顯示搜索結果（若有）
        matches = st.session_state.get("search_matches", [])
        selected_product_name: Optional[str] = None

        if matches:
            st.write(f"找到 {len(matches)} 個匹配，請從下方選擇要生成的產品：")
            product_choices = [f"{m['product_name']} ({m['category']})" for m in matches]
            chosen = st.selectbox("選擇產品（或留空使用原始輸入）", ["(使用原始輸入)"] + product_choices)
            if chosen != "(使用原始輸入)":
                selected_product_name = chosen.split(" (")[0]

            if st.checkbox("顯示匹配詳情"):
                for m in matches:
                    st.write(f"- {m['product_name']}  |  類別: {m['category']}  |  價格: {m.get('price','')}  | doc_id: {m.get('doc_id')}")

            if st.button("清除搜索結果"):
                st.session_state.search_matches = []
        else:
            st.info("尚未搜索或未找到匹配項。請輸入關鍵字並按『🔎 搜索產品』。")
        
        # 演算法選擇
        algorithm_mode = st.radio(
            "選擇生成模式",
            ["單一策略", "摸索所有策略"],
            help="單一策略速度快，摸索模式會嘗試多個策略"
        )
        
        if algorithm_mode == "單一策略":
            selected_strategy = st.selectbox(
                "選擇策略",
                list(StrategyName),
                format_func=lambda x: x.value
            )
        else:
            selected_strategy = None
        
        if st.button("🚀 生成內容", use_container_width=True):
            # 基於部分匹配選擇最終查詢字串
            product_query_final = (selected_product_name or product_query_input or "").strip()
            if not product_query_final:
                st.error("請輸入產品名稱或選擇匹配項目")
            else:
                with st.spinner("系統正在思考..."):
                    try:
                        # 檢索上下文（使用最終解析出的產品名稱或原始輸入）
                        product_context_docs = prod_retriever.invoke(product_query_final)
                        product_context = "\n---\n".join([doc.page_content for doc in product_context_docs])

                        rule_context_docs = rule_retriever.invoke("所有 AIO/SEO/E-E-A-T 行銷規則")
                        rule_context = "\n---\n".join([doc.page_content for doc in rule_context_docs])

                        json_schema_description = MarketingContent.model_json_schema()

                        if algorithm_mode == "單一策略":
                            # 單一策略模式
                            prompt = algorithm_explorer.get_strategy_prompt(
                                strategy_name=selected_strategy.value,
                                product_context=product_context,
                                rule_context=rule_context,
                                product_query=product_query_final,
                                json_schema=json_schema_description
                            )

                            response_obj = structured_llm.invoke(prompt)

                            # 記錄到 Memory
                            memory_manager.add_generation_record(
                                query=product_query_final,
                                product_name=response_obj.product_name,
                                strategy=selected_strategy.value,
                                result=response_obj.model_dump()
                            )

                            # 更新算法統計（示例數值，可由評分替換）
                            algorithm_explorer.update_algorithm_stats(
                                selected_strategy.value,
                                success=True,
                                metrics={"quality": 8.0}
                            )

                            strategy_used = selected_strategy.value
                        else:
                            # 摸索所有策略模式
                            def llm_call(prompt):
                                return structured_llm.invoke(prompt)

                            exploration_results = algorithm_explorer.explore_all_strategies(
                                llm_invoke_func=llm_call,
                                product_context=product_context,
                                rule_context=rule_context,
                                product_query=product_query_final,
                                json_schema=json_schema_description
                            )

                            # 選擇最佳策略
                            best_strategy = algorithm_explorer.select_best_strategy(exploration_results)

                            if best_strategy and exploration_results["results"][best_strategy]["success"]:
                                response_obj = exploration_results["results"][best_strategy]["result"]
                            else:
                                st.error("沒有成功的策略")
                                st.stop()

                            # 記錄到 Memory
                            memory_manager.add_generation_record(
                                query=product_query_final,
                                product_name=response_obj.product_name,
                                strategy=f"摸索_{best_strategy}",
                                result=response_obj.model_dump()
                            )

                            strategy_used = f"摸索_{best_strategy}"

                        # 在主區顯示結果
                        st.session_state.last_result = response_obj
                        st.session_state.last_strategy = strategy_used

                    except Exception as e:
                        st.error(f"生成失敗：{e}")
    
    with tab2:
        st.subheader("💾 Memory 系統")
        
        memory_options = st.radio(
            "Memory 操作",
            ["查看歷史", "查看反饋", "歷史搜尋"]
        )
        
        if memory_options == "查看歷史":
            records = memory_manager.get_all_records()
            if records:
                st.write(f"共 {len(records)} 條記錄")
                for record in records[-5:]:
                    with st.expander(f"{record['product_name']} - {record['timestamp'][:10]}"):
                        st.write(f"查詢: {record['query']}")
                        st.write(f"策略: {record['strategy']}")
                        st.write(f"評分: {record.get('user_score', 'N/A')}")
            else:
                st.info("暫無記錄")
        
        elif memory_options == "查看反饋":
            stats = memory_manager.aggregate_feedback_stats()
            st.write(f"平均評分: {stats.get('average_score', 0):.1f}/10")
            st.write(f"總反饋: {stats.get('total_feedback', 0)} 條")
            
            if stats.get("distribution"):
                st.bar_chart(stats["distribution"])
        
        elif memory_options == "歷史搜尋":
            search_query = st.text_input("搜尋產品名稱")
            if search_query:
                similar = memory_manager.get_similar_past_results(search_query, limit=10)
                for record in similar:
                    st.write(f"✅ {record['product_name']} ({record['strategy']})")
    
    with tab3:
        st.subheader("📤 自我更新")
        
        update_mode = st.radio(
            "更新模式",
            ["單個產品", "批量匯入"]
        )
        
        if update_mode == "單個產品":
            product_name = st.text_input("產品名稱")
            description = st.text_area("產品描述")
            features = st.text_area("功能（每行一個）").split("\n") if st.text_area("功能（每行一個）") else []
            price = st.text_input("價格（可選）")
            
            if st.button("添加產品"):
                if product_name and description:
                    result = vectorstore_updater.add_product(
                        product_name=product_name,
                        description=description,
                        features=[f for f in features if f],
                        price=price or None
                    )
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
        
        else:
            uploaded_file = st.file_uploader("上傳 JSON 檔案", type="json")
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    
                    # 判斷是產品還是規則
                    if isinstance(data, list) and len(data) > 0:
                        if "rule_text" in data[0]:
                            result = vectorstore_updater.batch_import_rules(uploaded_file.name)
                        else:
                            result = vectorstore_updater.batch_import_products(uploaded_file.name)
                        
                        st.success(result["message"])
                except Exception as e:
                    st.error(f"匯入失敗：{e}")
    
    with tab4:
        st.subheader("📊 統計信息")
        
        stat_type = st.radio(
            "統計類型",
            ["向量庫", "演算法", "Memory"]
        )
        
        if stat_type == "向量庫":
            vs_stats = vectorstore_updater.get_vectorstore_stats()
            st.write(f"總文檔數: {vs_stats.get('total_documents', 0)}")
            st.write(f"產品: {vs_stats.get('products', 0)}")
            st.write(f"規則: {vs_stats.get('rules', 0)}")
            st.write(f"總更新次數: {vs_stats.get('total_updates', 0)}")
        
        elif stat_type == "演算法":
            algo_report = algorithm_explorer.get_strategy_performance_report()
            for strategy, perf in algo_report.items():
                st.write(f"**{strategy}**")
                st.write(f"  成功率: {perf['success_rate']*100:.0f}%")
                st.write(f"  平均評分: {perf['average_score']:.1f}")
                st.write(f"  權重: {perf['weight']:.2f}")
        
        elif stat_type == "Memory":
            stats = memory_manager.aggregate_feedback_stats()
            st.json(stats)
    
    # 主區顯示結果
    st.header("📄 生成結果")
    
    if "last_result" in st.session_state:
        response_obj = st.session_state.last_result
        
        st.subheader(f"✅ {response_obj.product_name}")
        
        st.markdown(f"### {response_obj.catchy_title}")
        st.divider()
        
        st.markdown("#### E-E-A-T 經驗段落:")
        st.markdown(response_obj.experience_paragraph)
        
        st.markdown("#### 核心功能:")
        st.markdown("\n".join(f"- {item}" for item in response_obj.features_bullets))
        
        st.markdown("#### Q&A 部分:")
        for pair in response_obj.qa_pairs:
            st.markdown(f"**Q: {pair['q']}**")
            st.markdown(f"A: {pair['a']}")
        
        st.markdown("#### 語義標籤:")
        st.markdown(", ".join(response_obj.semantic_tags))
        
        st.markdown("---")
        
        # 反饋和評分
        col1, col2 = st.columns(2)
        with col1:
            user_score = st.slider("評分", 0, 10, 5, help="0 = 很差，10 = 完美")
        with col2:
            user_comment = st.text_input("評論")
        
        if st.button("保存評分"):
            record_id = memory_manager.persistent_records[-1]["id"]
            memory_manager.add_feedback(
                record_id=record_id,
                score=user_score,
                comment=user_comment
            )
            st.success("✅ 評分已保存")
        
        st.subheader("原始 JSON 輸出 (用於 API):")
        st.json(response_obj.model_dump_json())
    
    else:
        st.info("👈 從左側選擇「生成」標籤開始使用")