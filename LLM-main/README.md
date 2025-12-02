專案設定指南 (README.md) - 已修正

這是一個本地運行的 RAG 專案，用於生成 AIO/SEO 優化的行銷文案。請依照以下步驟在您的電腦上設定並運行此專案：

前提條件：安裝 Ollama

此專案依賴 Ollama 在本地運行開源 LLM。

    安裝 Ollama：前往 ollama.ai 下載並安裝適合您作業系統（Windows, macOS, Linux）的版本。

    拉取模型：安裝完成後，打開您的終端機（Terminal 或命令提示字元）並執行以下兩個命令，以下載專案所需的模型：
    Bash



Bash
    ollama pull llama3:8b
    ollama pull nomic-embed-text

    (請確保 Ollama 應用程式在後台保持運行)

專案運行步驟：

    建立專案資料夾：在您的電腦上建立一個新資料夾（例如 AIO_RAG_Project）。

    安裝 Python：確保您已安裝 Python 3.9 或更高版本。

    # if streamlit is installed in the active environment
streamlit run "C:\Users\alan8\Downloads\ncu\地端端\LLM-main\app.py"

    # OR explicitly via python module (safer inside virtualenv)
python -m streamlit run "C:\Users\alan8\Downloads\ncu\地端端\LLM-main\app.py"
完成：您的瀏覽器將自動打開一個新分頁，顯示本地運行的 Web 應用程式。
