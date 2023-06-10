import os
import streamlit as st

from PIL import Image
from llama_index import (
    Document,
    GPTVectorStoreIndex,
    GPTListIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    PromptHelper,
    StorageContext,
    load_index_from_storage,
    download_loader,
)
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS

# 必要なモジュールをインポート
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from utils import get_llm

# 関数の定義部分
def initialize_index(llm_name, model_temperature, api_key):
    """Create the GPTSQLStructStoreIndex object."""
    llm = get_llm(llm_name, model_temperature, api_key)

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm))

    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./initial_index"),
        service_context=service_context,
    )

    return index

# OpenAI APIキー、LLM名、モデルの温度を設定
api_key = os.getenv("OPENAI_API_KEY")  # これはあなたの実際のAPIキーに置き換えてください
llm_name = "text-davinci-003"  # あなたが使用したいモデル名に置き換えてください
model_temperature = 0.5  # モデルの温度を設定。0から1の範囲で設定可能

# 関数を呼び出して結果を表示
index = initialize_index(llm_name, model_temperature, api_key)
# print(index.as_query_engine().query("What is the New York City"))
