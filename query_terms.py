import os
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

from constants import DEFAULT_TERM_STR, DEFAULT_TERMS, REFINE_TEMPLATE, TEXT_QA_TEMPLATE
from utils import get_llm

def initialize_index_backend(llm_name, model_temperature, api_key):
    """Create the GPTSQLStructStoreIndex object."""
    llm = get_llm(llm_name, model_temperature, api_key)

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm))

    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./initial_index"),
        service_context=service_context,
    )

    return index

def query_terms_backend(llama_index, query_text):
    response = (
        llama_index
        .as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            text_qa_template=TEXT_QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
        )
        .query(query_text)
    )
    return str(response)


def extract_terms(documents, term_extract_str, llm_name, model_temperature, api_key):
    llm = get_llm(llm_name, model_temperature, api_key, max_tokens=1024)

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm),
                                                   chunk_size_limit=1024)

    temp_index = GPTListIndex.from_documents(documents, service_context=service_context)
    query_engine = temp_index.as_query_engine(response_mode="tree_summarize")
    terms_definitions = str(query_engine.query(term_extract_str))
    terms_definitions = [x for x in terms_definitions.split("\n") if x and 'Term:' in x and 'Definition:' in x]
    # parse the text into a dict
    terms_to_definition = {x.split("Definition:")[0].split("Term:")[-1].strip(): x.split("Definition:")[-1].strip() for x in terms_definitions}
    return terms_to_definition


# Set your parameters here
llm_name = "text-davinci-003"
model_temperature = 0.5
api_key = os.getenv("OPENAI_API_KEY")  # Replace with your actual OpenAI API key
query_text = "I gone to combined statistical area in the new york for COVID-19 pandemic."
term_extract_str = DEFAULT_TERM_STR

# Assuming `document_text` is a string that contains the text you want to process.
# This should be replaced with the actual text you want to process.

extracted_terms = extract_terms(
    [Document(query_text)],
    term_extract_str, 
    llm_name,
    model_temperature, 
    api_key
)

print(extracted_terms)
# llama_index = initialize_index_backend(llm_name, model_temperature, api_key)
# response = query_terms_backend(llama_index, query_text)
# print(response)