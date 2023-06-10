import os
import streamlit as st
from llama_index import Document,  LLMPredictor, ServiceContext, load_index_from_storage
from llama_index.indices.list import GPTListIndex

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI


DEFAULT_TERM_STR = (
    "Make a list of terms and definitions that are defined in the context, "
    "with one pair on each line. "
    "If a term is missing it's definition, use your best judgment. "
    "Write each line as as follows:\nTerm: <term> Definition: <definition>"
)

st.title("🦙 Llama Index Term Extractor 🦙")

setup_tab, upload_tab = st.tabs(["Setup", "Upload/Extract Terms"])


def get_llm(llm_name, model_temperature, api_key, max_tokens=256):
    os.environ['OPENAI_API_KEY'] = api_key
    if llm_name == "text-davinci-003":
        return OpenAI(temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens)
    else:
        return ChatOpenAI(temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens)

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


with setup_tab:
    st.subheader("LLM Setup")
    api_key = st.text_input("Enter your OpenAI API key here", type="password")
    llm_name = st.selectbox('Which LLM?', ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
    model_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, step=0.1)
    term_extract_str = st.text_area("The query to extract terms and definitions with.", value=DEFAULT_TERM_STR)

with upload_tab:
    st.subheader("Extract and Query Definitions")
    document_text = st.text_area("Or enter raw text")
    print(document_text)
    if st.button("Extract Terms and Definitions") and document_text:
        with st.spinner("Extracting..."):
            extracted_terms = extract_terms([Document(document_text)],
                                            term_extract_str, llm_name,
                                            model_temperature, api_key)
        st.write(extracted_terms)


def insert_terms(terms_to_definition):
    for term, definition in terms_to_definition.items():
        doc = Document(f"Term: {term}\nDefinition: {definition}")
        st.session_state['llama_index'].insert(doc)
    # TEMPORARY - save to disk
    st.session_state['llama_index'].storage_context.persist("initial_index")

@st.cache_resource
def initialize_index(llm_name, model_temperature, api_key):
    """Create the SQLStructStoreIndex object."""
    llm = get_llm(llm_name, model_temperature, api_key)

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm))

    index = load_index_from_storage(service_context=service_context)

    return index
  
if "all_terms" not in st.session_state:
    st.session_state["all_terms"] = DEFAULT_TERMS