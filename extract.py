from llama_index import Document, ListIndex, LLMPredictor, ServiceContext, load_index_from_storage

def get_llm(llm_name, model_temperature, api_key, max_tokens=256):
    os.environ['OPENAI_API_KEY'] = api_key
    if llm_name == "text-davinci-003":
        return OpenAI(temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens)
    else:
        return ChatOpenAI(temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens)

def extract_terms(documents, term_extract_str, llm_name, model_temperature, api_key):
    llm = get_llm(llm_name, model_temperature, api_key, max_tokens=1024)

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm),
                                                   chunk_size=1024)

    temp_index = ListIndex.from_documents(documents, service_context=service_context)
    query_engine = temp_index.as_query_engine(response_mode="tree_summarize")
    terms_definitions = str(query_engine.query(term_extract_str))
    terms_definitions = [x for x in terms_definitions.split("\n") if x and 'Term:' in x and 'Definition:' in x]
    # parse the text into a dict
    terms_to_definition = {x.split("Definition:")[0].split("Term:")[-1].strip(): x.split("Definition:")[-1].strip() for x in terms_definitions}
    return terms_to_definition