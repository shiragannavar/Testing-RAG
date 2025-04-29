import os
from logging_config import setup_logging
import logging
from datetime import datetime

# Configure structured logging
setup_logging()
logger = logging.getLogger(__name__)
import csv
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from astrapy.info import CollectionVectorServiceOptions
from phoenix.otel import register
import phoenix as px 
from openinference.instrumentation.langchain import LangChainInstrumentor
import pandas as pd 
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import tempfile, json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

def get_default_rag_chain(
        astradb_collection: str = "movies",
        collection_vector_service_options = None):
    # Configuration
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    #print(OPENAI_API_KEY)
    vectorstore = None 
    if collection_vector_service_options is not None: 
        vectorstore = AstraDBVectorStore(
            collection_vector_service_options=collection_vector_service_options,
            collection_name=astradb_collection,
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_ENDPOINT,
        )
    else:
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name=astradb_collection,
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_ENDPOINT,
        )
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",  # specify the model name here
        temperature=0
    )

    # Define the prompt template
    ANSWER_PROMPT = ChatPromptTemplate.from_template(
        """You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

        Context:
        {context}

        Question: "{question}"
        Answer:"""
    )

    # Create the chain
    chain = (
            {"question": RunnablePassthrough(), "context": vectorstore.as_retriever()}
            | ANSWER_PROMPT
            | llm
    )

    return chain

def run_eval(chain, ground_truth_file, run_name="rag-eval"):    
    qa_df = pd.read_csv(ground_truth_file)    
    for idx,row in qa_df.iterrows():
        q = row['question']
        logger.info("Asking question: %s", q)
        answer = chain.invoke(q)
        logger.info("Question answered: %s -> %s", q, answer)
    
def start_phoenix_session(project_name = "rag-eval"):
    try:
        session = px.launch_app()    
    except Exception:
        logger.warning("Phoenix already running, retrieving active session")
        session = px.active_session()
        logger.info("Retrieved active Phoenix session: %s", session)

    tracer_provider = register(
            project_name=project_name, 
            endpoint="http://localhost:6006/v1/traces",
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)    
    return session

def get_ragchecker_input(session: px.Session, 
                        phoenix_project_name: str, 
                        ground_truth_file: str,
                        ragchecker_file: str = 'output.json'):
    qa_df = pd.read_csv(ground_truth_file)
    gt_map = {}
    for idx,row in qa_df.iterrows():
        # Ensure we have string values even if original is float or NaN
        q = str(row['question'])
        a = str(row['answer'])
        gt_map[q.strip()] = a.strip()
    if session is None:
        session = px.Client(endpoint="http://localhost:6006")
    spans = session.get_spans_dataframe(project_name=phoenix_project_name) 
    trace_ids = spans["context.trace_id"].unique()
    rag_checker_results = []
    for trace_id in trace_ids:        
        question = spans["attributes.output.value"][spans["context.trace_id"]==trace_id][0].strip()
        answer = spans["attributes.output.value"][spans["context.trace_id"]==trace_id][-1].strip()
        context = spans["attributes.output.value"][ (spans["context.trace_id"]==trace_id) & (spans["span_kind"]=="RETRIEVER") ][0]
        docs = [ {"doc_id":idx, "text":doc} for idx,doc in enumerate(json.loads(context)["documents"])]
        result_entry = {
            'query_id': trace_id,
            'query': question,
            'gt_answer': gt_map[question],
            'response': answer,
            'retrieved_context': docs
        }        
        rag_checker_results.append(result_entry)
    output = {'results': rag_checker_results}
    with open(ragchecker_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(output, jsonfile, indent=2)
    logger.info("Wrote RAGChecker input to %s", ragchecker_file)
    return rag_checker_results

def compute_ragchecker_metrics(input_file_name, metrics_file_name, graph_file_name="metrics-graph.json"):
    with open(input_file_name) as fp:
        rag_results = RAGResults.from_json(fp.read())
    evaluator = RAGChecker(
        extractor_name="openai/gpt-4o-mini-2024-07-18",
        checker_name="openai/gpt-4o-mini-2024-07-18",
        batch_size_extractor=5,
        batch_size_checker=5
    )
    evaluator.evaluate(rag_results, all_metrics)    
    if metrics_file_name is not None: 
        with open(metrics_file_name, "w") as f:
            json.dump(rag_results.metrics, f)
    # Update metrics graph file with timestamped metrics
    timestamp = datetime.utcnow().isoformat() + "Z"
    # Load existing graph data or initialize structure
    if os.path.exists(graph_file_name):
        with open(graph_file_name) as gf:
            graph_data = json.load(gf)
    else:
        graph_data = {"timestamps": [], "overall_metrics": {}, "retriever_metrics": {}, "generator_metrics": {}}
    # Append timestamp
    graph_data["timestamps"].append(timestamp)
    # Append new metric values to corresponding lists
    for group in ["overall_metrics", "retriever_metrics", "generator_metrics"]:
        metrics_group = rag_results.metrics.get(group, {})
        gd_group = graph_data.setdefault(group, {})
        for key, value in metrics_group.items():
            gd_group.setdefault(key, []).append(value)
    # Write updated graph data
    with open(graph_file_name, "w") as gf:
        json.dump(graph_data, gf, indent=2)
    logger.info(f"Updated metrics graph file {graph_file_name} at {timestamp}")
    return rag_results.metrics
