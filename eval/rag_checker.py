import os
import csv
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.runnables import RunnablePassthrough

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from astrapy.info import CollectionVectorServiceOptions
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor


# Load environment variables
load_dotenv()

def get_default_rag_chain(
        astradb_collection: str = "collection",
        collection_vector_service_options = None):
    # Configuration
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    
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
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    # Define the prompt template
    ANSWER_PROMPT = ChatPromptTemplate.from_template(
        """You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

            Context:
            {context}

            Question: "{question}"
            Answer:"""
            )
    chain = {"context":vectorstore.as_retriever(), "question": RunnablePassthrough()} | ANSWER_PROMPT | llm 
    return chain 

def run_eval():
    tracer_provider = register(
        project_name="<unique-id>", 
        endpoint="http://localhost:6006/v1/traces",
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    # start the phoenix server
    # download GT
    # run eval for GT
    # download spans from Phoenix - use the same id
    # construct RAGChecker file format
    # ragchecker

if __name__ == "__main__":
    # Read the CSV file
    questions = []
    with open('qa_output.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            questions.append({'question': row['question'], 'gt_answer': row['answer']})

    results = []

    # Process each question
    for idx, q in enumerate(questions):
        user_question = q['question']
        gt_answer = q['gt_answer']
        query_id = str(idx + 1)  # Starting from 1

        # Retrieve relevant documents
        print(f"Processing question {query_id}: {user_question}")
        relevant_docs = retrieve_documents(user_question)

        # Initialize the response
        final_answer = "No relevant documents found."
        retrieved_context = []

        if relevant_docs:
            # Generate answer using the retrieved context
            final_answer = generate_answer(user_question, relevant_docs)

            # Prepare retrieved_context
            retrieved_context = []
            for doc_idx, doc in enumerate(relevant_docs):
                doc_id = str(doc_idx + 1).zfill(3)  # Start from '001' for every question
                retrieved_context.append({
                    'doc_id': doc_id,
                    'text': doc.page_content
                })

        # Build result entry
        result_entry = {
            'query_id': query_id,
            'query': user_question,
            'gt_answer': gt_answer,
            'response': final_answer,
            'retrieved_context': retrieved_context
        }

        results.append(result_entry)

    # Prepare the final JSON structure
    output = {'results': results}

    # Write to JSON file
    with open('output.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(output, jsonfile, indent=2)

    print("Processing complete. Results saved to output.json.")
