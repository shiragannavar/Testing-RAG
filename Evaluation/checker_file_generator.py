import os
import csv
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import AstraDB
from langchain_astradb import AstraDBVectorStore

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_ENDPOINT"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTRA_DB_COLLECTION = "pdf_vector"

# Initialize embedding model
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize AstraDB vector store
vectorstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
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

def retrieve_documents(query, top_k=4):
    """Retrieve top_k documents relevant to the query from AstraDB vector store."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    return docs

def generate_answer(question, context_documents):
    """Generate answer using LLM and provided context."""
    context = "\n\n".join([doc.page_content for doc in context_documents])
    chain = LLMChain(
        llm=llm,
        prompt=ANSWER_PROMPT,
    )
    answer = chain.run(context=context, question=question)
    return answer

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
