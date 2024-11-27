from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import pandas as pd
from pydantic import BaseModel, Field
from astrapy import DataAPIClient

# Set your OpenAI API key
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pydantic import BaseModel, Field

class GroundTruth(BaseModel):
    question: str = Field(..., title="Question")
    answer: str = Field(..., title="Answer")

class GroundTruthResponse(BaseModel):
    qa_pairs: list[GroundTruth] = Field(..., title="List of question and answer pairs")
    
def generate_ground_truth(docs: list, 
                          save_to_file: bool = True, 
                          file_name: str = 'qa_output.csv',
                          save_to_AstraDB: bool = False,
                          astra_collection_name: str = 'ground_truth'):

    prompt = PromptTemplate(
        input_variables=["doc"],
        template="""
        Analyze the given text, generate accurate question and answer pairs from the given text only. 
        Scope of question and answer should be solely based on the given text
        Generate atleast 1, upto 3 question, answer pairs for each text.
        
        Here is the text:
        {doc}
        """
        )
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    llm = llm.with_structured_output(GroundTruthResponse)
    chain = prompt | llm
    qa_list = []
    
    for idx, doc in enumerate(docs):
        try:
            output: GroundTruthResponse = chain.invoke(doc)          
            qa_list.extend(output.qa_pairs)        
        except Exception as e:
            print(f"An error occurred at index {idx}: {e}")
            continue
    
    if save_to_file:
        data = [{'question': g.question, 'answer': g.answer} for g in qa_list]
        df = pd.DataFrame(data)
        file_exists = os.path.isfile(file_name)
        df.to_csv(file_name, mode='a', index=False, header=not file_exists)

    if save_to_AstraDB:
        client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
        database = client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])
        collection = database.create_collection(astra_collection_name, check_exists=False)        
        collection.insert_many([{'question': g.question, 'answer': g.answer} for g in qa_list])

    return qa_list