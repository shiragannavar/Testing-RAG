from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import pandas as pd
from pydantic import BaseModel, Field
from astrapy import DataAPIClient
import google.generativeai as genai
import json

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
        Generate atleast 1, upto 3 question, answer pairs for each text. Do not create one work answers. All questions and answers should be in english.
        
        Here is the text:
        {doc}
        """
        )
    llm = ChatOpenAI(model_name='gpt-4o-mini-2024-07-18', temperature=0)
    llm = llm.with_structured_output(GroundTruthResponse)
    chain = prompt | llm
    qa_list = []
    i=0
    for idx, doc in enumerate(docs):
        try:
            i=i+1
            output: GroundTruthResponse = chain.invoke(doc)          
            qa_list.extend(output.qa_pairs)
            print(f"Generating {i}")
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


def generate_ground_truth_flash(images: list,
                                save_to_file: bool = True,
                                file_name: str = 'qa_output.csv',
                                save_to_AstraDB: bool = False,
                                astra_collection_name: str = 'ground_truth'):
    """
    This function takes a list of images in base64 format and queries the Google Gemini model (Flash)
    to generate Q/A pairs from the image content. The functionality (saving to file and AstraDB)
    is similar to generate_ground_truth.
    """

    # Initialize the Gemini model
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Prepare a prompt that instructs the model to return structured JSON output
    # We include a clear instruction to return the result in a JSON format that matches GroundTruthResponse.
    base_prompt = """
    Analyze the given image and generate accurate question and answer pairs solely based on the image content.
    Generate at least 3 and upto 10 question-answer pairs. Do not generate one-word answers. 
    All questions and answers should be in English. Be as detailed as possible and the answer should be available in the image content.
    In the question, do not mention document, page number, based on image or based on context, provided text etc. 
    Return the result as a JSON object strictly in the following format:
    {
      "qa_pairs": [
        {
          "question": "QUESTION_TEXT",
          "answer": "ANSWER_TEXT"
        }
      ]
    }
    """

    qa_list = []
    i = 0
    for idx, image_b64 in enumerate(images):
        try:
            i += 1
            # We send both the image and the instructions. The API expects a list of inputs.
            # The first element is the image in a dict specifying mime type and data.
            # The second element is the prompt string.
            response = model.generate_content([
                {'mime_type': 'image/png', 'data': image_b64},
                base_prompt
            ])

            # response.text should contain the model's output as text
            # We expect a JSON structure. Let's parse it.
            try:
                raw_text = response.text.strip()
                # Remove code fences if they exist
                if raw_text.startswith("```"):
                    # Remove the first line of backticks
                    raw_text = raw_text.split('\n', 1)[1]
                if raw_text.endswith("```"):
                    # Remove the trailing backticks
                    raw_text = raw_text.rsplit('\n', 1)[0]

                # Now parse the cleaned JSON
                data = json.loads(raw_text)

                # data = json.loads(response.text)
                # Validate with our Pydantic model
                ground_truth_response = GroundTruthResponse(**data)
                qa_list.extend(ground_truth_response.qa_pairs)
            except json.JSONDecodeError:
                print(f"Could not decode JSON at index {idx}: {response.text}")
            except Exception as e:
                print(f"Validation error at index {idx}: {e}")

            print(f"Generating {i}")
            if save_to_file and qa_list:
                data = [{'question': g.question, 'answer': g.answer} for g in qa_list]
                df = pd.DataFrame(data)
                file_exists = os.path.isfile(file_name)
                df.to_csv(file_name, mode='a', index=False, header=not file_exists)

            if save_to_AstraDB and qa_list:
                client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
                database = client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])
                collection = database.create_collection(astra_collection_name, check_exists=False)
                collection.insert_many([{'question': g.question, 'answer': g.answer} for g in qa_list])
        except Exception as e:
            print(f"An error occurred at index {idx}: {e}")
            continue


    print("Thanks")
    return qa_list