from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import os
import pandas as pd
import json
import pymupdf4llm

# Set your OpenAI API key
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#####
#Extract first 10 pages
#####

import PyPDF2

input_pdf_path = 'files/document.pdf'
output_pdf_path = 'files/output.pdf'

with open(input_pdf_path, 'rb') as input_file:
    pdf_reader = PyPDF2.PdfReader(input_file)
    pdf_writer = PyPDF2.PdfWriter()

    num_pages_to_copy = min(20, len(pdf_reader.pages))

    for page_num in range(num_pages_to_copy):
        page = pdf_reader.pages[page_num]
        pdf_writer.add_page(page)

    with open(output_pdf_path, 'wb') as output_file:
        pdf_writer.write(output_file)

md_text = pymupdf4llm.to_markdown(
    "files/output.pdf",
    page_chunks=True,
    # write_images=True,
)
print(md_text)

texts = [item['text'] for item in md_text]

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a helpful assistant that generates three accurate and concise question and answer pairs from the given text only. Do not add anything additional to answer the question.

Given the following text:

{text}

Please create a question that accurately reflects the key information in the text, and provide a detailed answer. Output your response in JSON format as:

{{
    "question": "Your generated question",
    "answer": "Your generated answer"
}},
{{
    "question": "Your generated question",
    "answer": "Your generated answer"
}},
{{
    "question": "Your generated question",
    "answer": "Your generated answer"
}}
"""
)

llm = ChatOpenAI(model_name='gpt-4', temperature=0)

chain = LLMChain(llm=llm, prompt=prompt)

qa_list = []

i=0
for idx, text in enumerate(texts):
    try:
        print("Processing")
        # print(text)
        output = chain.invoke(text)
        print("End")
        generated_text = output['text']

        fixed_generated_text = '[' + generated_text + ']'

        qa_pairs = json.loads(fixed_generated_text)

        qa_list.extend(qa_pairs)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error at index {idx}: {e}")
        print("Generated text was:")
        print(generated_text)
        continue
    except Exception as e:
        print(f"An error occurred at index {idx}: {e}")
        continue

df = pd.DataFrame(qa_list)

csv_file = 'qa_output.csv'
file_exists = os.path.isfile(csv_file)

df.to_csv(csv_file, mode='a', index=False, header=not file_exists)

print(df)
