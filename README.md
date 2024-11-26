Sure, here's the content in Markdown up to the License section:

```markdown
# PDF to Markdown Q&A Generator

This script converts a PDF file into markdown format and uses a Large Language Model (LLM) to generate question-and-answer pairs from the content of each page. The generated Q&A pairs are stored in a CSV file for easy access and further processing.

## Features

- **PDF Conversion**: Extracts text from the first 20 pages of a PDF file and converts it into markdown format.
- **LLM Integration**: Utilizes OpenAI's GPT-4 model to generate three accurate and concise question-and-answer pairs from the text of each page.
- **Data Storage**: Stores the generated Q&A pairs in a CSV file (`qa_output.csv`) for easy access and analysis.
- **Error Handling**: Includes robust error handling to manage JSON parsing errors and other exceptions during processing.

## Requirements

- Python 3.7 or higher
- OpenAI API Key
- The following Python libraries:
  - `langchain`
  - `openai`
  - `pymupdf4llm`
  - `PyPDF2`
  - `pandas`
  - `python-dotenv`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install all required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, install the packages individually:

   ```bash
   pip install langchain openai pymupdf4llm PyPDF2 pandas python-dotenv
   ```

4. **Set Up Environment Variables**

   - Create a `.env` file in the root directory of the project.
   - Add your OpenAI API key to the `.env` file:

     ```dotenv
     OPENAI_API_KEY=your-openai-api-key
     ```

## Usage

1. **Place Your PDF File**

   - Place the PDF file you want to process in the `files/` directory.
   - Ensure the file is named `document.pdf` or update the `input_pdf_path` variable in the script.

2. **Run the Script**

   Execute the script using Python:

   ```bash
   python your_script_name.py
   ```

   Replace `your_script_name.py` with the actual name of the script file.

3. **Output**

   - The script will extract text from the first 20 pages of the PDF (or fewer if the PDF has less than 20 pages).
   - It will generate three question-and-answer pairs for each page.
   - The results will be appended to `qa_output.csv` in the project directory.
   - The markdown version of the PDF content will be printed to the console.

## Configuration

- **Number of Pages to Process**

  To change the number of pages processed, modify the `num_pages_to_copy` variable:

  ```python
  num_pages_to_copy = min(20, len(pdf_reader.pages))  # Change 20 to your desired number
  ```

- **OpenAI Model Settings**

  To change the OpenAI model or adjust the temperature:

  ```python
  llm = ChatOpenAI(model_name='gpt-4', temperature=0)  # Adjust 'model_name' and 'temperature' as needed
  ```

- **File Paths**

  Update the input and output PDF paths if your files are located elsewhere:

  ```python
  input_pdf_path = 'files/your_input_file.pdf'
  output_pdf_path = 'files/your_output_file.pdf'
  ```

## Example Output

An example of the generated Q&A pairs:

```csv
question,answer
"What is the main functionality of this script?","The script converts a PDF file into markdown format and generates question-and-answer pairs using an LLM, storing the results in a CSV file."
"Which libraries are essential for running this code?","The code requires libraries such as langchain, openai, pymupdf4llm, PyPDF2, pandas, and python-dotenv."
"How does the script handle OpenAI API keys?","It uses the python-dotenv library to load the OpenAI API key from a .env file for secure access."
```

## Troubleshooting

- **JSON Parsing Errors**

  If you encounter a JSON parsing error, the script will notify you and print the problematic output. This usually happens if the LLM's response doesn't match the expected JSON format.

- **OpenAI API Key Issues**

  Ensure that your OpenAI API key is correctly set in the `.env` file and that your account has access to the GPT-4 model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
```
