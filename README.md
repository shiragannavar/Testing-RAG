# Ground Truth Generator and RAG Evaluator 

This repository provides a framework to simplify the generation of ground truth datasets and the evaluation of Retrieval-Augmented Generation (RAG) applications. It aims to assist developers in building and testing RAG applications with higher accuracy and reliability by automating ground truth creation and providing comprehensive evaluation metrics.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Ground Truth Data](#generate-ground-truth-data)
  - [Evaluate RAG Application](#evaluate-rag-application)
  - [View Evaluation Metrics](#view-evaluation-metrics)
- [Framework Overview](#framework-overview)
- [AstraDB Integration](#astradb-integration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Building RAG applications for testing is straightforward. However, developing production-ready RAG applications with high accuracy requires grounding with organizational data, which can be challenging. Developers often face difficulties in:

- **Creating Ground Truth Datasets**: Developers find creating ground truth dataset difficult. Manually building datasets is time-consuming and prone to errors.
- **Evaluating RAG Applications**: Assessing applications in terms of accuracy, relevance, hallucination, etc.
- **Data Management**: Storing and updating ground truth datasets efficiently.

Our framework addresses these challenges by providing tools to:

- **Easily Generate Ground Truth Datasets**: Automate the creation of question-answer pairs from your data.
- **Simplify Evaluation of RAG Applications**: Use built-in methods to evaluate and test your applications.
- **Offer LLM-Agnostic Solutions**: Compatible with various Large Language Models (LLMs).
- **Integrate with AstraDB**: Use AstraDB as a ground truth dataset store, eliminating the need for additional databases.

## Features

- **Automated Ground Truth Generation**: Generate datasets from your documents effortlessly.
- **Comprehensive Evaluation Metrics**: Evaluate applications on accuracy, relevance, hallucination, and more.
- **AstraDB Storage**: Store and manage your datasets directly in AstraDB.
- **Flask Web Interface**: Visualize evaluation metrics via a simple web application.
- **LLM-Agnostic Framework**: Compatible with different Large Language Models.

## Prerequisites

- Python 3.7 or higher
- OpenAI API Key
- AstraDB Application Token and Endpoint
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shiragannavar/Testing-RAG.git
   cd testing-rag
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add the following:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ASTRA_DB_APPLICATION_TOKEN=your_astradb_token
   ASTRA_DB_API_ENDPOINT=your_astradb_endpoint
   ```

## Usage

```python
python -m example_client.py
```

## Code Explanation

### Generate Ground Truth Data

Use/Modify the `groundtruth.ground_generator` module to generate ground truth question-answer pairs from your documents.

```python
import groundtruth.ground_generator as gg

# Sample documents
movies = [
    {
        "id": 653346,
        "title": "Kingdom of the Planet of the Apes",
        "content": "Several generations following Caesar's reign, apes – now the dominant species – live harmoniously while humans have been reduced to living in the shadows..."
    },
    {
        "id": 573435,
        "title": "Bad Boys: Ride or Die",
        "content": "After their late former Captain is framed, Lowrey and Burnett try to clear his name, only to end up on the run themselves..."
    }
]

# Extract content
texts = [movie["content"] for movie in movies]

# Generate ground truth data
qa_list = gg.generate_ground_truth(
    texts,
    save_to_AstraDB=True,
    save_to_file=True
)

print(qa_list)
```

This code:

- Extracts the `content` field from each movie in the `movies` list.
- Generates question-answer pairs using the `generate_ground_truth` function.
- Saves the generated pairs to a CSV file (`qa_output.csv`) and stores them in AstraDB.

### Evaluate RAG Application

Use the `eval.rag_checker` module to evaluate your RAG application.

```python
import time
import eval.rag_checker as rc

# Initialize RAG chain with AstraDB as vector store
chain = rc.get_default_rag_chain(astradb_collection='movies')

# Start Phoenix session for tracing
project_name = f"my-eval-app.{time.time()}"
print("Project Name: ", project_name)
session = rc.start_phoenix_session(project_name=project_name)

# Run evaluation using the generated ground truth data
rc.run_eval(chain, "qa_output.csv")

# Generate RAGChecker file with evaluation results
ragchecker = rc.get_ragchecker_file(None, project_name, "qa_output.csv")
```

This code:

- Initializes a RAG chain that retrieves context from AstraDB.
- Starts a Phoenix session for tracing and monitoring the evaluation process.
- Runs the evaluation using the ground truth data from `qa_output.csv`.
- Generates an `output.json` file containing the evaluation results.

### View Evaluation Metrics

Use the following code to evaluate the results and display the metrics via a Flask web application.

```python
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import tempfile, json

# Load evaluation results
with open("output.json") as fp:
    rag_results = RAGResults.from_json(fp.read())

# Set up the evaluator
evaluator = RAGChecker(
    extractor_name="openai/gpt-4-turbo",
    checker_name="openai/gpt-4-turbo",
    batch_size_extractor=10,
    batch_size_checker=10
)

# Evaluate results with selected metrics
evaluator.evaluate(rag_results, all_metrics)
print(rag_results)

# Save metrics to a temporary file
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
with open(temp_file.name, "w") as f:
    json.dump(rag_results.metrics, f)
```

#### Flask Application

```python
from flask import Flask, render_template
import json

# Flask application
app = Flask(__name__)

@app.route("/")
def display_metrics():
    # Read metrics from the temporary file
    with open(temp_file.name, "r") as f:
        metrics = json.load(f)
    return render_template("metrics.html", metrics=metrics)

if __name__ == "__main__":
    # Ensure Flask app starts after the rag_results are processed
    app.run(debug=False)
```

This code:

- Loads the evaluation results from `output.json`.
- Uses `RAGChecker` to compute various evaluation metrics.
- Saves the metrics to a temporary JSON file.
- Sets up a Flask web application to display the metrics.

Run the Flask app and navigate to `http://localhost:5000/` to view the evaluation metrics.

## Framework Overview

Our framework simplifies the process of building and evaluating RAG applications by:

- **Ground Truth Generation**: Automating the creation of ground truth datasets from your documents.
- **Evaluation Metrics**: Providing built-in methods to evaluate your RAG applications using various metrics.
- **Data Storage**: Integrating with AstraDB to store and manage your datasets efficiently.

### Why Use This Framework?

- **Ease of Use**: Generate datasets and evaluate applications with minimal code.
- **LLM Agnostic**: Compatible with different Large Language Models, allowing flexibility.
- **Strategic Development**: Use evaluation results to develop strategies for improving your applications.
- **No Additional Databases Needed**: Store your datasets in AstraDB, eliminating the need for other databases.

## AstraDB Integration

AstraDB is used as the ground truth dataset store, allowing developers to store and update the grounding dataset as new data is added to the knowledge repository.

- **Setup**:
  - Ensure you have an AstraDB account and obtain the application token and API endpoint.
  - Set the environment variables `ASTRA_DB_APPLICATION_TOKEN` and `ASTRA_DB_API_ENDPOINT`.

- **Usage**:
  - The ground truth generator saves QA pairs to AstraDB when `save_to_AstraDB` is set to `True`.
  - The RAG chain uses AstraDB as the vector store for retrieving context during evaluation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the Apache-2.0.

---

By using this framework, developers can focus on improving their RAG applications without worrying about the complexities of data generation and evaluation. It streamlines the entire process, from generating ground truth data to evaluating application performance, making it easier to build high-accuracy, production-ready RAG applications.
