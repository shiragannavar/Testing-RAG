from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import os



# initialize ragresults from json/dict
with open("/Users/adarsh.shiragannavar/Documents/GitHub/Testing-RAG/output.json") as fp:
    rag_results = RAGResults.from_json(fp.read())

# set-up the evaluator
evaluator = RAGChecker(
    extractor_name="openai/gpt-4",
    checker_name="openai/gpt-4",
    batch_size_extractor=10,
    batch_size_checker=10
)

# evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
evaluator.evaluate(rag_results, all_metrics)
print(rag_results)