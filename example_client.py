import groundtruth.ground_generator as gg
## Generate Groundtruth
movies = [
  {
    "id": 653346,
    "title": "Kingdom of the Planet of the Apes",
    "poster_path": "/gKkl37BQuKTanygYQG1pyYgLVgf.jpg",
    "content": "Kingdom of the Planet of the Apes\n\n\n     \n\n\n\n\n\nSeveral generations following Caesar's reign, apes \u2013 now the dominant species \u2013 live harmoniously while humans have been reduced to living in the shadows. As a new tyrannical ape leader builds his empire, one young ape undertakes a harrowing journey that will cause him to question all he's known about the past and to make choices that will define a future for apes and humans alike.\n\nLooks like we're missing the following data in ms-SG or en-US...\n\n\n\n\n\n"
  },
  {
    "id": 573435,
    "title": "Bad Boys: Ride or Die",
    "poster_path": "/nP6RliHjxsz4irTKsxe8FRhKZYl.jpg",
    "content": "Bad Boys: Ride or Die\n\n\n     \n\n\n\n\n\nAfter their late former Captain is framed, Lowrey and Burnett try to clear his name, only to end up on the run themselves.\n\nLooks like we're missing the following data in ms-SG or en-US...\n\n\n\n\n\n"
  }
]

texts = [ movie["content"] for movie in movies ]
df = gg.generate_ground_truth(texts, save_to_AstraDB=True, save_to_file=True)
print(df)


### Generate RAGChecker file

import time 
import eval.rag_checker as rc

chain = rc.get_default_rag_chain( astradb_collection='movies')
project_name = f"my-eval-app.{time.time()}"
print("Project Name: ", project_name)
session = rc.start_phoenix_session(project_name=project_name)
rc.run_eval(chain, "qa_output.csv")

# project_name="my-eval-app.1733305345.8140092"
ragchecker = rc.get_ragchecker_file(None, project_name, "qa_output.csv")

from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import tempfile, json

with open("output.json") as fp:
  rag_results = RAGResults.from_json(fp.read())

# set-up the evaluator
evaluator = RAGChecker(
  extractor_name="openai/gpt-4-turbo",
  checker_name="openai/gpt-4-turbo",
  batch_size_extractor=10,
  batch_size_checker=10
)

# evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
evaluator.evaluate(rag_results, all_metrics)
print(rag_results)
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
with open(temp_file.name, "w") as f:
    json.dump(rag_results.metrics, f)

from flask import Flask, render_template

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

