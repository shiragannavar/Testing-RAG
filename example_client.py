import json 
import groundtruth.ground_generator as gg

## Generate Groundtruth for a dataset
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

# texts = [ movie["content"] for movie in movies ]
# df = gg.generate_ground_truth(texts, save_to_AstraDB=True, save_to_file=True)
# print(df)


### Generate RAGChecker Metrics

import time 
import eval.rag_checker as rc

chain = rc.get_default_rag_chain( astradb_collection='movies')
project_name = f"my-eval-app.{time.time()}"
ragchecker_file = "ragchecker_input.json"
metrics_file_name = "metrics.json"
ground_truth_file = "gt.csv"

session = rc.start_phoenix_session(project_name=project_name)
rc.run_eval(chain, ground_truth_file)
rc.get_ragchecker_input(session=None, 
                        phoenix_project_name=project_name, 
                        ground_truth_file=ground_truth_file,
                        ragchecker_file=ragchecker_file)
rc.compute_ragchecker_metrics(input_file_name=ragchecker_file, 
                              metrics_file_name=metrics_file_name)

from flask import Flask, render_template

# Flask application
app = Flask(__name__)

@app.route("/")
def display_metrics():    
    with open(metrics_file_name, "r") as f:
        metrics = json.load(f)
    return render_template("metrics.html", metrics=metrics)

if __name__ == "__main__":
    # Ensure Flask app starts after the rag_results are processed
    app.run(debug=False)

