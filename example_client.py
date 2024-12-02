import groundtruth.ground_generator as gg
import eval.rag_checker as rag_checker

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



# Create a RAG chain
checker_metrics  = rag_checker.run_eval( chain=None, 
                     ground_truth_from_astra=True,
                     ground_truth_file=None,
                     phoenix_project_name="testing-rag",)

