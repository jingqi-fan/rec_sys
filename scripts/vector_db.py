import os
import pickle
from time import time

import duckdb
import torch

from configs.model import OUTPUT_MODEL_PATH

"""
This is a simple script to create movie embedding table in DuckDB and use the vss extension 
as the vector search engine. This is similar to the popular vector db such as Pinecone
"""

con = duckdb.connect(os.path.join(OUTPUT_MODEL_PATH, "movie_len.db"))
con.install_extension("vss")  # we need this extension to build HNSW index for ANN
con.load_extension("vss")
con.execute("SET hnsw_enable_experimental_persistence=true;")

# create the embedding table
table_creation_sql = """
    DROP TABLE IF EXISTS movie_len_embedding_table; 

    CREATE TABLE movie_len_embedding_table (
        movie_id INT,
        embedding FLOAT[64]
    );
"""
con.sql(table_creation_sql)

movie_len_embeddings = torch.load(os.path.join(OUTPUT_MODEL_PATH, "movie_embeddings_v2.pt"), weights_only=False)
with open(os.path.join(OUTPUT_MODEL_PATH, "movie_id_remapper.pkl"), "rb") as file:
    movie_id_remapper = pickle.load(file)
batch_size = 100
num_batch = movie_len_embeddings.shape[0] // batch_size
print(f"Total number of batch {num_batch}...")
start_time = time()
for i in range(num_batch):
    batch_idx = [i for i in range(i * batch_size, min((i + 1) * batch_size, movie_len_embeddings.shape[0]))]
    # we need to do a round of filter here to remove the zombie embeddings which does not have movie id associated
    batch_idx = [i for i in batch_idx if i in movie_id_remapper]
    movie_ids = [movie_id_remapper.get(i) for i in batch_idx]  # here we extract the movie ids out
    batch = movie_len_embeddings[batch_idx, :].tolist()
    insert_sql = "INSERT INTO movie_len_embedding_table VALUES "
    values = [f"({movie_id}, {embedding})" for movie_id, embedding in zip(movie_ids, batch)]
    if not values:  # skip if there is no items to insert
        continue
    insert_sql += ",".join(values) + ";"
    con.sql(insert_sql)

    if i % 25 == 0:
        time_eclipsed = time() - start_time
        print(f"Batch {i} finished..., time eclipsed {time_eclipsed:.4f}")

# create index on the table, here we use the HNSW index as the ANN algorithm
# HNSW refers to hierarchy navigable small world algorithm, and here is a good resource
# to learn more about how this algorithm works: https://www.pinecone.io/learn/series/faiss/hnsw/
index_creation_sql = """
    CREATE INDEX movie_len_embedding_hnsw_index
    ON movie_len_embedding_table
    USING HNSW(embedding)
    WITH (metric = 'cosine');
"""
con.sql(index_creation_sql)

# simple query check
select_sql = f"""
    SELECT movie_id, array_cosine_distance(embedding, {movie_len_embeddings[0].tolist()}::FLOAT[64]) as score
    FROM movie_len_embedding_table
    ORDER BY array_cosine_distance(embedding, {movie_len_embeddings[0].tolist()}::FLOAT[64])
    LIMIT 10;
"""
result = con.sql(select_sql).fetchall()
print(result)

con.close()
