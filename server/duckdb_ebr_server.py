import heapq
import os

import duckdb

from configs.model import OUTPUT_MODEL_PATH
import proto.ebr_pb2


class DuckEBRServer:

    def __init__(self):
        # build connection with duck db, load the table and index
        self.conn = duckdb.connect(os.path.join(OUTPUT_MODEL_PATH, "movie_len.db"))

    def send_request(self, query, num_candidate):
        """
        Use the DuckDB to perform the ANN search

        There should be some optimization on the multi embedding search but the
        vss extension does not support yet. It seems that if we use the similarity
        search joins the HNSW index is not utilized:

        https://duckdb.org/docs/stable/core_extensions/vss.html#bonus-vector-similarity-search-joins

        Thus we would just do search each document individually and merge the result
        together
        """
        # retrieve the query embedding
        select_sql = f"""
            SELECT movie_id, embedding
            FROM movie_len_embedding_table
            WHERE movie_id IN ({','.join(str(i) for i in query)});
        """
        query_embeddings = self.conn.sql(select_sql).fetchall()

        # use the array_cosine_distance as the distance function because we build
        # the index using cosine metrics, note here that it is the cosine distance,
        # which is 1 - cosine_similarity
        search_sql = """
            SELECT
                movie_id,
                array_cosine_distance(embedding, {query}::FLOAT[64]) as score
            FROM movie_len_embedding_table
            ORDER BY array_cosine_distance(embedding, {query}::FLOAT[64])
            LIMIT {num_candidate}; 
        """
        movie_ids, distances, indices = [], [], []
        for elem in query_embeddings:
            movie_id, embedding = elem[0], elem[1]
            query_result = self.conn.sql(search_sql.format(query=list(embedding), num_candidate=num_candidate)).fetchall()
            movie_ids.append(movie_id)
            indices.append([r[0] for r in query_result])
            distances.append([r[1] for r in query_result])

        # TODO: refactor this into a base class to be shared for each server
        heap, visited = [], set(movie_ids)
        for k in range(len(movie_ids)):
            heapq.heappush(heap, (distances[k][0], k, 0))  # distance, array_index, index
        result = []
        while len(result) < num_candidate:
            # get the next candidate to check
            dist, array_index, index = heapq.heappop(heap)
            candidate = indices[array_index][index]
            if candidate not in visited:
                visited.add(candidate)
                result.append({"id": candidate, "score": float(dist), "source_id": movie_ids[array_index]})
            # push the next candidate from the array into the heap
            next_index = index + 1
            if next_index < len(indices[array_index]):  # check if we have consumed all elements in this array
                heapq.heappush(heap, (distances[array_index][next_index], array_index, next_index))

        # To make the interface consistent, we reuse the entity we have define in protobuf for ebr server
        return [proto.ebr_pb2.SearchResult(id=c["id"], score=c["score"], source_id=c["source_id"]) for c in result]


if __name__ == "__main__":
    server = DuckEBRServer()
    result = server.send_request(query=[1, 2, 3], num_candidate=15)
    print(result)
