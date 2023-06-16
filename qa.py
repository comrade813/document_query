from requests_aws4auth import AWS4Auth
from haystack.document_stores import FAISSDocumentStore

from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.nodes import SentenceTransformersRanker
from haystack.pipelines import Pipeline
from haystack.utils import print_answers

document_store = FAISSDocumentStore(
    faiss_index_path="document_store.faiss"
)

retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                               model_format="sentence_transformers")          
ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False, context_window_size=1000)

p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
p.add_node(component=reader, name="Reader", inputs=["Ranker"])

query = "How much money did Judas agree to betray Jesus for?"
result = p.run(query=query, params={"Retriever": {"top_k": 20}, "Ranker": {"top_k": 10}, "Reader": {"top_k": 5}})

print_answers(result, details="medium")