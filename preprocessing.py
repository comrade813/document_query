from typing import Tuple
from haystack.nodes import TextConverter
from haystack.nodes import PreProcessor
from haystack.nodes import EmbeddingRetriever
import json
 
# Opening JSON file
settings = ""
with open('data.json') as json_file:
    settings = json.load(json_file)

file_name = settings["file"]
txt_converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
doc_web = txt_converter.convert(file_path=f"data/{file_name}.txt", meta=None)[0]

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=settings["document_length"],
    split_overlap=settings["document_length"]/2,
    split_respect_sentence_boundary=True,
    sent_end_chars=("\n"),
)

docs_txt = preprocessor.process([doc_web])

print(f"n_docs_input: 1\nn_docs_output: {len(docs_txt)}")

def set_passage_title(s: str) -> Tuple[str, str]:
    verses = s[:-1].split('\n')
    
    begin = verses[0].split('\t')[0]
    end = verses[-1].split('\t')[0]
    title = f"{begin} - {end} ({file_name})"

    for i in range(0, len(verses)):
        verses[i] = verses[i].split('\t')[1]

    return title, '\n'.join(verses)

from progress.bar import Bar

bar = Bar('Processing', max=len(docs_txt))

for i in range(0, len(docs_txt)):
    docs_txt[i].meta["title"], docs_txt[i].content = set_passage_title(docs_txt[i].content)
    bar.next()

bar.finish()


from haystack.document_stores import FAISSDocumentStore

db = settings["db"]
user = db["user"]
password = db["password"]
host = db["host"]
port = db["port"]
table = db["relation"]
document_store = FAISSDocumentStore(
    sql_url = f"postgresql://{user}:{password}@{host}:{port}/{table}"
)

document_store.write_documents(docs_txt)

print("Finished writing documents")

retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
                               model_format="sentence_transformers")

document_store.update_embeddings(retriever)
document_store.save(index_path="document_store.faiss")