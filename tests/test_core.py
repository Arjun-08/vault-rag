from src.chunking import chunk_records
from src.evaluation import exact_match,token_f1
def test_chunking():
    x=[{"document_id":"d","document_name":"d.txt","page":1,"text":"one two three four five six seven eight nine ten"}]
    assert len(chunk_records(x,5,1))>=2
def test_em(): assert exact_match("The cat.",["cat"])==1.0
def test_f1(): assert token_f1("the cat",["cat"])>0
