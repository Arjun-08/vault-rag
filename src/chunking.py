def chunk_records(records, chunk_size=650, overlap=100):
    if overlap >= chunk_size: raise ValueError("overlap must be smaller than chunk_size")
    chunks=[]; step=chunk_size-overlap
    for record in records:
        words=record["text"].split()
        for n,start in enumerate(range(0,len(words),step)):
            piece=words[start:start+chunk_size]
            if not piece: break
            chunks.append({"chunk_id":f'{record["document_id"]}_{record["page"]}_{n}',"document_id":record["document_id"],"document_name":record["document_name"],"page":record["page"],"text":" ".join(piece)})
            if start+chunk_size>=len(words): break
    print(f"[CHUNK] Created {len(chunks)} chunks")
    return chunks
