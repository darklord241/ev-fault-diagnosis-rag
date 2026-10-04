### overview

--- 

Document
    ↓
cleaning
    ↓
chunking
    ↓
embedding
    ↓
vector database
    ↓
retrieval

--- 
- Keep an FAQ as one chunk initially cuz there is no reason to separate them out 
- Split the PDF text into chunks because a page can contain several unrelated pieces of information while keeping the source metadata
- overall
    - target ≈ 500–800 words
    - small overlap ≈ 50–100 words
    - don't split FAQs
    - preserve page number
    - preserve source metadata

--- 
- what is chunking ? => we divide the document into smaller pieces 
- why ? => if we embed the entire document into the db then the retrieved result bury the actual relevant info required | we retrieve the relevant piece of the document only
- how do you decide the size of a chunk ? => small enough for precise retrieval, but large enough to preserve meaning.
- Chunking directly affects the quality of this retrieval.
- types : fixed size, fixed size + overlap, sentence based, paragraph based, semantic, structure aware/hierarchical chunking


### Progression 

- last stage : change this to token-based chunking because embedding/LLM models work in tokens, and token-aware chunking is more principled.
- next immediate : 
    - Structure-aware decides where a logical unit starts/ends.
    - Size-based acts as a constraint when that logical unit becomes too large.
- so for checking the retrieval response we can just not use a db for now cuz there are not many documents right now 

#### 1. fixed size(600) + overlap(100)

- this is just a baseline chunker 

- the faqs come out properly only cuz we r chunking each faq as its own chunk and this comes under 600 
- the problem is the pdf chunking right now is using pages as reference and in the future when the relevant info spans across multiple pages, so then the metadata source needs to hv multiple pg nums 
---
PDF
 ↓
extract pages
 ↓
combine text while preserving page boundaries
 ↓
chunk
 ↓
attach page/source metadata to each chunk
---
raw/processed documents
          ↓
      document type
          ↓
   segmentation strategy
       /          \
     FAQ          PDF
      │            │
   Q + A       structure-aware
                    │
             size constraint
                    │
                 chunks

---
- record = document/page extraction unit
- chunk = retrieval unit
--- 
- what embedding model we're using
- what retrieval model we're using
- whether we need hybrid lexical + vector search
- what metadata filters we need
- how many chunks we'll have
- whether we need parent/child retrieval
- how we'll evaluate retrieval
<br>So designing a rigid PostgreSQL schema now would be premature.
