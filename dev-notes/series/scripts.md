### overview 
this is for listing out what each python script does and how the overall picture looks like 

### just additional data overview 
~/el/ev-fault-diagnosis-rag/
└── scraper/
    └── data/
        ├── incoming/    # Newly collected files awaiting processing
        ├── raw/         # Raw HTML/PDF source files
        ├── metadata/    # Source sidecar metadata
        ├── processed/   # Extracted FAQ records and PDF page records
        └── chunked/     # Final chunk records used as retrieval input
        


### collection and staging 
- `scraper.py` : downloads documents from configured links from Ather Page using HTTP requests. it saves the response files in `data/raw` and writes metadata sidecars 
- `process-html.py` : This is an alternative collection route, useful when direct automated downloads are blocked.

### extraction and normalization 
- `document_processor.py` : 
    - Reads raw HTML and extracts embedded FAQ/topic data, including structured data such as faqData and topicData. It cleans the HTML content and writes processed FAQ records in JSONL format
    - It also uses provenance.py to attach source metadata.
- `pdf-processor.py` : 
    - Uses the external pdftotext utility to extract text from PDFs. It splits the extracted output using page boundaries and writes processed JSONL records, typically one record per PDF page.
    - Like the HTML processor, it attaches provenance metadata.
- `provenance.py` : 
    - this is not a pipeline stage which run by itself but is used by other scripts to load metadata sidecars 
    - the common source-of-truth mechanism that preserves where each piece of information came from.

### chunking 
- `chunker.py` : 
    Reads the processed JSONL records and writes chunked JSONL records.
    - FAQ records are kept as individual chunks.
    - PDF page records are split using the existing 600-word chunk size and 100-word overlap.
    - Provenance is carried forward into the resulting chunks.
    Think of it as: preparing appropriately sized text units for a future embedding and retrieval system.
    - Why does this matter? If you eventually embed an entire long manual as one giant text block, retrieval may be too coarse. Chunks let the retrieval system find a more specific passage relevant to a user's fault description.

### validation and diagnostic 
- `validate_pipeline.py` : 
    - Checks data consistency, provenance, sidecar matching, record and chunk IDs, content quality, and reconciliation between processed records and chunks.
    - Role: your main integrity checker. It helps catch missing metadata, inconsistent IDs, and data-quality problems.

- `analyze_chunks.py` : 
    - Despite its name, it currently reads data/processed/, not data/chunked/, and reports word-count statistics.
    - Role: exploratory data analysis. 

- `inspect-html.py` :
    - Examines raw HTML and searches for titles and FAQ-related script content.
    - Role: debugging the HTML extraction process, especially when the website's structure isn't obvious.

- `test-pdf.py` :
    - A manual PDF experiment using pypdf. It opens a hard-coded PDF and prints sample extracted text.
    - Role: experimenting with PDF extraction—not an automated unit test of your main PDF processor.

- `test_provenance.py` : 
    - Contains standard-library unittest tests for provenance loading and normalization, along with checks that chunking preserves provenance and IDs.
    - Role: automated regression tests that help ensure future changes don't break established behavior.

