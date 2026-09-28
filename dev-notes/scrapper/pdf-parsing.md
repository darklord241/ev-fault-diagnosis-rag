### test-pdf.py
- **pypdf** : A Python package used for reading and working with PDF files.
- `PdfReader` opens a PDF using its path and creates a reader object representing the document. The pages attribute provides access to the individual pages, 
- `extract_text()` is used to extract text from each page. `enumerate()` is used to keep track of the page number while iterating through the pages. 

### pdf-proccesor.py
- `subprocess.run()` : Executes an external program from Python. Here it is used to run the pdftotext command and capture its extracted text.

- `\f` : The form-feed character. pdftotext uses it as a page separator in its output. Splitting the extracted text using \f therefore gives a list where each element corresponds to one PDF page.
- Saving the extracted pages: Each page is stored as a JSONL record containing the source document name, source type, page number, and extracted text. The metadata allows the retrieved information to later be traced back to its original document and page. 