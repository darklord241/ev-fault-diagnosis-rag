### scrapper.py
- it was first made for getting data from both webpages n pdfs but ather official websites do not accept data mining and hence we received 403 forbidden source error 
-  `pathlib` is a library for managing the filesystem paths and is used here mainly to `mkdir` make dir and  write files 
- sources stores the URLs of the official sources that the scraper is intended to collect. This keeps the source list separate from the downloading logic and makes it easier to reproduce or modify the collection process.
- i changed the code in `make_filename` func cuz the old one which is commented out is complicated 
- in `get_extension` function, `.bin` is binary indication and hence all other extensions are just specialised and hence all files can also be kept as .bin 
- the use of dictionary is actually seen a lot 
- `response.raise_for_status()` func checks the status code and raises an exception if it's an error.
- `get_extension()` is used only for naming the content taken and why cant i just leave it like tht instead of adding extension ? : the extension is useful because it tells both you and software what the file is
- do i need to leave space here `raw_path = RAW_DIR / filename` like beside / : it is convention it seems 
- what is with and what is open like which lib func is it? : `with open(...) as file:` is the standard safe way to work with files cuz if no with then i have to close the file opened and open is a built in func 
- we are using `wb` which is write binary cuz some files dont just contain text but also other things
- what is raw_path.stem ? : it is related to the file name 
- why am i adding delay ? and if so then what is 2 indicating here ? : after finishing one download. and 2 is 2 seconds

### process_html.py 
- what is bs4 ? : python package called beautiful soup 4 used for parsing html/xml files 
- what is shutil ? : `pathlib` is used for managing file paths and checking metadata, while `shutil` is used for high-level file operations like copying, moving, and deleting files or directories.

### inspection of the html pages
- i am taking only faq from these pages now and the way they are structured are different so i have handle them separately 
- this structure is then used to parse into clean data with only the important bits 
- i am only currently taking faqs cuz they r properly structured and easier to parse than other stuff 

### document_processor.py
``` 
    print("FAQ data found.")

    for category, groups in faq_data.items():
        print(f"\nCategory: {category}")

        for group in groups:
            print(f"  Group: {group['name']}")
            print(f"  Questions: {len(group['questions'])}")
```
- the first version of it just checks if the faq are actually present and possible to be extracted and yeah there are 349 faqs across the 6 html pages 
```
    count = 0
    for category, groups in faq_data.items():
        for group in groups:
            for question in group["questions"]:
                print()
                print("Category:", category)
                print("Group:", group["name"])
                print("Question:", question["question"])
                print("Answer:", question["answer"])

                count += 1
                if count == 5:
                    return
```
- the second version just checks the html structure around the question so as to know what to remove and how 
- the third version then checks if the output is cleaned by adding `clean_html()` func which works using soup
- (committed) the fourth version extracts the faq and also error checks for none answer and saves it in `JSONL` format where each faq is across a line instead of in json sub objects format 
- the next version which was done before the commit checks the topicData

## Current Status

### Completed

- [x] Identified initial official Ather sources
- [x] Implemented automated PDF collection
- [x] Identified HTTP 403 responses for automated HTML requests
- [x] Collected required HTML pages through browser saving
- [x] Organized raw documents and metadata
- [x] Implemented process-html.py
- [x] Generated metadata for collected HTML documents
- [x] Inspected embedded HTML/JavaScript structure
- [x] Verified FAQ data exists in structured JSON
- [x] Verified the structure of all six HTML FAQ pages
- [x] Identified faqData structure for the main FAQ page
- [x] Identified topicData structure for the other FAQ pages
- [x] Implemented document_processor.py
- [x] Extracted structured FAQ records
- [x] Cleaned HTML from question/answer fields
- [x] Added handling for FAQ records with missing answers
- [x] Saved processed FAQ data in JSONL format
- [x] Processed PDF documents into page-aware text
- [x] Saved processed PDF pages in JSONL format

### In Progress

- [ ] Design common processed-document format
- [ ] Implement chunking
- [ ] Design database schema
- [ ] Store processed data in PostgreSQL + pgvector