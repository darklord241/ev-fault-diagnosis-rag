### scrapper.py
- it was first made for getting data from both webpages n pdfs but ather official websites do not accept data mining and hence we received 403 forbidden source error 
-  `pathlib` is a library for managing the filesystem paths and is used here mainly to `mkdir` make dir and  write files 
- sources is an object used to save the official sources in case of any error in future
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
- what is bs4 ? : 
- what is shutil ? : `pathlib` is used for managing file paths and checking metadata, while `shutil` is used for high-level file operations like copying, moving, and deleting files or directories.