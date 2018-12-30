So getting back to something very much in use from a business angle. Often we are left with a whole directory of PDF files and as Data scientist our job might be to extract usefull information from this pile of documents. Our first aim would be to find a good library or code snippet that would help us to extract pdf into python.  Below is the list that i compiled after a quick couple of hours of search in internet. 

* pyPDF2 module
* textract 
* tika package
* pdftotext
* tesseract
* python-Docx
* pdfminer

Lets look into how to use each library

***pyPDF2*** -  this blog post helped me with the initial code structure to extract pdf  [pyPDF2 sample code](https://www.blog.pythonlibrary.org/2018/06/07/an-intro-to-pypdf2).

this is a decent library with good resources to start around, but i'm yet to figure out a way to get all the texts in a python list format a bit more research is needed. I would get back to this library if i'm running out of luck with all the other libraries. 

***textract*** - i just ran out of luck in the very first step which is installing the library.  it threw some dependency error, I figured out that this repo is not being updated for quite some time. I still tried to install it by manually installing dependencies but without much luck. so after a good half an hour of research i decided to move on. 

***tika package*** -  this is the best python library for extracting text from PDF. i referred this blog post for getting my head around this library [Tika extract PDF](https://cbrownley.wordpress.com/2016/06/26/parsing-pdfs-in-python-with-tika/).  as per my preliminary research this appears to be the best library for extracting text from PDF.  I tried the code locally and it did work so very much happy with tika.

I thought of looking through other libraries just to know how good they are after all i should be utilising my time to research on this topic.

***pdfToText from xpdf*** - so i think this is more of a standalone tool instead of a ready to use library. i tried to install the python library but not much resources are available in the open forum.  i spend quite an awfull lot of time to install it correctly but again no luck. 

***Tesseract*** -   This is a very promising library a wrapper around the googles Tesseract OCR engine.  This should be great for images and i found quite a lot of interesting tutorials on how to read characters from image using tesseract but extracting pdf looks a bit tough, i couln't figure out a working code snippet to do this, perhaps it may be hidden. anyways i'll be considering this library if tika fails. Tesseract is very promising because of the number of people who have successfully used it to rad characters from image.

***Python-Docx*** - Again this one also looks like a standalone tool i tried installing the library but to my bad luck i couldn't get this library to work for me it is also throwing some dependency error and when i traced it back i met with dead end. i couldnt find the github repository itself in github.

***PDF-Miner*** - This one was a straight 2 minutes quick installation and and short tutorial mentioned in stack overflow post  [link to the question in stackoverflow](https://stackoverflow.com/questions/26494211/extracting-text-from-a-pdf-file-using-pdfminer-in-python). This one saved quite a lot of time. 

So my Final Pick is Tika, PDF Miner, pyPDF2 and then tesseract. Next step is to test the efficiency of PDF to text extraction.