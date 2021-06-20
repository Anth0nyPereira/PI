# CHANGE TO YOUR PATH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dbsPath = 'bolt://neo4j:12345@localhost:7687'
ocrPath = 'app/resources/tesseract-OCR/tesseract'
essPath = 'app/resources/elasticSearch/bin/elasticsearch.bat'

var = 1

ocr_ubuntu = r'/usr/bin/tesseract'
## Windows Iglesias:
if var == 1:
    ocrPath = r'D:\Programs\tesseract-OCR\tesseract'
    essPath = "D:\\Programs\\elasticsearch-7.12.0\\bin\\elasticsearch.bat"
## Windows Wei:
elif var == 2:
    ocrPath = r'D:\OCR\tesseract'
    essPath = "D:\Java\JavaEE\elasticsearch\elasticsearch-7.11.1\\bin\elasticsearch.bat"
## Ubuntu Anthony:
elif var == 3:
    ocrPath = ocr_ubuntu
    essPath = "/home/anth0nypereira/elasticsearch-7.12.1/bin/elasticsearch"
## Ubuntu Alexa:
elif var == 4:
    ocrPath = ocr_ubuntu
    essPath = "/home/alexis/Downloads/elasticsearch-7.12.1/bin/elasticsearch"
## Ubuntu Mariana:
else:
    ocrPath = ocr_ubuntu
    essPath = "/home/mar/Documents/UA/6-semester/PI/elasticsearch-7.12.0/bin/elasticsearch"
