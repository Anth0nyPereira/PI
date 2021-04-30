import json

from django.shortcuts import render
from elasticsearch_dsl import Index, Search, Q
from app.models import ImageES
from app.processing import uploadImages, findSimilarImages, getOCR, fs
from manage import es


def createIndex(request):
    i = Index(using=es, name=request.GET.get("index"))
    if not i.exists(using=es):
        i.create()

    return render(request, 'index.html')


def update(request):
    a = ImageES.get(using=es, id=request.GET.get("uri"))
    a.tags.append(request.GET.get("tag"))
    a.tags = list(set(a.tags))
    a.update(using=es, tags=a.tags)
    a.save(using=es)
    return render(request, 'index.html')


def search(request):
    tags = request.GET.get("tags")
    q = Q('bool', should=[Q('term', tags=tag) for tag in tags],
          minimum_should_match=1)

    s = Search(using=es, index='image').query(q)
    r = s.execute()

    for hit in r:
        print("score: %s, uri: %s, tags: %s"%(hit.meta.score,hit.uri, hit.tags))

    return render(request, 'index.html')


def upload(request):
    data = json.loads(request.body)
    uploadImages(data["path"])
    return render(request, 'index.html')

def deleteFolder(request):
    if request.method == "PUT":
        folderToBeDeleted = request.GET.get("f")
        fs.deleteFolderFromFs(folderToBeDeleted)
    return render(request, 'index.html')

def findSimilar(request):
    get = request.GET.get("path")
    findSimilarImages(get)
    return render(request, 'index.html')


def ocr(request):
    get = request.GET.get("path")
    getOCR(get)
    return render(request, 'index.html')

def exif(request):
    get = request.GET.get("path")
    getExif(get)
    return render(request, 'index.html')
