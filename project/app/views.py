import json
import os
import re
from collections import defaultdict

import cv2
from django.shortcuts import render, redirect
from elasticsearch_dsl import Index, Search, Q
from app.forms import SearchForm, SearchForImageForm, EditFoldersForm, PersonsForm, PeopleFilterForm
from app.models import ImageES, ImageNeo, Tag, Person, Location
from app.processing import getOCR, getExif, dhash, findSimilarImages, uploadImages, fs, deleteFolder#, frr
from app.utils import addTag, deleteTag, addTagWithOldTag
from manage import es
from app.nlpFilterSearch import processQuery
import re
import itertools



def landingpage(request):
    query = SearchForm()  # query form stays the same
    image = SearchForImageForm()  # fetching image form response
    folders = len(fs.getAllUris())
    path_form = EditFoldersForm()
    return render(request, "landingpage.html", {'form': query, 'image_form': image, 'folders': folders, 'path_form':path_form})

def updateTags(request, hash):
    newTagsString = request.POST.get("tags")
    newTags = re.split('\s|\s+|\t|#', newTagsString)
    newTags = [tag for tag in newTags if tag != ""]
    print(newTags)
    image = ImageNeo.nodes.get_or_none(hash=hash)
    oldTags = [x.name for x in image.tag.all()]
    print(oldTags)

    for indx, tag in enumerate(newTags):
        if tag not in oldTags:
            addTag(hash, tag)

    for tag in oldTags:
        if tag not in newTags:
            deleteTag(hash, tag)

    query = SearchForm()
    image = SearchForImageForm()
    results = {}
    for tag in Tag.nodes.all():
        results["#" + tag.name] = tag.image.all()
        count = 0
        for lstImage in results["#" + tag.name]:
            results["#" + tag.name][count] = (lstImage, lstImage.tag.all())
            count += 1

    return render(request, "index.html", {'form': query, 'image_form': image, 'results': results, 'error': False})


from app.utils import showDict


def index(request):
    if request.method == 'POST':  # if it's a POST, we know it's from search by image
        query = SearchForm()  # query form stays the same
        image = SearchForImageForm(request.POST, request.FILES)  # fetching image form response

        if image.is_valid() and image.cleaned_data["image"]:  # if search by image is valid
            imagepath = image.cleaned_data["image"]  # get inserted path
            image_array = findSimilarImages(imagepath)  # find similar images
            results = {"results": []}  # blank results dictionary
            for i in image_array:  # looping through the results of the search
                getresult = ImageNeo.nodes.get_or_none(hash=i)  # fetching each corresponding node
                if getresult:  # if it exists
                    results["results"].append((getresult, getresult.tag.all()))  # append the node and its tags
            return render(request, "index.html",
                          {'form': query, 'image_form': image, 'results': results, 'error': False})
        else:  # the form filled had a mistake
            results = {}  # blank results dictionary
            for tag in Tag.nodes.all():  # looping through all tag nodes
                results["#" + tag.name] = tag.image.all()  # inserting each tag in the dict w/ all its images as values
                count = 0  # counter
                for lstImage in results["#" + tag.name]:  # for each image of the value of this tag
                    results["#" + tag.name][count] = (lstImage, lstImage.tag.all())  # replace it by a tuple w/ its tags
                    count += 1  # increase counter
            return render(request, 'index.html',
                          {'form': query, 'image_form': image, 'results': results, 'error': True})
    else:
        if 'query' in request.GET:
            query = SearchForm()  # cleaning this form
            image = SearchForImageForm()  # fetching the images form
            query_text = request.GET.get("query")  # fetching the inputted query
            query_array = processQuery(query_text)  # processing query with nlp
            tag = "#" + " #".join(query_array)  # arranging tags with '#' before

            result_hashs = list(
                map(lambda x: x.hash, search(query_array)))  # searching and getting result's images hash
            results = {tag: []}  # blank results dictionary
            for hash in result_hashs:  # iterating through the result's hashes
                img = ImageNeo.nodes.get_or_none(hash=hash)  # fetching the reuslts nodes from DB
                if img is None:  # if there is no image with this hash in DB
                    continue  # ignore, advance
                tags = img.tag.all()  # get all tags from the image
                results[tag].append((img, tags))  # insert tags in the dictionary

            return render(request, "index.html",
                          {'form': query, 'image_form': image, 'results': results, 'error': False})

        else:  # first time in the page - no forms filled
            query = SearchForm()
            image = SearchForImageForm()

            results = {}
            for tag in Tag.nodes.all():
                results["#" + tag.name] = tag.image.all()
                count = 0
                for lstImage in results["#" + tag.name]:
                    results["#" + tag.name][count] = (lstImage, lstImage.tag.all())
                    count += 1
            return render(request, 'index.html',
                          {'form': query, 'image_form': image, 'results': results, 'error': False})


def delete(request, path):
    form = SearchForm()
    image = SearchForImageForm()
    pathf = EditFoldersForm()
    deleteFolder(path)
    folders = fs.getAllUris()
    return render(request, 'managefolders.html',
                  {'form': form, 'image_form': image, 'folders': folders, 'path_form': pathf})


def managefolders(request):
    if 'path' in request.GET:
        uploadImages(request.GET.get('path'))
        form = SearchForm()
        image = SearchForImageForm()
        pathf = EditFoldersForm()
        folders = fs.getAllUris()
        return render(request, 'managefolders.html',
                      {'form': form, 'image_form': image, 'folders': folders, 'path_form': pathf})
    else:
        form = SearchForm()
        image = SearchForImageForm()
        pathf = EditFoldersForm()
        folders = fs.getAllUris()
        return render(request, 'managefolders.html',
                      {'form': form, 'image_form': image, 'folders': folders, 'path_form': pathf})


def managepeople(request):
    if request.method == 'POST':
        filters = PeopleFilterForm(request.POST)
        print('entrou aqui...')
        print(filters)
        filters2 = filters.cleaned_data
        print('cleanded fore valid', filters2)

        showDict['unverified'] = filters2['unverified']
        showDict['verified'] = filters2['verified']
        """
        if filters.is_valid() and filters.has_changed():
            print('entrou aqui tmb')
            filters = filters.cleaned_data
            print(filters)

            #showDict = filters
            showDict['unverified'] = filters['unverified']
            showDict['verified'] = filters['verified']
            print(showDict)
        else:
            print('invalid...')
        """
        return redirect('/people')
        # return render(request, 'renaming.html', {'form': form, 'image_form': image, 'names_form': names})

    form = SearchForm()
    image = SearchForImageForm()
    names = PersonsForm()
    filters = PeopleFilterForm(initial=showDict)
    return render(request, 'renaming.html',
                  {'form': form, 'image_form': image, 'names_form': names, 'filters': filters})


def search(tags):
    q = Q('bool', should=[Q('term', tags=tag) for tag in tags], minimum_should_match=1)
    s = Search(using=es, index='image').query(q)
    return s.execute()


def alreadyProcessed(img_path):
    image = cv2.imread(img_path)
    hash = dhash(image)
    existed = ImageNeo.nodes.get_or_none(hash=hash)

    return True if existed else False


def upload(request):
    data = json.loads(request.body)
    uploadImages(data["path"])
    return render(request, 'index.html')


def searchtag(request):
    get = [request.GET.get('tag')]
    q = Q('bool', should=[Q('term', tags=tag) for tag in get], minimum_should_match=1)
    s = Search(using=es, index='image').query(q)
    execute = s.execute()
    for i in execute:
        print(i)
    return render(request, 'index.html')


def updateFolders(request):
    folders = fs.getAllUris()
    for folder in folders:
        uploadImages(folder)
    form = SearchForm()
    image = SearchForImageForm()
    pathf = EditFoldersForm()
    return render(request, 'managefolders.html',
                  {'form': form, 'image_form': image, 'folders': folders, 'path_form': pathf})


def update_faces(request):
    if request.method != 'POST':
        redirect('/people')

    form = PersonsForm(request.POST)
    if not form.is_valid():
        print('invalid form!!!')

    print(form.cleaned_data)
    data = form.cleaned_data

    imgs = int(len(form.cleaned_data) / 5)
    listt = []
    for i in range(imgs):
        # if not data['person_verified_%s' % str(i)]:
        #    continue
        thumbname = data['person_image_%s' % str(i)]
        new_personname = data['person_name_%s' % str(i)]

        # retirar isto abaixo dps!!!
        #new_personname = new_personname.split(' ')[0]
        old_personname = data['person_before_%s' % str(i)]
        verified = True
        if not data['person_verified_%s' % str(i)]:
            # continue
            new_personname = old_personname
            verified = False

        # if old_personname != new_personname:
        image_hash = data['person_image_id_%s' % str(i)]
        frr.changeRelationship(image_hash, new_personname, old_personname, thumbnail=thumbname, approved=verified)
        if old_personname != new_personname:
            frr.changeNameTagES(image_hash, new_personname, old_personname)

    frr.update_data()

    if 'reload' in request.POST:
        print('reload was called')
        frr.reload()

    return redirect('/people')


def dashboard(request):
    form = SearchForm()
    image = SearchForImageForm()
    person_number = 0
    for p in Person.nodes.all():
        person_number += 1

    location_number = 0
    for l in Location.nodes.all():
        location_number +=1

    results = {}
    counts = {}
    for tag in Tag.nodes.all():

        results["#" + tag.name] = tag.image.all()
        count = 0
        for lstImage in results["#" + tag.name]:
            results["#" + tag.name][count] = (lstImage, lstImage.tag.all())
            count += 1
        counts[tag.name] = len(results["#" + tag.name])

    countTags = dict(sorted(counts.items(), key=lambda item: item[1],
                            reverse=True))  # sort the dict by its value (count), from the greatest to the lowest
    if len(countTags) < 10:
        countTags = dict(itertools.islice(countTags.items(), len(countTags)))
    else:
        countTags = dict(itertools.islice(countTags.items(), 10))  # only want the first top 10 more common tags
    countTags = json.dumps(countTags)

    ## original tag source statistics
    countOriginalTagSource = {}
    allTagLabels = {"ocr": "text", "manual": "manual", "object": "objects", "places": "places",
                    "exif": "image properties", "folder": "folder's name", "breeds": "breed"}
    for tag in Tag.nodes.all():
        imgList = tag.image.all()
        for img in imgList:
            rel = img.tag.relationship(tag)
            originalTagSource = rel.originalTagSource
            originalTagSource = allTagLabels[originalTagSource]
            # print(tag.name, originalTagSource)
            if originalTagSource not in countOriginalTagSource:
                countOriginalTagSource[originalTagSource] = 1
            else:
                countOriginalTagSource[originalTagSource] += 1

    # print(countOriginalTagSource)
    for label in allTagLabels.values():
        if label not in countOriginalTagSource.keys():
            countOriginalTagSource[label] = 0

    countOriginalTagSource = dict(sorted(countOriginalTagSource.items(), key=lambda item: item[1]))
    print(countOriginalTagSource)
    return render(request, 'dashboard.html',
                  {'form': form, 'image_form': image, 'results': results, 'counts': countTags,
                   'countTagSource': countOriginalTagSource, 'numbers': {'person': person_number, 'location': location_number}})


def calendarGallery(request):
    form = SearchForm()
    image = SearchForImageForm()
    datesInsertion = {}
    datesCreation = {}
    previousImages = []
    for tag in Tag.nodes.all():
        imgList = tag.image.all()
        for img in imgList:
            if img not in previousImages:
                insertionDate = str(img.insertion_date)
                creationDate = str(img.creation_date)
                insertionDate = insertionDate.split(" ")[0]
                creationDate = creationDate.split(" ")[0]
                if insertionDate not in datesInsertion:
                    datesInsertion[insertionDate] = 1
                else:
                    datesInsertion[insertionDate] += 1
                if creationDate != "None":
                    if creationDate not in datesCreation:
                        datesCreation[creationDate] = 1
                    else:
                        datesCreation[creationDate] += 1

                previousImages += [img]

            else:
                continue

    datesInsertion = json.dumps(datesInsertion)
    datesCreation = json.dumps(datesCreation)
    return render(request, 'gallery.html',
                  {'form': form, 'image_form': image, 'datesInsertion': datesInsertion, 'datesCreation': datesCreation})


def objectsGallery(request):
    form = SearchForm()
    image = SearchForImageForm()
    allTags = []
    for tag in Tag.nodes.all():
        imgList = tag.image.all()
        for img in imgList:
            rel = img.tag.relationship(tag)
            originalTagSource = rel.originalTagSource
            if originalTagSource == "object" and tag.name not in allTags:
                allTags += [tag.name.lower()]

    allTags = sorted(allTags)

    return render(request, 'objectsGallery.html',
                  {'form': form, 'image_form': image, 'objectTags': allTags})


def peopleGallery(request):
    form = SearchForm()
    image = SearchForImageForm()
    allNames = []

    for person in Person.nodes.all():
        name = person.name
        imgList = person.image.all()
        for img in imgList:
            rel = img.person.relationship(person)
            verified = rel.approved
            if verified == True:
                allNames += [name]
                break
            else:
                break

    allNames = sorted(allNames)

    return render(request, 'peopleGallery.html',
                  {'form': form, 'image_form': image, 'people': allNames})


def scenesGallery(request):
    form = SearchForm()
    image = SearchForImageForm()
    allTags = []
    for tag in Tag.nodes.all():
        imgList = tag.image.all()
        for img in imgList:
            rel = img.tag.relationship(tag)
            originalTagSource = rel.originalTagSource
            if originalTagSource == "places" and tag.name not in allTags:
                allTags += [tag.name.lower()]

    allTags = sorted(allTags)

    return render(request, 'placesGallery.html',
                  {'form': form, 'image_form': image, 'placesTags': allTags})


def textGallery(request):
    form = SearchForm()
    image = SearchForImageForm()
    allTags = []
    for tag in Tag.nodes.all():
        imgList = tag.image.all()
        for img in imgList:
            rel = img.tag.relationship(tag)
            originalTagSource = rel.originalTagSource
            if originalTagSource == "ocr" and tag.name not in allTags:
                allTags += [tag.name.lower()]

    allTags = sorted(allTags)

    return render(request, 'textGallery.html',
                  {'form': form, 'image_form': image, 'textTags': allTags})

