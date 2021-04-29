import re
from app.models import *
from app.utils import getRandomNumber


class Node:
    def __init__(self, name: str, id: int):
        self.name = name  # folder name
        self.children = {}  # key: folder name, value: node
        self.parent = None
        self.id = id

        # if current node is the last one of an uri, it is terminated
        # ex: for uri C:\User\abcd\ef\, node 'ef' is terminated
        self.terminated = False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class SimpleFileSystemManager:
    def __init__(self):
        self.trees = {}  # key: folder name, value: node.

    def exist(self, uri: str):
        folders, root = self.splitUriAndGetRoot(uri)

        if root in self.trees:
            node = self.trees[root]
            for i in range(1, len(folders)):
                folder = folders[i].strip()
                if folder == "":
                    continue

                if folder in node.children:
                    node = node.children[folder]
                else:
                    return False
        else:
            return False

        return node.terminated

    def addFullPathUri(self, uri: str, ids: list):
        folders = re.split("[\\\/]+", uri)
        if folders[len(folders) - 1].strip() == "":
            folders = folders[:-1]

        if len(folders) != len(ids): return

        if self.exist(uri): return

        root = folders[0]

        if root in self.trees:
            node = self.trees[root]
        else:
            self.trees[root] = node = Node(root, ids[0])

        for i in range(1, len(folders)):
            folder = folders[i]

            if folder in node.children:
                node = node.children[folder]
            else:
                newNode = Node(folder, ids[i])
                newNode.parent = node
                node.children[folder] = node = newNode

        node.terminated = True

    # add a new folder under the given uri
    # given uri must exist in fileSystemManager
    def expandUri(self, uri, folder, id):
        folders, root = self.splitUriAndGetRoot(uri)

        if root in self.trees:
            node = self.trees[root]

            for i in range(1, len(folders)):
                folder = folders[i]
                if folder in node.children:
                    node = node.children[folder]

            newNode = Node(folder, id)
            newNode.terminated = True
            node.children[folder] = newNode

    def getLastNode(self, uri):
        folders, root = self.splitUriAndGetRoot(uri)

        if root in self.trees:
            node = self.trees[root]

            for i in range(1, len(folders)):
                folder = folders[i]
                if folder in node.children:
                    node = node.children[folder]
                else:
                    return None

            return node

    def createUriInNeo4j(self, uri):
        folders, root = self.splitUriAndGetRoot(uri)

        if root in self.trees:
            node = self.trees[root]
        else:
            savedNode = Folder(id_=getRandomNumber(), name=root, root=True,
                          terminated=True if len(folders) == 1 else False).save()
            self.trees[root] = node = Node(root, savedNode.id_)

        for i in range(1, len(folders)):
            folder = folders[i]
            if folder in node.children:
                node = node.children[folder]
            else:
                savedNode = Folder(id_=getRandomNumber(), name=folder,
                                   terminated=True if i == len(folders) - 1 else False).save()
                parent = Folder.nodes.get(id_=node.id)
                savedNode.parent.connect(parent)

                newNode = Node(folder, savedNode.id_)
                newNode.parent = node
                node.children[folder] = node = newNode

        return node

    def splitUriAndGetRoot(self, uri):
        folders = re.split("[\\\/]+", uri)
        if folders[len(folders) - 1].strip() == "":
            folders = folders[:-1]

        root = folders[0]
        return folders, root

    def deleteFolderFromFs(self, uri):
        if self.exist(uri):
            folders, root = self.splitUriAndGetRoot(uri)

            node = self.trees.pop(root)
            for i in range(1, len(folders)):
                folder = folders[i]
                node = node.children.pop(folder)

            folderstoBeDeleted = [Folder.nodes.get(id_=node.id)]

            while folderstoBeDeleted != []:
                f = folderstoBeDeleted.pop()
                images = f.getImages()

                if images is not None:
                    for image in images:
                        image.folder.disconnect(f)
                        if len(image.folder) == 0:
                            image.delete()

                childrenFolders = f.getChildren()
                if childrenFolders:
                    folderstoBeDeleted.extend(list(childrenFolders))
                f.delete()
