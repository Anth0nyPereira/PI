# Create your features here.
import random
from elasticsearch_dsl import Document, Text, Keyword, Index
from neomodel import StructuredNode, StringProperty, StructuredRel, IntegerProperty, config, \
    DateTimeProperty, FloatProperty, RelationshipTo, RelationshipFrom, OneOrMore, ZeroOrMore, BooleanProperty, \
    ArrayProperty
from neomodel import db
from manage import es

config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'


# for elastic search ↓
class ImageES(Document):
    uri = Text(required=True)
    hash = Text()
    tags = Text()

    class Index:
        name = 'image'

ImageES.init(using=es)
i = Index(using=es, name=ImageES.Index.name)
if not i.exists(using=es):
    i.create()


# for neo4j ↓
# relations
class WasTakenIn(StructuredRel):
    rel = 'Was taken in'
    latitude = FloatProperty()
    longitude = FloatProperty()
    altitude = FloatProperty()


class IsIn(StructuredRel):
    rel = 'Is in'


class HasA(StructuredRel):
    rel = "Has a"
    originalTagName = StringProperty()
    originalTagSource = StringProperty()


class DisplayA(StructuredRel):
    rel = 'Display a'
    coordinates = ArrayProperty()


# Nodes
class ImageNeo(StructuredNode):
    folder_uri = StringProperty(unique_index=True, required=True)
    name = StringProperty(required=True)
    creation_date = DateTimeProperty(default_now=False)
    insertion_date = DateTimeProperty(default_now=True)
    processing = StringProperty()
    format = StringProperty()
    width = IntegerProperty()
    height = IntegerProperty()
    hash = StringProperty(index=True)
    tag = RelationshipTo("Tag", HasA.rel, model=HasA, cardinality=OneOrMore)
    person = RelationshipTo("Person", DisplayA.rel, model=DisplayA, cardinality=ZeroOrMore)
    location = RelationshipTo("Location", WasTakenIn.rel, model=WasTakenIn)
    folder = RelationshipTo("Folder", IsIn.rel, model=IsIn)


class Tag(StructuredNode):
    name = StringProperty(unique_index=True, required=True)
    quantity = IntegerProperty(default=1)
    image = RelationshipFrom(ImageNeo, HasA.rel, model=HasA, cardinality=OneOrMore)


class Person(StructuredNode):
    name = StringProperty(required=True)
    image = RelationshipFrom(ImageNeo, DisplayA.rel, model=DisplayA, cardinality=OneOrMore)


class Country(StructuredNode):
    name = StringProperty(unique_index=True, required=True)
    city = RelationshipFrom('City', IsIn.rel, model=IsIn)


class City(StructuredNode):
    name = StringProperty(unique_index=True, required=True)
    country = RelationshipTo(Country, IsIn.rel, model=IsIn)
    location = RelationshipFrom('Location', IsIn.rel, model=IsIn)


class Location(StructuredNode):
    name = StringProperty(unique_index=True, required=True)
    image = RelationshipFrom(ImageNeo, WasTakenIn.rel, model=WasTakenIn)
    city = RelationshipTo(City, IsIn.rel, model=IsIn)


class Folder(StructuredNode):
    id_ = IntegerProperty(unique_index=True)
    name = StringProperty(required=True)
    root = BooleanProperty(default=False)
    terminated = BooleanProperty(default=False)
    parent = RelationshipTo("Folder", IsIn.rel, model=IsIn)
    children = RelationshipFrom("Folder", IsIn.rel, model=IsIn)
    images = RelationshipFrom("ImageNeo", IsIn.rel, model=IsIn)

    def getImages(self):
        query = "MATCH (i:ImageNeo)-[:`Is in`]->(f:Folder {id_:$id_}) RETURN i"
        results, meta = db.cypher_query(query, {"id_": self.id_})
        return [ImageNeo.inflate(row[0]) for row in results]

    def getChildren(self):
        query = "MATCH (c:Folder)-[:`Is in`]->(f:Folder {id_:$id_}) RETURN c"
        results, meta = db.cypher_query(query, {"id_": self.id_})
        return [self.inflate(row[0]) for row in results]

    def getFullPath(self):
        query = "MATCH (f:Folder {id_:$id_})-[*]-> (c:Folder) RETURN c.name"
        results, meta = db.cypher_query(query, {"id_": self.id_})
        return [path[0] for path in results]
