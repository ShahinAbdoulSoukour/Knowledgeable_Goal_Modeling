import os
import rdflib

from utils.sparql_queries import find_all_triples_q, find_all_triples_with_node_q, find_uri_from_label_q

def graph_extender(path_to_graph: str or os.PathLike) -> rdflib.Graph:
    graph = rdflib.Graph()
    graph.parse(path_to_graph)

    all_triples = graph.query(find_all_triples_q)

    triples_to_add = []

    for triple in all_triples:
        if str(triple["predicate"]) == "is a type of":
            subject = triple["subject"]
            subject_uri = graph.query(find_uri_from_label_q.format(str(subject))).bindings[0]["uri"]
            object = triple["object"]

            parent_relationships = graph.query(find_all_triples_with_node_q.format(str(object)))
            for parent_relationship in parent_relationships:
                relationship_predicate = parent_relationship["predicate"]
                relationship_predicate_uri = graph.query(find_uri_from_label_q.format(str(relationship_predicate))).bindings[0]["uri"]
                if parent_relationship["subject"] != subject and parent_relationship["object"] != subject:
                    if parent_relationship["subject"] == object:
                        relationship_object = parent_relationship["object"]
                        relationship_object_uri = graph.query(find_uri_from_label_q.format(str(relationship_object))).bindings[0]["uri"]
                        triples_to_add.append((subject_uri, relationship_predicate_uri, relationship_object_uri))
                    elif parent_relationship["object"] == object and str(parent_relationship["predicate"]) != "is a type of":
                        relationship_subject = parent_relationship["subject"]
                        relationship_subject_uri = graph.query(find_uri_from_label_q.format(str(relationship_subject))).bindings[0]["uri"]
                        triples_to_add.append((relationship_subject_uri, relationship_predicate_uri, subject_uri))

    for triple_to_add in triples_to_add:
        graph.add(triple_to_add)

    return graph