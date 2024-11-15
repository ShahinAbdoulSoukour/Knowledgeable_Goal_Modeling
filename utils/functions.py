import os
import time
from typing import Optional

import requests
import torch

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Pipeline, AutoModelForSeq2SeqLM
import pandas as pd

from utils.sparql_queries import find_all_triples_with_2_nodes_q


def detect_entailment(premise, hypothesis, model_name, model=None, tokenizer=None, max_length=512, full_results=False):
    """
    Labels two sentences, a premise and a hypothesis, as either **entailed**, **neutral** or **contradictory**.

    Parameters
    ----------
    premise :      str
                   The premise for the textual entailment evaluation.
    hypothesis :   str
                   The hypothesis for the textual entailment evaluation.
    model_name :   str or os.PathLike
                   Can be either:

                       - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                         Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                         user or organization name, like `dbmdz/bert-base-german-cased`.
                       - A path to a *directory* containing vocabulary files required by the tokenizer, for instance
                         saved using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                       - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                         single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                         applicable to all derived classes)
    model :        AutoModel, optional
                   If the model and the tokenizer have already been instantiated, you can directly pass them to the
                   function rather than re-creating them. In this case, you can leave `model_name` empty
    tokenizer :    AutoTokenizer, optional
                   If the model and the tokenizer have already been instantiated, you can directly pass them to the
                   function rather than re-creating them. In this case, you can leave `model_name` empty
    max_length :   int, default=256
                   The maximum number of tokens handled by the model referred to by `model_name`.
    full_results : bool, default=False
                   If True, returns a dict containing the scores for each label. Otherwise, returns only the best label
                   and the associated score.

    Returns
    -------
    tuple[str, float]|dict[str, float]
        A tuple composed of two elements:

            - A label representing the entailment relationship between the `premise` and the `hypothesis`. This label is
              either **entailment**, **neutral** or **contradiction**
            - The certainty score obtained by the model for the label identified as most probable.

        If `full_results` is True, returns a dict where the keys are the `entailment`, `neutral` and `contradiction`
        labels and values are the certainty scores obtained for each label.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    encoded_text = tokenizer(premise, hypothesis, return_tensors='pt', max_length=max_length, truncation=True).to(device)
    logits = model(encoded_text["input_ids"])["logits"][0]
    probs = torch.softmax(logits, -1).tolist()
    dict_predicted_probability = {'entailment': probs[0], 'neutral': probs[1], 'contradiction': probs[2]}

    if full_results:
        return dict_predicted_probability

    label = max(dict_predicted_probability, key=dict_predicted_probability.get)
    proba = max(dict_predicted_probability.values())

    return label, proba


def detect_entailment_api(premise, hypothesis, model_name, max_length=512, full_results=False, test_again=0):
    """
    Labels two sentences, a premise and a hypothesis, as either **entailed**, **neutral** or **contradictory**.

    Parameters
    ----------
    premise :      str
                   The premise for the textual entailment evaluation.
    hypothesis :   str
                   The hypothesis for the textual entailment evaluation.
    model_name :   str or os.PathLike
                   Can be either:
                       - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                         Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                         user or organization name, like `dbmdz/bert-base-german-cased`.
                       - A path to a *directory* containing vocabulary files required by the tokenizer, for instance
                         saved using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                       - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                         single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                         applicable to all derived classes)
    max_length :   int, default=256
                   The maximum number of tokens handled by the model referred to by `model_name`.
    full_results : bool, default=False
                   If True, returns a dict containing the scores for each label. Otherwise, returns only the best label
                   and the associated score.

    Returns
    -------
    tuple[str, float]|dict[str, float]
        A tuple composed of two elements:

            - A label representing the entailment relationship between the `premise` and the `hypothesis`. This label is
              either **entailment**, **neutral** or **contradiction**
            - The certainty score obtained by the model for the label identified as most probable.

        If `full_results` is True, returns a dict where the keys are the `entailment`, `neutral` and `contradiction`
        labels and values are the certainty scores obtained for each label.
    """
    API_URL = os.environ["API_URL_NLI"]
    API_TOKEN = os.environ["HF_TOKEN"]
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        'inputs': {
            'text': premise,
            'text_pair': hypothesis
        },
        "parameters": {
            'max_length': max_length,
            "return_all_scores": True
        },
        "options": {'wait_for_model': True}
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code < 200 or response.status_code > 399:
        if test_again < 50:
            time.sleep(5)
            return detect_entailment_api(premise, hypothesis, model_name, max_length=max_length, full_results=full_results, test_again=test_again+1)
        else:
            print(str(response.content))
            response.raise_for_status()

    results = response.json()

    if not full_results:
        return results[0]["label"], results[0]["score"]

    dict_predicted_probability = {i["label"]: i["score"] for i in results}
    return dict_predicted_probability



def triple_sentiment_analysis(triple, sentiment_task, neutral_predicates=None):
    """
    Predicts the sentiment (either `positive`, `negative` or `neutral`) expressed by a triple.

    Parameters
    ----------
    triple :             list[str]
                            #. A subject
                            #. A predicate
                            #. An object
    sentiment_task :     Pipeline
                         A Transformers pipeline for sentiment analysis.
    neutral_predicates : list[str], optional
                         A list of predicates inducing automatically neutral triples. Predicates from the RDF ontology
                         (e.g., `rdf:type`) are often relevant for this list.

    Returns
    -------
    tuple[str, float]
        A tuple composed of the most probable sentiment for the triple and its certainty score.
    """
    if neutral_predicates is None:
        neutral_predicates = []

    if triple[1] in neutral_predicates:
        return "neutral", 1

    sentiment = sentiment_task(" ".join(triple))[0]

    return sentiment["label"], sentiment["score"]



def triple_sentiment_analysis_api(triple, neutral_predicates=None, test_again=0):
    """
        Predicts the sentiment (either `positive`, `negative` or `neutral`) expressed by a triple through the HuggingFace API.

        Parameters
        ----------
        triple :             list[str]
                             A list composed of three strings:
                                 #. A subject
                                 #. A predicate
                                 #. An object
        neutral_predicates : list[str], optional
                             A list of predicates inducing automatically neutral triples. Predicates from the RDF ontology
                             (e.g., `rdf:type`) are often relevant for this list.
        Returns
        -------
        tuple[str, float]
            A tuple composed of the most probable sentiment for the triple and its certainty score.
    """
    API_URL = os.environ["API_URL_SENT"]
    API_TOKEN = os.environ["HF_TOKEN"]
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    if neutral_predicates is None:
        neutral_predicates = []

    if triple[1] in neutral_predicates:
        return "neutral", 1

    data = {
        "inputs": " ".join(triple),
        "parameters": {},
        "options": {"wait_for_model": True}
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code < 200 or response.status_code > 399:
        if test_again < 50:
            time.sleep(5)
            return triple_sentiment_analysis_api(triple, neutral_predicates=neutral_predicates, test_again=test_again+1)
        else:
            print(str(response.content))
            response.raise_for_status()

    sentiment = response.json()[0]

    return sentiment["label"], sentiment["score"]




def test_sentiment_analysis(triple, use_api, sentiment_task=None, neutral_predicates=None):
    """
    Function to select API-based or local sentiment analysis for a triple.

    Parameters
    ----------
    triple :             list[str]
                         A list composed of three strings representing the subject, predicate, and object.
    use_api :            bool
                         Check the presence of Hugging Face API keys
    sentiment_task :     Pipeline, optional
                         A Hugging Face Transformers pipeline for sentiment analysis (used only for local processing).
    neutral_predicates : list[str], optional
                         A list of predicates that automatically assign a neutral sentiment to triples.
    Returns
    -------
    tuple[str, float]
        A tuple containing the sentiment label and the certainty score.
    """
    if use_api:
        # Use API-based sentiment analysis
        return triple_sentiment_analysis_api(triple, neutral_predicates=neutral_predicates)
    else:
        # Use local sentiment analysis with the provided pipeline
        return triple_sentiment_analysis(triple, sentiment_task=sentiment_task, neutral_predicates=neutral_predicates)





def test_entailment(df, tokenizer, model_name, model_nli, use_api):
    """
    Performs NLI on a DataFrame containing pairs of premises and hypotheses, classifying each pair as entailment, neutral, or contradiction.
    Uses the Hugging Face API if available; otherwise, runs the model locally.

    Parameters
    ----------
    df :        pandas.DataFrame
                A DataFrame containing 'PREMISE' and 'HYPOTHESIS' columns with text data to analyze.
    tokenizer:  AutoTokenizer, optional
                The tokenizer instance used for the local NLI model.
    model_name: str or os.PathLike
                The name or path of the NLI model on Hugging Face, used by the API function.
    model_nli : transformers.AutoModelForSequenceClassification
                The NLI model used when running locally without the API.
    use_api :   bool
                Check the presence of Hugging Face API keys
    Returns
    -------
    pandas.DataFrame
        The original DataFrame with additional columns:
         - ENTAILMENT: Score for the entailment classification.
         - NEUTRAL: Score for the neutral classification.
         - CONTRADICTION: Score for the contradiction classification.
         - NLI_LABEL: The predicted label with the highest probability.
    """

    # apply the detect_entailment() function and create two new columns
    #df['ENTAILMENT_SCORES'] = df.apply(lambda row: detect_entailment_api(row['PREMISE'], row['HYPOTHESIS'], model_name, full_results=True), axis=1)

    # Test the entailment
    if use_api:
        print("\nHuggingFace API keys used for NLI.")
        # Use the API-based function
        df['ENTAILMENT_SCORES'] = df.apply(
            lambda row: detect_entailment_api(row['PREMISE'], row['HYPOTHESIS'], model_name, full_results=True),
            axis=1
        )
    else:
        print("\nNo HuggingFace API keys found for NLI.")
        # Use the local function
        df['ENTAILMENT_SCORES'] = df.apply(
            lambda row: detect_entailment(row['PREMISE'], row['HYPOTHESIS'], "",
                                          tokenizer=tokenizer, model=model_nli, full_results=True),
            axis=1
        )

    df['ENTAILMENT'] = df.apply(lambda row: row['ENTAILMENT_SCORES']['entailment'], axis=1).round(5)
    df['NEUTRAL'] = df.apply(lambda row: row['ENTAILMENT_SCORES']['neutral'], axis=1).round(5)
    df['CONTRADICTION'] = df.apply(lambda row: row['ENTAILMENT_SCORES']['contradiction'], axis=1).round(5)
    df = df.drop("ENTAILMENT_SCORES", axis=1)

    # display the dataframe with all entailments
    df = df.sort_values(by=['ENTAILMENT'], ascending=False)
    df['NLI_LABEL'] = df[['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']].apply(lambda x: x.idxmax(), axis=1)

    return df





def get_neighbors(triple, triple_serialized, graph):
    """
    Retrieves all neighboring triples in a knowledge graph that connect the subject and object of the given triple.
    Returns a DataFrame containing the original triple and its neighboring triples.

    Parameters
    ----------
    triple :            list[str]
                        The input triple.
    triple_serialized : list[list[str]]
                        The serialized form of the input triple, where each element is a list of three strings representing the components
                        (subject, predicate, object) in serialized format.
    graph :             rdflib.Graph
                        The RDFLib graph object to query for neighboring triples.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
            - TRIPLE_NEIGHBOR: Concatenated string representation of neighboring triples.
            - TRIPLE_NEIGHBOR_SERIALIZED: Serialized list format of each neighboring triple.
            - TRIPLE: The original triple.
            - TRIPLE_SERIALIZED: The serialized form of the original triple.

    """
    subject = triple_serialized[0][0]
    object = triple_serialized[-1][2]

    data_concatenate_triples = []

    query_results = graph.query(find_all_triples_with_2_nodes_q.format(subject, object))
    query_results_list = [[str(row["subject"]), str(row["predicate"]), str(row["object"])] for row in
                          query_results.bindings]  # triple

    data_concatenate_triples = {
        "TRIPLE_NEIGHBOR": [],
        "TRIPLE_NEIGHBOR_SERIALIZED": [],
        "TRIPLE": [],
        "TRIPLE_SERIALIZED": []
    }

    for t in query_results_list:
        if t not in triple_serialized:
            data_concatenate_triples["TRIPLE_NEIGHBOR"].append(f"{t[0]} {t[1]} {t[2]}")
            data_concatenate_triples["TRIPLE_NEIGHBOR_SERIALIZED"].append([t])
            data_concatenate_triples["TRIPLE"].append(triple)
            data_concatenate_triples["TRIPLE_SERIALIZED"].append(triple_serialized)

    df = pd.DataFrame(data_concatenate_triples)

    return df