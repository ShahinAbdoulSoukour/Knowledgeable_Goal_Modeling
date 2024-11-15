import requests
from fastapi import Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter

from typing import List, Optional

from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.responses import RedirectResponse, Response

from database import SessionLocal, engine
import models
from dotenv import load_dotenv

import os
os.environ['HF_HOME'] = os.getcwd() + "/cache/"
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.utils import logging
from sentence_transformers import SentenceTransformer
import torch
from anchor_points_extractor import anchor_points_extractor
from utils.sparql_queries import find_all_triples_q
from utils import test_entailment, test_sentiment_analysis
from graph_explorator import graph_explorator_bfs_optimized
from g2t_generator import g2t_generator
from graph_extender import graph_extender

import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
logging.set_verbosity_error()

#load_dotenv()
# --- Initialize inference servers
#API_URL_sent = os.environ["API_URL_SENT"]
#API_URL_nli = os.environ["API_URL_NLI"]
#API_TOKEN = os.environ["HF_TOKEN"]
#headers = {
#    "Accept": "application/json",
#    "Authorization": f"Bearer {API_TOKEN}",
#    "Content-Type": "application/json"
#}

#data_sent = {
#    'inputs': {
#        'text': '',
#        'text_pair': ''
#    }
#}
#data_nli = {
#    "inputs": " "
#}

#requests.post(API_URL_nli, headers=headers, json=data_nli)
#requests.post(API_URL_sent, headers=headers, json=data_sent)

load_dotenv()
# Check for Hugging Face API availability
api_token = os.getenv("HF_TOKEN")
api_url_sent = os.getenv("API_URL_SENT")
api_url_nli = os.getenv("API_URL_NLI")
use_api = bool((api_url_sent or api_url_nli) and api_token)


# --- Import models ---
if use_api:
    print("\n--> HuggingFace API keys found to perform NLI and sentiment analysis.")
    model_nli_name = None
    tokenizer_nli = None
    model_nli = None
    sentiment_model_path = None
    sentiment_task = None
else:
    print("\n--> No HuggingFace API key was found.")
    model_nli_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer_nli = AutoTokenizer.from_pretrained(model_nli_name)
    model_nli = AutoModelForSequenceClassification.from_pretrained(model_nli_name).to(device)

    sentiment_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path, device=device)

model_sts = SentenceTransformer('all-mpnet-base-v2')
model_g2t = T5ForConditionalGeneration.from_pretrained("Inria-CEDAR/WebNLG20T5B").to(device)
tokenizer_g2t = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)


# --- Import the Knowledge Graph (KG) ---
# TODO: upload a knowledge graph (RDF file)
domain_graph = graph_extender("./flooding_graph_V2.rdf")


# Templates (Jinja2)
templates = Jinja2Templates(directory="templates/")

# Router
router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

models.Base.metadata.create_all(bind=engine)


def get_subgoal_branch_from_triple_filtered(subgoal_id, db: Session):
    # Initialize the path and the stack
    path = [] # To store the results
    stack = [(subgoal_id, [])] # Initialized with the subgoal_id and an empty path

    # For each current_goal_id in the stack, the corresponding Goal is retrieved
    while stack:
        current_goal_id, current_path = stack.pop()
        current_goal = db.query(models.Goal).filter_by(id=current_goal_id).first()

        if current_goal is None:
            continue

        # Retrieve the filtered triples that created the current goal (--> the subgoal)
        # from the table: Triple_Filtered
        filtered_triple_list = []
        filtered_triples = db.query(models.Triple_Filtered).filter_by(subgoal_id=current_goal_id).all()

        if filtered_triples:
            for triple in filtered_triples:
                filtered_triple_list.append({
                    'high_level_goal_id': triple.high_level_goal_id,
                    'entailed_triples': triple.get_entailed_triples()
                })

        # The new path is constructed by prepending the current goal and its filtered triples to the current_path
        new_path = [(current_goal, filtered_triple_list)] + current_path

        # Updating the path
        # The path is updated with the new path
        path = new_path

        # Check for high-level goals
        # High-level goals for the current goal are retrieved (from the table: Hierarchy), and each high-level goal is added to the stack
        high_level_goals = db.query(models.Hierarchy).filter_by(subgoal_id=current_goal_id).all()
        for hlg in high_level_goals:
            stack.append((hlg.high_level_goal_id, new_path))

    # path --> it contains the subgoal, the filtered triples used to create it, and its high-level goals in the correct order
    return path


def find_relevant_information(request: Request, goal_type: str, refinement: Optional[str], highlevelgoal: str, hlg_id: int, filtered_out_triples_with_goal_id: List[str], beam_width: int, max_depth: int, db: Session) -> Response:
    # get the start time
    st = time.time()

    # Goal
    print('\nGoal:', highlevelgoal)

    # Max depth
    print('\nMax Depth:', max_depth)

    # Beam width
    print('\nBeam width:', beam_width)

    all_goal = db.query(models.Goal).all()

    goal_with_outputs = db.query(models.Goal).filter(models.Goal.goal_name == highlevelgoal).first()

    # If the high-level goal do not exist in the database
    if not goal_with_outputs:
        modified_filtered_triples = []

        print(filtered_out_triples_with_goal_id)

        # If there are some filtered triples, get the high-level goal id and the triples
        if filtered_out_triples_with_goal_id:
            triples_with_ids = []
            for item in filtered_out_triples_with_goal_id:
                goal_id, triple = item.split('_', 1)
                print("goal_id:")
                print(goal_id)
                triples_with_ids.append({
                    'goal_id': goal_id,  # High-level goal ID
                    'filtered_triple': triple  # Filtered triples (selected by the designer)
                })
            triples_with_ids_df = pd.DataFrame(triples_with_ids)  # Add the elements in a dataframe

            print("\nTRIPLES WITH IDS:")
            print(triples_with_ids_df.to_string())

            # If the dataframe is not empty
            if not triples_with_ids_df.empty:
                for row in triples_with_ids_df.itertuples():
                    print(row.filtered_triple)
                    modified_filtered_triples.append({
                        'high_level_goal_id': row.goal_id,  # High-level goal ID
                        'triple_filtered_from_hlg': row.filtered_triple.replace("', '", ' ').replace("['", "").replace(
                            "']", "")
                    })

                modified_filtered_triples_df = pd.DataFrame(modified_filtered_triples)

                subgoal_id = modified_filtered_triples_df.loc[
                    0, 'high_level_goal_id']  # Replace with the actual subgoal id
                subgoal_branch = get_subgoal_branch_from_triple_filtered(subgoal_id, db)

                print("\n")

                ###
                # Iterate through the subgoal_branch and append new rows to modified_filtered_triples
                for level, (goal, ft) in enumerate(subgoal_branch):
                    # Ensure ft is a list
                    if isinstance(ft, str):
                        ft = [ft]

                    if ft:
                        for triple_dict in ft:
                            triple = triple_dict['entailed_triples']
                            high_level_goal_id = triple_dict['high_level_goal_id']

                            # Check for redundancy before adding
                            if not any(d['triple_filtered_from_hlg'] == triple and d[
                                'high_level_goal_id'] == high_level_goal_id for d in modified_filtered_triples):
                                modified_filtered_triples.append({
                                    'high_level_goal_id': high_level_goal_id,
                                    'triple_filtered_from_hlg': triple
                                })

                # Convert the updated list to a DataFrame
                modified_filtered_triples_df = pd.DataFrame(modified_filtered_triples)
                print("\nUpdated MODIFIED TRIPLE (STRING TO LIST):")
                print(modified_filtered_triples_df.to_string())

        # --- Extract all triples in the KG ---
        query_results = domain_graph.query(find_all_triples_q)
        triples = [list(map(str, [row["subject"], row["predicate"], row["object"]])) for row in query_results.bindings]

        data = []
        for t in triples:
            subject = t[0]
            predicate = t[1]
            object = t[2]

            # simple triple
            triple = " ".join(t)
            # triples serialized
            triple_with_separator = [t]
            list_goal_triples = [(triple, highlevelgoal, triple_with_separator, subject, predicate, object)]

            for element in list_goal_triples:
                row = {'TRIPLE': element[0], 'GOAL': element[1], 'TRIPLE_SERIALIZED': element[2], 'SUBJECT': element[3],
                       'PREDICATE': element[4], 'OBJECT': element[5]}
                data.append(row)

        goal_triples_df = pd.DataFrame(data)

        ft = [item['triple_filtered_from_hlg'] for item in modified_filtered_triples]
        print("\nFILTERED TRIPLE(S):")
        print(ft)

        # --- Extract anchor points ---
        anchor_points_df = anchor_points_extractor(goal_triples_df, model_sts, ft).copy()
        print("\nANCHOR TRIPLES:")
        print(anchor_points_df)

        # --- Transform negative triples ---  --- Sentiment analysis ---
        anchor_points_df["SENTIMENT"] = anchor_points_df["TRIPLE_SERIALIZED"].apply(
            lambda triple: test_sentiment_analysis(triple[0], use_api, sentiment_task, neutral_predicates=["is a type of"])[0])
        anchor_points_df.rename(columns={'TRIPLE': 'PREMISE', 'GOAL': 'HYPOTHESIS'}, inplace=True)

        transformed_triples_premise = []
        goal_types = []  # to store the corresponding "GOAL_TYPE" based on whether the sentiment is negative or not.

        for triple, sentiment in zip(anchor_points_df["PREMISE"], anchor_points_df["SENTIMENT"]):
            if sentiment == "negative":
                ### Transformation
                transformed_triples_premise.append("Prevent that " + triple)
                goal_types.append("AVOID")
            else:
                transformed_triples_premise.append(triple)
                goal_types.append("ACHIEVE")

        # Create a new DataFrame with the transformed premises and goal types
        transformed_anchor_points = pd.DataFrame({
            "GOAL_TYPE": goal_types,
            "PREMISE": transformed_triples_premise
        })
        transformed_anchor_points["HYPOTHESIS"] = anchor_points_df["HYPOTHESIS"].values
        transformed_anchor_points["PREMISE_SERIALIZED"] = anchor_points_df["TRIPLE_SERIALIZED"].values
        transformed_anchor_points["SUBJECT"] = anchor_points_df["SUBJECT"].values
        transformed_anchor_points["PREDICATE"] = anchor_points_df["PREDICATE"].values
        transformed_anchor_points["OBJECT"] = anchor_points_df["OBJECT"].values

        # --- Print transformed anchor points ---
        print("\nTRANSFORMED ANCHOR POINTS:")
        print(transformed_anchor_points.to_string())

        # --- Test the entailment between the high-level goal (as hypothesis) and triples (as premise) ---
        entailment_result = test_entailment(transformed_anchor_points, tokenizer_nli, model_nli_name, model_nli, use_api)
        print("\nENTAILMENT RESULTS:")
        print(entailment_result.to_string())

        # --- Explore graph to improve contextualization ---
        entailed_triples_df = graph_explorator_bfs_optimized(entailment_result, highlevelgoal, domain_graph, model_sts,
                                                             model_nli_name, tokenizer_nli, model_nli, beam_width, max_depth, use_api)

        # --- ### ---
        #all_triples_entailed = [triple for triples in entailed_triples_df["SUBGOALS_SERIALIZED"].tolist() for triple in triples]

        #print("\nall_triples_entailed")
        #print(all_triples_entailed)

        all_triples_entailed = []
        unique_triples_entailed = []

        if not entailed_triples_df.empty:
            #all_triples_entailed.append(triples[0] for triples in entailed_triples_df["SUBGOALS_SERIALIZED"].tolist())
            #print('\nALL ENTAILED TRIPLES:')
            #print(all_triples_entailed)

            #for triple in all_triples_entailed:
            #    if (triple not in unique_triples_entailed) and (type(triple) is list):
            #        unique_triples_entailed.append(triple)

            #print("\nUNIQUE TRIPLES:")
            #print(unique_triples_entailed)

            triples_already_processed = []
            processed_data = []
            triples_to_process_grouped = []

            for row in entailed_triples_df.itertuples():
                triples_to_process = []

                for triple in row.SUBGOALS_SERIALIZED:
                    if (triple not in triples_already_processed) and (type(triple) is list):
                        triples_to_process.append(triple)

                if triples_to_process:
                    triples_to_process_grouped.append((triples_to_process, row.GOAL_TYPE))

                triples_already_processed.extend(triples_to_process)

            print('\nTRIPLES TO PROCESS:')
            print(triples_to_process)

            print('\nTRIPLES ALREADY PROCESSED:')
            print(triples_already_processed)

            print('\nTRIPLES TO PROCESS GROUPED:')
            print(triples_to_process_grouped)

            if triples_to_process_grouped:
                grps_of_triples = list(filter(lambda grp: len(grp[0]) >= 2, triples_to_process_grouped))
                if len(grps_of_triples):
                    predictions = g2t_generator([tripls_grp for tripls_grp, _ in grps_of_triples], model=model_g2t,
                                                tokenizer=tokenizer_g2t)
                    for i in range(len(triples_to_process_grouped)):
                        if len(triples_to_process_grouped[i][0]) == 1:
                            predictions.insert(i, "")
                else:
                    predictions = [""] * len(triples_to_process_grouped)

                for prediction, (triples, gt) in zip(predictions, triples_to_process_grouped):
                    processed_data.append({
                        "ENTAILED_TRIPLE": triples,
                        "GOAL_TYPE": gt,
                        "GENERATED_TEXT": prediction
                    })

            # Create DataFrame from the list of dictionaries
            processed_data_df = pd.DataFrame(processed_data)

            print('\nPROCESSED DATA (G2T):')
            print(processed_data_df.to_string())

            # Add the goal (as high-level goal) in the database (table: goal)
            new_goal = models.Goal(goal_type=goal_type, goal_name=highlevelgoal)
            db.add(new_goal)
            db.commit()

            # Add the entailed triples and the goal type in the database (table: outputs)
            for row in processed_data_df.itertuples():
                new_results = models.Outputs(generated_text=row.GENERATED_TEXT, goal_type=row.GOAL_TYPE,
                                             goal_id=new_goal.id)
                new_results.set_entailed_triple(row.ENTAILED_TRIPLE)
                db.add(new_results)
            db.commit()

            print("\nHigh-level goal added in the database!")

            # If certain triples are selected
            if modified_filtered_triples:
                for row in modified_filtered_triples_df.itertuples():
                    # Add the filtered triples (selected by the designer for creating subgoals) to the database
                    # (table: filtered_triple)
                    filtered_triple = models.Triple_Filtered(subgoal_id=new_goal.id,
                                                             high_level_goal_id=row.high_level_goal_id)
                    filtered_triple.set_entailed_triple(row.triple_filtered_from_hlg)
                    db.add(filtered_triple)
                db.commit()
                print("\nSubgoal added in the database!")

            if hlg_id != -1:
                # Add the high-level goal and the subgoal in the database (table: hierarchy)
                db_hierarchy = models.Hierarchy(high_level_goal_id=hlg_id, refinement=refinement,
                                                subgoal_id=new_goal.id)
                db.add(db_hierarchy)
                db.commit()
                print("\nUpdate the hierarchy!")

            # Save exploration parameters
            new_param = models.Exploration_Parameter(goal_id=new_goal.id, max_depth=max_depth, beam_width=beam_width)
            db.add(new_param)
            db.commit()
            print('\nParameter added in the database!')

            # Extract the entailed triples and the generated texts (to print)
            outputs = db.query(models.Outputs).filter(models.Outputs.goal_id == new_goal.id).all()

            with_generated_texts = False

            # Extract data into a list of dictionaries
            data = []
            for output in outputs:
                data.append({
                    'id': output.id,
                    'goal_id': output.goal_id,
                    'goal_type': output.goal_type,
                    'generated_text': output.generated_text,
                    'entailed_triple': output.get_entailed_triples()
                })
                if output.generated_text != "":
                    with_generated_texts = True

            # Create a DataFrame for storing all outputs
            outputs_df = pd.DataFrame(data)

            # get the end time
            et = time.time()

            # get the execution time
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')

            return templates.TemplateResponse('contextualization.html', context={
                'request': request,
                'highlevelgoal': highlevelgoal,
                'unique_triples_entailed': enumerate(unique_triples_entailed),
                'outputs': outputs_df,
                'goal_with_outputs': goal_with_outputs,
                'hlg_id': new_goal.id,
                'all_goal': all_goal,  # for the input (for autocompletion)
                'with_generated_texts': with_generated_texts,
                'beam_width': beam_width,
                'max_depth': max_depth
            })
        else:
            message = "No triple"
            print("\nNo triples!")

            new_goal = models.Goal(goal_type=goal_type, goal_name=highlevelgoal)
            db.add(new_goal)
            db.commit()

            if hlg_id != -1:
                db_hierarchy = models.Hierarchy(high_level_goal_id=hlg_id, refinement=refinement,
                                                subgoal_id=new_goal.id)
                db.add(db_hierarchy)
                db.commit()
                print("\nUpdate the hierarchy!")

            # Save exploration parameters
            new_param = models.Exploration_Parameter(goal_id=new_goal.id, max_depth=max_depth,
                                                     beam_width=beam_width)
            db.add(new_param)
            db.commit()
            print('\nParameter added in the database!')

            # get the end time
            et = time.time()

            # get the execution time
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')

            return templates.TemplateResponse('contextualization.html', context={
                'request': request,
                'highlevelgoal': highlevelgoal,
                'message': message,
                'hlg_id': new_goal.id,
                'goal_with_outputs': goal_with_outputs,
                'all_goal': all_goal,  # for the input (for autocompletion)
                'beam_width': beam_width,
                'max_depth': max_depth
            })
    else:
        return RedirectResponse(f"/contextualization/{goal_with_outputs.id}", status_code=302)


@router.get("/")
async def contextualization(request: Request, db: Session = Depends(get_db)):
    all_goal = db.query(models.Goal).all()
    return templates.TemplateResponse('contextualization.html', context={'request': request, 'all_goal': all_goal})


@router.get("/contextualization/{hlg_id}")
async def contextualization(request: Request, hlg_id: int, db: Session = Depends(get_db)):
    all_goal = db.query(models.Goal).all()

    goal_with_outputs = db.query(models.Goal).filter(models.Goal.id == hlg_id).first()
    params = db.query(models.Exploration_Parameter).filter(models.Exploration_Parameter.goal_id == hlg_id).first()

    if not goal_with_outputs:
        return RedirectResponse("/")

    highlevelgoal = goal_with_outputs.goal_name

    data = []

    with_generated_texts = False

    for output in goal_with_outputs.outputs:
        data.append({
            'id': output.id,
            'goal_id': output.goal_id,
            'goal_type': output.goal_type,
            'generated_text': output.generated_text,
            'entailed_triple': output.get_entailed_triples()
        })
        if output.generated_text != "":
            with_generated_texts = True

    # Create a DataFrame for storing all outputs
    outputs_df = pd.DataFrame(data)

    return templates.TemplateResponse('contextualization.html', context={'request': request,
                                                                         'highlevelgoal': highlevelgoal,
                                                                         'outputs': outputs_df,
                                                                         'goal_with_outputs': goal_with_outputs,
                                                                         'hlg_id': hlg_id,
                                                                         'all_goal': all_goal,
                                                                         'with_generated_texts': with_generated_texts,
                                                                         "params": params})


@router.post("/")
async def contextualization(request: Request, goal_type: str = Form(...), refinement: Optional[str] = Form(None), highlevelgoal: str = Form(...), hlg_id: int = Form(...), filtered_out_triples_with_goal_id: List[str] = Form([]), beam_width: int = Form(...), max_depth: int = Form(...), db: Session = Depends(get_db)):
    # The process is performed asynchronously in a parallel thread to allow the navigation in other parts of the app
    response = await run_in_threadpool(lambda: find_relevant_information(request, goal_type, refinement, highlevelgoal, hlg_id, filtered_out_triples_with_goal_id, beam_width, max_depth, db))
    return response
