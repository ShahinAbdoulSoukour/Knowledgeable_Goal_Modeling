import pandas as pd

from utils.functions import get_neighbors, test_entailment
import heapq
from sklearn.metrics.pairwise import cosine_similarity


# Function to convert lists to tuples, handling nested lists
def hashable_premise_serialized(premise_serialized):
    if isinstance(premise_serialized, list):
        return tuple(hashable_premise_serialized(i) for i in premise_serialized)
    return premise_serialized


def graph_explorator_bfs_optimized(df, goal, graph, model_sts, model_nli_name, tokenizer_nli, model_nli, beam_width, max_depth, use_api):
    entailed_triples_df = pd.DataFrame(columns=["GOAL_TYPE", "SUBGOALS", "SUBGOALS_SERIALIZED", "SCORE", "NLI_LABEL"])
    priority_queue = []
    visited = set()  # Use a set to track visited premises for faster lookups

    # Initialize the priority queue with initial triples
    for _, row in df[
        ["GOAL_TYPE", "PREMISE", "HYPOTHESIS", "PREMISE_SERIALIZED", "ENTAILMENT", "NLI_LABEL"]].iterrows():
        # Convert 'PREMISE_SERIALIZED' to a fully hashable tuple
        premise_serialized_tuple = hashable_premise_serialized(row["PREMISE_SERIALIZED"])

        if row['NLI_LABEL'] == "ENTAILMENT":
            entailed_triples_df.loc[len(entailed_triples_df)] = {
                "GOAL_TYPE": row['GOAL_TYPE'],
                "SUBGOALS": row["PREMISE"],
                "SUBGOALS_SERIALIZED": row["PREMISE_SERIALIZED"],
                "SCORE": row["ENTAILMENT"],
                "NLI_LABEL": row["NLI_LABEL"]
            }
        else:
            # Push a tuple containing the entailment score, row as a dictionary, and depth
            #heapq.heappush(priority_queue, (-row['ENTAILMENT'], row.to_dict(), 0))  # (negate score for max-heap, initial depth)

            # Push a tuple with four elements: (negative entailment score, unique idx, row dict, depth)
            heapq.heappush(priority_queue,
                           (-row['ENTAILMENT'], _, row.to_dict(), 0))  # (negate score for max-heap, unique ID)

    # BFS with beam search, depth check, and score comparison
    while priority_queue:
        print('\nPRIORITY QUEUE:')
        print(priority_queue)

        #current_entailment, current_row, depth = heapq.heappop(priority_queue)

        # Pop a tuple with four elements
        current_entailment, idx, current_row, depth = heapq.heappop(priority_queue)

        print('\n-- CURRENT ENTAILMENT:')
        print(current_entailment)

        current_entailment = -current_entailment  # Convert back to positive entailment score

        print('\nCURRENT ENTAILMENT:')
        print(current_entailment)

        print('\nCURRENT ROW:')
        print(current_row)

        print('\nDEPTH:')
        print(depth)

        # If depth exceeds max_depth, stop further exploration
        if depth >= max_depth:
            continue

        # Convert 'PREMISE_SERIALIZED' to a fully hashable tuple
        premise_serialized_tuple = hashable_premise_serialized(current_row["PREMISE_SERIALIZED"])

        # Avoid revisiting the same triple
        if premise_serialized_tuple in visited:
            continue
        visited.add(premise_serialized_tuple)

        print('\nVISITED:')
        print(visited)

        # Get triple neighbors
        neighbor_triples = get_neighbors(current_row["PREMISE"], current_row["PREMISE_SERIALIZED"], graph)

        if isinstance(neighbor_triples, pd.DataFrame) and neighbor_triples.empty:
            continue

        # Concatenate neighbors
        concatenated_triples = pd.DataFrame({
            "GOAL_TYPE": current_row["GOAL_TYPE"],
            "PREMISE": neighbor_triples["TRIPLE_NEIGHBOR"] + ". " + neighbor_triples['TRIPLE'],
            "HYPOTHESIS": goal,
            "PREMISE_SERIALIZED": neighbor_triples["TRIPLE_NEIGHBOR_SERIALIZED"] + neighbor_triples['TRIPLE_SERIALIZED']
        })

        # Beam search with similarity scoring
        # Compute the cosine similarity for each row based on PREMISE and HYPOTHESIS
        concatenated_triples['SIMILARITY_SCORE'] = concatenated_triples.apply(
            lambda row: cosine_similarity(
                model_sts.encode(row['PREMISE']).reshape(1, -1),  # Encode PREMISE
                model_sts.encode(row['HYPOTHESIS']).reshape(1, -1)  # Encode HYPOTHESIS (the goal)
            ).flatten()[0], axis=1  # Compute cosine similarity and flatten the result
        )
        concatenated_triples.sort_values(by='SIMILARITY_SCORE', inplace=True, ascending=False)

        # Beam search: explore only the top `beam_width` neighbors
        concatenated_triples = concatenated_triples.head(beam_width).drop('SIMILARITY_SCORE', axis=1)

        print('\nTHE TOP beam_width NEIGHBORS:')
        print(concatenated_triples)

        # Apply entailment test to all triple neighbors and sort results by entailment score
        entailment_concatenate_triples_result = test_entailment(concatenated_triples, tokenizer_nli, model_nli_name, model_nli, use_api)
        entailment_concatenate_triples_result.sort_values(by="ENTAILMENT", ascending=False, inplace=True)

        print("\nENTAILMENT TEST RESULTS (CONCATENATED TRIPLES):")
        print(entailment_concatenate_triples_result.to_string())

        # The top `beam_width` neighbors
        top_k_neighbors = entailment_concatenate_triples_result

        # Track the highest entailment score and check for entailment
        found_entailment = False
        previous_entailment_score = current_entailment  # Start with current entailment
        for _, neighbor_row in top_k_neighbors.iterrows():
            new_entailment = neighbor_row["ENTAILMENT"]

            # Check if this neighbor entails the goal
            if neighbor_row['NLI_LABEL'] == 'ENTAILMENT':
                # Store the entailing triple and stop further exploration for this anchor triple
                entailed_triples_df = pd.concat([
                    entailed_triples_df,
                    pd.DataFrame({
                        "GOAL_TYPE": [neighbor_row["GOAL_TYPE"]],
                        "SUBGOALS": [neighbor_row["PREMISE"]],
                        "SUBGOALS_SERIALIZED": [neighbor_row["PREMISE_SERIALIZED"]],
                        "SCORE": [neighbor_row["ENTAILMENT"]],
                        "NLI_LABEL": [neighbor_row["NLI_LABEL"]]
                    })
                ])
                found_entailment = True  # Set flag to indicate an entailment was found
                break  # Stop exploration for this current triple

            # Check if entailment score is increasing
            if new_entailment > previous_entailment_score:
                # If score improves, push this neighbor for further exploration
                heapq.heappush(priority_queue, (-new_entailment, idx, neighbor_row.to_dict(), depth + 1))
                previous_entailment_score = new_entailment  # Update the previous score
            else:
                # If the entailment score decreases, stop exploring further for this triple
                break

        # If we found an entailment, move on to the next anchor triple
        if found_entailment:
            continue

    # Return sorted results by entailment score
    print("\n=> ENTAILED TRIPLES:")
    print(entailed_triples_df.sort_values(['SCORE'], ascending=[False]).to_string())
    return entailed_triples_df