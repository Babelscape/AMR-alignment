import json
from tkinter import E
import numpy as np
from pydantic import NonPositiveFloat

INIT_TOKEN = 'Ġ'

relations_tokens = [f"{INIT_TOKEN}:compared-to", f"{INIT_TOKEN}:consist", f"{INIT_TOKEN}:op1", f"{INIT_TOKEN}:op2", f"{INIT_TOKEN}:op3", f"{INIT_TOKEN}:op4", f"{INIT_TOKEN}:op5", f"{INIT_TOKEN}:ARG0", f"{INIT_TOKEN}:ARG1", f"{INIT_TOKEN}:ARG2", f"{INIT_TOKEN}:ARG3", f"{INIT_TOKEN}:ARG4", f"{INIT_TOKEN}:ARG5", f"{INIT_TOKEN}:ARG6", f"{INIT_TOKEN}:ARG7", f"{INIT_TOKEN}:ARG8", f"{INIT_TOKEN}:ARG9", f"{INIT_TOKEN}:ARG10", f"{INIT_TOKEN}:ARG11", f"{INIT_TOKEN}:ARG12", f"{INIT_TOKEN}:ARG13", f"{INIT_TOKEN}:ARG14", f"{INIT_TOKEN}:ARG15", f"{INIT_TOKEN}:ARG16", f"{INIT_TOKEN}:ARG17", f"{INIT_TOKEN}:ARG18", f"{INIT_TOKEN}:ARG19", f"{INIT_TOKEN}:ARG20", f"{INIT_TOKEN}:accompanier", f"{INIT_TOKEN}:age", f"{INIT_TOKEN}:beneficiary", f"{INIT_TOKEN}:calendar", f"{INIT_TOKEN}:cause", f"{INIT_TOKEN}:century", f"{INIT_TOKEN}:concession", f"{INIT_TOKEN}:condition", f"{INIT_TOKEN}:conj-as-if", f"{INIT_TOKEN}:consist-of", f"{INIT_TOKEN}:cost", f"{INIT_TOKEN}:day", f"{INIT_TOKEN}:dayperiod", f"{INIT_TOKEN}:decade", f"{INIT_TOKEN}:degree", f"{INIT_TOKEN}:destination", f"{INIT_TOKEN}:direction", f"{INIT_TOKEN}:domain", f"{INIT_TOKEN}:duration", f"{INIT_TOKEN}:employed-by", f"{INIT_TOKEN}:era", f"{INIT_TOKEN}:example", f"{INIT_TOKEN}:extent", f"{INIT_TOKEN}:frequency", f"{INIT_TOKEN}:instrument", f"{INIT_TOKEN}:li", f"{INIT_TOKEN}:location", f"{INIT_TOKEN}:manner", f"{INIT_TOKEN}:meaning", f"{INIT_TOKEN}:medium", f"{INIT_TOKEN}:mod", f"{INIT_TOKEN}:mode", f"{INIT_TOKEN}:month", f"{INIT_TOKEN}:name", f"{INIT_TOKEN}:ord", f"{INIT_TOKEN}:part", f"{INIT_TOKEN}:path", f"{INIT_TOKEN}:polarity", f"{INIT_TOKEN}:polite", f"{INIT_TOKEN}:poss", f"{INIT_TOKEN}:purpose", f"{INIT_TOKEN}:quant", f"{INIT_TOKEN}:quarter", f"{INIT_TOKEN}:range", f"{INIT_TOKEN}:relation", f"{INIT_TOKEN}:role", f"{INIT_TOKEN}:scale", f"{INIT_TOKEN}:season", f"{INIT_TOKEN}:source", f"{INIT_TOKEN}:subevent", f"{INIT_TOKEN}:subset", f"{INIT_TOKEN}:superset", f"{INIT_TOKEN}:time", f"{INIT_TOKEN}:timezone", f"{INIT_TOKEN}:topic", f"{INIT_TOKEN}:unit", f"{INIT_TOKEN}:value", f"{INIT_TOKEN}:weekday", f"{INIT_TOKEN}:wiki", f"{INIT_TOKEN}:year", f"{INIT_TOKEN}:year2", f"{INIT_TOKEN}:snt0", f"{INIT_TOKEN}:snt1", f"{INIT_TOKEN}:snt2", f"{INIT_TOKEN}:snt3", f"{INIT_TOKEN}:snt4", f"{INIT_TOKEN}:snt5"]
spring_reference_tokens = [f"{INIT_TOKEN}I", f"{INIT_TOKEN}he", f"{INIT_TOKEN}she", f"{INIT_TOKEN}we", f"{INIT_TOKEN}you", f"{INIT_TOKEN}they", f"{INIT_TOKEN}our", f"{INIT_TOKEN}us", f"{INIT_TOKEN}its", f"{INIT_TOKEN}mine", f"{INIT_TOKEN}me", f"{INIT_TOKEN}your", f"{INIT_TOKEN}his", f"{INIT_TOKEN}him", f"{INIT_TOKEN}her", f"{INIT_TOKEN}their", f"{INIT_TOKEN}:it"]
spring_op_relations = [f"{INIT_TOKEN}:op1", f"{INIT_TOKEN}:op2", f"{INIT_TOKEN}:op3", f"{INIT_TOKEN}:op4", f"{INIT_TOKEN}:op5"]


from asyncore import write
import json
import argparse
import os   
import sys
import errno

from tqdm import tqdm


# method read multiple json object from a file
def read_json(file_name):
    sentence = []
    for line in open(file_name, 'r'):
        sentence.append(json.loads(line))
    return sentence


def check_exist_directory(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# build graph maps from tokenized graph
def build_graph_maps(graph_tokens):
    
    # create a map that aligned tokenized graph to the original graphs
    current_node_pos = 1
    node_pos_map = {}
    node_pos_stack = []

    target_node_map = {}
    node_pos = str(current_node_pos)
    target_node_map[2] = node_pos
    node_pos_stack.append(current_node_pos)
    current_node_pos = 1

    reentrancy_map = {}
    non_reentrancy_map = {'<pointer:0>':"1"}
    named_entities_map = {}

    is_lit = False
    is_named_entity = False
    name_entity_root = None
    prev_wiki = False
    
    for token_idx, token in enumerate(graph_tokens):
        if token == f"{INIT_TOKEN}<lit>":
            is_lit = True
        elif token == f"{INIT_TOKEN}</lit>":
            is_lit = False
        
        if not is_lit and token == f"{INIT_TOKEN})":
            is_named_entity = False
            name_entity_root = None

        if  token == f"{INIT_TOKEN}:wiki":
            current_node_pos += 1
            prev_wiki = True
        elif not is_lit:
            if token.startswith(f"{INIT_TOKEN}:"):
                next_token_idx = token_idx + 1
                while not graph_tokens[next_token_idx].startswith(INIT_TOKEN) or  graph_tokens[next_token_idx] == f"{INIT_TOKEN}<lit>" or graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}op")  or  graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}snt")   or (graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}s") and graph_tokens[next_token_idx + 1].startswith("n"))or graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}prep"):
                    next_token_idx += 1

            if token.startswith(f"{INIT_TOKEN}:") and graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}("):
                if token.startswith(f"{INIT_TOKEN}:name"):
                    is_named_entity = True
                    name_entity_root = node_pos + "." + str(current_node_pos)
                    if not (current_node_pos == 2 and not prev_wiki):
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                    else: 
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos)

                    
                    prev_wiki = False

                next_token_idx += 1
                node_pos += "." + str(current_node_pos)
                current_node_pos += 1
                node_pos_stack.append(current_node_pos)
                current_node_pos = 1
                token = graph_tokens[next_token_idx]
                target_node_map[next_token_idx] = node_pos
                target_node_map[token_idx] = node_pos + ".r"
                # node_span_map[node_pos] = pos2spans_map[input_word_pos_map[np.argmax(alignment_score[next_token_idx, :])]]
                # node_span_map[node_pos + ".r"] = pos2spans_map[input_word_pos_map[np.argmax(alignment_score[token_idx, :])]]

                non_reentrancy_map[graph_tokens[next_token_idx].replace(f"{INIT_TOKEN}", "")] = node_pos

                if is_named_entity:                    
                    named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                    named_entities_map.setdefault(name_entity_root, []).append(node_pos + ".r")

            elif token.startswith(f"{INIT_TOKEN}:") and not graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}("):
                node_pos += "." + str(current_node_pos)

                target_node_map[next_token_idx] = node_pos
                target_node_map[token_idx] = node_pos + ".r"
                # node_span_map[node_pos] = pos2spans_map[input_word_pos_map[np.argmax(alignment_score[next_token_idx, :])]]
                # node_span_map[node_pos + ".r"] = pos2spans_map[input_word_pos_map[np.argmax(alignment_score[token_idx, :])]]
                
                if is_named_entity:
                    named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                    named_entities_map.setdefault(name_entity_root, []).append(node_pos + ".r")

                if graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}<pointer"):
                    reentrancy_map[node_pos] = graph_tokens[next_token_idx].replace(f"{INIT_TOKEN}", "")

                current_node_pos += 1
                node_pos = ".".join(node_pos.split(".")[:-1])

        

            elif  token.startswith(f"{INIT_TOKEN})") and node_pos_stack:
                node_pos = ".".join(node_pos.split(".")[:-1])
                current_node_pos = node_pos_stack.pop()
    
    return target_node_map, reentrancy_map, non_reentrancy_map, named_entities_map


def create_graph_structure_tokens_map(target_node_map, graph_tokens):

    # create map relate node position to graph token
    graph_id_map = {}
    graph_nodes_map = {}

    for graph_idx, graph_token in target_node_map.items():
        next_token = 1 if graph_tokens[graph_idx].startswith(f"{INIT_TOKEN}<pointer") and not (graph_tokens[graph_idx + 1].startswith(f"{INIT_TOKEN}:") or graph_tokens[graph_idx + 1] == f"{INIT_TOKEN})") else 0

        graph_id = graph_tokens[graph_idx].replace(f"{INIT_TOKEN}", "")
        graph_node = graph_tokens[graph_idx + next_token].replace(f"{INIT_TOKEN}", "")
        
        next_token += 1
        is_prep_edge = graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN}prep"

        while (graph_idx + next_token) < len(graph_tokens) \
                and (not graph_tokens[graph_idx + next_token].startswith(INIT_TOKEN) \
                    or (is_prep_edge and (not graph_tokens[graph_idx + next_token].startswith(INIT_TOKEN) \
                        or graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN}prep")
                    or (graph_tokens[graph_idx].startswith(f"{INIT_TOKEN}<pointer") and graph_id != graph_node and not (graph_tokens[graph_idx + next_token].startswith(f"{INIT_TOKEN}:") or graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN})")))):

            graph_node += graph_tokens[graph_idx + next_token].lstrip(f"{INIT_TOKEN}")
            next_token += 1


        graph_id_map[graph_token] = graph_id
        graph_nodes_map[graph_token] = graph_node

    return graph_id_map, graph_nodes_map


def conver_leamr_isi(leamr_file, isi_file, graphs_list):

    # check if exist leamr fike
    if not os.path.exists(leamr_file):
        # print('Input file does not exist')
        sys.exit(1)

    # check if exist isi file
    check_exist_directory(isi_file)

    # build graph tokens map
    target_node_map, reentrancy_map, _, _ = build_graph_maps(graphs_list)

    # create graph structure tokens map
    _, graph_nodes_map = create_graph_structure_tokens_map(target_node_map, graphs_list)
    
    inv_reentrancy_map = {v + " " + graph_nodes_map[k + ".r"]: k for k, v in reentrancy_map.items()}
    
    with open(leamr_file, 'r') as f:
        data = json.load(f)
    with open(isi_file, 'w') as f:
        for sentence in data:
            alignments = data[sentence]
            output_alignments = ''
            for alignment in alignments:
                for token in alignment["tokens"]:
                    if "nodes" in alignment:
                        for node in alignment["nodes"]:
                            output_alignments += str(token) + '-' + node + ' '
                    if "edges" in alignment:
                        for triplet in alignment["edges"]:
                            if triplet[0] + " " + triplet[1] in inv_reentrancy_map:
                                if alignment["type"] not in ["relation", "reentrancy:primary"]:
                                    output_alignments += str(token) + '-' + inv_reentrancy_map[triplet[0] + " " + triplet[1]]
                                else:
                                    output_alignments += str(token) + '-' + inv_reentrancy_map[triplet[0] + " " + triplet[1]] + '.r '
                            else:                                
                                output_alignments += str(token) + '-' + triplet[2] + '.r '
            output_alignments = output_alignments[:-1]
            f.write(sentence + '\t' + output_alignments + '\n')
    # print('Done')

def is_consecutive_span(collapse_pos, pos):
    positions = collapse_pos.split('-')
    for position in positions:
        # if position is consecutive (abs diff is 1)
        if abs(int(position) - int(pos)) == 1:
            return True


def compare_alignment_with_gold(pred_alignments_path, gold_alignments_path):
    pred_alignment = read_json(pred_alignments_path)[0]
    gold_alignment = read_json(gold_alignments_path)[0]
    pred_filtered_alignment = {}
    
    for k,v in pred_alignment.items():
        new_values = []
        for alignment in v:
            if alignment["nodes"] or alignment["edges"]:
                new_values.append(alignment)
        
        pred_filtered_alignment[k] = new_values

    message = ""
    missing_things = {}
    # compare elements
    for sentence_id, alignments in gold_alignment.items():
        missing_things[sentence_id] = []

        for alignment in alignments:
            alignment_idx =  alignment["tokens"][0]
            alignment_gold = None
            for alignment_aux in pred_filtered_alignment[sentence_id]:
                if alignment_aux["tokens"][0] == alignment_idx:
                    alignment_gold = alignment_aux
                    break

            if not alignment_gold:
                missing_things[sentence_id].append(f"Recall, not found: {str(alignment['tokens'])}")

            else:
                pred_alignmets = set(alignment["nodes"])
                gold_alignments = set(alignment_gold["nodes"])

                # intersection
                intersection = pred_alignmets.intersection(gold_alignments)

                wrong_predictions = pred_alignmets - intersection
                missing_prediction = gold_alignments - intersection

                if missing_prediction:
                    missing_things[sentence_id].append(f"Recall Missing: {' '.join(missing_prediction)}")

                if wrong_predictions:
                    missing_things[sentence_id].append(f"Recall wrong: {wrong_predictions}")


    # compare elements
    for sentence_id, alignments in pred_filtered_alignment.items():

        for alignment in alignments:
            alignment_idx =  alignment["tokens"][0]
            alignment_gold = None
            for alignment_aux in gold_alignment[sentence_id]:
                if alignment_aux["tokens"][0] == alignment_idx:
                    alignment_gold = alignment_aux
                    break

            if not alignment_gold:
                missing_things[sentence_id].append(f"Precission, not found: {str(alignment['tokens'])}")

            else:
                pred_alignmets = set(alignment["nodes"])
                gold_alignments = set(alignment_gold["nodes"])

                # intersection
                intersection = pred_alignmets.intersection(gold_alignments)

                wrong_predictions = pred_alignmets - intersection
                missing_prediction = gold_alignments - intersection

                if missing_prediction:
                    missing_things[sentence_id].append(f"Precission missing: {' '.join(missing_prediction)}")

                if wrong_predictions:
                    missing_things[sentence_id].append(f"Precission wrong: {' '.join(wrong_predictions)}")

    for sentence_id, missing_things_list in missing_things.items():
        if missing_things_list:
            datos = '\n\t'.join(missing_things_list)
            message += f"{sentence_id}\n: {datos}\n"
            print(sentence_id)
            print('\t' + '\n\t'.join(missing_things_list))

    return message




def compare_alignment_with_gold_relation(pred_alignments_path, gold_alignments_path):
    pred_alignment = read_json(pred_alignments_path)[0]
    gold_alignment = read_json(gold_alignments_path)[0]

    missing_things = {}
    # compare elements
    for sentence_id, alignments in gold_alignment.items():
        missing_things[sentence_id] = []

        for alignment in alignments:
            alignment_idx =  alignment["tokens"][0]
            alignment_gold = None
            for alignment_aux in pred_alignment[sentence_id]:
                if alignment_aux["tokens"][0] == alignment_idx:
                    alignment_gold = alignment_aux
                    break

            if not alignment_gold:
                missing_things[sentence_id].append(f"Recall, not found: {str(alignment['tokens'])}")

            else:
                str_alignments = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment["edges"]]
                str_gold_alignment = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment_gold["edges"]]

                pred_alignmets = set(str_alignments)
                gold_alignments = set(str_gold_alignment)

                # intersection
                intersection = pred_alignmets.intersection(gold_alignments)

                wrong_predictions = pred_alignmets - intersection
                missing_prediction = gold_alignments - intersection

                if missing_prediction:
                    if sentence_id == "AUSTINS_EXAMPLE":
                        print(f"Recall Missing: {'    '.join(missing_prediction)}")
                        print(alignments)
                        print(pred_alignmets)
                        print(gold_alignments)
                        exit()
                        
                    missing_things[sentence_id].append(f"Recall Missing: {'    '.join(missing_prediction)}")

                if wrong_predictions:
                    missing_things[sentence_id].append(f"Recall wrong: {wrong_predictions}")


    # compare elements
    for sentence_id, alignments in pred_alignment.items():

        for alignment in alignments:
            alignment_idx =  alignment["tokens"][0]
            alignment_gold = None
            for alignment_aux in gold_alignment[sentence_id]:
                if alignment_aux["tokens"][0] == alignment_idx:
                    alignment_gold = alignment_aux
                    break

            if not alignment_gold:
                missing_things[sentence_id].append(f"Precission, not found: {str(alignment['tokens'])}")

            else:
                str_alignments = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment["edges"]]
                str_gold_alignment = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment_gold["edges"]]

                pred_alignmets = set(str_alignments)
                gold_alignments = set(str_gold_alignment)

                # intersection
                intersection = pred_alignmets.intersection(gold_alignments)

                wrong_predictions = pred_alignmets - intersection
                missing_prediction = gold_alignments - intersection

                if missing_prediction:
                    missing_things[sentence_id].append(f"Precission missing: {'    '.join(missing_prediction)}")

                if wrong_predictions or missing_prediction:
                    missing_things[sentence_id].append(f"Precission wrong: {'    '.join(wrong_predictions)}")

    for sentence_id, missing_things_list in missing_things.items():
        if missing_things_list:
            print(sentence_id)
            print('\t' + '\n\t'.join(missing_things_list))



def compare_alignment_with_gold_reentrancy(pred_alignments_path, gold_alignments_path):
    pred_alignment = read_json(pred_alignments_path)[0]
    gold_alignment = read_json(gold_alignments_path)[0]

    missing_things = {}
    alignments_tokes_gold = {}
    alignments_tokes_pred = {}

    # compare elements
    for sentence_id, alignments in gold_alignment.items():
        missing_things[sentence_id] = []

        alignments_tokes = {}
        for alignment in alignments:
            alignments_tokes.setdefault(alignment["tokens"][0], []).append(alignment)

        alignments_tokes_gold[sentence_id] = alignments_tokes

    for sentence_id, alignments in pred_alignment.items():
        alignments_tokes = {}
        for alignment in alignments:
            alignments_tokes.setdefault(alignment["tokens"][0], []).append(alignment)

            alignments_tokes = {}
            for alignment in alignments:
                alignments_tokes.setdefault(alignment["tokens"][0], []).append(alignment)

            alignments_tokes_pred[sentence_id] = alignments_tokes

    for sentence_id, alignments_tokes in alignments_tokes_gold.items():
        for token_pos, alignments in alignments_tokes.items():

            if not alignment_gold:
                missing_things[sentence_id].append(f"Recall, not found: {str(alignment['tokens'])}")

            else:
                str_alignments = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment["edges"]]
                str_gold_alignment = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment_gold["edges"]]

                pred_alignmets = set(str_alignments)
                gold_alignments = set(str_gold_alignment)

                # intersection
                intersection = pred_alignmets.intersection(gold_alignments)

                wrong_predictions = pred_alignmets - intersection
                missing_prediction = gold_alignments - intersection

                if missing_prediction:
                    if sentence_id == "AUSTINS_EXAMPLE":
                        print(f"Recall Missing: {'    '.join(missing_prediction)}")
                        print(alignments)
                        print(pred_alignmets)
                        print(gold_alignments)
                        exit()
                        
                    missing_things[sentence_id].append(f"Recall Missing: {'    '.join(missing_prediction)}")

                if wrong_predictions:
                    missing_things[sentence_id].append(f"Recall wrong: {wrong_predictions}")


    # compare elements
    for sentence_id, alignments in pred_alignment.items():

        for alignment in alignments:
            alignment_idx =  alignment["tokens"][0]
            alignment_gold = None
            for alignment_aux in gold_alignment[sentence_id]:
                if alignment_aux["tokens"][0] == alignment_idx:
                    alignment_gold = alignment_aux
                    break

            if not alignment_gold:
                missing_things[sentence_id].append(f"Precission, not found: {str(alignment['tokens'])}")

            else:
                str_alignments = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment["edges"]]
                str_gold_alignment = [s + " " + r + " " + o if r != None else s + " NONE " + o for s,r,o in alignment_gold["edges"]]

                pred_alignmets = set(str_alignments)
                gold_alignments = set(str_gold_alignment)

                # intersection
                intersection = pred_alignmets.intersection(gold_alignments)

                wrong_predictions = pred_alignmets - intersection
                missing_prediction = gold_alignments - intersection

                if missing_prediction:
                    missing_things[sentence_id].append(f"Precission missing: {'    '.join(missing_prediction)}")

                if wrong_predictions or missing_prediction:
                    missing_things[sentence_id].append(f"Precission wrong: {'    '.join(wrong_predictions)}")

    for sentence_id, missing_things_list in missing_things.items():
        if missing_things_list:
            print(sentence_id)
            print('\t' + '\n\t'.join(missing_things_list))

def generate_leamr_alignment(leamr_type, sentence_pos, nodes, edges, sentence_token_map, graph_tokens_map, reentrancy_map):
    if_nodes_edges = ", " if nodes and edges else ""
    if_edges = "(" if edges else ""
    if_edges_ends = "')" if edges else ""
    
    # if sentence_pos is list
    if not isinstance(sentence_pos, list):
        sentence_pos = sentence_pos.split('-')

    leamr_alignment = {'type': leamr_type, 
                        'tokens': [int(pos) for pos in sentence_pos], 
                        'nodes': [node_pos for node_pos in nodes], 
                        'edges': [[edge[:-4], graph_tokens_map[edge], edge[:-2] if edge[:-2] not in reentrancy_map else reentrancy_map[edge[:-2]]]  for edge in edges], 
                        'string': leamr_type + ' : ' + " ".join([sentence_token_map[int(pos)] for pos in sentence_pos]) 
                                        + ' => ' + ", ".join([graph_tokens_map[node_pos] for node_pos in nodes]) 
                                        + if_nodes_edges + if_edges + "'), (".join([ "'" + graph_tokens_map[edge[:-4]] + "', '" + graph_tokens_map[edge] + "', '" + graph_tokens_map[edge[:-2]]  for edge in edges]) + if_edges_ends}
    return leamr_alignment


# implement subgraphs rules:
# 1. Collapse name entities subgraphs
# 2. Collapse date-entity
# 3. Collapse quantity-entity
# 3. Collapse place holders (government-organization, person) same unique world
# 4. collapse 91 roles
def conver_isi_leamr( isi_aligment, sentence_token_map, graph_id_map, graph_tokens_map):
   
    leamr_subgraph_alignments = []
    leamr_relation_alignments = []    
    leamr_reentrancy_alignments = []

    word_nodes_map = {}
    reentrancy_alignment = {}

    for alignment in isi_aligment:
        sentence_pos = alignment[0]
        node_pos = alignment[1]
        if not graph_tokens_map[node_pos].startswith("<pointer"):
            word_nodes_map.setdefault(sentence_pos, set()).add(node_pos)
        else:
            reentrancy_alignment[sentence_pos] = node_pos + ".r"

    collapse_word_nodes_map = {}
    # iterate overitems  word_nodes_map
    collapse_named_entities = {}

    for sentence_pos, nodes in word_nodes_map.items():
        is_same = False
        is_entity = False
        for node_pos in nodes:
            if node_pos in graph_tokens_map and (graph_tokens_map[node_pos] == ":name" or graph_tokens_map[node_pos] == "name") \
                 or (graph_tokens_map[node_pos] in ["date-entity", "quantity-entity", "government-organization", "person", "expressive"]) \
                 or (graph_tokens_map[node_pos].endswith("-91")):
                is_entity = True
                break

        if is_entity:
            # print(sentence_pos)
            # print(sentence_token_map[sentence_pos])
            for collapse_sentence_pos, collapse_nodes in collapse_named_entities.items():
                # nodes intersection if

                if len(nodes.intersection(collapse_nodes)) > 0 and is_consecutive_span(collapse_sentence_pos, str(sentence_pos)):
                    is_same = True
                    # change name of key in collapse_word_nodes_map
                    collapse_named_entities[collapse_sentence_pos + "-" + str(sentence_pos)] = collapse_nodes.union(nodes)

                    # remve old key
                    del collapse_named_entities[collapse_sentence_pos]
                    break
        

            if not is_same:
                collapse_named_entities[str(sentence_pos)] = nodes
        else:
            for collapse_sentence_pos, collapse_nodes in collapse_word_nodes_map.items():
                # nodes intersection if

                if len(nodes.intersection(collapse_nodes)) > 0 and is_consecutive_span(collapse_sentence_pos, str(sentence_pos)):
                    is_same = True
                    # change name of key in collapse_word_nodes_map
                    collapse_word_nodes_map[collapse_sentence_pos + "-" + str(sentence_pos)] = collapse_nodes.union(nodes)

                    # remve old key
                    del collapse_word_nodes_map[collapse_sentence_pos]
                    break
        
            
            if not is_same:
                collapse_word_nodes_map[str(sentence_pos)] = nodes

    reentrancy_collapse = {}
    # iterate overitems  word_nodes_map
    for sentence_pos, node in reentrancy_alignment.items():
        is_same = False
        for collapse_sentence_pos, collapse_node in reentrancy_collapse.items():
            # nodes intersection if
            if node == collapse_node:
                is_same = True

                # change name of key in collapse_word_nodes_map
                reentrancy_collapse[collapse_sentence_pos + "-" + str(sentence_pos)] = node

                # remve old key
                del reentrancy_collapse[collapse_sentence_pos]
                break

        if not is_same:
            reentrancy_collapse[str(sentence_pos)] = node
            
    # print(graph_tokens_map)
    # print(collapse_word_nodes_map)
    # print(collapse_named_entities)


    delete_relations = []
    # generate missing relations alignments
    for sentence_pos, nodes in collapse_named_entities.items():
        generated_relations = set()

        for node_pos in nodes:
            if node_pos[:-2] in nodes and not node_pos+".r" in nodes:
                generated_relations.add(node_pos+".r")
                delete_relations.append(node_pos+".r")

        nodes.union(generated_relations)


    # combine collapse named entities with collapse word nodes
    collapse_word_nodes_map.update(collapse_named_entities)


    for sentence_pos, nodes_pos in collapse_word_nodes_map.items():
        nodes = [node_pos for node_pos in nodes_pos if not node_pos.endswith('.r') and not graph_tokens_map[node_pos].startswith("<pointer")]
        edges = [edge for edge in nodes_pos if edge.endswith('.r')]

        is_subgraph = True
        for edge in edges:
            if edge[:-2] not in nodes or edge[:-4] not in nodes:
                is_subgraph = False
                break
        

        # generate leamr alignment depending on the case
        if is_subgraph:
            leamr_subgraph_alignments.append(generate_leamr_alignment('subgraph', sentence_pos, nodes, edges, sentence_token_map, graph_tokens_map))

        else:
            # filter out delteded relations
            edges = [edge for edge in edges if edge not in delete_relations]
            leamr_subgraph_alignments.append(generate_leamr_alignment('subgraph', sentence_pos, nodes, [], sentence_token_map, graph_tokens_map))
            leamr_relation_alignments.append(generate_leamr_alignment('relation', sentence_pos, [], edges, sentence_token_map, graph_tokens_map))

    for sentence_pos, nodes_pos in reentrancy_collapse.items():
        leamr_reentrancy_alignments.append(generate_leamr_alignment('reentrancy', sentence_pos, [], [nodes_pos], sentence_token_map, graph_tokens_map))


    return leamr_subgraph_alignments, leamr_relation_alignments, leamr_reentrancy_alignments


# method extract the alignment from a json object
def extract_alignment(sentence_id, sentence_tokens, graph_tokens, alignment_score):
    # create numpy array from the alignment score
    alignment_score = np.array(alignment_score)
    
    # create a map that aligned tokenized sentence to the original sentence
    input_word_pos_map = {}
    pos = 0
    for word_idx, word in enumerate(sentence_tokens):
        if word.startswith(INIT_TOKEN) and word != f"{INIT_TOKEN}<s>" and word != f"{INIT_TOKEN}</s>" and not (word == f"{INIT_TOKEN}<" and (word_idx + 1) < len(sentence_tokens) and sentence_tokens[word_idx + 1] == "a"):
            input_word_pos_map[word_idx] = pos
            pos += 1

    # create a map that aligned tokenized graph to the original graphs
    current_node_pos = 1
    node_pos_map = {}
    node_pos_stack = []

    target_node_map = {}
    node_pos = str(current_node_pos)
    target_node_map[2] = node_pos
    node_pos_stack.append(current_node_pos)
    current_node_pos = 1

    if sentence_id == "nw.wsj_0012.1":
        pass


    is_lit = False
    for token_idx, token in enumerate(graph_tokens):
        if  token == f"{INIT_TOKEN}:wiki":
            current_node_pos += 1
        else:
            if token.startswith(f"{INIT_TOKEN}:"):
                next_token_idx = token_idx + 1
                while not graph_tokens[next_token_idx].startswith(INIT_TOKEN) or  graph_tokens[next_token_idx] == f"{INIT_TOKEN}<lit>" or graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}op")  or  graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}snt")   or (graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}s") and graph_tokens[next_token_idx + 1].startswith("n"))or graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}prep"):
                    next_token_idx += 1

            if token.startswith(f"{INIT_TOKEN}:") and graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}("):
                next_token_idx += 1
                node_pos += "." + str(current_node_pos)
                current_node_pos += 1
                node_pos_stack.append(current_node_pos)
                current_node_pos = 1
                token = graph_tokens[next_token_idx]
                target_node_map[token_idx] = node_pos + ".r"
                target_node_map[next_token_idx] = node_pos

            elif token.startswith(f"{INIT_TOKEN}:") and not graph_tokens[next_token_idx].startswith(f"{INIT_TOKEN}("):
                node_pos += "." + str(current_node_pos)
                target_node_map[token_idx] = node_pos + ".r"
                target_node_map[next_token_idx] = node_pos

                current_node_pos += 1
                node_pos = ".".join(node_pos.split(".")[:-1])

            elif  token.startswith(f"{INIT_TOKEN})") and node_pos_stack:
                node_pos = ".".join(node_pos.split(".")[:-1])
                current_node_pos = node_pos_stack.pop()


    alignment_entities_score = alignment_score.copy()

    # remove score from stop words from sentence
    stop_words_sentence = ['<s>', f'{INIT_TOKEN}the', '.', '</s>', f'{INIT_TOKEN};', 
                            f'{INIT_TOKEN}-', f'{INIT_TOKEN},', f'{INIT_TOKEN}@', f'{INIT_TOKEN}<pad>', 
                            f'{INIT_TOKEN};', f'{INIT_TOKEN}.', f'{INIT_TOKEN}:']

    for sentence_token_idx, sentence_token in enumerate(sentence_tokens):
        if sentence_token in stop_words_sentence:
            alignment_score[:,sentence_token_idx] = 0 


    # remove score from stop words from graph and wikinodes
    stop_words_graph =  [f'{INIT_TOKEN}(', f'{INIT_TOKEN})', '<s>', '</s>', f'{INIT_TOKEN}:wiki', f'{INIT_TOKEN}<lit>', f'{INIT_TOKEN}</lit>']   
    is_lit = False
    is_wiki = False
    is_broken_rel = False
    for graph_token_idx, graph_token in enumerate(graph_tokens):
        if graph_token == f"{INIT_TOKEN}:wiki":
            is_wiki = True
        elif graph_token == f"{INIT_TOKEN}<lit>":
            is_lit = True
        elif graph_token == f"{INIT_TOKEN}</lit>":
            is_wiki = False
            is_lit = False
        elif graph_token == f'{INIT_TOKEN}-' and graph_tokens[graph_token_idx - 1] == f"{INIT_TOKEN}:wiki":
            is_wiki = False
            alignment_score[graph_token_idx, :] = 0 
        elif graph_token == f'{INIT_TOKEN}:' and not is_lit:
            is_broken_rel = True
        elif is_broken_rel and graph_tokens[graph_token_idx - 1] != f'{INIT_TOKEN}:' and graph_token.startswith(f'{INIT_TOKEN}'):
            is_broken_rel = False

        if graph_token in stop_words_graph or (is_wiki and is_lit) or is_broken_rel:
            alignment_score[graph_token_idx, :] = 0 


        
    # identify compound tokens in the sentence and sum the values
    sentence_tokens_filter = [token_idx  for token_idx, token in enumerate(sentence_tokens) if not token.startswith(INIT_TOKEN) and token not in ['<s>', '</s>']]
    sentence_tokens_map = {}
    for token_idx in sentence_tokens_filter:
        sentence_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in sentence_tokens_map \
                                                        else sentence_tokens_map[token_idx - 1]

    for split_token_idx in sentence_tokens_filter:
        alignment_score[:, sentence_tokens_map[split_token_idx]] += alignment_score[:, split_token_idx]
        alignment_score[:, split_token_idx] = 0

    #identify compound tokens in the graph tokens and sum the values
    graph_tokens_filter = [token_idx  for token_idx, token in enumerate(graph_tokens) if not token.startswith(INIT_TOKEN)]
    graph_tokens_map = {}
    for token_idx in graph_tokens_filter:
        graph_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in graph_tokens_map \
                                                    else graph_tokens_map[token_idx - 1]

    for split_token_idx in graph_tokens_filter:
        alignment_score[graph_tokens_map[split_token_idx],:] += alignment_score[split_token_idx,:]
        alignment_score[split_token_idx,:] = 0


    # combine pointers with nodes values: not counting ")" nor links
    for graph_token_idx, graph_token in enumerate(graph_tokens):
        if graph_tokens[graph_token_idx - 1].startswith(f"{INIT_TOKEN}<pointer") \
            and graph_token != f"{INIT_TOKEN})" \
            and graph_token not in relations_tokens:
            alignment_score[graph_token_idx - 1,:] += alignment_score[graph_token_idx,:]
            alignment_score[graph_token_idx,:] = 0



    # alignment_output.append(head_alignment)
    results_layer = alignment_score.copy()
    results_layer[results_layer < 2] = 0        

    # extract sentence word related to word position to token in sentence
    sentence_words_map = {}
    for encoder_pos, sentence_token_pos in input_word_pos_map.items():
        sentence_word = sentence_tokens[encoder_pos].replace(f"{INIT_TOKEN}", "")
        next_token = 1
        while (encoder_pos + next_token) < len(sentence_tokens) and not sentence_tokens[encoder_pos + next_token].startswith(INIT_TOKEN):
            sentence_word += sentence_tokens[encoder_pos + next_token]
            next_token += 1

        sentence_words_map[sentence_token_pos] = sentence_word


    # create map relate node position to graph token
    graph_id_map = {}
    graph_nodes_map = {}

    for graph_idx, graph_token in target_node_map.items():
        next_token = 1 if graph_tokens[graph_idx].startswith(f"{INIT_TOKEN}<pointer") and not (graph_tokens[graph_idx + 1].startswith(f"{INIT_TOKEN}:") or graph_tokens[graph_idx + 1] == f"{INIT_TOKEN})") else 0

        graph_id = graph_tokens[graph_idx].replace(f"{INIT_TOKEN}", "")
        graph_node = graph_tokens[graph_idx + next_token].replace(f"{INIT_TOKEN}", "")

        next_token += 1
        while(graph_idx + next_token) < len(graph_tokens) and not graph_tokens[graph_idx + next_token].startswith(INIT_TOKEN):
            graph_node += graph_tokens[graph_idx + next_token]
            next_token += 1

        graph_id_map[graph_token] = graph_id
        graph_nodes_map[graph_token] = graph_node


    # extrsact isi alignment
    head_alignment = sentence_id
    alignments = []

    for graph_idx in range(len(results_layer)):
        for sentence_idx in range(len(results_layer[graph_idx])):
            if results_layer[graph_idx, sentence_idx] > 0:
                alignments.append((input_word_pos_map[sentence_idx], target_node_map[graph_idx]))

    # sort alignments by first element
    alignments = sorted(alignments, key=lambda x: x[0])

    # export leamr alignmet from isi
    leamr_subgraph_alignments, leamr_relation_alignments, leamr_reentrancy_alignments = \
        conver_isi_leamr(alignments, sentence_words_map, graph_id_map, graph_nodes_map)




    head_alignment += "\t" + " ".join([f"{str(x[0])}-{x[1]}" for x in alignments])
    return head_alignment, leamr_subgraph_alignments, leamr_relation_alignments, leamr_reentrancy_alignments


def get_son_relations(graph_nodes_map, node):
    son_nodes = []
    for graph_node in graph_nodes_map.keys():
        if graph_node.startswith(node) and (len(graph_node) - len(node) == 4 and graph_node[-1] == "r"):
            son_nodes.append(graph_node)
    return son_nodes

def get_son_nodes(graph_nodes_map, node):
    son_nodes = []
    for graph_node in graph_nodes_map.keys():
        if graph_node.startswith(node) and (len(graph_node) - len(node) == 2 and graph_node[-1] != "r"):
            son_nodes.append(graph_node)
    return son_nodes


# split the nodes in the list if they are not connected
def split_non_related_nodes(nodes, reentrancy_map, graph_tokens_map):
    # sort nodes by length
    nodes_sorted = sorted(nodes, key=lambda x: len(x))

    just_nodes = [node_pos for node_pos in nodes if not node_pos.endswith('.r')]
    just_edges = [edge for edge in nodes if edge.endswith('.r')]

    # sort nodes by length
    just_nodes_sorted = sorted(just_nodes, key=lambda x: len(x))
    just_edges_sorted = sorted(just_edges, key=lambda x: len(x))
    nodes_sorted = just_nodes_sorted + just_edges_sorted

    nodes_splits = [[nodes_sorted[0]]]
    relation_split = []
    is_neg = graph_tokens_map[nodes_sorted[0]] == "-"


    for node in nodes_sorted[1:]:
        is_node = False
        son_nodes = get_son_nodes(graph_tokens_map, node)
        reentrant_son = None

        is_multisentence = graph_tokens_map[node] in ["and", "or", "multi-sentence"]
        for son_node in son_nodes:
            if son_node in reentrancy_map:
                reentrant_son = reentrancy_map[son_node]
                break

        if node[-1] != "r":
            father_node_pos = 2
        else:
            father_node_pos = 4
        
        if not is_multisentence:
            for nodes_split in nodes_splits:
                if node[-1] == "r":
                    if node[:-2] in nodes_split and node[:-4] in nodes_split and not node[:-2] in reentrancy_map:
                        nodes_split.append(node)
                        is_node = True
                        break 

                elif node[:-2] in nodes_split or reentrant_son in nodes_split:
                    nodes_split.append(node)
                    is_node = True
                    break 
                
                for node_split in nodes_split:
                    if is_neg and node[:-2] == node_split[:-2]:
                        nodes_split.append(node)
                        is_node = True
                        break
                    
                if is_node:
                    break
        
        if not is_node and node[-1] == "r":
            relation_split.append(node)
        elif not is_node:
            nodes_splits.append([node])

    if relation_split:
        nodes_splits.append(relation_split)

    return nodes_splits

def extract_isi(node_span_map, span2pos_map):
    isi_alignments = []
    for node_pos,span  in node_span_map.items():
        if span in span2pos_map:
            for pos in span2pos_map[span]:
                isi_alignments.append((int(pos), node_pos))

    # sort alignments by second element
    isi_alignments = sorted(isi_alignments, key=lambda x: x[0])

    isi_alignments = " ".join(f"{str(pos)}-{node_pos}" for pos, node_pos in isi_alignments)

    return isi_alignments


def extract_leamr(node_span_map, span2pos_map, sentence_token_map, graph_id_map, graph_tokens_map, reentrancy_map):
   
    leamr_subgraph_alignments = []
    leamr_relation_alignments = []    
    leamr_reentrancy_alignments = []

    
    # create map relate node position to node type
    span_node_map = {}
    for k,v in node_span_map.items():
        span_node_map.setdefault(v, []).append(k)


    # invert reentrancy_map
    reentrancy_primary = [v for k,v in reentrancy_map.items()]
    reentrancy_primary = list(set(reentrancy_primary))


    for span_pos, nodes_pos in span_node_map.items():
        
        sentence_pos = span2pos_map[span_pos]
        splitted_nodes = split_non_related_nodes(nodes_pos, reentrancy_map, graph_tokens_map)
        subgraph_type =  "subgraph"
        dupl_subgraphs = []
        for nodes_split in splitted_nodes:

            nodes = [node_pos for node_pos in nodes_split if not node_pos.endswith('.r') and not graph_tokens_map[node_pos].startswith("<pointer")]
            edges = [edge for edge in nodes_split if edge.endswith('.r')]
            reentrancy = [node_pos + ".r" for node_pos in nodes_split if not node_pos.endswith('.r') and graph_tokens_map[node_pos].startswith("<pointer")]
            reentrancy_primary_node = [node_pos for node_pos in nodes_split if node_pos.endswith('.r') and node_pos[:-2] in reentrancy_primary]


            is_subgraph = True

            for edge in edges:
                if edge[:-2] not in nodes or edge[:-4] not in nodes:
                    is_subgraph = False
                    break
            
            is_multisentence = graph_tokens_map[nodes[0]] in ["and", "or", "multi-sentence"] if nodes else False

            # generate leamr alignment depending on the case
            if is_subgraph and nodes:
                if graph_tokens_map[nodes[0]] in dupl_subgraphs and not is_multisentence:
                    subgraph_type = "dupl-subgraph"

                leamr_subgraph_alignments.append(generate_leamr_alignment(subgraph_type, sentence_pos, nodes, edges, sentence_token_map, graph_tokens_map, reentrancy_map))
                
                dupl_subgraphs.append(graph_tokens_map[nodes[0]])
                
                
            else:
                # filter out delteded relations
                if nodes:
                    leamr_subgraph_alignments.append(generate_leamr_alignment("subgraph", sentence_pos, nodes, [], sentence_token_map, graph_tokens_map, reentrancy_map))
       
                if edges:
                    leamr_relation_alignments.append(generate_leamr_alignment('relation', sentence_pos, [], edges, sentence_token_map, graph_tokens_map, reentrancy_map))


            if reentrancy:
                for edge in reentrancy:
                    snt_pos = sentence_pos
                    if graph_tokens_map[edge[:-4]] in ["and", "or", "multi-sentence"]:
                        snt_pos = span2pos_map[node_span_map[edge[:-4]]]
                    elif graph_tokens_map[edge[:-2]] in ["and", "or", "multi-sentence"]:
                        snt_pos = span2pos_map[node_span_map[edge[:-2]]]

                    leamr_reentrancy_alignments.append(generate_leamr_alignment('reentrancy:control', snt_pos, [], [edge], sentence_token_map, graph_tokens_map, reentrancy_map))
            
            if reentrancy_primary_node:
                snt_pos = sentence_pos

                for edge in reentrancy_primary_node:
                    leamr_reentrancy_alignments.append(generate_leamr_alignment('reentrancy:primary', snt_pos, [], [edge], sentence_token_map, graph_tokens_map, reentrancy_map))

            
    # sort leamr_subgraph_alignments by first element
    leamr_subgraph_alignments = sorted(leamr_subgraph_alignments, key=lambda x: x["tokens"][0])
    leamr_relation_alignments = sorted(leamr_relation_alignments, key=lambda x: x["tokens"][0])

    return leamr_subgraph_alignments, leamr_relation_alignments, leamr_reentrancy_alignments




def extract_alignment_unsupervised(sentence_tokens, graph_tokens, alignment_score, spans_list):
    # create numpy array from the alignment score
    alignment_score = np.array(alignment_score)

    span2pos_map = {k:v for k,v in enumerate(spans_list)}
    pos2spans_map = {vv:k for k,v in enumerate(spans_list) for vv in v}
    # create a map that aligned tokenized sentence to the original sentence
    input_word_pos_map = {}
    pos = 0
    for word_idx, word in enumerate(sentence_tokens):
        if word != f"{INIT_TOKEN}<s>" and word != f"{INIT_TOKEN}</s>":
            input_word_pos_map[word_idx] = pos

            if word.startswith(INIT_TOKEN) and not (word == f"{INIT_TOKEN}<" and (word_idx + 1) < len(sentence_tokens) and sentence_tokens[word_idx + 1] == "a"):
                pos += 1

    
    target_node_map, reentrancy_map, non_reentrancy_map, named_entities_map = build_graph_maps(graph_tokens)

    reentrancy_map = {k: non_reentrancy_map[v] for k, v in reentrancy_map.items() if v in non_reentrancy_map}

    # remove score from stop words from graph and wikinodes
    stop_words_graph =  [f'{INIT_TOKEN}(', f'{INIT_TOKEN})', '<s>', '</s>', f'{INIT_TOKEN}:wiki', f'{INIT_TOKEN}<lit>', f'{INIT_TOKEN}</lit>', "Ã", "ĩ"]   
    is_lit = False
    is_wiki = False
    is_broken_rel = False

    alignment_score = np.squeeze(alignment_score)

    for graph_token_idx, graph_token in enumerate(graph_tokens):
        if graph_token == f"{INIT_TOKEN}:wiki":
            is_wiki = True
        elif graph_token == f"{INIT_TOKEN}<lit>":
            is_lit = True
        elif graph_token == f"{INIT_TOKEN}</lit>":
            is_wiki = False
            is_lit = False
        elif graph_token == f'{INIT_TOKEN}-' and graph_tokens[graph_token_idx - 1] == f"{INIT_TOKEN}:wiki":
            is_wiki = False
            alignment_score[:, :, graph_token_idx, :] = 0 
        elif graph_token == f'{INIT_TOKEN}:' and not is_lit:
            is_broken_rel = True
        elif is_broken_rel and graph_tokens[graph_token_idx - 1] != f'{INIT_TOKEN}:' and graph_token.startswith(f'{INIT_TOKEN}'):
            is_broken_rel = False

        if graph_token in stop_words_graph or (is_wiki and is_lit) or is_broken_rel:
            alignment_score[:, :, graph_token_idx, :] = 0 


    stop_words_input = ['<s>', '</s>', f'{INIT_TOKEN}<pad>', f'{INIT_TOKEN}-', f'{INIT_TOKEN},', f'{INIT_TOKEN}@', f'{INIT_TOKEN}.', ".", f'{INIT_TOKEN}:']
    for snt_token_idx, snt_token in enumerate(sentence_tokens):
        if snt_token in stop_words_input:
            alignment_score[:, :, :, snt_token_idx] = 0

    # identify compound tokens in the sentence and sum the values
    sentence_tokens_filter = [(token_idx, 1)  for token_idx, token in enumerate(sentence_tokens) if not token.startswith(INIT_TOKEN) and token not in ['<s>', '</s>']]
    sentence_tokens_map = {}
    for token_idx, repeated in sentence_tokens_filter:
        sentence_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in sentence_tokens_map \
                                                        else sentence_tokens_map[token_idx - 1]
    # ccreate 1 array of lenght of sentence with 1s
    length_compound_tokens = np.ones(len(alignment_score[0, 0, 0,:]))


    for split_token_idx, repeated in sentence_tokens_filter:
        alignment_score[:, :, :, sentence_tokens_map[split_token_idx]] += alignment_score[:, :, :, split_token_idx]
        alignment_score[:, :, :, split_token_idx] = 0
        length_compound_tokens[sentence_tokens_map[split_token_idx]] += 1
        length_compound_tokens[split_token_idx] = 1

    # extract sentence word related to word position to token in sentence
    sentence_words_map = {}
    for encoder_pos, sentence_token_pos in input_word_pos_map.items():
        sentence_word = sentence_tokens[encoder_pos].replace(f"{INIT_TOKEN}", "")
        next_token = 1
        while (encoder_pos + next_token) < len(sentence_tokens) and not sentence_tokens[encoder_pos + next_token].startswith(INIT_TOKEN):
            sentence_word += sentence_tokens[encoder_pos + next_token]
            next_token += 1

        sentence_words_map[sentence_token_pos] = sentence_word


    alignment_score = alignment_score[:4].sum(axis=0).sum(axis=0)
    # heads_to_select = [ 3,4, 5,6, 7, 11,  12, 15]

    # alignment_score = alignment_score[3].sum(axis=0) + alignment_score[7].sum(axis=0)

    # create map relate node position to graph token
    graph_id_map = {}
    graph_nodes_map = {}
    span_node_map = {}
    placeholders = ["person", "thing", "government-organization", "country", "city", "state", "relative-position", "before", "after", "nation"]
    role_nodes = []
    revisit_placeholder_nodes = []
    pos2alignment_map = {}

    named_entities_map_filter = [node for _, nodes in named_entities_map.items() for node in nodes]
    for graph_idx, graph_token in target_node_map.items():
        next_token = 1 if graph_tokens[graph_idx].startswith(f"{INIT_TOKEN}<pointer") and not (graph_tokens[graph_idx + 1].startswith(f"{INIT_TOKEN}:") or graph_tokens[graph_idx + 1] == f"{INIT_TOKEN})") else 0

        graph_id = graph_tokens[graph_idx].replace(f"{INIT_TOKEN}", "")
        graph_node = graph_tokens[graph_idx + next_token].replace(f"{INIT_TOKEN}", "")
        
        # copy tensor
        sum_alignments = alignment_score[graph_idx + next_token, :].copy()


        next_token += 1
        is_prep_edge = graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN}prep"

        while (graph_idx + next_token) < len(graph_tokens) \
                and (not graph_tokens[graph_idx + next_token].startswith(INIT_TOKEN) \
                    or (is_prep_edge and (not graph_tokens[graph_idx + next_token].startswith(INIT_TOKEN) \
                        or graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN}prep")
                    or (graph_tokens[graph_idx].startswith(f"{INIT_TOKEN}<pointer") and graph_id != graph_node and not (graph_tokens[graph_idx + next_token].startswith(f"{INIT_TOKEN}:") or graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN})")))):

            graph_node += graph_tokens[graph_idx + next_token].lstrip(f"{INIT_TOKEN}")
            if graph_tokens[graph_idx + next_token] != f"{INIT_TOKEN}-":
                sum_alignments += alignment_score[graph_idx + next_token, :].copy()
            
            next_token += 1


        graph_id_map[graph_token] = graph_id
        graph_nodes_map[graph_token] = graph_node

        # if all the element in tensor are 0
        if np.sum(sum_alignments) != 0:
            pos2alignment_map[graph_token] = sum_alignments


    node_span_map = {}
    node_span_map['1'] = pos2spans_map[input_word_pos_map[np.argmax(alignment_score[2, :])]]
    # calculate alignments for nodes
    for node, alignment_score in pos2alignment_map.items():
        if "role-91" in graph_nodes_map[node]:
            role_nodes.append(node)

        elif graph_nodes_map[node] == "amr-unknown" and f"{INIT_TOKEN}?" in sentence_tokens:
            sentence_tokens_np = np.array(sentence_tokens)
            node_span_map[node] = pos2spans_map[input_word_pos_map[np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}?")[0][0]]]

        elif graph_nodes_map[node] == ":condition" and (f"{INIT_TOKEN}if" in sentence_tokens):
            sentence_tokens_np = np.array(sentence_tokens)
            node_span_map[node] = pos2spans_map[input_word_pos_map[np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}if")[0][0]]]

        elif graph_nodes_map[node] == ":condition" and (f"{INIT_TOKEN}If" in sentence_tokens):
            sentence_tokens_np = np.array(sentence_tokens)
            node_span_map[node] = pos2spans_map[input_word_pos_map[np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}If")[0][0]]]

        elif graph_nodes_map[node] == ":purpose" and (f"{INIT_TOKEN}to" in sentence_tokens):
            sentence_tokens_np = np.array(sentence_tokens)
            alignment_aux = alignment_score[0].copy()*0
            pos_alignment_aux = 0
 
            for poss_align in np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}to")[0]:
                if alignment_score[poss_align] > alignment_aux:
                    alignment_aux = alignment_score[poss_align]
                    pos_alignment_aux = poss_align

            node_span_map[node] = pos2spans_map[input_word_pos_map[pos_alignment_aux]]


        elif node not in named_entities_map_filter:
            span_pos = pos2spans_map[input_word_pos_map[np.argmax(alignment_score)]]

            node_span_map[node] = span_pos
            # span_node_map[span_pos] = graph_token
            if graph_nodes_map[node] in placeholders:
                revisit_placeholder_nodes.append(node)



    named_entities_spans = set()
    # create correct named entity alignment
    for root_node, nodes in named_entities_map.items():
        node_span_map[root_node] = pos2alignment_map[root_node]*0
        for node in nodes:
            if  root_node + "." in node and node[-1] != "r":
                node_span_map[root_node] += pos2alignment_map[node]

        node_span_map[root_node] = pos2spans_map[input_word_pos_map[np.argmax(node_span_map[root_node])]]

        for node in nodes:
            node_span_map[node] = node_span_map[root_node]
        
        named_entities_spans.add(node_span_map[root_node])

    named_entities_spans = list(named_entities_spans)


    # Fix role alignment
    for role_node in role_nodes:

        relation = None
        for aux_relation in get_son_relations(graph_nodes_map, role_node):
            if aux_relation in graph_nodes_map and  graph_nodes_map[aux_relation] == ":ARG1":
                relation = aux_relation

        if relation is not None and relation not in named_entities_map_filter:
            span = pos2spans_map[input_word_pos_map[np.argmax(pos2alignment_map[relation])]]

            pos2spans_map[role_node] = span
            pos2spans_map[relation] = span
            pos2spans_map[relation[:-2]] = span

            if relation[:-4] not in named_entities_map_filter:
                node_span_map[relation[:-4]] = span
                node_span_map[relation[:-2] + ".r"] = span
 
        else:

            i = 1
            span_pos = None
            sum_alignments = pos2alignment_map[role_node]


            while span_pos == None and input_word_pos_map[sum_alignments.argsort()[-i]] in pos2spans_map:
                try:
                    span_pos_aux = pos2spans_map[input_word_pos_map[sum_alignments.argsort()[-i]]]
                except:
                    print(i)
                    print( np.count_nonzero(sum_alignments))
                    print(sum_alignments)
                    print(sentence_tokens)
                    print(sum_alignments.argsort()[-i])
                    print(input_word_pos_map)
                    print(pos2spans_map)
                    span_pos_aux = pos2spans_map[input_word_pos_map[sum_alignments.argsort()[-i]]]

                if span_pos_aux not in named_entities_spans:
                    span_pos = span_pos_aux

                i += 1

            if span_pos != None:
                node_span_map[role_node] = span_pos


    del_nodes = {}
    # Filter out non named nodes from named entities structure
    for node, span in node_span_map.items():
        son_relations = get_son_relations(graph_nodes_map, node)

        if (node not in named_entities_map_filter and span in named_entities_spans and node[-1] != "r") and not (node + ".1" in named_entities_map_filter and graph_nodes_map[node + ".1.r"] ==  ":mod"):
            del_nodes[node] = span

    # TODO: Fix this
    for node, spam in del_nodes.items():
        del node_span_map[node]
        continue


    # Fix placeholders alignment
    for node in revisit_placeholder_nodes:
        son_relations = get_son_relations(graph_nodes_map, node)
        # if father and son nodes are in the same span then align to their span
        if node + ".1" in node_span_map and node[:-2] in node_span_map and  node_span_map[node + ".1"] == node_span_map[node[:-2]]:
            node_span_map[node] = node_span_map[node + ".1"]

        elif node + ".2" in node_span_map and node[:-2] in node_span_map and  node_span_map[node + ".2"] == node_span_map[node[:-2]]:
            node_span_map[node] = node_span_map[node + ".2"]
        
        # Fix before after structure
        elif graph_nodes_map[node] in ["before", "after"] and node + ".1" in graph_nodes_map and graph_nodes_map[node + ".1"] == "now":
            node_span_map[node + ".1"] = node_span_map[node]

        else:
            for relation in son_relations:
                if relation in graph_nodes_map and  (graph_nodes_map[relation] == ":mod" and graph_nodes_map[relation[:-2]] in placeholders):
                    node_span_map[node] = node_span_map[relation[:-2]]
                    break  

    # Fix before now and after now  structure

    # spans_map = {}
    # # iterate over named entity spans
    # for root_node, all_nodes in named_entities_map.items():
    #     for node in all_nodes:
    #         spans_map.setdefault(root_node, []).extend(node_span_map[node])
# 
    # new_span_list = spans_map
    

    for root_node, all_nodes in named_entities_map.items():
        for node in all_nodes:
            node_span_map[node] = node_span_map[root_node]

    # fix ARG relations alignment
    for node, word in graph_nodes_map.items():
        if word.startswith(":ARG") and word.endswith("of") and node[:-2] in node_span_map:
            node_span_map[node] = node_span_map[node[:-2]]
        elif word.startswith(":ARG") and not word.endswith("of") and node[:-4] in node_span_map:
            node_span_map[node] = node_span_map[node[:-4]]

        elif node.endswith("r"):
            if graph_nodes_map[node] in [":mod", ":duration"] and node[:-2] in node_span_map:
                node_span_map[node] = node_span_map[node[:-2]]
            elif graph_nodes_map[node] in [":domain"] and node[:-4] in node_span_map:
                node_span_map[node] =  node_span_map[node[:-4]]
            elif graph_nodes_map[node].startswith(":op") and node[:-4] in node_span_map:
                node_span_map[node] =  node_span_map[node[:-4]]
            elif node[:-2] in node_span_map and node[:-4] in node_span_map and node_span_map[node[:-2]] == node_span_map[node[:-4]]:
                node_span_map[node] = node_span_map[node[:-2]]

   
    subgraphs_alignment, relation_alignment, reentrancy_alignments = extract_leamr(node_span_map, span2pos_map, sentence_words_map, graph_id_map, graph_nodes_map, reentrancy_map)
    isi_alignment = extract_isi(node_span_map, span2pos_map)

    node_word_pos_map = {}
    for node, alignment in pos2alignment_map.items():
        node_word_pos_map[node] = input_word_pos_map[np.argmax(alignment)]

    isi_alignments = []
    for node_pos,span  in node_word_pos_map.items():
        isi_alignments.append((int(span), node_pos))

    # sort alignments by second element
    isi_alignments = sorted(isi_alignments, key=lambda x: x[0])

    isi_alignment = " ".join(f"{str(pos)}-{node_pos}" for pos, node_pos in isi_alignments)


    return isi_alignment.strip(), subgraphs_alignment, relation_alignment, reentrancy_alignments




# method extract the alignment from a json object
def extract_alignment_using_spans(sentence_id, sentence_tokens, graph_tokens, alignment_score, spans_list):
    # create numpy array from the alignment score
    alignment_score = np.array(alignment_score)

    span2pos_map = {k:v for k,v in enumerate(spans_list)}
    pos2spans_map = {vv:k for k,v in enumerate(spans_list) for vv in v}
    
    # create a map that aligned tokenized sentence to the original sentence
    input_word_pos_map = {}
    pos = 0
    for word_idx, word in enumerate(sentence_tokens):

        if word != f"{INIT_TOKEN}<s>" and word != f"{INIT_TOKEN}</s>":
            input_word_pos_map[word_idx] = pos

            if word.startswith(INIT_TOKEN) and not (word == f"{INIT_TOKEN}<" and (word_idx + 1) < len(sentence_tokens) and sentence_tokens[word_idx + 1] == "a"):
                pos += 1

    
    node_span_map = {}


    # remove score from stop words from graph and wikinodes
    stop_words_graph =  [f'{INIT_TOKEN}(', f'{INIT_TOKEN})', '<s>', '</s>', f'{INIT_TOKEN}:wiki', f'{INIT_TOKEN}<lit>', f'{INIT_TOKEN}</lit>', "Ã", "ĩ"]   
    is_lit = False
    is_wiki = False
    is_broken_rel = False
    for graph_token_idx, graph_token in enumerate(graph_tokens):
        if graph_token == f"{INIT_TOKEN}:wiki":
            is_wiki = True
        elif graph_token == f"{INIT_TOKEN}<lit>":
            is_lit = True
        elif graph_token == f"{INIT_TOKEN}</lit>":
            is_wiki = False
            is_lit = False
        elif graph_token == f'{INIT_TOKEN}-' and graph_tokens[graph_token_idx - 1] == f"{INIT_TOKEN}:wiki":
            is_wiki = False
            alignment_score[graph_token_idx, :] = 0 
        elif graph_token == f'{INIT_TOKEN}:' and not is_lit:
            is_broken_rel = True
        elif is_broken_rel and graph_tokens[graph_token_idx - 1] != f'{INIT_TOKEN}:' and graph_token.startswith(f'{INIT_TOKEN}'):
            is_broken_rel = False

        if graph_token in stop_words_graph or (is_wiki and is_lit) or is_broken_rel:
            alignment_score[graph_token_idx, :] = 0 


    stop_words_input = ['<s>', '</s>', f'{INIT_TOKEN}<pad>', f'{INIT_TOKEN}-', f'{INIT_TOKEN},', f'{INIT_TOKEN}@', f'{INIT_TOKEN}.', ".", f'{INIT_TOKEN}:']

    for snt_token_idx, snt_token in enumerate(sentence_tokens):
        if snt_token in stop_words_input:
            alignment_score[:, snt_token_idx] = 0

    # identify compound tokens in the sentence and sum the values
    sentence_tokens_filter = [(token_idx, 1)  for token_idx, token in enumerate(sentence_tokens) if not token.startswith(INIT_TOKEN) and token not in ['<s>', '</s>']]
    sentence_tokens_map = {}
    for token_idx, repeated in sentence_tokens_filter:
        sentence_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in sentence_tokens_map \
                                                        else sentence_tokens_map[token_idx - 1]
    # ccreate 1 array of lenght of sentence with 1s
    length_compound_tokens = np.ones(len(alignment_score[0,:]))


    for split_token_idx, repeated in sentence_tokens_filter:
        alignment_score[:, sentence_tokens_map[split_token_idx]] += alignment_score[:, split_token_idx]
        alignment_score[:, split_token_idx] = 0
        length_compound_tokens[sentence_tokens_map[split_token_idx]] += 1
        length_compound_tokens[split_token_idx] = 1


    node_span_map['1'] = pos2spans_map[input_word_pos_map[np.argmax(alignment_score[2, :])]]

    target_node_map, reentrancy_map, non_reentrancy_map, named_entities_map = build_graph_maps(graph_tokens)

    reentrancy_map = {k: non_reentrancy_map[v] for k, v in reentrancy_map.items() if v in non_reentrancy_map}

    # extract sentence word related to word position to token in sentence
    sentence_words_map = {}
    for encoder_pos, sentence_token_pos in input_word_pos_map.items():
        sentence_word = sentence_tokens[encoder_pos].replace(f"{INIT_TOKEN}", "")
        next_token = 1
        while (encoder_pos + next_token) < len(sentence_tokens) and not sentence_tokens[encoder_pos + next_token].startswith(INIT_TOKEN):
            sentence_word += sentence_tokens[encoder_pos + next_token]
            next_token += 1

        sentence_words_map[sentence_token_pos] = sentence_word

    # create map relate node position to graph token
    graph_id_map = {}
    graph_nodes_map = {}
    span_node_map = {}
    placeholders = ["person", "thing", "government-organization", "country", "city", "state", "relative-position", "before", "after", "nation"]
    role_nodes = []
    revisit_placeholder_nodes = []
    pos2alignment_map = {}

    named_entities_map_filter = [node for _, nodes in named_entities_map.items() for node in nodes]
    for graph_idx, graph_token in target_node_map.items():
        next_token = 1 if graph_tokens[graph_idx].startswith(f"{INIT_TOKEN}<pointer") and not (graph_tokens[graph_idx + 1].startswith(f"{INIT_TOKEN}:") or graph_tokens[graph_idx + 1] == f"{INIT_TOKEN})") else 0

        graph_id = graph_tokens[graph_idx].replace(f"{INIT_TOKEN}", "")
        graph_node = graph_tokens[graph_idx + next_token].replace(f"{INIT_TOKEN}", "")
        
        # copy tensor
        sum_alignments = alignment_score[graph_idx + next_token, :].copy()

        next_token += 1
        is_prep_edge = graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN}prep"

        while (graph_idx + next_token) < len(graph_tokens) \
                and (not graph_tokens[graph_idx + next_token].startswith(INIT_TOKEN) \
                    or (is_prep_edge and (not graph_tokens[graph_idx + next_token].startswith(INIT_TOKEN) \
                        or graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN}prep")
                    or (graph_tokens[graph_idx].startswith(f"{INIT_TOKEN}<pointer") and graph_id != graph_node and not (graph_tokens[graph_idx + next_token].startswith(f"{INIT_TOKEN}:") or graph_tokens[graph_idx + next_token] == f"{INIT_TOKEN})")))):

            graph_node += graph_tokens[graph_idx + next_token].lstrip(f"{INIT_TOKEN}")
            if graph_tokens[graph_idx + next_token] != f"{INIT_TOKEN}-":
                sum_alignments += alignment_score[graph_idx + next_token, :].copy()
            
            next_token += 1


        graph_id_map[graph_token] = graph_id
        graph_nodes_map[graph_token] = graph_node


        # if all the element in tensor are 0
        if np.sum(sum_alignments) != 0:
            pos2alignment_map[graph_token] = sum_alignments

    # calculate alignments for nodes
    for node, alignment_score in pos2alignment_map.items():
        if "role-91" in graph_nodes_map[node]:
            role_nodes.append(node)
            
        elif graph_nodes_map[node] == "amr-unknown" and f"{INIT_TOKEN}?" in sentence_tokens:
            sentence_tokens_np = np.array(sentence_tokens)
            node_span_map[node] = pos2spans_map[input_word_pos_map[np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}?")[0][0]]]

        elif graph_nodes_map[node] == ":condition" and (f"{INIT_TOKEN}if" in sentence_tokens):
            sentence_tokens_np = np.array(sentence_tokens)
            node_span_map[node] = pos2spans_map[input_word_pos_map[np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}if")[0][0]]]

        elif graph_nodes_map[node] == ":condition" and (f"{INIT_TOKEN}If" in sentence_tokens):
            sentence_tokens_np = np.array(sentence_tokens)
            node_span_map[node] = pos2spans_map[input_word_pos_map[np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}If")[0][0]]]
        

        elif graph_nodes_map[node] == ":purpose" and (f"{INIT_TOKEN}to" in sentence_tokens):
            sentence_tokens_np = np.array(sentence_tokens)
            alignment_aux = alignment_score[0].copy()*0
            pos_alignment_aux = 0
 
            for poss_align in np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}to")[0]:
                if alignment_score[poss_align] > alignment_aux:
                    alignment_aux = alignment_score[poss_align]
                    pos_alignment_aux = poss_align

            node_span_map[node] = pos2spans_map[input_word_pos_map[pos_alignment_aux]]

        elif node not in named_entities_map_filter:
            try:
                span_pos = pos2spans_map[input_word_pos_map[np.argmax(alignment_score)]]
            except:
                print(alignment_score.shape)
                print(np.argmax(alignment_score))
                print(input_word_pos_map)
                print(pos2spans_map)
                span_pos = pos2spans_map[input_word_pos_map[np.argmax(alignment_score)]]

            node_span_map[node] = span_pos
            # span_node_map[span_pos] = graph_token
            if graph_nodes_map[node] in placeholders:
                revisit_placeholder_nodes.append(node)

    named_entities_spans = set()
    # create correct named entity alignment
    for root_node, nodes in named_entities_map.items():
        node_span_map[root_node] = pos2alignment_map[root_node]*0
        for node in nodes:
            if  root_node + "." in node and node[-1] != "r":
                node_span_map[root_node] += pos2alignment_map[node]

        node_span_map[root_node] = pos2spans_map[input_word_pos_map[np.argmax(node_span_map[root_node])]]

        for node in nodes:
            node_span_map[node] = node_span_map[root_node]
        
        named_entities_spans.add(node_span_map[root_node])

    named_entities_spans = list(named_entities_spans)


    # Fix role alignment
    for role_node in role_nodes:
        i = 1
        span_pos = None
        sum_alignments = pos2alignment_map[role_node]

        while span_pos == None:
            span_pos_aux = pos2spans_map[input_word_pos_map[sum_alignments.argsort()[-i]]]
            if span_pos_aux not in named_entities_spans:
                span_pos = span_pos_aux

            i += 1
        
        node_span_map[role_node] = span_pos

        
    del_nodes = {}
    # Filter out non named nodes from named entities structure
    for node, span in node_span_map.items():
        son_relations = get_son_relations(graph_nodes_map, node)

        if (node not in named_entities_map_filter and span in named_entities_spans and node[-1] != "r") and not (node + ".1" in named_entities_map_filter and graph_nodes_map[node + ".1.r"] ==  ":mod"):
            del_nodes[node] = span

    # TODO: Fix this
    for node, span in del_nodes.items():
        del node_span_map[node]
        continue

        del node_span_map[node]
        continue
        print(node, span)
        print(sentence_id)
        i = 1
        span_pos = None
        sum_alignments = pos2alignment_map[node]

        while span_pos == None:
            span_pos_aux = pos2spans_map[input_word_pos_map[sum_alignments.argsort()[-i]]]
            if span_pos_aux != span:
                span_pos = span_pos_aux

            i += 1
        
        node_span_map[node] = span_pos




    # Fix placeholders alignment
    for node in revisit_placeholder_nodes:
        son_relations = get_son_relations(graph_nodes_map, node)
        # if father and son nodes are in the same span then align to their span
        if node + ".1" in node_span_map and node[:-2] in node_span_map and  node_span_map[node + ".1"] == node_span_map[node[:-2]]:
            node_span_map[node] = node_span_map[node + ".1"]

        elif node + ".2" in node_span_map and node[:-2] in node_span_map and  node_span_map[node + ".2"] == node_span_map[node[:-2]]:
            node_span_map[node] = node_span_map[node + ".2"]
        
        # Fix before after structure
        elif graph_nodes_map[node] in ["before", "after"] and node + ".1" in graph_nodes_map and graph_nodes_map[node + ".1"] == "now":
            node_span_map[node + ".1"] = node_span_map[node]

        else:
            for relation in son_relations:
                if relation in graph_nodes_map and  (graph_nodes_map[relation] == ":mod" and graph_nodes_map[relation[:-2]] in placeholders):
                    node_span_map[node] = node_span_map[relation[:-2]]
                    break  

    # Fix before now and after now  structure

    # spans_map = {}
    # # iterate over named entity spans
    # for root_node, all_nodes in named_entities_map.items():
    #     for node in all_nodes:
    #         spans_map.setdefault(root_node, []).extend(node_span_map[node])
# 
    # new_span_list = spans_map
    

    for root_node, all_nodes in named_entities_map.items():
        for node in all_nodes:
            node_span_map[node] = node_span_map[root_node]

    # fix ARG relations alignment
    # for node, word in graph_nodes_map.items():
    #     if word.startswith(":ARG") and word.endswith("of") and node[:-2] in node_span_map:
    #         node_span_map[node] = node_span_map[node[:-2]]
    #     elif word.startswith(":ARG") and not word.endswith("of") and node[:-4] in node_span_map:
    #         node_span_map[node] = node_span_map[node[:-4]]
# 
    #     elif node.endswith("r"):
    #         if word in [":mod", ":duration"] and node[:-2] in node_span_map:
    #             node_span_map[node] = node_span_map[node[:-2]]
    #         elif graph_nodes_map[node] in [":domain"] and node[:-4] in node_span_map:
    #             node_span_map[node] =  node_span_map[node[:-4]]
    #         elif graph_nodes_map[node].startswith(":op") and node[:-4] in node_span_map:
    #             node_span_map[node] =  node_span_map[node[:-4]]
    #         elif node[:-2] in node_span_map and node[:-4] in node_span_map and node_span_map[node[:-2]] == node_span_map[node[:-4]]:
    #             node_span_map[node] = node_span_map[node[:-2]]

    for node, word in graph_nodes_map.items():
        if node.endswith("r"):
            if node[:-2] in node_span_map and node[:-4] in node_span_map and node_span_map[node[:-2]] == node_span_map[node[:-4]]:
                node_span_map[node] = node_span_map[node[:-2]]
            
            elif any(word.startswith(prefix) for prefix in [':ARG', ':op', ':snt']):
                if word.endswith("of") and node[:-2] in node_span_map:
                    node_span_map[node] = node_span_map[node[:-2]]
                elif node[:-4] in node_span_map:
                    node_span_map[node] = node_span_map[node[:-4]]
            
            elif word in [":domain"] and node[:-4] in node_span_map:
                node_span_map[node] =  node_span_map[node[:-4]]
            
            elif word in [":mod", ":duration", ':name',':polarity',':li'] and node[:-2] in node_span_map:
                node_span_map[node] = node_span_map[node[:-2]]

            elif word in [":pre-", ":conj-", ':poss', ':part']:
                if graph_nodes_map[node] == ":pre-":
                    token_label = graph_nodes_map[node].replace(':prep-', '').split('-')[-1]
                elif graph_nodes_map[node] == ":conj-":
                    token_label = graph_nodes_map[node].replace(':conj-', '').split('-')[-1]
                elif graph_nodes_map[node] in [":poss", ":part"]:
                    token_label = "'s"

                if f"{INIT_TOKEN}{token_label}" in sentence_tokens:
                    sentence_tokens_np = np.array(sentence_tokens)

                    alignment_aux = pos2alignment_map[node][0].copy()*0
                    pos_alignment_aux = 0

                    for snt_tok in token_label:
                        print( np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}{snt_tok}"))
                        print(f"{INIT_TOKEN}{snt_tok}")
                        print(sentence_tokens_np)
                        for poss_align in np.argwhere(sentence_tokens_np == f"{INIT_TOKEN}{snt_tok}")[0]:
                            if alignment_score[poss_align] > alignment_aux:
                                alignment_aux = alignment_score[poss_align]
                                pos_alignment_aux = poss_align

                    node_span_map[node] = pos2spans_map[input_word_pos_map[pos_alignment_aux]]

                
    subgraphs_alignment, relation_alignment, reentrancy_alignments = extract_leamr(node_span_map, span2pos_map, sentence_words_map, graph_id_map, graph_nodes_map, reentrancy_map)
    isi_alignment = extract_isi(node_span_map, span2pos_map)

    node_word_pos_map = {}
    for node, alignment in pos2alignment_map.items():
        node_word_pos_map[node] = input_word_pos_map[np.argmax(alignment)]

    isi_alignments = []
    for node_pos,span  in node_word_pos_map.items():
        isi_alignments.append((int(span), node_pos))

    # sort alignments by second element
    isi_alignments = sorted(isi_alignments, key=lambda x: x[0])

    isi_alignment = " ".join(f"{str(pos)}-{node_pos}" for pos, node_pos in isi_alignments)

    return isi_alignment, subgraphs_alignment, relation_alignment, reentrancy_alignments

if __name__ == '__main__':
    path = "/home/martinez/project/alignment-aware-mspring/alignments/"
    name = "attention"
    dataset = "leamr"
    model = "loss"
    path_file = f"{path}amr-release-3.0-alignments-test-leamr.txt_leamr_{name}-2.jsonl"
    path_file = f"{path}amr-release-3.0-alignments-test-leamr.txt_leamr_attention-0.8404.jsonl"
    # path_file = f"{path}amr-release-3.0-alignments-test-leamr.txt_leamr_gradients-0.8394.jsonl"
    path_file = f"{path}amr-release-3.0-alignments-test-leamr.txt_leamr_attention-12_0.8417.jsonl"
    path_file = f"{path}amr-release-3.0-alignments-test-leamr.txt_leamr_attention-0.8404.jsonl"
    path_file = f"{path}all_test.txt_unsupervised_attention-0.8400.jsonl"
    # path_file = f"{path}all_test.txt_unsupervised_gradient-0.8400.jsonl"
    # path_file = f"{path}{dataset}_test.txt_{model}_{name}-0.8400.jsonl"
    # path_file = f"{path}all_test.txt_leamr_attention-0.8363.jsonl"
    path_file = f"{path}{dataset}_test.txt_{model}_{name}_4layers-0.8386.jsonl"
    path_file = f"{path}amr-release-3.0-alignments-test-leamr.txt_leamr_attention-12_0.8417.jsonl"
    path_file = f"{path}{dataset}_test.txt_{model}_{name}-0.8587.jsonl"
    path_file = f"{path}{dataset}_test.txt_{model}_{name}-0.9269.jsonl"

    subgraph_gold_file = "/home/martinez/project/leamr/data-release/alignments/leamr_test.subgraph_alignments.gold.json"

    version = path_file.split("-")[-1].replace(".jsonl", "")
    alignments_isi = {}
    alignments_leamr_subgraph = {}
    alignments_leamr_relations = {}
    alignments_leamr_reentrancy = {}

    if dataset == "concensus":
        spans_file = "../data/alignment/amr-release-3.0-alignments-consensus-sentences.spans.json"
    else:
        spans_file = "../data/alignment/all_test.spans.json"
    
    # def read json file
    with open(spans_file, "r") as f:
        spans = json.load(f)

    with open(subgraph_gold_file, "r") as f:
        subgraph_gold = json.load(f)
 
    # read json file
    data = read_json(path_file)


    # extract alignment
    for sentence in tqdm(data):
        spans_sentence = [subgraph["tokens"] for subgraph in subgraph_gold[sentence["id"] ]]
        
        for span in spans[sentence["id"]]:
            for word in span:
# # # 
                is_span = False
                for subgraph in spans_sentence:
                    if word in subgraph:
                        is_span = True
                        break
                
                if not is_span:
                    spans_sentence.append([word])
# # # 
        # sort spans by first element
        spans_sentence = sorted(spans_sentence, key=lambda x: x[0])

        # alignment_isi ,alignment_leamr_subgraph, alignment_leamr_relations, alignment_leamr_reentrancy = \
        #    extract_alignment_unsupervised(
        #        sentence["encoder_tokens"],  
        #        sentence["tokens"], 
        #        sentence["alignments"],
        #        spans[sentence["id"]]
        #        # spans_sentence
        #    )    
# # 

        alignment_isi, alignment_leamr_subgraph, alignment_leamr_relations, alignment_leamr_reentrancy = \
            extract_alignment_using_spans(sentence["id"],
                sentence["encoder_tokens"],  
                sentence["tokens"], 
                sentence["alignments"],
                spans[sentence["id"]]
                # spans_sentence
            )
              
        alignments_isi[sentence["id"]] = alignment_isi
        alignments_leamr_subgraph[sentence["id"]] = alignment_leamr_subgraph
        alignments_leamr_relations[sentence["id"]] = alignment_leamr_relations
        alignments_leamr_reentrancy[sentence["id"]] = alignment_leamr_reentrancy
        
    # path_file = f"{path}amr-release-3.0-alignments-test-consensus.txt_leamr_{name}.jsonl"

    # # read json file
    # data = read_json(path_file)
    # 
    # # extract alignment
    # for sentence in data:
    #     alignment_isi, alignment_leamr_subgraph, alignment_leamr_relations, alignment_leamr_reentrancy = \
    #         extract_alignment(sentence["id"],
    #             sentence["encoder_tokens"],  
    #             sentence["tokens"], 
    #             sentence["alignments"]
    #         )  
# 
    #     alignments_isi.append(alignment_isi)
    #     alignments_leamr_subgraph[sentence["id"]] = alignment_leamr_subgraph
    #     alignments_leamr_relations[sentence["id"]] = alignment_leamr_relations
    #     alignments_leamr_reentrancy[sentence["id"]] = alignment_leamr_reentrancy

    # write to file
    print(f"{path}{dataset}.{name}_{model}_isi_alignment-{version}.tsv", "w")
    with open(f"{path}{dataset}.{name}_{model}_isi_alignment-{version}.tsv", "w") as f:
        for id, alignment in alignments_isi.items():
            f.write(id + "\t" + alignment + "\n")

    subgraph_prediction_file = f"{path}{dataset}.{name}_{model}_subgraph_alignment-{version}.json"
    relation_prediction_file = f"{path}{dataset}.{name}_{model}_relations_alignment-{version}.json"
    reentrancy_prediction_file = f"{path}{dataset}.{name}_{model}_reentrancy_alignment-{version}.json"

    print(subgraph_prediction_file)
    # write leamr alignment to json file
    with open(subgraph_prediction_file, "w") as f:
        json.dump(alignments_leamr_subgraph, f)
    
    with open(relation_prediction_file, "w") as f:
        json.dump(alignments_leamr_relations, f)
    
    with open(reentrancy_prediction_file, "w") as f:
        json.dump(alignments_leamr_reentrancy, f)

    
    subgraph_prediction_file = "/home/martinez/project/leamr/test/supervised-8567-gold/leamr_test.subgraph_alignments.json"
    subgraph_gold_file = "/home/martinez/project/leamr/gold/alignments-unanonymize/leamr_test.subgraph_alignments.gold.json"
    message = compare_alignment_with_gold(subgraph_prediction_file, subgraph_gold_file)

    subgraph_prediction_file = "/home/martinez/project/leamr/test/gold/leamr_test.subgraph_alignments.json"
    message2 = compare_alignment_with_gold(subgraph_prediction_file, subgraph_gold_file)

    with open("../data/alignment/supervised-errors.txt", "w") as f:
        f.write(message)

    with open("../data/alignment/leamr-errors.txt", "w") as f:
        f.write(message2)



        