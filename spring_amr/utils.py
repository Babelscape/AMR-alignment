from glob import glob
from pathlib import Path

import torch
from transformers import AutoConfig

from spring_amr.dataset import AMRDataset, AMRAlignmentDataset, AMRDatasetTokenBatcherAndLoader
from spring_amr.modeling_bart import AMRBartForConditionalGeneration, AMRAlignmentBartForConditionalGeneration
from spring_amr.tokenization_bart import AMRBartTokenizer, PENMANBartTokenizer, OLDPENMANBartTokenizer, OLDAMRBartTokenizer
from spring_amr.tokenization_mbart import AMRMBartTokenizer, PENMANMBartTokenizer
from spring_amr.modeling_mbart import AMRMBartForConditionalGeneration

import numpy as np

INIT_TOKEN = 'Ġ'

def permute_cross_attn_forward(cross_attn, model_name):
    # transform list of tensors (cross_attn) to tensor
    cross_tensor = torch.stack(cross_attn, dim=0)
    if "mbart" in model_name:
        return cross_tensor.permute(1, 0, 2, 3, 4)
    elif "bart-base" in model_name or "bart-large" in model_name:
        return cross_tensor.permute(1, 0, 2, 3, 4)
    else:
        return cross_tensor.permute(0, 1, 2, 3, 4)




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
    span2pos_map = {k:v for k,v in enumerate(spans_list)}
    pos2spans_map = {vv:k for k,v in enumerate(spans_list) for vv in v}

    # create a map that aligned tokenized sentence to the original sentence
    input_word_pos_map = {}
    pos = 0

    for word_idx, word in enumerate(sentence_tokens):
        if word.lstrip(INIT_TOKEN)  != "<s>" and word != "</s>":
            input_word_pos_map[word_idx] = pos

            if word.startswith(INIT_TOKEN) and not (word.lstrip(INIT_TOKEN)  == "<" and (word_idx + 1) < len(sentence_tokens) and sentence_tokens[word_idx + 1] == "a"):
                pos += 1

    target_node_map, reentrancy_map, non_reentrancy_map, named_entities_map = build_graph_maps(graph_tokens)

    reentrancy_map = {k: non_reentrancy_map[v] for k, v in reentrancy_map.items() if v in non_reentrancy_map}

    # remove score from stop words from graph and wikinodes
    stop_words_graph =  ['(', ')', '<s>', '</s>', ':wiki', '<lit>', '</lit>', "Ã", "ĩ"]   
    is_lit = False
    is_wiki = False
    is_broken_rel = False

    alignment_score = np.squeeze(alignment_score)


    for graph_token_idx, graph_token in enumerate(graph_tokens):
        lstrip_graph_token = graph_token.lstrip(INIT_TOKEN) 
        if lstrip_graph_token == ":wiki":
            is_wiki = True
        elif lstrip_graph_token == "<lit>":
            is_lit = True
        elif lstrip_graph_token == "</lit>":
            is_wiki = False
            is_lit = False
        elif lstrip_graph_token == '-' and graph_tokens[graph_token_idx - 1].lstrip(INIT_TOKEN) == ":wiki":
            is_wiki = False
            alignment_score[:, :, graph_token_idx, :] = 0 
        elif lstrip_graph_token == ':' and not is_lit:
            is_broken_rel = True
        elif is_broken_rel and graph_tokens[graph_token_idx - 1] != ':' and lstrip_graph_token.startswith(''):
            is_broken_rel = False

        if lstrip_graph_token in stop_words_graph or (is_wiki and is_lit) or is_broken_rel:
            alignment_score[:, :, graph_token_idx, :] = 0 


    stop_words_input = ['<s>', '</s>', '<pad>', '-', ',', '@', '.', ".", ':', "is", "are"]
    for snt_token_idx, snt_token in enumerate(sentence_tokens):
        if snt_token.lstrip(f"{INIT_TOKEN}") in stop_words_input:
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
def extract_alignment_using_spans(sentence_tokens, graph_tokens, alignment_score, spans_list):
    sentence_tokens, graph_tokens, alignment_score, spans_list = sentence_tokens, graph_tokens, alignment_score, spans_list

    alignment_score = alignment_score[:4].sum(axis=0).sum(axis=0)

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
    stop_words_graph =  [f'{INIT_TOKEN}(', f'{INIT_TOKEN})', f'{INIT_TOKEN}<s>', f'{INIT_TOKEN}</s>', '<s>', '</s>', f'{INIT_TOKEN}:wiki', f'{INIT_TOKEN}<lit>', f'{INIT_TOKEN}</lit>', "Ã", "ĩ"]   
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


    stop_words_input = ['<s>', '</s>', f'{INIT_TOKEN}<s>', f'{INIT_TOKEN}</s>', f'{INIT_TOKEN}<pad>', f'{INIT_TOKEN}-', f'{INIT_TOKEN},', f'{INIT_TOKEN}@', f'{INIT_TOKEN}.', ".", f'{INIT_TOKEN}:']

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



def instantiate_model_and_tokenizer(
        name='facebook/bart-large',
        checkpoint=None,
        additional_tokens_smart_init=True,
        dropout = 0.15,
        attention_dropout = 0.15,
        from_pretrained = True,
        init_reverse = False,
        collapse_name_ops = False,
        penman_linearization = False,
        use_pointer_tokens = False,
        raw_graph = False,
        mode = "old",
        language = "en_XX",
        direction = "amr",
):
    if raw_graph:
        assert penman_linearization

    skip_relations = False

    tokenizer_name = name

    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout

    tokenizer_type = None
    model_type = None


 
    if penman_linearization and language == "en_XX" and mode == "old":
        tokenizer_type = OLDPENMANBartTokenizer
        model_type = AMRBartForConditionalGeneration

    elif penman_linearization and language == "en_XX":
        tokenizer_type = PENMANBartTokenizer
        model_type = AMRAlignmentBartForConditionalGeneration

    elif penman_linearization:
        tokenizer_type = PENMANMBartTokenizer
        model_type = AMRMBartForConditionalGeneration
    elif language == "en_XX" and mode == "old":
        tokenizer_type = OLDAMRBartTokenizer
        model_type = AMRBartForConditionalGeneration

    elif language == "en_XX":
        tokenizer_type = AMRBartTokenizer
        model_type = AMRAlignmentBartForConditionalGeneration

    else:
        tokenizer_type = AMRMBartTokenizer
        model_type = AMRMBartForConditionalGeneration
    
    

    src_lang=language
    tgt_lang="en_XX"

    if penman_linearization:
        tokenizer = tokenizer_type.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            raw_graph=raw_graph,
            config=config,
            direction=direction,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            add_prefix_space=True,
        )
    else:
        tokenizer = tokenizer_type.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            config=config,
            direction=direction,
            src_lang=src_lang, 
            tgt_lang=tgt_lang,
            add_prefix_space=True,
        )

    model = model_type.from_pretrained(name, config=config) if from_pretrained else model_type(config)
    model.resize_token_embeddings(len(tokenizer))  

    if mode == "old":

        if additional_tokens_smart_init:
            modified = 0
            for tok, idx in tokenizer.encoder.items():
                tok = tok.lstrip(tokenizer.INIT)

                if idx < tokenizer.old_enc_size:
                    continue

                elif tok.startswith('<pointer:') and tok.endswith('>'):
                    tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

                elif tok.startswith('<'):
                    continue

                elif tok.startswith(':'):

                    if skip_relations:
                        continue

                    elif tok.startswith(':op'):
                        tok_split = ['relation', 'operator', str(int(tok[3:]))]

                    elif tok.startswith(':snt'):
                        tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                    elif tok.startswith(':ARG'):
                        tok_split = ['relation', 'argument', str(int(tok[4:]))]

                    else:
                        tok_split = ['relation'] + tok.lstrip(':').split('-')

                else:
                    tok_split = tok.split('-')

                tok_split_ = tok_split
                tok_split = []
                for s in tok_split_:
                    s_ = s + tokenizer.INIT
                    if s_ in tokenizer.encoder:
                        tok_split.append(s_)
                    else:
                        tok_split.extend(tokenizer._tok_bpe(s))

                vecs = []
                for s in tok_split:
                    idx_split = tokenizer.encoder.get(s, -1)
                    if idx_split > -1:
                        vec_split = model.model.shared.weight.data[idx_split].clone()
                        vecs.append(vec_split)

                if vecs:
                    vec = torch.stack(vecs, 0).mean(0)
                    noise = torch.empty_like(vec)
                    noise.uniform_(-0.1, +0.1)
                    model.model.shared.weight.data[idx] = vec + noise
                    modified += 1

        if init_reverse:
            model.init_reverse_model()

        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

    else: 


        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
        else:
            if additional_tokens_smart_init:
                modified = 0
                for tok in tokenizer.added_tokens_list:
                    idx = tokenizer.convert_tokens_to_ids(tok)

                    tok = tok.lstrip(tokenizer.INIT)

                    if idx < tokenizer.vocab_size:
                        continue

                    elif tok.startswith('<pointer:') and tok.endswith('>'):
                        tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

                    elif tok.startswith('<'):
                        continue

                    elif tok.startswith(':'):

                        if skip_relations:
                            continue

                        elif tok.startswith(':op'):
                            tok_split = ['relation', 'operator', str(int(tok[3:]))]

                        elif tok.startswith(':snt'):
                            tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                        elif tok.startswith(':ARG'):
                            tok_split = ['relation', 'argument', str(int(tok[4:]))]

                        elif mode=="amr":
                            tok_split = ['relation'] + tok.lstrip(':').split('-')
                        
                        else:
                            tok_split = ['relation'] + tok.lstrip(':').split('_')

                    else:
                        tok_split = tok.split('-')

                    tok_split_ = tok_split
                    tok_split = []
                    for s in tok_split_:
                        s_ = s + tokenizer.INIT
                        if (tokenizer.unk_token != s_ and tokenizer.convert_tokens_to_ids(s_) != tokenizer.unk_token_id):
                            tok_split.append(s_)
                        else:
                            tok_split.extend(tokenizer._tok_bpe(s))

                    vecs = []
                    for s in tok_split:
                        idx_split = tokenizer.convert_tokens_to_ids(s)
                        if idx_split != tokenizer.unk_token_id:
                            vec_split = model.model.shared.weight.data[idx_split].clone()
                            vecs.append(vec_split)

                    if vecs:
                        vec = torch.stack(vecs, 0).mean(0)
                        noise = torch.empty_like(vec)
                        noise.uniform_(-0.1, +0.1)
                        model.model.shared.weight.data[idx] = vec + noise
                        modified += 1

                if mode == "bmr":
                    bn_lemmas_map = {}
                    with open(f"./data/lemmas/lemmas_{language[:2].upper()}.tsv", "r") as f0:
                        for line in f0:
                            bn_lemmas_map["_" + line.strip().split("\t")[0][3:]] = line.strip().split("\t")[1][1:-1].split(", ")[:1]

                    for bn, lemmas in bn_lemmas_map.items():
                        idx = tokenizer.convert_tokens_to_ids(bn)
                        if idx != tokenizer.unk_token_id:
                            tok = tok.lstrip(tokenizer.INIT)

                            tok_split_ = lemmas
                            vecs = []
                            for s in tok_split_:
                                s_ = tokenizer.convert_tokens_to_ids(tokenizer.INIT + s )
                                if s_ != tokenizer.unk_token_id:
                                    vec_split = model.model.shared.weight.data[s_].clone()
                                    vecs.append(vec_split)
                                else:
                                    word_tokens = []

                                    vec_word = []
                                    for word in s.split("_"):
                                        word_ = tokenizer.convert_tokens_to_ids(tokenizer.INIT + s) 

                                        if word_ != tokenizer.unk_token_id:
                                            vec_split = model.model.shared.weight.data[word_].clone()
                                            vec_word.append(vec_split)
                                        else:
                                            vec_word_tok = []
                                            for word_tok in tokenizer._tok_bpe(word):
                                                word_tok_ = tokenizer.convert_tokens_to_ids(word_tok) 
                                                if word_tok_ != tokenizer.unk_token_id:
                                                    vec_split = model.model.shared.weight.data[word_tok_].clone()
                                                    vec_word_tok.append(vec_split)               
                                            
                                            vec_word.append(torch.stack(vec_word_tok, 0).mean(0))

                                    vecs.append(torch.stack(vec_word, 0).mean(0))

                            if vecs:
                                vec = torch.stack(vecs, 0).mean(0)
                                noise = torch.empty_like(vec)
                                noise.uniform_(-0.1, +0.1)
                                model.model.shared.weight.data[idx] = vec + noise
                                modified += 1

                    del bn_lemmas_map


            model.model.set_input_embeddings(model.model.shared)
            if init_reverse:
                model.init_reverse_model()

    return model, tokenizer



def instantiate_loader(
        glob_pattn,
        tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
):
    paths = []
    if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
        glob_pattn = [glob_pattn]
    for gpattn in glob_pattn:
        paths += [Path(p) for p in glob(gpattn)]
    if evaluation:
        assert out is not None
        Path(out).write_text(
            '\n\n'.join([p.read_text() for p in paths]))
    dataset = AMRDataset(
        paths,
        tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader



def instantiate_aligner_loader(
        glob_pattn,
        tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
):
    paths = []
    if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
        glob_pattn = [glob_pattn]
    for gpattn in glob_pattn:
        paths += [Path(p) for p in glob(gpattn)]
    if evaluation:
        assert out is not None
        Path(out).write_text(
            '\n\n'.join([p.read_text() for p in paths]))
    dataset = AMRAlignmentDataset(
        paths,
        tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader