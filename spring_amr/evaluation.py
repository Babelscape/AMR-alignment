import datetime
from pathlib import Path

import penman
from sacrebleu import corpus_bleu
import torch
from tqdm import tqdm
import smatch
from spring_amr.utils import *
from spring_amr.dataset import reverse_direction
import json

def extract_amr_alignment(loader, model, tokenizer, output_path):
    
    beam_size=1
    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = True

    padding_token_id = 1

    graphs = []
    isi_alignments = []
    leamr_graph_alignments = []
    leamr_relation_alignments = []
    leamr_reentrancy_alignments = []

    with tqdm(total=total) as bar:
        for x, y, extra in loader:

            # calculate logits
            with torch.no_grad():

                pred_graphs = extra['graphs']
                output = model(**x, **y)
                print(x)
                print(tokenizer.convert_ids_to_tokens(np.array(model.generate(**x).cpu())[0]))
                exit()

                cross_attn = output['cross_attentions'] 
                input_ids = x["input_ids"]
                decoder_input_ids = y["decoder_input_ids"]
                attention_mask = x["attention_mask"]

                inputs_tokens = [tokenizer.convert_ids_to_tokens(snt) for snt in np.array(input_ids.cpu())]
                decoder_inputs_tokens = [tokenizer.convert_ids_to_tokens(graph) for graph in np.array(decoder_input_ids.cpu())]
                cross_attns = np.array(permute_cross_attn_forward(cross_attn, "bart-base").cpu())   

                for graph, input_tokens, decoder_input_tokens, cross_attn, spans in zip(pred_graphs, inputs_tokens, decoder_inputs_tokens, cross_attns, extra["sentences"]):
                    
                    spans_list = [[idx] for idx, span in enumerate(spans.split(" "))]
                
                    alignments_isi ,alignments_leamr_subgraph, alignments_leamr_relations, alignments_leamr_reentrancy = \
                        extract_alignment_using_spans(
                            input_tokens,  
                            decoder_input_tokens, 
                            cross_attn,
                            spans_list
                        )

                    graph.metadata["alignment"] = alignments_isi
                    graphs.append(graph)
                    isi_alignments.append(alignments_isi)
                    leamr_graph_alignments.append(alignments_leamr_subgraph)
                    leamr_relation_alignments.append(alignments_leamr_relations)
                    leamr_reentrancy_alignments.append(alignments_leamr_reentrancy)

    print(f"Writting ISI alignment in file: {output_path}/isi_alignment.tsv")
    with open(f"{output_path}/isi_alignment.tsv", "w") as f:
        for graph, alignment in zip(graphs, isi_alignments):
            f.write(graph.metadata["id"] + "\t" + alignment + "\n")

    print(f"Writting LEAMR subgraph alignment in file: {output_path}/leamr_subgraph_alignment.jsonl")
    # write leamr alignment to json file
    with open(f"{output_path}/leamr_subgraph_alignment.tsv", "w") as f:
        json.dump(alignments_leamr_subgraph, f)
    
    print(f"Writting LEAMR relations alignment in file: {output_path}/leamr_relation_alignment.jsonl")

    with open(f"{output_path}/leamr_relation_alignment.tsv", "w") as f:
        json.dump(alignments_leamr_relations, f)

    print(f"Writting LEAMR reentracy alignment in file: {output_path}/leamr_reentrancy_alignment.jsonl")
    with open(f"{output_path}/leamr_reentrancy_alignment.tsv", "w") as f:
        json.dump(alignments_leamr_reentrancy, f)

    return graphs


def predict_amrs(
        loader, model, tokenizer, beam_size=1, tokens=None, restore_name_ops=False, return_all=False):

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = True

    if tokens is None:
        ids = []
        tokens = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ii = extra['ids']
                ids.extend(ii)
                with torch.no_grad():
                    out = model.generate(
                        **x,
                        max_length=1024,
                        decoder_start_token_id=0,
                        num_beams=beam_size,
                        num_return_sequences=beam_size)
                nseq = len(ii)
                for i1 in range(0, out.size(0), beam_size):
                    tokens_same_source = []
                    tokens.append(tokens_same_source)
                    for i2 in range(i1, i1+beam_size):
                        tokk = out[i2].tolist()
                        tokens_same_source.append(tokk)
                bar.update(nseq)
        # reorder
        tokens = [tokens[i] for i in ids]
        tokens = [t for tt in tokens for t in tt]

    graphs = []
    for i1 in range(0, len(tokens), beam_size):
        graphs_same_source = []
        graphs.append(graphs_same_source)
        for i2 in range(i1, i1+beam_size):
            tokk = tokens[i2]
            graph, status, (lin, backr) = tokenizer.decode_amr(tokk, restore_name_ops=restore_name_ops)
            graph.status = status
            graph.nodes = lin
            graph.backreferences = backr
            graph.tokens = tokk
            graphs_same_source.append(graph)
        graphs_same_source[:] = tuple(zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0]))))[1]

    for gps, gg in zip(graphs, loader.dataset.graphs):
        for gp in gps:
            metadata = gg.metadata.copy()
            metadata['annotator'] = 'bart-amr'
            metadata['date'] = str(datetime.datetime.now())
            if 'save-date' in metadata:
                del metadata['save-date']
            gp.metadata = metadata

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    if not return_all:
        graphs = [gg[0] for gg in graphs]

    return graphs

def predict_sentences(loader, model, tokenizer, beam_size=1, tokens=None, return_all=False):

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = False
    
    if tokens is None:
        ids = []
        tokens = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ids.extend(extra['ids'])
                x, y = reverse_direction(x, y)
                x['input_ids'] = x['input_ids'][:, :1024]
                x['attention_mask'] = x['attention_mask'][:, :1024]
                with torch.no_grad():
                    out = model.generate(
                        **x,
                        max_length=350,
                        decoder_start_token_id=0,
                        num_beams=beam_size,
                        num_return_sequences=beam_size)
                for i1 in range(0, len(out), beam_size):
                    tokens_same_source = []
                    tokens.append(tokens_same_source)
                    for i2 in range(i1, i1+beam_size):
                        tokk = out[i2]
                        tokk = [t for t in tokk.tolist() if t > 2]
                        tokens_same_source.append(tokk)
                bar.update(out.size(0) // beam_size)
        #reorder
        tokens = [tokens[i] for i in ids]

    sentences = []
    for tokens_same_source in tokens:
        if return_all:
            sentences.append([tokenizer.decode(tokk).strip() for tokk in tokens_same_source])
        else:
            sentences.append(tokenizer.decode(tokens_same_source[0]).strip())

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    return sentences

def write_predictions(predictions_path, tokenizer, graphs):
    pieces = [penman.encode(g) for g in graphs]
    Path(predictions_path).write_text('\n\n'.join(pieces).replace(tokenizer.INIT, ''))
    return predictions_path

def compute_smatch(test_path, predictions_path):
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return score[2]

def compute_bleu(gold_sentences, pred_sentences):
    return corpus_bleu(pred_sentences, [gold_sentences])
