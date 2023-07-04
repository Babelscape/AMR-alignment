import logging
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data, read_amr_data

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'labels': lm_labels}
    return x, y

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        raw_data=True
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        if raw_data:
            graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        else:
            graphs = read_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.ids = []

        for g in graphs:
            l, e = self.tokenizer.linearize(g)
        
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)
            self.ids.append(g.metadata['id'])

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        sample['graphs'] = self.graphs[idx]
        sample['sentences_id'] = self.ids[idx]

        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['sentences'])
               

    def collate_fn(self, samples, device=torch.device('cpu')):
        sentences = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(sentences, device=device)

        graphs = [s['graphs'] for s in samples]
        y, extra_y = self.tokenizer.batch_encode_graphs(graphs, device=device)
        extra.update(extra_y)

        xx = {k: v.tolist() for k, v in x.items()}
        yy = {k: v.tolist() for k, v in y.items()}


        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None

        extra['ids'] = [s['id'] for s in samples]

        extra['sentences_id'] = [s['sentences_id'] for s in samples]

        return x, y, extra



class AMRAlignmentDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        raw_data=True
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        if raw_data:
            graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        else:
            graphs = read_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.alignments = []
        self.ids = []

        for g in graphs:
            l, e = self.tokenizer.linearize(g)
        
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['tok']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['tok'])
            self.alignments.append(g.metadata['alignments'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)
            self.ids.append(g.metadata['id'])

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        sample['graphs'] = self.graphs[idx]
        sample['sentences_id'] = self.ids[idx]

        sample['alignments'] = self.alignments[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['sentences'])
    
    def _add_alingment(self, input_ids, decoder_input_ids, alignments):
        # create batch alignment tensor, size = len(enumerate(sentences)) x len(input_ids) x len(decoder_input_ids)
        batch_alignment = torch.zeros(len(input_ids), len(input_ids[0]), len(decoder_input_ids[0]))

        for sentence_idx, sentence in enumerate(input_ids):
            graph_tokens = self.tokenizer.convert_ids_to_tokens(decoder_input_ids[sentence_idx])
            alignment_map = {}
            for alignment in alignments[sentence_idx].split():
                 alignment_map.setdefault(int(alignment.split("-")[0]), []).append(alignment.split("-")[1])

            node_pos = ""
            current_node_pos = 1
            node_pos_stack = []
            
            target_node_map = {}

            node_pos = str(current_node_pos)

            node_pos_stack.append(current_node_pos)
            current_node_pos = 1
            for token_idx, token in enumerate(graph_tokens):
                if token.startswith("Ġ:") and token != "Ġ:":
                    next_token_idx = token_idx + 1

                    while not graph_tokens[next_token_idx].startswith("Ġ") or graph_tokens[next_token_idx].startswith("Ġop")  or  graph_tokens[next_token_idx].startswith("Ġsnt")   or (graph_tokens[next_token_idx].startswith("Ġs") and graph_tokens[next_token_idx + 1].startswith("n")) or graph_tokens[next_token_idx].startswith("Ġprep"):
                        next_token_idx += 1

                    if not graph_tokens[next_token_idx].startswith("Ġ(") and not graph_tokens[next_token_idx].startswith("Ġ<lit>"):
                        node_pos += "." + str(current_node_pos)
                        target_node_map.setdefault(node_pos, []).append(next_token_idx)
                        target_node_map.setdefault(node_pos + ".r", []).append(token_idx)
                        current_node_pos += 1
                        node_pos = ".".join(node_pos.split(".")[:-1])
                                      
                elif token == "Ġ:":
                    next_token_idx = token_idx + 4

                elif token.startswith("Ġ") and not (graph_tokens[token_idx - 1].startswith("Ġ:") or token.startswith("Ġ)") or token.startswith("Ġ(") or token.startswith("Ġop") or token.startswith("Ġsnt") or token.startswith("Ġs") and graph_tokens[token_idx + 1].startswith("n") or token.startswith("Ġprep")):
                    target_node_map.setdefault(node_pos, []).append(token_idx)

                if token.startswith("Ġ:") and (graph_tokens[next_token_idx].startswith("Ġ(") or graph_tokens[next_token_idx].startswith("Ġ<lit>")):
                    next_token_idx += 1
                    node_pos += "." + str(current_node_pos)
                    current_node_pos += 1
                    node_pos_stack.append(current_node_pos)
                    current_node_pos = 1
                    target_node_map.setdefault(node_pos+ ".r", []).append(token_idx)



                elif  (token.startswith("Ġ)") or token.startswith("Ġ</lit>")) and node_pos_stack:
                    node_pos = ".".join(node_pos.split(".")[:-1])
                    current_node_pos = node_pos_stack.pop()

            input_word_pos_map = {}
            pos = 0
            sentence_tokenized = self.tokenizer.convert_ids_to_tokens(input_ids[sentence_idx])

            for word_idx, word in enumerate(sentence_tokenized):
                if word.startswith("Ġ") and word != "Ġ<s>" and word != "Ġ</s>" and not (word == "Ġ<" and (word_idx + 1) < len (sentence_tokenized) and sentence_tokenized[word_idx + 1] == "a"):
                    if pos in alignment_map:
                        for node_position in alignment_map[pos]:
                            if node_position in target_node_map:
                                batch_alignment[sentence_idx][word_idx][target_node_map[node_position]] = 1

                    pos += 1

                elif not word.startswith("Ġ"):
                    input_word_pos_map[word_idx] = pos

        return batch_alignment

            

    def collate_fn(self, samples, device=torch.device('cpu')):
        sentences = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(sentences, device=device)

        graphs = [s['graphs'] for s in samples]
        y, extra_y = self.tokenizer.batch_encode_graphs(graphs, device=device)
        extra.update(extra_y)

        alignments = [s['alignments'] for s in samples]

        xx = {k: v.tolist() for k, v in x.items()}
        yy = {k: v.tolist() for k, v in y.items()}

        alignment = self._add_alingment(xx['input_ids'], yy['decoder_input_ids'], alignments)

        x['alignments'] = alignment.to(device)

        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None

        extra['ids'] = [s['id'] for s in samples]

        extra['sentences_id'] = [s['sentences_id'] for s in samples]

        return x, y, extra
    


class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
