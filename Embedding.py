import json
import os
from sys import prefix

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
# nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from scipy.optimize import linear_sum_assignment
import wandb

import pdb


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_BATCH_SIZE = 16    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 200  # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 30        # number of epochs to train (default: 10)
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
SEED = 42               # random seed (default: 42)
MAX_LEN = 512
SUMMARY_LEN = 1024
DEBUG = False
EVAL = False
EVAL_TRAIN = False
NUM_OUT = 1  # number of sampled output sentences
MOD_EVAL = TRAIN_EPOCHS
T5_MODEL_TYPE = "Vamsi/T5_Paraphrase_Paws"
T5_CHECKPOINT_PATH = "/projects/katefgroup/language_grounding/simbot/t5_best.pt"
SIMBOT_DIR = '/projects/katefgroup/language_grounding/simbot'
FINETUNED = False

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class NearestNeighbor:
    def __init__(self, T5_checkpoint_path=T5_CHECKPOINT_PATH, T5_model_type=T5_MODEL_TYPE, template_instructions='./templates.txt', device=DEVICE, seed=SEED) -> None:
        self.device = device
        self.T5_checkpoint = T5_checkpoint_path
        self.T5_model_type = T5_model_type
        self.template_instructions = template_instructions
        self.seed = seed
        self.parser_model, self.tokenizer = self.load_model()
        self.gram = ("NP: {<DT>?<JJ>*<NN>}") # chunking grammar
        self.chunker = nltk.RegexpParser(self.gram)
        
        # load template instructions
        with open(self.template_instructions) as f:
            self.neighbors = f.read().splitlines() 

        print(self.neighbors)
               
        # load template embeddings
        embeddings = []
        neighbor_noun_embeddings = []
        for neighbor in self.neighbors:
            sent_embedding, noun_phrase_embeddings = self.get_final_embedding(neighbor)
            embeddings.append(sent_embedding)
            neighbor_noun_embeddings.append(noun_phrase_embeddings)
            
        # stack embeddings
        self.neighbor_sentence_embeddings = torch.stack(embeddings)
        # neighbour noun phrase embeddings
        self.neighbor_noun_phrase_embeddings = neighbor_noun_embeddings
        
        print("--- NearestNeighbor initialized ---")
        
        
    def get_topk_results(self, noun_phrase_embeddings, neighbor_noun_embeddings_list, neighbors, k=10):
        # sort using hungarian loss
        losses = []
        for i in range(len(neighbor_noun_embeddings_list)):
            neighbor_noun_embeddings = neighbor_noun_embeddings_list[i]
            loss = self.get_hungarian_loss(noun_phrase_embeddings, neighbor_noun_embeddings)
            losses.append(loss)
        losses = torch.stack(losses)
        topk_idx = losses.topk(k)[1]
        topk_results = []
        for idx in topk_idx:
            topk_results.append(neighbors[idx])
        return topk_results
    
    
    def get_hungarian_loss(self, noun_phrase_embeddings, neighbor_noun_embeddings):
        # pdb.set_trace()
        # get cosine similarity matrix
        sim_matrix = []
        for neighbor_noun_embedding in neighbor_noun_embeddings:
            sim_row = []
            for noun_phrase_embedding in noun_phrase_embeddings:
                sim_row.append(torch.cosine_similarity(noun_phrase_embedding.unsqueeze(0), neighbor_noun_embedding.unsqueeze(0)))
            sim_matrix.append(torch.tensor(sim_row))
        sim_matrix = torch.stack(sim_matrix)
        
        # get hungarian loss
        row_ind, col_ind = linear_sum_assignment(-1 * sim_matrix)
        loss = sim_matrix[row_ind, col_ind].sum()
        return loss
        
    def get_noun_phrases(self, sentence):
        # tokenize sentence
        tokenized = nltk.word_tokenize(sentence) 
        # tag parts of speech
        tagged = nltk.pos_tag(tokenized)
        # chunk with grammar
        tree = self.chunker.parse(tagged)
        # extract noun phrases
        noun_phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                t = [w for w, t in subtree.leaves()]
                noun_phrases.append(' '.join(t))
        return noun_phrases
    
    def get_final_embedding(self, utterance):
        sentence_embedding = self.get_language_embedding(utterance)

        noun_phrases = self.get_noun_phrases(utterance)        
        # Convert Noun Phrases to embeddings
        noun_phrase_embeddings = []
        for noun_phrase in noun_phrases:
            noun_phrase_embeddings.append(self.get_language_embedding(noun_phrase))
        
        # Stack Embeddings to get final vector (N X 768)
        return (sentence_embedding, noun_phrase_embeddings)
        
    @torch.no_grad()
    def run_t5_parser(self, utterance):
        source = self.tokenizer.encode_plus(
            utterance,
            padding='max_length', max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.parser_model.generate(
            input_ids=source['input_ids'].flatten()[None].long().to(self.device),
            attention_mask=source['attention_mask'].flatten()[None].long().to(self.device),
            max_new_tokens=1024,
            num_beams=4,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1
        )[0]
        program = self.tokenizer.decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return program

    def load_model(self, finetuned=FINETUNED):
        tokenizer = AutoTokenizer.from_pretrained(self.T5_model_type)
        parser_model = AutoModelForSeq2SeqLM.from_pretrained(self.T5_model_type)
        
        if finetuned:
            parser_model.load_state_dict(torch.load(self.T5_checkpoint))
    
        parser_model = parser_model.eval().to(self.device)
        return parser_model, tokenizer

    # Extract language embedding from T5 model for a given utterance
    @torch.no_grad()
    def get_language_embedding(self, utterance):
        enc = self.tokenizer.encode_plus(
            utterance,
            padding='max_length', max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # forward pass through encoder only
        output = self.parser_model.encoder(
            input_ids=enc["input_ids"], 
            attention_mask=enc["attention_mask"], 
            return_dict=True
        )
        
        # get the final hidden states, shape is (batch_size, seq_len, hidden_size)
        out_embed = output.last_hidden_state.squeeze()
        
        # attention pooling over the sequence dimension
        out_embed = mean_pooling(out_embed, enc["attention_mask"].squeeze())
        
        return out_embed.squeeze()
    
    @torch.no_grad()
    def get_nearest_neighbors(self, utterance, k=5):
        sent_embedding, noun_phrase_embeddings = self.get_final_embedding(utterance)
        
        # Retrieve using sentence embedding
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos_sim(sent_embedding, self.neighbor_sentence_embeddings)
        sim, idx = torch.topk(sim, k=k)
        
        # Get topk neighbors
        neighbour_noun_embeddings_list = []
        neighbours_list = []
        for i in range(len(idx)):
            neighbour_noun_embeddings_list.append(self.neighbor_noun_phrase_embeddings[idx[i]])
            neighbours_list.append(self.neighbors[idx[i]])
            
        # Get topk neighbors using hungarian loss
        final_nearest_neighbors = self.get_topk_results(noun_phrase_embeddings, neighbour_noun_embeddings_list, neighbours_list, k)
        return final_nearest_neighbors        


if __name__ == '__main__':
    nn = NearestNeighbor()
    # neighbors = nn.get_nearest_neighbors(['go to the red box and pick up the red block'])
    neighbors = nn.get_nearest_neighbors('goto apple')
    print(neighbors)