<<<<<<< HEAD
#----------------------------------------IMPORT REQUIREMENTS----------------------------------------
import os
import sys
import random
import pickle

import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AdamW
)

import wandb

# import costumized modules
from compute import compute_acc
from visualize_score import plot_graph
from dump_datasets import mk_dataset, mk_dataset_xlnet
from dump_models import load_model, load_model_xlnet
from evaluate import test_model
from data_processing import regular

# save logs
sys.stdout = open('./stdout.txt', 'w')

# create required folders if not exists
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
createFolder('./best_models')
createFolder('./dump_datasets')
createFolder('./dump_models_tokenizer')
createFolder('./scores')
createFolder('./submissions')


#----------------------------------------SEED FIX----------------------------------------
def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)


#----------------------------------------MODEL----------------------------------------
# test various models
MODEL_NAME = 'bert-base-uncased'
# MODEL_NAME = 'bert-large-uncased'
# MODEL_NAME = 'xlnet-base-cased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert str(device) == 'cuda'

# by using pickle, load dump file if exist, else make dump file and load
# load for bert(uncased) or xlnet model(cased)
try:
    with open('./dump_models_tokenizer/' + MODEL_NAME + '.p', 'rb') as f:
        model = pickle.load(f)
        tokenizer = pickle.load(f)
    print('model exists => just load model')
except:
    print('exeption occur => download model')
    if MODEL_NAME == 'bert-base-uncased':
        model, tokenizer = load_model(MODEL_NAME)
    elif MODEL_NAME == 'xlnet-base-cased':
        model, tokenizer = load_model_xlnet(MODEL_NAME)

model.to(device)


#----------------------------------------HYPER PARAMETERS----------------------------------------
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=256

LEARNING_RATE = 5e-5
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
TRAIN_EPOCH = 3


#----------------------------------------WANDB----------------------------------------
wandb.init(project="groomProject1", entity="chohs1221")
wandb.run.name = 'bert3_lr' + str(LEARNING_RATE) + '_re'
wandb.config.learning_rate = LEARNING_RATE
wandb.config.epochs = TRAIN_EPOCH
wandb.config.batch_size = TRAIN_BATCH_SIZE


#----------------------------------------LOAD DATASETS----------------------------------------
# by using pickle, load dump file if exist, else make dump file and load
# load for bert(uncased) or xlnet model(cased)
try:
    if  MODEL_NAME == 'bert-base-uncased':
        with open('./dump_datasets/train_dev_dumps.p', 'rb') as f:
            train_pos = pickle.load(f)
            train_neg = pickle.load(f)
            dev_pos = pickle.load(f)
            dev_neg = pickle.load(f)
        print('dataset exists => just load datasets')
    elif  MODEL_NAME == 'xlnet-base-cased':
        with open('./dump_datasets/train_dev_dumps_xlnet.p', 'rb') as f:
            train_pos = pickle.load(f)
            train_neg = pickle.load(f)
            dev_pos = pickle.load(f)
            dev_neg = pickle.load(f)
        print('dataset exists => just load datasets')
except:
    print('exeption occur => make datasets')
    train_pos, train_neg, dev_pos, dev_neg = mk_dataset()
    if MODEL_NAME == 'bert-base-uncased':
        train_pos, train_neg, dev_pos, dev_neg = mk_dataset()
    elif MODEL_NAME == 'xlnet-base-cased':
        train_pos, train_neg, dev_pos, dev_neg = mk_dataset_xlnet()


#----------------------------------------DATA PREPROCESSING----------------------------------------
# Remove '_num_', !@#$ ... from datasets
train_pos = regular(train_pos)
train_neg = regular(train_pos)
dev_pos = regular(train_pos)
dev_neg = regular(train_pos)

#----------------------------------------TOKENIZE----------------------------------------
# seperate encoding to preprocess data before encoding
train_pos = [tokenizer.encode(line) for line in train_pos]
train_neg = [tokenizer.encode(line) for line in train_neg]
dev_pos = [tokenizer.encode(line) for line in dev_pos]
dev_neg = [tokenizer.encode(line) for line in dev_neg]


#----------------------------------------MAKE DATASETS----------------------------------------
# concatenate pos, neg dataset, costomize magic mathod
class SentimentDataset(object):
    def __init__(self, pos, neg):
        self.data = [pos_sent for pos_sent in pos] + [neg_sent for neg_sent in neg]
        self.label = [[1] for _ in range(len(pos))] + [[0] for _ in range(len(neg))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])

train_dataset = SentimentDataset(train_pos, train_neg)
dev_dataset = SentimentDataset(dev_pos, dev_neg)

#----------------------------------------DATA LOADER----------------------------------------
# argsort deleted <- no reason to sort
# attention masking for padding token
def collate_fn_style(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)

    attention_mask = torch.tensor([[1] * len(input_id) + [0] * (max_len - len(input_id)) for input_id in input_ids])
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids], batch_first=True)
    token_type_ids = torch.tensor([[0] * len(input_id) for input_id in input_ids])
    position_ids = torch.tensor([list(range(len(input_id))) for input_id in input_ids])
    labels = torch.tensor(np.stack(labels, axis=0))

    return input_ids, attention_mask, token_type_ids, position_ids, labels

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=TRAIN_BATCH_SIZE,
                                           shuffle=True, 
                                           collate_fn=collate_fn_style,
                                           pin_memory=True, num_workers=2)

dev_loader = torch.utils.data.DataLoader(dev_dataset, 
                                         batch_size=EVAL_BATCH_SIZE,
                                         shuffle=False, 
                                         collate_fn=collate_fn_style,
                                         num_workers=2)

#----------------------------------------TRAIN----------------------------------------
# record datas for highest accuracy & lowest loss
lowest_valid_loss = 9999.
highest_valid_acc = 0.
train_acc = []
train_loss = []
valid_acc = []
valid_loss = []

temp_train_acc = []
temp_train_loss = []

#train model
model.train()
for epoch in range(TRAIN_EPOCH):
    with tqdm(train_loader, unit="batch") as tepoch:
        for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):

            tepoch.set_description(f"Epoch {epoch}")

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)
            labels = labels.to(device, dtype=torch.long)

            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           labels=labels)

            loss = output.loss
            
            logits = output.logits
            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
            batch_labels = [int(example) for example in labels]
            
            acc = compute_acc(batch_predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            tepoch.set_postfix(acc=acc, loss=loss.item())
            
            temp_train_acc.append(acc)
            temp_train_loss.append(loss.item())
            if iteration != 0 and iteration % int(len(train_loader) / 100) == 0:

                # evaluate model
                model.eval()
                with torch.no_grad():
                    val_acc = []
                    val_loss = []
                    for input_ids, attention_mask, token_type_ids, position_ids, labels in dev_loader:
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        token_type_ids = token_type_ids.to(device)
                        position_ids = position_ids.to(device)
                        labels = labels.to(device, dtype=torch.long)

                        output = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    labels=labels)

                        logits = output.logits
                        batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                        batch_labels = [int(example) for example in labels]

                        val_acc.append(compute_acc(batch_predictions, batch_labels))
                        val_loss.append(output.loss)

                # compute accuracy, loss for train, valid datasets and record
                mean_train_acc = sum(temp_train_acc) / len(temp_train_acc)
                mean_train_loss = sum(temp_train_loss) / len(temp_train_loss)
                mean_val_acc = sum(val_acc) / len(val_acc)
                mean_val_loss = sum(val_loss) / len(val_loss)

                train_acc.append(mean_train_acc)
                train_loss.append(mean_train_loss)
                valid_acc.append(mean_val_acc)
                valid_loss.append(mean_val_loss)

                temp_train_acc = []
                temp_train_loss = []
                
                # WANDB
                wandb.log({"train_loss": mean_train_loss,
                           'train_acc': mean_train_acc,
                           'valid_loss': mean_val_loss,
                           'valid_acc': mean_val_acc})

                # save best models
                if highest_valid_acc < mean_val_acc:
                    highest_valid_acc = mean_val_acc
                    print('ACCURACY for highest valid acc: ', mean_val_acc)
                    print('LOSS for lowest valid acc: ', mean_val_loss)
                    model.save_pretrained('./best_models/model' + str(int(mean_val_acc*100)) + str(int(mean_val_loss*1000)))

                elif lowest_valid_loss > mean_val_loss:
                    lowest_valid_loss = mean_val_loss
                    print('ACCURACY for lowest valid loss: ', mean_val_acc)
                    print('LOSS for lowest valid loss: ', mean_val_loss)
                    model.save_pretrained('./best_models/model' + str(int(mean_val_acc*100)) + str(int(mean_val_loss*1000)))
                                        
                model.train()


#----------------------------------------SAVE SCORES----------------------------------------
# using pickle, save dump accuracy, loss file
accloss_filename = 'accloss' + str(int(mean_val_acc*100)) + str(int(mean_val_loss*1000)) + '.p'
with open('./scores/' + accloss_filename,'wb') as f:
    pickle.dump(train_acc, f)
    pickle.dump(train_loss, f)
    pickle.dump(valid_acc, f)
    pickle.dump(valid_loss, f)


#----------------------------------------TEST----------------------------------------
# test model
test_model(model, tokenizer, mean_val_acc, mean_val_loss, file_name = 'test_no_label', device='cuda')


#----------------------------------------SCORE VISUALIZE----------------------------------------
# plot accuracy, loss graph for train, valid datasets
plot_graph(accloss_filename)

# save logs
sys.stdout.close()
=======
#----------------------------------------IMPORT REQUIREMENTS----------------------------------------
import os
import sys
import random
import pickle

import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AdamW
)

import wandb

# import costumized modules
from compute import compute_acc
from visualize_score import plot_graph
from dump_datasets import mk_dataset, mk_dataset_xlnet
from dump_models import load_model, load_model_xlnet
from evaluate import test_model
from data_processing import regular

# save logs
sys.stdout = open('./stdout.txt', 'w')

# create required folders if not exists
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
createFolder('./best_models')
createFolder('./dump_datasets')
createFolder('./dump_models_tokenizer')
createFolder('./scores')
createFolder('./submissions')


#----------------------------------------SEED FIX----------------------------------------
def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)


#----------------------------------------MODEL----------------------------------------
# test various models
# MODEL_NAME = 'bert-base-uncased'
# MODEL_NAME = 'bert-large-uncased'
MODEL_NAME = 'xlnet-base-cased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert str(device) == 'cuda'

# by using pickle, load dump file if exist, else make dump file and load
# load for bert(uncased) or xlnet model(cased)
try:
    with open('./dump_models_tokenizer/' + MODEL_NAME + '.p', 'rb') as f:
        model = pickle.load(f)
        tokenizer = pickle.load(f)
    print('model exists => just load model')
except:
    print('exeption occur => download model')
    if MODEL_NAME == 'bert-base-uncased':
        model, tokenizer = load_model(MODEL_NAME)
    elif MODEL_NAME == 'xlnet-base-cased':
        model, tokenizer = load_model_xlnet(MODEL_NAME)

model.to(device)


#----------------------------------------HYPER PARAMETERS----------------------------------------
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=256

LEARNING_RATE = 5e-5
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
TRAIN_EPOCH = 2


#----------------------------------------WANDB----------------------------------------
wandb.init(project="groomProject1", entity="chohs1221")
wandb.run.name = 'xlnet2_lr' + str(LEARNING_RATE)
wandb.config.learning_rate = LEARNING_RATE
wandb.config.epochs = TRAIN_EPOCH
wandb.config.batch_size = TRAIN_BATCH_SIZE


#----------------------------------------LOAD DATASETS----------------------------------------
# by using pickle, load dump file if exist, else make dump file and load
# load for bert(uncased) or xlnet model(cased)
try:
    if  MODEL_NAME == 'bert-base-uncased':
        with open('./dump_datasets/train_dev_dumps.p', 'rb') as f:
            train_pos = pickle.load(f)
            train_neg = pickle.load(f)
            dev_pos = pickle.load(f)
            dev_neg = pickle.load(f)
        print('dataset exists => just load datasets')
    elif  MODEL_NAME == 'xlnet-base-cased':
        with open('./dump_datasets/train_dev_dumps_xlnet.p', 'rb') as f:
            train_pos = pickle.load(f)
            train_neg = pickle.load(f)
            dev_pos = pickle.load(f)
            dev_neg = pickle.load(f)
        print('dataset exists => just load datasets')
except:
    print('exeption occur => make datasets')
    train_pos, train_neg, dev_pos, dev_neg = mk_dataset()
    if MODEL_NAME == 'bert-base-uncased':
        train_pos, train_neg, dev_pos, dev_neg = mk_dataset()
    elif MODEL_NAME == 'xlnet-base-cased':
        train_pos, train_neg, dev_pos, dev_neg = mk_dataset_xlnet()


#----------------------------------------DATA PREPROCESSING----------------------------------------
# Remove '_num_', !@#$ ... from datasets
# train_pos = regular(train_pos)
# train_neg = regular(train_pos)
# dev_pos = regular(train_pos)
# dev_neg = regular(train_pos)

#----------------------------------------TOKENIZE----------------------------------------
# seperate encoding to preprocess data before encoding
train_pos = [tokenizer.encode(line) for line in train_pos]
train_neg = [tokenizer.encode(line) for line in train_neg]
dev_pos = [tokenizer.encode(line) for line in dev_pos]
dev_neg = [tokenizer.encode(line) for line in dev_neg]


#----------------------------------------MAKE DATASETS----------------------------------------
# concatenate pos, neg dataset, costomize magic mathod
class SentimentDataset(object):
    def __init__(self, pos, neg):
        self.data = [pos_sent for pos_sent in pos] + [neg_sent for neg_sent in neg]
        self.label = [[1] for _ in range(len(pos))] + [[0] for _ in range(len(neg))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])

train_dataset = SentimentDataset(train_pos, train_neg)
dev_dataset = SentimentDataset(dev_pos, dev_neg)

#----------------------------------------DATA LOADER----------------------------------------
# argsort deleted <- no reason to sort
# attention masking for padding token
def collate_fn_style(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)

    attention_mask = torch.tensor([[1] * len(input_id) + [0] * (max_len - len(input_id)) for input_id in input_ids])
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids], batch_first=True)
    token_type_ids = torch.tensor([[0] * len(input_id) for input_id in input_ids])
    position_ids = torch.tensor([list(range(len(input_id))) for input_id in input_ids])
    labels = torch.tensor(np.stack(labels, axis=0))

    return input_ids, attention_mask, token_type_ids, position_ids, labels

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=TRAIN_BATCH_SIZE,
                                           shuffle=True, 
                                           collate_fn=collate_fn_style,
                                           pin_memory=True, num_workers=2)

dev_loader = torch.utils.data.DataLoader(dev_dataset, 
                                         batch_size=EVAL_BATCH_SIZE,
                                         shuffle=False, 
                                         collate_fn=collate_fn_style,
                                         num_workers=2)

#----------------------------------------TRAIN----------------------------------------
# record datas for highest accuracy & lowest loss
lowest_valid_loss = 9999.
highest_valid_acc = 0.
train_acc = []
train_loss = []
valid_acc = []
valid_loss = []

temp_train_acc = []
temp_train_loss = []

#train model
model.train()
for epoch in range(TRAIN_EPOCH):
    with tqdm(train_loader, unit="batch") as tepoch:
        for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):

            tepoch.set_description(f"Epoch {epoch}")

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)
            labels = labels.to(device, dtype=torch.long)

            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           labels=labels)

            loss = output.loss
            
            logits = output.logits
            batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
            batch_labels = [int(example) for example in labels]
            
            acc = compute_acc(batch_predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            tepoch.set_postfix(acc=acc, loss=loss.item())
            
            temp_train_acc.append(acc)
            temp_train_loss.append(loss.item())
            if iteration != 0 and iteration % int(len(train_loader) / 100) == 0:

                # evaluate model
                model.eval()
                with torch.no_grad():
                    val_acc = []
                    val_loss = []
                    for input_ids, attention_mask, token_type_ids, position_ids, labels in dev_loader:
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        token_type_ids = token_type_ids.to(device)
                        position_ids = position_ids.to(device)
                        labels = labels.to(device, dtype=torch.long)

                        output = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    labels=labels)

                        logits = output.logits
                        batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                        batch_labels = [int(example) for example in labels]

                        val_acc.append(compute_acc(batch_predictions, batch_labels))
                        val_loss.append(output.loss)

                # compute accuracy, loss for train, valid datasets and record
                mean_train_acc = sum(temp_train_acc) / len(temp_train_acc)
                mean_train_loss = sum(temp_train_loss) / len(temp_train_loss)
                mean_val_acc = sum(val_acc) / len(val_acc)
                mean_val_loss = sum(val_loss) / len(val_loss)

                train_acc.append(mean_train_acc)
                train_loss.append(mean_train_loss)
                valid_acc.append(mean_val_acc)
                valid_loss.append(mean_val_loss)

                temp_train_acc = []
                temp_train_loss = []
                
                # WANDB
                wandb.log({"train_loss": mean_train_loss,
                           'train_acc': mean_train_acc,
                           'valid_loss': mean_val_loss,
                           'valid_acc': mean_val_acc})

                # save best models
                if highest_valid_acc < mean_val_acc:
                    highest_valid_acc = mean_val_acc
                    print('ACCURACY for highest valid acc: ', mean_val_acc)
                    print('LOSS for lowest valid acc: ', mean_val_loss)
                    # model.save_pretrained('./best_models/model' + str(int(mean_val_acc*100)) + str(int(mean_val_loss*1000)))

                elif lowest_valid_loss > mean_val_loss:
                    lowest_valid_loss = mean_val_loss
                    print('ACCURACY for lowest valid loss: ', mean_val_acc)
                    print('LOSS for lowest valid loss: ', mean_val_loss)
                    # model.save_pretrained('./best_models/model' + str(int(mean_val_acc*100)) + str(int(mean_val_loss*1000)))
                                        
                model.train()


#----------------------------------------SAVE SCORES----------------------------------------
# using pickle, save dump accuracy, loss file
accloss_filename = 'accloss' + str(int(mean_val_acc*100)) + str(int(mean_val_loss*1000)) + '.p'
with open('./scores/' + accloss_filename,'wb') as f:
    pickle.dump(train_acc, f)
    pickle.dump(train_loss, f)
    pickle.dump(valid_acc, f)
    pickle.dump(valid_loss, f)


#----------------------------------------TEST----------------------------------------
# test model
test_model(model, tokenizer, mean_val_acc, mean_val_loss, file_name = 'test_no_label', device='cuda')


#----------------------------------------SCORE VISUALIZE----------------------------------------
# plot accuracy, loss graph for train, valid datasets
plot_graph(accloss_filename)

# save logs
sys.stdout.close()
>>>>>>> a276db7324688919f66b41a3801e2fc014889596
