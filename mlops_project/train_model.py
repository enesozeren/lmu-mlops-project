import logging
from data.utils import get_datasets, preprocessing, b_metrics
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader,  RandomSampler, SequentialSampler
import random
import numpy as np
from tqdm import tqdm

# Suppress warnings from the transformers library
logging.basicConfig(level=logging.ERROR)

# Hyperparameters
random_seed = 76
batch_size = 16
epochs = 2

# Get the training, validation, and test datasets
(train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels) = get_datasets()


######################
### Pre-processing ###
######################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

token_id = []
attention_masks = []

for sample in train_tweets:
  encoding_dict = preprocessing(sample, tokenizer) 
  token_id.append(encoding_dict['input_ids'])
  attention_masks.append(encoding_dict['attention_mask'])

token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(train_labels)

train_set = TensorDataset(token_id,attention_masks, labels)
val_set = TensorDataset(token_id, attention_masks, labels)

train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )


################
### Training ###
################
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels = 2, 
    output_attentions = False, # don't return attention weights of model
    output_hidden_states = False, # don't return hidden states
)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08 
                              )

#model.cuda()

# Set random seed for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
#torch.cuda.manual_seed_all(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(epochs): 
    # ========== Training ==========
    model.train()
    
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for batch in tqdm(train_dataloader, desc = f"Epoch: {epoch}, Iter: "):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad() 

        # Forward pass
        train_output = model(b_input_ids, 
                             token_type_ids = None,
                             attention_mask = b_input_mask, 
                             labels = b_labels)

        # Backward pass
        train_output.loss.backward()
        optimizer.step()

        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
 
    print(f"Epoch {epoch} is finished.")

    # ========== Validation ==========
    model.eval()

    # Tracking variables 
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []
    val_logits = []
    val_probs = []
    val_preds = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
          # Forward pass
          eval_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate validation metrics
        b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        # Update precision only when (tp + fp) !=0; ignore nan
        if b_precision != 'nan': val_precision.append(b_precision)
        # Update recall only when (tp + fn) !=0; ignore nan
        if b_recall != 'nan': val_recall.append(b_recall)
        # Update specificity only when (tn + fp) !=0; ignore nan
        if b_specificity != 'nan': val_specificity.append(b_specificity)

        # Track logits, probabilities, and predictions
        val_logits.extend(logits)
        val_probs.extend(torch.softmax(torch.tensor(logits), dim=1)[:,1].tolist())
        val_preds.extend(torch.argmax(torch.softmax(torch.tensor(logits), dim=1), dim=1).tolist())

    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
    print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
    print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
    print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

    val_logits = np.array(val_logits)
    val_probs = np.array(val_probs)
    val_preds = np.array(val_preds) 

# Save model
torch.save(model.state_dict(), 'weights_config01.pth')


##################
### Test model ###
##################
token_id_test = []
attention_masks_test = []

for sample in test_tweets:
    encoding_dict = preprocessing(sample, tokenizer) 
    token_id_test.append(encoding_dict['input_ids'])
    attention_masks_test.append(encoding_dict['attention_mask'])

token_id_test = torch.cat(token_id_test, dim=0)
attention_masks_test = torch.cat(attention_masks_test, dim=0)
labels_test = torch.tensor(test_labels)

test_set = TensorDataset(token_id_test, attention_masks_test, labels_test)

test_dataloader = DataLoader(
    test_set,
    sampler=SequentialSampler(test_set),
    batch_size=batch_size
)

#model.load_state_dict(torch.load('weights_config01.pth'))
model.eval()
test_accuracy = []
test_precision = []
test_recall = []
test_specificity = []

for batch in tqdm(test_dataloader, desc="Testing: "):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        # Forward pass
        eval_output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = eval_output.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate evaluation metrics
    b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
    test_accuracy.append(b_accuracy)
    # Update precision only when (tp + fp) !=0; ignore nan
    if b_precision != 'nan': test_precision.append(b_precision)
    # Update recall only when (tp + fn) !=0; ignore nan
    if b_recall != 'nan': test_recall.append(b_recall)
    # Update specificity only when (tn + fp) !=0; ignore nan
    if b_specificity != 'nan': test_specificity.append(b_specificity)

print('\n\t - Test Accuracy: {:.4f}'.format(sum(test_accuracy)/len(test_accuracy)))
print('\t - Test Precision: {:.4f}'.format(sum(test_precision)/len(test_precision)) if len(test_precision)>0 else '\t - Test Precision: NaN')
print('\t - Test Recall: {:.4f}'.format(sum(test_recall)/len(test_recall)) if len(test_recall)>0 else '\t - Test Recall: NaN')
print('\t - Test Specificity: {:.4f}\n'.format(sum(test_specificity)/len(test_specificity)) if len(test_specificity)>0 else '\t - Test Specificity: NaN')