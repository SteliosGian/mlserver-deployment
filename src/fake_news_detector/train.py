"""
Model training
"""
import logging
import torch
import yaml
import time
import random
import zipfile
import torch.optim as optim
import preprocess as pp
import numpy as np
import pandas as pd
from utils import utils
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

logging.basicConfig(level=logging.INFO)

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def train(model: BertForSequenceClassification,
          optimizer: torch.optim,
          epochs: int,
          scheduler: SequentialSampler,
          seed: int,
          train_inputs: torch.Tensor,
          train_masks: torch.Tensor,
          train_labels: torch.Tensor,
          batch_size: int,
          validation_inputs: torch.Tensor,
          validation_masks: torch.Tensor,
          validation_labels: torch.Tensor
          ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 1

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        logging.info('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model = model.to(device)

        model.train()

        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = utils.format_time(time.time() - t0)

                # Report progress.
                logging.info('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        logging.info("")
        logging.info("Average training loss: {0:.2f}".format(avg_train_loss))
        logging.info("Training epoch took: {:}".format(utils.format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        logging.info("")
        logging.info("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_accuracy = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():

                outputs = model(b_input_ids,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = utils.calc_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        logging.info("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        logging.info("Validation took: {:}".format(utils.format_time(time.time() - t0)))

    # Save trained model
    utils.save_model(config['paths']['DESTINATION_DIR'], model)


if __name__ == "__main__":
    logging.info("Reading data")
    zf = zipfile.ZipFile("data/news.csv.zip")
    df = pd.read_csv(zf.open('news.csv'))

    logging.info("Encoding label")
    LABEL_COL = config['params']['LABEL_COLUMN']
    encoded_label = np.where(df[LABEL_COL] == "FAKE", 1, 0)

    preprocessor = pp.preprocess(df=df,
                                 max_length=config['params']['MAX_LENGTH'],
                                 padding=config['params']['PADDING'],
                                 test_size=config['params']['TEST_SIZE'],
                                 text_column=config['params']['TEXT_COLUMN'],
                                 pretrained_model=config['params']['PRETRAINED_MODEL']
                                 )

    logging.info("Tokenizing...")
    input_ids = preprocessor.tokenize()

    logging.info("Get attention masks")
    attention_masks = preprocessor.attention_masks(input_ids=input_ids)

    logging.info("Data train-test split")
    train_inputs, validation_inputs, train_labels, validation_labels = preprocessor.data_split(input_ids=input_ids,
                                                                                               encoded_label=encoded_label)

    logging.info("Masks train-test split")
    train_masks, validation_masks = preprocessor.mask_split(attention_masks=attention_masks,
                                                            encoded_label=encoded_label)

    logging.info("Converting data to tensors...")
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = preprocessor.convert_to_tensor(train_inputs,
                                                                                                                                     validation_inputs,
                                                                                                                                     train_labels,
                                                                                                                                     validation_labels,
                                                                                                                                     train_masks,
                                                                                                                                     validation_masks)
    logging.info("Loading BERT")
    model = BertForSequenceClassification.from_pretrained(config['params']['PRETRAINED_MODEL'],
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False,
                                                          )

    optimizer = optim.Adam(model.parameters(),
                           lr=2e-5,
                           eps=1e-8
                           )

    total_steps = len(train_inputs) * config['params']['EPOCHS']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps
                                                )
    logging.info("Start training...")
    train(model=model,
          optimizer=optimizer,
          epochs=config['params']['EPOCHS'],
          scheduler=scheduler,
          seed=1,
          train_inputs=train_inputs,
          train_masks=train_masks,
          train_labels=train_labels,
          batch_size=config['params']['BATCH_SIZE'],
          validation_inputs=validation_inputs,
          validation_masks=validation_masks,
          validation_labels=validation_labels
          )
