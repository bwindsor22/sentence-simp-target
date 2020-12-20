import logging
import random
import pandas as pd
import torch
from pprint import pprint
from nltk.translate.bleu_score import sentence_bleu

from transformers import EncoderDecoderModel, RobertaTokenizer, AdamW

"""
PARAMETERS AND LOGGING
"""
model_name = 'roberta-base'
run_name = 'enc-dec-org-norp'

epochs = 580
test_every = 20
batch_size = 8
learning_rate = 3e-5
device = 'cpu'

log_file_name = './{}-{}-epch-{}-batch-{}-lr-{}-out.log'.format(run_name, model_name, epochs, batch_size, learning_rate)
logging.basicConfig(filename=log_file_name, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

"""
DATASETS
"""
def get_df():
    # file_path = '/Users/bradwindsor/classwork/natural_language_processing/paper/org-norp-loc-batch-1,3,2.csv'
    file_path = '/Users/bradwindsor/classwork/natural_language_processing/paper/masked_dataset.csv'
    return pd.read_csv(file_path, engine='python')

def org_norp():
    # column_name = 'ORG NORP Simplified'
    column_name = 'org_norp'
    return get_data_for_column(column_name)


def org_loc():
    # column_name = 'ORG LOC Simplified'
    column_name = 'org_loc'
    return get_data_for_column(column_name)

def get_data_for_column(column_name):
    df = get_df()
    df = df[df[column_name] != 'pass']
    df = df.fillna('')
    df = df[df['text'] != '']
    df = df[df[column_name] != '']
    text = df[['text']].values
    simplified = df[[column_name]].values
    tups = [(a[0], b[0]) for a, b in zip(text, simplified)]
    return tups

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

"""
SET UP TRAINING DATA AND MODEL
"""
def tokenize(text, max_len=60):
    toks = tokenizer.encode(text)
    return pad_to_len(toks, max_len=max_len)


def pad_to_len(toks, max_len=60):
    if len(toks) <= max_len:
        return toks + [0] * (max_len - len(toks))
    return toks[:max_len]


data_encoded = org_norp()

train_len = int(0.9 * len(data_encoded))
train_data = data_encoded[:train_len]
test_data = data_encoded[train_len:]
logging.info('len train %d, len test, %d', len(train_data), len(test_data))

tokenizer = RobertaTokenizer.from_pretrained(model_name)
model_save_dir = './'

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
model.train()
model.to(device)


"""
TRAIN
"""
total_steps = epochs * int(len(train_data) / batch_size)
optimizer = AdamW(model.parameters(), lr=learning_rate)


# warmup_frac = 0.1
# scheduler = get_linear_schedule_with_warmup(optimizer, total_steps * warmup_frac, total_steps)

logging.info('first datapoint')
logging.info(str(train_data[0][0]))
logging.info(str(tokenize(train_data[0][0])))
logging.info(str(train_data[0][1]))
logging.info(str(tokenize(train_data[0][1])))
logging.info(str(train_data[1][0]))
logging.info(str(tokenize(train_data[1][0])))
logging.info(str(train_data[1][1]))
logging.info(str(tokenize(train_data[1][1])))
logging.info('beginning train')


# help with inference
def run_snip(snip, name):
    inp_ids = torch.tensor([tokenize(e[0]) for e in snip]).to(device)
    generated = model.generate(inp_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
    logging.info('~~~~~Evaluating on test~~~~~~~~~~')
    logging.info('%s data %s', name, str(snip))
    # logging.info('generated %s', str(generated))

    gen = generated.cpu().numpy()

    target_simplifications = [e[1] for e in snip]

    bleus = []
    for i, row in enumerate(gen):
        target = target_simplifications[i]
        output = tokenizer.decode(row)
        logging.info("target %s", str(target))
        logging.info("actual %s", str(output))
        cleaned_out = output.replace('<pad>', '').replace('<s>', ' ')
        bleus.append(sentence_bleu([target], cleaned_out))
    avg_bleu = sum(bleus) / len(bleus)
    logging.info("Average Bleu %s", str(avg_bleu))

cume_losses = []
for epoch in range(epochs):
    cume_loss = 0
    for chunk in divide_chunks(train_data, batch_size):
        inp_ids = torch.tensor([tokenize(e[0], max_len=60) for e in chunk]).to(device)
        out_ids = torch.tensor([tokenize(e[1], max_len=15) for e in chunk]).to(device)
        optimizer.zero_grad()
        out = model(input_ids=inp_ids, decoder_input_ids=out_ids, labels=out_ids)
        loss = out['loss']
        loss.backward()
        optimizer.step()
        # scheduler.step()
        logging.info('...loss %f', loss.item())
        cume_loss += loss.item()

    if epoch % 999 == 0 or epoch == epochs - 1:
        model.save_pretrained(model_save_dir)
    if epoch % test_every == 0:
        snip = test_data[:10]
        run_snip(snip, 'test')

    cume_losses.append(cume_loss)
    logging.info('epoch %d, loss %d', epoch, cume_loss)

losses_by_epoch = [(i, val) for i, val in enumerate(cume_losses)]
print('loss by epoch')
pprint(losses_by_epoch)


"""
RUN INFERENCE
"""
logging.info('End train inference on train and test')
snip = train_data[:10]

run_snip(snip, 'train')

snip = test_data[:10]

run_snip(snip, 'test')