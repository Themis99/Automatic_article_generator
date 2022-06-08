from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_metric
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import sys
import warnings
warnings.filterwarnings("ignore")

def clean(line):
    line = line.strip().replace("newline_char", " ")
    line = line.replace("( opens in new window )", "")
    line = line.replace("click to email this to a friend", "")
    line = line.replace("lick to share on whatsapp", "")
    line = line.replace("click to share on facebook", "")
    line = line.replace("share on facebook", "")
    line = line.replace("click to share on twitter", "")
    line = line.replace("click to share on pinterest", "")
    line = line.replace("click to share on tumblr", "")
    line = line.replace("click to share on google+", "")
    line = line.replace("feel free to share these resources in your social "
                        "media networks , websites and other platforms", "")
    line = line.replace("share share tweet link", "")
    line = line.replace("e-mail article print share", "")
    line = line.replace("read or share this story :", "")
    line = line.replace("share the map view in e-mail by clicking the share "
                        "button and copying the link url .     embed the map "
                        "on your website or blog by getting a snippet of html "
                        "code from the share button .     if you wish to "
                        "provide feedback or comments on the map , or if "
                        "you are aware of map layers or other "
                        "datasets that you would like to see included on our maps , "
                        "please submit them for our evaluation using this this form .", "")
    line = line.replace("share this article share tweet post email", "")
    line = line.replace("skip in skip x embed x share close", "")
    line = line.replace("share tweet pin email", "")
    line = line.replace("share on twitter", "")
    line = line.replace("feel free to weigh-in yourself , via"
                        "the comments section . and while you ’ "
                        "re here , why don ’ t you sign up to "
                        "follow us on twitter us on twitter .", "")
    line = line.replace("follow us on facebook , twitter , instagram and youtube", "")
    line = line.replace("follow us on twitter", "")
    line = line.replace("follow us on facebook", "")
    line = line.replace("play facebook twitter google plus embed", "")
    line = line.replace("play facebook twitter embed", "")
    line = line.replace("enlarge icon pinterest icon close icon", "")
    line = line.replace("follow on twitter", "")
    line = line.replace("autoplay autoplay copy this code to your website or blog", "")
    line = line.replace("( Newser )  ", "")
    return line

class ModelGeneration:
  def __init__(self,model_dir,max_len):

    self.model_dir = model_dir
    self.max_len = max_len

  def preprocesiing(self,dataset):
    new = []
    for i in dataset:
      new.append(str(i))

    new = pd.Series(new)
    #concatenate
    concatenated = '|||||'.join(new)

    return concatenated

  def generate(self,input_text):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(device)
    #Multi-document summarization
    model = PegasusForConditionalGeneration.from_pretrained(self.model_dir).to(device)
    tokenizer = PegasusTokenizer.from_pretrained(self.model_dir)

    model.config.length_penalty = 1
    model.config.max_length = self.max_len
    model.config.min_length = self.max_len

    Tokenized = tokenizer(input_text, truncation=True, padding=True,return_tensors='pt').to(device)
    summary_ids = model.generate(Tokenized['input_ids'],num_beams=10)
    Generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    return Generated_text

if __name__ == "__main__":

    # directories for model and input texts
    MODEL_DIR = './model_dir'
    DATASET_DIR = './data_texts.txt'
    MAX_LEN = 180

    data_texts = pd.read_csv(DATASET_DIR,delimiter="\t",header=None)[0]
    print(data_texts)

    generation_module = ModelGeneration(model_dir = MODEL_DIR, max_len = MAX_LEN)

    input_text = generation_module.preprocesiing(data_texts)

    generate = generation_module.generate(input_text = input_text)

    print(generate)