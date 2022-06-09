import pandas as pd
import text_extractor
import pegasus_inference

url_list = ['https://www.aljazeera.com/news/2022/4/2/brazil-several-killed-landslides-floods',
            'https://www.france24.com/en/americas/20220530-death-toll-rises-after-heavy-rainfall-in-brazil-sparks-floods-and-landslides',
            'https://www.cnbc.com/2022/05/31/death-toll-from-brazil-floods-at-least-91-with-dozens-lost.html']

text_list = []
for url in url_list:
    text_list.append(text_extractor.take_text(url))

data_texts = pd.Series(text_list)

# directories for model and input texts
MODEL_DIR = './model_dir'
DATASET_DIR = './data_texts.txt'
MAX_LEN = 180

generation_module = pegasus_inference.ModelGeneration(model_dir=MODEL_DIR, max_len=MAX_LEN)
input_text = generation_module.preprocesiing(data_texts)
generate = generation_module.generate(input_text=input_text)

print(generate)