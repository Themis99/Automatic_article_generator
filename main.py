import pandas as pd
import text_extraction
import text_synthesis


#Greece wildfiles June 2022 links
url_list1 = ['https://apnews.com/article/wildfires-greece-athens-climate-and-environment-2fa67008f59593655cb7677bbf174e7f',
             'https://abcnews.go.com/International/wireStory/greek-firefighters-battle-blaze-athens-day-85192862',
             'https://www.ot.gr/2022/06/04/english-edition/emergency-message-to-ano-voula-residents-to-evacuate-while-firefighters-battle-blaze-at-mt-hymettus-foothills/',
             'https://www.washingtonpost.com/world/greek-firefighters-battle-blaze-near-athens-for-second-day/2022/06/05/c9bb941a-e4c1-11ec-a422-11bbb91db30b_story.html']

# Brazil floods May 2022 links
url_list2 = ['https://www.aljazeera.com/news/2022/4/2/brazil-several-killed-landslides-floods',
            'https://www.france24.com/en/americas/20220530-death-toll-rises-after-heavy-rainfall-in-brazil-sparks-floods-and-landslides',
            'https://www.cnbc.com/2022/05/31/death-toll-from-brazil-floods-at-least-91-with-dozens-lost.html']

text_list = []
for url in url_list1:
    text_list.append(text_extractor.take_text(url))

data_texts = pd.Series(text_list)

# directories for model and input texts
MODEL_DIR = './model_dir'
MAX_LEN = 180

generation_module = text_synthesis.ModelGeneration(model_dir=MODEL_DIR, max_len=MAX_LEN)
input_text = generation_module.preprocesiing(data_texts)
generate = generation_module.generate(input_text=input_text)

print(generate)
