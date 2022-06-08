# Automatic_article_generator

In this project we use the PEGASUS model, pretrained on the multi-news dataset, to automatically compose articles related to natural disasters. This project is part of my contribution to the SnapEarth project when I worked for CERTH. 

## Project architecture
![Project architecture](https://user-images.githubusercontent.com/46052843/172661217-11814181-2332-4ce7-a0ac-e413c01a8016.png)
As shown in the architecture image, the user will be able to provide a list of URLs that can correspond to articles. Then the text from these URLs is extracted and fed into the next module where the text is composed. In this module the concatenation of the texts is made and then the concatenated tex fed into the model were a new text is generated.

## Research 
More specifically the task of this project is Multi-document summarization where we try to generate a summary from multiple texts in contrast with the single document summarization when we try to generate a summary from a single document. In this project the generated texts are more domain-specific and more specifically related to natural disasters news, so we choose the PEGASUS [1] model because it shows the best performance in the task of multi-document summarization for the Multi-news dataset [2] which is close to our domain. The Multi-news dataset is a collection of news articles from various sites with their corresponding summaries.

## Datasets and training
Because we wanted to make our model generate more domain-specific texts (Texts related to natural disasters news) we created a new dataset that includes texts from various news sites and their corresponding summaries. The process of creation of this dataset is similar to the process of creation of the Multi-news dataset [2]. Then we used the PEGASUS model, pretrained on the Multi-news dataset to train in our dataset.

## How to run the project
If you want to train the PEGASUS model, the dataset is provided in the file Dataset. The codes for the creation of the dataset are not provided.
[+++]

## Example results

## Contributors:
Reseacrh and implementation of the AI part made by Deep Belief
Data extraction from URLs were made from Goxr3plus


## Reference
[1] Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020, November). Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning (pp. 11328-11339). PMLR.

[2] Fabbri, A. R., Li, I., She, T., Li, S., & Radev, D. R. (2019). Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. arXiv preprint arXiv:1906.01749.
[2]
