# Automatic Article Generator

In this project we use AI to automatically compose text related to natural disasters from various news sites as sourse. This project is part of my contribution to the SnapEarth project when I worked for CERTH. The generated text contains information about a natural disaster and can be used as part of an article.

## Project architecture
![Project architecture](https://user-images.githubusercontent.com/46052843/172661217-11814181-2332-4ce7-a0ac-e413c01a8016.png)
As shown in the architecture image, the user will be able to provide a list of URLs that can correspond to articles. Then the text from these URLs is extracted and fed into the next module where the text composition is made. In this module the concatenation of the texts is made and then the concatenated text is used as input to the model were a new text is generated.

## Research 
More specifically the task of this project is Multi-document summarization where we try to generate a summary from multiple texts in contrast with the single document summarization when we try to generate a summary from a single document. In this project the generated texts are more domain-specific and more specifically related to natural disasters news, so we choose the PEGASUS [1] model because it shows the best performance in the task of multi-document summarization for the Multi-news dataset [2] which is close to our domain. The Multi-news dataset is a collection of news articles from various sites with their corresponding summaries. 

## Datasets and training
Because we wanted to make our model generate more domain-specific texts (Texts related to natural disasters news) we created a new dataset that includes texts from various news sites and their corresponding summaries. The process of creation of this dataset is similar to the process of creation of the Multi-news dataset [2]. Then we used the PEGASUS model, pretrained on the Multi-news dataset to train in our dataset.  

## How to execute the codes
If you want to train the PEGASUS model, you can execute the train_pegasus_disaster.py. The dataset to train the model is provided in the Dataset folder.
The Dataset is already splited in train and test set. Before train the model you can change a number of hyperparameters like the number of epochs, the batch size, 
the output lengh of the generated text etc

If you want to generate text from the you can provide a list of URLs from news sites related to specific event like fire/flood/volcano etc. The agruements that should be provided are the path for the model and the max lengh of the generated text. Feel free to play with diffent max lenghs. 

The file of the model is very big and you can find it in this link: https://drive.google.com/drive/folders/15zwW1V5MMrpBIPutdXZ8mOpiwbXIrw26?usp=sharing

I number of decoding parameters can also be changed from the user.

## Example results

## Contributors:
Reseacrh and implementation of the AI part made by Deep Belief



## Reference
[1] Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020, November). Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning (pp. 11328-11339). PMLR.

[2] Fabbri, A. R., Li, I., She, T., Li, S., & Radev, D. R. (2019). Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. arXiv preprint arXiv:1906.01749.
[2]
