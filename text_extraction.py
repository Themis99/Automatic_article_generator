from newspaper import Article

def take_text(ulr):
    article = Article(ulr)
    article.download()
    article.parse()
    article.text


    return article.text.replace("\n", "")

