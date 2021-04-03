import re

from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
t = Tokenizer(wakati=True)


def clean_html(html, strip=False):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=strip)
    return text


def remove_url(html):
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.findAll('a'):
        a.replaceWithChildren()
    return str(soup)


def tokenize(text):
    return t.tokenize(text)


def normalize_number(text, reduce=False):
    if reduce:
        normalized_text = re.sub(r'\d+', '0', text)
    else:
        normalized_text = re.sub(r'\d', '0', text)
    return normalized_text
