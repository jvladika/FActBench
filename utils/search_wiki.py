import requests
import time
import spacy
from typing import List


def search_wiki(claims: str):
    claims = [claims]

    nlp = spacy.load("en_core_sci_scibert")

    all_titles = list()
    all_snippets = list()

    idx = 0
    for claim in claims:

        text = claim[:300]

        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={text}&format=json"
            response = requests.get(url)
            data = response.json()
            search_results = data['query']['search']

            if len(search_results) <= 0:
                doc = nlp(text)
                ents = list(doc.ents)
                raise Exception("")

        except:
            try:
                doc = nlp(text)
                ents = list(doc.ents)
                new_query = ""
                for e in ents:
                    e = e.text
                    new_query += e
                    new_query += " "

                try:
                    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={new_query}&format=json"
                    response = requests.get(url)
                    data = response.json()
                    search_results = data['query']['search']
                except:
                    search_results = []

                if len(search_results) <= 0:
                    for e in ents:
                        e = e.text
                        new_query = e
                        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={new_query}&format=json"
                        response = requests.get(url)
                        data = response.json()
                        search_results.extend(data['query']['search'][:2])

            except:
                return list()

        titles = list()
        snippets = list()
        for result in search_results:
            title = result['title']
            titles.append(title)

            snippet = result['snippet']
            snippets.append(snippet)

        # print(claim, titles)

        all_titles.append(titles)
        all_snippets.append(snippets)
        idx += 1

    return titles

# claims = ["your list of atomic facts", "test"]
# search_wiki(claims)