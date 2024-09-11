import glob
import pandas as pd
from typing import List
from openai import OpenAI, BadRequestError
import os, sys, logging, time, json
import numpy as np
import re
from utils.search_wiki import search_wiki
from tqdm import tqdm

def get_openai_key():
    openai_tok = os.environ.get("OPENAI_API_KEY")
    assert openai_tok and openai_tok != "<openai_token>", "OpenAI token is not defined"
    return openai_tok.strip()
def flatten_hallucinations(hallucinations):
    flat_list = []
    for hallucination in hallucinations:
        if not hallucination:
            hal = []
        else:
            hal = [h['atom'] for h in hallucination]
        flat_list.append(hal)
    return flat_list
def regenerate_text(generations, hallucinations):
    print ("regenerating text")
    regenerations = []
    num_rate_errors = 0
    openai_client = OpenAI(api_key=get_openai_key())
    prompts = [(
f"Your task is to remove factually incorrect information from the provided text. Do not include any new information, simple remove the list of atomic facts from the given text."
                   f"\n Text: {g} \n\n Atomic Facts to remove: {h} \n New Text: ") for g,h in zip(generations,hallucinations)]
    for prompt in tqdm(prompts):
        received = False
        while not received:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0.2,
                )
                received = True
                topic = response.choices[0].message.content
                regenerations.append(topic)

            except:
                regenerations.append("")
                error = sys.exc_info()[0]
                num_rate_errors += 1
                if error == BadRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                logging.error("API error: %s (%d)" % (error, num_rate_errors))

    return regenerations
def get_wiki_topic(query:list) -> List:
    response = None
    wiki_topics = []
    num_rate_errors = 0

    openai_client = OpenAI(api_key=get_openai_key())
    prompts = [(f"Your task it to Map the provided text to an existing wikipedia article that describes it the best. Please only output the name of the article name make sure that the word is a valid wikipedia entry."
                f"\n Text: {q} \n Wikipedia article name: ") for q in query]
    for prompt in prompts:
        received = False
        while not received:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=10,
                    temperature=0.2,
                )
                received = True
                topic = response.choices[0].message.content
                wiki_topics.append(topic)

            except:
                wiki_topics.append("")
                error = sys.exc_info()[0]
                num_rate_errors += 1
                if error == BadRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                logging.error("API error: %s (%d)" % (error, num_rate_errors))

    wiki_topics = [wiki_topic.replace ('-', " ") for wiki_topic in wiki_topics]
    wiki_topics = [wiki_topic.replace('_', " ") for wiki_topic in wiki_topics]

    return wiki_topics

def csv_to_jsonl_for_factscore(results_dir):

    # get a list of all logs
    extension = 'csv'
    runs = glob.glob('{}/*.{}'.format(results_dir, extension))

    jsonl_paths = []
    for run in runs:
        path_d = run.replace('.csv', '.jsonl')
        jsonl_paths.append(path_d)

        df = pd.read_csv(run) #, encoding='ISO-8859-1')
        df = df.head(1)
        #df = df.loc[[3,18]]

        # add columns required by factscore
        df["topic"] = get_wiki_topic(df["prediction-summary"])
        #df["topic"] = search_wiki(df["prediction-summary"])
        df["cat"] = [[] for i in range(len(df))]

        # only keep the columns needed for factscore
        df = df[[ "article", "prediction-summary", "topic","cat"]]
        df.columns = ["input", "output", "topic", "cat"]


        #convert each row to a dict and write as a jsonl
        df_jsonl = df.to_dict('records')
        with open(path_d, 'w') as out:
            for ddict in df_jsonl:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)

    return jsonl_paths

#test_path = "/home/ubuntu/juraj/results/csvs/test/"
#json_paths = csv_to_jsonl_for_factscore(test_path)
#print(json_paths)