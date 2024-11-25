import argparse
import json
import logging
from typing import List, Optional
import sys
from FActScore.factscore.factscorer import FactScorer
from utils.wandb_utils import wandb_init_run, wandb_push_json, wandb_push_table
from utils.fscore_utils import csv_to_jsonl_for_factscore, regenerate_text, flatten_hallucinations
from datetime import datetime
import nltk
nltk.download('punkt_tab')
from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


fs_logs_available = {
    "human-eval": "folder",
}

class GenFact:
    def __init__(self, args: Optional[List[str]] = None):

        self.args = parse_options(sys.argv[1:] if args is None else args)

        logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.ERROR if self.args.print_rate_limit_error else logging.CRITICAL)

        self.log_dir = self.create_log_folder()
        self.fs = FactScorer(model_name=self.args.model_name,
                        data_dir=self.args.data_dir,
                        model_dir=self.args.model_dir,
                        cache_dir=self.args.cache_dir,
                        openai_key=self.args.openai_key,
                        cost_estimate=self.args.cost_estimate,
                        abstain_detection_type=self.args.abstain_detection_type,
                        grounding_provided=self.args.grounding_provided)

    def add_item(self, item) -> None:
        self.factscore_logs = item
    def run_factscrorer(self, grounding_provided:bool) -> dict:

        tot = 0
        topics, generations, atomic_facts, groundings = [], [], [], []
        with open(self.args.input_path) as f:
            for line in f:
                dp = json.loads(line)
                tot += 1
                if self.args.use_atomic_facts:
                    assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                    if dp["annotations"] is None:
                        continue
                    topics.append(dp["topic"])
                    generations.append(dp["output"])
                    atomic_facts.append(
                        [atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
                else:
                    topics.append(dp["topic"])
                    generations.append(dp["output"])
                if self.args.grounding_provided:
                    groundings.append(dp["input"])

                if self.args.n_samples is not None and tot == args.n_samples:
                    break
        topics = topics[:2]
        generations = generations[:2]
        groundings = groundings[:2]
        out = self.fs.get_score(topics=topics,
                           generations=generations,
                           groundings=groundings,
                           gamma=self.args.gamma,
                           atomic_facts=atomic_facts if self.args.use_atomic_facts else None,
                           knowledge_source=self.args.knowledge_source,
                           verbose=self.args.verbose,
                           grounding_provided=grounding_provided)
        print ("Using intrinsic Fact Checking")
        logging.critical("FActScore = %.1f%%" % (100 * out["score"]))
        if "init_score" in out:
            logging.critical("FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"]))
        logging.critical("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
        logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

        # Save out as a json file
        with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
            f.write(json.dumps(out) + "\n")

        self.factscore_logs = {"score": out["score"],"topics": topics, "decisions": out["decisions"], "wrong_facts": out["wrong_facts"], "groundings": groundings,
                               "generations": generations, "grounding_provided": grounding_provided}

        return self.factscore_logs

    def write_logs(self, out:json, fname:str):
        fname = os.path.join(self.log_dir,fname)
        with open(fname, 'w') as fp:
            json.dump(out, fp)

    def read_logs(self, fs_cache_dir:str)-> tuple:

        fs_cache_dir = "results/genfact/humaneval_summarization/folder/"
        #with open(os.path.join(fs_cache_dir, "factscore_vanilla.json")) as f:
        #    factscore_out_vanilla = json.load(f)
        factscore_out_vanilla = {'score': 0}

        with open(os.path.join(fs_cache_dir, "factscore_grounded.json")) as f:
            factscore_out = json.load(f)

        with open(os.path.join(fs_cache_dir, "deberta_grounded.json")) as f:
            deberta_out = json.load(f)

        with open(os.path.join(fs_cache_dir, "factscore_grounded_extrinsic.json")) as f:
            fs_extrinsic_out = json.load(f)
        #fs_extrinsic_out = dict()

        with open(os.path.join(fs_cache_dir, "deberta_grounded_extrinsic.json")) as f:
            db_extrinsic_out = json.load(f)

        return (factscore_out_vanilla, factscore_out, fs_extrinsic_out, deberta_out, db_extrinsic_out)


    def fs_get_extrinsic_af(self, topics, wrong_facts, groundings,generations, grounding_provided):
        # Check if the wrongly classified facts are "wrong" or just not present in the article.
        extrinsic_af = self.fs.get_extrinsic_af(topics=topics, wrong_facts=wrong_facts, groundings=groundings,
                                                  generations=generations, grounding_provided=grounding_provided)


        return extrinsic_af
    def fs_extrinsic_score(self,fs_extrinsic_af:dict):
        extrinsic_out = self.fs.get_extrinsic_score(topics = self.factscore_logs["topics"], extrinsic_facts=fs_extrinsic_af["extrinsic_facts"],
                                                    generations = self.factscore_logs["generations"],  verbose=False,
                                                    grounding_provided=False)
        return extrinsic_out

    def create_log_folder(self, ):
        date_time =  '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.now() )
        run_name = os.path.basename(self.args.input_path).replace('.jsonl','')

        folder = os.path.join("results","genfact", run_name, date_time)
        os.makedirs(folder, exist_ok=True)
        print(f"Run outputs would be locally stored at {folder}")
        return folder

    def get_updated_score(self, factscore_out, fs_extrinsic_af) ->float:
        decision_before = factscore_out["decisions"]
        decision_after = fs_extrinsic_af["decisions"]
        count = 0


        for idx, afs in enumerate(decision_after):
            if len(afs) > 0:
                for af in afs:
                    try:
                        if decision_before[idx][af["idx"]]['is_supported'] != af["is_supported"]:
                            #print(f"Updating the decision for the Atomic Fact: {af} for sample {idx}")
                            decision_before[idx][af["idx"]]['is_supported'] = af["is_supported"]
                    except Exception as e:
                        print (e)
                        print(f"Failed to Update the decision for the Atomic Fact: {af['atom']} for sample {idx}")
                        count += 1
        scores = [np.mean([d["is_supported"] for d in decisions if d is not None]) for decisions in decision_before if decisions is not None]
        hallucinations = [[d for d in decisions if not d["is_supported"]] for decisions in decision_before if decisions is not None]

        updated_score = np.mean(scores)
        logging.critical("FActScore After extrinsic check = %.1f%%" % (100 * updated_score))
        logging.critical(f"Updated decision on {str(count)} Facts after running Extrinsic check")
        #if "init_score" in extrinsic_out:
        #    logging.critical("FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"]))
        #logging.critical("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
        #logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))
        return updated_score, hallucinations


class DebertaNli:
    def __init__(self, score_out, decisions, groundings, fs):
        new_decisions = list()
        for dec in decisions:
            if dec is None:
                new_decisions.append([{'atom': 'This is a statement.', 'is_supported': True, 'idx': None, 'wiki_context': 'Title: Article\nText: The statement is true. \n'}])
            else:
                new_decisions.append(dec)

        self.decisions = new_decisions 
        self.score_out = score_out
        self.groundings = groundings
        self.fs = fs
        self.fs.register_knowledge_source()

        self.model_name = "tasksource/deberta-base-long-nli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)

    def get_nli_class(self, premise, hypothesis):
        model_input = self.tokenizer(premise, hypothesis, truncation=True, max_length=1024, return_tensors="pt")
        model_output = self.model(model_input["input_ids"].to(device)) 
        prediction_probs = torch.softmax(model_output["logits"][0], -1).tolist()
        prediction_probs = np.array(prediction_probs)

        max_index = np.argmax(prediction_probs)
        if  max_index == 0:
            nli_class = "entailment"
        elif max_index == 1:
            nli_class = "neutral"
        elif max_index == 2:
            nli_class = "contradiction"

        return nli_class


    def check_intrinsic(self,) -> dict:
        nli_decisions = list()
        deberta_scores = list()
        
        print("Deberta checking intrinsic facts.")
        for data_instance, article in tqdm(zip(self.decisions, self.groundings), total=len(self.decisions)):
            nli_results = list()

            for atom_instance in data_instance:
                atom_fact = atom_instance["atom"]
                premise = atom_fact
                hypothesis = article

                nli_class = self.get_nli_class(premise, hypothesis)
                nli_results.append(nli_class)

            nli_decisions.append(nli_results)
        
        new_decisions = list()
        wrong_facts = list()

        print("Intrinsic check done. Getting the wiki articles before extrinsic checking.")
        for decision, nli_prediction in tqdm(zip(self.decisions, nli_decisions), total=len(nli_decisions)):
            new_list = list()
            for dec, pred in zip(decision, nli_prediction):
                dec["nli_intrinsic"] = pred
                dec["nli_supported_intrinsic"] = True if pred == "entailment" else False
                new_list.append(dec)


            debscore = np.mean([d["nli_supported_intrinsic"] for d in decision])
            deberta_scores.append(debscore)
            new_decisions.append(new_list)

            #wfs = [{"atom":d["atom"], "idx":idx}  for idx, d in enumerate(new_list) if not d["nli_supported_intrinsic"]]
            wfs = [{"atom":d["atom"], "idx":idx}  for idx, d in enumerate(new_list) if d["nli_intrinsic"] == "contradiction"]
            wrong_facts.append(wfs)

            for d in new_list:
                #if not d["nli_supported_intrinsic"]:
                if d["nli_intrinsic"] == "contradiction":
                    passages = self.fs.search_passage_till_success(topic = '', atom = d["atom"], generation=d["atom"],
                                                               knowledge_source="enwiki-20230401")
                    context = ""
                    for psg_idx, psg in enumerate(reversed(passages)):
                        context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                    context = context.strip()

                    d["wiki_context"] = context


        self.decisions = new_decisions            
        self.score_out["decisions"] = new_decisions

        self.score_out["deberta_score_intrinsic"] = np.mean(deberta_scores)
        self.score_out["deberta_wrong_facts"] = wrong_facts

        logging.critical("Deberta Score (intrinsic) = %.1f%%" % (100 * np.mean(deberta_scores)))

        return self.score_out, np.mean(deberta_scores)
    

    def check_extrinsic(self, wrong_facts) -> dict:
        nli_decisions = list()
        deberta_scores = list()

        print("Deberta checking extrinsic facts.")
        for data_instance, wf in tqdm(zip(self.decisions, wrong_facts), total=len(self.decisions)):
            nli_results = list()

            fact_idx = 0
            wf_indices = [w["idx"] for w in wf]

            for atom_instance in data_instance:
                if fact_idx not in wf_indices:
                    nli_results.append(("none", fact_idx))
                    fact_idx += 1
                    continue

                atom_fact = atom_instance["atom"]
                atom_wiki_context = atom_instance["wiki_context"]

                premise = atom_fact
                hypothesis = atom_wiki_context

                model_input = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
                model_output = self.model(model_input["input_ids"].to(device)) 
                prediction_probs = torch.softmax(model_output["logits"][0], -1).tolist()
                prediction_probs = np.array(prediction_probs)

                max_index = np.argmax(prediction_probs)
                if  max_index == 0:
                    nli_class = "entailment"
                elif max_index == 1:
                    nli_class = "neutral"
                elif max_index == 2:
                    nli_class = "contradiction"
                nli_results.append((nli_class, fact_idx))

                fact_idx += 1

            nli_decisions.append(nli_results)

        new_decisions = list()
        for decision, nli_prediction in zip(self.decisions, nli_decisions):
            new_list = list()

            atom_dec_idx = 0
            wrong_fact_indices = [n[1] for n in nli_prediction if n[0] != "none"]
            for dec, pred in zip(decision, nli_prediction):

                if atom_dec_idx in wrong_fact_indices:
                    dec["nli_extrinsic"] = pred[0]
                    dec["nli_supported_extrinsic"] = True if pred[0] == "entailment" else False
                    dec["nli_final_supported"] = dec["nli_supported_extrinsic"]
                    new_list.append(dec)
                else:
                    dec["nli_final_supported"] = dec["nli_supported_intrinsic"]
                    new_list.append(dec)

                atom_dec_idx += 1

            #debscore = np.mean([d["nli_supported_extrinsic"] for d in decision])
            #deberta_scores.append(debscore)
            debscore = np.mean([d["nli_final_supported"] for d in new_list])
            deberta_scores.append(debscore)

            new_decisions.append(new_list)

        self.decisions = new_decisions            
        self.score_out["decisions"] = new_decisions
        self.score_out["deberta_score_final"] = np.mean(deberta_scores)
        logging.critical("Deberta score (final, after extrinsic) = %.1f%%" % (100 * np.mean(deberta_scores)))        

        return self.score_out, np.mean(deberta_scores)

def get_pooled_score(deberta_extrinsic_out, mode = None):
    decisions = deberta_extrinsic_out["decisions"]
    new_decisions = list()
    pooled_scores = list()

    for instance_decisions in decisions:
        new_list = list()
    
        for dec in instance_decisions:
            if mode == 'intrinsic':
                deberta_final = dec["nli_supported_intrinsic"]
            else:
                deberta_final = dec["nli_final_supported"]
            factscore_final = dec["is_supported"]

            #Pooled decision is True if both FactScore and NLI predictions are True.
            pooled_final = deberta_final and factscore_final
            dec["pooled_supported"] = pooled_final
            new_list.append(dec)

        poolscore = np.mean([d["pooled_supported"] for d in new_list])
        pooled_scores.append(poolscore)

        new_decisions.append(new_list)

    logging.critical("Pooled score (final) = %.1f%%" % (100 * np.mean(pooled_scores)))        

    return new_decisions, np.mean(pooled_scores)


def parse_options(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    #@todo: Fix these paths to be defined from the majorityvoter or command line and revert this to the original factscore code stucture
    parser.add_argument('--data_dir',
                        type=str,
                        default="./FActScore/.cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default="./FActScore/.cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default="./FActScore/.cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)

    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)
    parser.add_argument('--grounding_provided',
                        type=bool,
                        default=False)

    args = parser.parse_args()
    return args




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="GPT4-mini")
    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--grounding_provided',
                        type=bool,
                        default=False)
    args = parser.parse_args()


    csv_results_dir = "results/"
    #jsonl_paths = csv_to_jsonl_for_factscore(csv_results_dir)


    genFact = GenFact(args)

    base_run_id = wandb_init_run(run_path=args.input_path, config = genFact.args)

    if base_run_id in fs_logs_available.keys():
        fs_cache_dir = os.path.dirname(genFact.log_dir)
        fs_cache_dir = os.path.join(fs_cache_dir,fs_logs_available[base_run_id])
        genFact.log_dir = fs_cache_dir

        factscore_out_vanilla, factscore_out, fs_extrinsic_out, deberta_out, db_extrinsic_out = genFact.read_logs(fs_cache_dir)
        genFact.add_item(factscore_out)
        print (f"UPDATE: Run outputs would be locally stored at the updated location:  {fs_cache_dir}")

        fs_updated_score, fs_updated_wrong_facts = genFact.get_updated_score(factscore_out, fs_extrinsic_out)

        wandb_table = {"fs_wiki": factscore_out_vanilla["score"], "fs_grounded": factscore_out["score"],
                       "fs_grounded_wiki": fs_updated_score}
        pooled_decisions, pooled_score = get_pooled_score(db_extrinsic_out)
        wandb_push_json(wandb_table)


    else:

        print ("Running Vanilla FactScore")
        factscore_out_vanilla = genFact.run_factscrorer(grounding_provided = False )
        genFact.write_logs(factscore_out_vanilla, fname="factscore_vanilla.json")
        #factscore_out_vanilla = {"score":0}

        print ("Running Factscore with grounded document: Intrinsic")
        factscore_out = genFact.run_factscrorer(grounding_provided=args.grounding_provided)
        genFact.write_logs(factscore_out, fname="factscore_grounded.json")

        print("Running Factscore with grounded document: Extrinsic")
        fs_extrinsic_af = genFact.fs_get_extrinsic_af(topics = factscore_out["topics"], wrong_facts = factscore_out["wrong_facts"],
                    groundings=  factscore_out["groundings"], generations = factscore_out["generations"],
                                                grounding_provided= factscore_out["grounding_provided"])
        fs_extrinsic_out = genFact.fs_extrinsic_score(fs_extrinsic_af)
        genFact.write_logs(fs_extrinsic_out, fname="factscore_grounded_extrinsic.json")


        fs_updated_score, fs_updated_wrong_facts = genFact.get_updated_score(factscore_out,fs_extrinsic_out)
        wandb_table = {"fs_wiki": factscore_out_vanilla["score"], "fs_grounded": factscore_out["score"],
                       "fs_grounded_wiki": fs_updated_score, "dummy_value": 0}
        wandb_push_json(wandb_table, table_name="fs_table")



    #Creates new class for deberta predictions. Loads a model from HuggingFace.

    deberta_nli = DebertaNli(score_out = factscore_out,
                             decisions =  factscore_out["decisions"],
                              groundings = factscore_out["groundings"],
                              fs = genFact.fs)
    
    print("Loaded Deberta.")
    #Output is the same as factscore_out, but with a new attribute in dictionaries with NLI predictions. 
    #Gives intrinsic NLI score.
    deberta_out, deberta_intrinsic_score = deberta_nli.check_intrinsic()
    genFact.write_logs(deberta_out, fname="deberta_grounded.json")

    print("Deberta intrinsic done.\n\n")
    # Calculates the final pooled prediction (inside of pooled_decisions) and final pooled score.
    db_out_pooled_decisions, db_out_pooled_score = get_pooled_score(deberta_out, mode='intrinsic')
    genFact.add_item(db_out_pooled_decisions)

    #Checks the wrong facts with extrinsic checking over Wikipedia. Gives final NLI score.
    deberta_nli.score_out = deberta_out    
    deberta_extrinsic_out, deberta_final_score = deberta_nli.check_extrinsic(factscore_out["deberta_wrong_facts"])
    genFact.write_logs(deberta_extrinsic_out, fname="deberta_grounded_extrinsic.json")

    print("Deberta extrinsic done.\n\n")

    #Calculates the final pooled prediction (inside of pooled_decisions) and final pooled score.
    pooled_decisions, pooled_score = get_pooled_score(deberta_extrinsic_out)

    deberta_score_dict = {
                   "deberta_grounded": factscore_out["deberta_score_intrinsic"], "intrinsic_pooled_score":db_out_pooled_score,
                "deberta_grounded_wiki": deberta_final_score,"extrinsic_pooled_score":pooled_score}
    wandb_push_json(deberta_score_dict, table_name="db_table")

    db_regenerations = regenerate_text(factscore_out["generations"], flatten_hallucinations(fs_updated_wrong_facts))
    #db_regenerations = ['']*len(factscore_out["generations"])
    halucinations = ['\n'.join(hal) for hal in flatten_hallucinations(fs_updated_wrong_facts)]
    wandb_table = {"generations": factscore_out["generations"], "fs_hallucinations": halucinations,
                    'regenerations':db_regenerations}
    wandb_push_table(wandb_table)

    print("done")

