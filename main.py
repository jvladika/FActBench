import argparse
import json
import logging
from typing import List, Optional
import sys
from FActScore.factscore.factscorer import FactScorer
from utils.wandb_utils import wandb_init_run, wandb_push_json, wandb_push_table
from utils.fscore_utils import csv_to_jsonl_for_factscore
from datetime import datetime
import nltk
import numpy as np
import os
nltk.download('punkt_tab')
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
                    if decision_before[idx][af["idx"]]['is_supported'] != af["is_supported"]:
                        print(f"Updating the decision for the Atomic Fact: {af} for sample {idx}")
                        decision_before[idx][af["idx"]]['is_supported'] = af["is_supported"]
                        count += 1
        scores = [np.mean([d["is_supported"]  for d in decisions]) for decisions in decision_before]
        hallucinations = [[d for d in decisions if not d["is_supported"]] for decisions in decision_before]

        updated_score = np.mean(scores)
        logging.critical("FActScore After extrinsic check = %.1f%%" % (100 * updated_score))
        logging.critical(f"Updated decision on {str(count)} Facts after running Extrinsic check")
        #if "init_score" in extrinsic_out:
        #    logging.critical("FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"]))
        #logging.critical("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
        #logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))
        return updated_score, hallucinations


class DebertaNli:
    def __init__(self, score_out, decisions, groundings):

        self.score_out = score_out
        self.decisions = decisions
        self.groundings = groundings

        self.model_name = "tasksource/deberta-base-long-nli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)


    def run_nli(self,) -> dict:
        nli_decisions = list()

        for data_instance, article in zip(self.decisions, self.groundings):
            nli_results = list()

            for atom_instance in data_instance:
                atom_fact = atom_instance["atom"]

                premise = atom_fact
                hypothesis = article

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
                nli_results.append(nli_class)

            nli_decisions.append(nli_results)
        
        new_decisions = list()
        for decision, nli_prediction in zip(self.decisions, nli_decisions):
            new_list = list()
            for dec, pred in zip(decision, nli_prediction):
                dec["nli_prediction"] = pred
                new_list.append(dec)

            new_decisions.append(new_list)
        
        self.score_out["decisions"] = new_decisions

        return self.score_out


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
                        default="retrieval+ChatGPT")
    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--grounding_provided',
                        type=bool,
                        default=False)
    args = parser.parse_args()


    csv_results_dir = "results/"
    jsonl_paths = csv_to_jsonl_for_factscore(csv_results_dir)


    genFact = GenFact(args)
    #wandb_init_run(run_path=args.input_path, config = genFact.args)


    print ("Running Vanilla FactScore")
    factscore_out_vanilla = genFact.run_factscrorer(grounding_provided = False )
    genFact.write_logs(factscore_out_vanilla, fname="factscore_vanilla.json")



    print ("Running Factscore with grounded document")
    factscore_out = genFact.run_factscrorer(grounding_provided=args.grounding_provided)
    genFact.write_logs(factscore_out, fname="factscore_grounded.json")


    fs_extrinsic_af = genFact.fs_get_extrinsic_af(topics = factscore_out["topics"], wrong_facts = factscore_out["wrong_facts"],
                groundings=  factscore_out["groundings"], generations = factscore_out["generations"],
                                            grounding_provided= factscore_out["grounding_provided"])
    fs_extrinsic_out = genFact.fs_extrinsic_score(fs_extrinsic_af)
    genFact.write_logs(factscore_out, fname="factscore_grounded_extrinsic.json")


    updated_score, updated_wrong_facts = genFact.get_updated_score(factscore_out,fs_extrinsic_out)

    #WANDB part was giving me errors so I temporarily commented it out.
    ''' 
    wandb_table = {"fs_wiki": factscore_out_vanilla["score"], "fs_grounded": factscore_out["score"],
                   "fs_grounded_wiki": updated_score}
    wandb_push_json(wandb_table)

    wandb_table = {"generations": factscore_out["generations"], "hallucinations": updated_wrong_facts }
    wandb_push_table(wandb_table)
    '''

    #Creates new class for deberta predictions. Loads a model from HuggingFace.
    deberta_nli = DebertaNli(score_out = factscore_out,
                             decisions =  factscore_out["decisions"],
                              groundings = factscore_out["groundings"])
    
    #Output is the same as factscore_out, but with a new attribute in dictionaries with NLI predictions. 
    deberta_out = deberta_nli.run_nli()

    #deberta_extrinsic_out = fs_extrinsic_af












        


