import argparse
import json
import logging
from typing import List, Optional
import sys
from FActScore.factscore.factscorer import FactScorer
from utils.wandb_utils import wandb_init_run
from utils.fscore_utils import csv_to_jsonl_for_factscore
import nltk
nltk.download('punkt_tab')
from dotenv import load_dotenv
load_dotenv()
class GenFact:
    def __init__(self, args: Optional[List[str]] = None):

        self.args = parse_options(sys.argv[1:] if args is None else args)

        logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.ERROR if self.args.print_rate_limit_error else logging.CRITICAL)

        self.fs = FactScorer(model_name=self.args.model_name,
                        data_dir=self.args.data_dir,
                        model_dir=self.args.model_dir,
                        cache_dir=self.args.cache_dir,
                        openai_key=self.args.openai_key,
                        cost_estimate=self.args.cost_estimate,
                        abstain_detection_type=self.args.abstain_detection_type,
                        grounding_provided=self.args.grounding_provided)

    def run_factscrorer(self,) -> dict:

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
                           grounding_provided=self.args.grounding_provided)
        logging.critical("FActScore = %.1f%%" % (100 * out["score"]))
        if "init_score" in out:
            logging.critical("FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"]))
        logging.critical("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
        logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

        # Save out as a json file
        with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
            f.write(json.dumps(out) + "\n")

        self.factscore_logs = {"topics": topics, "wrong_facts": out["wrong_facts"], "groundings": groundings,
                               "generations": generations, "grounding_provided": self.args.grounding_provided}
        return self.factscore_logs

    def fs_get_extrinsic_af(self, topics, wrong_facts, groundings,generations, grounding_provided):
        # Check if the wrongly classified facts are "wrong" or just not present in the article.
        extrinsic_af = self.fs.get_extrinsic_af(topics=topics, wrong_facts=wrong_facts, groundings=groundings,
                                                  generations=generations, grounding_provided=grounding_provided)

        #logging.critical("FActScore = %.1f%%" % (100 * extrinsic_out["score"]))
        #if "init_score" in extrinsic_out:
        #    logging.critical("FActScore w/o length penalty = %.1f%%" % (100 * out["init_score"]))
        #logging.critical("Respond ratio = %.1f%%" % (100 * out["respond_ratio"]))
        #logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))
        return extrinsic_af
    def fs_extrinsic_score(self,fs_extrinsic_af:dict):
        extrinsic_out = self.fs.get_extrinsic_score(topics = self.factscore_logs["topics"], extrinsic_facts=fs_extrinsic_af["extrinsic_facts"],
                                                    generations = self.factscore_logs["generations"],  verbose=False,
                                                    grounding_provided=False)
        return extrinsic_out

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
                        default="/Users/anumafzal/PycharmProjects/FactSumm/FActScore/.cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default="/Users/anumafzal/PycharmProjects/FactSumm/FActScore/.cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default="/Users/anumafzal/PycharmProjects/FactSumm/FActScore/.cache/factscore/")
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
    wandb_init_run(run_path=args.input_path, config = genFact.args)

    factscore_out = genFact.run_factscrorer()
    fs_extrinsic_af = genFact.fs_get_extrinsic_af(topics = factscore_out["topics"], wrong_facts = factscore_out["wrong_facts"],
                groundings=  factscore_out["groundings"], generations = factscore_out["generations"],
                                            grounding_provided= factscore_out["grounding_provided"])
    fs_extrinsic_out = genFact.fs_extrinsic_score(fs_extrinsic_af)


    deberta_out = factscore_out
    deberta_extrinsic_out = fs_extrinsic_af







        


