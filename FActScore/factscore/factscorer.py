import argparse
import string
import json
import numpy as np
import os
import logging
import sys
from tqdm import tqdm
from FActScore.factscore.abstain_detection import is_response_abstained
from FActScore.factscore.atomic_facts import AtomicFactGenerator
from FActScore.factscore.clm import CLM
from FActScore.factscore.npm import NPM
from FActScore.factscore.openai_lm import OpenAIModel
from FActScore.factscore.retrieval import DocDB, Retrieval
from typing import List, Optional
from utils.search_wiki import search_wiki
from utils.fscore_utils import get_wiki_topic

class FactScorer(object):

    def __init__(self,
                 model_name="GPT4-mini",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 grounding_provided=False,
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm",
                              "GPT4-mini"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type
        self.grounding_provided = grounding_provided

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            pass 

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "GPT4" in model_name:
            self.lm = OpenAIModel("gpt-4o-mini",
                                  cache_file=os.path.join(cache_dir, "GPT4o-mini.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        #assert name not in self.retrieval, f"{name} already registered"
        if name not in self.retrieval:
            if db_path is None:
                db_path = os.path.join(self.data_dir, f"{name}.db")

            if data_path is None:
                data_path = os.path.join(self.data_dir, f"{name}.jsonl")

            cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

            print (db_path)
            self.db[name] = DocDB(db_path=db_path, data_path=data_path)
            self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
            if "npm" in self.model_name:
                cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
                embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
                self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                     "npm-single",
                                     cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        # @todo: Anum update this
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  topics,
                  generations,
                  groundings,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False,
                  grounding_provided=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "GPT4o-mini.pkl"))

            # estimate the total cost of atomic fact generation
            #total_words = 0
            #for gen in generations:
            #    total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            #self.print_cost_estimates(total_words, task="atomic fact generation", model="GPT-4o mini")

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            print ("Generating Atomic Facts.")
            for topic, gen in tqdm(zip(topics, generations),total = len(generations)):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                if isinstance(gen,str) == False:
                    gen = ''
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        # todo: update this
        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts, grounding in zip(topics, generations, atomic_facts, groundings):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source,
                                                   cost_estimate=self.cost_estimate, grounding=grounding,
                                                   grounding_provided=grounding_provided)

            #self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        wrong_facts = []

        print("Getting score from FS.")
        for topic, generation, facts, grounding in tqdm(zip(topics, generations, atomic_facts, groundings), total=len(generations)):
            #print (f"Running for the follow facts. {facts}")
            if facts is None:
                decisions.append(None)
                wrong_facts.append([])
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source, grounding=grounding,
                                           grounding_provided=grounding_provided)
                score = np.mean([d["is_supported"] for d in decision])
                wrong_fact = [{"atom":d["atom"], "idx":idx}  for idx, d in enumerate(decision) if not d["is_supported"]]

                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                wrong_facts.append(wrong_fact)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "wrong_facts": wrong_facts,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def get_extrinsic_af(self, topics, wrong_facts, groundings, generations = None,  verbose=False, grounding_provided=False):
        if verbose:
            topics = tqdm(topics)

        scores = []
        decisions = []
        extrinsic_facts = []

        for topic, generation, facts, grounding in tqdm(zip(topics, generations, wrong_facts, groundings),total = len(generations)):
            #print(f"Running for the follow wrongly classified facts to check if they are intrinsic or extrinsic. {facts}")
            if facts is None:
                decisions.append(None)
                extrinsic_facts.append([])
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source=None, grounding=grounding,
                                           grounding_provided=grounding_provided, check_extrinsic=True)
                score = np.mean([d["is_supported"] for d in decision])
                extrinsic_fact = [{"atom":d["atom"], "idx":d["idx"]}  for idx, d in enumerate(decision) if not d["is_supported"]]


                decisions.append(decision)
                scores.append(score)
                extrinsic_facts.append(extrinsic_fact)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        extrinsic_af = {"score": np.mean(scores),
               "decisions": decisions,
               "extrinsic_facts": extrinsic_facts,
               }
        #print ("The following wrongly classified facts are Extrinsic: \n {}".format(extrinsic_facts))

        return extrinsic_af

    def get_extrinsic_score(self, topics, extrinsic_facts, generations = None,  verbose=False, grounding_provided=False):
        if verbose:
            topics = tqdm(topics)

        knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)
        scores = []
        decisions = []
        extrinsic_hallucinated_facts = []


        for topic, generation, facts in tqdm(zip(topics, generations, extrinsic_facts), total = len(generations)):
            #print(f"Checking the wrongly classified text through extrinsic fact checking using knowledge source {knowledge_source} for the following facts. \n {facts}")
            if facts is None:
                decisions.append(None)
            else:

                decision = self._get_score(topic, generation, facts, knowledge_source=knowledge_source,
                                           grounding_provided=grounding_provided, check_extrinsic=False, get_topic_per_af = True)
                score = np.mean([d["is_supported"] for d in decision])
                extrinsic_hallucinated_fact = [{"atom":d["atom"], "idx":d["idx"]}  for idx, d in enumerate(decision) if not d["is_supported"]]


                decisions.append(decision)
                scores.append(score)
                extrinsic_hallucinated_facts.append(extrinsic_hallucinated_fact)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        self.extrinsic_out = {"score": np.mean(scores),
               #"respond_ratio": respond_ratio,
               "decisions": decisions,
               "wrong_facts": extrinsic_hallucinated_facts,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        #print ("The following facts are still classified as hallucinations after Extrinsic Fact Checking: \n {}".format(extrinsic_facts))
        return self.extrinsic_out


    def search_passage_till_success(self, topic, atom, generation, knowledge_source) -> List:

        try:
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            #print ("got passage using  provided topic")
        except:
            # try all suggested topics until a match is found
            #print("didn't get passage using  provided topic")
            topics = search_wiki(generation)
            if not topic:
                # if we are in the get topic per AF mode. The LLM generated topic would be tested first.
                llm_generate_topic = get_wiki_topic([atom])[0]
                topics.insert(0,llm_generate_topic)

            success = False
            idx = 0
            #print ( f"trying out {len(topics)} topics from search_Wiki function.")
            #if len(topics) == 1:
            #    print ("Problem found")
            #    print ("Generation: ", generation)
            #    print ("Topics:", topics)
            while (success == False):
                if idx >= len(topics):
                    print (f"Exhausted all options for this generation, no DB topic match found! Generation: {generation} \n\n Assigning dummpy empty passage")
                    return []
                try:
                    topic = topics[idx]
                    passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=3)
                    idx += 1
                    success = True
                    #print (f"topic found: {topic}")
                except :
                    #print (e)
                    idx += 1
        return passages

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, grounding = None, grounding_provided=False,
                   cost_estimate=None, check_extrinsic = False, get_topic_per_af = False):
        decisions = []
        always_use_same_passage = False
        passages = None
        total_words = 0
        for atomic_fact in atomic_facts:
            if not isinstance(atomic_fact, str):
                atom = atomic_fact["atom"]
                idx = atomic_fact["idx"]
            else:
                atom = atomic_fact
                idx = None
            atom = atom.strip()
            if self.lm:
                if grounding_provided:
                    if isinstance(grounding, str):
                        grounding = [grounding]
                    passages = [{'title': 'Article', 'text': ground} for ground in grounding]
                elif always_use_same_passage and passages is not None:
                    passages = passages

                elif get_topic_per_af:
                    # passing empty topic would force the retriever the get a llm generated topic
                    passages = self.search_passage_till_success(topic = '', atom = atom, generation=atom,
                                                       knowledge_source=knowledge_source)
                else:
                    passages = self.search_passage_till_success(topic, atom, generation, knowledge_source)
                    always_use_same_passage = True

                if check_extrinsic:
                    definition = "Does the provided text contain any information related to the Input statement?.\n\n"
                else:
                    definition = "Answer the question about {} based on the given context.\n\n".format(topic)

                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."

                if check_extrinsic:
                    definition.replace("Title: Article", "")
                    prompt = "{}\n\nInput: {} \n This statement is discussed in the above provided text. True or False?\nOutput:".format(definition.strip(), atom.strip())
                else:
                    prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue

                output = self.lm.generate(prompt)

                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3

            decisions.append({"atom": atom, "is_supported": is_supported, "idx": idx, "wiki_context": context})


        if cost_estimate:
            return total_words
        else:
            return decisions

if __name__ == '__main__':

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
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
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

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot==args.n_samples:
                break
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # Save out as a json file
    with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
        f.write(json.dumps(out) + "\n")

