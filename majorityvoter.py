import json
import glob
import pandas as pd
import argparse
from FActScore.factscore.factscorer import main
class MajorityVoter:
    def __init__(self):
        self.results_dir = "results/"
        self.jsonl_paths = None
        self.factscore = None

    def prepare_for_factscore(self):

        # get a list of all logs
        extension = 'csv'
        runs = glob.glob('{}/*.{}'.format(self.results_dir, extension))

        self.jsonl_paths = []
        for run in runs:
            path_d = run.replace('.csv', '.jsonl')
            self.jsonl_paths.append(path_d)

            df = pd.read_csv(run, encoding='ISO-8859-1')
            df = df.head(10)

            # add columns required by factscore
            df["topic"] = "the provided article"
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

        return self.jsonl_paths


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


    test = MajorityVoter()
    jsonl_paths = test.prepare_for_factscore()
    main(args)


        


