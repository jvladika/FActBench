import json
import glob
import pandas as pd
import argparse
from FActScore.factscore.factscorer import factscrorer_main
class MajorityVoter:
    def __init__(self):

        self.jsonl_paths = None
        self.factscore = None

def prepare_for_factscore(results_dir):

    # get a list of all logs
    extension = 'csv'
    runs = glob.glob('{}/*.{}'.format(results_dir, extension))

    jsonl_paths = []
    for run in runs:
        path_d = run.replace('.csv', '.jsonl')
        jsonl_paths.append(path_d)

        df = pd.read_csv(run, encoding='ISO-8859-1')
        df = df.head(20)
        #df = df.loc[[3,15,16,18]]

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

    return jsonl_paths




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
    jsonl_paths = prepare_for_factscore(csv_results_dir)

    # Call FActscores
    factscore_out = factscrorer_main(args)
    deberta_out = tuple()

    majVot = MajorityVoter()



        


