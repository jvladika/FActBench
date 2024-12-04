import glob
import json
import os
import re
import shutil
from time import sleep
from typing import List, Optional, Tuple

import pandas as pd
import wandb

runs = [
]


def load_wandb_tables_by_project(entity: str, project: str, out_dir: str, selected_run_ids: Optional[List[str]] = None):
    # initialize API client
    api = wandb.Api()
    failed_runs = []

    # make sure the directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get all runs in the project
    runs = api.runs(path=f"{entity}/{project}")

    for run in runs:
        try:
            if selected_run_ids and run.id not in selected_run_ids:
                continue
            if run.state != 'finished':
                continue

            json_config = json.loads(run.json_config)
            print (run.name)


            model_name = run.name.split('_')[1]
            table_name = f"summary_table"
            task_name = "summarization"
            ds_name = json_config["dataset_name"]["value"]

            filename = f"{run.id}_MODEL_{model_name}_DS_{ds_name}_TASK_{task_name}.csv"
            filepath = os.path.join(out_dir, filename)

            table_artifact = run.logged_artifacts()
            table_name = "summary_table"

            for tab_art in table_artifact:
                table_dir = tab_art.download()
                if table_name in table_dir:
                    break

            table_path = f"{table_dir}/{table_name}.table.json"
            with open(table_path) as file:
                json_dict = json.load(file)
            df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
            df.to_csv(filepath, index=False)

            print(f"Data from run {run.id} saved to {filepath}")
        except:
            print (f"problem with {run.id} {run.name}")
            failed_runs.append(run.id)

    sleep(1)
    artifacts_root_dir = "artifacts"
    shutil.rmtree(artifacts_root_dir)
    return failed_runs


def load_wandb_domain_adaptation_summarization_data(out_dir:str, entity: str, project:str, selected_run_ids: Optional[List[str]] = None):


    return load_wandb_tables_by_project(entity=entity, project=project, out_dir=out_dir, selected_run_ids=selected_run_ids)



def main():
    out_dir = "../results"
    entity = "anum-afzal-technical-university-of-munich"

    project = "factcheck-summarization"

    failed_runs = load_wandb_domain_adaptation_summarization_data(out_dir, entity=entity, project=project)
    print (failed_runs)

if __name__ == "__main__":
    main()
