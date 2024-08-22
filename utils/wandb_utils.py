import randomname as randomname
import os
import wandb

def wandb_setup():

    wandb_token_key: str = "WANDB_TOKEN"

    # wandb setup
    wandb_tok = os.environ.get(wandb_token_key)
    assert wandb_tok and wandb_tok != "<wb_token>", "Wandb token is not defined"
    wandb.login( key=wandb_tok)

def wandb_init_run(run_path:str, config = None, wandb_project_name = "GenFact", entity = "anumafzal"
                   ):
    wandb_setup()
    wandb_mode = "online"
    wandb_run = os.path.basename(run_path).replace('.jsonl','')
    model = wandb_run.split('_')[2]
    ds = wandb_run.split('_')[4]
    task = wandb_run.split('_')[-1]

    wandb_run_name = randomname.get_name() + '_' + '_'.join(
        [model, ds, task])

    wandb.init(project=wandb_project_name, entity=entity, config=vars(config), name=wandb_run_name,
               mode=wandb_mode, group=task)



