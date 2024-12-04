import argparse
import os
import json


def config_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to generate and run necessary execution files"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=[
            "lrz-dgx-a100-80x8",
            "lrz-v100x2",
            "lrz-dgx-1-v100x8",
            "lrz-dgx-1-p100x8",
            "lrz-hpe-p100x4",
        ],
        help="GPU partition to use",
        required=True,
    )
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use", default=1)
    parser.add_argument(
        "--max_time", type=int, help="Maximum time for execution in minutes", default=60
    )
    parser.add_argument("--input_path", type=str, help="Experiment for which GenFact should be evaluated",
                        default="results/vxfjdpx6_MODEL_meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo_DS_pubmed_TASK_summarization.jsonl")

    return parser.parse_args()


def get_exec_str(args) -> str:
    print (args)
    model_name = (args['input_path'].split('/')[1]).split('_')[2]
    ds_name  = (args['input_path'].split('/')[1]).split('_')[4]
    print (model_name, ds_name)
    return f"{ds_name}-{model_name}"



if __name__ == "__main__":
    parser = config_parser()
    print (parser)
    input_path = parser.input_path


    # get full run path from the config json file
    exec_path = get_exec_str(input_path)
    aug_exec_path = os.path.join("lrz", "runs", exec_path)

    # mkdir aug exec path
    if not os.path.exists(aug_exec_path):
        os.makedirs(aug_exec_path)

    # create dump files
    dump_out_path = os.path.join(aug_exec_path, "dump.out")
    dump_err_path = os.path.join(aug_exec_path, "dump.err")
    os.system(f"touch {dump_err_path}")
    os.system(f"touch {dump_out_path}")

    og_path_container = "/dss/dssfs04/lwp-dss-0002/t12g1/t12g1-dss-0000/"

    # create sbatch file
    sbatch_path = os.path.join(aug_exec_path, "run.sbatch")
    with open(sbatch_path, "w") as sbatch_file:
        sbatch_file.write("#!/bin/bash\n")
        sbatch_file.write("#SBATCH -N 1\n")
        sbatch_file.write(f"#SBATCH -p {parser.gpu}\n")
        sbatch_file.write(f"#SBATCH --gres=gpu:{parser.num_gpus}\n")
        sbatch_file.write("#SBATCH --ntasks=1\n")
        sbatch_file.write(f"#SBATCH -o {dump_out_path}\n")
        sbatch_file.write(f"#SBATCH -e {dump_err_path}\n")
        sbatch_file.write(f"#SBATCH --time={parser.max_time}\n\n")

        srun_command = f"srun --container-image ~/demo.sqsh --container-mounts={og_path_container}:/mnt/container torchrun --nproc_per_node={parser.num_gpus} --standalone ~/FactSumm/main.py --input_path {input_path}"

        sbatch_file.write(f"{srun_command}\n")

    # submit sbatch job
    os.system(f"sbatch {sbatch_path}")
