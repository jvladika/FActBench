# FActBench

### FactScore instructions

Initial setup
```
cd FActScore
python -m venv fscore
source fscore/bin/activate
pip install --upgrade factscore
python -m spacy download en_core_web_sm
python -m factscore.download_data
pip install -r requirements.txt
```

Wikipedia dump file download (20 GB)
```
cd FActScore/.cache/factscore
pip install gdown
gdown 1mekls6OGOKLmt7gYtHs0WGf5oTamTNat
```

Test run for Factscore
```
cd FActScore
echo "your_openai_key" > api.key
python -m factscore.factscorer --input_path data/unlabeled/ChatGPT_first20.jsonl --model_name retrieval+ChatGPT --openai_key api.key
```
see more options at https://github.com/shmsw25/FActScore


Test run for Adapted version of factscore

Update the paths inside the parse_options() of factscorer.py with absolute path of your working directory
```   
 parser.add_argument('--data_dir',
                        type=str,
                        default="/Users/anumafzal/PycharmProjects/FactSumm/FActScore/.cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default="/Users/anumafzal/PycharmProjects/FactSumm/FActScore/.cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default="/Users/anumafzal/PycharmProjects/FactSumm/FActScore/.cache/factscore/")
```
```
python main.py --input_path {path_to_jsonl} --model_name retrieval+ChatGPT --grounding_provided True --openai_key api.key

```

Example 
```
python main.py --input_path /Users/anumafzal/PycharmProjects/FactSumm/results/3z2i7njs_MODEL_gpt-3.5-turbo-1106_DS_pubmed_TASK_summarization.jsonl --model_name GPT4-mini --grounding_provided True --openai_key api.key

```

python main.py --input_path /home/ubuntu/juraj/results/gpt4omini.jsonl --model_name GPT4-mini --grounding_provided True --openai_key api.key

for running on lrz:
python lrz/deploy.py --input_path /dss/dsshome1/lxc0A/ga27wos2/FactSumm/results/ec27kf6l_MODEL_mistralai-Mixtral-8x7B-Instruct-v0.1_DS_pubmed_TASK_summarization.jsonl --gpu lrz-dgx-a100-80x8 --max_time 2000
