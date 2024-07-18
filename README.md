# FactSumm

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

Test run
```
cd FActScore
echo "your_openai_key" > api.key
python -m factscore.factscorer --input_path data/unlabeled/ChatGPT_first20.jsonl --model_name retrieval+ChatGPT --openai_key api.key
```
see more options at https://github.com/shmsw25/FActScore


