# mlfinal

### pad過的training
https://drive.google.com/open?id=1Y7zsE0JCnX6do7YRH9eSdUmGOrjXQP4r<br>
download to `raw_data/`

## Models
public best: https://drive.google.com/file/d/1S-bA69PoP0fd5InlYgYYq-RA3qXBKQVC/view?usp=sharing<br>
private best: https://drive.google.com/file/d/1JFSSDGkBW8nbn34tQS4z6OoJx6_yx8oS/view?usp=sharing


## train word2vec
`python -m src.train_word2vec --sg`

## train rnn
`python -m src.main --model stacked_gru_2_dot`

## infer rnn
### 1. 下載public best、private best至models/底下
### 2. 執行py檔，如以下兩個指令
 `python -m src.infer --model models/public_best -o predictions/public_best.csv`
 `python -m src.infer --model models/private_best -o predictions/private_best.csv`

## merge
`python -m src.merge --model_name models/stacked_gru_2_dot/stacked_gru_2_dot_002_0.3374 models/stacked_gru_2_dot/stacked_gru_2_dot_003_0.3374 models/stacked_gru_2_dot/stacked_gru_2_dot_004_0.3388 models/stacked_gru_2_dot/stacked_gru_2_dot_005_0.3357 models/stacked_gru_2_dot/stacked_gru_2_dot_006_0.3367 models/stacked_gru_2_dot/stacked_gru_2_dot_007_0.3374 models/stacked_gru_2_dot/stacked_gru_2_dot_008_0.3374`

## environments
python3.6+ is required<br>
see requiements.txt for package and version details<br>
`pip install -r requirements.txt`
