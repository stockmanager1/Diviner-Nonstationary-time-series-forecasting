# Long term 5G network traffic forecasting via modeling non-stationarity with deep learning

## 원본 링크
원본(https://github.com/CapricornGuang/Diviner-Nonstationary-time-series-forecasting)

## 모델 수정
사이버 보안 경진대회에 참가하기 위해서 일부 코드를 변경했습니다.

## Requirements

- Python 3.6+
- numpy == 1.21.6
- pandas == 1.3.5
- scikit_learn == 1.0.2
- torch == 1.12.0+cu113

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```
## Usage
Here we present a simple demo of training your Diviner model.
```bash
#Traing <Diviner model> on ETTh1 dataset from scratch.
python -u main.py --model=diviner --data=ETTh1 --predict_length=336 --enc_seq_len=30 --out_seq_len=14 --dec_seq_len=14 --dim_val=24 --dim_attn=12 --dim_attn_channel=48 --n_heads=6 --n_encoder_layers=3 --n_decoder_layers=2 --batch_size=32 --train_epochs=100 --use_gpu --smo_loss --dynamic --early_stop --shuffle --verbose --out_scale

#Testing <Diviner model> on ETTh1 dataset from scratch.
python -u main.py --model=diviner --data=ETTh1 --predict_length=336 --enc_seq_len=30 --out_seq_len=14 --dec_seq_len=14 --dim_val=24 --dim_attn=12 --dim_attn_channel=48 --n_heads=6 --n_encoder_layers=3 --n_decoder_layers=2 --batch_size=32 --train_epochs=100 --use_gpu --smo_loss --dynamic --early_stop --shuffle --verbose --out_scale --test 
--load_check_points=.\checkpoints\ETTh1\336\diviner_checkpoints.ckpt 
```

