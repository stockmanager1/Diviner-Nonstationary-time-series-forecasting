# Long term 5G network traffic forecasting via modeling non-stationarity with deep learning

## 원본 링크
원본(https://github.com/CapricornGuang/Diviner-Nonstationary-time-series-forecasting)

## 모델 수정
사이버 보안 경진대회에 참가하기 위해서 일부 코드를 변경했습니다.

[모델의 한글 설명](https://velog.io/@stockmanager1/Long-term-5G-network-traffic-forecasting-viamodeling-non-stationarity-with-deep-learning)


p.s 강 모델 설명할때 테블릿으로 마지막으로 한번 더 정리해보고 그걸 올려야겠다. 손으로 못쓰니까 너무 불편하네 ㅠㅠㅠ

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


## 수정 사항

```
#원본 모델에서는 학습 데이터의 MAE,MSE 평가 결과 값만을 도충합니다. 하지만 중간에 exp_diviner 파일을 수정해 예측을 수행한 값을 원본 스케일로 돌리고 이를 넘파이로 저장 #하는 코드를 추가해 이제 원본 논문 코드의 결과 값을 시각화 할 수 있습니다.
#따라서 우선 위 Usage를 진행 하신 다음 colab환경을 기준으로 아래 코드를 작성하시면 됩니다. 

import numpy as np

predictions = np.load('/content/predict_values.npy',allow_pickle=True)

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(predictions[0], marker='o', linestyle='-')
#plt.axhline(y=0, color='r', linestyle='--', label='Threshold') 
plt.title('PREDIECT')
plt.xlabel('Time Step')
#plt.yticks([])  
plt.ylabel('Value')
#plt.legend()  
plt.grid(True)
plt.show()
```

