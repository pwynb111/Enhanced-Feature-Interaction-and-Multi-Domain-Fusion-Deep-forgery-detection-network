# EFMID-Net Enhanced Feature Interaction and Multi-Domain Fusion Deep forgery detection network

## Approach

![Enhanced Feature Interaction and Multi-Domain Fusion Deep forgery detection network
](Framework.jpg)

## Dependencies
```
pip install requirement.txt
```

## Data Preparation
1. Download the original dataset from [FF++](https://github.com/ondyari/FaceForensics).
<!---2. Download the landmark detector from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks).-->
2. Extract frames from FF++ videos. The processed dataset can be downloaded from [FF++](https://pan.baidu.com/s/1ZHm-WCiPjor2Tz2IsuojvA) (code: 7s5s)(Cited from [Locate-and-Verify](https://github.com/sccsok/Locate-and-Verify)).
4. Run the code in folder *./process* to get the aligned images and masks.

##  Pre-trained Model
The pre-trained model of EFMID-Net will be made publicly available in this repository soon. Please stay tuned for updates.


## Results

Our model achieved the following performance on:

| Training Data | Backbone        | FF++       | Celeb-DFv1   | Celeb-DFv2       | 
| ------------- | --------------- | ---------- | ----------   | ---------- |
| FF++          | Xception        | 0.998      | 0.938        | 0.995     |
Note: the metric is *frame-level AUC*.
## Training

To train our model from scratch, please run :

```
python3  train.py --opt ./config/FF++.yml --gpu *
or
python -m torch.distributed.launch --nproc_per_node * --nnode * train.py --opt ./config/FF++.yml
```

