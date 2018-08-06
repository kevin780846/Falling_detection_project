# Human Falling Detection by Anomaly Detection with Auto-Encoder
---
Detect falling event by surveillance camera

Requirements:
- python 3.5
- keras 2.1
- tensorflow 1.4


## Download dataset & model weights
### Include
- model weights
- Training and testing npy file
- Four perspectives of video
Each perspectives contain: 1 training normal video, 1 testing normal video, 12 testing falling video
- link: https://drive.google.com/open?id=1-U4gkkCs0tqajHScJKTO_-PNQCyy8MEG

## Python file explanation
- build_model.py: build auto-encoder model
- train_model.py: train file
- roc_analysis.py: plot roc curve and get threshold for detect falling event
- falling_detection.py: plot reconstruction error curve of full falling video

