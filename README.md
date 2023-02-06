# Embedding
Code and trained model for "Learning to Class-adaptively Manipulate Embeddings for Few-shot Learning"

# Requirements
- numpy  1.21.2
- scipy  1.3.0
- torch  1.6.0
- torchvision  0.7.0
- python 3.7.3

# Train
```
- 5-way 1-shot:
python3 main.py  --gpu 0   --N-shot 1  --data ./data/miniImageNet/  --checkpoint  your_model_name


- 5-way 5-shot:
python3 main.py  --gpu 0   --N-shot 5  --data ./data/miniImageNet/  --checkpoint  your_model_name

```
# Test
```
- 5-way 1-shot:
python3 main.py  --gpu 0  --N-shot 1 --data ./data/miniImageNet/  --test 1  --resume  your_model_name/checkpoint.pth.tar  

- 5-way 5-shot:
python3 main.py  --gpu 0  --N-shot 5 --data ./data/miniImageNet/  --test 1  --resume  your_model_name/checkpoint.pth.tar  


```
# Trained models:
```
- 5-way 1-shot:
 ./trained_model_1shot/checkpoint.pth.tar  
- 5-way 5-shot:
 ./trained_model_5shot/checkpoint.pth.tar  
 ```
# My Email:
```
- zhoufei@mail.nwpu.edu.cn
```
