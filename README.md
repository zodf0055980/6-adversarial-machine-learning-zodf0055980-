# Adversarial Machine Learning
## 作法
首先需要安裝相對應的 tensorflow , Keras 版本
```
$ pip install tensorflow==1.15
$ pip install keras==2.3.1
```
使用 fgsm 做攻擊手法，透過執行 `attack.py` 去產生攻擊的圖片
```
$ python3 attack.py cat1
...
```
所產生的結果 : 

| 攻擊前 | noise | 攻擊後 |
| -------- | -------- | -------- |
| ![image](cat1.jpg) | ![image](cat1noise.png)  | ![image](cat1adv.png) |
| ![image](cat2.jpg) | ![image](cat2noise.png)  | ![image](cat2adv.png) |
| ![image](dog1.jpg) | ![image](dog1noise.png)  | ![image](dog1adv.png) |

準確率預測
![image](attack_predict.png)

reference : https://github.com/soumyac1999/FGSM-Keras
## lab require
The widespread use of artificial intelligence (AI) in today's computer systems incurs a new attack vector for computer security, as the AI can be fooled to take incorrect or even unsafe actions.

In this mini-project, you are given an image classifier (<em>predict.py</em>), which is pre-trained to recognize dogs and cats. After installing python3 and [keras](https://keras.io/), you can use the following command with filenames of the images you want to classify as the arguments.

```bash
$ python3 predict.py cat1.png cat2.png dog1.png
```

it will show the classification probabilities as follows:

![image](classification_result.png)

The classifier identifies the dog image correctly. Now, if we add some noises that are imperceptible to human beings to the dog image. The resulting image file is named *adversarial_example.png*. Let us rerun the classifier.

| ![image](example.png)  | ![image](adversarial_example.png)   |
|---|---|
| Example | Adversarial Example   |

Interestingly, the *adversarial_example.png* is miss-classified as a cat!

![image](adversarial_classification.png)

For the mini-project, you are given three images (cat1.png, cat2.png, dog1.png). Please create their adversarial versions, which will be misclassified by the classifier.

Hint: https://www.tensorflow.org/tutorials/generative/adversarial_fgsm  
