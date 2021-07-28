# Captcha-Recognition

### Introduction 
This project is divided into 2 tasks: 
a) Train two models to recognize handwritten English letters and compare their performances;
b) Use the best trained classifier in the first task to identify which letters compose a corrupted image (captcha)

### Dataset

*For task a):*
It consists of images of handwritten English alphabet and the corresponding labels. In total 124800 images and labels are present. 
For each label 4800 different images are available.

*For task b):* 
It is composed of a series of 4 letters in a corrupted image of size 30 × 140.

### Methodology

*Task a)*
K-NN was utilized as a baseline and compared with the classification accuracy of a 2-D CNN.

*Task b)*
First, noise was removed from captcha. Then, bounding boxes were used to divide the image into 4 letters. 
Hence, predictions were made per each letter in the captcha. 

### Results 

*Task a)*
| *Accuracy Score(%)*| Validation set  | Test set |
| ------------- | ------------- | ------------- |
| **K-NN**  | 82.9% | 84.8%  |
| **CNN** | 95.01% | 95.4%  |

