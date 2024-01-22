# On the Fly Neural Style Smoothing for Risk-Averse Domain Generalization

Achieving high accuracy on data from domains unseen during training is a fundamental challenge in domain generalization (DG). While state-of-the-art DG classifiers have demonstrated impressive performance across various tasks, they have shown a bias towards domain-dependent information, such as image styles, rather than domain-invariant information, such as image content. This bias renders them unreliable for deployment in risk-sensitive scenarios such as autonomous driving where a misclassification could lead to catastrophic consequences. To enable risk-averse predictions from a DG classifier, we propose a novel inference procedure, Test-Time Neural Style Smoothing (TT-NSS), that uses a “style-smoothed” version of the DG classifier for prediction at test time. Specifically, the style-smoothed classifier classifies a test image as the most probable class predicted by the DG classifier on random re-stylizations of the test image. TT-NSS uses a neural style transfer module to stylize a test image on the fly, requires only black-box access to the DG classifier, and crucially, abstains when predictions of the DG classifier on the stylized test images lack consensus. Additionally, we propose a neural style smoothing (NSS) based training procedure that can be seamlessly integrated with existing DG methods. This procedure enhances prediction consistency, improving the performance of TT-NSS on non-abstained samples. Our empirical results demonstrate the effectiveness of TT-NSS and NSS at producing and improving risk-averse predictions on unseen domains from DG classifiers trained with SOTA training methods on various benchmark datasets and their variations.

<hr>
This repository contains the codes used to run the experiments presented in our paper "On the Fly Neural Style Smoothing for Risk-Averse Domain Generalization". 
In this repository we describe how to obtain the data used for our experiments and the commands used to run experiments with different settings.

### Obtaining the data:
    1. For PACS and Office-Home we download the data using the code from https://github.com/facebookresearch/DomainBed.
    2. For VLCS we obtained the data from https://github.com/belaalb/G2DM#download-vlcs
    3. For Wikiart we downloaded the data following the instructions from https://github.com/gs18113/AdaIN-TensorFlow2#download-style-images

### Description of the codes in different folders
<hr>

#### AdaIN training: 
This folder contains the code to train the adaIN-based style transfer network using MS-COCO for content images and WikiArt as Style images. The trained decoder is utilized for other style smoothing-based evaluation and training experiments.  Command to run: 

    python adaIN_training.py
    
<hr>

#### Single domain generalization: 
This folder contains the codes to train and evaluate domain generalization models in a single source domain setting.
Within the folder we provide the code for different datasets.
    
a. To run the model training with vanilla ERM algorithm, navigate into the specific dataset folder and run the following command 
    
    python vanilla_erm.py --SOURCE 0. 
    
The flag --SOURCE requires the domain number that should be used for training. All other domains will be used for
evaluation.

b. To run the model training with neural style smoothing, use the following command

    python KL_style_consistency_training.py --SOURCE 0 
    
for training on domain with number 0.

c. To evaluate the models trained with ERM use the following command. 
    
    python eval_adaIN_augmented_classifier.py --SOURCE 0 --MODE 0
    
To evaluate on Wikiart stylized or corrupted version of the dataset use wikiart_eval_adaIN_augmented_classifier.py 
or corrupted_eval_adaIN_augmented_classifier.py respectively for the same flags.

d. To evaluate NSS-trained models use the command in (c) above by changing the MODE flag to 1 instead of 0.
        
<hr>

#### Multi domain generalization:
This folder contains the codes to train and evaluate domain generalization models in a multi-source domain setting.
Within the folder, we provide the code for different datasets.

a. To run the model training with vanilla ERM algorithm, navigate into the specific dataset folder and run the following command 
    
    python vanilla_erm_M.py --TARGET 0
    
The flag --TARGET requires the domain number that should be used for evaluation. All other domains will be used for
training.
    
b. To run the model training with neural style smoothing, use the following command

    python KL_style_consistency_training_M.py --TARGET 0 
    
for training on all domains except 0 which will be used for testing.

c. To evaluate the models trained with ERM use the following command. 

    python eval_adaIN_augmented_classifier_M.py --TARGET 0 --MODE 0
    
To evaluate on Wikiart stylized or corrupted version of the dataset use wikiart_eval_adaIN_augmented_classifier_M.py 
or corrupted_eval_adaIN_augmented_classifier_M.py respectively for the same flags.

d. To evaluate NSS-trained models use the command in (c) above by changing the MODE flag to 1 instead of 0.
    
#### Citing

If you find this useful for your work, please consider citing
<pre>
<code>
@inproceedings{mehra2024fly,
  title={On the Fly Neural Style Smoothing for Risk-Averse Domain Generalization},
  author={Mehra, Akshay and Zhang, Yunbei and Kailkhura, Bhavya and Hamm, Jihun},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3800--3811},
  year={2024}
}
</code>
</pre>
