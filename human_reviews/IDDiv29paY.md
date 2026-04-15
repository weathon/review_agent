# Fooling Contrastive Language-Image Pre-Training with CLIPMasterPrints

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3

## Abstract
Models leveraging both visual and textual data such as Contrastive Language-Image Pre-training (CLIP), are the backbone of many recent advances in artificial intelligence. In this work, we show that despite their versatility, such models are vulnerable to what we refer to as fooling master images. Fooling master images are capable of maximizing the confidence score of a CLIP model for a significant number of widely varying prompts, while being either unrecognizable or unrelated to the attacked prompt for humans. We demonstrate how fooling master images can be mined using stochastic gradient descent, projected gradient descent, or gradient-free optimisation. Contrary to many common adversarial attacks, the gradient-free optimisation approach allows us to mine fooling examples even when the weights of the model are not accessible. We investigate the properties of the mined fooling master images, and find that images trained on a small number of image captions potentially generalize to a much larger number of semantically related captions. Finally, we evaluate possible mitigation strategies and find that vulnerability to fooling master examples appears to be closely related to a modality gap in contrastive pre-trained multi-modal networks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper examines the possibility of attacking pre-trained CLIP models, i.e. generating a fooling image to maximize its embedding cosine similarity with some given prompts. Three possible methods of attacking including stochastic gradient descent, latent variable evolution, and projected gradient descent are considered. Experiments are done on imagenet showing the effectiveness of proposed methods. It is also found that fooling images trained on a small number of image captions potentially generalize to a much larger number of semantically related captions.

### Strengths
Measuring the similarities between images and text is an important topic and it is surprising to see that with simple adversarial attacks, we are able to generate a fooling image that can simultaneously maximize its cosine similarity across multiple different prompts.

The use of latent variable evolution in the context of adversarial attack is interesting. With LVE, it does not require access to the model weights and thus overcoming the limitations of common gradient-based methods such as SGD and PGD.

The paper is well-written and easy to follow.

### Weaknesses
I am not very convinced by the setting of the problem. In particular, I am not convinced why we need a fooling image that maximizes its possibility with many other prompts. What are the potential concerns of this vulnerability?

It seems that the methodology is not different enough with the existing literature. In particular, as admitted in the paper, none of SGD, LVE, or PGD is a technical contribution of this paper, and it seems that this paper is just evaluating those methods in the context of fooling CLIP.

Only original CLIP models are evaluated in the paper but there are many later improvement such as TCL (https://arxiv.org/pdf/2202.10401.pdf), ALBEF (https://arxiv.org/abs/2107.07651), BLIP (https://arxiv.org/abs/2201.12086), and so on. Do they have the same vulnerability? How does this vulnerability scale with the size of pre-training data (https://arxiv.org/abs/2210.08402)?

### Questions
The title is misleading. If I understand correctly, the paper does not have pre-training experiments but only carry out evaluations of pre-trained CLIP.

Please see weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies the adversarial attacks of CLIP models and proposes CLIPMasterPrints, a type of images that can maximize the CLIP scores with a wide range of prompts. This paper proposes three ways to mine these CLIPMasterPrints, SGD, PGD and gradient-free optimization when the model weights are inaccessible. Details experiments on image recognition tasks to show that extracted CLIPMasterPrints can fool the pretrained CLIP model on wide categories. The authors also study how to mitigate the attack risks from CLIPMasterPrints by mitigating the modality gap between text and image encoder in CLIP

### Strengths
1. This paper proposes a new type of adversarial attacks for CLIP models CLIPMasterPrints. Given the fact that CLIP is the foundational vision-language models that have wide application, this topic plays an important role in mitigating the risks of misusing CLIP.

2. This paper proposes several technical ways to mine the CLIPMasterPrints, from gradient based methods to non-gradient methods, which could cover diverse scenarios based on if the CLIP weights are accessible or not.  

3. This paper also studies the way to reduce the risks of CLIPMasterPrints. Although the solution points to the existing finding (multimodality gap), it is still good to see the solution.

### Weaknesses
1. Limited experiments: this paper also conducts experiments on ImageNet. It is interesting to see if the conclusion still holds for other recognition dataset ( note CLIP is evaluation on dozens of datasets). Moreover, this paper also uses CLIP ViT-L models. It also interesting to see the performance on CLIP ResNet models.

### Questions
See Weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper focus on mining images referred as “fooling master images” or “CLIPMasterPrints” that fool the CLIP model by obtaining higher image-text similarity scores compared to clean image-text scores. These CLIPMasterPrints are optimized to obtain hight similarity scores across different text embeddings. Authors show that these CLIPMasterPrints also generalize to semantically related text prompts that are not directly considered in the optimization process. They attribute the rationale for existence of such CLIPMasterPrints  to the not well aligned CLIP image-text embeddings. They countermeasure CLIPMasterPrints  by adjusting the CLIP alignment i.e. by shifting the centroids of image and text embeddings.

### Strengths
-	Simple approach to successfully fool against different text embeddings.
-	Paper is easy to read and understand.
-	Illustrations are clear.

### Weaknesses
Prior works have already shown that CLIP is vulnerable to adversarial attacks (Noever & Miller Noever, 2021; Daras &Dimakis, 2022; Goh et al., 2021). This work on fooling master images is a variant of such adversarial attacks utilizing already existing optimization algorithms like SGD, PGD and, LVE (Latent Variable Evolution) to craft an image with perturbations. The difference here is that adversarial objective aims to fool different text embeddings. Furthermore, countermeasuring CLIPMasterPrints is performed by shifting the centroids of image and text embeddings that is proposed in prior work Liang et al. (2022) (cited in the paper). Therefore, this limits the originality, novelty and technical contributions of this work.

### Questions
Despite the concept of CLIPMasterPrints for CLIP is interesting, the paper does not meet the criteria for conference acceptance. I suggest this paper to be a fit for workshop.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
