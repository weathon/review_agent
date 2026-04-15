# Dissolving Is Amplifying: Towards Fine-Grained Anomaly Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
In this paper, we introduce \textit{DIA}, dissolving is amplifying. DIA is a fine-grained anomaly detection framework for medical images. We describe two novel components in the paper. 
First, we introduce \textit{dissolving transformations}. Our main observation is that generative diffusion models are feature-aware and applying them to medical images in a certain manner can remove or diminish fine-grained discriminative features such as tumors or hemorrhaging.
Second, we introduce an \textit{amplifying framework} based on contrastive learning to learn a semantically meaningful representation of medical images in a self-supervised manner. The amplifying framework contrasts additional pairs of images with and without dissolving transformations applied and thereby boosts the learning of fine-grained feature representations.
DIA significantly improves the medical anomaly detection performance with around 18.40\% AUC boost against the baseline method and achieves an overall SOTA against other benchmark methods. Our code is available at \url{http://}.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper targets medical anomaly detection. To deal with fine-grained anomalies, this paper uses pretrained diffusion model to provide a data augmentation tool by deleting the fine-grained details. The augmentation is then applied to an existing framework CSI. The experiments on multiple medical diagnosis datasets show the effectiveness of the proposed method.

### Strengths
1.	This paper focuses on a specific application with reasonable motivations and insights.
2.	Experiments show the effectiveness of the proposed method.

### Weaknesses
1.	This paper is somewhat incremental and lacks novelty. This paper proposes to use diffusion model as a data augmentation tool to provide fine-grained data. The following method is highly similar with CSI, an existing method. To offer a clearer perspective, the authors should provide a detailed comparison with CSI, highlighting the key distinctions between these two approaches. Does the contribution come from the proposed augmentation?
2.	The concept presented in this paper has been explored extensively in the literature. Many previous works have investigated the use of fine-grained or natural synthetic anomalies as alternatives to basic augmentations to enhance performance. For instance, in [a], the method incorporates Poisson image editing to seamlessly blend scaled patches of various sizes from separate images. It would be valuable to investigate how such Poisson image editing performs within the CSI framework and whether the proposed diffusion model-based augmentation outperforms these methods when applied within the same CSI framework.
3.	This paper seems to be incomplete or abbreviated. For instance, in Figure 4 on Page 8, two non-medical datasets, CIFAR10 and CIFAR100, are employed, yet these datasets are not introduced or explained elsewhere in the paper. Additionally, the results are not compared with any baseline methods, such as CSI. 

[a] Schluter et al. "Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization." ECCV 2022.

### Questions
See the weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a medical image anomaly detection method, namely dissolving is implying (DIA). It combines two learning paradigm: diffusion models and contrastive learning, for anomaly detection. By pre-trained a diffusion model to the target dataset, images are processed by the reverse diffusion process. The data with added diffusion noise is then taken as the negative samples in contrastive learning. Anomaly scores are calculated based on the similarity between the query and its nearest training samples.

### Strengths
(1) The combination of diffusion models and contrastive learning is quite interesting. Instead of using the diffusion model for image reconstruction, adopting a diffusion model for noise injection is interesting.

(2) The paper is easy to follow.

(3) Experiments are conducted on six medical image datasets from different imaging modalities and organs.

### Weaknesses
(1) The paper employs a diffusion model to introduce noise into the data. What advantages does this approach offer compared to the straightforward addition of random noise to images? Is there any experimentation or ablation study that demonstrates the benefits of using the diffusion model for noise injection?

(2) There is a need for a more in-depth discussion of the proposed method. What are the underlying reasons for the effectiveness of the model? A deeper exploration of the working principles would enhance the paper.

(3) The paper lacks several crucial ablation studies. For instance, what is the impact of altering the number of diffusion steps on the anomaly detection performance? Can you quantify the specific contributions of the two loss terms towards achieving the final results?

(4) It's important to address the computational cost associated with training a diffusion model. How does this cost compare to previous methods in the field? Moreover, when using the diffusion model to inject noise into images, is it possible to substitute a pre-trained model instead of a data-specific one? If so, what performance difference would be observed if a general pre-trained diffusion model were utilized?

(5) In the realm of visual anomaly detection, prior works have showcased the effectiveness of methods like PatchCore. It would be beneficial to include PatchCore in the comparative analysis to provide a comprehensive overview of the proposed method's performance relative to existing approaches.

(6) SOTA anomaly detection methods are able to generate anomaly maps. I am wondering if the propose DIA method also generates a accurate anomaly localization results?

### Questions
Please refer to the weakness section for my questions.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes DIA, a new anomaly detection framework for medical images. It applies the idea of a reverse process of the diffusion model to create a dissolving transformation, which removes fine-grained features like tumors from the input image, as opposed to eliminating noise in the traditional diffusion model. The generated negative pairs are then used within a contrastive learning framework.

### Strengths
The proposed idea of employing the reverse process of a generative diffusion model to remove fine-grained features from the original images and generate negative pairs for contrastive learning is novel and sound.

### Weaknesses
-	The theoretical basis for the dissolving transformation through the reverse diffusion process to remove fine-grained features is not well explained, while visual validation is partly provided in Figure 1.

-	The authors specifically target anomaly detection for medical images throughout the manuscript, but it is not clear whether and why the proposed method is better suited for medical images, but not for other domains. 

-	The previous methods compared in the experimental section primarily focus on anomaly detection using the MVtec dataset, which differs from medical images. For instance, there are various deep learning-based methods designed specifically for detecting anomalies in medical images. The authors should consider including relevant studies focused on medical images or expand their experiments to datasets in other domains.

### Questions
-	It would be interesting to see the results when applying the proposed method to the data used in other comparative models. 

-	The results indicate that downsampling high-resolution images to 32x32 leads to the best performance, but it is unclear why performance is comparatively worse when downsampling to 64x64 or 128x128. If fine-grained features are indeed crucial, higher resolutions should yield better performance.

-	Adding results related to the changes in the diffusion step 't' in Section 5.1 would be beneficial.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents a contrastive learning based unsupervised anomaly detection approach. The authors first argue that diffusion models erase fine-grained discriminative features. Based on this assumption the authors add “dissolved samples” (samples with further reverse diffusions) as negative samples to the contrastive framework. These dissolved samples enforce the contrastive learner to differentiate subtleness in the input images. The proposed approach is tested on public datasets, and it demonstrates overall improved performances compared with existing approaches,

### Strengths
The problem setting of learning fine-grained subtleness for anomaly detection is of significant clinical potential for medical imaging applications. 

This paper is well-written with overall sufficient clarity. 

The insight that diffusion models tend to remove fine-grained discriminative features is interesting. The idea of taking the input images for further reverse diffusions is a bit surpring at first sight but it is reasonable if we assume diffusion models pull reversed samples towards the distributions of input data.

### Weaknesses
A very critical sanity check is needed to validate the dissolving transforms: Would simple gaussian smoothings and/or additive noises on the image space or shallow-layer feature space lead to similar “dissolving” effect as with diffusion models? The general idea is still to remove fine-grained subtleness by smoothing or altering the content. But if simple gaussian smoothing/additive noises can reach similar effects, there would be no need to bother training a diffusion model. This is the major reason I remain slightly negative at this stage. 


Some terms are not self-explanative: E.g., what does diffusion models to be “feature-aware” mean in the abstract? 


The term *positive synthesis* and *negative synthesis* may not be sufficiently representative: the authors may consider using the terms *reconstructive approaches* or *generative approaches* to summarize DAE-/AnnoGAN-/Diffusion-based models, etc. and using *discriminative approaches* to summarize synthetic-anomoly-based approaches. Also, the authors may want to discuss approaches that synthesize anomalies based on handcrafted assumptions on anomaly distributions: [1,2]. Performance comparisons with those methods are also welcomed, if the authors find them necessary. 

The dissolving operation serves as the core of the approach while intuitively it subjects to the selection and training details of the base diffusion models. Therefore, more details of the diffusion model (e.g., subtypes of diffusion models, number of steps, computational overhead, etc.) and how they dictate the final anomaly detection results need to be discussed. 

Eq. 10 lacks clarity: Do we compute the mean scores under all $k$’s or do we take the max? For $S_{con}$, why does the feature norm need to be multiplied?

### Questions
The authors are encouraged to discuss more on the employed diffusion models, in together with the associated computational costs.

“Figure” (Table) 4.: The absolute number of training samples should be provided.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
