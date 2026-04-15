# Dataset Distillation in Latent Space

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 5, 3

## Abstract
Dataset distillation (DD) is a newly emerging research area aiming at alleviating the heavy computational load in training models on large datasets, as it tries to distill a large dataset into a small and condensed one so that models trained on the distilled dataset can perform comparably with those trained on the full dataset in downstream tasks. Among the previous works in this area, there are three key problems that hinder the performance and availability of the existing DD methods: high time complexity, high space complexity, and low info-compactness. In this work, we simultaneously attempt to settle these three problems by moving the DD processes from conventionally used pixel space to latent space. Encoded by a pretrained generic autoencoder, latent codes in the latent space are naturally info-compact representations of the original images in much smaller sizes. After transferring three mainstream DD algorithms to latent space, we significantly reduce time and space consumption while achieving similar performance, allowing us to distill high-resolution datasets or target at greater data ratio that previous methods have failed. Besides, within the same storage budget, we can also quantitatively deliver more info-compact latent codes than pixel-level images, which further boosts the performance of our methods.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Dataset distillation is known to be expensive in both memory and training time. The authors in this paper propose to address this issue directly through an auto-encoder, where the actual DD only happens in the latent space. The distilled codes can be used to further reconstruct and obtain training images. The authors demonstrate the efficiency and promising training results on the proposed method.

### Strengths
+ The proposed method tackles the DD problem from another angle. The latent code distillation makes a lot of sense in terms of efficiency and can potentially help the field on larger datasets
+ The authors demonstrate that the proposed method indeed can achieve descent performance with good efficiency
+ The authors' writing is pretty clear and easy to follow

### Weaknesses
- The algorithm seems to be heavily depending on the quality of the pretrained autoencoder, causing another layer of complexity in the distillation procedure.
- In a more general field, language or other modality, where AEs are not that popular, the proposed method can be limited in terms of contribution or usage.
- It seems that the authors only focus on a subset of DD algorithm, how would the latent DD perform using FrePo [1] or momentum-based BPTT [2]? It would be nice if authors can add the comparison and discussion on these two as well.

[1] Dataset Distillation using Neural Feature Regression

[2] Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks

### Questions
See above.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work aims to address three challenges in Dataset Distillation: high time and space complexities, and low data compactness. They proposed LatentDD to move the distillation from pixel space to latent space, leveraging a pretrained autoencoder from stable diffusion. The LatentDD method significantly reduce time and space requirements in DD tasks, allowing the distillation of higher resolution datasets and offer more info-compact latent codes within the same storage limits.

### Strengths
This paper is well-motivated, and well-organized. The observation that "distilling dataset in the original space (e.g. pixel space for image datasets) will inevitably condense high-frequency detailed information into limited storage budget, which is usually unnecessary for downstream tasks" is a solid point to serve as motivation for method design. The authors also provide comprehensive experiments on various datasets.

### Weaknesses
1. While this method has demonstrated its effectiveness for high-resolution dataset distillation, there are no experiments and results comparison on lower resolution datasets such as CIFAR10/100. It leaves concern of whether using an autoencoder from stable diffusion for DD impacts the performance of distillation for such datasets.
2. To my knowledge, coreset selection does not belong to dataset distillation, and dataset distillation usually refers to the optimization based methods that distill the data into compact synthetic sets. Therefore, the statement in the introduction: "Some DD methods select a subset from the full dataset according to certain rules (Feldman et al., 2013; Welling, 2009; Sener & Savarese, 2018; Aljundi et al., 2019; Zhou et al., 2023), usually referred to as coreset selection" seems confusing.
3. In the introduction, P1: "computationally intensive bilevel optimization problem" and P2: "space complexity, i.e. DD needs to store the whole computation graph" however, I don't see the present method design that directly aims to address the computation issue of bilevel optimization and I did not see the latentDD method evaluated on bilevel optimization based DD methods. Besides, current method still needs to store the entire computation graph.
4. All three methods chosen by the authors all falls into surrogate objective DD frameworks which seems limiting and the meta-learning based methods are ignored. It would be a lot more convincing to include methods related to meta-learning based DD as well.
5. Eqn.2 is for meta-learning DD framework, and the authors did not evaluate their latentDD method on meta-learning DD framework, which seems kinda disconnected to list Eqn.2 here. 
6. The authors claim: "all the previous works have distilled datasets in pixel space" which does not seem accurate. Check [1] for more details.
7. The authors did not report the full dataset (of latent codes) performance, making it hard to evaluate the performance gap between original dataset and distilled ones.
8. Following 7, since the full dataset (of latent codes) performance is unknown, and there's also no experiment with the performance evaluation of the initial latent code (post-autoencoding and pre-distillation), it's unclear to me whether the performance would be good enough just with the latent code itself even without the distillation process. I think it will be interesting to see how much of the performance gain it has during the distillation process, or this distillation process actually hurts the initial latent code's info-compact ability.
9. The cross-architecture results reported in table 6 seems confusing, eg., the performance of ResNet18 (56.00) and VGG11 (49.32) performance are consistently better than the ConvNet results (46.72), even the distilled data comes from ConvNet.
10. The authors only report IPC=1 for Res=512, would be great to see the performance of IPC=10. 

[1] Cazenavette, George, et al. "Generalizing Dataset Distillation via Deep Generative Prior." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
Check weaknesses section for more details.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the challenges in Dataset Distillation (DD) by transitioning from pixel space to info-compact latent space. This shift reduces time and space requirements while maintaining performance, enabling high-resolution dataset distillation. The paper's method delivers more info-compact latent codes within the same storage constraints, enhancing efficiency.

### Strengths
The paper identifies three primary challenges in dataset distillation: high time complexity, high space complexity, and the retention of unnecessary high-frequency information. The authors claim to introduce a pioneering framework that directly addresses these issues by conducting dataset distillation in the latent space, rather than the pixel space.

### Weaknesses
● The authors assert that they are the first to successfully address these three challenges in dataset distillation. What specific limitations or hindrances have prevented existing works from generalizing solutions to these problems? Is the proposed method the sole solution, or are there alternative approaches that merit consideration?

● I've noticed that the paper exclusively presents performance experiments on the Sub-ImageNet dataset. Given the existence of prior works that have addressed the full ImageNet condensation problem efficiently, and the authors' claim of efficiency, it would be valuable to see if the authors can tackle this challenging task and report the results.

● How could the third challenge be solved in this framework (distilling dataset in the
original space (e.g. pixel space for image datasets) will inevitably condense high-frequency detailed
information into limited storage budget, which is usually unnecessary for downstream tasks.)? Is it tested on the experiments?

● As far as my current knowledge goes, there are existing works on matching the latent space in the dataset distillation (DD) framework. Could you highlight the primary distinctions between your work and these existing approaches that set your method apart?

### Questions
Methodology and Experimental Setup:

a. Could you provide a more detailed description of the methodology used in your experiments, including specific hyperparameters, model architectures, and training protocols?

b. How were the datasets prepared and preprocessed before conducting experiments, and what criteria were used for data selection and cleaning?

Comparative Analysis:

a. In the context of your efficiency claims, can you offer a direct quantitative comparison of your method with existing dataset distillation (DD) frameworks, highlighting the advantages and limitations?

b. Given the broader landscape of DD research, can you elaborate on how your approach compares with other methods in terms of scalability and generalization to different datasets and architectures?

Scalability and Generalization:

a. To address scalability, how does your method perform when applied to datasets with higher resolutions or complex network architectures?

b. Can you discuss the generalization capabilities of your proposed method, particularly in the context of training on models directly (without transfer learning) and its applicability to large-scale datasets like ImageNet?

Latent Space vs. Pixel Space:

a. What are the main advantages of conducting dataset distillation in the latent space, as opposed to the pixel space, and how does this impact the retention of high-frequency information?

b. Could you explain the rationale for not performing experiments on the full ImageNet dataset, given your assertion of efficiency, and how your method could address this challenging task?

Distinguishing Features:

a. In light of existing works that match the latent space in the DD framework, what key differentiating features or innovations characterize your approach?

b. Can you clarify the specific mechanisms or techniques that set your method apart from prior works, leading to the successful resolution of the identified challenges in DD?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes to perform dataset distillation in the latent space instead of the pixel space. The proposed method first encodes real images of target dataset into the latent codes. Then three representative (pixel level) distillation methods are adapted to distill latent codes. After that, the distilled latent codes are fed into the decoder to get the distilled images.

### Strengths
This paper shows that, performing the distillation in latent space costs less resources than the distillation in pixel space, without sacrificing much performance.

### Weaknesses
Most comparisons in this paper are UNFAIR.

In dataset distillation area, previous works compare the performance under the same IPC (image per class) settings, which means that the AMOUNT of the distilled images fed into the evaluation network is fixed.

This paper proposes ‘LPC’ (latent per class) and claims that 1 IPC=12 LPC since their size are the same (latent codes have lower resolution). Then the authors compare their method’s performance (12*n LPC) with previous works (n IPC), which means that the method proposed in this paper actually uses twelve times more images than previous methods for evaluation.

I think comparing performances under the same ‘storage consumption’ settings rather than IPC are unfair and unacceptable. Otherwise, we can perform the distillation first and then use an auto-encoder to compress the distilled images, such that we can use more distilled data under the same ‘storage consumption’ setting; accordingly, the performance is improved. Then the development of dataset distillation might turns toward finding a stronger auto-encoder.

### Questions
In Table 2, the time consumption of LatentDC/DM/MTT is evaluated under the same IPC settings or the proposed LPC settings?

I think the selling point of this paper should be: Performing distillation in latent space is quicker, low-cost, and will not sacrificing performance drastically. It is fine to perform worse than previous works since it is hard to acquire lower cost and better performance at the same time. Please stop performing the evaluation under ‘LPC’ settings. I suggest the authors to perform a fair comparison and highlight their contributions better (such as low cost).

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
