# Unsupervised Discovery of Object-Centric Neural Fields

- Decision: Reject
- Scores: 8, 6, 6, 3

## Abstract
We study inferring 3D object-centric scene representations from a single image. While recent methods have shown potential in unsupervised 3D object discovery from simple synthetic images, they fail to generalize to real-world scenes with visually rich and diverse objects. This limitation stems from their object representations, which entangle objects' intrinsic attributes like shape and appearance with extrinsic, viewer-centric properties such as their 3D location. To address this fundamental bottleneck, we propose unsupervised discovery of Object-Centric neural Fields (uOCF). uOCF focuses on learning the intrinsics of objects and models the extrinsics separately. Our approach significantly improves systematic generalization, thus enabling unsupervised learning of high-fidelity object-centric scene representations from sparse real-world images. To evaluate our approach, we collect three new datasets including two real kitchen environments. Extensive experiments show that uOCF enables unsupervised discovery of visually rich objects from a single real image, allowing applications such as 3D object segmentation and scene manipulation. Impressively, uOCF even demonstrates zero-shot generalizability to unseen, more difficult objects. We attach an overview video in our supplement.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel approach called unsupervised discovery of Object-Centric neural Fields (uOCF) for inferring 3D object-centric scene representations from a single image. To generalize to the real-world scenes, the paper focuses on disentangling the learning of object intrinsics and the extrinsic separately.  The proposed approach significantly improves systematic generalization, enabling unsupervised learning of high-fidelity object-centric scene representations from sparse real-world images. The approach allows for the discovery of visually rich objects from a single real image, allowing for applications such as 3D object segmentation and scene manipulation.

### Strengths
1. The proposed 3-stage training process is technically sound and is proven to be effective. 
2. It is nice to see the effectiveness of the proposed approach in terms of zero-shot generalizability on various datasets.
3. This paper is well-written and well-organized, providing a detailed explanation of the proposed approach. Its evaluation of various datasets proves the generalization ability.
4. The proposed method enables unsupervised learning of high-fidelity object-centric scene representations from sparse real-world images, which has potential applications in 3D object segmentation and scene manipulation.

### Weaknesses
1. The main additional contribution of this paper should be disentangling the learning of object intrinsics and extrinsic compared with the previous method uORF. The performance comparisons in the experimental parts demonstrate the effectiveness of the whole pipeline. However, it is not very clear to me which parts contribute most to the performance gain.  The object-centric sampling mentioned in Sec. 3.3 seems reasonable and should lead to better rendering performance. But I didn't see any relevant ablation study on this.  
2. The authors did not provide a detailed analysis of the limitations or failure cases of the proposed approach, which could limit its applicability in certain scenarios.
4. The paper does not provide a clear explanation of the training process and hyperparameters used in the experiments, which could limit its reproducibility and further research.

### Questions
1. How to decide the hyperparameter K in Eq.(1)? How to guarantee the redundant slots do not affect the learning of other slots? 
2. Did you test with transparent objects? Would it also work properly?
3. Did you test the geometric quality of the learned object representations? Can your method recover accurate shape of the objects?
4. Which module do you think contributes most to the final performance improvement? Could you provide more detailed ablation study on this?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies unsupervised identification of objects in 3D radiance space from a single image. Particularly, given an input image, the encode will learn per-object fectures and per-object locations in 3D space. Each object will be encoded into a seperate NeRF model. By composing all individual 3D objects  and background and rendering back to 2D images, the entire network will be supervised by multiview images by photometric losses. Experiments are conducted on collected datasets and promising results are obtained.

### Strengths
1) The paper studies a critical problem in object-centric learning without any human annotations, especially discovering objects in 3D space. 

2) Using posed multi-views as supervision signals to learn radiance fields is reasonable to provide more information and constraints for object discovery. 

3) Additional datasets are provided to evaluate the proposed method. They are supposed to be beneficial in the community in the future.

### Weaknesses
1. About the motivation

In the beginning, the paper states that the bottleneck of existing works is the entanglement of object attributes such as shape and appearance with extrinsic properties such as object location. However, there is a lack of concrete evidence to support this claim.  In this paper, why is separately learning per-object locations significantly better than others?


2. About the method

2.1. In sec 3.1, the description of Encoder/Latent Extraction Module seems not matched with the provided Figure 2. It's unclear how the two sets of feature maps f_g and f_l are fed into the second module. 

2.2. In Latent Extraction Module, for each object, its position p_i^{wd} in the world frame will be learned. In this case, what is the world coordinate? Is it predefined for the whole set of multi-view images of every single scene?  In this case, given two different input images of the same scene, the same 3D object needs to learn exactly the same 3D location. In this way, the network actually needs to learn a separate coordinate system conditioning on every input image, which seems not that sensible if I understand correctly. 

2.3. The paper models every object/background by a separate NeRF. How is such strategy able to deal with the potential under/over segmentation of 3D objects? For example, what if two chairs are grouped into a single NeRF (the center of two chairs may be learned as the combined object center)? What if a single chair is learned as two objects at the very beginning? Why will the entire background structure (usually complex) must be grouped into a single NeRF? In practice, there are typically a variable number of objects in each 3D scene to discover. There could be objects that are visible in input image, but invisible in multi-view images, or otherwise. 

2.4. The paper states that it incorporates two novel techniques: 1) object-centric prior learning and 2) object-centric sampling. However, in page 5, such two techniques are very briefly described by two small paragraphs. As to object-centric prior learning, what types of priors are planned to be learned? why is it helpful for the subsequent object discovery? As to object-centric sampling, how does it work in  detail and why is it important? 


3. About the experiments

3.1. For the collected three datasets. How many objects in each image? From the samples in Figure 4, it seems every image has exactly four objects.

3.2. As to the two real-world datasets Kitchen-easy/hard,  the objects are rather simple in terms of both shape and appearance. It's suggested to evaluate on more complex real-world images such as ScanNet[1]. 

3.3. For the evaluation metrics of object segmentation, only ARI related scores are reported. It's suggested to report additional metrics such as AP scores as analyzed by [2] in the community of unsupervised object segmentation. 

3.4. For the experimental results in Tables 1/2/3, why are the existing methods inferior in both segmentation and novel view synthesis? It's suggested to give more concrete discussions. 

3.5. There is a lack of ablation studies to analyze the effectiveness of the proposed components. 

[1] ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes, CVPR'17
[2] Promising or elusive? unsupervised object segmentation from real-world single images, NeurIPS'22


4. Minor suggestions:

4.1. In page 2, the claimed first contribution seems not very meaningful at all. 
4.2. For equations 1/2/3/4/5, it's suggested to use mathematical symbols rather than English words. It's a bit hard to track the meanings.
4.3. The work ONeRF [3] should be discussed, as it also uses NeRF for unsupervised 3D object discovery. 

[3] ONeRF: Unsupervised 3D Object Segmentation from Multiple Views, arXiv'22

### Questions
Details given above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a method to infer 3D object centric representations from a single RGB image. Unlike previous works, the proposed method disentangles the intrinsic and extrinsic properties of an object when learning the scene representation, which allows them to generalize better. The paper shows various application of their method - novel view synthesis, segmentation, object removal and scene rearrangement.

### Strengths
1. The paper is clearly written and has good figures which help in getting the point across.
2. The main novelty of the work is in disentangling the intrinsic and extrinsic properties of an object when learning object centric representations. This helps the method to generalize better. 
3. The paper also releases 3 new datasets to help test this approach which can help the research community to build on top of the method.
4. The paper shows excellent results on different tasks - scene segmentation, novel view synthesis, object re-arrangement, object removal.

### Weaknesses
The major weakness of the paper is the lack of ablations performed. The paper mentions various design choices, however, I didn't find any ablations in the main paper justifying any of them. For example, the paper mentions that 3 stage training helps in getting a better performance, however there is no ablation to justify that. What if you skip the stage 2 and directly train on stage 3 after learning object priors using stage 1? How much will the performance degrade? How much does stage 2 training actually help here?
Also, authors mention learning a global representation using ViT and a local representation using shallow UNet. Again, I didn't find an ablation justifying using two separate networks for these representations. Why not use some intermediate layer (or the spatial features) from ViT as local features? What's the intuition behind shallow UNet? Is it to prevent features from different objects entangling with each other? The authors should provide such ablations/intuitions in the main paper.

### Questions
I would like authors to address the points raised in the weakness section.
1. What is the impact of stage 2 training? How much value is it adding? What if you skip stage 2 training and directly train on real datasets in stage 3?
2. What is the significance of using a separate shallow UNet for local features? Why not just use the intermediate features from ViT as local features.
3. How do authors deal with pose ambiguity. Most of the objects used in the paper are symmetric (plates, chairs, etc). Will that symmetry cause problems or make the method stuck in some local minima?

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper addresses the challenge of learning single-view RGB object-centric representations with multi-view supervision tasks. The author presents an object-centric learning framework built upon the foundation of uORF, which incorporates Slot-attention, a pre-trained DINOv2 encoder, and compositional NeRF.

The training process is carried out in multiple stages. 
In the initial stage, the model is trained on synthetic scenes, each featuring a single object. 
Subsequently, the training extends to synthetic scenes containing four objects. 
Finally, the model undergoes further training on a real-world dataset characterized by a substantial domain gap in comparison to the synthetic dataset.

### Strengths
The exploration of object-centric learning within real-world datasets is an important research direction. This study takes a step in the pursuit of this objective. The writing is largely lucid and comprehensible. Notably, the reconstruction and segmentation outcomes outperform previous methods, particularly in scenes featuring intricate textures.

### Weaknesses
This model is essentially an integration of prior research efforts, with key components such as slot-attention, DINO, and object-compositional NeRF having been individually introduced in previous works. Consequently, the training pipeline itself lacks a significant degree of novelty.

I have major concerns regarding the utilization of a pre-trained DINO encoder. DINO is already proficient at extracting meaningful and somewhat object-aware features from images, leading to the possibility that the model merely aligns these features with latent variables. This raises questions about how much the proposed pipeline genuinely contributes to object discovery.

The "l_pos" loss, which calculates the positional disparity between object poses from different viewing directions, relies on the fact that, during the initial training stage, there is only one object per scene, eliminating object-matching challenges. However, strictly speaking, the use of a single-object dataset disqualifies the proposed pipeline as an unsupervised model while making the learning problem considerably less challenging. To apply this method to a general dataset, one must manually select scenes containing only one object, a process akin to labeling. In other words, the training pipeline appears tailored specifically for the proposed dataset. While the author presents object-centric prior learning as a contribution, it might be perceived more as a limitation.

The fixed number of objects in each scene restricts the method, while most object-centric learning frameworks merely set a maximum number of objects in the scenes.

The limitations of the work are not adequately discussed or disclosed.
The ablation study is rather limited in scope.

### Questions
Could the author provide insights into why the proposed method demonstrates superior background reconstruction quality compared to uORF? While the inference pipeline shares many similarities with uORF, the reconstruction quality appears to be significantly enhanced.

It would be valuable to know the specific value of the parameter "$K" employed during the training process.

Is it possible for the author to conduct training and evaluation of the proposed method on the Room-Chair dataset, which was utilized in the uORF paper?

Moreover, it would be beneficial to observe results on real-world datasets without the pre-training on synthetic datasets. It remains unclear why pre-training is deemed essential, especially when facing a substantial domain gap.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
