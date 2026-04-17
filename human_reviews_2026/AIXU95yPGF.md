# Physics-Inspired Reconfiguring Multimodal Learning Networks

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Despite recent progress, current multimodal fusion methods still face three practical issues: gradient interference between task and fusion objectives, fragility under missing modalities, and rigidity from enforcing uniform feature dimensions across modalities.
We present Physics-Inspired Multimodal Reconfiguration (PMR), a Poisson–Nernst–Planck (PNP)–inspired structured prior for fusion. Drawing from the principles of conservation and single-potential-driven flow, PMR embeds these as (i) an information-preservation regularizer and (ii) a unified scalar potential that shapes gradient updates, mitigating interference between task and fusion objectives. This unified potential drives disentanglement of shared and modality-specific subspaces. A three-stage mapping (dissolution → dissociation → concentration) instantiates the prior to separate and reconstruct features, improving robustness to missing modalities and naturally supporting unequal feature dimensions.
Across audio, image, video, and text, PMR consistently outperforms competitive baselines on classification and cross-modal retrieval, demonstrating the efficacy of a physics-inspired hybrid prior for multimodal learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new method to fuse information in multimodal representation learning for retrieval and classification tasks. It is inspired by the Poisson-Nernst-Planck structured prior and it is validated on image-video, audio-video retrieval tasks and image classification tasks.

### Strengths
-	Multiple tasks are considered to validate the models (retrieval, classification) and various baselines are given.
-	The proposed model apparently gives competitive results across all benchmarks.

### Weaknesses
-	The justification for the proposed method is mysterious, with a lot of background from theoretical physics that has nothing to do with the final loss. The parallel between “Poisson-Nernest-Planck” and the fusion module is far-fetched and never justified throughout the experiments. This module corresponds more or less to a simple MLP with some neurons arbitrarily set to zero to extract the “shared” or “specific” information from different modalities (which is never imposed directly with the proposed reconstruction losses). 
-	A lot of hyper-parameters are introduced (such as the length of the splits for the shared/specific latent vectors for each modality or the actual structure of the mapping between the extracted features and the output of the fusion module) but they are not discussed or mentioned in the experimental section. How did you pick them and what are the final choices?
-	Two reconstruction losses are mentioned in equation 6. Which one did you use in the end and why? Why did you introduce a cosine similarity in one case and an Euclidean distance in another? What is the rational?
-	In equation 5, you mention that k is the “next modality of m”. So how do you handle the case where m is the “last” modality (assuming that an order exists between modalities) ? 
-	In your reconstruction losses, how do you make sure that the “specific” information does not contain “shared” information as well, which would be enough to minimize your reconstruction loss. 
-	In your experiments, I would expect the feature extractors to be state-of-the-art foundation models (such as DINOv3, CLIP…) in order to have strong baselines in retrieval or classification tasks. 
Overall, this paper pretends to draw a parallel between equations from theoretical physics (drift-diffusion transport, field-charge coupling) and deep neural networks but their analysis is never justified theoretically or empirically. I have also strong doubts about the fairness and reproducibility of their experiments considering the little implementation details given and the high number of hyperparameters introduced by the proposed fusion module.

### Questions
Please, see my previous comments.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents PMR, a physics-inspired multimodal fusion framework derived from the Poisson–Nernst–Planck equations. PMR aims to address three key challenges in multimodal learning: gradient conflicts between task and fusion objectives, robustness to missing modalities, and the limitations of uniform feature dimensions. The framework unifies task and preservation losses through a single scalar potential and introduces a novel three-stage feature reconfiguration process (dissolve–dissociate–concentrate). Experimental results show that the method provides stable optimization and enhances model flexibility and robustness. PMR demonstrates consistent improvements over baseline methods on classification and retrieval tasks across various modalities including audio, image, video, and text.

### Strengths
1. The paper provides a novel perspective on multimodal fusion by leveraging principles from physics.
2. The addressed challenges are both relevant and critical to the field of multimodal learning.
3. The paper gives a comprehensive summary of related work and situates the proposed method well within prior research.

### Weaknesses
1. While Chapter 3 introduces a novel perspective, the connection between its formulations and the algorithm in Chapter 4 is not clearly established. The mathematical formulations seem more like background information rather than serving as a foundation for the proposed algorithm.
2. In Chapter 4, it is not clear why only two mapping networks are sufficient to distinguish between shared and specific features, or whether their separation can be strictly guaranteed.
3. The algorithmic section lacks sufficient detail and clarity; it appears more akin to a multi-task learning approach rather than a fundamentally new method for multimodal fusion.
4. The experimental results are not particularly compelling. In Tables 3 and 4, the performance is comparable to DrFuse. Additionally, the results in Table 5 show the unimodal approach outperforming others in some cases, which is insufficiently explained.

### Questions
Please check the above section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes PMR (Physics-Inspired Multimodal Reconfiguration), a new multimodal fusion framework inspired by the Poisson–Nernst–Planck equations, which enforces information conservation and uses a unified scalar potential to jointly optimize task and fusion objectives, thereby mitigating gradient interference. By implementing a three-stage dissolve–dissociate–concentrate process, PMR supports unequal feature dimensions, improves robustness to missing modalities, and consistently outperforms strong baselines across audio, image, video, and text tasks.

### Strengths
(1) It's very interesting to design a new neural network using the framework of the Poisson–Nernst–Planck equations.
(2) The paper is well-written and nicely presented.

### Weaknesses
(1) The experiments in the paper are very weak, all conducted on some rather outdated datasets.  
(2) The paper lacks comparison with state-of-the-art multimodal large models. Current multimodal models are capable of processing multimodal information and extracting embeddings to tackle a wide range of downstream tasks.  
(3) It also omits comparisons with other multimodal fusion approaches, such as ImageBind ("ImageBind: One Embedding Space To Bind Them All," CVPR 2023) and OnePeace ("ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities," arXiv 2023).  

Overall, the paper presents an interesting idea, but it lacks large-scale, compelling experimental validation.

### Questions
I checked many of the methods compared in the paper (e.g., MDF-FND) and found that none of them reported results on the corresponding datasets. I’m curious: where do the results reported in the paper come from? Did the authors implement these methods themselves? If so, how was fairness ensured? Additionally, could the authors clarify how they selected these baseline methods for comparison?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose PMR, a physics-inspired multimodal fusion framework that is built upon a scalar potential that minimizes entropy and preserves the system's energy. The PMR imitates the PNP theory well by its 3-stage mapping reconfiguration. The experiments demonstrate the effectiveness of the framework design, and the ablations are sufficient and convincing.

### Strengths
1. The paper is well-written.
2. The authors explain the PNP theory well and use it to develop a new multimpdal encoding framework PMR.
3. The experiments demonstrate the effectiveness of the proposed method.
4. The ablation is sufficient to validate the efficacy of the components in PMR.

### Weaknesses
Major:
1. Line 137-147, consistently minimizing the entropy for all modalities means that we hypothesize all modalities contribute the same to the final task. But does this really apply to real-world scenarios, when there are always some modalities that are dominant, while some are just complementary?

2. For Eq.(4,5,6), how to ensure that the b(m) is meaningful to separate the shared and specific features? Is b(m) learnable? And how? (In Figure. 3 I saw b is a hyperparameter.) But according to the PNP model, the b for different substances should be different, which represents a balanced state for different substance pairs; however, in PMR, b is the same for different modalities. 

3. The whole process of PMR looks like just a separate and shared encoder for different modality features and uses a straightforward fusion strategy to obtain joint embeddings. The connection between PNP and PMR is not that close, especially considering that some core mechanisms are different, as mentioned in point 2 that the boundary b is the same for all modalities.

4. It looks like PMR could fit any number of modalities; however, in the experiments, the authors only showcase two-modality experiments. Some tri-modal validation could be better. For example, CMU-MOSI(a+v+t) and UCF101-Three(rgb, optical flow, and rgb diff). 

Minor:
1. The Z should be V in line 121?

### Questions
1. Line 187, how to understand this sentence: "Increasing the effective length enlarges the region where drift dominates diffusion"?
2. In Table 2, how to understand: (1) Feature ratio and learning splitting at b; (2) Magnification factor (effective length)
3. In line 449, what is the actual meaning of nxb? What about without P_seperate, i.e., b=1? Or b=0?

### Soundness
3

### Presentation
3

### Contribution
3
