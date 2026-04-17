# Privacy-Aware Hybrid Image Synthesis with Local-Cloud Collaboration

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Recent advances in diffusion-based generative models have enabled high-quality and personalized image synthesis. However, protecting user privacy while enabling efficient image generation remains a major challenge when deploying diffusion models on edge devices. Cloud-based inference risks exposing sensitive content, whereas fully local execution demands excessive computation and memory. This calls for a collaborative edge–cloud paradigm that can balance both concerns. In this work, we propose PrivInfer, a training-free, privacy-preserving inference framework that enables efficient edge-cloud collaboration for image generation. PrivInfer decomposes the generation process by region: privacy-sensitive areas are processed locally, while non-sensitive regions are offloaded to the cloud. This design reduces on-device computation while minimizing privacy risks. To ensure secure cross-device interaction, we introduce a more secured mechanism that shares structural information without exposing raw features. We further develop a ring-based masking strategy to structurally isolate private content during convolution, and a heterogeneous-step scheme that enables low-step local models to leverage high-fidelity cloud features. Extensive experiments show that PrivInfer significantly reduces inference steps and computational load on edge devices, while maintaining high generation fidelity and strong privacy protection, which offers a practical solution for private and efficient diffusion model deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PrivInfer, a training-free, privacy-preserving inference framework for efficient diffusion-based image generation through local–cloud collaboration. The key idea is to divide image synthesis by spatial regions—sensitive areas (e.g., faces) are processed locally on the user’s device, while non-sensitive regions are offloaded to the cloud. To enable this collaboration securely and efficiently, PrivInfer introduces Gram Matrix Communication, which transmits only aggregated attention statistics instead of full feature maps (Keys/Values), preventing exposure of private visual features. Moreover, it proposes Ring-Based Masking, which structurally isolates private regions during convolution operations, ensuring that private pixels do not leak into shared computations. Finally, it suggests Heterogeneous-Step Asynchronous Interaction, whcih allows local models to use cached or server-guided features and operate with fewer denoising steps than the cloud model—reducing computation and latency. The proposal has been experimentally evaluated on text-to-image (MJHQ-30K) and image-to-image (COCO 2017) tasks.

### Strengths
1. The paper proposes a hybrid inference paradigm that enables privacy-preserving and computationally efficient diffusion-based image generation through collaborative processing between local and cloud models. Its central idea—region-wise decomposition of image synthesis and secure cross-device interaction via Gram matrix communication—is a novel contribution that effectively mitigates privacy risks without retraining. 

2. The presentation is clear and well organized, with figures and explanations that make the framework easy to follow. 

3. The significance of this work lies in addressing a timely and practical challenge—how to deploy large generative models in privacy-sensitive and resource-limited settings—making it highly relevant to both academic research and real-world applications in edge AI, healthcare, and personalized content generation.

### Weaknesses
1. There is a lack of formal privacy guarantees for the proposed Gram matrix communication mechanism. While transmitting Gram matrices instead of full key–value features reduces direct exposure of private representations, this approach provides only structural or empirical privacy rather than provable privacy protection. In particular, although the Gram matrix is non-invertible in theory and hides exact feature vectors, an adversary with sufficient side information (e.g., model parameters or multiple correlated Gram matrices) could potentially approximate sensitive features or infer partial content correlations. Thus, the method enhances privacy awareness but does not ensure rigorous security comparable to differential privacy, homomorphic encryption, or secure multi-party computation. Clarifying the threat model and providing either theoretical analysis or empirical validation of privacy leakage would strengthen the paper’s claims.

2. It is standard practice to include the code implementation, either as a zipped file in the supplementary material or through an anonymous link. Providing access to the implementation enhances transparency and significantly improves the reproducibility of the results.

3. In the experiments, private areas are either synthetically defined (e.g., the central region of the image) or derived from dataset bounding boxes, rather than detected based on privacy semantics such as faces or personally identifiable details. Moreover, in datasets like COCO, bounding boxes typically cover the entire object, while in real-world scenarios only subregions of an object (e.g., a person’s face rather than their whole body) may be privacy-sensitive. The framework therefore assumes that the private region is known and uniformly defined in advance, which may limit its applicability to more complex or realistic privacy settings. Incorporating or discussing finer-grained or automatic privacy-region identification would strengthen the work’s practical relevance.

4. While the proposed design conceptually reduces transmitted information through Gram matrix aggregation, ring-based masking, and asynchronous updates, the paper does not provide quantitative measurements of the actual communication volume or latency. In realistic edge–cloud scenarios, network bandwidth and transmission time can become the dominant bottlenecks, potentially offsetting the computational savings achieved on the local device. A more detailed evaluation of the communication–computation trade-off, including scalability with image resolution and network conditions, would be essential to assess the practical efficiency of PrivInfer.

Minor weaknesses:
1. The manuscript uses the acronym MLP without defining it at first use. Please expand it to multilayer perceptron the first time it appears.

2. The mapping  ϕ(⋅) in Eq. (4) is undefined.

### Questions
1. The paper claims privacy protection through Gram matrix communication, but it remains unclear what type of adversary or attack model is assumed. Could the authors clarify whether the privacy guarantee is meant to be empirical or formal (provably non-invertible under certain assumptions)? Have the authors evaluated or simulated potential feature reconstruction attacks from Gram matrices to empirically verify the claimed privacy level?

2. The framework assumes the private region is known in advance, but in practice, privacy-sensitive content may be unknown or fine-grained (e.g., a face rather than a whole person). Could the authors elaborate on how their method would integrate with automated privacy-region detection (e.g., face or object segmentation)? How would the system handle partially sensitive regions within a bounding box?

3. While the proposed mechanisms conceptually reduce transmission load, the paper does not provide quantitative measurements of communication cost or latency. Could the authors report empirical data on (a) the amount of data transmitted between client and server per image, (b) communication time under realistic bandwidths, and (c) how performance scales with image resolution or network delay?

4. Table 1 includes metrics both with respect to the ground truth and to the original model outputs. Could the authors clarify why both are used and whether they correspond to different evaluation goals (absolute vs. relative fidelity)? Would results measured only against ground truth lead to similar conclusions about quality retention?

5. The experiments focus on the SANA model and specific datasets. Could the authors comment on whether PrivInfer can generalize to other diffusion architectures (e.g., SDXL, Flux) or modalities (e.g., video generation)? Are there architectural dependencies that might limit portability?

6. Since PrivInfer involves real-time collaboration between local and cloud models, what are the authors’ thoughts on potential security and trust issues during communication? For instance, could model inversion or man-in-the-middle attacks compromise the privacy guarantees, and how might encryption or secure channels mitigate this?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
PrivInfer proposes training-free local-cloud collaboration for diffusion image generation. The method partitions images into privacy-sensitive and non-sensitive regions, processing them locally and on cloud respectively. Cross-device communication transmits only Gram matrices rather than raw key-value features. Ring-based masking isolates private content during convolution. Heterogeneous-step scheduling allows local models to use fewer inference steps while leveraging cloud features. On COCO 2017, five local steps achieve LPIPS 0.198 versus original 20-step outputs, compared to 0.316 for single-device five-step inference. MACs reduce based on privacy region ratio (estimated 11.11% from COCO bounding boxes).

### Strengths
- Training-free deployment removes need for model fine-tuning or retraining.
- Five-step local inference approaches 20-step quality when leveraging cloud features (Table 3: LPIPS 0.198 vs 0.316 for pure local).
- Ring-based masking provides explicit isolation for convolutional privacy boundaries.
- MACs analysis demonstrates computational reduction potential (Table 2: 0.84T vs 21.65T).
- Method compatible with existing diffusion architectures using linear attention.

### Weaknesses
- My primary concern is that the paper's privacy claims rest heavily on a theoretical property (Eq. 7-8) without sufficient empirical validation. While orthogonal ambiguity is noted, it's known from style transfer that Gram matrices still encode rich texture information (like hair or skin patterns). The argument would be much stronger if the paper included experiments simulating reconstruction attacks to show what is practically recoverable.
- Similarly, the effectiveness of the ring-based masking was not clearly demonstrated. The analysis (p.5) seems to only account for a single 3x3 kernel. However, in a deep U-Net, receptive fields expand significantly, creating complex pathways for leakage. The paper would be more convincing with ablation studies comparing ring vs. no-ring baselines and quantifying leakage rates at different layers.
- The results for the heterogeneous-step scheduling (Table 3) appear contradictory. While it improves LPIPS against the original 20-step output, it sometimes degrades quality against the ground truth (0.717 vs 0.685). This strongly suggests a feature distribution mismatch between the asynchronous steps.
- The proposed communication strategy is explicitly tied to linear attention. This potentially limits the method's applicability, as many modern diffusion models use other architectures (like flash attention). The paper would benefit from a discussion on how the framework could adapt to these other models.

### Questions
- To help substantiate the privacy claims, could the authors provide reconstruction experiments? Specifically, what level of visual information (e.g., textures, patterns) is an adversary practically able to recover given the intercepted Gram matrices and mask coordinates?
- For broader applicability, could the authors elaborate on how the Gram-only communication strategy might be adapted for models using non-linear attention (like flash attention or full attention)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose PrivInfer, a framework that balances privacy protection and computational efficiency in image generation systems.  The framework processes privacy-sensitive regions locally while offloading non-sensitive areas to cloud servers by using only gram matrices rather than raw features. They also use a ring-based masking strategy to structurally isolate private content during convolution operations. Additionally they propose a training scheme, where the local and server models run at different steps.

### Strengths
1. The paper proposes a solution for an important problem: training models on privacy data via servers pose a big privacy risk. The method of  transmitting only gram matrices is a clever trick used in this scenario.
2. Apart from privacy protection, having server and local process at different frequencies enable fewer training iterations.
3. The technique also works with pre-trained models and suitable for real world deployment.

### Weaknesses
1. The paper does not have enough ablation studies:
a. Bandwidth requirements: How is the communication overhead with different image sizes?
b. Private image size: How does this method perform in case of varying privacy sub-image sizes?

2. It is unclear how the performance and privacy trade-off works in this method. Can we control the amount of privacy in this method?

3. The paper does not provide mathematical guarantees that gram matrices are privacy preserving. For instance, multiple gram matrices at multiple training steps, can it be used to reconstruct the data?

### Questions
1. Can you provide theoretical guarantees or bounds on what information is preserved/hidden by the Gram matrix transmission?
2. Could accumulating of Gram matrices across multiple timesteps enable reconstruction of private data?

### Soundness
3

### Presentation
3

### Contribution
4
