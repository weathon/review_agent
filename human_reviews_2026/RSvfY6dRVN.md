# Learning and Reusing Abstract Latent Actions in a Hippocampal-Entorhinal-Inspired World Model

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 2, 8, 4

## Abstract
Humans are capable of abstracting dynamic experiences into structured representations, facilitating both the inference of shared patterns by observing similar transition dynamics and the transfer of these structures across varied contexts. The hippocampal-entorhinal circuit, widely known for its role in spatial navigation, also supports the representation of abstract conceptual spaces crucial for non-spatial cognitive processes. This function emerges from the distinct yet integrated encoding of content-specific details by the hippocampus and abstract structures by the entorhinal cortex, facilitating structural generalization across varied contexts. Although the hippocampal-entorhinal circuit has been previously explored as a predictive system for binding contents, the process for concurrently extracting abstract structures from continuous real-world dynamics remains largely understudied. In this work, we propose a computational model inspired by the hippocampal-entorhinal circuit, capable of simultaneously inferring latent actions to form abstract structures and constructing predictive world models from real-world video sequences. Our model combines an inverse model for extracting abstract latent actions with a hippocampal-entorhinal-inspired coupling model that separately encodes contents and structures, leveraging action-driven path integration for prediction. Experimental results demonstrate that our model effectively captures abstract latent actions, reuses them robustly across diverse contexts, and achieves reliable predictive performance in both familiar and novel environments. Additionally, our analysis of latent representations from 3D object rotation datasets highlights why latent actions extracted through entorhinal cortex representations demonstrate greater abstraction and reusability. This work provides novel insights into the brain-inspired mechanisms underlying the self-supervised learning of abstract latent actions and world models from real-world dynamics, illuminating cognitive processes essential for transfer learning and data-efficient learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel architecture of biological implementation of world models, inspired by neuroscience technology: HPC-MEC circuits -- Hippocampus (HPC) binds content-specific information from contents and Medial Entorhinal Cortex (MEC) extracts abstract structures among them. The proposed architecture consists of two main components: HPC-MEC coupling model and Inverse Model. HPC-MEC coupling model has a hierarchical encoder-decoder architecture and mainly works for extracting low-level dimensional MEC embeddings, so as to interact with latent actions using CANN. On the other hand, inverse model extracts latent actions between the given two MEC embeddings. As experimental results, this approach generally outperforms other strong baselines on 7 different tasks, along with OOD generalization tasks. Interestingly, authors also provide latent space analysis on rotated object tasks, which implies the model's feature shapes abstract structures while in-class differentiation is formed.

### Strengths
1. **Strong Novelty.** Drawing from HPC-MEC circuit behavior observed in neuroscience to implement a biological world model represents a solid research contribution. The connection is not forced, and the motivation is well-justified. The approach also offers interesting points of departure for future work.

2. **Clear Illustration and Explanation.** Figures 1 and 2 effectively present the overall approach in a comprehensive manner. The figure quality is excellent and shows a polished presentation. Furthermore, even when explaining complex architectural details, the paper maintains clarity through step-by-step descriptions of the actual architecture's components. The explanations are detailed but accessible, making it easy to follow the proposed framework and related work.

3. **Comprehensive Experiment Design.** The evaluation uses three different categories and various objectives across 7 different datasets, providing substantial evidence of the model's behavior. Notably, the rotation task and OOD task evaluations, which specifically leverage the HPC-MEC circuit properties, demonstrate significant advantages and offer valuable insights for future research.

### Weaknesses
- **Downstream Tasks with Learned World Models.** The paper only evaluates the quality of generated video output compared to ground truth video, which may not sufficiently demonstrate the model's ability to capture world physics. While this is acceptable given the paper's focus on more fundamental problems, including a few downstream tasks (e.g., robotic task performance using this world model) could provide stronger evidence of the advantages that the HPC-MEC module offers for robotics applications.

### Questions
- The latent space analysis was quite impressive. To clarify, in Figure 5-B, g^{gen} shows overlapped representations between two colored apples, but p^{gen} successfully differentiates them. How can this behavior be explained?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The method learns an action representation by adding additional layers in the bottleneck of VAR [1], a frozen autoregressive video VQ-VAE. The method uses the encoder to obtain observation embeddings, then infers latent actions via an inverse model and applies them within a hierarchical Continuous Attractor Neural Network (CANN) structure for path integration. The model is evaluated on Something-Something v2, COIL-100, and simulated OOD benchmarks both qualitatively and in terms of SSIM and LPIPS against three world modeling baselines.

[1] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

### Strengths
1. The method avoids any action labels and builds the latent-action space through inverse modeling in the bottleneck of VAR.

2. Evaluations are on real and simulated datasets and the authors motivate the method by action transfer across visual domains. 

3. The motivation and method are motivated by biology (HPC-MEC circuit) and qualitative evidence for emergence of separation between HPC and MEC in terms of content features and geometric features is provided (Fig 5).

### Weaknesses
1. My main concern is that the frozen VQ-VAE backbone differs from those used in LAPA, Moto, or AdaWorld but no comparison is shown to the pretrained autoregressive VAR model itself [1]. This makes it unclear whether performance improvements and action stem from the new architecture or from feature quality of the original VAR.

2. Following my point above, the hierarchical design (transformers + CANN module) is heavy yet yields modest practical advantages. There is no ablation to show how the pretrained VQ-VAE performs without these components. Furthermore, it is unclear whether the CANN-based path integration outperforms a simpler architecture (e.g. MLP) of equal capacity when trained under identical conditions. 

3. The qualitative evaluation results do not show robustness or realism for all types of actions (e.g. Fig. 4 D and E). Furthermore, the autoregressive generated trajectories seem to be prone to compounding error (e.g. Fig 4 C).

4. Although the method claims to learn latent action representations, no metric is provided for decoding actions from the learned latents. Perceptual similarity metrics are not a suitable proxy for direct action decoding.

5. The neuroscience background is verbose and disconnected from the rest of the paper.

[1] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

### Questions
1. How does the model compare directly to the underlying VAR without adding HPC-MEC layers?

2. Can you provide quantitative disentanglement metrics (e.g., mutual information between latent actions and pixel-space transformations or decoding actions from HPC vs MEC tokens) rather than qualitative UMAPs?

3. Does fine-tuning the backbone and decoder jointly while learning the HPC-MEC layers improve performance?

4. Can you provide compute and wall-clock time comparison between the method and the baselines (including original VAR)?

5. Does the model enable any improvement in downstream RL and control tasks relative to baselines like LAPA?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a cognitive-/neuro-science-inspired artificial neural network architecture that implements a disentangled world model that learns abstract latent actions to predict next-state observations purely from video frames in a self-supervised manner.

The architecture uses a pretrained frozen vector-quantized autoencoder to generate initial frame embeddings, which are then processed by a stack of temporal causal transformers with dimensionality reduction in the latent codes. A Continuous Attractor Neural Network (CANN) predicts the delta to the next latent state given a latent action state and is regularized to be close to this delta. The latent action comes from an action model that generates this latent action code given two consecutive latent codes.

### Strengths
- Reproducibility: The provided source code is structured and easy to understand

 - Learning of visually independent latent actions

 - Strong autoregressive performance

 - Ablations show the relevance of the used predictive CANN model

### Weaknesses
- The authors claim “MEC captures shared dynamics across objects, while the HPC retains object-specific information,” but as I understand, MEC and HPC are both temporal causal transformers, with HPC having a narrow bottleneck, so how is this claim justified?

- Temporal recurrence in Figure 2 is a bit misleading since a causal transformer is not really recurrent in the RNN sense, so I would rather call this temporal information flow or something in that spirit.

- The ablation models need to be explained in more detail; for example, what exactly is meant by “unified space model”?

- More details on how the CANN was ablated would be helpful. I am not fully convinced that a transformer-like architecture with the same inputs and regularizations as the CANN would perform much worse.

- The abstract is a bit hard to read and could be addressed more toward a computer science audience. Think of it more like an advertisement of your paper; currently the cognitive and neuro-science terminology makes it a bit hard to parse what the paper will be about and what the exact contributions are.

- Figure 2 “dashed green arrow” should be “dashed purple arrow.”

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a self-supervised framework that can simultaneously learn latent actions and a world model directly from real-world videos. The approach is inspired from the hippocampal - entorhinal (HPC - MEC) circuit, a biological system known for encoding spatial and abstract structures in the brain. The model separates representations into content-specific (HPC) and structure-specific (MEC) components. Latent actions are learned via an inverse model operating in the MEC latent space, where the authors employ Continuous Attractor Neural Network (CANN) dynamics to represent velocity-like transformations.

The proposed model is evaluated on Something-Something V2, COIL-100, and several robotic simulation datasets (e.g., Franka Kitchen, Push-T). Results show strong generalization and transfer - latent actions extracted from one context (e.g., a hand motion or object rotation) can be reused to predict analogous dynamics in new visual scenes. Quantitative results on SSIM and LPIPS confirm better visual fidelity and lower perceptual error than baselines such as LAPA, Moto, and AdaWorld.

### Strengths
(+) The paper offers an elegant synthesis between computational neuroscience and self-supervised world modeling. This connection gives the model interpretability and conceptual depth beyond typical deep learning architectures.

(+) Modeling latent actions as velocity operators within a CANN-inspired latent manifold provides a semantically meaningful alternative to tokenized or discrete latent actions (as in VQ-VAE methods).

(+) The system successfully transfers learned latent actions to unseen contexts: including cross-object, cross-domain, and cross-dataset generalization. Performance on OOD datasets like COIL-100 and Franka Kitchen is notably strong relative to prior work.

(+) The authors conduct extensive analyses (e.g., UMAP visualization, in-class structural sharing, cosine similarity of transitions) that clarify how HPC and MEC embeddings differ in representational function.

### Weaknesses
(-) Evaluation remains perception-oriented. While the model demonstrates visual prediction and generalization, it has not yet been tested in interactive or control settings where learned latent actions must guide decision-making or robotic policies. The connection to actual action planning thus remains conceptual rather than empirical.

(-) Limited ablation on factorization design. The paper assumes the separation between content (HPC) and structure (MEC) is beneficial but provides only one ablation with a unified latent space. Addition studies, such as varying the coupling strength or testing partial disentanglement, would strengthen the argument for this design.

(-) Biological analogy not fully justified. While the HPC–MEC analogy is intellectually appealing, the correspondence between biological circuits and deep modules is mostly metaphorical. The paper could temper claims about biological plausibility and focus more on empirical validation.

### Questions
How does the choice of latent action dimension and number of CANN modules affect generalization and reconstruction quality?

Can the proposed framework be extended to goal-conditioned prediction or hierarchical planning (e.g., using latent actions as building blocks for policies)?

Does the system exhibit any drift or instability when performing long-horizon autoregressive predictions without visual feedback?

Have the authors compared training with and without the fixed pretrained VQ-VAE encoder to assess the benefit of freezing visual features?

Could the learned latent actions be composed or interpolated to form new dynamics (e.g., blending rotation and translation)?

### Soundness
3

### Presentation
2

### Contribution
2
