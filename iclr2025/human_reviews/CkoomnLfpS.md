## Human Reviewer 1

### Summary
The paper studies the problem of using neural networks as implicit neural representations (INRs). To address the overlooking of the sequential nature between neural layers, the authors propose representing neural network parameters from a dynamic graph perspective. Based on this core idea, the paper develops Dynamic Neural Graph Encoder and INR2JLS. Finally, the authors conduct experiments on various tasks to verify the proposed method.

### Strengths
- The paper studies INRs from a dynamic graph perspective, which is a new perspective.

- The paper is well-structured.

### Weaknesses
- Although the paper uses a new perspective (dynamic graph) to study INRs, regarding the weight parameters as dynamic graphs may not be an effective approach. The neural graph between different layers changes significantly, making it challenging to capture their sequential and evolving characteristics accurately.

- Are there any experiments that prove the improvement introduced by capturing the sequential nature between layers?

- In the related work section, the paper lacks a deeper analysis to describe differences between the proposed method and static graph counterparts.

- The proposed method can regard transformers as dynamic neural graphs, but there lack experiments on the transformers architecture to validate the proposed method.

- Minor typos: “dynmaic” in lines 181, 663, and 691 should be “dynamic”

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper proposes a novel approach to model neural networks as dynamic graphs for capturing layer-by-layer dependencies. The authors introduce an RNN-based method named the Dynamic Neural Graph Encoder (DNG-Encoder) for capturing temporal dynamics, which can mirror forward passes and preserve sequential characteristics of neural networks. Additionally, the authors present a new framework named INR2JLS, which can map neural weights and data into a unified latent space for enhancing the quality of representations. Extensive experiments are conducted to demonstrate the effectiveness and significant improvements of the proposed method.

### Strengths
- Different from traditional methods, this paper introduces a novel appoach (The Dynamic Neural Graph Encoder, DNG-Encoder) to model neural networks as dynamic graphs, effectively capturing temporal dependencies across layers and providing more accurate representations for forward passes of neural networks.
- The INR2JLS framework proposed in this paper can map neural weights and data into a unified latent space, which can enhance the quality of representations and improve the model performance particularly for challenging applications like implicit neural representation (INR) classification.

### Weaknesses
- In Section 3.1, the authors build on the work proposed by [1] and suggest that the natural symmetries in graphs align with neuron permutation symmetries in neural networks. However, as this paper focus on dynamic graphs, differing from the static graph setting in [1], the claim of invariance or equivariance to permutation symmetries requires further proof in the context of dynamic graphs.

- In Section 4.1, the authors use an RNN-based method to model the dynamic behaviors of neural networks; however, gradient vanishing and explosion are common issues in RNN-based methods. Specifically, as the size of neural networks or graphs increases, the dynamic model need to operate over more timesteps, increasing the likelihood of these issues occurring. It would be valuable to explain how the proposed methods how to address these two challenges, supported by a theoretical analysis.

- In Section 7, the authors provide only the experimental results on computational complexity. A more detailed theoretical analysis of time and space complexity should be included for fair comparisons with baseline methods [1], [2], and [3].

[1] Kofinas M, Knyazev B, Zhang Y, et al. Graph Neural Networks for Learning Equivariant Representations of Neural Networks[C]//The Twelfth International Conference on Learning Representations.

[2] Zhou A, Yang K, Jiang Y, et al. Neural functional transformers[J]. Advances in neural information processing systems, 2024, 36.

[3] Zhang D W, Kofinas M, Zhang Y, et al. Neural networks are graphs! graph neural networks for equivariant processing of neural networks[J]. 2023.

### Questions
- Please provide further theoretical proof of invariance or equivariance to permutation symmetries within the context of dynamic graphs.

- Please explain how the proposed method addresses gradient vanishing and explosion issues, both experimentally and theoretically.

- Please demonstrate the scalability of the proposed method, showing its deployment on large-scale neural networks and graphs.

- Can the proposed methods be applied to other types of neural networks?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper proposes to model neural network weights as dynamic neural graphs. Such an approach addresses the limitations of previous "static" neural graphs by more closely mimicking the forward pass of MLPs/CNNs and thereby simplifying the task of learning from the weights. The paper proposes to use dynamic GNNs to learn from dynamic neural graphs and shows better results on the tasks such as INR classification and predicting CNN generalization.

### Strengths
1. The observation that static neural graphs are not well aligned with the forward pass is original and interesting.
2. The overall idea of using temporal GNNs is logical and novel in this context.
3. The idea of joint weight and image space (INR2JLS) is interesting and novel.
4. The experiments show improvements over the baselines.

### Weaknesses
1. The motivation of processing INR weights in the Intro is not convincing. For example, the authors say "This observation has motivated  us to investigate the potential for directly processing INRs to uncover information about the data they encode." It's not very clear uncovering which data the authors imply and why we need to uncover them. It seems that INR classification appeared as a task in the previous literature mainly because it's a convenient testbed for this kind of methods. But recent papers in this domain often add other more practically relevant use-cases (e.g. learning to optimize in Kofinas et al. or processing the weights of diverse transformers in [a]), which makes the motivation of these methods more convincing.
2. The paper [a] (ICLR 2024) is not discussed, however, it proposed an approach very similar to neural graphs of Kofinas et al. 
3. As mentioned in 1 above, [a] showed the application to diverse transformer architectures, which could be leveraged in this submission to enhance experiments.
4. Using timestamps is not well justified because the layers in neural networks, while sequential, do not have the notion of time. For example, there is no need to obtain node/edge embedding at continuous times. And usually temporal/dynamic GNNs are used for continuous time prediction. Perhaps, the idea of using timestamps could be more justified for networks such as neural ODEs.
5. More ablations could be done (potentially 3-5 ablations of different model components). For example, is it possible to provide results of INR2JLS with some baseline weight encoders like NFN/NG/etc? Can the authors ablate the GRU (Eq. 6)? 


References:

[a] Graph Metanetworks for Processing Diverse Neural Architectures", ICLR 2024

### Questions
Regarding Table 4, do the baselines use any augmentation? If not, is comparison in Table 1 fair?

Does the number of heads in multi-head message function needs to be predefined before training DNG-Encoder? Does it mean once it's trained, it cannot be applied to CNNs with larger kernels? Does any of the experiments in the paper have a task where the GNN has to generalize to larger kernels?
Any difference between multi-head message function and using "towers" from MPNN Gilmer et al. (2017)?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3