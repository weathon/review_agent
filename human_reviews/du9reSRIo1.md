# RouteFinder: Towards Foundation Models for Vehicle Routing Problems

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
This paper introduces RouteFinder, a comprehensive foundation model framework to tackle different Vehicle Routing Problem (VRP) variants. Our core idea is that a foundation model for VRPs should be able to represent variants by treating each as a subset of a generalized problem equipped with different attributes. We propose a unified VRP environment capable of efficiently handling any attribute combination. The RouteFinder model leverages a modern transformer-based encoder and global attribute embeddings to improve task representation. Additionally, we introduce two reinforcement learning techniques to enhance multi-task performance: mixed batch training, which enables training on different variants at once, and multi-variant reward normalization to balance different reward scales. Finally, we propose efficient adapter layers that enable fine-tuning for new variants with unseen attributes. Extensive experiments on 48 VRP variants show RouteFinder achieves competitive results. Our code is openly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces RouteFinder, a foundation model for multiple VRP variants, using a transformer-based encoder with Global Attribute Embeddings. Key innovations include Mixed Batch Training (MBT), Multi-Variant Reward Normalization, and Efficient Adapter Layers (EAL) for adaptability to new problem variants. While it shows competitive results across 24 VRP variants, some limitations remain in performance on certain benchmarks.

### Strengths
1.RouteFinder is a significant step toward a general model for VRP, reducing the need for separate models for each variant and improving scalability.

2.The use of Global Attribute Embeddings and a transformer-based encoder helps the model differentiate between VRP variants and adapt effectively.

3.Mixed Batch Training (MBT) improves stability by training on multiple variants at once.

### Weaknesses
1.Despite generalization, RouteFinder lags behind specialized models on certain VRP variants, such as LKH, HGS, or SISR. A deeper analysis of these trade-offs is needed.

2.The limitations of MBT in highly imbalanced datasets are not well explored. More ablation studies would be helpful to understand its optimal conditions.

3.There is a lack of discussion on how the different model components interact. Further exploration of these interactions would be valuable.

### Questions
1.How does the difference between AL and EAL affect the outcome, given that EAL, by maintaining a predefined zero matrix, doesn't seem to have an impact on AL, which later fills in the matrix during training? It would be helpful to include empirical results or visualizations that demonstrate the effects of these two approaches at different stages of training.

2.Could you discuss the potential trade-offs between the two methods and scenarios where one might be more advantageous than the other?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces ROUTEFINDER, a foundation model designed to tackle various Vehicle Routing Problem (VRP) variants through a unified framework. It presents several key components, including a Transformer-based model architecture, Global Attribute Embeddings, Mixed Batch Training, and Efficient Adapter Layers for fine-tuning unseen attributes. While the paper proposes an interesting approach and demonstrates competitive experimental results across multiple VRP variants, there are concerns regarding the depth of innovation in the proposed methods and the clarity of explanations in some sections.

### Strengths
1. The authors propose a unified VRP environment that can handle multiple VRP variants efficiently, demonstrating a strong attempt at generalization in a complex combinatorial optimization domain.
2. The introduction of Efficient Adapter Layers (EAL) presents a lightweight yet powerful method to fine-tune the model for unseen attributes, which is crucial for practical applications that require adaptability.

### Weaknesses
1. Figure 4.1 lacks sufficient explanation. There is a need for a more detailed description of Figure 4.1. 
2. The proposed unified VRP environment and attribute dynamic composition approach, while having some differences compared to MTPOMO and MVMoE in handling different attributes, does not present substantial changes. In fact, MTPOMO and MVMoE can also achieve similar functionality.
3. The improvements to the Transformer architecture are mainly module-level substitutions, such as the use of RMS Normalization, SwiGLU activation, and FlashAttention. While these changes enhance training stability and convergence, they largely involve the application of existing techniques rather than introducing significant innovation. 
4. Lack of evaluation on large-scale.

### Questions
1. The design of Global Attribute Embeddings, such as Open and Limit attributes, is a notable feature. However, in MTPOMO and MVMoE, these attributes are already considered in decoder. What is the significance of redundantly including these embeddings in the encoder as well in this work?
2. The meaning of $l_l$ is unclear. Does $l_{rand(0,1)}<1/2$ represent random sampling of half of the attributes?
3. The explanation of why encountering different problem variants in a single batch leads to more stable training (as shown in Figure 4.3) is not well substantiated. Additional evidence is needed to support this claim.
4. The computation of the average reward $\bar{r}^{(k)}_t$ needs to be explained in more detail.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A foundation model for vehicle routing problems was presented in the manuscript. Mixed training, global feature embedding in a transformer were proposed to increase the training advantage. The foundation model is better than MTPOMO, MVMoE and AL in uniform data, while the improvement in CVRPLIB is quite marginal.

### Strengths
The foundation model is better than MTPOMO, MVMoE and AL in uniform data. The normalization of rewards and layers makes a lot of sense to standardize the training of different vehicle routing problems that take on different features. Extending zero-shot and few-shot capabilities of a foundation model advances the frontier of large optimization models.

### Weaknesses
The foundation model is not validated comprehensively. The zero-shot validation for a mere M constraint is not convincing as it is very relevant to constraints in training batch. More constraints in zero-shot validation are supposed to make a more solid work. The zero-shot didn't account for MTPOMO, MVMoE that were only compared in Table 1. Besides uniform data, the foudanation model is validated in CVRPLIB ignoring more practical vehicle routing problems like soloman VRPTW data. The improvement is incremental and still fall behind HGS-VRP.

### Questions
1. Could authors explain their rationale for selecting the mixed backhaul (M) constraint for zero-shot validation, rather than other practical constraints like multiple depots? Please discuss any other choices of validation task related to real-world VRP applications.

2. What is training time of the foundation model, MTPOMO, MVMoE? MTPOMO and MVMoE are retrained by the same data and same training resource, including dataset and training duration? 

3. How does the difference of AL and EAL influence outcome, as EAL keeping a predefined zero matrix seems not to influence AL that puts in the matrix in later phase? It's better to include empirical results or visualizations showing the effects of these different approaches at various stages of training.

4. Please discuss any potential trade-offs between the two methods and situations where one might be preferable over the other.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a unified approach called RouteFinder to model different Vehicle Routing Problem (VRP) scenarios using a transformer-based encoder and global attribute embeddings. This approach provides a scalable solution for diverse routing challenges. Mixed Batch Training (MBT) and Multi-Variant Reward Normalization are introduced to enable efficient training across multiple problem variants. Specifically, Multi-Variant Reward Normalization balances reward scales to ensure stable learning. Efficient Adapter Layers (EAL) enhance the model's adaptability, allowing it to handle new VRP variants with unseen attributes without full retraining. This paper includes extensive experiments that demonstrate the competitive performance of RouteFinder compared to state-of-the-art baselines.

### Strengths
1. The introduction of RouteFinder as a foundation model for various VRP variants is a significant step forward, offering flexibility and reducing the need for specialized models for each variant. This can lead to more accessible and scalable VRP solutions across diverse scenarios.

2. The model employs a transformer-based encoder with Global Attribute Embeddings, improving the understanding and differentiation of VRP tasks, which translates to better adaptability and solution quality.

3.  Mixed Batch Training (MBT) allows RouteFinder to train on multiple VRP variants simultaneously, supporting diverse distributions and ensuring model stability. The Multi-Variant Reward Normalization further strengthens robustness, addressing reward scale differences effectively across VRP types.

4.  EAL enables fine-tuning of the pre-trained RouteFinder model to handle new VRP variants, providing a lightweight and promising solution for unseen attributes. This feature is especially advantageous in real-world applications, where new constraints may arise.

5. Extensive experimentation on 24 VRP variants showcases the competitive performance of RouteFinder, indicating its effectiveness and superior performance compared to recent baselines in multi-task VRP solutions.

### Weaknesses
1.  Although RouteFinder demonstrates strong generalization across VRP variants, some performance trade-offs arise when compared to models designed for specific problem variants. Additional analysis could help address these trade-offs, especially in large-scale deployments.
2.  While MBT is an efficient training method, the paper could provide a more detailed examination of its limitations, particularly in cases of significantly imbalanced attribute distributions. Including more ablation studies might clarify MBT’s optimal conditions.

3.The ablation studies primarily focus on individual components. Exploring potential interactions among these components might offer insights into possible synergistic or conflicting effects, which could enhance the model's overall efficiency.

4. The use of transformers and attribute composition may make RouteFinder somewhat of a "black box." Adding interpretability methods, like attention visualization, could improve the practical usability and trustworthiness of the model by shedding light on its decision-making processes.

### Questions
1. It seems there is not yet a neural method that could be trained on a single VRP variant or task and outperform the strong traditional method like LKH, HGS or SISR on that task, then what is the point in training a unified model across multiple tasks with much performance drop?
2. This paper is quite similar to 'RouteFinder: Towards Foundation Models for Vehicle Routing Problems, ICML Workshop'. And what is the major difference?
3. In the main body, only O, L and M are considered as the global attribute. However, in appendix, late time window and depot location are also considered as global attribute. Why there is such an inconsistence?
4. The authors claim that RMS Norm could speed up convergence, SwiGLU could help model the data relationship. However, those claims are not fully supported in the paper.
5. In Table 5.1, the objective value of VRPBLTW is 350.808. Please elaborate what happened to the model that leads to such an inferior result.

### Soundness
2

### Presentation
2

### Contribution
2
