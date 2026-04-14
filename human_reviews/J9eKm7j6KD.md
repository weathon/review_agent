# Words in Motion: Extracting Interpretable Control Vectors for Motion Transformers

- Decision: Accept (Poster)
- Scores: 5, 3, 3, 8, 5

## Abstract
Transformer-based models generate hidden states that are difficult to interpret. In this work, we analyze hidden states and modify them at inference, with a focus on motion forecasting. We use linear probing to analyze whether interpretable features are embedded in hidden states. Our experiments reveal high probing accuracy, indicating latent space regularities with functionally important directions. Building on this, we use the directions between hidden states with opposing features to fit control vectors. At inference, we add our control vectors to hidden states and evaluate their impact on predictions. Remarkably, such modifications preserve the feasibility of predictions. We further refine our control vectors using sparse autoencoders (SAEs). This leads to more linear changes in predictions when scaling control vectors. Our approach enables mechanistic interpretation as well as zero-shot generalization to unseen dataset characteristics with negligible computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper aims to interpret hidden states and control them at inference. It assesses whether human-interpretable features are embedded in hidden states of motion transformers using neural collapse. The latent space properties are used to fit control vectors for each interpretable feature. The control vectors are optimized using sparse autoencoding and enforcing sparsity results in a more linear relationship between control vector temperatures and forecasts.

### Strengths
1. The paper introduces an approach for interpreting hidden states in motion forecasting models by leveraging neural collapse and linear probing.

2. The paper uses sparse autoencoders to optimize control vectors and enhance the linearity between control temperatures and forecasts.

3. Applying the method to self-driving cars and motion prediction shows its relevance to real-world applications.

### Weaknesses
1. The related work section does not clearly and effectively highlight the similarities and differences between prior research and your work.

2. The study lacks baseline comparisons. Are there existing methods that could be used to evaluate your proposed approach?

3. The paper lacks novelty, as it does not introduce linear probes or control vectors but simply applies these existing methods to motion control.

### Questions
1. Will using a sparse autoencoder significantly increase computational costs?

2. This paper is not well written. Some of the design motivations are not clearly explained.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces a method for extracting interpretable representations in transformer-based motion forecasting models: (i) leverage neural collapse to identify a structured latent space where features form interpretable clusters, (ii) use sparse autoencoders to optimize "control vectors" for each interpretable feature. This method is evaluated on three motion forecasting models, demonstrating effective feature manipulation and zero-shot generalization.

### Strengths
- The paper addresses the under-explored area of mechanistic interpretability in motion forecasting, providing a timely study at this intersection.
- The paper presents thorough quantitative and qualitative experiments, demonstrating the effectiveness of the interpretability method across different models and scenarios.

### Weaknesses
- While the paper is new in connecting mechanistic interpretability with motion forecasting, the technical components (neural collapse and sparse autoencoders) have been studied in LLMs. The claim of `a significant step towards mechanistic interpretability and controllability of transformer models` (L452) seems quite overstated. Further clarification on the unique technical contributions would strengthen the paper.
- The paper extends interpretability techniques from LLMs to the motion domain; however, the contexts differ fundamentally: LLMs deal with discrete tokens, while motion forecasting operates in a continuous space. The proposed method quantizes continuous variables like speed and acceleration into discrete categories, which might oversimplify the difference. A deeper discussion on the impact of this discretization and the influence of specific parameters (e.g., in L160) would enhance understanding of the challenges and significance of applying mechanistic interpretability to motion forecasting.

### Questions
- How sensitive are the results to the thresholds mentioned in Appendix A.3?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors have proposed finding an interpretable feature in the motion transformer and then enabling the control of the prediction based on the found interpretable features. To this end, authors claim that the interpretable feature can be found with the linear probing the hidden state, then use a control vector combined with sparse autoencoder to control the output of the motion transformer.

### Strengths
The problem is well-motivated: building an interpretable and controllable motion prediction network.

### Weaknesses
1. I think the overall writing needs to be improved in multiple aspects.
- First, it is hard to understand why the author has focused on motion transformers in the introduction. After reading section 3 (method part), I can understand why the author needs an interpretable method for motion transformers, but in the introduction, it is explained as an application area of their method, not the main focus.
- While the author suggests they have used neural collapse to measure the human interpretable feature, it is not known what neural collapse is in the introduction. 

2. Neural collapse is the most important term in the paper, but it is not defined clearly (or clearly mathematically), so it is unable to understand what the authors are using. Note that the authors have argued that they have used neural collapse as a metric, so we need a mathematical way to measure it. I think the author should define the following in a clear (or mathematical) way.
- What is "Neural collapse"?
-  What is an "interpretable feature"?
- What is "motion features"?

3. While the author mentioned they have used sparse autoencoder (SAE), it is not clear which one they have used [1,2,3]. 
- Which layer did the author train the SAE? SAE is usually trained on the hidden state of the transformer [1,2,3], and the layer is always selected carefully.
- Can the author report that the SAE is trained well? For instance, the reconstruction error of the SAE. (I think the author mentioned it is reported in the Appendix, but could not find it).

4. Can the author explain the goal of the motion transformer? What is the input of the motion transformer and output of the motion transformer?

5. It is not clear why the author used SAE for the collability. There are several activation steering methods [4,5]. I think it is great to claim the benefit of using SAE over other methods. 

Overall, I think the paper needs to improve in terms of writing, especially the introduction and the method part. Moreover, the author could improve the overall experimental details and analysis mentioned above.

Reference\
[1] Scaling and evaluating sparse autoencoders\
[2] Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders\
[3] Improving Dictionary Learning with Gated Sparse Autoencoders\
[4] In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering\
[5] In-Context Learning Creates Task Vectors

### Questions
See the weakness above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a method to interpret and control transformer-based motion forecasting models by analyzing hidden states through the lens of neural collapse. Using linear probes and control vectors, it maps hidden states to interpretable features, such as speed and direction, enabling insight into the model’s decision-making and the ability to manipulate forecasts without retraining. This approach enhances model interpretability, though its reliance on neural collapse and associated computational demands present some (minor) challenges, and testing other base models beyond SAE vs. PCA would be interesting.

### Strengths
- **Interpretability:** The method takes motion transformers, and maps hidden states to human-interpretable features, thus clarifying the model’s decision-making process. In general, interpretability is an important area.

- **Controllability:** Control vectors allow manipulation of specific motion features (e.g., speed, acceleration) at inference time without retraining, enabling intuitive model adjustments.

- **Zero-shot Generalization:** The interpretable control vectors support generalization to unseen scenarios, like different driving styles or environments, compensating for domain shifts in the data.

- **Nice integration across disciplines:** I like using neuro-inspired linear probing and their quantization method using natural language. I think this shows an elegant combination of techniques that can be leveraged to build interpretable systems.

### Weaknesses
- **Reliance on Neural Collapse:** The method's effectiveness depends on well-defined hidden state clusters. If neural collapse is weak, the extracted features and control vectors may be less reliable.

- **Limited Feature Scope:** The approach primarily focuses on basic motion features (e.g., speed, acceleration, direction) and could be expanded to capture more complex motion patterns and interactions with the environment.

- **Limited Baselines**: Currently, they primarily focus on comparisons for the control vectors are to PCA, but do not consider methods that inherently have dynamics, such as temporal convolutional networks (TCNs), nonlinear embedding methods that consider time-series inputs (one prominent example would be Schneider et al. 2023 Nature), or RNNs. While not critical for their argument, they should minimally discuss this in a limitation/discussion section. 

- **Computational Overhead:** Using sparse autoencoders increases computational demands, potentially extending training time and resource requirements; it would be ideal to estimate the computational resources used and those needed to utilize their code and models that they note will be released.

### Questions
See weaknesses, above. I would be happy for the authors to provide commentary on the baselines and computational resources, then I would be happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper focuses on the interpretability and control of transformer-based motion forecasting models. The authors aim to interpret hidden states of motion transformers and control them during inference. They use neural collapse to assess whether human-interpretable features are embedded within hidden states. The authors fit control vectors for steering during inference for the interpretable features. They further finetune these vectors using sparse autoencoders. They show that enforcing sparsity leads to a more linear relationship between the strength of the steering and interpretable features. They apply this approach to various motion forecasting models with different fusion mechanisms and environment representations. Finally, they address domain shifts using the control vectors and enable zero-shot generalization.

### Strengths
1. The paper applies interpretability approaches to transformer-based models beyond the natural language domain and propose neural collapse as a metric of interpretability.

2. The authors use sparse autoencoder-based steering for improving control vector linearity for motion control and zero-shot generalization.

3. The authors have done significant work to apply the proposed method to multiple motion forecasting architectures and datasets.

### Weaknesses
1. The paper relies heavily on neural collapse as a measure for the model learning clusters of interpretable features. The authors should verify the feature clusters are indeed distinct by comparing the within-class and between-class variance.
2. The L1 sparsity in the training objective is known to induce feature shrinkage and result in poor reconstructions (Wright and Sharkey, 2024). The authors should compare l1, l2, and reconstruction loss for other SAE architectures that do not induce feature shrinkage such as TopK or JumpReLU SAEs, and use the best-performing architecture to finetune control vectors and show their impact on control linearity. If training new SAEs seem out-of-scope, then they should report the % loss recovered from the SAEs and ensure that they are in an acceptable range.
3. The authors should test their approach on other naturally occurring domain shifts like various traffic densities, scenes with different weather conditions, etc.
4. Details on the training of the sparse autoencoders such as choice of learning rate and schedule, batch size, number of epochs, etc. as missing.
5. Other minor comments:
           a. The authors give a brief explanation of neural collapse in paragraph 2 of the introduction but do not explicitly mention the term.
           b. Along similar lines, the authors use terms like domain shift and fusion mechanisms for the transformer architectures and zero- 
               shot generation, they should be properly defined and contextualized for their specific problem.
           c. The reference for sparse autoencoders used in Section 4.4 is for the Gated SAEs paper, however, the authors do not use that 
               specific architecture. The reference should be changed appropriately.

### Questions
1. In the introduction, the authors claim that they identify that interpretable features are embedded in hidden states of transformer-based models. Since this is a well-known observation, the reviewer is curious about how the work adds to the existing knowledge in the mechanistic interpretability literature.
2. In the conclusion, the authors claim that they take a significant step towards the mechanistic interpretability and controllability of the transformer models. This is a bold claim - how do the results from the paper justify the claim and add to the pre-existing knowledge in the broader transformer interpretability literature?
3. How do different feature quantization thresholds affect results? The rationale for these specific thresholds should be better justified by showing the dataset statistics. The authors mention the choice was based on insights from Seff et al., 2023 - what were the specific insights the authors used for their choice of classes?
4. In Section 3.2, the authors claim to use the mean of the standard deviation to measure collapse - is this correct or should this metric be just the standard deviation?`
5. For the control vectors found using pca, what is the variance explained in each case?
6. The implications of neural collapse on the post-training processes of a model like transferability and generalization is still an open question (Vignesh Kothapalli, 2024). Can the authors better justify the use of neural collapse to show the validity of their probe accuracies?
7. How would the control effects change when combining multiple control vectors (e.g., speed + direction)?
8. How do the control vectors found using one approach compare to the other, in terms of their cosine similarity?
9. What was the intuition behind choosing the temperature range of [-50,50]?
10. Can the authors comment on how to extend their method to continuous control features rather than discrete classes as they have used in the paper?
12. Will the authors release both code and pre-trained models, including the trained sparse autoencoders?

### Soundness
2

### Presentation
3

### Contribution
2
