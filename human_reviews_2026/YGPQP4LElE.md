# Provenance Networks: End-to-End Exemplar-Based Explainability

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
We introduce provenance networks, a novel class of neural models designed to provide end-to-end, training-data-driven explainability. Unlike conventional post-hoc methods, provenance networks learn to link each prediction directly to its supporting training examples as part of the model’s normal operation, embedding interpretability into the architecture itself. Conceptually, the model operates similarly to a learned KNN, where each output is justified by concrete exemplars weighted by relevance in the feature space. This approach enables systematic studies of memorization versus generalization, facilitates the identification of mislabeled or anomalous data points, and enhances robustness to input perturbations. By jointly optimizing the primary task and the explainability objective, provenance networks offer insights into model behavior that traditional deep networks cannot provide. While the model introduces additional computational cost and currently scales to moderately sized datasets, it provides a complementary approach to existing explainability techniques. In particular, it addresses critical challenges in modern deep learning, including model opaqueness, hallucination, and the assignment of credit to data contributors, thereby improving transparency and trustworthiness in neural models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Provenance Networks, neural architectures that aim to deliver training-data-driven explainability by learning, during standard end-to-end training, to retrieve concrete training exemplars that purportedly support each prediction. The core mechanism is an index-prediction branch that maps an input to the most relevant training example(s), combined with a primary task branch (e.g., classification). The authors study single-branch and two-branch variants, introduce a label-mixing parameter α to balance memorization and generalization, and evaluate the approach on MNIST, FashionMNIST, CIFAR-10/100, Stanford Dogs, and a VAE-based image generation setting.

### Strengths
1. The label-mixing α yields a transparent memorization–generalization spectrum with concrete outcomes
2. Visualizations of learned embeddings and cluster structures illustrate instance-level organization and intra-class variation 
3. For certain distortions, partial label mixing can outperform an isolated CNN, an interesting empirical phenomenon worth deeper study

### Weaknesses
1. The work equates retrieving visually similar training samples with explanation, but does not provide faithfulness metrics, human evaluations, or causal tests (e.g., counterfactual influence, provenance under data poisoning) to substantiate that linkage as a reliable explanation 
2. No empirical comparisons against influence functions, data Shapley, or modern training data attribution methods appear; this hinders assessing whether the proposed approach yields superior or complementary provenance quality
3. Claims about mitigating hallucinations, IP credit attribution, and applicability to LLMs are not evaluated; the current evidence is confined to small vision datasets and a modest-generation VAE experiment
4. Robustness comparisons do not include strong baselines (adversarial training, augmentations), and dataset debugging/anomaly detection via entropy lacks ground-truth validation or quantitative metrics

### Questions
1. How will you formally evaluate whether retrieved exemplars truly explain predictions?
2. Beyond stating applicability, can you demonstrate a retrieval-augmented LLM setup where provenance networks concretely reduce hallucinations and provide traceable credit attribution, with clear metrics

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces provenance networks, a novel neural network architecture designed for end-to-end, exemplar-based explainability.The core strength lies in its ability to explicitly link predictions to training data indices, akin to a differentiable KNN, providing unique insights into model behavior and enabling diverse applications like dataset debugging and membership inference. The mechanism for controlling the memorization-generalization trade-off is well-demonstrated. However, the paper acknowledges significant scalability limitations due to the index head size, only partially mitigated by subset sampling. Furthermore, the clarity of some mathematical formulations, particularly around the mixing strategy and loss functions, could be improved.

### Strengths
1. This paper introduces a novel end-to-end provenance mechanism.
- The core idea of integrating an "index branch" to predict training sample indices alongside the main task is a novel departure from post-hoc explainability methods, baking interpretability into the model's forward pass.
- Framing the network as a learned, differentiable KNN provides an intuitive conceptual model and connects deep learning with case-based reasoning paradigms.
- Unlike feature attribution methods (LIME, SHAP, Gradients), provenance networks provide explanations in terms of concrete training examples, which can be more intuitive and actionable for users.

2. This paper demonstrated versatility across multiple applications.
-  Experiments show that intermediate levels of memorization (tuned via $\alpha$) can improve robustness to certain input distortions compared to a standard classifier, suggesting a benefit beyond just explainability.
- Integrating the provenance mechanism into a VAE demonstrates the potential for tracking the influence of training samples on generated outputs, relevant for issues like hallucination and content attribution.
- The near-perfect AUC achieved using the index branch confidence for membership inference highlights the model's strong memorization capabilities and provides a clear signal for distinguishing training members from non-members.

3. This paper provides a systematic analysis of design choices.
- The paper explores both single-branch and two-branch (class-conditional and independent) architectures, analyzing their respective strengths and weaknesses, particularly regarding scalability and performance on different datasets.
- The analysis of how model capacity and the degree of parameter sharing between branches impact performance provides valuable insights into architectural design trade-offs for these multi-task networks.
- The investigation into training on subsets (10%-90%) demonstrates a practical approach to mitigate the scalability challenge, showing that high classification and reasonable retrieval performance can be maintained with significantly fewer index head parameters.

### Weaknesses
1. This paper provides a limited theoretical analysis of provenance learning.
- This paper has no derivation or proof linking index prediction with model generalization capacity. 
- This paper has no formal metric to quantify “explainability improvement.”  
- The paper lacks a complexity analysis for the joint optimization of $\lambda_{class}$ and $\lambda_{index}$.
- In Section 2.1, the method introduces memorization by using biased sampling. It also introduces random variance, and if the selected samples are not representative, the actual effect may be questionable.

2. This paper does not provide the computational & memory complexity analysis.
- The fundamental limitation is the output layer size of the index branch, which scales linearly with the number of training samples (N) or the maximum samples per class (M), making it potentially infeasible for very large datasets.
- While subset sampling helps (reducing parameters by up to 70-50% while maintaining accuracy), it means the model cannot retrieve provenance for excluded samples. The non-monotonic behavior of the Top-1 index accuracy with subset size also suggests challenges in selecting the optimal subset. Selecting a subset from the training dataset remains a challenge.
- The paper acknowledges computational cost but doesn't quantify it. Training and inference overhead compared to standard models, especially for the large index head computations (K in Section 2.2), is not reported.

3. This paper has a limited empirical comparison and scope.
- All experiments are on 2-D images; no text, tabular, or temporal data, although this paper claims the method is applicable for other modalities, including LLMs. 
- This paper claims this method helps mitigate hallucinations and enables fair credit attribution to content creators. However, there is no evidence. The primary task is classification; no segmentation, detection, or regression.  
-  The related work discusses influence functions and data Shapley but provides no empirical runtime or accuracy comparison against these methods, even on smaller datasets where they might be feasible. The comparison remains conceptual.

4. This paper has some writing issues.
- There are some inconsistent terms and math symbols. For example, in line 103 and line 107, there are two kinds of 'appendix'. In line 136, $\hat{z}_z$, in line 151, $\hat{y}_y$, and $\hat{z}_{z|y}$ seem not good. In Figure 2, the capitals are not consistent.
- The table caption should be above the table.
- In table 2, the class-conditional and class-independent are unclear.

### Questions
-  Why was the specific mixing strategy (sampling own index vs. random same-class index) chosen over alternatives, such as sampling based on feature similarity within the class?
-  Could the index branch predict an embedding similarity score directly instead of a categorical index, potentially mitigating the large output layer issue? How might this affect performance?
-  For subset sampling (Sec 3.3), were alternatives to stratified random sampling (e.g., core-set selection, uncertainty sampling) considered for choosing the representative subset?
- Could you provide quantitative measures of computational overhead (e.g., increase in training time, inference latency, memory usage) compared to baseline models without the index branch?
-  In the VAE application (Sec 4.5), how strongly correlated is the quality/realism of the generated image with the index prediction accuracy or confidence?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel class of neural models termed provenance networks, which is designed to provide end-to-end, training-data-driven explainability. The model operates analogously to a learned KNN. It enables joint end-to-end learning and facilitates the analysis of memorization–generalization trade-offs. It also supports the detection of mislabeled or anomalous samples and the verification of training-set membership.

### Strengths
1. This paper proposes a novel neural models in which each prediction can be directly linked to its supporting training samples, thereby injecting interpretability into the model architecture. This design is conceptually inspiring in the field of explainable AI.
2. This paper provides a concise and clear explanation of the provenance network. Specifically, it is illustrated as a shared backbone architecture with two branches: one dedicated to the main task and the other responsible for predicting the index of the input sample. 
3. The paper clearly distinguishes between single-branch and dual-branch networks, and clarifies that label mixing is only applied during the training of the single-branch network.

### Weaknesses
1. The datasets used in the experiments (MNIST, FashionMNIST, and LFW) are relatively small and simple. The paper lacks experiments on large-scale benchmark datasets such as ImageNet or DomainNet.
2. The authors mention that input samples are not always mapped to their own indices but occasionally to other indices within the same class. This phenomenon raises a question: what impact does such occasional misalignment have? Should specific techniques be adopted to mitigate its effect?
3. In Section 3.1, this paper state that with a lower label mixing coefficient $\alpha$, the network tends to memorize individual samples. Why does this happen? The authors should provide a more detailed explanation of the underlying reasons.
4. In Section 3.3, the authors claim that to address the fundamental scalability limitations of provenance networks, they investigate whether the system remains effective when trained only on a strategically selected subset of training data. However, it is unclear how this setup helps resolve the scalability issue. The authors should elaborate on the reasoning behind this connection.
5. To study how model size affects the relationship between generalization and memorization in provenance networks, the authors compare two models with 4M and 80M parameters. However, this comparison overlooks practical considerations: in real-world scenarios, a 4M-parameter model is likely a CNN architecture, whereas an 80M-parameter model is likely a Transformer, making the comparison less meaningful.
6. Overall, as a work on explainable AI, the theoretical contribution of this paper remains limited. On one hand, several causal claims throughout the paper lack sufficient justification. On the other hand, the architectural design, network taxonomy, and experimental setup are not well-supported by theoretical derivations.A more rigorous approach would be to first formalize the target problem using mathematical formulations, derive it step by step, and ultimately demonstrate why your provenance network is necessary to explain or address it. Alternatively, the authors could strengthen the logical and theoretical foundations of the paper in other ways.

### Questions
Why does the network tend to memorize individual samples when the label mixing coefficient $\alpha$ is set to a lower value?

If input samples are not always mapped to their own indices but occasionally to other indices within the same class, what would happen in such cases? Would this phenomenon undermine the rigor of the paper?

From my perspective, this paper lacks sufficient causal explanations and justifications for some of its claims or propositions. Should this interpretation be inaccurate, could the authors provide further clarification and supporting evidence?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a novel neural network called Provenance Networks, designed to predict the most relevant training sample index while performing classification or generation tasks, thus establishing a direct link between the input and training data. The authors propose both single-branch and two-branch architectures and experimentally validate them on multiple tasks, including image classification and image generation, primarily based on MNIST, FashionMNIST, and CIFAR-10/100. However, some issues need to be emphasized.

### Strengths
1. This work emphasizes achieving end-to-end sample-level interpretability by directly indexing training samples.
2. It combines prediction tasks with index prediction, performs joint training through multi-task loss, and analyzes the balance between memorization and generalization.
3. The authors apply this framework to tasks such as image classification and generation, demonstrating its potential applicability across multiple scenarios.

### Weaknesses
1. Theoretically, the Provenance mechanism proposed in this work predicts sample indices using a supervised paradigm, essentially similar to the retrieval module in memory networks or matching networks. What is the fundamental difference between this work and memory-augmented models, or prototype networks?
2. Regarding dataset selection, all experiments are based on small datasets such as MNIST and FashionMNIST, with shallow CNN/VAE classification models. There is a lack of testing on mainstream architectures (such as ResNet, ViT, and transformers) and large-scale datasets (such as ImageNet, COCO, and OpenImages). This is considered "toy" testing, reducing the credibility of the experimental conclusions.
3. In terms of comparison methods, there is no quantitative comparison with methods based on feature attribution (such as GradCAM), memory networks, kNN prototypes, or RAG classes. This fails to verify whether the model outperforms existing interpretable models.
4. The index prediction branch's output dimension explodes as the training set size increases, but the paper does not provide any measured data on training time, inference time, memory usage, etc. The claim of scalability is therefore questioned.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2
