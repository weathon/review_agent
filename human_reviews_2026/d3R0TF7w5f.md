# Mitigating Privacy Risk via Forget Set-Free Unlearning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Training machine learning models requires the storage of large datasets, which often contain sensitive or private data.
Storing data is associated with a number of potential risks which increase over time, such as database breaches and malicious adversaries.
Machine unlearning is the study of methods to efficiently remove the influence of training data subsets from previously-trained models.
Existing unlearning methods typically require direct access to the "forget set"---the data to be forgotten-and organisations must retain this data for unlearning rather than deleting it immediately upon request, increasing risks associated with the forget set.
We introduce partially-blind unlearning---utilizing auxiliary information to unlearn without explicit access to the forget set.
We also propose a practical framework Reload, a partially-blind method based on gradient optimization and structured weight sparsification to operationalize partially-blind unlearning.
We show that Reload efficiently unlearns, approximating models retrained from scratch, and outperforms several forget set-dependent approaches. On language models, Reload unlearns entities using <0.025\% of the retain set and <7\% of model weights in <8 minutes on Llama2-7B.
In the corrective case, Reload achieves unlearning even when only 10\% of corrupted data is identified.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Existing study requires the model provider to keep the requested unlearned data information until the unlearning is completed. This paper explores and defines a new unlearning setting, Partially Blind Unlearning (PBU), in which no direct access to the unlearning data is required. Under this setting, the author implements a three-fold method that leverages cached gradients from the training stage. The method RELOAD consists of three steps, combining previous studies. RELOAD gives the unlearned model under PBU by following 1. Compute gradient difference, perform a single gradient ascent step, and then fine-tune on retained datasets. The author then provides comprehensive experiments on both classical unlearning tasks unlearning on vision models, and unlearning tasks on language models.

### Strengths
1. The paper provides a new privacy-oriented perspective on the problem of unlearning. The author made an important observation and defined a new setting for exploring safe unlearning. The motivation is well-explained. 
2. The paper integrates several existing methods in solving new and practically important problems. This provides a more modular and interpretable unlearning algorithm. 
3. The evaluation is comprehensive and representative. The experiments marked the performance of RELOAD on both small-scale vision models and on language models.

### Weaknesses
1. In Section 2.3, the author states that we can infer the loss gradient on the forget dataset with the original model by computing the difference between the loss gradient on the original dataset and the retained dataset. This, however, works with several assumptions that are not explicitly stated, such as the assumption that loss is additively computed across samples. I am concerned that for contrastive learning tasks or models trained with layer normalization, this formula may not function as intended. I would appreciate further discussion on this point.
2. While the overall unlearning method provides more interpretability compared to existing unlearning methods and achieves promising results, the methodological innovation is somewhat incremental. The core part of the method, including gradient ascent, selective weight re-initialization, and fine-tuning are existing method. The contribution of the method lies mostly in how to combine the methods under PBU setting.

### Questions
It seems like RELOAD is achieving good results in the baseline section. The setting for the experiments is not strict to PBU. This indicates that  RELOAD is producing better results and outperforms baseline methods that do use the forget set. If so, why is PBU important to this method?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles the problem of machine unlearning without retaining the forget set, a long-standing practical limitation in enforcing the “right to be forgotten.” The authors introduce RELOAD, a partially-blind unlearning algorithm that relies only on cached gradients from the final training step and the retained dataset, avoiding the need to store sensitive data. Through gradient ascent, selective reinitialization, and fine-tuning, RELOAD achieves performance comparable to retraining from scratch while being significantly more efficient. Experiments on image classification and large language models demonstrate promising results.

### Strengths
1. The paper formalises the partially-blind unlearning setting, which is a realistic and privacy-preserving variant of traditional unlearning.
This addresses an important gap between regulatory demands (e.g., GDPR) and existing technical capabilities.
2. The motivation connecting dataset risk and model risk is conceptually strong and provides a clear societal justification for this work.
3. RELOAD elegantly combines gradient-based unlearning, structured sparsity, and fine-tuning in a simple yet effective pipeline.
The “knowledge value” mechanism for selective reinitialization is particularly interesting.
4. Across diverse tasks (CIFAR, SVHN, and Llama-2), RELOAD shows comparable or better performance than methods requiring the actual forget set. Its efficiency (<8 min for Llama2-7B) suggests real practical potential.

### Weaknesses
1. Limited theoretical justification. The paper mainly relies on intuition and empirical validation. While the derivation of ∇L(Dforget) = ∇L(D) − ∇L(Dretain) is sound, the guarantees of approximate unlearning (e.g., bounds on residual influence) are not formally analysed.
2. Dependence on cached gradients. Storing full-model gradients at the end of training may be expensive for large-scale models, potentially offsetting some of the claimed efficiency or privacy advantages.
3. While results are impressive, the experiments could be broadened — e.g., include more realistic privacy benchmarks or human-sensitive data domains.
4. Some ablation studies (e.g., varying α in reinitialization, or using partial gradient caching) are deferred to the appendix but would strengthen the main text.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RELOAD, a partially-blind unlearning (PBU) method that aims to remove the influence of a forget set without access to the forget data. The method includes three parts: 1) an ascent step using cached final-training gradients, 2) re-initialisation of parameters with low “knowledge value,” and 3) fine-tuning on the retain set to maintain model utility. Experiments include classic unlearning, entity unlearning, and corrective unlearning.

### Strengths
1. The problem of forget-set-free unlearning is novel and well-motivated. The RELOAD enables machine unlearning without retaining the raw forget set by using cached final-step gradients.

2. The empirical results are strong across benchmarks. RELOAD can preserve model utility while improving forget quality.
 
3. The method is efficient. For Llama-2-7B, it uses 7% of weights and <0.025% of retained data, finishing in 8 minutes on a single GPU. This result suggests that the method is practical.

### Weaknesses
1.	The paper does not provide a theoretical justification for why a single gradient-ascent update using $\nabla_\theta L(D)-\nabla_\theta L(D_{\text{retain}})$ and selective re-initialization of low-KV weights can remove the influence of the forget data. As a result, it is unclear when RELOAD succeeds or fails beyond the reported scenarios.

2.	The paper claims RELOAD “allow user data to be immediately removed when a request for deletion is made, eliminating the continued accumulation of dataset risk,” but RELOAD requires the retain set to compute gradients and to perform fine-tuning. This means that the retained data is still at risk, and the ideal method should not use any training data during the unlearning process.

3.	The robustness of RELOAD across different numbers of requests is unclear. When the forget set is extremely small (e.g., unlearning only one data point), $\nabla_\theta L(D_{\text{forget}})=\nabla_\theta L(D)-\nabla_\theta L(D_{\text{retain}})$ becomes a very small residual between two nearly identical large gradients; this makes the ascent step near zero and may fail to truly forget that sample. In addition, when the forget request is large (e.g., 30% of the data), the ascent and re-initialization would degrade model utility, while fine-tuning may fail to recover the utility given the limited retain set.

### Questions
Please see the Weaknesses section for all questions and clarification requests.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a significant limitation in existing machine unlearning techniques, namely the requirement to retain access to the “forget set” (the data to be removed) during the unlearning process. The authors introduce the concept of partially-blind unlearning (PBU) and propose RELOAD, a framework that performs unlearning without direct access to the forget set. RELOAD leverages cached gradients from the final epoch of training and combines gradient ascent, selective weight reinitialization, and fine-tuning on retained data. The paper claims strong empirical results showing that RELOAD can efficiently approximate retraining from scratch and even outperform some forget set-dependent approaches, including applications to both image classification and large language models.

### Strengths
* The problem formulation of unlearning without the forget set is timely, novel, and practically relevant for privacy compliance (e.g., GDPR).

* Methodologically sound integration of gradient ascent, weight reinitialization, and fine-tuning into a coherent framework.

* Empirical results show promising efficiency and competitive performance, especially for large models such as Llama2-7B.

### Weaknesses
* Limited robustness to model updates

The method assumes availability of final-epoch gradients representing the entire training data. In real-world pipelines where models are fine-tuned or continuously updated, these cached gradients may no longer capture the forget set’s influence, reducing unlearning effectiveness. Evaluating RELOAD under fine-tuning or continual learning scenarios is necessary.

* Unclear privacy guarantees

Although RELOAD is “partially blind,” cached gradients can still leak sensitive information. Prior works (e.g., Geiping et al., 2020) show that gradient inversion can reconstruct data. The paper lacks a quantitative privacy leakage analysis to support its safety claims.

* Limited ablation and sensitivity study

While some ablations are included, deeper exploration of how hyperparameters (e.g., ascent rate, reset proportion) affect performance is missing. This limits confidence in robustness across settings.

* Storage overhead

The approach removes the need to store the forget set but introduces the requirement to store full-model gradients, which can be large for modern networks. The practical feasibility of gradient caching is not analyzed.

### Questions
1. How would RELOAD perform if the model undergoes fine-tuning or incremental updates after initial training?

2. What is the approximate storage cost of ∇θL(D) for large models such as Llama2-7B, and could this be mitigated via gradient compression?

3. Have the authors tested for information leakage from cached gradients using existing gradient inversion techniques?

4. How does RELOAD handle overlapping or correlated data between forget and retain sets?

5. Can the method scale to federated or distributed training settings where only local gradients are available?

### Soundness
2

### Presentation
3

### Contribution
2
