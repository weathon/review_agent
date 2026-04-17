# Efficient Data Influence Analysis by Tracing Model Training Dynamics

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2, 4

## Abstract
Quantifying the impact of training data is essential for understanding model behavior and optimizing the training process. Despite extensive research into influence estimation, existing methods often rely on repeated training or gradient analysis, which results in prohibitive computational and memory costs and limits their applicability to large-scale models and datasets. In this paper, we explore a new perspective on influence estimation by distilling influence signals from training dynamics, i.e., the model’s predictions on individual examples throughout training. We propose an influence estimation approach that uses contrastive learning to project the observed influence into a representation space, where the proximity between data points reflects their influence strength. Our approach is simple, efficient, and scalable, requiring neither gradient computation nor assumptions about the optimizers.  We validate our approach across various tasks and datasets, demonstrating its ability to estimate influence effectively, scalability to large models, and utility in downstream applications, such as mislabeled data identification, influential data selection, and data attribution.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes INFTRACE, a gradient-free algorithm for estimating data influence by distilling training dynamics into a learned embedding space. Instead of relying on gradients or Hessians, it tracks how each batch affects the prediction of validation samples during training and then learns a contrastive embedding where positively influenced validation samples lie closer to the training batch. Influence between any training and validation pair is computed as the negative distance between their learned embeddings. The authors demonstrate the effectiveness of their approach on diverse models (RoBERTa and LLaMA-3-8B) and tasks (mislabeled data detection, data selection).

### Strengths
- INFTRACE offers a gradient-free alternative to influence estimation that avoids costly Hessian or gradient computations, making it applicable to large models such as LLaMA-3-8B
- Comprehensive experiments across a variety of models and tasks

### Weaknesses
- Lack of principled formulation and interpretability:

  - INFTRACE fails to account for the sign or direction of influence both in its algorithmic design and final scores. 1) In the contrastive learning framework, validation samples with large negative $\Delta p$ values are always treated as negatives samples and therefore forced to be far away from the batch embedding. This does not look like a principled design. 2) The INFTRACE scores are negative for all training samples and therefore offers no interpretation as to which training samples are helpful or harmful. This is a significant drawback and contrasts with most existing data attribution methods.

  - INFTRACE uses a weighted sum of training embeddings instead of modeling how each individual training sample affects a validation example (as in TracIn). This aggregated representation blurs the source of influence within the batch. Moreover, the inverse-confidence weighting scheme is heuristic, lacking both theoretical justification and ablation analysis.

  - From an information-theoretical perspective, relying solely on prediction deltas ($\Delta p$) is unlikely to yield more informative or faithful influence estimates than gradient-based approaches such as TracIn, which leverage richer, first-order information.

- Gross ignorance of prior work: The paper overlooks a substantial body of recent literature on dynamic influence estimation. TracIn itself is a dynamic method that tracks influence along the training trajectory, yet this is not properly acknowledged. In addition, SGD-influence [1] and Simfluence [2] are two prominent approaches that the authors fail to cite or compare against. The paper also ignores recent efforts to scale these dynamic methods to LLMs (e.g., [3, 4]). I encourage the authors to check the recent survey on data attribution [5] for a more comprehensive view of this rapidly evolving area. 

- The empirical results (in particular Figure 5 and Table 2) do not show statistically significant improvement over baselines. The authors may consider incorporating additional evaluation metrics, such as the Linear Datamodeling Score (LDS).


References

[1] Hara, Satoshi, Atsushi Nitanda, and Takanori Maehara. "Data cleansing for models trained with SGD." Advances in Neural Information Processing Systems 32 (2019).

[2] Guu, Kelvin, et al. "Simfluence: Modeling the influence of individual training examples by simulating training runs." arXiv preprint arXiv:2303.08114 (2023).

[3] Wang, Jiachen T., et al. "Data shapley in one training run." ICLR 2025   

[4] Wang, Jiachen T., et al. "Capturing the Temporal Dependence of Training Data Influence." ICLR 2025   

[5] Deng, Junwei, et al. "A Survey of Data Attribution: Methods, Applications, and Evaluation in the Era of Generative AI." (2025).

### Questions
I don’t have further questions for now. My primary concerns lie in the design principles of INFTRACE and the lack of comparison with recent, strong baselines (e.g., [3] and [4]).

### Soundness
2

### Presentation
2

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
This paper introduces INFTRACE, an efficient method for data influence analysis that avoids the high costs of traditional gradient-based or retraining-based approaches. The core idea is to distill influence signals from training dynamics. INFTRACE then uses contrastive learning to map data points into a low-dimensional representation space, where the proximity between two points directly reflects the influence strength between them. The authors demonstrate that this gradient-free and optimizer-independent method is scalable and effective on models like RoBERTa and LLaMA-3-8B for tasks including mislabeled data detection and influential data selection

### Strengths
The idea of using contrastive learning to learn a dedicated "influence representation space" is both interesting and novel. This "influence embedding" perspective is a fresh contribution to the field.

The paper is well-motivated, clearly identifying the computational and storage bottlenecks of existing methods. The proposed INFTRACE, being gradient-free and optimizer-independent, directly addresses this "efficiency" motivation, which is supported by the complexity analysis.

### Weaknesses
The primary validation in Section 3.1 (Fig. 3) relies on correlating the estimated influence with the empirical $\Delta p$. This is, to some extent, circular reasoning, as the model was explicitly trained to capture signals from $\Delta p$. A more robust validation would compare the estimated influence to a "ground truth" measure of influence, such as the actual change in model parameters or loss from Leave-one-out (LOO) retraining, even if only on a small-scale setup.

The method appears to require re-training for each specific task, dataset, and model. The learned embeddings are specific to the training dynamics of a particular run. This limits the method's generalizability, as it cannot be readily applied to a new, pre-trained model or dataset without re-collecting all training dynamics and re-training the contrastive model.

The accuracy of INFTRACE is dependent on the quality of the learned representations, which in turn depends on the amount of data (i.e., the number of triplets) used to train the contrastive learning model. While the paper ablates the sampling ratio $\lambda$ (Fig. 4, middle), it does not explicitly discuss how the total number of collected triplets affects the final accuracy of the influence estimation. This is a key parameter for practical application.

### Questions
1.Could the authors elaborate on the feasibility of comparing INFTRACE to Leave-one-out (LOO) on a smaller-scale experiment? This would significantly strengthen the claim that the method captures "true" influence rather than just its own proxy metric ($\Delta p$).

2.Could the authors provide an experiment or discussion on the impact of the contrastive learning dataset size on the accuracy of the influence estimation? This would provide practical guidance on how much data need to be collected for the embeddings to stabilize.

3.The paper notes that REPSIM performs poorly on the LLaMA task, while INFTRACE excels (Table 3), especially at R@100. Could the authors elaborate on why this is the case? 

4.The choice of $\Delta p$ (change in probability of the correct class) is clear for classification tasks. However, for continuous, auto-regressive generation tasks (like with LLaMA), how is $\Delta p$ precisely defined?  Is it the average change in log-probability for all tokens in the generated sequence, or the change in probability of a specific target class?

5.The rationale for choosing $\Delta p$ over $\Delta l$ is given in Sec 2.4 (lines 240-248), arguing it is more interpretable, bounded, and computationally cheaper. However, could the authors provide experimental evidence? For example, what happens if INFTRACE is trained to model the gap in $\Delta l$ instead of $\Delta p$? Would its performance on downstream tasks degrade?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes INFTRACE, an approach to estimate data influence by observing model training dynamics. The method collects the change in prediction probabilities on validation examples throughout the training process. It then uses a contrastive learning objective to learn an embedding space where the proximity between data points is intended to reflect their influence strength. The utility of this method is evaluated on tasks like mislabeled data identification and influential data selection.

While the work is based on a simple and reasonable design, the central claims of efficiency are not supported and appear to be incorrect. The method introduces a heavy computational overhead, the complexity analysis in Table 1 is flawed and incomplete, the empirical evaluations are weak, and the results are not significant enough to justify the costs. The paper does not have enough contribution for a full publication. The authors should consider expanding the scope, addressing the severe computational costs, and re-evaluating the method for a workshop or as a short paper submission.

### Strengths
1. The paper presents a novel perspective on influence estimation by 
evaluating on validation examples at each training step rather than relying on standard gradient-based or retraining methods.

2. The method design is straightforward and intuitive, and it could be useful in specific practical use cases where full training process is accessible.

### Weaknesses
1. The paper's title and abstract claim the method is "efficient." However, the proposed approach appears to be significantly more computation-heavy than many existing methods. INFTRACE requires consistently evaluating the model on validation examples at every training step to collect probability changes. This introduces a very high computational overhead that is not adequately acknowledged.

2. The complexity analysis in Table 1 is unclear, incomplete, and misleading. It fails to properly account for the high cost of INFTRACE's own data collection phase (i.e., step-by-step validation), which is a critical part of its training cost. The table is missing many important baselines, such as TracIn, [Data Shapley in One Training Run], and [The Mirrored Influence Hypothesis].


3. The empirical results are not significant. The performance differences shown in Figure 5 and Tables 2/3 are marginal. Given the significantly higher computation cost of INFTRACE and its dependency on having access to the entire training process, these slight improvements are insufficient to demonstrate advantages over existing approaches.

3. The analysis of the learned representations (Figure 2) seems potentially irrelevant. The t-SNE visualization primarily shows that the embeddings separate samples by their class labels. This can be a trivial result, as it could be achieved by simply concatenating a one-hot label encoding to the features. It is unclear what the "initial embeddings" represent (e.g., from a randomly initialized model? If so, the random distribution is trivial). Most importantly, how is this label-separation property connected o the task of capturing sample influence, which is the stated goal of the embeddings?

4. The correlation plot shows the learned representation compressed the influence relationships. However, it does not sufficiently motivate why this compression is necessary or beneficial for the task of influence estimation. Why do we need to do this?

### Questions
In Table 1, why do Retraining-based methods have computational complexity of $O(DDM)$?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes INFTRACE, a new method for estimating the influence of training data without relying on gradients or retraining. Instead, it leverages training dynamics and uses contrastive learning to distill these signals into an embedding space, where distance represents influence strength.

### Strengths
1. The proposed is efficient compared to gradient-based and retraining-based alternatives, providing another good zero-order data attribution method.
2. Provided comprehensive experiments across datasets and model scales.

### Weaknesses
1. Missing references: the beginning of the paper as well as Section 2 all mentioned most influential examples. However, the reference are incomplete given some noticeable works by a quick search, e.g., [1] and [2]. Other missing references such as various efficient alternatives (Line 44) does not include/discuss representative references such as [3, 4], and training dynamics influence such as [5, 6, 7]. Some arguments are not well-justified and existing works suggest the contrary, making poisoning this work in the literature difficult.
2. Conceptual/Factual Error: For instance Equation (1) and (2) both define $\theta' = \arg\max_{\theta} \ell$, which is confusing and should be $\arg\min$. Moreover, Equation (1) is not how the original influence function (Koh & Liang 2017) is defined.
3. Unclear motivation: The zero-order approximation of the influence function, such as TracIn [5], has already been proposed, where no training gradient or retraining is required. I see little discussion on why the contrastive learning is required while there is an easy-to-use alternative that measures what you want directly.

[1] Yuzheng Hu, Pingbang Hu, Han Zhao, and Jiaqi W. Ma. Most influential subset selection: Challenges, promises, and beyond.

[2] Huang, Jenny Y., David R. Burt, Yunyi Shen, Tin D. Nguyen, and Tamara Broderick. Approximations to worst-case data dropping: unmasking failure modes.

[3] Grosse, R., Bae, J., Anil, C., Elhage, N., Tamkin, A., Tajdini, A., Steiner, B., Li, D., Durmus, E., Perez, E. and Hubinger, E., 2023. Studying large language model generalization with influence functions.

[4] Wang, Jiachen Tianhao, Tong Wu, Dawn Song, Prateek Mittal, and Ruoxi Jia. Greats: Online selection of high-quality data for llm training in every iteration.

[5] Pruthi, Garima, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent.

[6] Bae, Juhan, Wu Lin, Jonathan Lorraine, and Roger Grosse. Training data attribution via approximate unrolled differentiation.

[7] Wang, Jiachen T., Dawn Song, James Zou, Prateek Mittal, and Ruoxi Jia. Capturing the temporal dependence of training data influence.

### Questions
My main question is related to Weakness 1 and 3:
1. Can you provide a detailed discussion on the related literature and how your method compared to the existing works?
2. Can you elaborate why the proposed method (e.g., contrastive learning) is required, while the "ground truth" influence effect you want to measure, i.e., Equation (2), can be obtained from the training dynamics?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces INFTRACE, a data influence estimation framework that utilizes Contrastive Learning (CL) to distill influence signals from model training dynamics. INFTRACE projects influence into a low-dimensional embedding space, demonstrating strong correlation with empirical influence and showing utility in mislabeled data identification, coreset selection, and data attribution for large models.

### Strengths
1. The framework introduces a novel perspective by distilling influence signals directly from model training dynamics, which is both an interesting finding and a practical approach.

2. The method demonstrates strong utility across critical downstream applications, including mislabeled data detection, harmful data detection, and coreset selection, with validation across different model scales.

3. The contrastive learning formulation bypasses the computational bottlenecks of gradient-based methods, making influence estimation tractable for large-scale models.

### Weaknesses
1. My concern is the practical utility of coreset selection. Performance saturates at 50% of data on SNLI and SST-2, with the 10% INFTRACE coreset achieving lower accuracy than simply using 50% of data. I wonder why practitioners would prefer a 10% coreset in this case when using more data yields better performance. Also, why does saturation occur so quickly—is this a model capacity issue or data redundancy? Another concern is that evaluation is limited to in-domain test sets. I wonder if INFTRACE-selected coresets would maintain their advantages when evaluated on out-of-domain NLI tasks.

2. I wonder if the authors compared against computing influence as embedding similarity at earlier layers (e.g., word embeddings) without contrastive training. This ablation seems critical to justify whether contrastive learning offers unique value over simpler content-based similarity approaches.

3. While computational complexity is discussed, I wonder about the actual runtime. The paper would be stronger if it included wall-clock comparisons of CL training + influence calculation versus baselines across different scales to validate the efficiency claims.

4. Table 2 shows detection rates, but I'm curious whether the these amount of mislabeled data actually hurts performance. It would be helpful to report baseline accuracy on clean data and accuracy with 1%, 5%, 10% mislabeling to quantify how much degradation INFTRACE helps mitigate.

5. How sensitive is the method to the absolute |D_val| size? Also, how is the initial D_val selected from the full dataset?

6. In Figure 5, DataInf and RepSim use colors too similar to other baselines (GradDot, TracIn), making visual comparison difficult. Using more distinct colors would improve readability.

### Questions
Please see the Weaknesses for the details.

### Soundness
2

### Presentation
2

### Contribution
2
