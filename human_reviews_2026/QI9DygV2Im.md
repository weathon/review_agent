# Old but Gold: Adaptive Coreset Selection for Robust Dataset Compression

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
The computational and storage costs of large-scale datasets present a significant bottleneck in modern artificial intelligence (AI). While dataset distillation and coreset selection aim to mitigate this by compressing the original datasets into small ones, both have critical limitations. Dataset distillation produces synthetic images that exhibit architectural overfitting and poor transferability to downstream tasks. Conversely, existing coreset selection methods rely on fixed scoring functions, leading to redundant sample selection and performance saturation as the data budget increases. To address these challenges, we propose Adaptive Coreset Selection (ACS), a novel framework that learns an optimal selection strategy for a given budget. ACS employs a multi-stage approach, first building a foundational set of representative samples and then iteratively training models on the selected images to identify hard samples. This adaptive process ensures the final coreset balances representativeness and diversity. We demonstrate the efficacy of ACS on CIFAR-10 and ImageNet, where it outperforms state-of-the-art dataset distillation and coreset selection methods. Notably, on CIFAR-10 with 200 images-per-class, ACS surpasses all baselines by 2\%p in validation accuracy and shows superior generalization to downstream tasks, establishing it as a more robust and scalable solution for dataset compression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Adaptive Coreset Selection (ACS). ACS iteratively builds a coreset of examples. First selects representative examples and then iteratively adds hard examples as the model gets trained. ACS is compared against coreset selection and dataset distillation methods. Methods are evaluated on the CIFAR10 and ImageNet.

### Strengths
- Paper is well written and understandable.
- The method proposed challenges the idea of static coreset selection with good arguments.
- Authors do a good job in showing weaknesses of related work.

### Weaknesses
Insufficient evidence for cost claims and practicality: Paper asserts high “financial and computational costs” but provides no concrete measurements (Questions 1).

Comparison to related work: Paper seems to claim novelty on building the coreset dynamically, however that has been proposed in the past (Questions 2, 3). Additionally, comparison to more recent methods would benefit the paper [2,3,4,6].

Decision reasoning: choices such as starting with easy examples are stated without evidence or reasoning, no experiments as why not to start with hard examples.

Please find further weaknesses in Questions.

### Questions
1. “The financial and computational costs associated with storing and training on such data present major obstacles (Strubell et al., 2019), limiting the adoption of advanced models.” (line 036). However, there are no measurements supporting this claim. For example, could you report time-to-accuracy and wall-clock time benchmarks? How are costs reduced if one must first pretrain an entire model on the full dataset? What is the overhead of using ACS? How much can training time be reduced end-to-end?

 2. “Current approaches employ the same greedy selection strategy regardless of budget size” (Line 053): [3] have adapted these methods to perform dynamic sampling during training and to sample from both static and dynamic distributions.


3. Could you clarify how your method differs from Forgetting as adapted in [3]


4. “Fig. 2b supports our claim. Images selected with EL2N cluster tightly in the feature space, leaving other regions uncovered.” (Line 246): If the goal is to cover all regions, why not maximize distribution coverage using an explicit metric (e.g. Maximum Mean Discrepency (MMD))? More broadly, how do you determine that maximizing distribution coverage is desirable, shouldn’t this depend on the data distribution and the subset size?


5. “Unlike existing methods that selects all samples at once using a fixed score, ACS considers the coreset holistically and dynamically adjusts it selection strategy based on what has already been selected.” (Line 263): Dynamic sampling has already been proposed; see [5-6] and [3]. Please clarify the novelty relative to these approaches.


6. “In the first segment, we select the easiest samples,” (Line 298). Why begin with easy examples, and how is this choice validated? [1] provide theoretical conditions under which one should start with easy versus hard examples.


7. What is the motivation for stratified sampling? How do you determine that having equal numbers per class is preferable to allocating more samples to class A than to class B? For example, when in your case having 10 IPC, why is it not optimal to take 15 images of class A and only 5 images of class B?


8. In general, reporting IPC together with the selection/pruning ratio is more informative. For CIFAR-10/100 this is easy to compute, but otherwise stating that a model is trained on “2%” of the data is more informative than stating it is trained on 10 IPC.


9. “This indicates smaller models are better suited for our algorithm, as larger models require more data to learn meaningful representations” (Line 455): I do not believe this follows from Table 5. To support the claim, it would help to show performance when each model is trained on the entire dataset; otherwise, the result could reflect architectural differences rather than an interaction with ACS. Additionally, given scaling laws, shouldn’t one prefer a method that scales with the number of model parameters?


[1] Sorscher, Ben, et al. "Beyond neural scaling laws: beating power law scaling via data pruning." Advances in Neural Information Processing Systems 35 (2022): 19523-19536.

[2] Kolossov, Germain, Andrea Montanari, and Pulkit Tandon. "Towards a statistical theory of data selection under weak supervision." (ICLR’24)

[3] Okanovic, Patrik, et al. "Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning." The Twelfth International Conference on Learning Representations.

[4] Abbas, Amro Kamal Mohamed, et al. "Effective pruning of web-scale datasets based on complexity of concept clusters." The Twelfth International Conference on Learning Representations.

[5] Mirzasoleiman, Baharan, Jeff Bilmes, and Jure Leskovec. "Coresets for data-efficient training of machine learning models." International Conference on Machine Learning. PMLR, 2020.

[6] Qin, Ziheng, et al. "InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning." The Twelfth International Conference on Learning Representations.

### Soundness
2

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
3

### Summary
This paper tackles the limitations of fixed-score coreset selection and dataset distillation for dataset compression, proposing an Adaptive Coreset Selection (ACS) framework. ACS employs a multi-stage strategy: firstly it selects representative easy samples, then iteratively adds hard examples identified through models which trained on the current subset, progressively constructing a diverse and representative coreset. Comprehensive experiments on CIFAR-10 and ImageNet demonstrate that ACS outperforms state-of-the-art dataset distillation and coreset selection methods, especially at large data budgets, and achieves better generalization on downstream and OOD tasks.

### Strengths
1. **Clear identification of limitations:** Both dataset distillation and standard coreset selection methods have clear limits. Dataset distillation often leads to overfitting because it uses synthetic data. Standard coreset methods rely on fixed scoring rules, which can pick too many similar samples. This causes redundancy and makes the selection less effective.
2. **Methodological innovation**: ACS uses a multi-stage, adaptive scoring method which updates how important each sample is as the coreset grows. Most earlier approaches assume samples are independent, but ACS breaks that assumption. The paper explains this idea clearly. It also shows it well in the pipeline diagram.
3. **Comprehensive experimental validation**: The paper provides robust empirical evidence  on both CIFAR-10 and ImageNet for various images-per-class setups, covering accuracy, transferability, OOD robustness, and ablation on architectural backbones and segmentation strategies.
4. **Qualitative analysis**: Visualization of selected images from early and late stages demonstrates that ACS achieves both representative base coverage and challenging sample inclusion—evidence for improved diversity.
5. **Reproducibility**: Hyperparameters are fully provided in Appendix B, and the algorithm is presented in clear pseudocode, supporting claims of transparency in implementation.

### Weaknesses
1. **Lack of formal theoretical analysis of generalization**: Apart from the intuitive rationale and empirical results, there’s no formal proof or bound concerning why ACS-selected subsets would achieve better generalization or diversity than other approaches. For instance, there’s no analysis of marginal gains or redundancy reduction, nor is there a clear link to submodular optimization or curriculum learning theory.
2. **Scalability and compute cost**: While the paper claims ACS is scalable, the multi-stage process (involving multiple model trainings) likely incurs noticeably larger compute than single-pass, fixed-score methods. There is no analysis (empirical or theoretical) on computational cost versus performance, nor is wall-clock time or total FLOPs reported in any table.
3. **Algorithmic ablations lacking**: Apart from backbone and segment number (see **Table 5** and **Table 6**), there is no thorough ablation of, for example, whether selecting only from misclassified samples at later stages is essential, or how sensitive ACS is to the exclusion of hard or easy samples in various regimes. This makes it hard to judge whether the method’s advantage is robust to minor tweaks.
4. **Notational clarity**: While the pseudocode in Appendix A and equations throughout are generally sound, some ambiguity remains. For example, the notational switch between $\mathcal{C}_t$ (cumulative coreset) and $\mathcal{S}_t$ (current segment) could be more visually separated throughout the main text and algorithm. Additionally, the loss function notation is at times overloaded.

### Questions
1. **Choice and generality of scoring function**: Have you experimented with alternative context-aware scoring metrics beyond classification loss, such as prediction margin, uncertainty (entropy), or uncertainty-gradient products? Would such measures yield better/robuster coresets? Please comment on the generalizability of the principle.
3. **Diversity metrics**: Beyond qualitative t-SNE and segment visualizations, have you quantified the diversity of ACS-selected samples (e.g., feature-space coverage, cluster spread, redundancy metrics)? Could you add such assessments to strengthen the evidence for ACS’s claims?
5. **Hyperparameter scheduling**: Is there any principled scheme to set the number of segments ($T$) based on data or model properties? Have you considered automatic tuning or meta-learning approaches?
6. **Sensitivity to model/backbone choice**: Table 5 suggests varying robustness, but how stable is ACS performance across a wider range of architectures, especially for current large transformer models? Is there a risk of over-specialization?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Adaptive Coreset Selection (ACS), a novel framework for dataset compression that overcomes limitations of both dataset distillation and existing coreset methods. ACS adaptively selects samples in stages, balancing representativeness and diversity by iteratively training models and selecting misclassified examples. It outperforms state-of-the-art methods on CIFAR-10 and ImageNet. Moreover, ACS also shows good generalization and robustness across architectures and downstream task.

### Strengths
- This work reveals the fundamental limitations of dataset distillation and traditional coreset selection methods, and proposes ACS—a more robust and scalable data compression framework—demonstrating that carefully selected real images outperform synthetic ones in terms of cross-architecture generalization and out-of-distribution robustness.  
- We introduce the first adaptive coreset selection framework based on context-aware scoring, which breaks away from the assumption of fixed sample scores. It models sample importance as a dynamic property that evolves with the already-selected samples and naturally preserves diversity through a multi-stage iterative strategy.  
- The method consistently surpasses existing coreset selection and dataset distillation approaches on CIFAR-10 and ImageNet, and continues to improve even under high budgets, significantly outperforming random selection baselines.  
- Baseline methods are reproduced following standard protocols and using publicly available codebases (e.g., DeepCore), ensuring strong reproducibility.

### Weaknesses
1. The proposed method heavily relies on the previously selected batch of data, which introduces a significant efficiency bottleneck. While this has minimal impact on small-scale datasets (e.g., CIFAR-10), the time cost for sample selection grows dramatically as dataset size increases. For instance, on ImageNet, the method achieves performance nearly on par with random selection but at a substantially higher computational cost. The authors should take this issue seriously: if applied to even larger datasets (e.g., LAION-400M), the method’s advantages may diminish entirely. Therefore, a comparison of actual runtime against baseline methods is essential.
2. We know that random sampling remains highly effective on large-scale datasets and is compatible with any model architecture. In contrast, the method proposed in this paper relies heavily on the backbone architecture. Although the authors conducted related experiments, it remains unclear whether their approach exhibits strong generalization in terms of transferability. For example, why not use a ViT architecture for sample selection? How would sampling perform with an MLP-based model? Are models with the same architecture inherently more advantageous? The authors need to provide more theoretical insights and experimental analyses to address these questions.
3. Why not compare against some dynamic dataset pruning methods [1]?

[1] InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning

### Questions
See weaknesses

### Soundness
3

### Presentation
3

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
- This paper introduced a multi-stage coreset selection framework ,named “Adaptive Coreset Selection (ACS)”, that adjusts its selection criteria based on model performance at different stages, in an adaptive manner.
- The method selects coreset in an easy to hard manner. It selects the easy and most representative samples first and then iteratively train models on selected samples to select the harder samples.
- It aims to address performance saturation at higher images-per-class setting selection for coreset methods.
- The paper compares performance against various coreset selection and dataset distillation methods.

### Strengths
1. The problem motivation is clearly articulated, regarding addressing performance saturation for higher selection budgets.
2. The method is intuitive and straightforward, borrowing ideas from curriculum learning. 
3. The method’s practical implementation does not require any complex optimisation.
4. The paper has provided ablation studies for backbone architectures (Table 5) and number of segments (Table 6).

### Weaknesses
**Limited Novelty**

- The methodology combines aspects from already well-established techniques such as curriculum learning and iterative hard example mining (standard practice in bootstrapping and active learning).

**Limited comparison baselines**

- The coreset selection methods used as baseline are used from DeepCore library , which was published in 2022. Recent coreset selection works which have shown better performance than these methods have not been considered. Some of the works are listed below:

A) Moderate Coreset: A Universal Method of Data Selection for Real-world Data-efficient Deep Learning (ICLR 2023)

B) Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy (NeurIPS, 2023)

C) Coverage-Centric Coreset Selection for High Pruning Rates (ICLR, 2023)

E) Data Pruning via Moving-one-Sample-out (NeurIPS 2023)

F) Noise-free Loss Gradients: A Surprisingly Effective Baseline for Coreset Selection (TMLR, 2025)

**Mismatch in results reported**

- Referring to Table 3, the accuracy values reported for SRe2L and RDED are significantly lesser than what these papers have reported in their results. For example, for ImageNet-1K, RDED reports (Table 2 of RDED paper) an accuracy of 42.0% and 56.5% for IPC=10 and IPC=50 respectively. While, Table 3 of this paper has reported 12.5% and 29.8% respectively, which are significantly less and misleading in nature for comparison purposes. A similar discrepancy is observed for SRe2L as well.

**Lack of timing analysis**

- The multi-stage nature of coreset selection raises a question regarding computational overhead in terms of GPU usage and time required for coreset selection. A comparative analysis of GPU requirement and timing for coreset selection with other baseline methods would be very helpful in understanding impact of ACS. 

**Number of segments**
 
- From Table 6, number of training segments required to achieve higher performance for CIFAR-10 is 10 for IPC=100. The paper does not provide similar results for ImageNet-1K dataset. 

**Limited datasets for comparison**

- Other standard benchmark datasets such as CIFAR-100 and Tiny ImageNet are not considered. A performance comparison on these two datasets would be insightful regarding efficiency of ACS on datasets of various number of classes.

### Questions
- Please refer to the weaknesses section of the review.

### Soundness
2

### Presentation
3

### Contribution
2
