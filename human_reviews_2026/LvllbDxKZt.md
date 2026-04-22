# Align-SAM: Seeking Flatter Minima for Better Cross-Subset Alignment

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 2

## Abstract
Sharpness-Aware Minimization (SAM) has proven effective in enhancing deep neural network training by simultaneously minimizing the training loss and the sharpness of the loss landscape, thereby guiding models toward flatter minima that are empirically linked to improved generalization. From another perspective, generalization can be seen as a model’s ability to remain stable under distributional variability. In particular, effective learning requires that updates derived from different subsets or resamplings of the same data distribution remain consistent. In this work, we investigate the connection between the flatness induced by SAM and the alignment of gradients across random subsets of the data distribution, and propose \textit{Align-SAM} as a novel strategy to further enhance model generalization. Align-SAM extends the core principles of SAM by promoting optimization toward flatter minima on a primary subset (the training set), while simultaneously enforcing low loss on an auxiliary subset drawn from the same distribution. This dual-objective approach leads to solutions that are not only resilient to local perturbations but also robust against distributional shifts in each training iteration. Empirical evaluations demonstrate that Align-SAM consistently improves generalization across diverse datasets and challenging settings, including scenarios with noisy labels and limited data availability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper propses Align-SAM, a variation of SAM which modifies the SAM update to better align gradients between different subsets of the training set. The paper demonstates theoretically that indeed their algorithm leads to better alignment as well as the same convergence rate as SAM. Empirically, the method outperforms SAM variants.

### Strengths
The main strength of this paper in my view are the strong empirical results. Align-SAM's strong empirical performance is demonstrated on numerous datasets, architectures and against several baseliens. The method is also intuitive and is theoretically motivated.

### Weaknesses
The paper doesn't too have many weaknesses in my view. First, in Figure 2, the tradeoff coefficient range cuts off on the lower side at 0.25. Is it possible to consider a tradeoff coefficient of zero? Relatedly, is there a study varying the size of \rho? Second, several of the figures and tables are too small- I hope the authors use the extra page to enlarge these (e.g. Table 2, 3, 5, Figure 1).

Minor comments
- Is "Agnostic-SAM" in Figure 2 another name for Align-SAM?

### Questions
- What is the behavior of the method when the tradeoff coefficient is 0?
- Is there a study on how changing \rho affects performance?
- See minor comments above

### Soundness
3

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
The paper proposes Align-SAM, a novel optimization method that extends SAM to further enhance generalization under different subsets. Align-SAM guides the model toward solutions that minimize sharpness and loss on a primary training subset while simultaneously maintaining low loss on an independently drawn auxiliary subset sampled from the same distribution. This approach views generalization through the lens of cross-subset alignment and is designed to find minima that are not only resilient to local perturbations but also robust against distributional variability in each training iteration. Empirical results demonstrate consistent performance gains over SAM and its variants across diverse and challenging settings.

### Strengths
1. **Theoretical Foundation and Methodological Proposal**: The authors approach generalization from a novel perspective, framing it as an alignment across random subsets drawn from the same data distribution. Building on this viewpoint, the methodology is supported by Theorem 1, an extension of the PAC-Bayes theorem, which establishes an upper bound on generalization loss using an independently drawn auxiliary subset. Furthermore, Theorem 2 provides theoretical justification that the proposed iterative update scheme (Equations 4–6) successfully maximizes the dot product of the gradients computed on the primary batch and the auxiliary batch, thereby encouraging these gradients to become more congruent.

2. **Extensive Experiments**: Align-SAM is evaluated extensively across diverse and challenging settings, demonstrating consistent effectiveness. The experiments include: training from scratch on large-scale datasets (ImageNet, Food101), mid-sized datasets (CIFAR-100/10), transfer learning using various architectures (ResNet, EfficientNet, ViT-Adapter-S), and evaluation of robustness against noisy labels (CIFAR-10/100). Additionally, domain generalization tests show robustness under domain shifts (ImageNet-R and ImageNet-C). Ablation studies reinforce the core mechanism, showing that Align-SAM achieves significantly flatter loss landscapes and lower Hessian eigenvalues compared to SAM and SGD.

### Weaknesses
# Major Concerns

> 1. **Insufficient Explanation of Preliminaries**:
> The theoretical foundation and the proposed Align-SAM heavily rely on familiarity with SAM, making the manuscript very difficult to follow for readers unfamiliar with SAM's specifics. Specifically, in Theorem 1, rather than stating "Under some mild conditions similar to SAM (Foret et al., 2021)," the specific conditions necessary for the theorem's validity should be explicitly mentioned, as is standard for a formal theorem.

> 2. **Necessity of the Auxiliary Batch Set**: 
> Align-SAM introduces an auxiliary batch requiring additional computational overhead due to extra forward and backward passes compared to standard SAM. Even though authors report Table 7 to alleviate the issues with this, there is still a question for the efficiency  of Align-SAM (please refer to W3(b) for details).


> 3. **Questions Regarding Fair Comparison**:
>
>> a. **Baseline Explanation**: The experimental baselines require further description. The paper compares Align-SAM against VASSO, GSAM, and LookSAM. Information should be provided (briefly) on the core methods utilized by each of these techniques, and crucially, whether they require an auxiliary set. This information is necessary for the reader to assess the fairness of the comparison and the complexity trade-offs.
>
>> b. **Practicality of Training Time**: Table 7 reports training time overhead based on CIFAR100 experiments. However, the large-scale ImageNet experiments utilize batch sizes of $\|B_t \|=2048$ and $\|B_a \|=512$. Given the scale of ImageNet and the fourfold increase in batch sizes, it is likely that the actual training time overhead is significantly larger. This raises doubts about the practical relevance of Table 7's results for large-scale implementations like ImageNet.


# Minor Concerns
> 4. **The manuscript requires general refinement:**
>
>> a. **Reference citation formats appear incorrect** or duplicated in the "Related Works" section.
>
>> b. **Notation is sometimes omitted**, e.g. $L$ in Theorem 1.
>
>> c. In Line L308, the text refers to "Asgnostic-SAM." Please confirm if this is a **typo** and should be "Align-SAM" or "Align-ASAM".
>
>> d.  In the formulation of the Taylor expansion leading to Equation (7) (L235), the step often involves an equality (=) based on a first-order  (L221). It is suggested to clarify whether the equality should be changed to an approximation (≈), and if so, discussion regarding the magnitude of the error term should be included for completeness.

### Questions
1. Could you elaborate on the reasoning why, in the statement of Theorem 1 (L172), $$\mathcal{L}_{\mathcal{D}} (\theta^* | \mathcal{S}^{a})$$ is used as the objective to minimize (Equation 3), rather than $\mathcal{S})$.

2. In line with W3(a), could you briefly explain the core mechanisms of baseline methods and explicitly state whether they require an auxiliary set?

### Soundness
2

### Presentation
2

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
This paper revisits Sharpness-Aware Minimization (SAM) from the perspective of cross-subset alignment. The authors argue that good generalization requires not only finding flat minima on the training subset, but also maintaining stable performance across independently sampled subsets from the same data distribution. To achieve this, they propose Align-SAM, which incorporates an auxiliary mini-batch and encourages gradient alignment between the primary and auxiliary subsets during training. The method is theoretically supported by a PAC Bayes generalization bound and a gradient congruence theorem. Extensive experiments validate the effectiveness of the proposed method.

### Strengths
- The paper presents a clear motivation. It aims to reduce sharpness and loss on the primary subset while maintaining low loss on the auxiliary subset, aligning well with the goal of improving generalization.
- The paper is theoretically grounded. It builds on PAC-Bayesian theory to formalize how cross-subset alignment reduces generalization error, and provides mathematical guarantees showing that the proposed update rule enhances gradient congruence across subsets, thereby offering a solid theoretical foundation for the method.
- The experiments span a wide range of tasks, including standard classification, transfer learning, and learning with label noise, demonstrating the robustness and effectiveness of the proposed method across diverse scenarios.

### Weaknesses
- According to Algorithm 1, the proposed method requires four backward passes for each parameter update. Although the authors state that using a much smaller auxiliary batch size $|B^a|$ than the target batch size $|B^t|$ can mitigate the computational overhead, this strategy may necessitate a large overall batch size (e.g., Table 7 uses a batch size of 512), and large-batch training may potentially harm performance.

- The experiments primarily evaluate CNN-based architectures (e.g., WideResNet, PyramidNet, DenseNet). Including results on transformer-based models such as ViT would further strengthen the paper, given their broad adoption and distinct optimization characteristics.

- More representative SAM-based methods, such as SAF [1], GAM [2], ImbSAM [3], CC-SAM [4], and Focal-SAM [5] should be included in the related work for a more comprehensive review.
- There are some typos, such as in line 307, where "Asgnostic-SAM" should be corrected to "Agnostic-SAM"

-----

[1] Sharpness-Aware Training for Free, NeurIPS 2022

[2] Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization, CVPR 2023

[3] ImbSAM: A Closer Look at Sharpness-Aware Minimization in Class-Imbalanced Recognition, ICCV 2023

[4] Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition, CVPR 2023

[5] Focal-SAM: Focal Sharpness-Aware Minimization for Long-Tailed Classification, ICML 2025

### Questions
Please see above.

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
This paper proposes Align-SAM, a new optimization method based on Sharpness-Aware Minimization (SAM). The authors introduce the idea of "cross-subset alignment," arguing that good generalization requires a model trained on one subset of data to also perform well on an auxiliary, independently drawn subset. The proposed Align-SAM method extends SAM by optimizing for flat minima on a primary training batch while simultaneously enforcing low loss on this auxiliary batch. This is implemented as a modification to the SAM update rule that incorporates gradients from both batches, with the goal of encouraging gradient alignment. The authors provide empirical results across various settings, including image classification, transfer learning, and training with noisy labels, claiming consistent generalization improvements.

### Strengths
**Thorough Evaluation:** The paper presents a wide range of experiments, evaluating the method on standard classification, transfer learning, domain generalization, and noisy labels.

### Weaknesses
- **Marginal Gains:** The central claim of "consistent improvement" is overstated. In many key experiments (e.g., CIFAR classification and transfer learning), the performance gains over strong baselines like SAM and ASAM are marginal and often fall within the reported standard deviation. These minor benefits do not appear to justify the significant computational overhead of an extra forward/backward pass per step.
- **Lack of Novelty and Missing Discussion:** The novelty of the update rule is questionable, as it resembles previous works like Lookahead SAM [1] and Lookbehind SAM [2]. The authors fail to adequately discuss or compare against these highly relevant methods in the main text, making it difficult to assess the originality and relative performance of the proposed contribution.

[1] Yu, Runsheng, Youzhi Zhang, and James Kwok. "Improving sharpness-aware minimization by lookahead." *Forty-first International Conference on Machine Learning*. 2024.

[2] Mordido, Gonçalo, et al. "Lookbehind-SAM: k steps back, 1 step forward." *arXiv preprint arXiv:2307.16704* (2023).

### Questions
1. Should the authors clarify that this paper focus on *in-distribution* generalization, given that *out-of-distribution* generalization is a distinct concept?
2. What are the meaning of $\nabla\_\theta \mathcal{L}\_{B\_t}(\tilde{\theta}\_l^t)$ and $\nabla\_\theta \mathcal{L}\_{B\_a}(\tilde{\theta}\_l^a)$? Do we want to align the gradient of $B\_t$ and $B\_a$ at $\theta\_l$, i.e., $\nabla\_\theta \mathcal{L}\_{B\_t}({\theta}\_l)$ and $\nabla\_\theta \mathcal{L}\_{B\_a}({\theta}\_l)$, instead?
3. The comparison appears to be unfair because Align-SAM effectively operates with a larger batch size than the baseline methods.
4. Could you please add standard deviations to Table 1 and Table 3 for a more rigorous comparison? Also, given its importance, should the analysis of Figure 1 (currently in Appendix A.3) be moved to the main text?
5. How does Align-SAM compare to a simpler baseline that adds a direct (weighted) gradient alignment regularizer to the standard SAM loss?
6. Could the authors clarify the intuition for why Align-SAM improves performance under noisy labels, especially when the test dataset follows a different distribution?
7. Missing related work about SAM
- Yu, Runsheng, Youzhi Zhang, and James Kwok. "Improving sharpness-aware minimization by lookahead." Forty-first International Conference on Machine Learning. 2024.
- Nguyen, Tuan H., et al. "Changing the training data distribution to reduce simplicity bias improves in-distribution generalization." Advances in Neural Information Processing Systems 37 (2024): 68854-68896.

### Soundness
2

### Presentation
1

### Contribution
1
