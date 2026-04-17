# What is the role of memorization in Continual Learning?

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Memorization impacts the performance of deep learning algorithms. Prior works have studied memorization primarily in the context of generalization and privacy. This work studies the memorization effect on incremental learning scenarios. Forgetting prevention and memorization seem similar. However, one should discuss their differences. We designed extensive experiments to evaluate the impact of memorization on continual learning. 
We clarified that learning examples with high memorization scores are forgotten faster than regular samples. Our findings also indicated that memorization is necessary to achieve the highest performance. However, at low memory regimes, forgetting regular samples is more important. We showed that the importance of a high-memorization score sample rises with an increase in the buffer size. 
We introduced a memorization proxy and employed it in the buffer policy problem to showcase how memorization could be used during incremental training. We demonstrated that including samples with a higher proxy memorization score is beneficial when the buffer size is large.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the role of memorization in continual learning. The authors argue that memorization can both enhance performance and accelerate forgetting. The main contribution of the paper lies in proposing a memorization proxy and validating its usefulness in buffer policy design.

### Strengths
1. The paper explores the relationship between memorization and forgetting in continual learning.
2. Since existing evaluation methods for memorization scores are not directly applicable to continual learning, the authors propose a memorization score proxy as an approximate substitute.
3. For memory-based continual learning methods, the proposed approach provides insights for guiding example selection.

### Weaknesses
1. Most of the paper’s conclusions are empirically derived, lacking theoretical support for the relationship between memorization and forgetting in continual learning.
2. The proposed proxy in Eq. 3 assumes a positive correlation with memorization degree. This assumption lacks theoretical justification, and the reported correlation coefficient indicates only a moderate relationship. Such a weak correlation could introduce noise into buffer example selection, making the results difficult to interpret.
3. One of the paper’s core contributions is the memorization-aware replay method; however, the experimental evidence does not strongly support the claim that “storing high-memorization samples is beneficial.” In fact, the top-k strategy performs significantly worse than bottom-k and mid-k in nearly all benchmarks, suggesting that storing high-memorization samples is detrimental when the buffer size is small. Moreover, mixing 10% of top-k samples into the bottom-k or mid-k buffers leads to negligible (or even negative) gains, all within the margin of standard deviation. Thus, the conclusion that “the importance of high-memorization samples increases with larger buffer size” seems weakly supported by the data.
4. To convincingly demonstrate that “memorization is necessary to achieve the highest performance,” an experiment capable of actively suppressing memorization without affecting pattern learning would be required. However, such isolation of the memorization variable is extremely challenging, and the current experiments cannot achieve this level of control.
5. The paper defines memorization at the sample level, yet its discussion of forgetting seems to operate at the representation level. For instance, is a high-memorization sample forgotten because it relies on fragile or rare features? If the model memorizes many such samples, might it gradually learn more generalizable representations to capture them, thereby improving overall performance? This conceptual ambiguity leaves it unclear whether memorization should ultimately be suppressed or encouraged.

### Questions
Please refer to Weaknees.

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
This paper investigates the role of memorization in the context of continual learning (CL). The authors define a memorization score for individual examples (based on prior definitions from Feldman et al.), and study how examples with high memorization scores behave under incremental training regimes. The work thus provides empirical insight into how “memorization” and “forgetting” interact in CL, and offers practical suggestions for buffer‐selection policies.

### Strengths
1. Through well-designed experiments on standard continual learning benchmarks (such as the CIFAR100 split), the authors reveal a pronounced correlation between memory scores, buffer size, and forgetting behavior.
2. The explicit study of how memorized examples behave under continual learning is a valuable contribution.
3. I think the paper is well-written and generally easy to follow.

### Weaknesses
1. Despite robust experimental validation, the paper offers a limited theoretical explanation for why highly memorized samples are more prone to forgetting in continual learning.
2. This study argues that memory should be considered in incremental training, but it does not explore how much additional computational or memory overhead would be introduced by calculating memory scores or implementing proxy strategies in practical applications.

### Questions
1. Have you tested the proxy memorization buffer policy on non‐vision tasks (e.g., NLP, tabular, reinforcement‐learning streams) to assess generality?
2. Can you provide insight into why high‐memorization examples are more quickly forgotten in continual sequences? Is it because they occupy more capacity, are less “representative”, or because the model shifts representation more aggressively?
3. In continual learning, retaining generalizable features is often more important than memorizing individual examples. Can you explain the differences and connections between these two?

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
4

### Summary
This paper explores the role of memorization in continual learning (CL), examining how it influences forgetting and overall performance. While both memorization and forgetting prevention aim to retain knowledge, the paper clarifies that memorization refers to learning specific samples—often atypical or rare—whereas CL seeks to preserve generalizable knowledge across sequential tasks. Using Feldman’s definition of memorization score, it shows that samples with higher memorization are forgotten faster, that memorization increases with the number of classes, and that wider networks tend to memorize more while deeper ones memorize less. Since computing exact memorization scores is computationally expensive, the paper proposes a proxy based on the iteration when a sample is first learned and remains correctly classified thereafter, which strongly correlates with Feldman’s estimator. It integrates this proxy into a “memorization-aware experience replay” buffer policy, finding that storing low- or medium-memorization samples improves performance when memory is limited, while high-memorization samples become useful when the buffer is large.  Experiments on Split-CIFAR10, CIFAR100, and Tiny ImageNet confirm that memorization is necessary to reach high accuracy but can also accelerate forgetting after distribution shifts.

### Strengths
1. Provides a novel and focused investigation of memorization in continual learning, a topic rarely examined beyond generalization or privacy contexts.

2. Clearly distinguishes memorization from forgetting prevention, offering conceptual clarity that helps refine the theoretical framing of continual learning.

3. Includes comprehensive experiments across multiple datasets (CIFAR10, CIFAR100, Tiny ImageNet), architectures (ResNet18/34/50), and memory settings, ensuring robust and generalizable findings.

4. Introduces a computationally efficient memorization proxy that correlates strongly with Feldman’s estimator, enabling large-scale empirical evaluation.

5. Demonstrates how memorization-aware buffer policies can guide experience replay strategies and improve continual learning performance under varying memory budgets.

6. Presents well-structured and reproducible methodology, with code availability and detailed hyperparameter reporting.

### Weaknesses
1. The paper is at times difficult to follow, and several descriptions lack precision. For example, in Figure 5, it is unclear what the training iterations on the y-axis represent. Upon checking the appendix, it appears that the y-axis corresponds to the proposed training-iteration–based memorization proxy, but this is not clearly stated in the main text. If this interpretation is incorrect, please clarify what is actually plotted. The manuscript would benefit from improved figure captions and clearer explanations of the plotted variables.

2. While the paper aims to study memorization in continual learning, all memorization scores are computed offline using Feldman’s estimator under stationary training and are not updated dynamically during incremental training. The scores are then tracked across tasks as the data distribution shifts. Consequently, the claim—that “memorized samples are forgotten faster”—rests on static pre-training memorization scores, which may not reflect what is actually memorized by the continually updated model. This design limits the causal validity of the conclusion that memorization directly influences forgetting in continual learning.

3. The proposed proxy—defined as the first iteration at which a sample is correctly classified and remains so thereafter—correlates with Feldman’s estimator (r ≈ 0.8), but may actually capture learning dynamics (e.g., sample learning speed or gradient stability) rather than memorization itself. For example, early-learned and consistently correct samples could receive low proxy scores even if they are later forgotten, while atypical or noisy samples that stabilize late may appear highly “memorized.” The paper would benefit from a qualitative analysis (e.g., visualization of high-proxy samples or representation trajectories) to confirm that high proxy values indeed correspond to memorized behavior as per Feldman’s definition—memorization without generalization.

4. Although the empirical findings (e.g., that highly memorized samples are forgotten faster) are intriguing, the paper does not provide a theoretical rationale or mechanistic explanation for these effects. Are the observed patterns due to feature drift, gradient interference, or representational collapse? Furthermore, the claim that “wider models memorize more but forget less” appears contradictory to the earlier finding that memorization accelerates forgetting, yet this paradox remains unresolved. A more formal or analytical treatment would significantly strengthen the contribution.

Minor Comments:

1. Figure 1 (Right Panel): The plot is difficult to read due to small font and contrast. Please improve the figure’s legibility.

2. Section 3.3 (First Line): The equation reference or variable notation is not clearly formatted; please revise for consistency.

3. Reproducibility: Upon acceptance, please consider open-sourcing the trained model checkpoints along with the code, as this would greatly facilitate future research and validation.

4. Table 1: Please include citations for all baseline methods (e.g., Reservoir Sampling, Rainbow Memory, BCSR, PBCS) directly in the table for easier cross-referencing.

### Questions
1. Could you please clarify precisely what the y-axis (“training iterations”) in Figure 5 represents? Does it correspond to the proposed training-iteration–based memorization proxy, or to actual training steps? If it is the proxy, please make this explicit in the main text and caption.

2. Since all memorization scores are computed offline using Feldman’s estimator under stationary training, how do you justify using these static scores to analyze forgetting in a dynamic continual-learning setup? Have you attempted to recompute or approximate memorization scores during incremental training to confirm that the same samples remain “memorized” over time?

3. How do you ensure that the proposed proxy (first correct-classification iteration) truly measures memorization rather than training difficulty or convergence speed? Have you examined qualitative examples or visualization of samples with high proxy scores to confirm that they exhibit the behavior expected of memorized examples?

4. To the best of my understanding, the correlation between the proxy and Feldman’s estimator (r ≈ 0.8) was computed on a limited subset of CIFAR-100 samples. Could you provide per-class statistics to assess the robustness of this correlation across the full dataset?

5. Can you elaborate on the underlying mechanism explaining why high-memorization samples are forgotten faster? Have you explored whether this effect is driven by feature drift, gradient interference, or differences in representational overlap between tasks?

6. The paper reports that wider models memorize more but forget less, which appears inconsistent with the claim that memorization accelerates forgetting. Could you clarify how these two findings coexist, or provide additional analysis (e.g., gradient alignment or curvature metrics) to reconcile them?

7. When reporting accuracy for “memorized samples,” are these measurements made on the training set or a held-out validation split? 

8. Will you release the trained model checkpoints upon acceptance? Doing so would help the community reproduce and extend your findings.

9. Could you improve the readability of Figure 1 (right panel) and revise Section 3.3’s notation for clarity? Also, please include citations for all baselines (Reservoir, Rainbow Memory, BCSR, PBCS) directly in Table 1 for easier cross-reference.

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
The paper investigates the empirical relationship between sample-level memorization and catastrophic forgetting in class-incremental continual learning (CL). The well-known memorization score proposed by Feldman is used for the study. The central claims of the paper are: (1) Samples with high memorization scores, computed in a stationary (full-dataset) setting, are forgotten more rapidly during incremental training than their low-memorization counterparts. (2) Memorization increases as the number of classes in a dataset grows, as model width grows, and decreases with model depth. (3)  In Continual Learning, the trend is reversed, namely, deeper models forget more while wider models forget less. (4) The experiments suggest that for small buffers, prioritizing samples with low or medium proxy scores is beneficial, while the utility of high-memorization samples increases with larger buffer sizes.

### Strengths
* This works is one of the few existing works studying memorization in continual learning, which I found extremely interesting and important, and potentially significant. In this sense, the work would be original and significant. However, in my view the paper suffers from fundamental flaws that invalidate its primary conclusions (see weaknesses).

### Weaknesses
* While the paper's title suggests a board study of memorization role in continual learning, the focus is only on class-incremental continual learning, so the stated findings at best could only be applicable to this narrow problem. Thus, the study conducted in Section 3.2 on the role of varying number of classes could be complete irrelevant to board continual learning setups.

* In a similar vein, the paper calculates memorization scores in an offline, stationary setting (i.e., training on the full CIFAR100 dataset) and then using these static scores to analyze sample behavior during dynamic incremental training. This is a significant methodological compromise. The paper itself concedes this flaw in Appendix H, stating "It may suggest that data with high memorization scores computed offline doesn't necessarily correspond to data that is actually memorized in incremental training". A proper study would investigate the process of memorization within CL, which would be the more fundamental question.

* The study in section 3.2 is fundamentally flawed in the following sense: Note that the memorization score definition in eq 1 and eq 2 are model and data distribution dependent. That is, for a given architecture and given data distribution they assess the memorization of different samples. Thus, one cannot use these equations to compare the memorization of different models and/or different data distributions. When changing the model size (depth or width) or the number of classes, we are changing the architecture and data distribution. Thus, I find the majority of the results in the paper, particularly those from section 3.2 invalid.

* The central finding that "memorized samples are forgotten at significantly faster rates" and the stated conundrum in section 6 may be an artifact of the "regularization-free" setting used for continual learning, i.e. using SGD with momentum set to 0.0 and, most critically, weight decay set to 0.0 vs a regularized setting used for offline/ non-continual learning setting. This regularization-free setting is known to promote memorization, rather than an intrinsic property of continual learning. I further encourage the authors to look into the literature on plasticity loss and approaches akin to weight decay/L2-regularization.

* The paper uses eq 3 as a proxy of memorization score. This approach basically uses the first iteration $j$ after which the sample $i$ is always classified correctly. While intuitive, I find its implementation, described in Algorithm 1 flawed. The paper states that checking every sample at every iteration is too costly. The algorithm, therefore, only updates the status of a sample $i$ when it appears in the current minibatch. This transforms the proxy from a deterministic function of the training trajectory into a highly stochastic one, dependent on the minibatch sampling order. A sample could be "learned" at iteration $j=100$ but, by chance, not be sampled again until $j=5000$, at which point its proxy score $v_i$ would be recorded as 5000, not 100 (see line 10 of Algorithm 1). The impact of this substantial noise and bias, which fundamentally separates the implementation (Algorithm 1) from the definition (Eq. 3), is not analyzed.

* Figure 5 shows only a weak, "moderate" correlation (Pearson $r=0.594$) between the proxy and the "Original Memorization Score" stated in eq 1. Note the indirect connection presented in Figure 8 (strong correlation ($r=0.808$) between eq 2 and eq 3) and Figure 7a (strong correlation ($r=0.757$) between eq 1 and eq 2) in the Appendix are not adequate to substantiate the validity of eq 3.

* The performances reported in Tables are too low and have a huge variance to support the conclusions and findings. For instance, the statement "we can see that selecting lower and mid proxy memorization scores obtains better results than reservoir sampling" is not substantiated in my view in light of the results reported in the tables.

### Questions
(1) The paper claims to use $k/n=0.5$ and $u=250$ (number of networks trained to approximate the expectation). This implies that for each sample $i$, $2 \times u = 500$ networks must be trained. For the CIFAR100 dataset ($n=50000$), computing this score for all samples would require $500 \times 50000 = 25,000,000$ network-trainings! The paper reports a training time of 10 minutes per network, implying a total compute of approximately 4.16 million GPU-hours. This is irreconcilable with the paper's reported computational budget of "over 3500 neural networks" or the total project time of 1476 hours.

### Soundness
2

### Presentation
2

### Contribution
2
