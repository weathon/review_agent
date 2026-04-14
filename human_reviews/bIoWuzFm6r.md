# Machine Unlearning for Streaming Forgetting

- Decision: Reject
- Scores: 3, 5, 5, 6

## Abstract
Machine unlearning aims to remove knowledge derived from the specific training data that are requested to be forgotten in a well-trained model while preserving the knowledge learned from the remaining training data. Currently, machine unlearning methods typically handle all forgetting data in a single batch, removing the corresponding knowledge all at once upon request. However, in practical scenarios, requests for data removal often arise in a streaming manner rather than in a single batch, leading to reduced efficiency and effectiveness in existing methods. Such challenges of streaming forgetting have not been the focus of much research. In this paper, to address the challenges of performance maintenance, efficiency, and data access brought about by streaming unlearning requests, we introduce an online unlearning paradigm,  formalizing the unlearning as a distribution shift problem. We then estimate the altered distribution and propose a novel online unlearning algorithm to achieve efficient streaming forgetting without requiring access to the original training data. Theoretical analyses confirm an $O(V_T\sqrt{T} + \Delta_T)$ error bound on the streaming unlearning regret, where $V_T$ represents the cumulative total variation in the optimal solution over $T$ learning rounds and $\Delta_T$ represents the cumulative total divergence between remaining and forgetting data distributions. This theoretical guarantee is achieved under mild conditions without the strong restriction of convex loss function. Experiments across various models and datasets validate the performance of our proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper addresses the challenge of machine unlearning, a process where users can request that their data be removed from a model’s knowledge in compliance with recent regulatory requirements. Unlike traditional batch-based approaches to unlearning, which handle all deletion requests in a single step, this work focuses on the streaming scenario where requests arrive sequentially. The authors frame the problem through an online learning lens, where each unlearning request is treated as a distribution shift to efficiently remove data incrementally. They propose a novel online unlearning algorithm that does not require access to the original training data, ensuring that knowledge derived from forgotten data is removed while preserving insights from the remaining data. The paper's theoretical analysis provides guarantees on error bounds by quantifying the cumulative regret based on total variation and divergence shifts between data to be forgotten and retained. The method achieves guaranteed regret bounds without assuming convexity of the loss function. Experimental results across various datasets and models demonstrate the method's practicality and efficiency in real-world streaming unlearning scenarios.

### Strengths
* The online learning perspective is novel, and in retrospective makes a lot of sense for sequential unlearning requests. 
* The algorithm is practical and seems on par with baselines according to the experimental results.

### Weaknesses
* **Heuristic Approach to Unlearning.** The paper analyzes one heuristic approach to unlearning (Problem (1)), which jointly minimizes the loss on the retain and the KL divergence (from the model obtained via retraining from scratch) on the forget data. Unfortunately, it is not clear that this heuristic attains the objective of unlearning, which is *being statistically indistinguishable from retraining without the forget data*. There are rigorous definitions of the unlearning objective (Ginart et al. 2019, Guo et al. 2020, Sekhari et al. 2021) which quantify the level of statistical indistinguishability and which are algorithmically attainable, and several works analyzing sequential (adaptive) requests towards this objective (Gupta et al. 2021, Chourasia et al. 2023). It is a missed opportunity to provide theoretical guarantees on the regret of the algorithm, without any guarantees on the central issue at hand, which is unlearning.

* **Insufficient Experimental Validation of Unlearning.** While there is no theoretical unlearning guarantee as mentioned above, there is not enough experimental evidence on unlearning neither. This is typically done through membership inference attacks, which are deferred to the appendix where they are not discussed enough, nor is the specific membership inference attack explained. The latter point is quite important, since it is not trivial to adapt membership inference attacks for machine unlearning (Hayes et al., 2024).

* **Meaningfulness of the Regret Bound.** It is not clear when the regret bound is meaningful, even assuming that the additive term (sum of divergence of retain datasets from forget datasets) is negligible. The term $V_T$ is the sum of distance between successive optimal parameters, and one would expect each summand to be in the order of $\Omega_T(1)$, which would make the final regret bound superlinear and thus vacuous.

* **Clarity of Theoretical and Algorithmic Components.** The paper is quite unclear/hard to read at several occurrences: 

1. For example, Theorem 1 claims that "$\hat{R}_t(w)$ is equivalent to $R_t(w)$" but the meaning of the so-called equivalence is not given. 
2.  Moreover, the KL divergence in line 111 is not well-defined a priori since it is not clear that the problem is a classification task and whether the arguments should be the distribution over classes or the parameter/model distributions. 
3. Line 7 of Algorithm 1 is unclear: "Calculate the shift distribution risk and population risk for forgetting data"; it is not clear what is exactly is the shift distribution risk nor how to calculate it. 
4. $div(D_t, F_t)$ appears in Theorem 2 and Lemma 2 without being defined; "the divergence" is not a precise description. 
5. Lines 357-359 on the comparison with the results of Sekhari et al. (2021) are not justified and unclear. 
6. In the experiments setting, the "JS" metric is not explained and assumed to be known by the reader.

## References
- Ginart et al. (NeurIPS 2019). Making ai forget you: Data deletion in machine learning.
- Guo et al. (ICML 2020). Certified data removal from machine learning models.
- Sekhari et al. (NeurIPS 2021). Remember what you want to forget: Algorithms for machine unlearning.
- Gupta et al. (NeurIPS 2021). Adaptive machine unlearning.
- Chourasia et al. (ICML 2023). Forget unlearning: Towards true data-deletion in machine learning.
- Hayes et al. (2024). Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy.

### Questions
in addition to addressing the concerns above, here are some questions and suggestions:
* Can the term $V_T$ be analyzed for a simple convex problem? This would help clarify the practical relevance of the regret bound.

* Could the authors expand on the membership inference attack used in experiments? Which specific attack was employed, and were any adjustments made to suit the unlearning setting?

* Future work might benefit from exploring advanced membership inference techniques or theoretical unlearning guarantees, even if initially focused on convex problems.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a technique for online data unlearning. Thorough mathematical analysis shows that using this method under mild assumptions provides competitive regret bound guarantees. Some empirical evidence is presented to support theoretical results.

### Strengths
S1. The streaming setup is interesting and a novel way to think about the problem that seems useful. I think this paper would be interesting to both the “unlearning” community as well as the online learning community.

S2. The theoretical derivations have clear assumptions that do not seem excessive. Bounded gradients (Lipschitz) is a mild assumption in general. The theoretical derivations are correct, at least from my checking. (I do have some minor concerns listed later.)

S3. Some evidence for the practical effectiveness versus other methods is demonstrated in the experimental results.

### Weaknesses
W1. While the theoretical grounding is excellent, the experiments are too small scale to be convincing and the statistical analysis is poor or missing. The use of only small benchmarks raises doubts as to whether the relative improvement to baselines is generalizable beyond the small image datasets. Can the authors explain the choice of datasets? Why not use other datasets that are larger (such as ImageNet or variations thereof)? Why focus entirely on image datasets? Since the algorithm operates in an online fashion, it should, in theory, be able to handle very large datasets in chunks. There is a disconnect between the motivation of learning efficiently in the introduction section and the use of small datasets, where efficient methods are unnecessary, as full retraining only takes a couple of minutes. To properly transition from the presented motivation to the experiment section, at least one large dataset should be examined.

 W2. The evaluation metrics are not defined. What do the RA, UA, TA, JS, GAP initializations mean? Does MIA refer to Membership Inference Attack?

W3. Proper statistical testing is needed to confirm that the method indeed improves over the second best baseline in most cases. For example, looking at RA column in MNIST: the 0.5% improvement with std of 0.2% can be possibly due solely to noise. The 0.5% improvement in a test set of 10k images corresponds to a mere 50 images; this is not a large enough margin to provide convincing evidence that the method is actually advancing the SOTA.

 W4. Ablations: The ablation results in Table 2 are slightly concerning. In the text, the authors claim that "these experimental results indicate that the distribution shift loss and the forgetting data gradient contribute significantly to the unlearning process". However, looking at the tables, we often see that removing a component  actually improves performance metrics. Why is this? Does this not contradict the claim that all model components are useful, or at least provide conflicting evidence? There needs to be a more in-depth analysis and discussion of these results.

W5. Writing: The presentation of the paper could be improved significantly. In Table 1, it appears that white space partially covers some numbers.  Some figure captions need to provide much more information so that it is clear which results are being presented. The text in some of the figures is far too small, to the point of being illegible at normal magnification of the text. There are more grammatical errors and typos than would be expected in a submitted paper, especially with the availability of very effective grammar-checking tools.

### Questions
Q1. Eq. (1) and Line 111: The term $\mathcal{R} = 1/|F|\sum_{(x,y)\in F} d_{KL}(f(w), f(w^*))$ is confusing to me. If $w^*$ is defined as optimum $\mathcal{L}(D-L, w)$, wouldn't the KL term go to zero simply by minimizing the loss and then setting $w=\argmin \mathcal{L}(D-L, w)$?

Also, this seems at odds with Kurmanji et al., 2023. In their work they have a KL term on the subset of the data that they wish to forget weighted negatively (which makes sense because you would want to penalize similar outputs between the old and new models on that data subset) and positive KL penalty on the subset of the data they want to remember. This also makes sense because for that subset of data you want to prevent forgetting. Again, it would make things much easier to understand if there were a proper mathematical definition of “unlearning” in this paper.

Q2. What is the definition of forgetting for the authors? Please provide it clearly in the paper. Some papers define it as a statistical indistinguishability in model performance between unseen and "unlearned" data. In other works (Kurmanji et al., 2023 – see paper for full reference), "unlearning" relates to performing much worse on the unlearned data compared to the past while maintaining performance on the remaining data. This paper does not make clear the exact mathematical definition of unlearning that is being adopted. Kurmanji et al., 2023 has a good discussion on this issue.

Q3. I followed the proofs in the appendix carefully and they seem correct. I have a question regarding Theorem 2. Where does the inequality in Line 898 come from?

Q4. Eq. (5): Why is $w_t$ on the LHS? We are minimizing over $w_t$ on the RHS to obtain minimal risk. Also, $w_t^*$ should be the argmin not the min value of the loss.

Q5. Eq. (7): Second denominator: I assume $Q^{0}(y)$ should be $Q_{0}(y)$. Also, $Q(y|x)$ is defined but $Q(y)$ and $Q(x)$ are not. Can you please define them?

Q6. Eq. (12): Please define the norm in the main text to improve readability. It breaks the flow if the reader needs to go to appendix to determine which norm is used here.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper focuses on machine unlearning, in particular in the settings where requests for data removal arise in a streaming manner. To address the new challenges of performance maintenance, efficiency and data access brought about by this new streaming setting, this paper introduced a new online unlearning paradigm. Specifically, a new algorithm named SAFE was proposed. The performance of SAFE was analyzed with a regret upper bound guarantee, and numerically evaluated and compared to baseline methods.

### Strengths
- This paper considers a practical setting for machine unlearning, i.e., the streaming setting where new data may continue arrive or be generated during the unlearning process. This new settings bring many new challenges compared to the widely-studied static/batched settings. This paper addressed these new challenges via proposing a new SAFE algorithm. 
- The performance of SAFE algorithm was (a) analyzed with a regret upper bound guarantee; and (b) numerically evaluated using several datasets and compared to baseline methods.

### Weaknesses
- The performance analysis of SAFE is relatively weak or may be very straightforward given the existing literature. For instance, the bound on the error of the t-th rounds is simply bounded by $O(\sqrt{T})$, which is straightforward and should not be a tight bound. 
- As noted by the authors, the accumulated regret is also not tight and compared to existing results. 
- Despite that MNIST, FASHION, and CIFAR-10 have been widely used, it may be more interesting to consider more complex datasets such as (Tiny)-ImageNet. In addition, all these datasets focus on the image classification task, it may be also interesting to consider datasets for other tasks, such as NLP tasks.

### Questions
- Can you elaborate the tightness of your regret bounds? Also what are the coefficients associated in the bounds with respect to $T$? Should the bound be written as $\tilde{\mathcal{O}}(\sqrt{T})$? 
- How do you know the value of $V_T$ in advance? which depends on all the $T$ rounds and the data arrive in a streaming manner? Likewise for the value of $\Delta_T$. In addition, are their values constant and independent of $T$?
- Can your algorithm operate without knowing the values of $V_T$ and $\Delta_T$ in advance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors consider a online machine unlearning problem, where requests for data remove arise in a streaming manner rather than in a single batch. This causes new challenges, and this paper solves this problem by viewing unlearning as a distribution shift problem. They estimate the altered distribution and propose a online unlearning method without having access to the original training data. Theoretical analysis is also conducted without the strong restriction of convex loss function. Experiments are conducted using 4 different datasets, confirming the advantage of the proposed method.

### Strengths
1. This paper is well written and easy to follow. The presentation and the organization of the paper is good.

2. This paper tackles an important problem, which is to conduct unlearning in a streaming manner. This problem has not been well studied in the unlearning literature.

3. The idea of viewing the unlearning as distribution shift problem has not been well explored before. The approach is relatively novel compared to existing works.

4. The paper also provides theoretical insights into the proposed online unlearning methodology.

### Weaknesses
1. Technical contributions: In Line 65, the authors mention that only two studies (Zhao et al., 2024; Li et al., 2021) have directly tackled the problem of stream data forgetting. However, it is not clear what are the new technical novelties of this work compared to previous works. It would be great if the authors can better illustrate their technical contributions and advantages in the revised paper.

2. Baselines: During experiments, it is not clear why the above two works (Zhao et al., 2024; Li et al., 2021) are not considered as baselines. It would be good to make a comparison with these closely related works.

3. Datasets: The datasets considered in this paper is relatively small. I recommend the authors to consider ImageNet related datasets, at least miniImageNet or tinyImageNet.

### Questions
My questions can be found in the weakness above.

### Soundness
3

### Presentation
3

### Contribution
2
