# An Empirical Study and Theoretical Explanation on Task-Level Model-Merging Collapse

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Model merging unifies independently fine-tuned LLMs from the same base, enabling reuse and integration of parallel development efforts without retraining.
However, in practice we observe that merging does not always succeed: certain combinations of task-specialist models suffer from catastrophic performance degradation after merging. We refer to this failure mode as  merging collapse. Intuitively, collapse arises when the learned representations or parameter adjustments for different tasks are fundamentally incompatible, so that merging forces destructive interference rather than synergy.
In this paper, we identify and characterize the phenomenon of task-level merging collapse, where certain task combinations consistently trigger huge performance degradation across all merging methods. 
Through extensive experiments and statistical analysis, we demonstrate that representational incompatibility between tasks is strongly correlated with merging collapse, while parameter-space conflict metrics show minimal correlation, challenging conventional wisdom in model merging literature.
We provide a theoretical explanation on this phenomenon through rate-distortion theory with a dimension-dependent bound, establishing fundamental limits on task mergeability regardless of methodology.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Model merging is a popular method to efficiently build a multi-task model. However, it usually leads to a performance degradation. In this paper, they propose three key research questions, including the consistency of performance degradation across merging techniques, the potential reason for performance degradation, and the factors affecting the merging performance. They conduct extensive experiments to answer those questions. Besides, a theoretical framework is proposed to analyze the relationship between representational incompatibility and merging collapse.

### Strengths
- The experiments in this paper are comprehensive, including diverse models and datasets.
- Besides empirical results, this paper proposed a theoretical framework to analyze the factors for merging collapse, revealing more insights to understand it.
- The research questions proposed in this paper are valuable and interesting, and the author gives a clear answer supported by extensive experiments.

### Weaknesses
- This paper considers the decoder-only models and encoder-decoder models, so it would be better to consider encoder-only models. An interesting direction would be investigating the effect of different model architectures.
- Some metrics in Sec.2 are a bit confusing to me. Please see the questions.
- The RQ1 is valuable but not that interesting. Many papers have shown that merging techniques will lead to performance degradation.

### Questions
- The definition of some metrics in Sec.2 is unclear.
    - Regarding the parameter sign change ratio in Sec.2, under what situation would this be an issue? For example, consider a two-dimensional space where one weight vector is $[1, 0]$ and the other is $[-1, 1]$, then merging them leads to a vector in-between. Since the ratio is 50% in this case, the author would think it is harmful but in my opinion, I dont know why this is an issue. Could you please give more explanation?
    - For the parameter magnitude change ratio, if the task vectors are $[1, 1]$ and $[2, 2]$, this ratio could be large, but I think there is not an issue as well.
- There are some new merging techniques that the author can discuss [1-4] (maybe after the rebuttal phase).
- [5] studies the generalization bound of model merging. The author may discuss it in this paper.

[1] Stoica, George, et al. "Model merging with svd to tie the knots." *arXiv preprint arXiv:2410.19735* (2024).

[2] Zhang, Haobo, and Jiayu Zhou. "Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging." *arXiv preprint arXiv:2505.22934* (2025).

[3] Huang, Chenyu, et al. "Emr-merging: Tuning-free high-performance model merging." *Advances in Neural Information Processing Systems* 37 (2024): 122741-122769.

[4] Tang, Anke, et al. "Merging models on the fly without retraining: A sequential approach to scalable continual model merging." *arXiv preprint arXiv:2501.09522* (2025).

[5] Li, Hongkang, et al. "When is task vector provably effective for model editing? a generalization analysis of nonlinear transformers." *arXiv preprint arXiv:2504.10957* (2025).

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
3

### Summary
The paper investigates task-level collapse in model merging and shows through broad experiments across methods and backbones that the phenomenon is widespread and largely driven by the specific task rather than the merging rule. It then offers a rate–distortion perspective with a dimension-dependent lower bound on representation distortion, explaining why even convex merges can fail under realistic separations of task representations. Finally, it introduces HiddenSim and its multi-task extension MDS to quantify representation similarity, demonstrating strong correlation with merging loss and practical value for screening incompatible task combinations.

### Strengths
1. Breadth of evidence for collapse. Systematic experiments across multiple merging rules and backbones show the collapse phenomenon is widespread and primarily task-driven rather than method-driven.

2. Representation-space explanation that matches practice. A rate–distortion analysis yields a dimension-dependent lower bound on representation distortion for convex merges, providing a principled reason merges can fail and aligning with the empirical signal.

3. Actionable pre-screening tools. HiddenSim and its multi-task extension MDS correlate strongly with merging loss and enable swapping out incompatible tasks before merging.

### Weaknesses
1. Theory relies on strong assumptions. The lower-bound analysis assumes linear-mode connectivity and last-layer linearity.

2. Narrow design of HiddenSim. It uses few samples and only the last layer with L2 distance. Including ablations over sample size, layer choice, pooling, and alternative metrics would help.

3. Results are centered on GLUE and Lots-of-LoRAs. The authors may include non-classification or code/generation tasks and additional model backbones to test generality.

### Questions
See weakness. I'm willing to increase the rating once my concerns are well addressed.

### Soundness
2

### Presentation
2

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
The paper investigates the failure of model merging, which occurs due to catastrophic performance degradation after merging. The authors focus their experiments on GLUE tasks and Lots-of-LoRAs checkpoints. 
They define merging loss as the relative difference between the performance of the merged model and that of the fine-tuned model on the same task.
The study explores whether merging collapse is more correlated with the tasks or with the merging techniques, finding a statistically significant correlation with the tasks.
They further assess the relationship between merging loss and four metrics that capture parameter update conflicts (Parameter Magnitude Change Ratio, Parameter Sign Change Ratio, Conflicting Parameter Magnitude Change Ratio, and Average Cosine Similarity) and find no statistically significant correlation with any of them.
Instead, they discover that merging loss is correlated with a new metric they introduce, called Hidden State Distance Similarity, which measures the distance between hidden states across tasks.
Finally, they prove a theorem showing that, under the assumption of linear mode connectivity, the minimum achievable hidden-state distortion is bounded.

### Strengths
- The paper provides a well-structured empirical investigation of model merging failures. By systematically testing correlations between merging loss, task identity, and several parameter-space metrics, it offers strong evidence that task characteristics primarily drive performance degradation.
- The paper introduced a novel metric which  connects merging performance to the geometry of model representations and correlates with performance degradation in merging. 
- The provide a theoretical result that adds a principled foundation to its empirical findings.

### Weaknesses
- The paper does not engage with existing literature tackling the same issue from complementary angles. Works such as for intance Task Singular Vectors [1], and Iso-C [2] directly analyze model merging collapse through task subspace geometry and rank alignment, yet are not cited or discussed. This omission limits the contextualization of the proposed study and leaves unclear how it connects to or extends prior understanding of merging failure.
- Evaluating the hidden-state diameter $\Delta$ or the distortion $\delta_{max}$ requires access to hidden representations across all fine-tuned models and the entire input distribution, which is generally infeasible in realistic multi-task or large-scale settings.
Thus, although the bound provides theoretical insight into when merging is doomed to fail, it is difficult to operationalize as a predictive or diagnostic tool.
- The statement of the theorem is not clear; some symbols are not well explained beforehand. 
The symbol $D$ (and $D^*$) is introduced without a clear definition. 
It appears to correspond to some notion of ``distortion'' or hidden-state mismatch, but no explicit mathematical formulation is given. 
Likewise, the term distortion is used interchangeably with $\delta_{\max}(\hat{\theta})$ in the main theorem, but this identification is never formally stated. 
In the proof, $\delta_i(\hat{\theta})$ is not defined; I assumed it was meant to be
$
\delta_i(\hat{\theta}) = \mathbb{E}_X \big\| H_i(X) - h(X; \hat{\theta}) \big\|_2^2 .
$
Moreover, the proof contains some small issues in Step~1: the paper's wording makes it sound as if the coefficients $\alpha$ are fixed \emph{a priori} and then one finds the center of the convex hull. 
Under this interpretation, it is not true that $h(x; \bar{\theta})$ coincides with $c(x)$. 
What is true is that, by Jung's Theorem, there exists a set of coefficients (depending on $x$) such that the convex combination (that is, the center) minimizes the worst-case distance to all $H_i(x)$. 
If this same set of coefficients defines a parameter merge, then this merge achieves the minimum possible worst-case hidden-state distortion. 
In any case, the statements appear correct to me. 
- table 7-10 are not clear, what are the columns in this tables? Do they correspond to the task of one group? Moreover are never mentioned in the paper. 

Typos: 
- Line 317 starts with a comma.
- Line 063 missing $\mathbb{R}^d$

[1]Gargiulo, Antonio Andrea, et al. "Task singular vectors: Reducing task interference in model merging." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2] Marczak, Daniel, et al. "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces." Forty-second International Conference on Machine Learning.

### Questions
- Table 1, why merging losses have posivite values= even if merging loss was define as negative? 
- What are the  exact 25 groups of 8 checkpoint IDs used in the paper?

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
The paper studies merging collapse where merging task specialist models causes performance degradation. They find that some tasks are incompatible to merging where they fail irrespective of the merging method. They also find that representational incompatibility, especially measured through hidden states similarity correlates well with merging collapse.

### Strengths
Strenghts:
- Good study across merging methods and domains 
- Concretely find that task incompatibility causes merging collapse 
- Propose a hidden similarity metric for guiding merging

### Weaknesses
- There is no significant understanding on merging collapse as proposed in abstract. It is known that task incompatilibity causes merging failure, as it cannot maintain linear mode connectivity. 
- Merging is interesting in generalization perspective, the paper doesn’t study anything related to that. If the goal is to only to retain performance of existing models, then even past works have seen that it is not possible to always retain complete performance i.e, merging collapse will happen.
- The paper finds that some tasks are not possible to merge, but is it stil true at scale? Studying this phenomenon would more your finding more concrete. 
- Using hidden state similarity is limited because of it assumes access to datasets that created the experts in the first place. If you have access the experts’ data, we can train a multitask model that performs well on all tasks instead of merging them.

### Questions
How does MDS metric guide merging? Do you rank the experts based on the merged model and current expert?

### Soundness
3

### Presentation
3

### Contribution
2
