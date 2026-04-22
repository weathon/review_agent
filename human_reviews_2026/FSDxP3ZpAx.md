# LoRA Provably Reduces Forgetting and Enables Adapter Merging in Multiclass Linear Classification

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Low–Rank Adaptation (LoRA) has become the de-facto parameter-efficient fine-tuning algorithm.  
Besides training-efficiency, practitioners observe two striking benefits: *(i)* remarkable resistance to *catastrophic forgetting*, and *(ii)* independently trained adapters can be *merged* into a single model that performs well on multiple tasks. Despite their practical importance, these phenomena have lacked rigorous theoretical explanation. In this work, we provide the first theoretical justification for the two aforementioned phenomena by analyzing the structure of LoRA solutions in multiclass linear classification problems for orthogonal tasks.  
Our analysis shows that, under suitable weight regularization, the optimal LoRA adapter aligns exactly with the \emph{max-margin} (hard-margin SVM) solution for the fine-tuning data. This alignment lets us track in closed form how the normalized margins on the pre-training data, fine-tuning data and their union vary with the regularization parameter.
For *(i)*, we observe a trade-off: decreasing the regularization parameter enlarges the fine-tuning margin while proportionally shrinking the pre-training margin, never collapsing it to zero.
Concerning *(ii)*, we view the merged weights through the same margin lens, we prove why merging succeeds and derive optimal mixing coefficients that maximize the margin on the union of all tasks.
Finally, we numerically validate our theory across multiple deep learning architectures and task configurations. The empirical results closely match our theoretical predictions. 
Taken together, our results give the first principled explanation for LoRA’s resistance to forgetting and its surprising merging ability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a theoretical analysis of the Low-Rank Adaptation (LoRA) algorithm in the context of multiclass linear classification. The authors aim to offer a principled explanation for two important empirical findings: (i) LoRA's resistance to catastrophic forgetting, and (ii) the ability to merge independently trained adapters into a single model that performs well on multiple tasks. The paper builds on the concept of maximizing the margin in linear classification and derives closed-form expressions for optimal LoRA adapters, showing that LoRA’s effectiveness in forgetting reduction and adapter merging can be understood through the lens of margin theory.

### Strengths
1. The paper has a good the high-level motivation, i.e., to explain LoRA’s empirical behavior from first principles, which is timely and relevant to the community.
2. Given the strong simplifying assumptions, the theoretical derivations are internally consistent. The link between the regularized LoRA objective and hard-margin SVM theory is clearly explained.

### Weaknesses
1. The Assumption 2.3 is a strong and unrealistic assumption in practical scenarios. The author provide the some numerical evidence in Appendix I.1 by using the pearson correlation coefficient, it may not be sufficient to support such a restrictive orthogonal assumption. Same for the Assumption 3.1. 
2. This paper only consider the multiclass linear classification task, but the usage of LoRA are often focus on the large scale model such as Large Language Models on the generation tasks.
3. The experiments are conducted on LoRA with only the one layer of the model (final linear), can the result expand to multiple layer as many exsiting usage of LoRA?

### Questions
1. This paper are based on the ''orthogonal tasks'', which lacks of the detailed explaination. Are the orthogonal tasks refers to the task that satisfied the Assumptions 2.3? If that, the reference that ''across all original tasks'' in the introduction (line 58) seems problematic as many of them do not have this assumption.
2. While the paper provides a deep theoretical analysis, it could benefit from a more detailed discussion of the practical implications of LoRA's findings for real-world applications, such as incremental learning or continual learning. How do the theoretical insights apply to large-scale models like LLMs (Large Language Models)? 
3. How performance changes as correlation of task increases?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a theoretical study on how LoRA exhibits resistance to catastrophic forgetting and enables the effective merging of separately fine-tuned models. Starting from a set of simplifying assumptions and leveraging the theory of soft and hard margins in SVMs, they develop formal theorems and proofs showing that, when models are fine-tuned with LoRA under norm regularization on the LoRA matrices, three distinct training regimes emerge, each leading to different model behaviors. The analysis further demonstrates that LoRA imposes specific boundaries between pretraining and finetuning phases depending on the strength of the regularization term, and that optimal scaling factors exist which *i)* allow the resulting model to find the most effective boundary between pretraining and finetuning tasks, thus mitigating catastrophic forgetting, or *ii)* yield optimal merging performance across multiple fine-tuned models. Theoretical insights are supported by experiments on various vision architectures using the CIFAR-100 dataset.

### Strengths
- Mathematical completeness: assumptions, lemmas and theorems are clearly stated, and proofs are provided
- Overall, the paper is well written, the notation is precise and not too complex at the same time

### Weaknesses
1. Experimental section: in my opinion, this is the main weakness of the paper. The authors make several simplifying assumptions to obtain more tractable mathematics (see next weakness), which, while potentially too restrictive and simplistic, could be acceptable if the experimental section demonstrated that the theoretical insights generalize to more realistic and complex settings. However, this is not the case. To validate their findings, the authors conduct linear probing experiments: *i.e.*, they fine-tune only the final classification layer with LoRA while keeping the rest of the model frozen. This setup is both unrealistic, since LoRA is typically applied to fine-tune the backbone (particularly the Attention modules) rather than the classifier head, and overly simplistic. Consequently, the empirical results have limited practical relevance. Moreover, given that the stated goal of the paper is to explain how and why LoRA mitigates catastrophic forgetting and facilitates model merging, the comparison baseline should include Full Fine-Tuning (FFT). The authors briefly mention FFT in Section 3.2, noting that comparing LoRA to FFT is not their goal: this, however, appears inconsistent with the paper’s central motivation.


2. Strong assumptions, which may rarely hold in practice:
   - Assumption 2.1, *i.e.* "The input data dimension is larger than the total number of classes", is reasonable.
   - Assumption 2.2, while somewhat restrictive, is still acceptable if it helps maintain mathematical tractability: LoRA can and has been used on datasets with more classes than LoRA’s rank.
   - Assumptions 2.3 and 2.4 are overly restrictive, and lack justification that they are met in realistic scenarios. Even Section I.1 (page 31 of the Appendix), which attempts to validate part of Assumption 2.3 by showing that features are orthogonal, does so in a toy setting (see previous weakness) and thus fails to demonstrate that these assumptions hold in real-world cases.

Minor weaknesses (did not affect my overall evaluation):

3. Figures 2 and 5 reference Theorem 4, which does not appear anywhere in the paper; this is likely a typo, and the authors probably meant Theorem 3.

4. The distinction between $M_K^{(\overline{K})}$ and $M_{\overline{K}}$ is unclear. The authors define $M_K^{(\overline{K})}$ as the first $\overline{K}$ columns of the $K$-ETF matrix $M_K$; however, it is not evident how this differs from $M_{\overline{K}}$.

### Questions
Does the theory provided by the authors also translate into more practical settings and use cases, for instance those in the Model Merging Literature? Especially considering the finetuning operations (finetuned backbone, in many cases with LoRA) and datasets.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper provides a theoretical analysis of LoRA under a multiclass linear classification setting to explain two empirical observations: its resistance to catastrophic forgetting and the success of adapter merging. The authors show that under certain regularization regimes, the LoRA solution aligns with the hard-margin SVM direction, and derive margin-based expressions to justify these behaviors. Limited experiments on linear-probe settings support the analysis.

### Strengths
1. Addresses two important empirical properties of LoRA with an explicit mathematical formulation.
2. Clear presentation and rigorous derivations under simplified assumptions.
3. Offers interpretable margin-based insights and closed-form solutions for merging coefficients.

### Weaknesses
1. The theoretical foundation is too shallow, restricted to a single linear layer without deeper or nonlinear analysis, which weakens its relevance to real LoRA behavior in large models.
2. The experimental validation is limited and cannot convincingly demonstrate how LoRA reduces forgetting or benefits merging in practice.
3. The regularization analysis is intuitive rather than novel; showing that a stronger regularizer constrains updates is already well understood.
4. Strong orthogonality assumptions further reduce practical applicability.

### Questions
Does the proposed analysis or conclusion still hold when LoRA is applied across all layers or on larger-scale models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a theoretical study of LoRA in a simplified multi-class linear classification setting. Under strong orthogonality and separability assumptions, it shows that, with appropriate regularization, LoRA can remain resistant to catastrophic forgetting, and that independently trained LoRA adapters can be merged into a single model that maintains good performance across multiple tasks.

### Strengths
- The analysis is clear and logically structured.
- The margin-based analysis yields closed-form expressions for both the optimal regularization parameter and the adapter-merging coefficients.
- The paper further distinguishes different penalty regimes to reveal the trade-off between preserving the pre-trained task and fitting the new task.

### Weaknesses
- The analysis relies on a number of very strong and idealized assumptions. These assumptions are unlikely to hold under realistic data distributions, and the paper currently does not provide large-scale empirical evidence to indicate how robust the theory is once these assumptions are relaxed.
- The paper only treats a simple multi-class linear classification head with frozen features, whereas in practice LoRA is mostly used inside attention blocks on non-orthogonal, high-dimensional representations; it would be helpful to at least discuss how the results might extend to that setting.
- The experimental section is relatively light and limited to small vision-style settings.

### Questions
Could the authors analyze how the current conclusions change when the orthogonality and related assumptions are relaxed to a more realistic, approximately orthogonal setting?

### Soundness
3

### Presentation
3

### Contribution
2
