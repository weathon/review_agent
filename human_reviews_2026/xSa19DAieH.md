# Stable-LoRA: Stabilizing Feature Learning of Low-Rank Adaptation

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 4

## Abstract
Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient method for fine-tuning Large Langauge Models. It updates the weight matrix as $W=W_0+sBA$, where $W_0$ is the original frozen weight, $s$ is a scaling factor and $A$,$B$ are trainable low-rank matrices. Despite its robust empirical effectiveness, the theoretical foundations of LoRA remain insufficiently understood, particularly with respect to feature learning stability. In this paper, we first establish that, LoRA can, in principle, naturally achieve and sustain stable feature learning (i.e., be self-stabilized) under appropriate hyper-parameters and initializations of $A$ and $B$. However, we also uncover a fundamental limitation that the necessary non-zero initialization of $A$ compromises self-stability, leading to suboptimal performances. To address this challenge, we propose Stable-LoRA, a weight-shrinkage optimization strategy that dynamically enhances stability of LoRA feature learning. By progressively shrinking $A$ during the earliest training steps, Stable-LoRA is both theoretically and empirically validated to effectively eliminate instability of LoRA feature learning while preserving the benefits of the non-zero start. Experiments show that Stable-LoRA consistently outperforms other baselines across diverse models and tasks, with no additional memory usage and only negligible computation overheads. The code is available at https://github.com/Yize-Wu/Stable-LoRA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a theoretical analysis of LoRA’s fine-tuning dynamics and argues that LoRA can achieve self-stabilization under proper initialization and hyperparameter choices. The authors further propose *Stable-LoRA*, a simple weight-shrinkage strategy applied to the A matrix during early training to mitigate instability. Experimental results show consistent improvements across several reasoning and QA benchmarks, demonstrating the method’s effectiveness and efficiency.

### Strengths
The paper offers a clear theoretical perspective on the stability of LoRA fine-tuning, supported by a consistent use of the γ-function framework to characterize scaling behavior. The derivations are mathematically well-structured and connect intuitively to optimization dynamics. The proposed Stable-LoRA method is simple yet effective, introducing negligible computational overhead while improving training stability across tasks. The work also contributes conceptually by bridging theoretical analysis and practical optimization design within the PEFT paradigm.

### Weaknesses
1. About Assumption 1: 

   If the activation $Z$ is normalized under a fan-in scaling scheme  (i.e., each input element scaled by $1/\sqrt{n}$, which is common in linear layers or attention projections),  then each component of $Z$ becomes $\Theta(n^{-1/2})$.  

   Consequently, the product $g_A^t Z$ involves a summation over $n$ such terms,  resulting in an overall magnitude of $\Theta(\sqrt{n})$.  This implies that $\gamma[g_A^t Z] = \tfrac{1}{2}$ rather than $1$.  Could the authors clarify whether Assumption 1 still holds under common fan-in scaling conventions?  In particular, does the assumed normalization (e.g., layer normalization, residual connections, or other schemes)  preserve the activations at the $\Theta(1)$ scale?

2. In Section 3.2, the authors assume that $\gamma[A_0 Z] \le \gamma[\eta] + 1 \Rightarrow \gamma[A_t Z] = \gamma[\eta] + 1.$ However, “≤” only provides an upper bound and does not guarantee equality during training. This step effectively assumes convergence of the recursive relation $\gamma[A_t Z] = \max(\gamma[A_{t-1}Z], \gamma[\eta] + 1),$ without proving that the sequence will reach equality. Could the authors provide a justification (analytical or empirical) showing that the recursion indeed converges to $\gamma[A_t Z] = \gamma[\eta] + 1$ rather than remaining strictly below that bound?

3. The paper claims that when only one condition in Eq. (5) holds, $\gamma[A_t Z] = \gamma[B_t] + 1 \Rightarrow \gamma[\delta_1] = \gamma[\delta_2],$ which is said to be “undesirable.” However, δ₁ and δ₂ represent the contributions of matrices A and B to the output update ΔYₜ. Having the same γ-scale simply implies that both pathways contribute symmetrically, which is not necessarily detrimental to learning stability. Could the authors clarify why $\gamma[\delta_1] = \gamma[\delta_2]$ must be considered suboptimal? Is there theoretical or empirical evidence that symmetric scaling between A and B leads to instability?

### Cons

1. The baselines considered in Table 1 are too limited. Recent PEFT methods such as DoRA[1] and AdaLoRA  also address optimization stability and efficiency. Including these methods would provide a fairer and more comprehensive comparison.

2.  The backbones used are relatively small (≤ 3B parameters). To substantiate the claim of general stability, experiments on larger or more recent architectures (e.g., Llama-3.2-8B, Mistral-7B, or Mixtral-8×7B) are necessary. This would also help assess scalability and compatibility with modern large-model training dynamics.

   [1] Liu, X., Li, Y., Zhou, T., Wang, K., & Qiu, X. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv preprint arXiv:2402.09353.

   [2] Zhang, R., Xu, H., Cui, Y., Liu, T., & Zhang, Y. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*. arXiv preprint arXiv:2303.10512.

## Typos

1. Lines 101: There is a dot(.) at the begining of 101 lines. Maybe its should be placed on the end of Eq. (1). 

2. Lines 202: The same issue as in line 101 — punctuation should not appear at the beginning of a new line.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Stable-LoRA, a novel optimization strategy that enhances the training stability of LoRA. The authors first theoretically demonstrate that LoRA possesses self-stabilizing properties under appropriate hyperparameters and initialization, but commonly used non-zero initialization often undermines this stability. To address this, Stable-LoRA dynamically shrinks matrix A during the early training phase. Experimental results demonstrate the effectiveness of Stable-LoRA and with negligible additional computational or memory overhead.

### Strengths
* This paper is solid in its theoretical contribution. The motivation and concepts are well illustrated, making the work easy to follow. The algorithm design is simple yet elegant, and the stability stopping criterion is theoretically justified.

* Empirical results demonstrate both the effectiveness and stability of the proposed method. Moreover, it is computationally efficient, introducing only a minor additional runtime overhead. The approach is also compatible with existing LoRA setups without requiring any architectural modifications.

### Weaknesses
* The experimental settings and details are somewhat limited and unclear. First, how are the experiments on the QA datasets conducted? Is the model fine-tuned on a mixed training dataset and then evaluated on several benchmarks, or is it fine-tuned on one QA dataset and tested accordingly? If it is the latter case, I would suggest conducting additional experiments on general language understanding and dialogue datasets such as WizardLM to better assess the model’s generalization ability. Moreover, the experimental settings for the reasoning tasks are also limited.

* How many steps does Stable-LoRA require to reach its stable mode? I think the shrinkage of the LoRA matrix $A$ shares a similar intuition with the weight decay mechanism in AdamW. How about comparing Stable-LoRA with AdamW using a relatively large weight decay parameter when the model shows instability?

* Finally, Stable-LoRA should also be applicable to other LoRA variants, such as AdaLoRA. Have the authors tried this? If the method generalizes well to other setups, it would be quite interesting.

### Questions
See details in the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates whether LoRA can achieve and sustain stable feature learning. By introducing a $\gamma$-function as the main analytical tool, the authors prove that stable feature learning is attainable under specific hyperparameter settings and when $A = B = 0$. However, in practices, $A$ and $B$ cannot be set to 0 at the same time. To mitigate this issue, the paper proposes a progressive shrinkage strategy that gradually reduces $A$ during the early training stages. Extensive experiments suggest that the proposed Stable-LoRA method consistently outperforms baseline models.

### Strengths
* The paper introduces the $\gamma$-function as a novel analytical construct for understanding LoRA’s stability behavior.
* Theoretical analysis yields an interpretable condition—$A=B=0$—under which stable feature learning can be achieved.
* The proposed progressive-shrink mechanism is a practical solution to approximate the ideal condition $A=B=0$ and address the constratins that in practices, we cannot set both $A$ and $B$ to 0.

### Weaknesses
* Definition 1 lacks rigorous justification. While it aligns with prior empirical observations, it represents only one possible stability condition. The framework built upon it may have limited generalizability, which can only be supported by broader empirical studies.
* The definition of the $\gamma$-function appears mathematically infeasible. Although it seems inspired by logarithmic properties (e.g., 
$\log(x) + \log(y) = \log(x \times y)$ and $\log(x + y)$ is dominated by $\max(\log(x), \log(y)$). The equation $\gamma[v + v'] = \max(\gamma[v], \gamma[v'])$ may not always hold, even with a hidden constant in the $\Theta(\cdot)$. Thus, the $\gamma$-function may be valid for qualitative reasoning, but not for formal proof.
* Empirical results (e.g., Table 1) show only moderate improvements without reporting standard deviations or confidence intervals. It is therefore difficult to determine whether the observed gains are statistically significant or within expected variance.
* The writing quality could be improved. Several sections are hard to parse and would benefit from clearer exposition and more precise mathematical notation.

### Questions
1. Line 47: Should the asymptotic complexity be $O(n)$ instead of $O(n^0)$,  if this is "with respect to model width $n$" as stated in line 46.
2. Line 131: The definition of $r[v]$ by $v=\Theta(n^{\gamma[v]})$ is kind of informal. Written in this way, it implicitly invites a proof of existence.
3. Line 132: $\gamma[\overrightarrow{v}]:=\max(v_i, 0 < i < k)$. Should it be $0\le i$?

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
LoRA is the method that is the most commonly use to finetune LLMs. However it does have some training challenges. This study looks at those problems and proposes a clear analysis, especially around the initialization of LoRA--and propose a novel solution. Empirical results are provided to support the claim.

** Strength **
- very important problem in the world of LLM and post-training especially
- the experiments are quite sound to validate the study
- the interpretation of LoRA is interesting albeit hard to grasp for readers not familiar with the field

** Weakness **
- enormous typo (e.g. one right in the title, adaption, or in the abstract, Langauge) which makes the entire script feel quite odd
- could the author discuss a possible link with https://arxiv.org/pdf/2410.09692? who also checked at initialization and training dynamics?
- lacking some geometric/visual intuition that would help reader better grasp the results
- needs better scoping of limitation and future work

### Strengths
Please see summary

### Weaknesses
Please see summary

### Questions
Please see summary

### Soundness
3

### Presentation
2

### Contribution
3
