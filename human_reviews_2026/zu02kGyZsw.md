# Quality-constrained Entropy Maximization Policy Optimization for LLM Diversity

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Recent research indicates that while alignment methods significantly improve the quality of large language model(LLM) outputs, they simultaneously reduce the diversity of the models' output. Although some methods have been proposed to enhance LLM output diversity, they often come at the cost of reduced performance. In this work, we first theoretically demonstrate that the alignment task can be decomposed into two distributions: quality and diversity. To enhance the diversity of LLM outputs while ensuring quality, we propose the Quality-constrained Entropy Maximization Policy Optimization (QEMPO). QEMPO aims to maximize the output entropy of the policy while ensuring output quality. By adding different constraints to QEMPO, we obtain different policies. To optimize policies, we propose both online and offline training methods. Experiments validate that QEMPO achieves performance comparable to or even better than RLHF while improving output diversity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method that tackles the problem of reduced output diversity in LLM alignment. The proposed approach, QEMPO, directly maximizes the entropy of the output distribution while keeping alignment quality as a constraint.

### Strengths
- General writing clarity is good.
- Experimental setups are clearly described, hyperparameters descriptions seem complete, experiments should be reproducible.
- Enough models are used in experiments.

### Weaknesses
* **W1) Missing detailed discussion on the limitations of RLHF**. This paper would benefit from a more thorough discussion of why RLHF is not enough to ensure output diversity. I mean it is rather clear after the preliminaries that RLHF never "explicitly" optimizes the entropy of the output distribution. However, a deeper discussion is missing. For example, I do not fully agree with the claim "[RLHF] optimization does not inherently encourage diversity" (line 130) so far. As your optimization perspective in P1 (line 133) suggests, RLHF minimizes KL divergence between the current model and a reference model. Shouldn't this already (implicitly) encourage a diversity close to the one of the reference model? What is the actual reason RLHF has a negative impact on output diversity / entropy? Could the problem also be in the trained reward, e.g. that it does not correctly assign high rewards to alternative responses as well? 

* **W2) Unclear diversity definition**. In Section 3.1 you state that the goal of output diversity should be that the probability of all 'correct' sequences must be the same. However, shouldn't the goal of diversity still support responses with varying probabilities? While models might be able to provide alternative responses, why can't some responses be more likely than others? In case your diversity definition is commonly used, could you point to other works that have used this definition before?

* **W3) Weak discussion of experimental results**. In general, the results across models (Table 2) do not consistently suggest that the proposed approach is better than the related work SPL, which also seems to provide better output diversity than RLHF. While this might be okay, a detailed and more careful analysis is missing. For example, what could be reasons for the instability of SPL? How do you ensure the comparison to SPL is fair? Do you tune hyperparameters for the baseline as well? In addition, are your results in Table 2 statistically significant, could you e.g. provide details on the standard deviation (similar for Figure 3)? Overall, the paper would benefit from a more careful discussion of differences to the SPL baseline and of the experimental results in general. This is rather critical for assessing the contribution of this paper.

**Minor weaknesses**

* **Low-quality section 4.2**. While the overall paper is in a good state regarding writing-quality, section 4.2 is repetitive and confusing, sentences are repeated (lines 359-364), making the results difficult to follow. 
* **Unclear paragraph in lines 210-212**. I think $\pi_{ref}$ and $\pi_{QEMPO-KL}$ are mixed-up in the discussion, making the argument difficult to understand.
* **Explanation on $\lambda$**. I think it would be helpful to explicitly say that $\lambda$ is the Lagrange multiplier coming from the constraint optimization problem (around Proposition 1 in lines 127-128). Currently this only becomes clear from reading the Appendix. Generally, pointers to proofs in the appendix would also be helpful.
* **Additional references**. Removing the dependency on a reference model (when going from QEMPO-KL to QEMPO in section 3.3) is not new and has been proposed for example by SimPO. I think discussing this parallel would be helpful, including potential negative side-effects (or at least referencing this parallel).
* **Clear preliminaries**. Currently the paper assumes that $\pi$, $x$ and $y$ are clear from context. I think a short introduction in the preliminaries would be helpful to clarify statements such as "Since $\pi$ is an LLM, it can be regarded as a distribution" (L686). 
* **Formatting of Table 1**. Table 1 is significantly larger than the page constraints.
* **Typo.** In line 163, do you mean a small subset of $Y_w$ (not $y_w$)?

In general, clearly stating potential limitations and broader impact of the proposed entropy optimization approach would also be helpful.

### Questions
- Q1) Could you clarify why entropy maximization under quality constraint is the better perspective? Why is it not sufficient to optimize the RLHF objective under an additional entropy constraint (i.e. the other way around), or simply to maximize both at the same time (RLHF and entropy)? Is QEMPO-KL not just RLHF + entropy maximization after applying the Lagrange multiplier?
- Q2) What do you mean with "policy gradient methods" (e.g. line 006 and in Corollary 3.1 in line 173)? Do you refer to a set of methods? You also write "We further demonstrate that optimization objectives like *Policy Gradient* inherently only optimize the quality in alignment tasks." (line 166). Could you specify what exactly you refer to (and potentially cite it)?
- Q3) How does QEMPO affect general model utility?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper intends to solve the issue that the diversity of LLMs' outputs after alignment (fine tuning) significantly reduces. It theoretically demonstrates that the alignment task is essentially requiring improving diversity and quality. Then, it proposes a fine-tuning method (with both offline and online versions) such that both quality and diversity can be preserved after fine-tuning.

### Strengths
The paper is studying an interesting question.

### Weaknesses
1. If I understand correctly, the definitions of $Y_l$ and $Y_w$ are not well-defined.
   - It seems that $Y_l$ and $Y_w$ represent "bad" and "good" responses when deriving theory, where "good" and "bad" are treated as absolute concepts. However, in equation (7) and in section 4.1, $y_w$ and $y_l$ are instead relatively good and bad responses, since the evaluation is based on pairwise comparisons. There seems to be a conceptual gap between these two interpretations, and the paper does not address it clearly.
   - Furthermore, defining $Y_l$ and $Y_w$ is actually nontrivial. If they are meant to include all possible outputs, then the cardinalities $|Y_w|$ and $|Y_l|$ are not guaranteed to be finite. Then a bunch of issues follow. Say, the entire proof of Proposition 2 should be revised. On the other hand, if $Y_w$ and $Y_l$ refer only to subsets of outputs (e.g., within the domain of specific LLMs), then the authors should justify why those particular subsets are considered.
2. The proof of Proposition 2 seems to rely on several assumptions that are not explicitly stated.
   - The **diversity** criterion on page 3 only requires equal probabilities for $Y_w$ but not $Y_l$. However, in the proof of Proposition 2, this condition appears to be applied to both sets.
   - Why does $\frac{1-\varepsilon}{|Y_w|}>\frac{\varepsilon}{|Y_l|}$? This is not obvious since the inequality depends on both values of $\varepsilon$, and $|Y_w|$ and $|Y_l|$. This issue also relates to the problem mentioned in 1.
3. I feel like Proposition 4 and 6 deliver little messages. Since the entropy of a policy depends on hyper parameters, the theoretical results would be more meaningful to include a comparative statement. For example, "when the KL divergence (or expected reward) is fixed, the entropy of XXX is larger than that of XXX."
4. Using Qwen-instruct as base models is somewhat not standard since the Qwen base models already have instructional following capability.
5. The paper's writing quality should be significantly improved. For example, a right quotation mark is utilized where a left quotation mark should be used in the first paragraph. Also, notations are not consistent even on the same page:
   - Proposition 2 uses "KL" while Proposition 3 uses "D_{KL}". 
   - In Table 1, some subscripts use "\text{ref}" while some use "\mathrm{ref}".

### Questions
See weakness.

### Soundness
1

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
The paper addresses the reduction of output diversity commonly observed in alignment methods such as RLHF and DPO. To mitigate this issue, it introduces Quality-constrained Entropy Maximization Policy Optimization (QEMPO) and its variant QEMPO-KL. QEMPO maximizes the entropy of the policy to enhance diversity while ensuring output quality, whereas QEMPO-KL further constrains the divergence from the reference policy. Analytical expressions for the optimal policies are derived, and both offline (DPO-style) and online (RLHF-style) training objectives are presented. Experiments on Qwen and Llama models show that QEMPO(-KL) improves output diversity metrics while maintaining quality comparable to RLHF.

### Strengths
- Clearly motivated by the observed diversity reduction problem in LLM alignment.

- Proposes both offline and online optimization variants (QEMPO, QEMPO-KL) with corresponding closed-form solutions.

- Empirical results on multiple model scales (1B–8B) show consistent diversity improvements across lexical, syntactic, and semantic dimensions.

### Weaknesses
The theoretical analysis appears problematic in several parts (although proofs in Appendix were not fully verified).


- Proposition 2 seems to restate the alignment objective as depending on both “quality” and “diversity,” but the result largely follows from how π* is constructed. It is assumed to be uniform over acceptable responses and negligible elsewhere. Under this assumption, the KL term naturally decomposes into an entropy term and a monotone function of the probability mass on Y_w. While mathematically consistent, the statement feels somewhat tautological and informal rather than a general theoretical insight.

- Propositions 3 and 5: The constraints R (minimum quality) and K (maximum KL divergence) are explicitly set to match the values achieved by the RLHF solution. This makes QEMPO(-KL) the maximum-entropy policy among those that already satisfy RLHF’s quality and distance-to-reference levels. While mathematically consistent, it also seems to take RLHF as the implicit reference operating point, so the comparison mainly shows that one can smooth RLHF’s output distribution without violating its constraints. It is therefore not obvious that this constitutes a general superiority over RLHF, rather than an alternative operating point on the same quality/safety frontier.



- (Minor) Reformulating the RLHF KL-regularized objective as a quality-constrained KL minimization problem (Proposition 1) is mathematically correct but essentially a standard Lagrangian dual form. This formulation is well known in the ML literature. Thus, while it provides a convenient foundation for introducing QEMPO, it does not constitute a novel theoretical contribution by itself, even though the authors claim that “we offer a new way to look at the optimization objective of RLHF”.

### Questions
- In Proposition 4 and 6, the proof infers higher entropy from pairwise ratio inequalities (Eq. 44). However, such pairwise comparisons do not necessarily imply global entropy increase. Could the authors clarify this point?

- The distinction from prior entropy-regularized RL methods such as EnTRPO and ERC-TRPO seems mainly formal, since both also control reward, KL, and entropy simultaneously. Could the authors clarify what substantive methodological difference or novelty QEMPO(-KL) introduces beyond a different Lagrangian formulation?

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
The paper addresses the growing issue that alignment methods such as RLHF and DPO—while improving the quality, helpfulness, and safety of large language model (LLM) outputs—tend to reduce their diversity. To tackle this, the authors introduce a reinforcement learning framework explicitly designed to preserve or enhance diversity while maintaining quality.

They propose two variants:
- **QEMPO (Quality-constrained Entropy Maximization Policy Optimization):** maximizes the entropy of the model’s output distribution, subject to a *quality constraint*. This encourages more varied responses without sacrificing correctness.
- **QEMPO-KL:** an extension that introduces an additional *KL constraint* relative to a reference policy, controlling divergence from the base model and stabilizing optimization.

Both methods are grounded in a theoretical decomposition of the alignment objective into “quality” and “diversity” components. Analytical derivations show that QEMPO yields the highest entropy, followed by QEMPO-KL and RLHF. The methods are implemented in both offline (DPO-like) and online (policy-gradient-based) training regimes. Empirical results on Qwen and Llama models demonstrate consistent diversity improvements with comparable or slightly better quality scores relative to RLHF.

### Strengths
- **Simple and practical modification of RLHF:** The approach extends existing RLHF pipelines with minimal implementation overhead.
- **Code availability:** Implementations for both learning modes are clearly described, supporting reproducibility.

### Weaknesses
- **No related works at all:**  
  The paper omits nearly all prior work on *quality–diversity trade-offs*, both in evaluation and training, leaving the contribution insufficiently contextualized. This includes several theoretical and empirical studies on balancing diversity and reward, entropy-based evaluation, and diversity-aware optimization methods. These works include dozens of papers, among which I highlight a few important ones and recent contributions below:
  - **Sajjadi, M. S. M. et al.**  
  *Assessing Generative Models via Precision and Recall*  
  NeurIPS 2018 (NIPS 2018), 2018.  
  [arXiv:1806.00035](https://arxiv.org/abs/1806.00035)

   - **Kynkäänniemi, T. et al.**  
  *Improved Precision and Recall Metric for Assessing Generative Models*  
  NeurIPS 2019 (Advances in Neural Information Processing Systems 32), 2019.  
  [arXiv:1904.06991](https://arxiv.org/abs/1904.06991)

   - **Khalifa, M. et al.**  
  *A Distributional Approach to Controlled Text Generation*  
  ICLR 2021 (camera-ready version), 2020.  
  [arXiv:2012.11635](https://arxiv.org/abs/2012.11635)

   - **Le Bronnec, F. et al.**  
  *Exploring Precision and Recall to Assess the Quality and Diversity of LLMs*  
  ACL 2024 (62nd Annual Meeting of the Association for Computational Linguistics, Long Papers), 2024.  
  [arXiv:2402.10693](https://arxiv.org/abs/2402.10693)

   - **Sun, H. et al.**  
  *Inverse-RLignment: Large Language Model Alignment from Demonstrations through Inverse Reinforcement Learning*  
  arXiv preprint (LLM alignment via inverse RL), 2024.  
  [arXiv:2405.15624](https://arxiv.org/abs/2405.15624)

   - **Shypula, A. et al.**  
  *Evaluating the Diversity and Quality of LLM Generated Content*  
  arXiv preprint (LLM diversity/quality evaluation), 2025.  
  [arXiv:2504.12522](https://arxiv.org/abs/2504.12522)

   - **Verine, A. et al.**  
  *Improving Diversity in Language Models: When Temperature Fails, Change the Loss*  
  ICML 2025 (Forty-Second International Conference on Machine Learning), 2025.  
  [arXiv:2508.09654](https://arxiv.org/abs/2508.09654)

   - **Wang, T. et al.**  
  *On the Effect of Sampling Diversity in Scaling LLM Inference*  
  arXiv preprint (scaling inference / Best-of-N with diverse prompts), 2025.  
  [arXiv:2502.11027](https://arxiv.org/abs/2502.11027)
  

- **Presentation quality:** The manuscript has formatting problems (tables and figures are not properly fitted to the page) and numerous typos, including in the abstract. These issues detract from readability and professionalism.  
- **Limited discussion of limitations:** There is little reflection on potential downsides of entropy maximization rather than recall per se, such as incoherent or low-quality generations when constraints are too loose.

### Questions
--

### Soundness
2

### Presentation
2

### Contribution
1
