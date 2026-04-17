# The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Recent efforts to accelerate LLM pretraining have focused on computationally-efficient approximations that exploit second-order structure. This raises a key question for large-scale training: how much performance is forfeited by these approximations? To probe this question, we establish a practical upper bound on iteration complexity by applying full Gauss-Newton (GN) preconditioning to transformer models of up to 150M parameters. Our experiments show that full GN updates yield substantial gains over existing optimizers, achieving a 5.4x reduction in training iterations compared to strong baselines like SOAP and Muon. Furthermore, we find that a precise layerwise GN preconditioner, which ignores cross-layer information, nearly matches the performance of the full GN method. Collectively, our results suggest: (1) the GN approximation is highly effective for preconditioning, implying higher-order loss terms may not be critical for convergence speed; (2) the layerwise Hessian structure contains sufficient information to achieve most of these potential gains; and (3) a significant performance gap exists between current approximate methods and an idealized layerwise oracle.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors of this manuscript investigate the practical upper bound of second-order optimization methods for training Large Language Models (LLMs). They establish this bound by applying a full Gauss-Newton (GN) preconditioner to transformer models of up to 150M parameters, comparing its iteration complexity against strong baselines like SOAP and Muon. They report that the full GN update achieves a significant reduction in training iterations (e.g., 5.4x over SOAP). Furthermore, they find that a precise layerwise GN preconditioner, which ignores cross-layer information, achieves nearly identical performance to the full GN method.

### Strengths
1- Studies showing the capability and limitation of current and ideal second-order optimization methods are very useful and insightful.

2- The paper is well-written and addresses a clear and important question: what is the practical performance ceiling for second-order methods in LLM training?

3- The finding that a precise layerwise Gauss-Newton preconditioner nearly matches the full GN method is a very strong and useful insight, as it suggests where future efforts in developing practical algorithms should be focused.

### Weaknesses
1- The current study lacks a comparison to an important family of second-order optimizers: the KFAC [1] family, which is orthogonal to the Shampoo-family methods (SOAP, Muon) that were analyzed. KFAC optimizers have been successfully scaled to large models, such as in KAISA [2] and MKOR [3]. It is important to compare the capabilities of such methods and understand how they relate to the Gauss-Newton "oracle" presented in this work.

2- In line 117, the authors claim that the most common optimizer used for LLMs is Adam, while AdamW [4] is more commonly used and is the default for many libraries (e.g., HuggingFace Trainer). While the authors use AdamW as a baseline, a direct comparison including both Adam and AdamW as baselines would be insightful to clarify the impact of decoupled weight decay relative to the second-order gains.

4- This study mainly focuses on the beginning part of the training process, where the loss decreases very rapidly (e.g., Figures 1 & 2 show convergence to a relatively high loss of 3.25-3.4). This is in contrast to the most dominant and expensive part of LLM pretraining, where the loss decrease is very gradual and almost plateaus. This focus is a limitation and leads to questioning the practicality of the findings for the full, end-to-end training process. I understand that scaling these optimizers (especially the full GN method) is very expensive, but without experiments that explore this later-stage convergence, the findings on iteration complexity gains remain inconclusive.

---

[1] Martens, J., & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature. arXiv:1503.05671.

[2] Pauloski, J. G., et al. (2021). KAISA: An Adaptive Second-Order Optimizer Framework for Deep Neural Networks. SC21.

[3] Mozaffari, M., et al. (2023). MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates. NeurIPS 2023.

[4] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019.

### Questions
1- How does the performance and scaling of the full Gauss-Newton oracle compare to the KFAC family of optimizers [1], including scaled implementations like KAISA [2] and MKOR [3]?

2- Could the authors provide an ablation that includes both Adam and the baseline AdamW [4] to clarify the impact of decoupled weight decay in this context?

3- How do the observed iteration complexity gains from GN hold up in the later stages of training, after the initial rapid loss decrease and into the long-tail plateau phase?

---

[1] Martens, J., & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature. arXiv:1503.05671.

[2] Pauloski, J. G., et al. (2021). KAISA: An Adaptive Second-Order Optimizer Framework for Deep Neural Networks. SC21.

[3] Mozaffari, M., et al. (2023). MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates. NeurIPS 2023.

[4] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the upper bound potential of second-order optimization for large language models (LLMs) using full Gauss-Newton (GN) preconditioning. By implementing a memory-feasible GN method via Jacobian-vector products, the authors estimate the theoretical limit of convergence speed relative to modern optimizers like SOAP, Muon, and AdamW.
Experiments on 45M- and 150M-parameter LLaMA models show 5.4× faster convergence and larger critical batch sizes for GN compared to SOAP. The layerwise GN variant nearly matches full GN, implying that most curvature information is local and that higher-order loss terms are unnecessary.  
Overall, the paper provides a rigorous and well-documented study that clarifies how far second-order optimization could improve LLM training efficiency, even if the proposed method itself is not yet practical at scale.

### Strengths
* Clearly defined and novel research question: measuring the upper bound of second-order optimization performance for LLMs.
* Strong empirical methodology, including detailed hyperparameter sweeps and compute transparency.
* Insightful findings: full GN drastically improves iteration efficiency, and layerwise GN nearly replicates its gains.
* Excellent documentation and reproducibility, with appendices covering training setups, tuning, and resources.
* Practical relevance: establishes a benchmark for how close current optimizers are to theoretical limits.

### Weaknesses
* Experiments are limited to 150M-parameter models, far smaller than modern LLMs.
* The study focuses on iteration count, not wall-clock time or energy cost, so real-world efficiency gains remain unclear.
* Comparisons may be slightly biased since the GN method uses Muon as an inner optimizer, giving it an inherent advantage.
* No multiple-run variance reporting; results could benefit from confidence intervals.
* Theoretical analysis of why layerwise GN performs nearly as well as full GN is missing.

### Questions
* How much slower is each GN iteration in wall-clock time compared to SOAP or Muon?
* Would the results generalize to models with stronger inter-layer dependencies (e.g., MoE architectures)?
* How sensitive are the results to the choice of inner optimizer (Muon vs. AdamW)?
* Can mixed-precision or blockwise GN approximations retain most of the benefits?
* Does the GN method still outperform others in smaller-batch or lower-data regimes?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides an empirical study of the application of the Gauss-Newton algorithm in LLM training. The paper studies a valid approximation method for lowering the memory and computational cost of the original Gauss-Newton method, making it feasible for LLM training. Empirical results show potential of this second-order approach, especially under large batch settings.

### Strengths
1. The motivation of the paper is interesting and inspiring, bringing second-order optimization back to the regime of LLM training, and shows promising performance.
2. The idea for approximating the full Gauss-Newton is interesting, making the algorithm practical and applicable to practical problems.

### Weaknesses
1. The notations and claims in the paper may need clearer explanations. I suggest the authors include more intuition on how this approximation is valid in approximating the updates of real Gauss-Newton and what the major differences are between the proposed approach and the real Gauss-Newton method.
2. For the experiment part, I have two major concerns. Firstly, as an empirical study of the application of the Gauss-Newton algorithm, the experiment scale seems not large enough, since the paper only examines models of size 45M and 150M. Also, it seems that only when a very large batch size (compared to the full token number) is adopted, the benefits of the Gauss-Newton method are clear, which seems somewhat limited.

### Questions
1. In practice, I am curious about how we should tune the hyperparameter $ N $ (the inner loop length)? Schedule choices may also be a concerning point. There seems to be still a long way to go to put this second-order algorithm into practical training settings.
2. What about the wall-clock training time of the algorithm? How is it compared to Muon or Adam?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper test the performance of full 2nd-order method (Gauss-Newton) on LLM pretraining. The aim of the paper is to explore the potential of classical 2nd-order methods in mordern pretrain tasks.  Although the proposed implementation is still compute intensive, this trial is meaningful and provides valuable guidance for the future optimizer design.

### Strengths
The topic of "exploring the 2nd-order methods in LLM pretrain" is crucially important. The presentation is clear. The experiments are sound. The message is valuable and quite worth sharing with the community.

### Weaknesses
The presentation can be further improved, as I explained below.

### Questions
**Summary:** The paper test the performance of full 2nd-order method (Gauss-Newton) on LLM pretraining. The aim of the paper is to explore the potential of classical 2nd-order methods in mordern pretrain tasks.  Although the proposed implementation is still compute intensive, this trial is meaningful and provides valuable guidance for the future optimizer design. 

**Strength:**  The topic of "exploring the 2nd-order methods in LLM pretrain" is crucially important. The experiments are sound. The message is valuable and quite worth sharing with the community. 

**Disclaimer:** Many may criticize that the computational cost of this paper remains high, which is true. However, I do not consider it a critical issue. For me, the directional guidance provided by this work outweighs its computational limitations. The reasons are as follows:

1. Although traditional second-order optimization methods have a rich theoretical foundation, there have been almost no meaningful attempts to apply them directly to modern neural networks or large language models, except a few early attempts on RNNs 10 years ago by Martins and Sustkever.  For modern LLM pretrain, we only know that several heavily approximated versions of Gauss-Newton can work in practice (e.g., Muon and Adam), but it remains unclear whether the non-approximated versions would be effective. This script gives a postive answer on a meaningful setting (nanoGPT speedrun).

2. Approximated methods often develop new characteristics that lead to new theoretical interpretations—for instance, Muon exhibits properties related to matrix orthogonalization. Researchers tend to construct new theoretical frameworks to explain these methods rather than treating them as approximations of classical approaches. While this promotes academic diversity and inspire new thoughts, it also creates confusion in algorithm design: Should we follow classical theories or newer principles? This paper demonstrates that following classical theory can indeed lead to success. This directional guidance is highly valuable, and I believe this work has the potential to inspire many future optimizer designs.

In summary, I think this is an important paper that provides important directional guidance to future optimizer design. The computational overhead is a minor issue since the main contribution lies in "guidance" rather than "an algorithm itself".  I vote for clear acceptance. 

**Questions related to presentation:**  The paper presentation is not prefect and can be further improved.

**Question 1:**
 I am quite confused by Section 7.2. LINEARIZED MODEL METHOD. 

1-1. What is the explicit form of this Algorithm 2? As an analogy to previous sections: the minimization over 2nd-order Taylor of the loss over the linearized model gives  w= w - G-1 grad, does algorithm 2 admit closed-form expression, and what is the closed-form if it exists?

1-2. Is algorithm 2 still a 2nd-order algorithm? 

1-3. What is the implication on the loss landscape if Algorithm 2 works well?

1-4. How do you solve the subproblem in Section 7.2. 

1-5. The name "Algorithm 2: Linearized model method" is quite misleading. Both Algorithm 1 and 2 are based on the Linearized model, an  if I understand correctly, Algorithm 2 actually incorporates higher-order loss function information than Algorithm 1.  The name "Linearized model method" may leave the impression that "Algorithm 1 incorporates higher-order information than Algorithm 2", but actually it is the other way around (correct me if wrong). 


**Question 2:** The algorithm presentation (Algorithm 1 and Algorithm 2) is confusing. It is impossible to understand the line "Update", e.g., in line 231. I suggest adding a hyperlink to tell readers which equation is used in defining this "Update" operator. Similarly, the line "Line search" is confusing as well. In optimization theory, line search usually refers to a particular procedure of choosing the stepsize of GD. But here you mean something different. Please add more detailed description.


**Question 3:** "We start all runs from the same AdamW post-warmup checkpoint." I dont understand this sentence. What do you mean by "all runs" and what is "AdamW post-warmup checkpoint"? Further, why not use random initialization?

**Questions related to experiments:**

**Question 4:** When computing L(θ_t + α(θ̂ − θ_t)) during line search, do you reuse the same data as in the inner loop, a held-out/proxy mini-batch, or freshly sampled data? Please specify for both full GN and layerwise GN

**Question 5:** Ablation study on line search. I’m wondering whether other optimization algorithms also rely on line search. If they don’t, why does line search improve the performance of Gauss–Newton? In addition, how much of Gauss–Newton’s performance gain comes from the algorithm itself rather than from the use of line search?

**Question 6:** The current script uses  GN^-1: w = w  - G^-1 grad. How about GN^-1/2: w = w  - G^-1/2 grad? Is there any feasible way to implement GN^-1/2? How would you expect it to perform?

**Question 7:** How would GN perform if you keep training on more tokens? Will the benefit shrink as the token size reaches 1x or 2x Chinchilla, or even more tokens?

**Question 8:**  In Figure 1, the x-axis is the number of outer steps, while Gauss–Newton uses a much larger batch size (240 M tokens vs. 40 M for AdamW/Muon) and each “step” includes several inner optimizer iterations and line-search evaluations. This makes the comparison on a per-step basis potentially misleading. I suggest reporting loss versus total tokens processed.

**Code:** No code is provided in the submission. I strongly suggest the authors provide an anonymous code link.  I will check the code, and I believe it is the most efficient way to help address my questions above.

I am more than happy to raise my score to 10 if my questions are addressed.

### Soundness
4

### Presentation
4

### Contribution
4
