# RSPO: Regularized Self-Play Alignment of Large Language Models

- Decision: Reject
- Scores: 2, 8, 2

## Abstract
Self-play alignment has emerged as an effective approach for fine-tuning large language models (LLMs), formulating preference optimization as a two-player game. However, the regularization with respect to the reference policy, which is crucial for mitigating over-optimization, has been insufficiently investigated in self-play alignment. To study the impact of different regularization strategies, we propose **Regularized Self-Play Policy Optimization (RSPO)**, a novel framework that unifies prior methods and enables simple plug-and-play regularizers, meanwhile preserving convergence to Nash equilibrium of the corresponding regularized game. We observe that RSPO with appropriate regularizers can substantially improve the length-controlled win rate (LCWR) on AlpacaEval-2 across a range of base models, while also achieving consistently superior performance on Arena-Hard, MT-Bench, ArmoRM, and response diversity. In particular, RSPO improves unregularized self-play baseline (SPPO) on AlpacaEval-2 LCWR from $28.5\\%$ to $ 35.4\\%$ with base model Mistral-7B, from $38.77\\%$ to $43.66\\%$ with LLaMA-8B, and from $50.54\\%$ to $51.83\\%$ with Gemma-2B. Combining simplicity, convergence guarantees, and significant empirical gains, RSPO offers a strong foundation for exploring regularized self-play in language model alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Regularized Self-Play Policy Optimization (RSPO), a new framework for aligning large language models (LLMs) using self-play. The authors argue that existing self-play alignment methods have not adequately explored regularization against a reference policy, which is critical for preventing over-optimization. RSPO addresses this by providing a flexible and simple way to incorporate various "plug-and-play" regularizers into the self-play loss function. The framework is theoretically grounded, with proofs showing that it maintains convergence to a Nash Equilibrium of the regularized game. Empirically, the paper demonstrates that RSPO with well-chosen regularizers significantly improves performance over unregularized baselines (like SPPO) across several benchmarks (AlpacaEval-2, Arena-Hard, MT-Bench) and base models (Mistral-7B, LLaMA-8B, Gemma-2B). The authors also analyze the distinct effects of different regularizers (e.g., forward vs. reverse KL) and show that their combination can lead to the best results.

### Strengths
- The paper does an excellent job of motivating the need for better regularization in self-play alignment. The problem of over-optimization, especially with imperfect preference models, is well-known in RLHF, and its insufficient investigation in the context of self-play is a relevant research gap.

- The proposed RSPO framework is elegant in its simplicity. The idea of adding a modular, plug-and-play regularization term to an existing self-play loss is appealing. Crucially, this simplicity does not come at the cost of theoretical rigor. The authors provide solid theoretical guarantees, linking RSPO to Mirror Descent and proving last-iterate convergence, which adds significant credibility to the method.

- The experimental setup is thorough and convincing. The authors evaluate RSPO on multiple popular benchmarks and against strong baselines. The use of different base models demonstrates the general applicability of the method. The ablation studies on different regularizers and their temperatures (Figure 4) provide valuable insights into their distinct effects on win rate and response length.

- The results presented are impressive. RSPO consistently outperforms the unregularized SPPO baseline and other methods across the board. The significant win rate improvements on benchmarks like AlpacaEval-2 (e.g., 28.5% to 35.4% for Mistral-7B) clearly demonstrate the practical benefits of the proposed approach.

### Weaknesses
In my opinion, this paper has a major and unavoidable issue. **All the theories in this paper (e.g., RSPO is flexible for general regularization) have already been thoroughly studied in the field of game theory**, yet the authors introduce them into the LLM field without any citations. Admittedly, the presentation of this paper is excellent. However, directly copying results from the game theory field contributes nothing to the community.

[1] Kenshi Abe, Mitsuki Sakamoto, Kaito Ariu, Atsushi Iwasaki. Boosting Perturbed Gradient Ascent for Last-Iterate Convergence in Games ICLR 2025.

[2] Kenshi Abe, Kaito Ariu, Mitsuki Sakamoto, Atsushi Iwasaki. Adaptively Perturbed Mirror Descent for Learning in Games. ICML 2024.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a framework called Regularized Self-Playing Strategy Optimization (RSPO) for alignment fine-tuning of Large Language Models (LLMs). Existing self-play alignment methods have not sufficiently explored regularization terms to mitigate “over-optimization” issues. RSPO unifies existing self-play approaches by introducing a plug-and-play regularization term while ensuring the algorithm converges to the Nash equilibrium of the corresponding regularized game. Experimental results demonstrate that RSPO with appropriately chosen regularization terms significantly outperforms the unregularized baseline model (SPPO) and achieves performance improvements across multiple models.

### Strengths
1. Theoretically rigorous and experimentally comprehensive, it identifies and resolves the critical issue of “over-optimization.”
2. As a practical framework for LLM alignment, RSPO demonstrates excellent flexibility and scalability, laying the groundwork for subsequent exploration of more complex regularization methods.

### Weaknesses
1. Diversity evaluation (self-BLEU) is limited to a single metric and lacks complementary measures such as entropy.
2. While the baseline comparison includes mainstream methods like SimPO, it does not evaluate other recent variants. We recommend expanding the comparison to enhance persuasiveness.

### Questions
1. In the experiment, only PairRM (0.4B) was used as the preference model. Has RSPO performance been tested with larger preference models (e.g., 70B RM)? If the preference model is more accurate, does the need for regularization decrease?
2. Could you provide λ tuning details, such as the search range, to facilitate reproduction?
3. Computational overhead (Appendix E.3) remains unquantified. Could you provide training times for RSPO vs. SPPO for reference?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose Regularized Self-Play Policy Optimization (RSPO), a novel framework that flexibly allows for "plug-and-play" integration of general regularization terms into the standard self-play loss. The paper claims to provide theoretical guarantees for this approach, asserting that RSPO is an implementation of Generalized Magnetic Mirror Descent (GMMD) and thus achieves last-iterate convergence to the Nash Equilibrium of the corresponding regularized game. Empirically, RSPO is shown to significantly outperform the unregularized SPPO baseline across various models and benchmarks.

### Strengths
1. The paper tackles the significant and practical issue of over-optimization in state-of-the-art self-play alignment methods (like SPPO), which largely lack the explicit regularization needed for stable, long-run training.

2. A key strength of RSPO is its simplicity. It allows researchers to add any regularization term ($\lambda R(\pi_{\theta}, \mu)$) directly to the existing SPPO loss, making it highly practical and easy to implement for experimenting with diverse regularizers (e.g., forward KL, reverse KL, $\chi^2$).

3. Extensive experiments showing that RSPO significantly outperforms the unregularized SPPO baseline on multiple benchmarks (AlpacaEval-2, Arena-Hard). It empirically proves its ability to solve the over-optimization problem (Figure 2, left) and provides valuable in-depth analysis on how different regularizers distinctly impact model behavior.

### Weaknesses
1. The claimed theoretical equivalence between RSPO and GMMD (Section 3.2, Appendix A.4) is invalid. In Appendix A.4 (Eq.41--46), the authors attempt to prove that $\nabla L_{RSPO} \propto \nabla L_{GMMD}$, but Eq.41 already defines $\nabla L_{GMMD} = \frac{1}{2} \nabla L_{RSPO}$; thus, the proof is circular and does not demonstrate any real equivalence between the two algorithms. In addition, the substitution $y \sim \pi_\theta \to y \sim \pi_t$ is made without importance-sampling correction or a small-step-size assumption, introducing unacknowledged bias. Since RSPO modifies the SPPO MSE surrogate by adding a regularizer, while GMMD performs a single mirror-descent proximal update, they are different optimization procedures. Consequently, the last-iterate convergence guarantee of GMMD cannot be directly inherited by RSPO, which should instead be described as a heuristic method inspired by GMMD rather than a theoretically equivalent one.
2. Even if we accept the GMMD proof, Prop. 3.1 hinges on Assumption A.1 (1-relative strong convexity w.r.t. entropy). The paper explicitly notes that Forward KL violates this assumption, yet the best-performing Mistral-7B setup uses IS-For.+Rev. KL. No sufficient conditions are provided to ensure that the mixed regularizer remains relatively strongly convex. Hence, the theoretical guarantee does not apply to the paper’s strongest empirical results.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
