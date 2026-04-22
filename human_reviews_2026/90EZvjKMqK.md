# Security-Constrained Fine-tuning: Preventing Knowledge Restoration in Unlearned Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Large language models face a critical vulnerability through relearning attacks, where adversaries exploit fine-tuning to restore knowledge that was intentionally removed via unlearning procedures. Current post-hoc safety evaluations detect violations only after fine-tuning completion, creating security gaps and computational waste.
We introduce a safety-constrained fine-tuning framework that proactively prevents relearning attacks by formulating defense as constrained optimization. Legitimate fine-tuning objectives are optimized subject to explicit constraints preventing restoration of forgotten knowledge. We present an efficient *Constraint-Aware Gradient Descent* algorithm that replaces intractable nonlinear constraints with first-order Taylor approximations, yielding convex quadratic subproblems with closed-form solutions.
Comprehensive experiments on Llama models demonstrate robust defense against relearning attack scenarios while maintaining legitimate fine-tuning performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the problem of LLMs relearning forgotten knowledge after machine unlearning. The authors propose a constrained fine-tuning framework that treats safety/unlearning as a hard constraint during optimization rather than a post-hoc check. They derive a constraint-aware gradient update based on safe gradient flow, preventing the model from reducing the “forget loss” during fine-tuning, thereby blocking relearning attacks. Experiments on LLaMA models under the TOFU benchmark show that the method effectively preserves unlearning while maintaining utility on normal data.

### Strengths
1. The paper formulates the problem as a constrained optimization task and introduces an optimization algorithm to address it, providing a meaningful and clear perspective.
2. Theoretical guarantees are thoroughly presented, with detailed proofs ensuring that the constraint is satisfied during optimization.
3. The problem is modeled in a clear manner, and the algorithm is described with mathematical rigor and precision.

### Weaknesses
1. The constrained optimization formulation and gradient correction mechanism largely overlap with existing work [1]. Thus, the core optimization idea is not conceptually new.
2. The problem positioning is somewhat unclear. Although the paper frames the method as safety- or unlearning-related, the focus is specifically on defending against relearning after unlearning. This is neither a purely unlearning problem nor a standard safety problem, and the connection to safety is indirect. Moreover, for the relearning attack setting, no baseline methods are provided, making it difficult to assess the actual effectiveness of the approach.
3. The experiments are limited in scope. The paper only evaluates relearning mitigation, without including complexity analysis, runtime evaluation, ablation studies, or sensitivity to hyperparameters. As a result, the empirical validation is insufficient.

[1] Allibhoy, A., & Cortés, J. (2023). Control-barrier-function-based design of gradient flows for constrained nonlinear programming. IEEE Transactions on Automatic Control, 69(6), 3499-3514.

### Questions
1. Since the paper frames the method in the context of safety-constrained fine-tuning, could the authors provide results on safety-centric benchmarks or tasks (e.g., harmful content prevention, jailbreak robustness, refusal alignment), rather than only focusing on the relearning scenario after unlearning? This would help clarify whether the method contributes to safety itself, rather than only mitigating relearning.
2. The optimization framework appears strongly related to prior work on constrained gradient flows and control barrier functions. Are the contributions mainly in applying this framework to the relearning/unlearning setup, or does the method introduce any fundamentally new theoretical ideas, constraint formulations, or algorithmic mechanisms beyond existing constrained optimization literature?

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
4

### Summary
This paper aims to address the retraining attack towards machine unlearning on large language models (LLMs). The background is that LLM maintainers may need to unlearn some training samples for a given LLM. However, as revealed by existing works, such unlearning is incomplete and there remain some residuals in LLMs; the unlearned samples can be rapidly recovered by simple fine-tuning. This procedure is named the retraining attack. To address this problem, they propose a new constrained-based optimization method. The core idea is to formalize the constraint terms beyond the original objective function, and such a subjection condition can be derived to an extra gradient optimization direction. Theoretically, the authors prove that under specific conditions their method can ensure the satisfaction of the condition under the promise of control barrier function. They provide some experiments on LLM unlearning using the TOFU dataset for empirical validation.

### Strengths
+ This paper considers an interesting problem, i.e., how to achieve better unlearning with higher quality.
+ The proposed algorithm is somewhat novel and comes with a theoretical guarantee.

### Weaknesses
+ The organization of this paper seems terrible. There are lots of unnecessary repetitions between Section 4 and Sections 1 & 2. I strongly suggest the authors consider reformatting Section 4 by removing all unnecessary emphasis on contributions (there are too many), related work introductions (you have already clarified them before), and Remark 4.1.
+ The threat model explored in this paper is unclear. When I consider it, I think it could be impractical. Retraining attack towards unlearning, in my opinion, is more about unlearning analysis than a realistic attack. Under the retraining attack scenario, who is the adversary? Is it the model maintainer or user? Who is the defender? Does this paper aim to provide a more robust/powerful unlearning than existing methods, making the forgotten dataset harder to recover, or is this strategy added exactly at the fine-tuning (i.e., re-training attack) process? If it is the latter, why would the attacker choose to use your algorithm (which is a defense) in his/her retraining attack procedure?
+ Significant details are missing. Even after reading the whole paper, I cannot find the exact formulation of $L_{fgt}$ shown in Equation (8). You can skip the definition of $L_{ce}$ as it is at least noted as $CE$, letting me know that it is cross-entropy-based fine-tuning, but what are the formulations of $L_{fgt}$, $L_{rtn}$, and $L_{Safety}$ used in your experiments?
+ Regarding experiments, there are some concerns too. The first one is that the baseline is overly simple. The experiments only compare with one baseline, i.e., no defense. I suggest the authors include existing unlearning methods that consider the residual (if there are any) and some simple baselines. For instance, they could use a simple convex combination of the objective function and the subjected objective function as the training loss as a baseline.
+ My second concern about the experiments is their results. Under the same model backbone with different datasets, why is {} significantly lower than others? From 57 to 90, it is a really big gap and not reasonable. Also, for {}, why does retrained exhibit a higher forgetting score than unlearned? I regard the retrained line as the results of a retrained model, which should be an ideal situation, i.e., the upper bound of any unlearning methods. Additionally, why does llama3.1-8B on $D_{nor}$ exhibit a forgetting score of 49 under naive fine-tuning? For the same setting with other models, it is zero.
+ Minor issues. There are many misuses of `citep` and `citet` in the paper. Also, I cannot get timely explanations of notations. For instance, what is $()_+$ shown in line 210? I cannot get the explanation until I read the next equation which also uses this notation.

### Questions
Please refer to the weakness.
If there are some significant misunderstandings, I'd like to raise my score.

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
4

### Summary
The paper proposes a novel framework for security-constrained fine-tuning of large language models (LLMs), motivated by the vulnerability of unlearned models to “relearning attacks,” where adversaries reintroduce deliberately forgotten knowledge via fine-tuning. The authors formulate fine-tuning as a constrained optimization problem, where standard objectives are optimized subject to constraints that explicitly prevent restoration of unwanted information. They introduce a Constraint-Aware Gradient Descent algorithm, which efficiently approximates intractable nonconvex constraints by leveraging first-order Taylor expansions to yield closed-form update steps, and demonstrate empirically—using the TOFU benchmark and Llama models—that their approach can block relearning attacks while largely maintaining legitimate task performance.

### Strengths
- Principled Approach: The re-casting of safe fine-tuning as an explicit constrained optimization problem is well-motivated and offers a methodological improvement over heuristic or purely post-hoc safety checks. By directly incorporating security or compliance constraints into the training objective, the approach has clear benefits for both practical deployment (e.g., commercial LLM APIs) and theoretical understanding.

- Algorithmic Clarity and Efficiency: The closed-form corrective step for constraint satisfaction is derived rigorously (as presented in Section 3.2 and detailed in the Appendix), offering theoretical guarantees (Theorem 3.1) under reasonable assumptions and practical implementation advantages—especially when constraints and objectives share structure (see discussion following Eq. 5 and Algorithm 1).

- Strong Empirical Demonstration: The experimental evaluation (Section 5 and Table 1) convincingly shows that, on the TOFU benchmark, the proposed method can block relearning attacks nearly completely (Forgetting metric remains $\sim0$ for Ours vs. $\gg0$ for naive methods), while still permitting high performance on legitimate customization objectives (ROUGE-L and Model Utility metrics remain competitive in non-adversarial scenarios). The ablation in Table 2 further details the practical sensitivity to constraint dataset size.

- Good Use of Visualizations: Figure 1 concisely and effectively illustrates the threat model, progression from unlearning to relearning attacks, and the intended protection offered by the proposed method—contributing to clarity of exposition for readers less familiar with the problem space.

- Methodological Rigor: Mathematical aspects are well worked out, with the key corrective update and step size derived and justified; explicit conditions for anytime constraint satisfaction in practice (Theorem 3.1, Eq. 8) are provided and explained.

- Reproducibility and Transparency: Implementation details (Appendix A.2.1) are given with explicit hyperparameters, LoRA setup, hardware environment, and use of the open-unlearning codebase, supporting practical reproducibility.

### Weaknesses
- Limited Scope of Experiments and Baselines: The experimental analysis is limited primarily to the TOFU benchmark and Llama models across three sizes, with a single dominant baseline (naive fine-tuning, i.e., no defense). The exclusion of additional recent competitive defenses, especially those specifically designed for robust unlearning (e.g., MUSE, OpenUnlearning, and methods from Fan et al. 2025; Zhan et al. 2023), seriously weakens claims of broad practical superiority or generalizability.

- Missing Direct Comparison to Recent Benchmarks: Notable recent benchmarks for knowledge persistence and unlearning, such as RWKU (Jin et al., 2024), are not employed, and related-state-of-the-art methods uncovered in the provided search context (e.g., Are We Really Unlearning? by Hsu et al., 2025) are absent from both related work discussion and experimental evaluation. This omission leaves it unclear whether the framework’s strengths generalize beyond TOFU and standard Llama models.

- Mathematical/Algorithmic Questions: While the formulation is clear, Section 3.2 and the proof in Appendix A.1 depend on $g$ and $f$ being continuously differentiable and on the $L_g$-Lipschitz property holding. In the LLM setting, with high-dimensional, non-smooth parameter landscapes and potential batch-level stochasticity, how these assumptions translate and whether the guarantees persist are not empirically probed. Moreover, constraint satisfaction hinges on choices of $\alpha^k$ and $\eta^k$ (see Eq. 8 and Appendix A.2.1), but little practical guidance beyond using $\alpha^k=1/\eta^k$ is given—leaving questions about robustness to hyperparameter selection.

- Underdeveloped Discussion of Trade-offs: Table 1 shows that, especially for the largest (8B) Llama model and in pure adversarial settings, the constraint-aware method can result in degraded Model Utility—even a collapse to zero. The explanation is brief: “this can be avoided with further hyperparameter modifications”—but no details or concrete mitigation strategies are provided. This trade-off between safety and utility deserves much deeper quantitative and qualitative investigation. Furthermore, Figure 1, while helpful as a conceptual overview, could benefit from added detail or supplementary visualizations (such as training dynamics or constraint violation plots).

- Insufficient Exploration of Broader Applicability: While the paper claims broad generalizability beyond the relearning scenario (see Remark 4.1), the methodology is only demonstrated concretely for the relearning/TOFU setting. No empirical evidence or pilot results are presented for privacy-preserving personalization, regulatory constraint scenarios, or alignment-preserving fine-tuning—limiting the demonstrable scope of utility at this stage.

- Incomplete Literature Positioning: Although the related work section (Section 2) is detailed, several recent empirical and benchmark efforts (RWKU, Hsu et al. 2025) and especially works exploring the persistence of residual knowledge in unlearning (Wei et al. 2023, Jin et al. 2024) are not cited or contrasted—even as these have material implications for evaluation and baseline comparison.

- Algorithmic Clarity and Pseudocode: While Algorithm 1 provides a reasonable high-level - outline, some notational choices (e.g., the usage of inner products and matrix norms in update directions) and practical training adjustments are only briefly described. More detailed pseudo-code or reference implementations would further improve clarity and usability.

- Constraint Dataset Selection and ‘Coverage’: Table 2 ablates constraint dataset size but relies on the TOFU ‘forget10’ subset. There is no analysis of the effect of distribution mismatch between constraint/evaluation and potential adversarial data, or guidelines for constructing constraint sets in more realistic, open-set environments.

- Potential for Over-constraining: With a strong constraint dataset or high $\alpha$, the method may lead to a substantial loss of model utility for non-adversarial objectives (as partially seen in Table 1 for the 8B model). The balance between utility and safety is acknowledged as tunable but not robustly quantified.

### Questions
- Generality to Non-TOFU Datasets & Real-World Scenarios: Beyond TOFU and synthetic ‘cloned’ datasets, how well does the method perform in realistic, noisy data environments, or when forget/constraint sets are not perfectly curated? Any results or insights into cross-domain/generalization, especially with distributional drift between constraint and attack sets?

- Hyperparameter Sensitivity and Robustness: 
How robust is performance (with respect to both constraint satisfaction and model utility) to the choice of $\alpha^k$ and $\eta^k$? Is the anytime constraint guarantee still approximately realized in practice under batch SGD and noisy gradients? Would using adaptive or learned schedules for these parameters improve results?

- Scalability to Larger Models and Real-World Workloads: 
Practical impact on compute, memory, and training time when scaling to billion-param models or production settings, where constraints may involve far larger and more diverse safety sets. Any insights or measured overheads?

- Utility-Safety Trade-off Quantification: 
Is there a more principled or quantitative way to control or visualize the trade-off between constraint tightness (e.g., lower $\varepsilon$ or larger constraint dataset) and model utility on retained tasks? Could this be integrated into hyperparameter selection or automated tuning?

- Attack Adaptivity and Constraint Bypass: 
Have the authors considered more sophisticated adversarial attacks aimed at bypassing explicit constraints (e.g., using paraphrases, alternative prompts, or multi-turn generation) rather than direct fine-tuning on the forget set? How would the current approach handle such cases?

- Open-Set Constraint Handling: What guidance, if any, can the authors provide for constructing constraint datasets in settings where the scope of unwanted knowledge is ambiguous or not fully enumerated? Any ideas for learning constraints or constraint sets adaptively?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
the work proposes a new finetuning method that should increase a model's robustness to finetuning attacks. to achieve this, the authors propose a constrained gradient-descent algorithm, which is evaluated with Llama 3 on the TOFU dataset.

### Strengths
- Well motivated problem setting and derivation of constraints and objectives

### Weaknesses
- Experiments only on outdated Llama models
- No baselines considered
- Evaluation only on a single dataset? (TOFU)

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
1
