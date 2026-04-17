# CoLD: Counterfactually-Guided Length Debiasing for Process Reward Models in Mathematical Reasoning

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Process Reward Models (PRMs) play a central role in evaluating and guiding multi-step reasoning in large language models (LLMs), especially for mathematical problem solving. However, we identify a pervasive length bias in existing PRMs: a tendency to assign higher scores to more verbose reasoning steps, regardless of their semantic content or logical validity. This bias undermines the reliability of reward predictions and leads to overly verbose outputs during inference. To address this issue, we propose CoLD (Counterfactually-Guided Length Debiasing), a unified framework that mitigates length bias based on counterfactual reasoning and causal graph analysis through three components: (1) an explicit length-penalty module, (2) a trainable bias estimator to capture spurious length-related signals, and (3) a joint training strategy that disentangles semantic correctness from superficial length features. Extensive experiments on MATH500 and GSM-Plus show that CoLD consistently reduces reward–length correlation, improves accuracy in step selection, and encourages more concise, logically valid reasoning. These results demonstrate the effectiveness and practicality of CoLD in improving the fidelity and robustness of PRMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the critical issue of 'length bias' in PRMs where verbose reasoning steps are incorrectly favored. The authors introduce CoLD, a novel framework that uses a causal inference perspective to model and mitigate this bias. CoLD employs a dedicated bias estimator and a specialized joint training objective to disentangle the true reward for logical correctness from spurious, length-related artifacts. Extensive experiments on several math reasoning benchmarks demonstrate that the proposed CoLD framework significantly improves Best-of-N style TTS evaluation, offering a robust solution to a key challenge in process reward modeling.

### Strengths
*   **Novel principled causal formulation:** The paper's innovation lies on its reframing of length bias through a causal lens. It formally models step length as a confounder and establishes a counterfactual objective to isolate semantic correctness from spurious correlations. This provides a strong theoretical foundation that elevates the work beyond prior heuristic-based debiasing approaches.

*   **Modular architecture and extensive ablation:** CoLD features a highly modular architecture that enhances its practical utility. The extensive ablation studies compellingly demonstrate that its components.

* **Consistent Improvement**: Experiments show that CoLD consistently achieves test-time scaling (TTS) improvements.

### Weaknesses
* **Extra computational cost**: CoLD introduces substantial additional computational overhead, yet it does not yield compelling performance gains—especially given that the paper evaluates using Best-of-N, a high-inference-cost strategy. This severely limits its practical applicability.

* **Limited generalization**: Results in Tables 1 and 2 show that the method performs more noticeably on larger LLMs, which may be attributed to their inherent reasoning diversity. The marginal improvements on smaller models reveal weak generalization capability.

* **Online test**: Works such as EurusPRM and DeepSeekMath have already explored integrating PRMs with online methods like GRPO and reinforcement learning, and the current paper offers insights into length bias. Why does it not investigate how length bias affects online PRM utilization or the benefits of CoLD in such dynamic settings? Addressing this would significantly enhance CoLD’s practical value, whereas the current approach remains confined to static test-time scaling.

* **Baselines**: More advanced LLMs should be considered: Models such as Qwen3 should be included in the evaluation. For instance, the chosen baselines already exhibit relatively low performance on Math500 under current research standards.

### Questions
Please see weaknesses.

### Soundness
4

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
This paper examines the issue of length bias in PRMs for mathematical reasoning, where longer reasoning steps receive higher reward scores regardless of logical validity. The authors propose CoLD, a framework composed of three parts: a length-penalty term, a learnable bias estimator network, and a joint training procedure with correlation regularization to separate semantic correctness from superficial verbosity. Experiments on MATH500 and GSM-Plus show slightly higher accuracy and shorter generated reasoning steps compared with baseline PRMs.

### Strengths
- The experimental setup is reproducible, and datasets are public.

### Weaknesses
1. The method does not implement true counterfactual inference or causal adjustment; it is a multi-loss regularization scheme.

2. The paper’s central idea "jointly learning a bias estimator with a reward model to remove spurious correlation with response length" appears conceptually similar to several prior studies addressing length bias in reward modeling.

- **Shen et al. (2023)**, *Loose Lips Sink Ships: Mitigating Length Bias in RLHF* (arXiv: [2310.05199](https://arxiv.org/abs/2310.05199)), separates a “length expert” and a “reward expert” under a Product-of-Experts formulation to decouple length bias. CoLD’s “bias estimator + main PRM joint training” follows a related structure, though applied to mathematical reasoning.  

- **Huang et al. (2024)**, *Post-hoc Reward Calibration: A Case Study on Length Bias* (arXiv: [2409.17407](https://arxiv.org/abs/2409.17407)), estimates a length-bias function and subtracts it from the reward, which is mathematically close to CoLD’s formulation  
\[
r^*(x) = r_\theta(x) - c·σ(b_φ(x)) - αℓ(x)
\]  
but achieved without retraining.  

- **Bu et al. (2025)**, *Adaptive Length Bias Mitigation in Reward Models for RLHF* (Findings of ACL NAACL 2025, [PDF](https://aclanthology.org/2025.findings-naacl.169.pdf)), introduces a dynamic penalty for verbosity bias that pursues a similar goal from a simpler perspective.  

  While CoLD extends these ideas to process reward models, the paper does not clearly articulate its methodological distinctions or provide empirical comparisons with these baselines. This potential overlap weakens the perceived novelty and limits the strength of the claimed contribution.

3. No justification of parameterization or stability (Line 825). The hyper-parameters play critical roles but are not theoretically motivated.The paper provides no sensitivity study or guidance for setting them, making reproducibility fragile. A method presented as “causally principled” should not rely on manual tuning to achieve the effect.

4. Limited scope of evaluation. The experimental evaluation is narrow and lacks strong baselines.  
All experiments are conducted only on math-specific reasoning datasets (MATH500 and GSM-Plus), without testing on other reasoning or RLHF-related settings. Moreover, the paper compares CoLD only with simple internal variants and does not include key external baselines. Without such comparisons, it is difficult to assess whether CoLD offers any meaningful improvement over existing methods or if its gains merely result from additional regularization.

### Questions
Refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies and tackles a previously overlooked issue in Process Reward Models (PRMs) for mathematical reasoning—length bias, where longer reasoning steps receive higher rewards regardless of their semantic or logical validity. The authors design controlled experiments to empirically confirm this bias, then propose CoLD (Counterfactually-Guided Length Debiasing), a framework that mitigates length bias via causal analysis and counterfactual reasoning. CoLD integrates three complementary modules:
1. A length-penalty term to explicitly penalize verbosity.
2. A bias estimator that models spurious length-dependent effects.
3. Joint training between the PRM and estimator to disentangle correctness from surface length cues.

Extensive experiments on MATH500 and GSM-Plus datasets, across multiple policy and reward models (e.g., Llama-3-70B, MetaMath-Mistral-7B, Muggle-Math-13B), demonstrate that CoLD substantially reduces reward–length correlation and improves reasoning accuracy, producing shorter yet more correct reasoning trajectories.

### Strengths
1. Novel problem identification: The paper is the first to formally identify and empirically validate length bias in PRMs—a real and impactful issue that undermines reasoning evaluation reliability.

2. Causal-theoretic rigor: The causal graph formulation (S→L→P vs. S→C→P) clearly distinguishes spurious and genuine causal pathways, grounding the proposed framework in principled reasoning.

3. Systematic debiasing design: The three-module CoLD framework elegantly integrates heuristic (penalty), learnable (bias estimator), and structural (joint training) debiasing mechanisms.

4. Strong empirical results: Consistent improvements (+2–4% accuracy, −30% average length) across multiple datasets and models demonstrate robustness and practicality.

5. Practical significance: The framework can be applied post-hoc to existing PRMs without full retraining, which is valuable for real-world RLHF and inference-time alignment systems.

### Weaknesses
1. Limited theoretical novelty: While the combination of counterfactual reasoning and PRM debiasing is novel, each component (penalty, bias estimator, joint training) builds on existing ideas.

2. Lack of interpretability: The paper does not deeply analyze what the Bias Estimator learns or visualize its internal representations.

3. Domain specificity: Experiments are restricted to mathematical reasoning; it is unclear if CoLD generalizes to code reasoning, logical argumentation, or multimodal PRMs.

4. Missing behavioral analysis: The paper evaluates reward quality but not how CoLD impacts policy generation dynamics in RL training (e.g., entropy, verbosity collapse).

5. Presentation density: Some equations (Eq.3–9) are overly packed, and could be better supported with more intuitive diagrams or pseudo-code explanations.

### Questions
1. **Bias Estimator Interpretability**  
   Could you provide more interpretability analysis of the *Bias Estimator*? For example, visualizing attention or gradient sensitivity to input length could clarify whether it truly captures spurious surface-level cues rather than semantic ones.

2. **Cross-domain Generalization**  
   Have you examined whether **CoLD** generalizes beyond mathematical reasoning — e.g., to code reasoning or logical argumentation tasks? Such evidence would strengthen the generality claim.

3. **Counterfactual Sensitivity**  
   In your counterfactual formulation (Eq. 10–13), have you tested sensitivity to the choice of counterfactual reference length \\( \ell^* \\)?

4. **Interaction with RL Fine-tuning**  
   Could you provide insights on how **CoLD** interacts with reinforcement fine-tuning methods (e.g., PPO, GRPO)? Does debiasing reward signals change convergence speed or policy entropy?

5. **Potential Over-penalization**  
   Are there any cases where **CoLD** over-penalizes necessary elaboration, resulting in shorter but incomplete reasoning?

### Soundness
2

### Presentation
4

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
To tackle the verbosity bias of process reward models, this paper proposes to postprocess PRM predictions with lengh penalty and learnt bias estimator.

### Strengths
1. The motivation is clear.
2. The experiment result supports the paper's claim.

### Weaknesses
1. Unclear and insufficient discussion on implementation and design choice. See questions below.
2. Verbosity bias have been observed long time ago in outcome-based reward models and LLM judges. Although this paper tackles PRM, there seems no specific design utilizing the special nature of process modeling.

### Questions
1. Implementation:
- How to add Gaussian noise on tokens (sec 3.3)?
- How exactly do gradient pass through Pearson correlation (eq 4, 5)? Is anything (e.g. inner expectation, standard deviations) detached?
- There are so many hyperparameters (e.g. c, alpha in eq 3, lambda  n eq 6, 7). How to decide them in practice?
2. Design:
- Eq 3 is a complex heuristic. Why choose such a complex design? Can simpler objectives (e.g. a straightforward $r_\theta(x) + b_\phi(x)$) do the trick? 
- Why don't train eq 3 end-to-end?
- Optimizing the bias estimator with Pearson correlation violates Goodhart's law (When a measure becomes a target, it ceases to be a good measure). Is there any comment on this?

### Soundness
2

### Presentation
1

### Contribution
1
