=== CALIBRATION EXAMPLE 28 ===

# Harsh Critic Review
Now I have enough to write a thorough review. Let me compile my analysis.

---

## Section-by-Section Critical Review

### Title & Abstract

The title claims a "robust safety guarantee," which implies a formal, non-trivial theoretical result. The actual guarantees provided (Theorem 4.1: convergence to a stationary point of any smooth SGD objective; Theorem 4.2: parameter-space deviation scales linearly in perturbation magnitude) are entirely standard and do not constitute a "safety guarantee" in any operationally meaningful sense. A guarantee on the *parameter* norm deviating by at most εG/L under data perturbation says nothing about whether the *outputs* of the model remain safe. This discrepancy between the title's promise and what is actually proved is a significant honesty problem.

The abstract states the method "substantially improves safety with only minor degradation in utility." Looking at Table 4, GSM8K on Llama2 degrades from 37.4% (CSFT) to 34.5% (CSFT+PA), a ~7.8% relative drop in reasoning capability. Whether this is "minor" is debatable, but more importantly, the abstract omits that CSFT+PA is consistently *worse* than CSFT on every utility metric — this is a systematic, not incidental, trade-off that deserves upfront acknowledgment.

---

### Introduction & Motivation

The core motivation — that CSFT enforces token-level distributional similarity and thereby "neglects the semantic nature of text generation" — is asserted but not demonstrated. The proposed PA loss also operates purely at the token level (Equation 5 compares log-probabilities of individual tokens). It is unclear in what meaningful sense CSFT+PA captures "semantics" while CSFT does not; both methods are entirely distributional at inference time. The authors should either demonstrate this claim empirically (e.g., showing that CSFT+PA generalizes better to paraphrased harmful prompts not seen in training) or abandon this framing.

The Related Work section contains a notably incomplete paragraph (lines 97–99): *"He analyzes policy instability in RL-trained LLMs via reward-to-policy continuity… Unifies explanations for failures like deceptive reasoning and instruction ignoring."* This reads as a raw citation note rather than integrated scholarship and signals insufficient engagement with the broader safety literature. Methods like SafeLoRA, VaccineBooster, DARE, and others that also address safety-preserving fine-tuning are entirely absent from the related work.

---

### Method / Approach

**PA Loss Formulation (Equation 5):** The PA loss penalizes cases where the model assigns lower probability to the aligned token *y_{t,aligned}* than to its own generated token *y_{t,θ}*. This is conceptually similar to token-level DPO, yet the paper does not carefully discuss this relationship. The token *y_{t,θ}* must be sampled from the current model *during training*, introducing an online generation step that is computationally expensive for LLMs. The paper never acknowledges or quantifies this overhead.

**Adaptive Weight μ_t (Equation 6):** The weight μ_t = KL(π_θ(y_t|x,y_{<t}) ∥ π_aligned(y_t|x,y_{<t})) requires computing the full vocabulary-level KL divergence at every token position for every training example. For modern LLMs with vocabularies of ~32K–128K tokens, this is a significant per-step cost. No mention is made of this computational burden, and no wall-clock training time is reported.

**Circular Dependency:** The adaptive weight μ_t depends on π_θ, and π_θ is being optimized. When computing gradients of L_PA with respect to θ (Equation in Appendix), the authors compute ∇_θ μ_t, which itself requires differentiating through the KL computation. In practice, this is almost certainly detached (stop-gradient), as implied by the analogous treatment of w_t in the CSFT gradient. However, the paper never explicitly states whether μ_t is treated as a fixed constant or is differentiated through — this is a material omission for reproducibility.

**Scheduling Coefficient δ_epoch (Equation 7):** This coefficient increases linearly from 0.1 to 0.3 over training. This is entirely ad hoc — why these values? Why linear? No ablation is provided over this choice (see Experiments section). Setting δ=0.3 means PA contributes at most 30% of the gradient signal. The interaction between this cap and the adaptive weight μ_t is not discussed.

**CSFT Loss (Equation 2):** The paper refers extensively to the CSFT loss from Qi et al. (2024) but Equation 2 is never fully legible due to the PDF parser (the core expression is split across lines 196–210). Based on context, w_t = 2(1 − σ(β_t(log π_θ − log π_aligned))) is a token-level importance weight. The value of β_t is never defined or ablated in the experiments.

---

### Theoretical Results

**Theorem 4.1 (Convergence):** This is a completely standard non-convex SGD convergence result (lim inf_k E[‖∇L‖²] = 0) under bounded gradients, Lipschitz continuity, and the Robbins-Monro learning rate schedule. The proof in the appendix is a textbook application of the descent lemma. This theorem provides no insight specific to safety alignment — *any* smooth bounded loss optimized by SGD satisfies it. The authors' own framing ("the proof follows the standard stochastic optimization framework") confirms this. Calling this a "convergence guarantee" in the context of safety is misleading.

**Theorem 4.2 (Robustness Bound):** This theorem has several critical weaknesses:

1. **The convexity assumption is untenable.** Section 4.2 explicitly assumes L_Total(θ) is convex and satisfies the Polyak-Łojasiewicz inequality. LLMs with billions of parameters are highly non-convex. The authors acknowledge "convexity is a foundational assumption… relaxed in practice for deep learning," but this relaxation invalidates the entire proof, not just the convergence rate.

2. **The "proof" is essentially trivial.** The argument (lines 418–425) shows that ‖∇L(θ*,ε)‖ ≤ εG ≤ L‖θ*,ε − θ*‖, and hence ‖θ*,ε − θ*‖ ≤ εG/L. This is a direct consequence of Lipschitz gradient continuity alone and holds for *any* differentiable loss function under the stated assumptions. PA is not used in the proof itself — the sentence "The inclusion of L_PA(θ) further improves robustness by reducing effective sensitivity to noise" is stated verbally but never formalized or quantified. There is no derivation of how PA changes the constants G or L.

3. **The robustness metric is wrong for the stated goal.** Measuring robustness in parameter space (‖θ*,ε − θ*‖) says nothing about output-level safety. Two parameter vectors close in Euclidean norm can produce wildly different output distributions in high-dimensional softmax settings. The relevant robustness notion for LLM safety is behavior-level stability (e.g., ASR under distributional shift), not parameter-space proximity.

In short, the theoretical section provides standard boilerplate convergence analysis that would apply to any smooth training objective, and a robustness proof whose key claim about PA's benefit is unproven.

---

### Experiments & Results

**Pre-Experiments (Sections 5.1 and 5.2):** Both pre-experiments use toy LSTM models (not LLMs) with synthetic tasks to demonstrate that PA reduces KL divergence and token probability difference. These experiments are not well-motivated: (a) the finding that minimizing PA loss reduces distributional divergence is tautological — that is what the loss is designed to do; (b) LSTM alignment dynamics are not representative of transformer-based LLMs; (c) there is no claim or demonstration that smaller distributional divergence in these toy settings implies improved safety robustness in real fine-tuning scenarios. These experiments feel like padding rather than genuine validation.

**Main Safety Evaluation (Table 3):** The results are the paper's strongest contribution. Reductions in ASR are significant in several conditions (e.g., harmful examples: from 3.6% to 2.7% on Llama2; trigger-based backdoor: from 6.9% to 3.3% on Llama2). However:

- For the identity-shifting attack on Gemma1.1, CSFT already achieves 9.1% ASR, and CSFT+PA achieves 8.8% — a reduction of just 0.3 percentage points. This is presented as "3.3% relative improvement" using misleading relative framing.
- For the trigger-free backdoor on Llama2, CSFT achieves 1.9% ASR and CSFT+PA achieves 1.5% — again a marginal absolute improvement.
- **No variance, standard deviation, or statistical significance is reported anywhere.** With differences this small (0.3pp), without error bars it is impossible to determine whether these improvements are meaningful.
- Only two models are tested (Llama2-7B-Chat, Gemma1.1). No results with larger models or more recent architectures (e.g., Llama-3, Mistral).
- No comparison is made against recent safety-preserving fine-tuning baselines beyond SFT and CSFT: SafeLoRA, VaccineBooster, RepNoise, DARE, and related methods from 2023–2024 are absent.

**Utility Evaluation (Table 4):** CSFT+PA is uniformly worse than CSFT on all utility benchmarks. The 7.8% relative GSM8K drop on Llama2 (37.4→34.5) suggests non-trivial reasoning degradation. The claim that drops are "within 8% and acceptable" is the authors' editorial judgment, not a principled threshold.

**Missing Ablations:**
- No ablation over δ_epoch scheduling (linear vs constant vs other schedules, or varying the 0.1–0.3 range).
- No ablation over the weight β_t in the CSFT loss.
- No ablation over training data composition or poisoning rate.
- No analysis of what happens as the proportion of adversarial examples increases.

**Hyperparameter Transparency:** The experimental section does not report learning rates, batch sizes, number of training epochs, LoRA rank (if LoRA is used), or the size of the safety/attack dataset. Reproducibility is severely hampered.

---

### Writing & Clarity

Section 4.2 introduces the convexity assumption prominently but then quietly walks it back ("convexity is… relaxed in practice for deep learning") while still using it as the basis for the formal theorem. This is intellectually inconsistent. The reader is left uncertain whether the theorem holds for the actual experimental setting.

The Related Work paragraph about Xu (2025) (lines 97–99) is incoherent — it reads as if the authors copied bullet points from a citation rather than writing prose.

Table 1 (comparison of methods on distributional vs preference alignment) has a checkmark format that is never filled in the parsed version, but is conceptually too simple: it reduces the comparison to two binary attributes, ignoring the nature, strength, and cost of each alignment mechanism.

---

### Limitations & Broader Impact

The paper has no explicit Limitations section. Key unacknowledged limitations:
1. **Computational cost** of online token sampling and KL computation during training.
2. **Scope of safety evaluation**: ASR is measured only against the specific attack datasets from Qi et al. (2024). Jailbreaking via prompt engineering, red-teaming, or adversarial suffixes is not evaluated.
3. **Scalability**: All experiments use 7B-parameter models. The approach's effectiveness at larger scales is unknown.
4. **Generalizability**: Only two utility benchmarks (one summarization, one structured output, one math) are tested. Safety-utility trade-offs may differ substantially in other domains.
5. **The "safe outputs" assumption**: The PA loss assumes the reference model (π_aligned) produces safe outputs for any prompt in the fine-tuning dataset. If the reference model also fails on some prompts, the PA loss provides misleading supervision.

---

### Overall Assessment

This paper addresses a genuine and important problem — maintaining safety alignment under domain-specific fine-tuning — and the core experimental observation (that adding a token-level preference loss on top of CSFT reduces ASR across several attack categories) has practical interest. However, the paper has three major problems that, in combination, likely place it below the ICLR acceptance bar in its current form. First, the theoretical contributions (Sections 4.1 and 4.2) are either standard boilerplate (convergence) or formally problematic (robustness under a convexity assumption that is explicitly acknowledged to be unrealistic for LLMs), and the claim that PA *specifically* improves the robustness bound is never quantitatively established. Second, the experimental evaluation lacks statistical rigor (no error bars), is missing important recent baselines, omits key ablations on architectural choices, and fails to report sufficient implementation details for reproducibility. Third, the framing throughout overstates the contribution: "robust safety guarantee" in the title, "semantics-aware" motivation that isn't supported by the token-level formulation, and "minor utility degradation" for a consistent and measurable drop on every benchmark. To be competitive at ICLR, the authors would need to: ground the robustness claims empirically rather than via an inapplicable convexity argument, add comparisons against the broader safety-preserving fine-tuning literature, report confidence intervals, provide full ablations, and tone down the theoretical claims to match what is actually proved.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **CSFT+PA**, a training framework that combines Constrained Supervised Fine-Tuning (CSFT) with Preference Alignment (PA) to enhance the robustness of Large Language Models (LLMs) against adversarial fine-tuning attacks. The method introduces an adaptive loss term that biases the model toward safe outputs from a pre-trained reference using token-level preference signals, rather than relying solely on distributional similarity. Experiments demonstrate substantial reductions in Attack Success Rates (ASR) across multiple attack vectors while maintaining acceptable utility levels on downstream tasks.

### Strengths
1.  **Addressing a Critical Safety Problem:** The paper tackles the specific and timely issue of safety alignment degradation during domain fine-tuning. By explicitly testing against Harmful Example, Identity Shifting, and Backdoor Poisoning attacks, it provides relevant evaluations for current deployment concerns (Table 3).
2.  **Strong Empirical Safety Gains:** The results show significant improvements in safety robustness. For instance, on Gemma1.1, the ASR for Harmful Example attacks drops from 1.3% (CSFT) to 0.6% (CSFT+PA), and Backdoor Poisoning ASR reduces from 1.9% to 0.9% (Table 3).
3.  **Balanced Safety-Utility Evaluation:** Unlike many safety papers that focus solely on refusal metrics, this work explicitly evaluates downstream task performance (Samsum, SQL, GSM8K). The trade-off analysis in Table 4 demonstrates that safety gains do not come at a prohibitively high cost to utility (e.g., <8% relative drop on most tasks).
4.  **Theoretical Analysis:** The inclusion of convergence and robustness analyses (Section 4 and Appendix A) aims to ground the practical method in optimization theory, providing a formal perspective on why the loss function behaves stably under perturbations.

### Weaknesses
1.  **Theoretical Assumptions vs. Deep Learning Reality:** The robustness proofs (Theorem 4.2, Section A.2.5) rely heavily on assumptions of **convexity** and the **Polyak-Łojasiewicz (PL) inequality** (Assumption in Section 4.2). These conditions are generally not satisfied by deep neural networks with overparameterized Transformers. Claiming "robustness guarantees" based on these assumptions may mislead readers about the practical validity of the bounds in the LLM regime.
2.  **Limited Baselines:** The experimental comparison focuses primarily on SFT and CSFT. It lacks comparison with other alignment strategies designed for robustness, such as DPO (Direct Preference Optimization) specifically applied to safety, or RLHF-based fine-tuning. This makes it difficult to assess whether the combination of CSFT + PA offers unique advantages over standard preference optimization for safety.
3.  **Pre-Experiment Model Mismatch:** Pre-experiments 1 and 2 utilize **LSTM** architectures to validate the loss function components, whereas the main experiments use LLMs (Transformers). The optimization landscape and gradient dynamics of LSTMs differ significantly from Transformers; validating the loss efficacy on LLMs requires more direct evidence beyond the theoretical argument.
4.  **Scheduling Mechanism Justification:** The scheduling coefficient $\delta_{epoch}$ (Equation 7) increases linearly to a cap of 0.3. The paper does not provide an ablation study or theoretical justification for this specific linear schedule versus other decay schedules or dynamic balancing methods used in multi-objective optimization, leaving the hyperparameter choice somewhat heuristic.

### Novelty & Significance
**Novelty:** The core contribution lies in the specific formulation of the **Preference-Augmented CSFT** loss function. While using preference data (DPO-style) and distributional constraints (CSFT-style) separately is known, combining them into a unified, dynamically weighted loss for *adversarial robustness* rather than just performance alignment is a distinct and meaningful approach. The design of the adaptive weight $\mu_t$ based on KL divergence adds a layer of technical innovation.

**Significance:** Safety alignment that withstands domain fine-tuning is a high-priority goal for the community, as it addresses the "forgetting" of safety instructions (Qi et al., 2024). Demonstrating a method that significantly lowers ASR while preserving utility has direct relevance for deploying LLMs in sensitive downstream tasks.

### Suggestions for Improvement
1.  **Refine Theoretical Claims:** Revise the Robustness Analysis to explicitly acknowledge that the convexity/PL assumptions are idealizations. Instead of claiming a "Robustness Bound" for the model, frame it as a "Robustness Guarantee for the Objective Function under Convex Approximations." This maintains rigor without overpromising on deep learning behavior.
2.  **Expand Baselines:** Include a comparison with **DPO** fine-tuned for safety robustness. Even a negative result (DPO performing similarly but with less overhead) would strengthen the paper by clarifying the unique value proposition of the proposed method (e.g., computational efficiency or stability vs. RL/HF).
3.  **Clarify Pre-Experiment Relevance:** Either expand the pre-experiments to use Transformer architectures to better mirror the main task, or provide a stronger justification in the text for why LSTM dynamics are sufficient to validate the loss function's property regarding distributional alignment in Transformers.
4.  **Ablation on Scheduling:** Conduct an ablation study on the scheduling strategy for $\delta_{epoch}$ (e.g., constant weight vs. linear increase vs. cosine annealing) to empirically show that the proposed schedule is optimal for safety-utility trade-offs.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add DPO/RLHF Baselines:** Compare against Direct Preference Optimization (DPO) and RLHF, as claiming "preference alignment" benefits without comparing to standard preference methods undermines the novelty and efficacy claims.
2. **Ablate Loss Components:** Remove the adaptive weight $\mu_t$ and scheduling coefficient $\delta_{epoch}$ individually to prove these specific mechanisms drive performance rather than generic regularization.
3. **Measure False Refusal Rates:** Evaluate refusal rates on benign datasets (e.g., AlpacaEval) because standard utility benchmarks like GSM8K do not capture safety-induced over-refusal on safe prompts.
4. **Use Standardized Safety Benchmarks:** Replace vague "Harmful Example Attacks" with established benchmarks like AdvBench or SafeBench to ensure results are reproducible and comparable to SOTA safety research.

### Deeper Analysis Needed (top 3-5 only)
1. **Justify Convexity Assumptions:** The robustness proof relies on convexity and PL inequalities on non-convex LLM landscapes; provide empirical verification of these conditions or explicitly frame them as theoretical approximations.
2. **Clarify Preference Pair Construction:** Explicitly analyze how $y_{aligned}$ is generated versus $y_{\theta}$ during training, as the current description blurs the line between this method and standard knowledge distillation.
3. **Plot Safety-Utility Pareto Frontier:** Show the trade-off curve between Attack Success Rate and Utility across varying hyperparameters rather than single points to prove the balance is genuinely superior.

### Visualizations & Case Studies
1. **Qualitative Failure Cases:** Provide side-by-side output examples showing where the model incorrectly refuses benign queries or fails to reject harmful ones under adversarial fine-tuning.
2. **Training Loss Dynamics:** Plot the convergence curves of $L_{CSFT}$ and $L_{PA}$ separately to verify that the scheduling coefficient $\delta_{epoch}$ modulates the loss contributions as theorized.
3. **Token Probability Shifts:** Visualize the probability distribution changes for specific unsafe tokens before and after fine-tuning to demonstrate the mechanistic effect of the preference loss.

### Obvious Next Steps
1. **Report Compute Overhead:** Quantify the additional training time and memory costs introduced by the online preference generation compared to standard CSFT to assess practical viability.
2. **Test Reference Model Robustness:** Evaluate performance when the safety-aligned reference model is weak or corrupted to verify if the method relies too heavily on reference quality.
3. **Hyperparameter Sensitivity Analysis:** Analyze how safety and utility metrics vary with changes to the scheduling coefficient and adaptive weight bounds to guide future implementation.

# Final Consolidated Review
## Summary

This paper proposes CSFT+PA, a framework combining Constrained Supervised Fine-Tuning with Preference Alignment to improve LLM safety robustness against adversarial fine-tuning attacks. The method adds a preference-based loss that adaptively biases the fine-tuned model toward safe outputs from a reference model, using token-level probability comparisons weighted by KL divergence. Experiments on Llama2-7B and Gemma1.1 demonstrate substantial reductions in Attack Success Rate across harmful example, identity shifting, and backdoor poisoning attacks, with modest utility degradation.

## Strengths

- **Strong empirical safety improvements:** Table 3 shows consistent ASR reductions across multiple attack types on both Llama2 and Gemma1.1. For trigger-based backdoor attacks, ASR drops from 6.9% to 3.3% on Llama2 (52.3% relative improvement) and from 1.9% to 0.9% on Gemma1.1 (52.6% relative improvement). The Harmful Example attack reductions are also substantial (88.9%→2.7% on Llama2, 81.6%→0.6% on Gemma1.1).

- **Systematic evaluation across attack categories:** The paper evaluates on three distinct attack types (Harmful Example, Identity Shifting, Backdoor Poisoning with and without triggers) plus utility benchmarks, providing a more comprehensive safety-utility trade-off analysis than typical single-metric safety papers.

- **Novel loss formulation with adaptive weighting:** The PA loss (Equation 5) combined with the adaptive weight μ_t based on KL divergence (Equation 6) represents a technically interesting approach to dynamically emphasizing alignment where the model most deviates from the reference distribution.

## Weaknesses

- **Theoretical contributions are weak and mismatched to claims:** Theorem 4.1 is a standard SGD convergence result (lim inf of gradient norm equals zero) that applies to any smooth loss function—nothing specific to safety. Theorem 4.2 assumes convexity and the PL inequality, which the paper acknowledges "is relaxed in practice for deep learning," undermining the theorem's applicability to actual LLM training. Critically, the robustness bound measures parameter-space deviation (‖θ* − θ*_ε‖), not output-space safety—two parameter vectors close in Euclidean norm can produce very different behaviors in high-dimensional softmax models. The claim that PA "further improves robustness by reducing effective sensitivity to noise" in the robustness analysis is stated verbally but never formally quantified.

- **"Semantics-aware" motivation is unsubstantiated:** The introduction claims PA addresses the limitation that CSFT "neglects the semantic nature of text generation." However, Equation 5 shows PA operates purely at the token level (comparing log-probabilities of individual tokens), just like CSFT. Both methods are distributional at inference time. There is no empirical demonstration (e.g., generalization to paraphrased harmful prompts) that PA captures semantics any better than CSFT.

- **Missing statistical significance and insufficient experimental rigor:** No error bars, confidence intervals, or multiple-run statistics are reported. With absolute improvements as small as 0.3 percentage points in some conditions (Identity Shifting on Gemma1.1: 9.1%→8.8%), statistical significance cannot be assessed. Hyperparameters (learning rates, batch sizes, epochs, LoRA rank) are not reported, hampering reproducibility.

- **Pre-experiments do not validate LLM behavior:** Sections 5.1 and 5.2 use toy LSTM models to demonstrate PA reduces KL divergence, but LSTM optimization dynamics differ substantially from Transformers. Demonstrating that PA reduces distributional divergence is tautological—this is what the loss is designed to do—and says nothing about safety generalization in actual LLMs.

- **Missing baselines and ablations:** No comparison with other safety-preserving fine-tuning methods (SafeLoRA, VaccineBooster, DARE, DPO for safety) beyond SFT and CSFT. No ablation on the scheduling coefficient δ_epoch (why 0.1 to 0.3? why linear?), the β_t parameter in CSFT loss, or the interaction between μ_t and δ_epoch.

- **Computational overhead unquantified:** Equation 6 requires computing full vocabulary-level KL divergence at every token position for every training example (vocabularies of 32K–128K tokens). The gradient of μ_t is computed (not detached) in Appendix A.1.3, adding computational cost. No wall-clock training time or memory overhead is reported.

## Nice-to-Haves

- Comparison with DPO/RLHF-based safety alignment baselines would clarify the unique value of the CSFT+PA combination.

- Evaluation of false refusal rates on benign datasets (e.g., standard instruction-following benchmarks) to capture safety-induced over-refusal.

- Testing on larger models (beyond 7B parameters) to assess scalability.

- Ablation on δ_epoch scheduling strategy (linear vs. constant vs. other schedules).

- Report compute overhead and training time compared to standard CSFT.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Related work incompleteness claims:** The harsh critic claimed SafeLoRA, VaccineBooster, and DARE are absent from related work. Without external verification, I cannot confirm these methods exist or are relevant to this paper's scope. The paper does cite relevant work (Qi et al. 2024, Xu 2025, Rafailov et al. 2023).

- **Overstatement about utility degradation framing:** While the paper's "minor degradation" claim is subjective, the 7.8% relative GSM8K drop is explicitly reported in Table 4, and the paper acknowledges the trade-off. The abstract's framing may be optimistic but is not factually incorrect.

- **Formatting nitpicks about Equation 2 parsing:** The harsh critic mentioned the equation was not fully legible in the parser—this is explicitly noted as a parser artifact, not a paper problem.

## Novel Insights

The adaptive weighting mechanism μ_t = D_KL(π_θ ∥ π_aligned) is a clever design: it automatically up-weights the PA loss contribution when the model has drifted further from the reference distribution, creating a self-correcting feedback loop. This contrasts with static regularization approaches and could inspire similar adaptive mechanisms in other alignment contexts. The token-level preference formulation (Equation 5) resembles DPO but applied at the token level rather than sequence level, which may be better suited for fine-grained safety control—though this relationship is not discussed in the paper.

## Suggestions

1. **Ground theoretical claims empirically:** Either remove the convexity-based robustness theorem or explicitly frame it as "if the loss landscape were approximately convex locally" and provide empirical verification of PL-like conditions during actual training.

2. **Add statistical significance:** Report mean ± standard deviation over at least 3-5 random seeds for all ASR and utility metrics.

3. **Include at least one safety-preservation baseline beyond SFT/CSFT:** Comparing against DPO with safety preference data or another recent method would establish the method's value relative to the broader literature.

4. **Report complete hyperparameters:** Learning rates, batch sizes, number of epochs, whether LoRA is used and at what rank, and dataset sizes are essential for reproducibility.

5. **Conduct key ablations:** Minimally, ablate the δ_epoch schedule and the adaptive weight μ_t to show these mechanisms contribute meaningfully beyond simpler alternatives.

# Actual Human Scores
Individual reviewer scores: [0.0, 4.0, 4.0, 2.0]
Average score: 2.5
Binary outcome: Reject
