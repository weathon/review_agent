# ICLR Benchmark Results

Date: 2026-04-05 02:37
Critic/Merger: qwen/qwen3.6-plus:free (OpenRouter)
Neutral: qwen/qwen3.6-plus:free, Related Work: qwen/qwen3.6-plus:free:online (OpenRouter)

## B6HnApgkP3

- GT: Reject (avg 2.0)
- Predicted: N/A (2.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a post-hoc adaptation of the embedded Feature Selection Layer (FSL) for interpreting frozen, pre-trained neural networks on tabular data. By inserting a trainable 1:1 dense scaling layer before the fixed backbone and optimizing it via backpropagation, the method learns non-negative feature relevance weights without altering the original model parameters. The authors evaluate this approach against standard gradient/perturbation-based attribution methods across synthetic and real-world high-dimensional, low-sample-size (HDLSS) datasets using predictive performance, feature selection precision/recall (PIFS/PSFI), stability metrics, and weighted t-SNE visualizations.

## Strengths
- **Practical, non-invasive design for deployed models:** The method explicitly addresses the need to generate feature attributions for frozen networks without retraining or modifying the base architecture (Section 1, 3.1). This constraint is highly relevant for real-world pipelines where model parameters cannot be altered post-deployment.
- **Rigorous empirical design across diverse regimes:** The evaluation systematically contrasts synthetic datasets with known ground truth (XOR, SynthA) against challenging real-world HDLSS benchmarks (Liver, Breast, Spam). Reporting cross-validated stability (Jaccard, Spearman, Pearson), feature selection recovery (PIFS/PSFI), and predictive metrics across multiple folds demonstrates a thorough experimental framework (Tables 1–6, 8–9).
- **Transparent acknowledgment of limitations:** The authors clearly delineate the method's boundaries, explicitly noting its instability in multi-class settings and restriction to fixed-meaning tabular features (Section 6.1). This conceptual awareness appropriately scopes the contribution and strengthens the paper's credibility.

## Weaknesses
- **Over-reliance on weighted t-SNE and silhouette scores for interpretability validation:** The paper heavily uses silhouette coefficients on 2D t-SNE projections as primary evidence of "visual and clustering-based interpretability." t-SNE is a stochastic, non-linear dimensionality reduction technique that distorts global distances; tight clustering in a 2D t-SNE embedding measures projection quality, not attribution faithfulness or recovery of the model's true decision logic. Without standard XAI fidelity metrics (e.g., deletion/insertion curves, ROAR, or perturbation-based faithfulness), the claim that the learned weights meaningfully interpret the frozen model remains empirically unsubstantiated.
- **Under-specified training objective and reproducibility gaps:** Section 3.2 outlines the forward pass but omits the explicit loss function formulation (whether training uses ground-truth labels or distills the frozen model's logits) and critical optimization hyperparameters (optimizer, learning rate schedule, epoch count, and how the L1 regularization strength λ is tuned). Section 3.3 also lacks empirical justification for changing initialization from 1/n to 1.0. These omissions hinder exact replication and obscure whether stability variance stems from the optimization setup or the method itself.
- **Stability trade-offs are reported but under-analyzed:** The post-hoc FSL consistently underperforms other methods on Spearman and Pearson stability metrics across real-world datasets (Tables 4, 9), yet the paper largely dismisses this as a secondary caveat. In interpretability research, unstable feature rankings directly undermine trustworthiness, especially in the high-stakes medical domains cited in the introduction. The paper lacks analysis of *why* the optimization yields unstable masks (e.g., identifiability issues on frozen non-linear backbones, sensitivity to initialization) or explores mitigation strategies.
- **Positioning lacks differentiation from standard linear probing/adapters:** The proposed mechanism is functionally equivalent to training a single-layer adapter or linear probing head on a frozen backbone. The paper does not situate the work within this established literature or demonstrate what theoretically or empirically distinguishes the FSL’s ReLU+L1 constraints from training a standard linear explainer head. Without this contextualization, the technical novelty appears incremental.

## Nice-to-Haves
- Provide a compute/runtime comparison against closed-form attribution methods (e.g., Integrated Gradients, SHAP) to quantify whether the optimization-based overhead justifies its use in practice.
- Include ablation studies on the L1 regularization strength (λ), ReLU activation, and weight initialization to empirically validate their impact on stability and feature recovery.
- Test the method across multiple base architectures (e.g., MLPs of varying depth/width) to verify that the learned weights generalize beyond a single unspecified backbone.

## Novel Insights
The reviews converge on a critical tension in post-hoc interpretability research: the paper successfully demonstrates that fine-tuning an input scaling layer is computationally feasible and preserves predictive performance, but conflates geometric class separability with attribution correctness. The core missing link is a rigorous analysis of *optimization dynamics on frozen backbones*. When a scalar weight vector is trained against a fixed, highly non-linear function, the loss landscape likely contains many equivalent minima that preserve accuracy but yield wildly different feature masks. Exploring this identifiability problem—perhaps through loss landscape analysis, stability regularization, or connections to linear probing theory—would transform the work from an applied adaptation into a substantive contribution to XAI optimization.

## Suggestions
- Replace or supplement t-SNE/silhouette evaluation with standard attribution faithfulness benchmarks (e.g., MOFO, ROAR, deletion-insertion AUC) to objectively measure whether high-weight features actually drive model predictions.
- Explicitly state the loss function used for post-hoc training and provide a detailed reproducibility appendix covering optimizer choice, learning rates, training epochs, λ selection protocol, and base network architectures.
- Compare post-hoc FSL directly to a standard linear probing baseline (training a simple linear layer without ReLU/L1 constraints) to isolate whether the FSL's architectural choices provide measurable advantages in feature recovery or stability.
- Analyze the reported stability degradation by investigating whether adding a consistency regularization term (e.g., penalizing weight variance across perturbed mini-batches) or ensembling multiple post-hoc runs can mitigate ranking volatility without sacrificing predictive fidelity.

---

## pPa5mIFj5p

- GT: Reject (avg 2.5)
- Predicted: N/A (3.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes Entropy-Aware Speculative Decoding (EASD), a training-free modification to standard speculative decoding (SD) that introduces a dynamic penalty during the verification step. When both draft and target models exhibit high Shannon entropy and substantial top-𝑛 token overlap, EASD zeros out the draft's top token in the target distribution and renormalizes, forcing alternative sampling. Evaluated across six challenging reasoning benchmarks with Qwen2.5 model pairs, EASD consistently improves average accuracy over standard SD and reward-guided baselines, occasionally exceeding the target model's greedy performance while claiming comparable inference efficiency.

## Strengths
- **Lightweight, training-free SD enhancement:** EASD requires no auxiliary models, fine-tuning, or architectural changes. It leverages only the native probability outputs of the draft and target models, making it highly practical for real-world deployment where compute budgets are tight.
- **Consistent empirical improvements across diverse reasoning tasks:** Across 32B and 72B target scales, EASD achieves stable average gains (+3.5 to +3.7 over single models) on highly structured benchmarks like AIME24, GPQA-Diamond, and OlympiadBench, demonstrating robustness to task domain.
- **Transparent experimental analysis:** The paper provides thorough ablations isolating draft entropy and overlap contributions, hyperparameter sensitivity sweeps, and honest failure case studies. The data-driven threshold calibration procedure and clear pseudocode enhance reproducibility and methodological transparency.

## Weaknesses
- **Critical inconsistency between temperature settings and entropy computation:** Section 4.1 explicitly states that EASD (like SD/RSD) is run with `temperature = 0` and `top-p = 1`. At $T=0$, the output distribution becomes nearly deterministic (effectively a delta function), yielding Shannon entropy close to zero. Yet EASD's penalty trigger (Eq. 4) requires $H_t, H_d > \tau_H \approx 2$, and Appendix E shows $\tau_H=2$ as a default. The paper does not clarify whether entropy is computed on raw logits before temperature application, on a separate $T>0$ pass, or if the implementation actually uses $T>0$. This is a fundamental reproducibility and logical gap that must be resolved.
- **Deliberate violation of exact sampling guarantees without explicit framing:** Standard SD preserves the exact target distribution by construction. EASD explicitly modifies $P_t$ via zeroing and renormalization (Eq. 6-7) to enable "surpassing" the target model. While this is an intentional design choice for accuracy, the paper fails to explicitly position EASD as a *distribution-shifting decoding strategy* rather than a strict acceleration framework. The trade-off between breaking exact target alignment and gaining reasoning accuracy should be foregrounded, not implied.
- **Contradictory efficiency claims and mismatched baseline comparisons:** The abstract claims efficiency is "proven" comparable to SD, but Appendix D Table 5 reveals EASD actually runs slower than the single-model baseline on GPQA (13 tok/s vs. 14 tok/s). High penalty rates under domain uncertainty can negate SD's parallelization advantage. Additionally, comparing greedy EASD ($T=0$) against Majority Voting and Beam Search ($T=0.7$, $N=16$) conflates accuracy gains from distributional steering with those from explicit test-time compute scaling. A compute-matched baseline or explicit acknowledgment of regime differences is necessary.
- **Unverified core assumption regarding the trigger heuristic:** The method assumes that "high entropy + high top-𝑛 overlap" reliably indicates an error-prone decision point prone to low-confidence propagation. However, no empirical analysis correlates trigger activation with actual token-level correction rates versus false penalty rates. High overlap typically signals model agreement; penalizing consensus during uncertainty may disrupt valid reasoning paths as often as it prevents errors. Without precision/recall analysis of the trigger, the mechanism remains an unverified heuristic.

## Nice-to-Haves
- **Statistical significance reporting:** Report mean ± variance over multiple random seeds or bootstrap resampling to demonstrate that the ~1–3% average gains are statistically robust and not within standard benchmark variance.
- **Cross-architecture draft/target pairs:** Validate the entropy-overlap heuristic on structurally dissimilar model pairs (e.g., LLaMA-3 drafting for Mixtral) to ensure the trigger isn't exploiting family-specific token distribution similarities.
- **Dynamic/online threshold adaptation:** Explore rolling entropy percentiles or domain-agnostic scaling for $\tau_H$ and $\tau_O$ to reduce reliance on matched validation calibration and improve zero-shot plug-and-play utility.

## Novel Insights
EASD reframes speculative decoding from a passive acceleration pipeline that strictly mimics a target distribution into an active collaborative decoding framework. The observation that "high agreement under high uncertainty" often signals a latent reasoning trap rather than model consensus is a conceptually sharp inversion of standard decoding intuition. By deliberately injecting controlled distributional shift at ambiguous decision points, the method demonstrates that lightweight, inference-time steering can escape local optima in long-horizon reasoning without the overhead of trained reward models or architectural changes.

## Suggestions
- Clarify exactly where and how Shannon entropy is computed relative to the temperature=0 setting. If entropy is calculated on pre-temperature logits or a separate sampling pass, explicitly state this in Section 4.1 and Eq. 2-3.
- Replace the overclaim "prove that the efficiency of EASD is comparable to that of SD" in the Abstract and Section 1 with "empirically demonstrate" or "maintain", and explicitly discuss the efficiency regression on high-uncertainty tasks like GPQA in the main text or limitations.
- Add a dedicated paragraph or subsection analyzing the precision of the penalty trigger: report the percentage of penalized tokens that led to correct downstream answers vs. those that disrupted valid trajectories, directly testing the assumption that high entropy + high overlap correlates with error propagation.

---

## HGdg76iVo6

- GT: Reject (avg 4.5)
- Predicted: N/A (5.7/10)
- Match: N/A

### Final Review

## Summary
This paper proposes RLKV, a reinforcement learning-based method to identify "reasoning heads" in LLMs that require full KV cache for chain-of-thought (CoT) consistency, while compressing the remaining heads. By optimizing lightweight gating adapters with GRPO, an L1 sparsity penalty, and two stabilization techniques (curriculum sampling and adaptive penalty weighting), the method achieves 20–50% KV cache reduction with near-lossless performance. The approach consistently outperforms existing token-dropping and head-reallocation baselines across multiple reasoning models and benchmarks, while introducing a more faithful dynamic-budget evaluation protocol for variable-length reasoning tasks.

## Strengths
- **Rigorous motivation and problem framing**: The controlled comparison between instruct and reasoning model variants effectively isolates extended CoT generation as the primary driver of compression failure, distinguishing algorithmic degradation from model capability differences. The empirical distinction between token-dropping (causing repetitive loops) and head-reallocation (causing over-extended failures) clearly motivates the need for reasoning-aware head identification.
- **Sound RL formulation with robust stabilization**: Reducing the optimization space to $L \times H$ gating adapters while freezing model weights makes RL-based head identification tractable. The authors correctly identify and resolve the sparse reward vs. dense penalty conflict, and the ablation studies concretely demonstrate that curriculum sampling and reward-adaptive penalty weighting are necessary to prevent training collapse.
- **Methodologically rigorous evaluation protocol**: The paper proactively addresses a known flaw in prior KV cache work by converting fixed-budget baselines to dynamic budgets proportional to sequence length. This adjustment yields a fairer comparison for reasoning tasks with highly variable output lengths and is thoroughly justified in the text and appendix, preventing artificially inflated baseline performance.

## Weaknesses
- **Lack of statistical validation for strong empirical claims**: The paper reports that RLKV occasionally surpasses the full KV cache baseline (e.g., on AIME24) and hypothesizes that compressing non-reasoning heads removes noise. Given the stochastic nature of RL training and the sensitivity of head identification to reward variance, the evaluation reflects a single training run without seed variance or confidence intervals. It remains unclear whether this improvement is statistically significant or within the natural performance noise of the benchmark.
- **Unquantified computational trade-off and missing heuristic baselines**: While adapter tuning is parameter-efficient, the identification phase requires 22–40 GPU-hours on A100s for repeated CoT sampling. The paper does not compare this cost against lighter, single-pass proxy methods for head selection (e.g., attention magnitude aggregation, activation variance, or gradient-based importance), nor does it provide a breakeven analysis showing how many inference calls are needed to offset the upfront RL training cost.
- **Static, model-level head masking limits domain adaptability**: RLKV learns a single $\alpha$ mask per model applied uniformly across all inputs. However, the appendix MMLU-Pro results show notable degradation on Law and Physics even at low sparsity, suggesting that "reasoning heads" are likely domain- or task-dependent rather than universally critical. The lack of conditional or step-adaptive routing restricts the method's applicability to more diverse or dynamic reasoning workloads.

## Nice-to-Haves
- Evaluate multi-request continuous batching in modern serving frameworks (e.g., vLLM or SGLang) to demonstrate the true throughput and memory-scaling benefits of compression, rather than relying on fixed-batch PyTorch latency measurements.
- Provide mechanistic visualizations or analyses of the identified reasoning heads (e.g., layer-wise $\alpha$ heatmaps, attention distance distributions, or targeted perturbation studies) to characterize their functional roles beyond compression sensitivity.
- Investigate cross-domain transferability by identifying heads on one domain (e.g., code) and evaluating compression performance on another (e.g., math), to test whether the method captures general reasoning circuitry or dataset-specific artifacts.
- Explore the interaction between RLKV and low-bit KV cache quantization to assess whether the two compression paradigms are complementary or if numerical errors compound under aggressive memory reduction.

## Novel Insights
The paper effectively repurposes RLVR—not for generation policy optimization, but for structural discovery of functional attention circuitry. By framing head identification as a sparse resource allocation problem guided by verifiable reasoning rewards, it demonstrates that only a small, stable subset of KV heads is essential for maintaining CoT consistency. This challenges the implicit assumption that all heads contribute equally to long-context retention and provides empirical evidence that strategically compressing non-essential heads can sometimes reduce interference, offering a new perspective on the heterogeneous, functionally specialized nature of reasoning pathways in modern LLMs.

## Suggestions
- Run the RLKV training pipeline across 3–5 different random seeds and report mean performance with standard deviation on AIME24 and Math500. Explicitly test whether the "surpassing full KV cache" claim persists across seeds or falls within the standard deviation of the uncompressed baseline.
- Implement a lightweight, single-pass baseline for head selection (e.g., cumulative attention score magnitude or activation variance over a small calibration set) and report its sparsity-performance curve alongside RLKV. Include a brief cost-benefit analysis quantifying the inference-volume breakeven point where RL training overhead is justified by memory/throughput gains.

---

## bhPaXhWVKG

- GT: Reject (avg 5.3)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
MermaidFlow introduces a declarative, graph-based intermediate representation (using the Mermaid DSL) for agentic workflow planning, paired with a safety-constrained evolutionary programming framework to search for optimal multi-agent structures. By enforcing type and role consistency during graph mutations and leveraging static compilation checks, the method decouples planning from execution, achieving higher valid-generation rates, improved token efficiency, and consistent performance gains over code-centric workflow synthesis baselines across standard math and code reasoning benchmarks.

## Strengths
- **Well-motivated abstraction that addresses a real bottleneck:** The shift from imperative Python/JSON workflows to a statically typed Mermaid graph cleanly separates structural planning from low-level execution. This directly mitigates the brittleness and entangled semantics that plague existing code-centric generation methods (Sec 1 & 2).
- **High search validity and efficiency via constraint-preserving operators:** The defined EP operators (substitution, insertion, rewiring, crossover) are explicitly designed to preserve type and interface compatibility. Coupled with soft/hard validation checks, this yields >90% success in generating executable Python code, significantly outperforming the ~50% validity rate of direct Python search (AFlow) and requiring roughly half the token budget to reach comparable scores (Sec 5.3).
- **Exceptional transparency and reproducibility:** The appendices provide exhaustive implementation details, including full prompt templates, algorithmic pseudocode, node-type schemas, validation rules, and end-to-end case studies (Appendices A–B). This level of documentation is well above standard for the field and strongly supports independent replication.

## Weaknesses
- **Overstated theoretical "guarantees" vs. practical implementation:** The introduction claims the framework "guarantee[s] static graph-level correctness across the entire generation process," supported by Lemma 1 (closure under operators). However, Section 4.1 explicitly states that LLM-generated candidates can violate constraints and require a checker with iterative regeneration. This is a robust rejection-sampling/validation loop, not a mathematical guarantee by construction. While the practical outcome is strong, the framing conflates theoretical operator closure with the stochastic reality of LLM generation, which weakens the technical precision expected at ICLR.
- **Unvalidated and unablated LLM-as-Judge selection mechanism:** To avoid expensive rollouts during evolution, the method uses an LLM judge to score candidate graphs. While pragmatic, the paper provides no empirical validation of the judge's alignment with actual execution performance (e.g., Spearman correlation with rollout scores) and no ablation comparing it against simpler heuristics like direct validation-score ranking or random sampling. Without this, it remains unclear whether search efficiency stems from the structured graph space or the subjective filtering of the LLM judge.
- **Modest gains lack statistical rigor:** MermaidFlow outperforms strong baselines consistently, but absolute improvements are modest (~1–3%). With results averaged over only three runs and no reported standard deviations or significance testing, these gains could reasonably be attributed to run-to-run LLM stochasticity or prompt variance rather than a statistically robust architectural advantage. For competitive benchmarks where margins are tight, variance reporting and paired significance tests are necessary to substantiate claims of superiority.
- **Residual execution brittleness in the translation layer:** The framework positions itself as cleanly separating planning from execution, yet the Mermaid-to-Python translation step still relies entirely on an LLM (Appendix E). This reintroduces a failure surface where a syntactically valid, semantically sound graph can yield runtime errors due to flawed code generation. The authors acknowledge this and propose a rule-based compiler for future work, but in the current pipeline, the end-to-end guarantee of execution correctness remains incomplete.

## Nice-to-Haves
- **Out-of-distribution / cross-domain generalization:** Testing whether a workflow evolved on one domain (e.g., MATH) generalizes to another (e.g., GSM8K or a novel reasoning task) would strengthen the claim that the declarative representation captures transferable structural priors.
- **Full compute/latency accounting:** Providing a breakdown of wall-clock time and API costs per pipeline stage (graph generation, judging, retries, translation) would clarify the practical overhead of the multi-step EP loop compared to single-pass or iterative code-generation baselines.
- **Operator activation & contribution analysis:** Quantifying how often each EP operator is sampled and its delta in performance would reveal which transformations drive meaningful exploration versus those that act as dead weight.
- **Judge-to-ground-truth calibration:** A simple scatter plot or correlation metric between LLM-judge scores and actual validation accuracy would quickly validate the proxy's reliability.

## Novel Insights
The paper demonstrates that imposing static verifiability as an inductive bias fundamentally reshapes the optimization landscape for LLM-driven program synthesis. By constraining the search to a declarative, compiler-checkable graph space, the method transforms a high-variance, brittle string-edit problem into a structured recombination problem. The empirical observation that validity rates jump from ~50% (code-based) to >90% (graph-based) while halving token consumption suggests that syntactic and type constraints do not restrict the search space meaningfully; rather, they prune catastrophic failure modes that waste LLM inference capacity. This points to a broader design principle for agentic systems: lightweight, domain-appropriate formal constraints can dramatically improve search efficiency and stability without sacrificing expressiveness, provided the representation remains close enough to natural generation patterns.

## Suggestions
1. **Temper "guarantee" language and clarify Lemma 1's scope:** Revise the introduction and Section 4.1 to accurately frame the pipeline as "enforcing static validity via constrained generation and automated validation/rejection," rather than claiming a generation guarantee. Explicitly state that Lemma 1 describes the closure property of the *algorithmic operator definitions*, while the LLM instantiation relies on iterative checking.
2. **Add minimal statistical reporting and judge validation:** Report mean ± standard deviation for Table 1 and Figure 3. At minimum, run a Spearman correlation analysis between LLM-judge scores and actual validation scores on a held-out subset, and add a simple ablation table swapping the judge for direct score-based selection or random sampling. This would decisively isolate the search space contribution from the ranking heuristic.
3. **Clarify the translation bottleneck's scope:** In the main text limitations or future work, explicitly acknowledge that while graph validity is statically enforced, end-to-end execution correctness remains partially LLM-dependent due to the translation step. Briefly discuss how a deterministic Mermaid-to-Python compiler (as proposed in Appendix E) would fully realize the claimed planning/execution decoupling.
4. **Standardize baseline optimizer conditions (if feasible):** While token efficiency strongly isolates the representation benefit, ensuring that compared code-based baselines use identical optimization LLMs, temperature, and prompt complexity (or explicitly accounting for differences) would preempt concerns about confounding variables in absolute performance gains.

---

## ZfdnZhOP0k

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (6.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces HUBBLE, a fully open-source suite of LLMs (1B and 8B parameters, trained on 100B or 500B tokens) with controlled, randomized insertions of copyright, privacy, and test-set data at varying duplication levels and training phases. The authors establish two empirical findings: diluting sensitive data by training on larger corpora, and ordering it to appear earlier in pretraining, both reduce final-model memorization. The suite is further leveraged to create confounder-free benchmarks for membership inference and machine unlearning, demonstrating that current unlearning methods lack the precision required for surgical data removal and inadvertently degrade neighboring in-distribution examples.

## Strengths
- **Causal, confounder-controlled experimental design:** By strictly decontaminating the base corpus and injecting perturbations with randomized duplication counts (0× to 256×), the paper enables precise dose-response measurement of memorization. This directly addresses the confounding limitations of prior observational studies that struggle to disentangle repetition from intrinsic data simplicity.
- **Fully open, reproducible artifact release:** The authors release all model checkpoints, perturbation datasets, training/evaluation code, and detailed GPU/compute accounting. This transparency exceeds typical ICLR standards and provides a much-needed, standardized anchor for the academic memorization research community.
- **Benchmark design that isolates critical failure modes:** The randomized insertion scheme eliminates the spurious temporal cues that invalidate prior MIA benchmarks (e.g., WIKIMIA). Furthermore, the unlearning benchmark's use of a "Keep" set (in-distribution near-neighbors) cleanly exposes a fundamental limitation in current unlearning methods: they catastrophically degrade neighboring knowledge rather than performing targeted removal.

## Weaknesses
- **Timing/ordering claim is confounded with learning rate schedule and optimization phase:** The paper concludes that "ordering sensitive data to appear early in training reduces memorization risks." However, early training inherently operates in a high-learning-rate, high-gradient-noise, and rapidly shifting optimizer state regime, which is known to stabilize representations less effectively than late-stage fine-tuning. Without controlling for the optimization phase (e.g., via a counter-factual learning rate schedule or constant-LR ablation during early insertion), it remains unclear whether the observed reduction in memorization is a function of temporal ordering or simply the under-optimization dynamics of early training.
- **Unlearning benchmark lacks evaluation on general downstream capabilities:** While the benchmark effectively quantifies degradation on the "Forget" and "Keep" sets, it does not report post-unlearning performance on standard capability suites (e.g., MMLU, HellaSwag, or reasoning tasks). Without this, it is impossible to distinguish whether the observed degradation is truly localized to the insertion manifold or reflects broader catastrophic forgetting that would render unlearned checkpoints practically unusable.
- **Cross-domain interference validation is restricted to the 1B/100B scale:** The authors train single-domain models to verify that joint perturbation insertion does not cause capacity competition. This check is only performed on 1B models trained on 100B tokens. Given that larger models exhibit sharper loss landscapes, different representational crowding dynamics, and greater capacity for feature interference, the assumption that joint-domain insertion yields independent causal effects at the primary 8B/500B scale is not empirically verified.

## Nice-to-Haves
- Clarify how the perturbation insertion protocol (Fig 1), which guarantees perturbations are never split across training sequence boundaries, might artificially inflate verbatim memorization compared to fragmented, naturally occurring web data.
- Temper the "dilution" framing to acknowledge trade-offs: while relative memorization risk decreases with larger corpora, total compute costs increase, and the absolute volume of other potentially sensitive data seen by the model also scales.
- Evaluate stronger adversarial extraction methods (e.g., iterative beam search or red-teaming prompts) to stress-test the explicitly stated "lower bounds" on memorization.
- Provide multi-seed runs or variance estimates for low-duplication effects (e.g., 1× vs 0×), though recognized as computationally expensive for pretraining at this scale.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Reframe or ablate the "ordering" mitigatory claim to explicitly disentangle temporal positioning from the high-LR/high-noise optimization regime characteristic of early training phases. If an LR-controlled ablation is infeasible, explicitly state that the finding is conditional on standard cosine decay schedules.
- Include standard downstream capability metrics (MMLU, HellaSwag, commonsense reasoning) in the unlearning benchmark results. Quantifying general utility degradation alongside "Keep" set degradation is necessary to establish the benchmark's practical viability and guide future algorithm development.
- Extend the interference validation to at least the 8B/500B configuration, or explicitly frame the joint-domain design as a resource-constrained approximation and discuss how scale-dependent capacity bottlenecks might alter interference patterns in future work.

---

