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

## o8kbAXPu7P

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces TRANSFIR, a framework for inductive reasoning on Temporal Knowledge Graphs (TKGs) that specifically targets emerging entities arriving without historical interactions. It proposes a three-stage pipeline: (i) classifying entities into latent semantic clusters using a vector-quantized (VQ) codebook over frozen textual embeddings, (ii) encoding relation-guided interaction chains via a Transformer to capture sequential dynamics, and (iii) transferring temporal reasoning patterns within clusters to update entity representations dynamically. Across four standard benchmarks and 17 baselines under a strict chronological split, TRANSFIR demonstrates strong empirical gains, particularly in preventing representation collapse for cold-start entities.

## Strengths
- **Well-Motivated Problem Formulation & Empirical Diagnosis:** The paper rigorously quantifies the prevalence of emerging entities (~25% under the proposed split) and introduces the "Collapse Ratio" metric to empirically demonstrate how standard TKG models suffer severe representation collapse when lacking historical signals. This three-angle investigation (Data, Representation, Feasibility) provides a strong, evidence-based foundation for the proposed cold-start setting.
- **Targeted Architectural Pipeline for Zero-History Reasoning:** The Classification–Representation–Generalization design directly addresses the core challenge. Decoupling stable semantic priors (frozen text) from dynamic temporal patterns (interaction chains) and linking them via VQ clustering offers a principled way to inject meaningful signal into otherwise untrained entity representations. The approach is algorithmically clear, with documented linear complexity scaling.
- **Comprehensive Empirical Validation & Reproducibility:** The experimental section adheres to high empirical standards, evaluating across graph-based, path-based, and static inductive baselines under identical strict chronological splits. Results show consistent and substantial improvements (+28.6% avg. MRR on emerging entities), supported by thorough ablations, hyperparameter sensitivity, efficiency profiling, and multiple PLM encoder tests. Full code and detailed appendices are provided.

## Weaknesses
- **Unspecified Handling of Strictly Empty Interaction Histories:** The core problem formalization defines an emerging entity $e$ at query time $t_q = t_e(e)$ (its first appearance). Under this definition, the interaction window $[t_q - T, t_q)$ is strictly empty. However, Section 4.2 proceeds to encode the Interaction Chain $C_q$ using a Transformer with relation-guided attention (Eq. 8), where the softmax denominator becomes mathematically undefined for sequence length $n=0$. The paper does not specify a fallback mechanism (e.g., defaulting to the cluster prototype, a learned zero-history token, or direct text embedding bypass) for this exact zero-history edge case, leaving a critical implementation gap in the method that claims to solve it.
- **Absence of Known-Entity Evaluation & Risk of Representation Interference:** Table 1 and the primary results exclusively report performance on emerging entity triples. The pattern transfer mechanism updates $\tilde{\mathbf{h}}_e$ for *all* entities $e \in E$ at every timestamp (Algorithm 1, Line 19) by blending static embeddings with dynamic cluster prototypes. Without reporting metrics on known/warm entities, it is impossible to assess whether this global update scheme causes catastrophic interference or degrades standard transductive reasoning performance. A method that boosts cold-start accuracy at the expense of established entity performance would have limited practical utility in evolving TKGs.
- **Conflation of Static Textual Priors with Temporal Transfer Gains:** TRANSFIR's VQ clustering and initial representation rely entirely on frozen pretrained language model embeddings. While effective, the reported performance gains could be partially driven by the high-quality static semantic signal rather than the proposed temporal interaction chain and VQ transfer mechanism. The paper lacks a strong control baseline (e.g., direct relational scoring using only frozen text embeddings + relation projection) to isolate the marginal contribution of the dynamic pipeline. Consequently, the exact value added by the temporal transfer component remains ambiguous.

## Nice-to-Haves
- **VQ Codebook Stabilization:** Incorporating standard VQ stabilization techniques (e.g., exponential moving average for codebook updates, random restarts for dead codes) would improve training robustness, though the current commitment/codebook loss formulation is standard.
- **Cluster-Semantic Alignment Quantification:** Reporting Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI) between VQ cluster assignments and ground-truth entity types/metadata would strengthen the claim that clusters capture meaningful semantic structures suitable for pattern transfer.
- **Temporal Trajectory of Cluster Prototypes:** Plotting the evolution of dynamic cluster prototypes $\mathbf{c}_k^{dyn}$ across snapshots would visually verify that the model is learning shifting temporal patterns rather than static cluster centroids.
- **Relation-Frequency Stratification:** Breaking down performance by relation rarity would clarify whether TRANSFIR learns generalizable temporal logic or primarily exploits frequent, high-co-occurrence patterns.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- **Explicitly define the zero-history fallback mechanism** in Section 4.2. Clarify how $\mathbf{h}^{IC}_{e_q}$ is computed or bypassed when $C_q$ is empty, and update Algorithm 1 to reflect this edge-case handling.
- **Include a full evaluation on known entities** alongside emerging entity results. Reporting MRR/Hits@k for the standard/full test split (or explicitly stratifying known vs. unseen) is essential to verify that dynamic pattern transfer does not degrade well-learned representations.
- **Add a strong text-prior control baseline** (e.g., $\phi(e_s, r, e_o) = \text{MLP}([\mathbf{h}_{e_s}^{text} \parallel \mathbf{h}_r \parallel \mathbf{h}_{e_o}^{text}])$) to the experimental table. This will cleanly isolate how much performance stems from the frozen textual prior versus the proposed temporal chain encoding and VQ-driven generalization.

---

## SkmkGKEZ1U

- GT: Reject (avg 0.5)
- Predicted: N/A (1.7/10)
- Match: N/A

### Final Review

## Summary
O-Forge proposes a practical pipeline that couples frontier LLMs with Mathematica’s `Resolve` function to automate the verification of asymptotic inequalities and series decompositions. By delegating the creative step of identifying subdomain breakpoints to the LLM and offloading the mechanical verification to quantifier elimination over the reals, the system targets a documented bottleneck in analytic number theory and theoretical computer science. The authors provide a functional CLI/web interface and demonstrate the approach on case studies inspired by recent commentary from Terence Tao.

## Strengths
- **Well-justified division of cognitive labor:** The paper accurately identifies domain/series decomposition as the primary creative bottleneck in asymptotic analysis, while recognizing that subdomain verification is mechanical. Separating these tasks mirrors successful AI4Math paradigms but applies them to a distinct, research-relevant niche beyond contest mathematics.
- **Pragmatic engineering for target users:** The system is explicitly designed for non-programmer researchers, featuring a clean LaTeX-to-verification pipeline, structured prompting, and a public web interface. This lowers the adoption barrier for mathematicians who lack formal verification or command-line expertise.
- **Defensible toolchain selection:** The authors provide a clear, experience-based rationale for choosing Mathematica’s `Resolve` over SMT solvers and Lean tactics, correctly noting that transcendental functions and nonlinear asymptotic bounds frequently break open-source or linear-restricted verifiers.

## Weaknesses
- **Insufficient quantitative evaluation and missing baselines:** The empirical section claims the tool is “robust” after testing on “40-50 easier problems,” yet reports zero success rates, latency metrics, failure analyses, or computational cost breakdowns. For ICLR, system claims require rigorous empirical grounding. The absence of baselines (e.g., LLM-only proof attempts, heuristic/random splitting, or other CAS wrappers) makes it impossible to isolate the performance gain attributable to the pipeline architecture from the raw capabilities of the underlying LLM or CAS.
- **Opaque critical implementation components:** The pipeline’s success hinges on two underspecified elements. First, Step 3 (“Regime-wise simplification”) relies on “elaborate Mathematica code” to extract leading-order terms and enforce positivity, yet the algorithm is not disclosed. Second, the structured prompt template in Sec 4 is left empty. Without disclosing these components, the technical contribution cannot be properly assessed or reproduced.
- **Missing domain coverage verification and heuristic constant bounding:** The framework verifies each proposed subdomain independently but does not automatically check whether the union of LLM-suggested subdomains actually covers the original domain or whether they are disjoint. Gaps, overlaps, or invalid boundaries would silently invalidate the global proof. Additionally, bounding the constant $C$ to a fixed grid ($1$ to $10^4$) injects an arbitrary numerical cutoff into a purportedly symbolic verification pipeline, creating false negatives for inequalities requiring larger or irrational constants.
- **Discrepancy between abstract and methodology:** The abstract advertises an “In-Context Symbolic Feedback loop,” but the methodology describes a strictly single-pass, linear cascade. If no iterative refinement or retry mechanism exists, the terminology is misleading.

## Nice-to-Haves
- Implement an automated retry/feedback mechanism where CAS failures or timeouts trigger alternative LLM proposals with explicit error constraints, as mathematical proving rarely succeeds on the first heuristic guess.
- Cross-validate `Resolve` outputs against open-source CAS or proof assistants on a shared subset to mitigate trust gaps inherent to closed-source quantifier elimination, and report the agreement rate.
- Provide empirical or theoretical justification for the observation that “the number of decompositions grows linearly with the number of variables,” as this claim is currently unsupported by data or analysis.
- Clarify the evaluation methodology by categorizing the test problems by complexity (variable count, function class, transcendental presence) to better map the system’s operational boundaries.

## Novel Insights
The paper’s core contribution is recognizing that the primary friction in research-level asymptotic analysis is not proof verification, but the creative identification of domain decompositions, and that this specific cognitive load aligns optimally with LLM generative strengths when explicitly decoupled from mechanical verification. This targeted separation of concerns offers a pragmatic, immediately deployable bridge from contest-math AI to research-math automation, accepting the trade-off of closed-source verifiers in exchange for out-of-the-box capability on transcendental bounds.

## Suggestions
- Release the test problem suite as a standardized benchmark with difficulty labels, and report quantitative metrics including pass@1/pass@k rates, average wall-clock time, API costs, and a structured failure taxonomy (LLM parsing error, `Resolve` timeout, $C$-grid exhaustion, invalid split).
- Disclose the complete LLM backend/version, temperature settings, exact prompt templates, and the algorithm/code for Step 3 (leading-term extraction) in the appendix or supplementary materials.
- Implement a lightweight symbolic check to verify that $\bigcup D_i = D$ and that subdomains are non-overlapping and singularity-free before running `Resolve`.
- Replace or supplement the fixed $C$-grid search with a native $\exists C > 0$ quantifier query in `Resolve`. If this proves unstable or computationally prohibitive, explicitly justify the numerical cap as a practical heuristic and document its failure boundaries.

---

## xVB9AivJa5

- GT: Reject (avg 1.3)
- Predicted: N/A (3.2/10)
- Match: N/A

### Final Review

## Summary
Blueprint-Bench introduces a model-agnostic benchmark that evaluates 2D spatial reasoning by tasking AI systems with generating standardized floor plans from sequences of interior photographs. Using a computer-vision extraction pipeline and a graph-similarity composite metric, the authors compare leading LLMs, image generation models, and agentic workflows across 50 apartments. The results reveal a consistent performance gap where frontier models perform near a null-input baseline, substantially below human capability.

## Strengths
- **Well-motivated evaluation of out-of-distribution spatial reasoning:** The task deliberately uses an in-distribution input modality (real-world interior photographs) while requiring an out-of-out-distribution structured output, effectively probing genuine spatial reconstruction rather than memorized pattern matching. This provides a valuable complement to highly alien benchmarks like ARC.
- **Transparent, automated, and model-agnostic scoring pipeline:** The two-step extraction and composite scoring methodology (Sections 2.2–2.3) are thoroughly documented, including precise parsing rules, door detection logic, and room size ranking. This ensures reproducibility and enables fair cross-architecture comparison without manual evaluation bias.
- **Candid acknowledgment of design trade-offs and limitations:** Section 2.4 explicitly addresses known scoring constraints (size-rank penalty cascades, shape ignorance, strict formatting rigidity) and justifies why alternative approaches (LLM-based labeling, point-sampling geometry metrics) were abandoned due to current model unreliability. This level of methodological self-awareness strengthens the paper's academic rigor.

## Weaknesses
- **Conflation of instruction-following/compliance with spatial reasoning:** The benchmark's core claim rests on measuring "spatial intelligence," yet the strict rendering constraints (exact 3px lines, pure RGB values, mandatory 10px dots) mean a substantial portion of failures stem from pixel-level formatting non-compliance rather than topological misunderstandings (e.g., GPT-4o and NanoBanana failures in Figure 6). While the authors acknowledge this as a deliberate trade-off for scoring robustness, it fundamentally weakens the assertion that low scores purely reflect a spatial blind spot. Without a separate validity filter or conditional scoring, the benchmark currently measures a hybrid of instruction-following and spatial reconstruction.
- **Unjustified composite scoring weights:** The final 0–1 score combines six components using fixed weights (50% edge overlap, 20% degree correlation, 10% density, 10% room count, 5% door count, 5% orientation) without theoretical grounding, empirical validation, or sensitivity analysis. This makes it impossible to determine whether reported model deficits are driven by genuine topological errors, minor size misestimations, or arbitrary metric prioritization. For a benchmark intended to track progress, unvalidated weighting schemes risk obscuring which specific spatial sub-capabilities are actually improving.
- **Misaligned baseline definitions weaken capability gap claims:** The paper references a "random baseline" (Section 3, Figure 5) but defines it in Section 2.2 as floor plans generated by models *without image input*—a null-input/unconditional baseline, not a formally defined random graph or size-rank sampling process. Additionally, human performance (Figure 7) is reported on only 12 apartments versus the full 50 used for models. These discrepancies complicate direct statistical comparisons and weaken the quantitative framing of the human-AI performance gap.

## Nice-to-Haves
- **Component-level score decomposition:** Reporting per-metric breakdowns (edge overlap vs. size rank vs. door orientation) would provide finer-grained diagnostics without requiring full pipeline redesign.
- **Input aggregation analysis:** Clarifying how models that typically accept 1–4 images processed ~20 images per apartment (e.g., stitching, summarization, context truncation) and ablation across aggregation strategies could disentangle spatial reasoning limits from context-window bottlenecks.
- **Complexity correlation:** Calculating a layout complexity metric (e.g., cyclomatic number, room count, or non-rectangular boundaries) and correlating it with model scores would reveal whether failures are uniform or topology-dependent.
- **Human protocol transparency:** Specifying annotator expertise, time limits, and inter-annotator agreement would strengthen the human baseline's reliability as a ceiling reference.

## Novel Insights
The benchmark effectively isolates a consistent, architecture-agnostic failure mode: even models trained on massive multimodal corpora struggle to synthesize coherent 2D topological structures from fragmented visual inputs. Crucially, the finding that agentic iterative refinement fails to bridge this gap—often because agents hallucinate self-corrections or fail to visually verify intermediate outputs—suggests that spatial grounding remains a structural deficit in current scaling paradigms rather than a simple single-pass or context-window constraint. This implies that emergent "general intelligence" in image and language models does not yet encompass the geometric and topological synthesis required for physical environment understanding.

## Suggestions
1. **Decouple formatting compliance from spatial scoring in results:** Add a secondary analysis or appendix table that only scores outputs passing a basic format-validity filter, explicitly reporting spatial accuracy conditional on rule adherence. This would isolate whether the "blind spot" is primarily spatial or instructional.
2. **Validate scoring weights via sensitivity analysis:** Perturb the component weights (e.g., uniform weighting ±10% shifts, or PCA-derived weights) and report whether model rankings remain stable. This will demonstrate that the composite metric robustly reflects spatial fidelity rather than designer priors.
3. **Align baseline definitions:** Rename the "random baseline" to "null-input/unconditional baseline" for accuracy, and ideally include a true random graph/size-ranking baseline for reference. If expanding the human evaluation to 50 apartments is infeasible, explicitly frame the 12-sample human result as a preliminary ceiling estimate rather than a direct statistical comparison to the 50-sample model aggregate.

---

## ASwVqmDOv9

- GT: Reject (avg 5.0)
- Predicted: N/A (7.1/10)
- Match: N/A

### Final Review

## Summary
This paper introduces GMem, a framework that conditions diffusion transformers on an external memory bank of self-supervised image representations ("snippets"). By framing conditioning strength as a continuum between weak class labels and strong oracle targets, the authors demonstrate that intermediate-density semantic features straighten probability flow trajectories, accelerating optimization. GMem achieves substantial empirical gains, including a 50× training speedup on ImageNet-256 (FID 1.53), rapid ~20K-step domain adaptation, and a bank-free text-to-image pipeline that matches strong baselines with a fraction of paired data and compute.

## Strengths
- **Compelling empirical efficiency and quality:** The paper delivers robust, cross-backbone validation of its core premise. Reaching FID 1.53 on ImageNet-256×256 in ~160 epochs without classifier-free guidance, alongside consistent gains on CIFAR-10 and ImageNet-512, establishes GMem as a highly effective acceleration strategy for modern diffusion transformers.
- **Clear conceptual framing of conditioning strength:** Sections 3.1–3.2 successfully articulate conditioning density as a control knob for flow curvature and optimization landscape complexity. This theoretical lens elegantly unifies class-conditional, instance-conditional, and text-to-image paradigms, which is strongly supported by the bidirectional conversion experiments between GMem and standard DiTs.
- **Practical downstream utility and architectural flexibility:** Beyond training speedup, GMem decouples generalization from memorization, enabling test-time domain adaptation via snippet interpolation, lightweight SVD compression, and rapid fine-tuning to specialized domains (faces, medical, anime) with minimal computational overhead.
- **Thorough ablation and transparency:** The systematic evaluation of masking strategies, solver choices (SDE vs. ODE), vision encoders, and bank capacity scaling demonstrates methodological rigor. Detailed hyperparameter reporting across backbones and resolutions supports reproducibility.

## Weaknesses
- **Mathematically underspecified noise-to-index sampling mechanism:** Section 4.3 defines the sampling index as `i = Φ(z)`, where `Φ` is the univariate standard normal CDF. However, `z` is a high-dimensional tensor (e.g., 4×32×32 for SD-VAE latents). The paper does not specify how this tensor is reduced to a scalar input for `Φ` without discarding entropy, introducing bias, or violating the uniform coverage assumption. This is a critical reproducibility gap that obscures the true sampling distribution and requires formal definition or an alternative deterministic mapping.
- **Incomplete compute accounting for T2I efficiency claims:** The abstract claims a `1/9` reduction in training cost and `1/17` in data volume compared to PixArt-α. However, this only accounts for the fine-tuning stage of the T2S adapter on 0.2M paired samples. It omits the substantial pretraining cost on the 1.28M ImageNet base model. For compute efficiency claims to hold and be fairly compared against end-to-end trained baselines, the paper must report the cumulative GPU-days (pretraining + adaptation) or clearly frame the comparison as a *pretraining + lightweight alignment* advantage rather than a direct compute-to-compute ratio.
- **Attribution of speedup conflates memory conditioning with orthogonal alignment losses:** The primary 50× speedup demonstrations stack GMem onto REPA (representation alignment). While the paper shows additive benefits over REPA in Table 2, it lacks a clean isolation of GMem's marginal contribution when starting from a vanilla DiT/Flow backbone. Without a `Base-DiT + GMem vs. Base-DiT` head-to-head at matched FID targets, it remains difficult to disentangle how much acceleration stems from the explicit memory bank versus the pre-existing representation alignment priors.

## Nice-to-Haves
- Provide a direct empirical comparison against retrieval-augmented diffusion baselines (e.g., KNN-Diffusion, RDM) to demonstrate how GMem's semantic masking and SSL feature extraction differ from raw retrieval in optimizing the denoising objective.
- Expand the T2I evaluation beyond GenEval to include holistic perceptual alignment and aesthetic metrics (e.g., CLIPScore, HPSv2, ImageReward) to better capture the quality/compute trade-off on small paired datasets.
- Formalize the "flow straightening" hypothesis with quantitative trajectory metrics (e.g., integrated path length, velocity field divergence) computed on real latent embeddings, moving beyond the toy GMM visualizations to ground the theoretical intuition in high-dimensional data geometry.
- Analyze the sensitivity of the T2S adapter depth and the capacity of the frozen text encoder to isolate whether data efficiency gains stem from the snippet-space bottleneck or the strength of the underlying language model.
- Release or detail the exact preprocessing, normalization, and indexing pipeline for the 1.28M snippet bank to lower the barrier for independent validation of the reported efficiency gains.

## Novel Insights
GMem reframes the role of external memory in generative modeling by positioning it as a tunable regularizer along a conditioning entropy continuum. Rather than treating memory as a static retrieval database, the framework leverages intermediate-density, self-supervised snippets to inject just enough semantic grounding to rectify high-curvature probability flows, while preserving the stochasticity needed to prevent distributional collapse. This establishes a principled bridge between weakly conditioned generation (high flexibility, slow convergence) and strongly conditioned generation (fast convergence, high overfitting risk), effectively decoupling the transformer's capacity for geometric generalization from the explicit memorization of training statistics. The resulting architecture enables test-time paradigm conversion and rapid domain adaptation by simply swapping or manipulating the external bank, offering a fresh perspective on how to structure conditioning signals for compute-constrained diffusion.

## Suggestions
- **Clarify the indexing mechanism:** Explicitly define the mathematical operation that maps the high-dimensional noise tensor `z` to a scalar for the CDF `Φ(z)`. If a hash, channel reduction, or learned projection is used, formalize it and provide an ablation or theoretical justification for why it preserves sampling uniformity and does not bias the generative posterior.
- **Report full compute budgets:** For all efficiency claims, especially the T2I pipeline, report the cumulative training compute (ImageNet pretraining + T2S fine-tuning) alongside the fine-tuning-only numbers. This will provide a transparent, apples-to-apples comparison with end-to-end baselines and strengthen the paper's efficiency narrative.
- **Isolate the memory bank's contribution:** Include a training run of a standard DiT/Flow backbone with GMem conditioning (without REPA) and report convergence curves. Isolating the memory bank's effect will solidify the claim that explicit semantic snippets, independent of orthogonal alignment losses, are the primary driver of accelerated optimization.

---

## pzXAS6Tf2r

- GT: Reject (avg 4.8)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
HiViBiX proposes a visually conditioned framework for translating mono music audio into binaural stereo by learning an intermediate Ambisonics-style representation (predicting X, Y channels and gain parameters) rather than directly generating left/right spectrograms. The method conditions a latent encoder-decoder via a hierarchical visual module that fuses local object crops, global scene semantics, monocular depth, and positional encodings. Extensive experiments on three standard benchmarks demonstrate consistent state-of-the-art reconstruction metrics, with supplementary out-of-domain generalization experiments highlighting practical deployment readiness.

## Strengths
- **Structurally motivated spatial bottleneck:** Framing mono-to-binaural conversion as a learnable Ambisonics decoding problem provides a physically grounded, interpretable architecture that moves beyond heuristic end-to-end stereo mapping. The explicit separation of spatial channels and gain parameters introduces valuable inductive bias for spatial control.
- **Hierarchical multimodal visual conditioning:** The encoder's design strategically combines local source crops (via object detection), global scene context, depth estimation, and Fourier positional features. Cross-attention that queries global features against local cues is a well-reasoned approach for grounding spatial audio in both environmental context and specific sound sources.
- **Strong empirical performance & practical generalization:** The method achieves clear improvements across FAIR-Play, Music-Stereo, and YT-Music on multiple objective metrics. The systematic ablation (Table 3) cleanly demonstrates additive contributions, and the lightweight out-of-domain adapter (YOLOv11 + open-class detection) shows thoughtful engineering for real-world audio scenes.

## Weaknesses
- **Missing binaural-specific evaluation metrics:** The core contribution claims "precise spatialization" via an Ambisonics-inspired structure, yet quantitative evaluation relies entirely on waveform/spectrogram reconstruction metrics (STFT L2, envelope distance, SNR). These measure signal fidelity but do not validate perceptual spatial quality. Crucially, domain-standard spatial metrics such as Interaural Time/Level Difference (ITD/ILD) error, azimuth localization accuracy, or spatial consistency scores are absent. Without them, the claim that the learned representation improves spatial positioning remains empirically unverified.
- **Ablation restricted to a single dataset split:** Table 3 evaluates component contributions (Ambisonics representation, loss design, visual modalities) using only the first split of FAIR-Play. Given the dataset's limited scale and documented performance variance across splits, this restricts confidence in the reported additive gains and leaves the architectural justifications untested on more diverse or dynamic datasets like YT-Music.
- **Underpowered subjective validation:** The included user study (13 participants, 20 samples) lacks statistical reporting (confidence intervals, significance testing, variance) and standardized listening environment controls. For a paper centered on immersive audio quality, the perceptual evidence is insufficiently rigorous to support claims of superior spatial realism, especially when objective metrics already plateau or show marginal gains on certain datasets.

## Nice-to-Haves
- Quantify the impact of visual pipeline failures (e.g., YOLO misdetections, inaccurate depth maps) on audio degradation to empirically validate the claimed robustness of the hierarchical conditioning.
- Correlate learned position/gain parameters with ground-truth source coordinates or simulated spatial layouts to verify the Ambisonics-like channels meaningfully decouple panning from spectral content rather than acting as an entangled latent mapping.
- Provide a temporal consistency analysis across video frames to measure performance decay in dynamic sequences, complementing the acknowledged single-frame limitation.
- Clarify the SNR formulation and report in standard logarithmic (dB) units if comparing against literature that uses perceptual loudness scales, though the current linear reporting is internally consistent.

## Novel Insights
The paper successfully shifts the paradigm of visually guided binaural synthesis from black-box stereo regression to a physics-inspired intermediate representation. By treating Ambisonics not as a strict ground-truth target but as a structured inductive bottleneck, the model learns a latent decomposition that naturally separates spatial positioning from source identity. This approach highlights a broader insight for multimodal audio AI: constraining generative architectures with domain-specific signal priors (here, spherical harmonic decomposition) can yield more controllable and interpretable synthesis, provided the conditioning pipeline is sufficiently hierarchical to capture both global acoustic context and local source geometry. The out-of-domain success further suggests that spatial grounding generalizes well when visual priors are decoupled from strictly musical categories.

## Suggestions
- Integrate explicit spatial evaluation metrics into the main results: compute ITD/ILD errors or run automated localization accuracy tests using standard binaural evaluation toolkits to directly validate the spatial claims of the Ambisonics bottleneck.
- Replicate the component ablation across at least two splits of FAIR-Play and include a compact version on YT-Music to verify that visual and structural components generalize beyond controlled indoor recordings.
- Expand the subjective evaluation to a standardized listening test (≥30 participants, controlled acoustic environments, explicit rating dimensions for localization, externalization, and timbral fidelity) and report confidence intervals and paired statistical tests.

---

## N8ntZEb4Ap

- GT: Withdrawn (treated as Reject) (avg 2.3)
- Predicted: N/A (2.4/10)
- Match: N/A

### Final Review

## Summary
AutoNFS proposes an end-to-end differentiable feature selection framework that couples a Gumbel-Sigmoid masking network with a task-specific predictor and a cardinality penalty. By jointly optimizing a global feature mask and the downstream objective with temperature annealing, the method automatically determines a minimal sufficient feature subset without requiring a pre-specified feature budget $k$. The approach is validated on corrupted OpenML benchmarks and high-dimensional metagenomic datasets, demonstrating competitive predictive performance while selecting substantially fewer features.

## Strengths
- **Practical automatic cardinality discovery:** The integration of a continuous sparsity penalty with temperature-annealed Gumbel-Sigmoid sampling effectively removes the need for manual $k$ specification, streamlining feature selection pipelines that traditionally require expensive hyperparameter sweeps or iterative retraining. (Evidence: Sec 3.3-3.4; empirical results show consistent feature reduction across diverse datasets without manual budget tuning.)
- **Comprehensive empirical validation across domains:** The paper evaluates performance on 11 tabular datasets under three controlled corruption schemes (random, Gaussian noise, second-order interactions) and 24 real-world metagenomic datasets. The inclusion of diagnostic metrics like misselection error rates and per-feature predictive power drop strengthens the practical credibility of the learned masks. (Evidence: Sec 4.1-4.2, Fig 3, Table 2.)
- **Strong reproducibility foundations:** The training procedure is explicitly algorithmized, hyperparameter optimization spaces and schedules are fully documented, and an anonymous code repository is provided, aligning well with modern empirical ML standards. (Evidence: Algorithm 1, Appendix C, code link.)

## Weaknesses
- **Flawed baseline comparison isolates noise removal rather than selection quality:** The benchmark evaluates all baseline methods on the full, noise-corrupted feature set without applying any feature selection, while AutoNFS removes 50%+ of the corrupted attributes. This experimental setup conflates the benefits of automatic noise filtering with the intrinsic quality of the selection mechanism. Without matched-sparsity evaluations or properly tuned sparse baselines, the central claim that AutoNFS achieves superior predictive performance while selecting fewer features cannot be verified and likely overstates the method's advantage.
- **Overstated claims regarding automaticity and computational scaling:** The paper frames cardinality determination as "automatic," yet Appendix F demonstrates that both the number of selected features and final accuracy are highly sensitive to the sparsity penalty weight $\lambda$, shifting the manual tuning problem from $k$ to $\lambda$. Additionally, the claim of "nearly constant computational overhead regardless of input dimensionality" contradicts the architecture: the masking network contains a linear projection from a 32-dimensional embedding to $D$ logits, which incurs at least $\mathcal{O}(D)$ forward/backward complexity. The reported sublinear scaling exponent ($\alpha \approx 0.08$) is derived from wall-clock timings dominated by batch processing overhead, GPU kernel latencies, and the small fixed dimension of the embedding bottleneck, making the complexity claim theoretically inaccurate.
- **Insufficient differentiation from established differentiable selectors:** The core methodology closely parallels existing techniques such as Stochastic Gates and $\ell_0$-relaxed neural networks, which also employ continuous relaxations of binary gates paired with sparsity penalties to automatically learn feature cardinality without fixing $k$. The paper cites these methods but does not empirically compare against them or articulate the algorithmic/practical advantages of AutoNFS's global embedding bottleneck and independent Gumbel-Sigmoid sampling over independent gate optimization, leaving its novelty contribution ambiguous.
- **Lack of statistical robustness in empirical reporting:** Main results report single-run point estimates without standard deviations, confidence intervals, or statistical significance testing. Given the stochastic nature of Gumbel-Sigmoid sampling, mini-batch optimization, and embedding initialization, marginal ranking differences (e.g., 2.1 vs. 3.6) and small accuracy improvements (e.g., +0.8 pp on metagenomic data) cannot be claimed as consistent or reliable without multi-seed variance reporting and formal statistical validation.

## Nice-to-Haves
- Report Pareto fronts (accuracy vs. feature count) across a range of $\lambda$ values rather than relying on a fixed $\lambda=1$, to fully characterize the sparsity-performance trade-off and demonstrate robustness across different noise regimes.
- Evaluate the selected feature subsets on stronger, modern tabular architectures (e.g., FT-Transformer or tree-ensemble hybrids) to ensure that performance gains are not merely compensating for the limited capacity of the shallow evaluation MLP.
- Analyze mask stability under high feature multicollinearity to determine whether the element-wise sampling arbitrarily distributes importance among correlated features or converges to consistent, interpretable subsets.

## Novel Insights
The paper's use of a single 32-dimensional learnable embedding vector to generate all feature selection logits implicitly imposes a low-rank constraint on the selector's weight space. While presented as an efficiency measure, this architecture likely acts as an implicit regularizer, forcing feature importance logits to share gradient information and preventing the sparse mask from overfitting to idiosyncratic noise in high-dimensional regimes. Coupled with the exponential temperature annealing schedule, this design cleanly translates discrete combinatorial subset selection into a continuous exploration-exploitation curriculum. However, this insight also highlights a trade-off: the bottleneck may restrict the selector's ability to resolve highly specific, independent features in datasets with complex, uncorrelated signal structures, a limitation the empirical evaluation does not currently probe.

## Suggestions
1. Re-run the benchmark with properly tuned sparse baselines (e.g., thresholded Lasso/LassoNet, RFE) that actually perform feature selection, and report results at matched feature counts or as accuracy-vs-sparsity Pareto curves to isolate the quality of AutoNFS's selection mechanism from noise removal.
2. Add direct empirical comparisons against core differentiable feature selection methods (e.g., STG, $\ell_0$-relaxed networks, Concrete Autoencoders) under identical training budgets, architectures, and hyperparameter search spaces to substantiate novelty claims.
3. Report mean $\pm$ standard deviation across at least 5 random seeds for all main benchmark results, and supplement average rank scores with statistical significance tests (e.g., Wilcoxon signed-rank or paired bootstrap intervals) to validate claimed superiority.
4. Replace wall-clock complexity fitting with theoretical FLOPs and parameter count derivations per training step to accurately characterize computational scaling, and reframe claims to acknowledge that cardinality discovery depends on tuning $\lambda$ or propose a lightweight adaptive schedule for it.

---

## K5t8PfzwFR

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (3.1/10)
- Match: N/A

### Final Review

## Summary
This paper introduces the Vision-based Inverse Dynamics (VID) dataset and a corresponding benchmark for predicting human joint torques directly from monocular real images, aiming to transition biomechanical analysis out of constrained laboratory environments. The authors provide ~63k synchronized frames with kinematic and dynamic annotations, propose a baseline network combining spatial probabilistic pose estimation with a temporal Transformer regressor, and establish a three-tiered evaluation protocol.

## Strengths
- **Addresses a clear, high-impact bottleneck:** The shift from lab-bound mocap/sEMG pipelines to direct image-to-torque estimation is well-motivated and practically relevant for sports science, rehabilitation, and robotics. The paper correctly identifies the deployment gap in current inverse dynamics methods.
- **Meticulous dataset curation & transparency:** The dataset derivation process includes explicit synchronization protocols, Savitzky-Golay smoothing, and cubic spline outlier correction. The appendix provides highly usable metadata, including precise marker placements, action distributions per subject, and torque variable definitions, significantly lowering the barrier for downstream research.
- **Granular, interpretable evaluation protocol:** Moving beyond a single aggregate metric, the three-level protocol (overall, joint-specific, action-specific mPJE) provides structured insight into model behavior across different biomechanical conditions and movement complexities.

## Weaknesses
- **Train/test split risks severe subject-level data leakage:** Section 5.1 specifies an 8:2 training/testing split but does not state whether it is randomized at the frame, trial, or subject level. Given the conclusion explicitly defers "cross-subject generalization" to future work, this strongly implies subject overlap in the split. In biomechanical learning, frame-level mixing allows the model to memorize subject-specific anthropometry and motion dynamics, artificially depressing error and invalidating generalization claims. A dataset/benchmark paper cannot claim real-world applicability without subject-disjoint evaluation.
- **Uncontrolled & mismatched baseline comparison:** The paper compares VID-Net (end-to-end image input) against Dino and ImDy, which natively consume ground-truth mocap/marker trajectories. While Section 5.1 acknowledges this modality gap, the baselines are not adapted to share the same visual kinematic inputs, nor is a standard `Pose Estimator → Torque Regressor` pipeline reported as a controlled reference. The reported ~39.8% improvement therefore conflates superior joint position recovery with actual inverse dynamics modeling, making it impossible to validate the proposed torque regression architecture's contribution.
- **Internal inconsistencies & missing architectural details:** Section 3 states the dataset was sampled from "seven types of movements, from which approximately 100 consecutive frames per trial were extracted." This directly contradicts Appendix Table 6, which lists 16 distinct action types with per-subject frame counts ranging from ~80 to over 2,000. Additionally, Equation 2 includes height ($H$) and mass ($M$) as inputs, but Section 4.2 completely omits how or where these anthropometric scalars are injected into the network. Resolving monocular scale ambiguity is fundamental to physically meaningful torque estimation, and this implementation gap severely limits reproducibility and physical grounding.

## Nice-to-Haves
- **Error decomposition & biomechanical plausibility checks:** Isolating vision-induced kinematic error from dynamics regression error (e.g., by feeding ground-truth poses to TorqueInferNet) would clarify where failures originate. Plotting full temporal torque waveforms rather than aggregated mPJE would expose phase lags, peak clipping, or unphysical discontinuities common in pure MSE regression.
- **Physics-informed regularization or GT pipeline details:** Incorporating temporal smoothness penalties, joint limit constraints, or ground reaction force consistency into the loss would better align the model with biomechanical priors. Providing explicit OpenSim solver settings, musculoskeletal scaling protocols, and ground reaction force estimation methods would further bolster label trust.
- **Standard statistical reporting:** While single-run reporting is common, providing subject- or trial-level standard deviations would contextualize the mPJE numbers given the high inter-subject biological variance in joint torque generation.

## Novel Insights
The paper inadvertently highlights a critical methodological gap at the intersection of computer vision and biomechanics: current evaluation pipelines conflate kinematic recovery accuracy with dynamic inference capability. By comparing image-based torque estimation against mocap-based methods without equalizing the kinematic input quality, it becomes impossible to determine whether performance gains stem from better 3D pose lifting or genuine physics-aware temporal modeling. Advancing this field requires decoupling these stages, establishing strict subject-disjoint splits to prevent anatomical memorization, and treating anthropometric parameters not as static metadata but as explicit, structurally integrated scaling factors within the network architecture.

## Suggestions
1. **Redesign the benchmark split & re-report results:** Restructure the train/validation/test partitions to be strictly subject-disjoint (leave-one-subject-out or explicit unseen-subject holdouts). Report the resulting mPJE to demonstrate true cross-anatomy generalization. If performance drops significantly, reframe the paper's contribution as a foundational dataset with a strong supervised baseline rather than a deployment-ready solution.
2. **Equalize baseline inputs for a fair ablation:** Adapt Dino and ImDy (or a standard inverse dynamics pipeline) to consume the exact same predicted 3D markers/poses generated by VID-Net's auxiliary heads. This isolates the torque regression capability from the vision backbone and validates whether the Transformer-based temporal modeling genuinely outperforms prior dynamics predictors under identical kinematic uncertainty.
3. **Resolve documentation & architectural gaps:** Correct the action/frame count inconsistency in Section 3 to match Appendix Table 6. Explicitly describe how $H$ and $M$ are injected into the network (e.g., broadcasted feature vectors, FiLM layers, or late-stage concatenation). Update Equation 2 to match the actual end-to-end formulation ($\tau = \text{VID}(I_{t-T:t})$) to eliminate the stated contradiction regarding supervision vs. inference inputs.

---

## DjxNqXsApM

- GT: Reject (avg 3.0)
- Predicted: N/A (3.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Ordered Sparse Autoencoders (OSAE), which apply a nested dropout-style prefix sampling objective to hard Top-$m$ SAEs to enforce a strict, canonical ordering of latent dimensions. The authors prove that under standard sparse dictionary learning assumptions (spark condition, non-negative codes, and frequency-aligned ground truth), the method resolves permutation non-identifiability and guarantees exact ordered feature recovery. Empirically, OSAE demonstrates substantial improvements in cross-seed feature consistency and early-prefix stability on synthetic data and LLM activations, while reducing dictionary incompleteness, though with measurable trade-offs in late-prefix utilization and reconstruction.

## Strengths
- **Principled reduction of permutation ambiguity:** By treating every latent dimension as its own prefix group and optimizing over all prefixes in expectation, OSAE directly targets the permutation invariance that plagues standard SAE reproducibility. This is a clean, theoretically motivated extension of Matryoshka SAEs and nested dropout.
- **Strong identifiability guarantee:** Theorem 3.1 rigorously connects the prefix-sampling loss to unique basis recovery under established sparse dictionary learning conditions (spark $>2m$, non-negativity, support-frequency ordering). The proof structure (Lemma 3.1 + perturbation argument) is mathematically sound and provides a principled foundation beyond heuristic regularization.
- **Empirical validation of consistency gains:** Prefix-orderedness and stability metrics consistently improve over baseline Matryoshka variants across synthetic settings, Gemma-2 2B, and Pythia-70M. The cross-dataset same-seed experiments and the SAE stitching analysis (reducing novel features from 73.8% to 33.8%) convincingly demonstrate that the ordering constraint shrinks the solution space and mitigates feature incompleteness.
- **Introduction of the FIFR error metric:** The Frequency-Invariant Feature Reconstruction Error (Appendix B.4) correctly identifies that global MSE underweights rare features in skew/Zipfian regimes. This metric better correlates with ground-truth dictionary stability than $\ell_2$ loss and is a valuable diagnostic contribution to the SAE evaluation toolkit.

## Weaknesses
- **Late-prefix degradation and ambiguous feature utilization:** Stability drops sharply after early prefixes (~128 out of 4096) and reconstruction loss can be higher than baselines. The paper attributes this to features being "less significant" but does not empirically disentangle whether this reflects natural signal decay in the data or optimization-induced gradient starvation/collapse from the strict prefix objective and unit-sweeping schedule. Without activation statistics or gradient norm analysis across indices, it remains unclear if the later dictionary capacity is effectively utilized or artificially suppressed.
- **Gap between theoretical assumptions and LLM activation regimes:** The identifiability guarantee requires exact non-negative codes, strict Top-$m$ sparsity, and ground-truth features ordered by activation frequency. LLM activations are signed, approximately sparse, and rarely follow a clean frequency-to-meaning mapping. While acknowledged in the limitations section, the paper lacks empirical analysis of how violations of these assumptions (e.g., introducing signed activations, mixed polysemy, or frequency mismatches) degrade ordered recovery or stability in practice. This leaves the theoretical contribution somewhat detached from the empirical setting.
- **High sensitivity to training schedules and under-specified optimization details:** Performance heavily depends on the prefix sampling distribution $p_{\text{ND}}(\ell)$ and the unit-sweeping freezing schedule, with no ablations quantifying their individual impact on the stability-reconstruction trade-off. Additionally, the paper does not specify how gradients are routed through the non-differentiable Top-$m$ operator (e.g., straight-through, auxiliary L0 regularization, or gating), which is essential for reproducibility and evaluating the true optimization geometry of the proposed loss.

## Nice-to-Haves
- Qualitative interpretability checks (e.g., activation maximization or steering tests on early vs. late prefix features) to assess whether the enforced ordering aligns with semantic or abstraction hierarchies, rather than purely statistical frequency.
- Benchmarking on established SAE evaluation suites (e.g., SAEBench) to contextualize the consistency gains against standard sparsity-L0 trade-offs and downstream feature utility metrics.
- Scaling the SAE stitching analysis across multiple dictionary size ratios to verify that the reduction in novel features reflects genuine coverage improvements rather than capacity bottlenecks.

## Novel Insights
The strict prefix objective fundamentally reframes sparse coding as a sequential feature curriculum rather than a parallel dictionary factorization. This constraint actively shrinks the permutation equivalence class, forcing the optimizer to converge on a canonical basis that prioritizes high-signal, high-frequency features early. Coupled with the FIFR metric, the work highlights a critical diagnostic blind spot in standard SAE training: global reconstruction error can remain deceptively low while rare but mechanistically meaningful features are consistently underfit or missed. Ordering, therefore, serves not just as a consistency prior, but as an implicit regularizer that exposes frequency-driven optimization imbalances in overcomplete dictionaries.

## Suggestions
- **Diagnose late-prefix dynamics:** Provide activation frequency histograms, gradient magnitude traces, and reconstruction contribution breakdowns per prefix index. This will clarify whether the post-~128 stability drop stems from natural data tail sparsity or from gradient starvation/collapse induced by the training schedule.
- **Ablate scheduling hyperparameters:** Systematically vary the prefix sampling distribution $p_{\text{ND}}(\ell)$ (e.g., uniform, geometric, adaptive) and the unit-sweeping freeze interval, reporting their effects on early vs. late stability and global reconstruction. This will provide practical guidelines for practitioners.
- **Clarify gradient routing for Top-$m$:** Explicitly document the differentiable approximation used for the hard Top-$m$ masking (e.g., straight-through gradients, continuous relaxation, or auxiliary sparsity penalties) and include a complete hyperparameter table (optimizer, LR schedule, batch size, exact prefix distribution form) to meet ICLR reproducibility standards.

---

## aZFjp5wqck

- GT: Reject (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
AgentVQA introduces a unified, offline multiple-choice benchmark aggregating 13,400 questions across five agentic domains (Web, Robotics, Egocentric Video, Games, and Spatial Understanding). By standardizing diverse trajectory and video tasks through a VLM-assisted hard-negative generation pipeline, it provides a scalable, deterministic alternative to compute-heavy online simulators. Evaluation across 15 VLMs reveals significant performance ceilings, substantial rank divergences from general-purpose benchmarks, and a detailed error taxonomy, offering actionable diagnostics for agentic model development.

## Strengths
- **Principled Construction & Empirical Validation:** The benchmark curation is rigorously justified, with subsampling analysis demonstrating <1% performance variance at 1,000 instances, robustness to option shuffling (~1.9% std dev), and clear refutation of random-guessing baselines (~25%). The pipeline effectively bridges fragmented domain-specific datasets and interactive simulators while preserving deterministic evaluation.
- **Diagnostic Divergence & Calibration Effect:** The benchmark successfully isolates agentic capabilities from passive visual understanding, evidenced by significant rank divergences from MMMU/GPQA and meaningful alignment with online environments (OSWorld, VideoGameBench). The MCQ format demonstrates a clear "calibration effect," rescuing scores penalized by brittle open-ended parsers while tightening overly lenient continuous metrics (e.g., IoU thresholds).
- **Actionable Error Taxonomy & Reasoning Insights:** The semi-automated classification reveals critical, domain-specific failure patterns (e.g., 46% grounding errors in Web Agents vs. 40% high-level reasoning breakdowns in Games). Importantly, the analysis challenges the ubiquity of chain-of-thought prompting, demonstrating it can degrade accuracy on perceptual tasks or induce recursive loops, pointing toward adaptive inference architectures.

## Weaknesses
- **Unaddressed Training Data Contamination:** The 14 source datasets are publicly available and widely integrated into VLM pretraining and instruction-tuning corpora. The paper lacks any decontamination filtering (e.g., embedding-based deduplication) or evaluation on temporally held-out/unseen trajectory splits. This risks inflating scores for frontier models and undermines both the "challenging" claim and the cross-benchmark divergence analysis.
- **Unquantified Human-in-the-Loop Verification:** The benchmark's difficulty relies on VLM-generated hard negatives, and its diagnostic insights rely on VLM-classified error modes. While manual review is mentioned, the paper provides no quantitative metrics on verification coverage, inter-annotator agreement, or failure rates of the automated pipeline. Without these statistics, the calibrated "hardness" of distractors and the reliability of the error taxonomy remain vulnerable to generator bias and annotation inconsistency.
- **Confounding Evaluation Hyperparameters:** The evaluation uses a temperature of 0.8 for reasoning models versus 0.2 for non-reasoning models. While potentially intended to accommodate diverse CoT sampling, this disparity acts as an uncontrolled variable when directly comparing performance gaps. Claims that reasoning variants underperform on perceptual tasks or outperform on strategic tasks remain partially confounded without a temperature-controlled ablation or explicit justification of cross-family evaluation fairness.
- **Anecdotal Characterization of "Thinking Loops":** The paper identifies repetitive reasoning cycles in certain models (e.g., GLM, Kimi) and uses this to recommend architectural decoupling of perception and planning. However, the finding is supported by only a single qualitative example in the appendix. Without quantifying the frequency and domain distribution of these loops across the 15 evaluated models, the observation remains illustrative rather than empirically substantiated.

## Nice-to-Haves
- Quantify distractor difficulty by reporting empirical selection frequencies per negative type or establishing a human/strong-model baseline to rigorously validate the "hard negative" claim.
- Strengthen offline-to-online predictive claims by reporting confidence intervals for correlation coefficients or employing robust rank-correlation methods when analyzing outlier sensitivity.
- Add a dedicated Limitations section explicitly detailing the inherent constraints of the static MCQ format for evaluating long-horizon planning, environmental non-determinism, and dynamic tool use, positioning the benchmark as a complementary diagnostic rather than a full simulator replacement.

## Novel Insights
The paper compellingly demonstrates that discretizing continuous agentic decision-making into rigorously calibrated MCQs does not dilute diagnostic value; rather, it exposes a "calibration effect" where the format filters out evaluation noise caused by brittle open-ended parsers and lenient grounding metrics. This reveals a fundamental semantic-spatial execution gap in current VLMs: models can often articulate correct high-level strategies but consistently fail at the low-level perceptual grounding required to act on them, suggesting that future agentic scaling must prioritize specialized grounding pre-training over purely architectural reasoning enhancements.

## Suggestions
- Implement and report explicit data decontamination protocols or construct a strictly held-out evaluation split to ensure reported scores reflect genuine zero-shot generalization rather than corpus familiarity.
- Provide quantitative metrics for the human verification step (e.g., percentage of samples reviewed, Cohen’s/Fleiss’ Kappa for error classification) to validate both the negative generation and diagnostic pipelines.
- Conduct a temperature sensitivity ablation across model families to isolate the impact of sampling parameters from intrinsic capabilities, particularly when contrasting reasoning and non-reasoning variants.
- Systematically quantify the incidence and domain distribution of "thinking loops" to substantiate the recommendation for adaptive inference mechanisms and decoupled perception-planning architectures.

---

## anLMfUzl0C

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (0.0/10)
- Match: N/A

### Final Review

## Summary
The paper introduces a confidence-driven RAG framework that quantifies LLM knowledge boundaries using internal hidden states before token generation. By measuring confidence shifts induced by retrieved contexts, it constructs a preference dataset to fine-tune a reranker and deploys Confidence-Based Dynamic Retrieval (CBDR) to adaptively skip retrieval for high-confidence queries. Experiments on Natural Questions and HotpotQA demonstrate improved context selection, modest accuracy gains, and reduced retrieval overhead when aligned with a target LLM.

## Strengths
- Efficient Preference Signal Construction: Using single-pass hidden-state confidence shifts to build reranker supervision data avoids the multi-round sampling overhead of prior methods like SEAKR, offering a computationally lean and conceptually clear pipeline (Sec 2.2, Sec 3.2).
- Transparent Mechanistic Analysis of Model Dependency: The paper rigorously investigates and explains why the fine-tuned reranker's benefits vanish on Qwen2.5, demonstrating through controlled experiments that cognitive preferences for context types (e.g., semantic match vs. factual density) differ significantly across architectures (Sec 5.2, Appendix G).
- Actionable Accuracy-Efficiency Trade-off Mapping: The CBDR threshold analysis reveals non-obvious deployment dynamics, such as how excessive retrieval at extreme $\beta$ values actively degrades accuracy due to knowledge conflicts, providing valuable engineering insights for real-world system tuning (Table 3, Sec 5.4).

## Weaknesses
- Uncalibrated Probability Metrics Undermine Thresholding & Ranking — The confidence probe is trained as a binary classifier on factual correctness, yet its softmax output is treated as a continuous confidence metric for both the $\beta$ retrieval gate and the $\Delta Conf$ preference ranking. Binary logits are notoriously poorly calibrated; without explicit calibration (e.g., temperature scaling or Platt scaling) or calibration reporting, the margin-based ranking and threshold decisions may reflect uncalibrated model artifacts rather than reliable epistemic uncertainty, risking unstable retrieval gating.
- Risk of Rewarding Fluent but Factually Incorrect Contexts — The preference signal assumes a monotonic relationship between confidence increase and factual utility. LLMs frequently exhibit higher confidence when exposed to highly plausible, stylistically coherent, but factually conflicting information. Without quantifying the false-positive preference rate or explicitly disentangling surface plausibility from factual grounding, the reranker may learn to prioritize persuasive hallucinations, undermining the core claim that confidence shifts reliably identify beneficial knowledge.
- Narrow Empirical Scope & Missing Direct Preference-Aligned Comparisons — Evaluation is restricted to two 7–8B instruction-tuned models and two factoid QA datasets. While the cross-model limitation is acknowledged, the lack of validation on reasoning-heavy, long-form, or domain-specific corpora weakens robustness claims. Moreover, despite positioning the confidence-shift signal as a superior alternative to methods like SEAKR, DPA-RAG, or RE-PLUG, the paper lacks head-to-head empirical comparisons. Without them, it remains unclear whether the single-pass signal matches or exceeds the accuracy of existing rationale- or probability-based preference alignment strategies.
- Qualitative Efficiency Metrics Weaken Practical Claims — The core motivation emphasizes RAG efficiency, yet Table 3 relies on qualitative cost labels ("High/Mid/Low") and reports retrieval savings only as a skip-ratio percentage. The overhead of running both the target LLM forward pass and the probe on every query, plus the reranker when triggered, is not quantified in wall-clock latency, throughput, or FLOPs. This omission makes it difficult to verify whether the framework genuinely delivers the "substantially enhancing practical utility" claimed in the abstract.

## Nice-to-Haves
- Reporting variance metrics (e.g., multiple random seeds, bootstrap confidence intervals) for marginal accuracy and retrieval gains would help assess stability, though single-run evaluation is common in large-scale LLM benchmarking.
- Exploring adaptive or data-driven threshold calibration for $\beta$ (e.g., validation-set Pareto optimization or confidence percentile tracking) could improve robustness over the current fixed grid search, particularly under distribution shift.
- Investigating whether lightweight adapter fine-tuning or confidence signal normalization could partially decouple the reranker from the target LLM, reducing the need to reconstruct the preference dataset for every new base model.

## Novel Insights
The most compelling insight is the mechanistic demonstration that RAG rerankers optimized purely on query-passage semantic similarity frequently misalign with the downstream LLM's actual cognitive processing preferences. By exposing this misalignment through internal state probing, the paper reveals that different LLMs possess distinct, quantifiable biases toward contextual features (e.g., one model tolerating lower semantic matching for richer factual chains, while another strictly enforces high lexical alignment). This finding fundamentally shifts reranker evaluation from static, input-level relevance to dynamic, model-aware utility maximization, establishing that optimal external knowledge integration is inherently architecture-specific rather than universally transferable.

## Suggestions
- Apply and report calibration metrics (e.g., Expected Calibration Error) for the confidence probe, and rerun the threshold & ranking experiments with calibrated probabilities to ensure the continuous signal reliably reflects true utility margins.
- Include at least one recent preference-aligned reranking baseline (e.g., SEAKR or a rationale-distillation method) in Table 2/3 comparisons to empirically verify whether the confidence-shift signal provides competitive or superior accuracy relative to existing LLM-preference approaches.
- Systematically measure the false-positive preference rate by analyzing how often $\Delta Conf > 0$ correlates with a final incorrect answer, specifically under retrieved contexts that are fluent but factually contradictory to ground truth.
- Replace qualitative cost labels with concrete latency and computational overhead metrics (e.g., ms/query, tokens/second, or GPU memory footprint) for the full CBDR pipeline to rigorously substantiate the framework's efficiency claims for production deployment.

---

## VlTHxRcP3A

- GT: Reject (avg 1.0)
- Predicted: N/A (2.8/10)
- Match: N/A

### Final Review

## Summary
This paper systematically compares linear and nonlinear (LSTM, FingerFlex) decoders for predicting 3D hand position and velocity from Neuropixels spike trains in mice performing a reach-to-grab task. By introducing artificial temporal lags and analyzing the geometry of learned decoder weights, the authors demonstrate that position decoding consistently outperforms velocity, and that position-predictive models learn smooth, lag-invariant linear projections while velocity-predictive models yield fragmented, delay-specific solutions. The work concludes that both model families may function primarily as static projectors onto behaviorally correlated manifolds rather than capturing true temporal neural dynamics.

## Strengths
- **Principled temporal perturbation framework:** The lag-sweep approach (-200ms to +200ms with post-hoc realignment to movement onset) provides a clean, interpretable diagnostic for probing causal alignment versus superficial spatial correlation, directly testing whether models leverage genuine temporal structure.
- **Informative representational analysis:** Visualizing linear decoder weights via UMAP and cosine similarity across lag conditions (Figure 3) offers a novel, geometry-level contrast between position and velocity decoding. The finding that position weights form a smooth continuum while velocity weights fracture into delay-specific clusters provides concrete evidence that target signal statistics directly shape learned projection spaces.
- **Methodologically careful evaluation protocol:** The use of nested cross-validation, retention of negative $R^2$ values without truncation, and the explicit handling of single-trial variability (via concatenated-sequence evaluation) demonstrate rigorous experimental design and prevent the overoptimistic performance estimates common in neural decoding.

## Weaknesses
- **Velocity derivation likely confounds neural encoding with signal processing artifacts:** The paper applies an 8th-order central difference filter to "get smoother velocity" (Sec 2). High-order finite difference operators approximate derivatives and inherently amplify high-frequency noise; they do not smooth. This likely inflates target variance and tracking noise, artificially degrading velocity $R^2$ and potentially causing the observed "fragmented" weight manifolds. The core claim that velocity is harder to decode neurally cannot be disentangled from this methodological artifact without a control using standard kinematic smoothing (e.g., Savitzky-Golay, spline fitting, or APT's built-in velocity outputs).
- **Unsupported "static projector" conclusion lacks null controls:** The conclusion that decoders "primarily act as projectors... without truly capturing the underlying temporal dynamics" (Sec 3.2) is prematurely drawn from the observation that LSTM and linear models perform similarly across small lags. Neural spike trains and kinematic traces are strongly autocorrelated; high instantaneous correlations mean shallow models can achieve strong performance without complex temporal integration. Without temporal permutation baselines, trial-shuffled controls, or analysis of LSTM temporal receptive fields/weights over time, the data cannot distinguish static projection from temporal modeling that simply yields diminishing returns on already-predictive instantaneous features.
- **Internal contradiction and missing architectural specifications undermine reproducibility:** Section 3 states the "number of training epochs is 1000," but the very next sentence reads, "Each deep learning model was trained for 3000 epochs." Additionally, critical LSTM/FingerFlex hyperparameters (sequence/window length, hidden dimensions, number of layers, dropout, gradient clipping) are omitted. Applying an ECoG-optimized CNN to binned spike trains without domain-adaptation details further obscures whether architectural mismatch or genuine representational limits explain the linear/nonlinear parity.
- **Extreme data scale limits claims about generalizable architecture behavior:** The analysis relies on 3 sessions from a single mouse. While the authors acknowledge this in Section 4.1, the paper's conclusions make broad claims about decoder architectures ("rather than interpreting the underlying dynamics... both linear and nonlinear models tend to function as projection mechanisms"). Single-animal, short-duration results cannot rule out idiosyncratic recording conditions, brain region specificity, or task-phase confounds, making these architecture-level generalizations premature for an ML venue.

## Nice-to-Haves
- Report UMAP hyperparameters (`n_neighbors`, `min_dist`, metric) and random seeds to ensure the "smooth vs. fragmented" visualization is reproducible.
- Include confidence intervals or non-parametric permutation tests for cross-architecture $R^2$ comparisons to formally substantiate claims of "no noticeable improvement" or "markedly superior" performance.
- Visualize temporal receptive fields (decoder weights across input time bins) or LSTM impulse responses to directly inspect whether models develop meaningful temporal filters or collapse to near-static mappings.
- Clarify whether downstream BCI or scientific applications should prioritize trial-wise vs. concatenated $R^2$, and justify the choice in context.

## Novel Insights
The paper reveals that the frequency content and temporal structure of the *target variable*—rather than solely neural encoding properties—dictate the geometry of learned decoding manifolds. Position, being a slowly varying, low-pass signal, naturally aligns with a smooth, lag-invariant subspace that simple linear projectors can stably map. Velocity, a high-frequency derivative, forces decoders to learn rapidly shifting, delay-specific linear filters, resulting in fragmented weight-space topology regardless of model capacity. This observation reframes the linear vs. nonlinear debate in neural decoding: when target labels are high-variance and temporally misaligned relative to model receptive fields, increased architectural complexity does not stabilize representations. Instead, representational continuity emerges from the alignment between label smoothness and the effective integration window of the decoder.

## Suggestions
1. Replace or rigorously justify the 8th-order velocity derivative. Compute velocity using standard kinematic smoothing (e.g., Savitzky-Golay with validated window length or tracking software's native differentiation), and report how this change affects $R^2$ and weight-space fragmentation.
2. Add temporal null baselines: train decoders on trial-shuffled or time-scrambled behavioral targets to establish chance $R^2$ and quantify how much performance actually derives from autocorrelation vs. genuine neural-behavioral alignment.
3. Resolve the epoch count contradiction and report full architecture specifications (LSTM layers/hidden units, sequence length, FingerFlex adaptation details for spike trains). 
4. Soften mechanistic claims to align with the data: reframe the "projector" conclusion as a strong empirical observation under current data conditions, and explicitly state that distinguishing static projection from temporal integration requires permutation controls or receptive field analysis in future work.

---

## lSM6MtjQcM

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces AetherCode, a competitive programming benchmark comprising 456 recent problems sourced from premier OI and ICPC contests (2024–2025). To address systemic evaluation bias in prior code benchmarks, the authors implement a hybrid test-case construction pipeline combining a Generator-Validator agent system with expert curation, achieving 100% True Positive and True Negative Rates against a corpus of >30k human submissions. Evaluations across 17 frontier reasoning and non-reasoning models reveal strong discriminative power, consistent stratification by reasoning capability, and clear domain-specific bottlenecks.

## Strengths
- **Rigorous classifier-style test-case validation:** The paper departs from traditional quantity-based test suite metrics by framing validation as a binary classification task (TPR/TNR) against a curated corpus of correct/incorrect human solutions. Achieving 100% on both metrics meaningfully addresses the documented evaluation bias from incomplete or invalid test cases in legacy benchmarks.
- **High-fidelity, contamination-resistant problem sourcing & taxonomy:** Systematic collection of recent premier contest problems with manual proofreading, Markdown+LaTeX conversion, and expert-driven categorization across 10 major domains and 144 fine-grained tags enables precise diagnostic tracking of model capabilities and reduces data contamination risks.
- **Actionable empirical stratification:** Beyond aggregate scores, the evaluation clearly differentiates reasoning vs. non-reasoning models, identifies concrete domain bottlenecks (e.g., consistent struggles with Computational Geometry and Tree Structures), and surfaces distinct failure patterns (e.g., correctness-efficiency trade-offs in Claude models), providing targeted signals for model development.

## Weaknesses
- **Unverified central motivation & missing human baselines:** The paper claims existing benchmarks "overstate model proficiency" and that a "significant gap" exists compared to elite humans, yet provides no empirical head-to-head comparison against benchmarks like LiveCodeBench on identical models, nor any human performance baseline (e.g., official solve rates, medalist scores) on AetherCode. This leaves the core motivation assertion qualitatively framed rather than empirically anchored, weakening the paper's foundational argument.
- **TNR robustness is bounded to human failure distributions:** The 100% TNR guarantee is measured exclusively against ~30k historical *human* submissions. LLMs exhibit qualitatively different failure modes (e.g., hallucinated algorithmic logic, structural prompt-following errors, or inefficient but syntactically correct code). Without stress-testing against LLM-generated incorrect solutions or explicitly quantifying the marginal gain of the expert curation phase over the G-V agent, the generalization of the TNR metric to future LLM failure patterns remains unproven.
- **Evaluation protocol ambiguities (language handling & environment calibration):** The prompt template specifies `{LANGUAGE}`, but the execution environment details only `gcc 9.4.0, C++17`. It is unclear whether non-C++ submissions are evaluated, how time limits are calibrated for slower languages (e.g., Python multipliers standard in official CP judging), or how the 2-core/4GB Docker sandbox aligns with official contest memory/time limits. This introduces potential confounders where TLE/MLE verdicts may reflect environment or language policy mismatches rather than genuine algorithmic reasoning deficits.
- **Difficulty normalization transparency & fine-grained statistical volatility:** Harmonizing difficulty across fundamentally different contest formats (IOI's multi-day partial scoring vs. ICPC's pass/fail ACM style) relies on expert judgment and solve rates without explicit quantitative normalization methodology. Additionally, fine-grained category analyses (Table 4) are derived from small absolute counts (e.g., ~48 Tree problems), making low-percentage differences highly volatile and prone to overinterpretation without reported uncertainty metrics.

## Nice-to-Haves
- Report mean ± variance or bootstrapped confidence intervals across the 4 evaluation runs, particularly for low-hit-rate categories, to prevent overinterpretation of minor percentage gaps.
- Expand the qualitative failure diagnosis beyond `o4-mini-high` to include the top 2–3 reasoning models, clarifying whether observed inefficiency/compile error patterns are model-specific or systemic.
- Clarify how time/memory limits were calibrated for the sandbox environment relative to official contest constraints, and briefly discuss how the benchmark could be extended to support partial-credit scoring (OI-style) or iterative agentic workflows.

## Novel Insights
AetherCode demonstrates that benchmarking fidelity can outweigh sheer scale: a compact, expert-validated suite with rigorously constructed test cases exposes capability boundaries that larger, mutation-based or externally-hosted evaluations obscure. The clear stratification between reasoning and non-reasoning models, coupled with consistent domain bottlenecks (notably in abstract algorithmic composition like Geometry and Tree Problems), suggests that current LLMs still lack robust efficiency-aware reasoning rather than mere code generation fluency. Furthermore, the substantial performance gains observed with increased sampling (Pass@1 to Pass@4) indicate a paradigm shift where frontier models are transitioning from deterministic algorithmic derivation to stochastic solution-space exploration, marking a new bottleneck for achieving reliable competitive programming proficiency.

## Suggestions
- Conduct a head-to-head evaluation on a strictly overlapping problem subset with LiveCodeBench or CodeContests to quantitatively validate the "overstatement" claim, and incorporate official human performance metrics (e.g., median solve rates by contest tier) to empirically anchor the difficulty labels and substantiate the human-model gap.
- Explicitly document language support policies (including any time multipliers), detail how time/memory limits were stress-tested for algorithmic complexity constraints, and release the full evaluation harness, G-V agent prompts, and verified solution corpus under a clear academic license to ensure reproducibility and community adoption.

---

## mHRuCmc9lo

- GT: Accept (Poster) (avg 7.3)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary
This paper develops a minimax decision-making framework for acting on forecasts that satisfy only partial ($\mathcal{H}$-) calibration guarantees. By formulating the decision problem as a convex optimization over the ambiguity set of outcome distributions consistent with $\mathcal{H}$-calibration, the authors derive a closed-form characterization of the optimal robust policy via duality. The central theoretical contribution is a "sharp transition" result: once $\mathcal{H}$ contains the indicator functions of the decision regions, the minimax-optimal policy collapses exactly to the plug-in best response. The framework is applied to structural calibration properties arising from standard squared-loss training and bin-wise post-hoc calibration, with empirical validation on 1D regression tasks under worst-case, calibration-respecting outcome distributions.

## Strengths
- **Elegant theoretical collapse result:** Theorems 4.1 and 4.2 rigorously prove that decision calibration, a tractable and widely studied weaker notion, is sufficient to collapse the minimax robust policy to the plug-in best response. This elevates decision calibration from a swap-regret guarantee to a full minimax optimality condition, providing clear decision-theoretic justification for its use in high-dimensional settings where full calibration is intractable.
- **Strong grounding in standard ML pipelines:** Proposition 4.4 correctly identifies that any model with a linear final layer trained to a first-order stationary point of squared loss inherently satisfies a self-orthogonality $\mathcal{H}$-calibration property. This insight usefully demonstrates that the robust decision framework can be deployed without algorithmic forecaster modification or post-hoc recalibration, leveraging "free" structure from ubiquitous training protocols.
- **Clean problem formulation and tractable characterization:** The ambiguity set construction and dual characterization (Theorem 3.1) provide a principled bridge between aggressive best-response and conservative minimax strategies. The pointwise computability of the robust policy and the explicit interpolation property make the framework practically actionable rather than purely abstract.

## Weaknesses
- **Empirical validation mismatches the core motivation and lacks algorithmic transparency:** The paper motivates the framework by highlighting the exponential sample complexity of full calibration in *high dimensions* ($d > 1$), yet all experiments are confined to 1D regression proxies. This leaves the computational overhead, conditioning, and practical scaling of the dual optimization for multiclass or vector-outcome settings entirely unverified. Furthermore, Section 5 reports utility under worst-case distributions tailored to each policy but omits the algorithmic procedure for constructing these calibration-respecting adversaries. Without specifying whether these are derived via projection, linear programming, or sampling, reproducibility is hindered and it is unclear whether the reported utility margins stem from the robust policy design or artifacts of the adversary generation process.
- **Terminology overclaims regarding "distribution shift":** The abstract and introduction frame the robust policy as handling "calibration-preserving distribution shift." However, the experiments do not evaluate covariate shift, temporal drift, or real-world domain transfer. Instead, they synthesize worst-case conditional expectations that strictly satisfy the population moment constraints on a static test split. This validates the duality construction under worst-case realizations, not robustness to actual distributional change. The terminology should be corrected to avoid implying guarantees that the experimental setup does not support.

## Nice-to-Haves
- **Integrate approximate $\mathcal{H}$-calibration analysis into the main narrative:** Appendix B provides valuable $O(mL\epsilon)$ bounds for $\epsilon$-approximate calibration and $\epsilon$-bin-wise calibration. Since finite-sample estimation of calibration moments from a held-out split inherently introduces slack, briefly discussing how this affects the sharp transition threshold in the main text would significantly strengthen the practical deployment narrative.
- **Visualize the adversarial belief shift and transition regime:** Plotting $q^\star(v) - v$ across the forecast domain would empirically verify how the dual optimization tilts probability mass toward worst-case boundaries. Additionally, graphing the utility advantage of the robust policy as a function of increasing calibration violation magnitude (controlled $\epsilon$-relaxation) would empirically validate the theoretical crossover point.
- **Contextualize against standard recalibration pipelines:** While out of the primary scope of deriving decision rules, a brief comparison of the proposed robust policy against plug-in best-response applied after standard post-hoc methods (e.g., isotonic regression or histogram binning) would help practitioners gauge the marginal benefit of the minimax framework over established calibration workflows.

## Novel Insights
The paper reframes calibration not as a binary target but as a quantifiable spectrum of information that directly dictates optimal conservatism. The most significant insight is the identification of a precise threshold where "trust" becomes minimax-optimal: once the calibration tests explicitly cover the decision regions relevant to the downstream agent, the adversarial ambiguity set collapses sufficiently that no policy can outperform the plug-in best response. This collapses what could be an infinite hierarchy of robust policies into a binary regime, offering a clear, theoretically grounded target for both forecaster design and downstream decision system engineering. It demonstrates that robustness to partial calibration does not always require complex conservative adjustments; it often requires simply ensuring the forecaster's calibration tests align with the decision-maker's utility geometry.

## Suggestions
- Provide pseudocode or a clear algorithmic description in Section 5 for constructing the worst-case, calibration-respecting outcome distributions used in Table 1, and release the corresponding code to ensure full reproducibility.
- Correct the terminology throughout to refer to "worst-case admissible realizations" or "calibration-consistent outcome perturbations" rather than "distribution shift," and explicitly state that the experiments validate theoretical minimax guarantees rather than real-world covariate/concept shift robustness.
- Add a brief discussion or footnote in Section 3 addressing the computational complexity and conditioning of the dual optimization as $d$ and $|A|$ scale, and consider adding a small multiclass or multi-output regression experiment to demonstrate that the dual solver remains stable and tractable beyond the 1D setting.

---

## Ilnbgf1eeS

- GT: Withdrawn (treated as Reject) (avg 1.0)
- Predicted: N/A (1.8/10)
- Match: N/A

### Final Review

## Summary
This paper addresses the Lottery Ticket Hypothesis (LTH) initialization dependency by training pruned networks using Bayesian methods, specifically Hamiltonian Monte Carlo (HMC) and Stochastic Variational Inference (SVI), rather than rewinding to the original weights. The authors demonstrate that distributional training can recover or exceed lottery-ticket performance across extreme sparsity levels (up to 99%) on small architectures, while empirically showing that carefully tuned SVI offers a computationally scalable alternative to full-batch HMC for larger models like ResNet-18.

## Strengths
- **Conceptual decoupling of masks from fixed initializations:** The paper accurately identifies a major practical limitation of LTH and logically reframes initialization as a prior distribution. Treating the weight space probabilistically to search for optimal posteriors, rather than relying on a single rewound sample, provides a principled theoretical pathway for evaluating mask quality independent of arbitrary starting points.
- **Pragmatic demonstration of SVI over HMC for scalable sparse training:** The empirical contrast between HMC and SVI is highly informative. By showing that SVI, when properly regularized via population-based KL-weight tuning, matches or exceeds lottery-ticket performance on ResNet-18 within standard training times, the paper successfully bridges a theoretically exact sampling framework with scalable deep learning practice.

## Weaknesses
- **Mismatch between asymptotic claims and empirical statistical validation:** The title and abstract heavily emphasize HMC's theoretical convergence guarantees and assert that "Bayes always wins," yet Section 5 explicitly documents practical non-convergence (up to 5% accuracy variance for LeNet-5 after 2800+ samples). Furthermore, Table 1 reports single-run metrics without standard deviations or seeds for the primary methods. Without multi-seed statistical reporting, the claimed 3–4% improvements over lottery initialization cannot be distinguished from stochastic variance, severely undermining the empirical reliability of the core contribution.
- **Unclear inference protocol and methodological inaccuracies regarding gradients:** Section 3.2 incorrectly states that the method avoids "backpropagation based gradient descent training." HMC fundamentally requires exact gradients of the log-posterior, which are computed via backpropagation to simulate Hamiltonian dynamics; the distinction is between gradient-based *optimization* and gradient-based *sampling*, not gradient elimination. More critically, the paper never specifies the test-time mechanism: if predictions use the posterior mean as a single weight matrix, the approach functions as a deterministic optimizer, raising doubts about whether full Bayesian machinery was necessary or merely acted as an implicit regularizer. If full Bayesian model averaging is required for inference, the computational overhead scales linearly with the number of samples, completely negating the storage and FLOPs efficiency gained from 99% pruning.

## Nice-to-Haves
- **Diagnostic analysis of posterior variance collapse:** Plotting or analyzing whether learned weight variances concentrate to near-zero values during training would clarify if the Bayesian model captures meaningful uncertainty or effectively degenerates into a point-estimate smoother.
- **Standard LTH recovery baselines:** Including comparisons against simple learning-rate rewinding or early-bird scheduling would help contextualize whether the Bayesian objective provides a distinct optimization advantage or simply matches strong deterministic fine-tuning heuristics.
- **MCMC convergence diagnostics:** Providing trace plots or effective sample size (ESS) metrics for the high-variance LeNet-5 runs would empirically characterize the reported convergence failure beyond anecdotal accuracy dips.

## Novel Insights
The paper's empirical trajectory reveals a subtle but important insight about sparse network optimization: what matters for recovering highly pruned masks is not asymptotically exact posterior sampling, but rather the implicit smoothing of the non-convex loss landscape provided by distributional objectives. The necessity to manually balance the KL-divergence term in SVI to achieve competitive performance suggests that the core benefit of the Bayesian approach lies in preventing the optimizer from collapsing into sharp, initialization-dependent minima that trap standard SGD. By integrating over weight uncertainty, the method effectively bypasses the local geometry constraints that make lottery tickets highly sensitive to their initial draw, reframing LTH success as a property of accessible posterior modes rather than unique weight configurations.

## Suggestions
- Explicitly define the test-time inference protocol (posterior mean point-estimate vs. predictive model averaging) and report the corresponding parameter count/latency to verify whether pruning efficiency gains are preserved in deployment.
- Retrain all HMC and SVI configurations across multiple random seeds (≥5), report means with standard deviations, and temper the title/abstract claims to accurately reflect finite-sample empirical performance rather than asymptotic guarantees.
- Correct Section 3.2 to accurately describe HMC as a gradient-informed MCMC sampler (dependent on backpropagation for leapfrog steps) and document how Batch Normalization statistics are handled during stochastic Bayesian sampling on ResNet-18.

---

## kkBOIsrCXh

- GT: Accept (Poster) (avg 8.0)
- Predicted: N/A (6.7/10)
- Match: N/A

### Final Review

## Summary
This paper introduces NavFoM, a unified vision-language-action model for embodied navigation trained on over 12.7 million samples spanning diverse tasks (VLN, object search, tracking, autonomous driving) and robot embodiments (quadrupeds, UAVs, wheeled robots, cars) without task-specific fine-tuning. The architecture extends standard video-VLMs with Temporal-Viewpoint Indicator (TVI) tokens for explicit multi-view/time alignment and a Budget-Aware Temporal Sampling (BATS) strategy to manage historical context under fixed token budgets, achieving state-of-the-art or highly competitive results across seven public benchmarks and validated real-world deployments.

## Strengths
- **Strong cross-task and cross-embodiment generalization without fine-tuning:** NavFoM achieves consistent zero-shot improvements across fundamentally different navigation paradigms, e.g., improving VLN-CE RxR single-view SR from 51.8% to 57.4% and reaching 45.2% SR on HM3D-OVON, matching or exceeding specialized baselines (Tables 1, 2, 7, 8). This demonstrates that a single unified policy can effectively bridge heterogeneous navigation domains when trained at scale.
- **Practical, well-motivated architectural components with clear empirical validation:** TVI tokens cleanly decouple viewpoint, temporal, and base embeddings to enable seamless co-training with Image/Video QA data, while BATS provides a mathematically grounded exponential sampling curve that preserves recent context under strict token budgets. Ablations (Table 8, Fig 4) confirm both components are necessary and outperform naive baselines like uniform sampling or token merging.
- **Comprehensive real-world validation and deployment transparency:** Beyond simulation, the paper reports results on 110 reproducible physical test cases across multiple robot platforms, alongside detailed deployment metrics (19.1 GB VRAM at 5 Hz on RTX 4090, successful 4-bit quantization to 10.7 GB, and Jetson Thor compatibility). This practical focus, combined with clear training compute documentation (4,032 H100-hours) and open dataset sourcing, strongly supports reproducibility and real-world applicability.

## Weaknesses
- **Unverified impact of noisy pseudo-labeled navigation data:** The training set includes ~2.03M web-video navigation samples with VLM-generated instructions and SLAM trajectories, which the authors acknowledge are "imperfect." However, no ablation isolates this subset or quantifies how its noise affects convergence and final performance. Without controlled filtering or ablation, it remains unclear whether reported gains stem from genuine real-world coverage or are confounded by label artifacts and trajectory inconsistency.
- **Lack of statistical robustness reporting for main results and ablations:** All primary benchmark tables and component ablations report single-point metrics without standard deviations, confidence intervals, or multi-seed averages. Given the fine-tuning nature of the pipeline and the high variance inherent in continuous navigation tasks, the absence of variance metrics makes it difficult to assess whether improvements (e.g., +5.6% SR on RxR) are statistically significant or susceptible to seed-dependent fluctuations.

## Nice-to-Haves
- Analyze gradient or feature interference between jointly trained navigation tasks and QA objectives, or experiment with dynamic loss weighting to replace the static $\beta=10$ multiplier.
- Provide token utilization or attention heatmaps showing how the LLM allocates focus across camera views and historical timesteps under TVI/BATS guidance, rather than collapsing to recent frames.
- Quantify the sim-to-real performance gap by explicitly measuring success/failure rate drops caused by real-world perception challenges (motion blur, lighting shifts, sensor noise) compared to simulation rollouts.
- Profile end-to-end latency and memory throughput across varying sequence lengths (short indoor VLN vs. long-horizon driving) to empirically validate BATS scaling claims beyond theoretical bounds.

## Novel Insights
The paper demonstrates that achieving robust cross-embodiment navigation does not require complex, dynamics-aware control heads or embodiment-specific architectures; instead, careful token-level organization of temporal, spatial, and modality information within a standard VLM backbone is sufficient to unify highly heterogeneous navigation tasks. The consistent zero-shot gains from single-view to multi-view across all evaluated tasks further suggest that cross-task generalization in embodied AI is heavily bottlenecked by context alignment and history management rather than policy capacity, and that explicit token organization paired with budget-aware sampling can effectively mitigate these bottlenecks without architectural overhaul.

## Suggestions
1. **Ablate the Sekai web-video navigation subset:** Train and evaluate with/without this ~2M sample subset, or apply incremental noise-filtering thresholds, to quantitatively isolate its contribution to generalization versus potential performance degradation from imperfect trajectories/instructions.
2. **Report multi-seed statistics:** Re-run the primary benchmark evaluations and key ablations (TVI variants, BATS vs. baselines) across at least three random seeds and report mean ± standard deviation. This will solidify confidence in the claimed improvements and meet standard empirical rigor expectations.
3. **Briefly analyze training stability under fixed loss balancing:** Given the known scale mismatch between MSE trajectory loss and cross-entropy QA loss, provide a short empirical analysis of convergence curves or gradient norms across tasks to demonstrate that the fixed $\beta=10$ weighting does not cause catastrophic interference or task dominance during the single-epoch fine-tuning phase.

---

## 1NZ3DHF9nT

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.9/10)
- Match: N/A

### Final Review

## Summary
Fast-dLLM v2 introduces a data-efficient post-training framework that adapts pretrained autoregressive LLMs into block-diffusion models using only ~1B fine-tuning tokens. By combining complementary masking, a hybrid block-causal attention scheme, and hierarchical KV caching with confidence-aware parallel decoding, the method achieves substantial inference speedups (up to 2.5×) over standard AR decoding while maintaining competitive performance across reasoning, code, and instruction-following benchmarks.

## Strengths
- **High Data Efficiency & Practical Adaptation Pathway:** The core contribution is a remarkably efficient adaptation pipeline. By leveraging a block-wise architecture that structurally aligns with pretrained AR models and requiring only ~1B tokens of fine-tuning (versus ~500B cited for full-attention dLLM adaptations), the paper demonstrates a low-compute, highly accessible route to diffusion-style decoding without retraining from scratch. (Sec 4.1, App A.1)
- **Rigorous Inference Engineering & Validation:** The hierarchical caching mechanism (block-level + sub-block DualCache) and confidence-threshold tuning are systematically validated. The ablation studies (Tables 2–4, Fig 6) clearly isolate the contributions of padding, complementary masking, and sub-block sizing, demonstrating that caching yields pure efficiency gains without degrading outputs, and that tuning the sub-block size (optimally to 8) effectively recovers performance lost to training-inference block-size mismatches.
- **Strong Empirical Gains in Reasoning & Instruction Following:** The 7B variant establishes a new efficiency-quality frontier for diffusion LLMs, notably improving upon both the original Qwen2.5-7B and strong dLLM baselines (LLaDA, Dream) on mathematical reasoning (GSM8K: 83.7 vs 71.4/78.6) and instruction adherence (IFEval: 83.7 vs 71.4/81.0), while matching AR throughput scaling on A100/H100 hardware. (Table 1, Fig 5)

## Weaknesses
- **Conflation of "Lossless" Quality and Accelerated Speedup:** The abstract and introduction claim "lossless adaptation" alongside a "2.5× speedup," but these metrics are derived from disjoint experimental settings. The primary accuracy results (Table 1) are reported at a confidence threshold of 1.0 (standard sequential denoising), while the efficiency claims and throughput charts rely on a threshold of 0.9 to enable parallel unmasking. Figure 4 explicitly shows a measurable GSM8K accuracy drop (~3 points) at this accelerated threshold. The paper lacks a comprehensive table evaluating all benchmarks at the accelerated threshold, leaving it unclear whether the practical 2.5× speedup incurs unacceptable quality degradation on knowledge-heavy or complex reasoning tasks.
- **Performance Regression on Knowledge-Intensive Benchmarks:** While the model excels in structured reasoning and code, it noticeably underperforms the Qwen2.5-7B-Nemo-FT baseline (standard next-token prediction fine-tuned on identical data/steps) on MMLU (61.4 vs. 69.5) and GPQA (31.9 vs. 34.2). This suggests that the block-diffusion objective and complementary masking strategy may inadvertently sacrifice factual retention or dense knowledge recall compared to a conventional SFT baseline of equivalent compute, a systematic trade-off that is currently unacknowledged.
- **Under-Specified Speedup Conditions:** The throughput advantages of diffusion models are highly sensitive to generation length due to iterative sampling overhead, yet the exact output sequence length, batch sizes, and software optimization level (e.g., custom Triton kernels vs. eager mode) yielding the peak 2.5× figure are not consistently reported alongside the latency charts. Without these parameters and a clear baseline optimization context, it is difficult to contextualize the raw latency gains against heavily optimized AR production baselines.

## Nice-to-Haves
- **Comparison to Speculative Decoding:** A direct empirical comparison against state-of-the-art AR speculative decoding methods (e.g., EAGLE, Medusa) at the 7B scale would help contextualize whether the ~2× block-parallelism gain is competitive with, or complementary to, the current industry standard for low-latency LLM serving.
- **Open-Ended Generation & Stochasticity Analysis:** Including perplexity, repetition rates, or diversity metrics on open-ended prompts would help verify that the diffusion objective does not induce subtle mode collapse. Quantifying output variance compared to deterministic AR decoding would also clarify the practical reliability of the model.
- **Architectural Generalization & VRAM Profiling:** A brief discussion or micro-experiment on how the token-shift and complementary masking recipe transfers to architectures with different RoPE implementations or MoE layers would strengthen the framework's generality. Additionally, reporting peak VRAM usage per batch size would clarify the actual serving footprint of the dual-cache mechanism.

## Novel Insights
The paper effectively demonstrates that block-diffusion is not merely a theoretical intermediate step, but a highly practical engineering compromise that unlocks AR-compatible KV caching while preserving bidirectional intra-block refinement. The finding that complementary masking paired with strict block-causal boundaries solves the training/inference representation mismatch—and that sub-block decoding acts as a tunable "resolution dial" to recover performance during inference—reveals a nuanced understanding of how discrete diffusion temporal dynamics align with transformer attention patterns. This shifts the dLLM paradigm away from requiring massive from-scratch training toward a lightweight, highly compatible post-training adaptation strategy.

## Suggestions
- Provide a benchmark table reporting accuracy across all evaluated tasks (GSM8K, MMLU, HumanEval, etc.) using the accelerated confidence threshold (e.g., 0.9). Transparently quantify the accuracy-speed trade-off to ground the "lossless" claim, and explicitly adjust the language to reflect the precise conditions under which quality is preserved versus sacrificed.
- Detail the exact generation length, batch settings, and baseline optimization levels used for the peak speedup claims. Explicitly discuss the training-time computational overhead of concatenating $x_t$ and $x_0$ (resulting in $2L$ sequences) compared to standard SFT, and address the observed MMLU/GPQA regression relative to the NTP-FT baseline in a formal Limitations section.

---

## zwfpyw345l

- GT: Reject (avg 0.5)
- Predicted: N/A (1.4/10)
- Match: N/A

### Final Review

## Summary
The paper proposes a hierarchical code representation for reinforcement learning that integrates token-level Transformer encoding with function-level (AST) and module-level (dependency graph) attention, optimized end-to-end via PPO. It claims that this multi-granular state representation captures both local syntax and global program semantics, demonstrating improved sample efficiency and code generation quality across completion, repair, and algorithmic tasks compared to flat or single-level baselines.

## Strengths
- **Well-aligned multi-granular architecture**: The explicit decomposition of code processing into token, function, and module hierarchies aligns naturally with software engineering principles. The ablation results empirically validate this design, showing measurable performance drops when each level is removed, supporting the core hypothesis that granular structure matters.
- **End-to-end RL optimization**: By jointly training the hierarchical encoder with the policy objective rather than relying on frozen pre-trained embeddings, the model adapts its state representation directly to reward signals. This design correlates with consistent improvements in both policy metrics and downstream code quality.
- **Practical scalability analysis**: The empirical efficiency study suggests that hierarchical state aggregation mitigates the memory and compute scaling bottlenecks typically encountered by flat sequence models when processing larger codebases, addressing a relevant deployment concern.

## Weaknesses
- **RL formulation mismatch and under-specification**: Section 5.3 explicitly states the use of Proximal Policy Optimization (PPO), yet Equation (6) presents the vanilla REINFORCE gradient objective. This theoretical mismatch, combined with missing MDP details (action space definitions per task, reward computation for "semantic correctness", transition dynamics, and episode termination), renders the training pipeline irreproducible and obscures the actual optimization signal.
- **Naive state aggregation undermines hierarchical claims**: Equation (5) reduces the multi-level attention mechanism to a linear concatenation of level-specific embeddings `[h_CLS ∥ f_main ∥ m_root ∥ g_CDG]`. This readout discards cross-level relational dependencies and fails to leverage the proposed attention weights to dynamically synthesize information, weakening the architectural contribution.
- **Insufficient empirical validation and statistical rigor**: The ablation study is restricted solely to the program repair task, failing to demonstrate that hierarchical components generalize across all three claimed environments. Furthermore, all main results report single scalars without standard deviations, contradicting the stated use of paired t-tests and masking the known instability of PPO in discrete code-generation spaces. Section 7.1 also contains explicit placeholder text ("9 Need to discuss several limitations of this study").
- **Outdated baselines and ambiguous dataset usage**: The comparisons rely heavily on pre-2021 architectures (CodeBERT, Tree-LSTM, flat Transformers) without controlling for compute budget or parameter-efficient fine-tuning, making it unclear if gains stem from architecture or training parity. Additionally, Section 5.1 claims use of the APPS benchmark but cites a 2024 paper on "Webapp1k," creating fundamental ambiguity about the experimental setup.
- **Unsubstantiated complexity claims and presentation issues**: The assertion of linear memory scaling versus quadratic growth for baselines lacks empirical FLOPs/VRAM profiling or explicit attention sparsity guarantees, which is critical since standard GATs and Transformers do not naturally scale linearly without modifications. These technical gaps are compounded by persistent grammatical errors, unverified terms ("complexity raising functions", "hierarchical cherry-picking"), and a lack of quantitative reporting for representation analysis.

## Nice-to-Haves
- Comparative evaluation against supervised fine-tuning (SFT) baselines of matched scale to isolate the specific contribution of RL optimization versus representation capacity.
- Quantitative representation quality metrics (e.g., cluster separation scores, k-NN semantic retrieval accuracy) to supplement qualitative t-SNE plots.
- Attention heatmaps overlaid on concrete AST/CDG structures for failure cases to illustrate whether hierarchical weights capture logical error propagation paths.
- Explicit documentation of graph construction pipelines (parsers, edge-type definitions) and full PPO hyperparameter tables (clip range, rollout length, reward scaling).

## Novel Insights
None beyond the paper's own contributions. The review primarily identifies a gap between the intuitive appeal of structure-aware RL representations and the methodological rigor required to prove their necessity over simpler, scale-driven approaches. The architecture serves as a concrete engineering integration, but without mechanistic evidence of how hierarchical biases alter credit assignment or stabilize policy gradients, the contribution remains empirically descriptive rather than fundamentally insightful.

## Suggestions
1. Correct the optimization mismatch by aligning Equation (6) with the true PPO surrogate objective or updating the text to reflect REINFORCE. Provide a complete MDP specification including action space formulation per task, exact reward computation, and environment dynamics.
2. Replace the concatenation readout in Equation (5) with a cross-level gating or attention fusion mechanism that actively utilizes hierarchical weights, and substantiate scalability claims with explicit memory/runtime profiling or demonstrated sparsity constraints.
3. Expand Table 2 to cover all three tasks and report means with standard deviations over 3–5 random seeds for all main results. Include variance shading on learning curves and resolve the APPS/Webapp1k citation discrepancy.
4. Establish strict training parity by detailing compute budgets, baseline adaptation strategies (e.g., LoRA vs. full fine-tuning), and incorporating at least one modern code representation baseline to contextualize the architectural gains. Remove all placeholder text and conduct thorough professional editing to clarify notation and terminology.

---

## 4Grhy3DAZi

- GT: Withdrawn (treated as Reject) (avg 2.0)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Introspective Adversarial Learning (IAL), an iterative alignment framework that autonomously generates preference data via a Player-Advisor self-refinement loop, thereby reducing reliance on human-annotated preference pairs. To address the documented Bradley-Terry failure mode where iterative preference optimization suppresses the absolute log-likelihood of target responses, the authors propose SPACP, which incorporates a hinge-like penalty term into the objective. Experiments on 7B and 3B models show IAL improves scores on the Open LLM Leaderboard and MT-Bench over DPO, SPIN, and SPA baselines, with full code and hyperparameters provided for reproduction.

## Strengths
- **Empirically validated stabilization of iterative preference optimization:** The paper correctly identifies a critical failure mode in standard BT/DPO training: maximizing relative preference margins inherently degrades the absolute log-likelihood of target responses after initial convergence. Figure 6 and Appendix B.3 clearly demonstrate that the SPACP penalty term stabilizes the "Real Reward" trajectory, preventing optimization from collapsing target probabilities while maintaining margin growth.
- **Transparent and reproducible self-play pipeline:** The Player-Advisor generation loop is conceptually straightforward and practically implementable. The authors provide open-source code, explicit hardware/dataset configurations, and cross-family validation (Qwen-2.5-3B in the appendix), meeting high community standards for experimental transparency.

## Weaknesses
- **Marginal gains presented without statistical validation:** The reported improvements over strong baselines are small (e.g., +0.42% average on Open LLM Leaderboard vs. SPIN-iter2; +0.15% on MT-Bench vs. PPO) yet reported as single-run results without standard deviations or multiple seeds. Given the inherent variance in benchmark scoring and iterative self-training dynamics, these deltas cannot support claims of "consistent superiority" without variance metrics or significance testing. The current presentation overstates the reliability of the findings.
- **Direct contradiction between efficiency claims and reported overhead:** The abstract and introduction explicitly frame IAL as a "scalable and efficient pathway," yet Table B.5 documents a per-iteration training cost of ~19 hours compared to ~4.6 hours for standard DPO (~4x slower due to the generation phase). This substantial compute multiplier fundamentally conflicts with the efficiency narrative and is not adequately justified or contextualized in the main text.
- **Incremental technical differentiation and baseline gap:** The SPACP penalty operates as a hinge constraint on target log-likelihood, which is conceptually proximate to recently proposed constrained DPO variants (e.g., DPO-Positive). While the paper cites these works, it lacks a head-to-head empirical comparison or clear theoretical differentiation to establish a distinct advantage. Furthermore, comparing primarily against SFT and older self-play methods, rather than the officially DPO-aligned Zephyr checkpoint on MT-Bench (~7.1), leaves the true practical alignment delta ambiguous and makes it difficult to assess IAL's standing against current SOTA.
- **Unverified alignment gains vs. fixed reward model overfitting:** The entire feedback loop relies exclusively on a single external judge (PairRM) to rank self-generated responses. Without tracking the distribution of generated pair margins, monitoring KL divergence expansion, or correlating improvements with independent human/multi-judge evaluations, it remains unclear whether observed gains reflect robust alignment progress or systematic optimization to PairRM's specific scoring biases across iterations.

## Nice-to-Haves
- Qualitative case studies showing `(prompt, initial response, advisor suggestion, refined response)` to verify whether the Advisor generates actionable, specific critiques or merely adds verbose filler that exploits the reward model.
- Plotting the distribution of pairwise score margins `(r(y') - r(y))` across iterations to confirm the self-play loop generates increasingly informative preference contrasts rather than collapsing to trivial pairs.
- Standardizing and clarifying policy notation in Section 3.3 (explicitly mapping the update schedule for $\pi_\theta$, $\pi_t$, and $\pi_{\text{ref}}$) to streamline the derivation of Equation 10 for readers replicating the bilevel optimization.

## Novel Insights
The paper's most valuable contribution lies in empirically mapping the degeneration dynamics of iterative Bradley-Terry optimization: as models chase relative preference margins, they inevitably sacrifice the absolute log-likelihood of ground-truth or target responses, leading to a characteristic "oscillatory downward" trajectory in Real Reward. By demonstrating that a simple non-negative hinge penalty acts as an effective relative "no-regression" constraint, the work provides a practical diagnostic and stabilization mechanism that highlights the necessity of preserving target probability mass in self-play alignment loops.

## Suggestions
- **Run multi-seed evaluations:** Execute ≥3 independent training runs for IAL and key baselines, reporting mean ± standard deviation. This is necessary to statistically validate whether the observed <1% deltas are reproducible or fall within standard benchmark variance.
- **Correct the efficiency narrative:** Explicitly acknowledge the ~4x computational overhead per iteration in the main text, and provide a cost-benefit justification (e.g., analyzing whether the autonomous data generation yields higher sample efficiency that offsets the additional compute).
- **Strengthen empirical validation:** Include a direct comparison against the officially DPO-tuned Zephyr checkpoint on MT-Bench, and analyze how PairRM score distributions and response diversity evolve across iterations to rule out severe reward model overfitting or sycophancy amplification.

---

## jrY5Sh1pIh

- GT: Withdrawn (treated as Reject) (avg 2.0)
- Predicted: N/A (3.4/10)
- Match: N/A

### Final Review

## Summary
The paper proposes SCD (Soft top-k Contrastive Decoding), a data-centric detoxification pipeline that rewrites toxic corpus text by adaptively subtracting the top-k logit disparities between a base model and a fine-tuned "toxic" model, followed by semantic-toxicity fusion ranking. Evaluated across GPT2-XL, LLaMA2-7B, OPT-6.7B, and Falcon-7B, the method achieves consistent, state-of-the-art reductions in toxicity probability and expected maximum toxicity while demonstrating out-of-distribution generalization. The core contribution reframes contrastive decoding as a corpus-rewriting tool rather than an inference-time constraint, offering an architecture-agnostic pathway to source-level data hygiene.

## Strengths
- **Significant & Consistent Toxicity Reduction:** SCD reliably achieves the lowest TP and EMT scores across four distinct LLM families, often halving toxicity metrics compared to strong baselines (LM-Steer, DEXPERTS, UNIDETOX) while maintaining robust OOD generalization on unseen toxicity categories.
- **Architecture-Agnostic Data Intervention:** By decoupling detoxification from inference-time hooks, RLHF, or model-specific parameter edits, SCD produces a rewritable corpus that theoretically integrates into standard pretraining/fine-tuning workflows without requiring architectural modifications or safety-specialized heads.
- **Empirical Robustness to Divergence Choices:** Detoxification performance remains stable across five distinct distributional discrepancy measures (Wasserstein, KL variants, JS, TVD, vanilla CD), minimizing hyperparameter sensitivity and simplifying practical deployment.

## Weaknesses
- **Mathematical Formulation & Reproducibility Gaps:** The methodology mixes probability and logit spaces without theoretical justification. Equation 1 computes the scaling factor $\alpha$ from probability distributions (`p_base`, `p_toxic`), while Equation 5 applies $\alpha$ directly to logits. Furthermore, Equation 5 applies a magnitude/absolute-value operation to toxic model logits (`|s(x_t; θ_toxic)|`), which discards the directional sign information fundamental to contrastive decoding's negative feedback mechanism. The "weighted sum" for the Semantic-Toxicity Fusion Ranking also lacks specified coefficients or normalization steps. Coupled with a clear numerical inconsistency in the abstract (reporting GPT2-XL baseline TP as 0.42 instead of 0.54), these ambiguities hinder reproducibility and raise concerns about theoretical soundness.
- **Downplayed Fluency & Diversity Degradation:** The text characterizes quality loss as "slight" and the corpus replacement as "near-lossless," yet empirical metrics reveal substantial linguistic degradation. GPT2-XL's PPL increases by ~22%, and Falcon-7B's PPL more than doubles (10.72 to 24.96), alongside sharp declines in n-gram diversity (e.g., Dist-1 dropping ~40%). This trade-off is intrinsic to aggressive logit subtraction and must be transparently quantified, as it directly challenges the practical viability of the detoxified data for high-coherence training pipelines.
- **Scale Misalignment with "Corpus-Level" Claims:** The paper advocates for source-level corpus detoxification applicable to broad pretraining, yet experiments and downstream fine-tuning are conducted on a randomly sampled subset of only 640 texts. This instruction-tuning scale is insufficient to validate the method's efficacy on true web-scale corpora, nor does it isolate the impact of SCD rewriting from the confounding effects of small-batch supervised fine-tuning.

## Nice-to-Haves
- Provide qualitative examples or lightweight human evaluation to contextualize the PPL/diversity metrics (e.g., distinguishing between minor syntactic awkwardness vs. severe semantic drift or repetition collapse).
- Report wall-clock time or token-throughput overhead for generating the detoxified corpus to contextualize the "drop-in" practicality against cheaper prompt-only or classifier-filtering baselines.
- Add a formal limitations section explicitly addressing dependence on Detoxify's classifier biases, computational cost of step-wise divergence computation during autogeneration, and risks of semantic over-correction.
- Include standard deviations in result tables to match the text's claim of reporting them, and provide a sensitivity analysis for the fixed top-$k$ parameter ($k=10$).

## Novel Insights
The work effectively operationalizes contrastive decoding as a proactive data-cleaning instrument rather than a reactive inference guardrail. By dynamically isolating and penalizing only the most distributionally divergent toxic tokens—instead of hard-masking or relying on global probability adjustments—SCD demonstrates that targeted logit suppression, when coupled with multi-candidate semantic reranking, can systematically purify training corpora across architectures. This establishes a practical paradigm where toxicity is treated as a correctable distributional artifact during data curation, shifting the safety burden from model alignment procedures to foundational data hygiene.

## Suggestions
- Correct the abstract's baseline toxicity value to match Table 1 (0.54, not 0.42) and ensure all textual claims align precisely with tabulated results.
- Reformulate Equations 1–5 to consistently operate in logit space. Explicitly justify or modify the use of absolute value/magnitude on toxic logits, as standard contrastive decoding relies on signed differences to preserve semantic likelihood while suppressing anti-expert modes.
- Specify the exact weighting coefficients, normalization procedure, and selection thresholds for the Semantic-Toxicity Fusion Ranking to guarantee reproducibility.
- Temper claims of "near-lossless" quality and "corpus-level" applicability to accurately reflect the measured PPL/diversity trade-offs and the 640-sample experimental scale, and explicitly position the method as a targeted rewriting strategy validated on focused datasets rather than web-scale pretraining corpora.

---

## GiItKTlJIB

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a systematic chain-of-thought (CoT) deletion framework to probe reasoning faithfulness in LLMs applied to physics problem-solving. By intercepting mid-generation scratchpads and removing varying percentages of tokens, the authors demonstrate that contemporary reasoning models tolerate 40–60% CoT deletion before accuracy collapses, compensating through a "cramming" behavior that increases final answer length. The work argues that accuracy-centric benchmarks are insufficient for scientific domains and calls for evaluation protocols that measure reasoning fidelity rather than end-task correctness.

## Strengths
- **Well-targeted methodology for AI-for-Science evaluation:** Physics provides a stringent, structured testbed where reasoning failure modes are highly consequential. The systematic sweep across three distinct deletion strategies (end, random, domain-tagged), three open reasoning models, and three benchmarks of increasing difficulty yields clear, consistent phenomenological trends.
- **Clear empirical documentation of compensatory "cramming":** The consistent inverse relationship between CoT retention and final answer length, paired with rising lexical overlap metrics, provides concrete evidence that models opportunistically regenerate missing traces rather than strictly depending on them. This directly challenges prevailing assumptions about CoT necessity and interpretability in scientific reasoning tasks.
- **Solid experimental transparency:** The inclusion of convergence analysis to determine sampling sufficiency, explicit reporting of hyperparameters (temperature, top-p), and full prompt templates in the appendix demonstrates good experimental hygiene and facilitates baseline replicability.

## Weaknesses
- **Lexical overlap metrics are misaligned with structured scientific reasoning:** The use of bag-of-words Jaccard similarity and Manhattan distance to quantify information overlap (§2.4, §4.2) captures only terminological persistence, not logical or structural fidelity. In physics, lexical overlap does not distinguish between correct equation derivation, dimensional consistency checks, or plausible but incorrect substitutions. Interpreting rising overlap as evidence of "systematic attempts to reconstruct truncated reasoning" overstates what these metrics measure, weakening the paper's core faithfulness claims.
- **Evaluation confounds verbosity with correctness and lacks statistical robustness:** The primary performance metric relies on a Claude-family judge scoring 0–1 based on "correctness, derivation accuracy, logic, formatting, and clarity" (§2.4). This inherently biases scores toward longer, more verbose outputs, making it difficult to disentangle genuine solution recovery from defensive hedging or filler generation. Compounded by a low calibration baseline (N=5) and the absence of confidence intervals on the deletion sweep curves, the precise robustness thresholds (40–60%) and claims of "sharp drops" lack statistical substantiation.
- **Underspecified deletion protocol and evaluator circularity:** The paper states it "intercepts CoT mid-generation" but omits the inference mechanics (e.g., KV cache modification, autoregressive pausing, token budget enforcement), limiting reproducibility. Furthermore, the physics-aware deletion strategy relies on the same model family used for primary evaluation to identify and remove domain-specific tokens. Without deterministic ablation or decoupling the tagger from the judge, observed trends could reflect the tagger's selection preferences rather than genuine model resilience.

## Nice-to-Haves
- Including a direct (no-CoT) answer baseline would help contextualize whether the 40–60% deletion threshold represents true scratchpad dispensability or simply reversion to the model's inherent parametric knowledge. While prompting styles are compared in §3.1, explicitly mapping that baseline against the deletion sweeps would strengthen the interpretation of "robustness."
- Providing side-by-side qualitative examples contrasting successful logical reconstruction against jargon-heavy or repetitive outputs under heavy deletion would clarify whether "cramming" reflects structured recovery or unproductive verbosity.

## Novel Insights
The paper convincingly demonstrates that for current reasoning LLMs, chain-of-thought traces function more as optional, high-bandwidth scaffolding than as strict computational dependencies. The systematic tolerance for heavy token deletion, paired with compensatory lengthening and surface-level lexical recovery, suggests that models maintain a redundant parametric reservoir of physics knowledge that can be opportunistically queried when prompting scaffolds fail. This reveals a critical epistemic risk for AI-for-science: models can produce long, lexically dense, and superficially coherent derivations after losing critical intermediate steps, creating an illusion of faithful reasoning that accuracy-only metrics fail to detect.

## Suggestions
- Detail the exact inference pipeline for mid-generation token deletion, specifying how the KV cache or context window is modified, whether sampling is deterministicized pre- and post-intervention, and how tokens are budgeted across the truncated prefix and final answer phase.
- Supplement or replace the bag-of-words overlap analysis with equation-aware or structural matching (e.g., AST parsing, symbolic canonicalization, or constrained concept overlap) to properly assess logical recovery, or explicitly reframe the overlap results as measuring terminological redundancy rather than faithful reconstruction.
- Decouple the physics-aware tagger from the evaluation model, report confidence intervals across deletion sweeps, and consider a length-controlled scoring ablation to isolate whether "cramming" genuinely improves solution fidelity or merely inflates LLM-as-a-judge scores through verbosity.

---

## FMjeC9Msws

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (7.4/10)
- Match: N/A

### Final Review

## Summary
This paper presents the first large-scale systematic study (>400,000 GPU-hours) establishing a predictive framework for reinforcement learning compute scaling in LLMs. By fitting sigmoidal compute-performance curves, the authors isolate how pipeline architecture, loss formulation, numerical precision, and data handling affect asymptotic performance and compute efficiency, consolidating these findings into the SCALERL recipe. The work demonstrates that stable RL training follows predictable trajectories, enabling accurate extrapolation from mid-scale runs to 100,000 GPU-hour trajectories across both dense and MoE architectures.

## Strengths
- **Unprecedented empirical scale & staged experimental design:** The >400k GPU-hour campaign is rigorously structured into small-scale ablations, 16k GPU-hour leave-one-out (LOO) validations, and full-scale 100k GPU-hour runs. This multi-stage isolation of components (e.g., PipelineRL, CISPO, FP32 logits, zero-variance filtering) provides a highly controlled foundation for disentangling algorithmic and system-level scaling effects.
- **Actionable, empirically validated predictive framework:** The parameterization of RL scaling into asymptotic ceiling ($A$), efficiency exponent ($B$), and midpoint ($C_{\text{mid}}$) is intuitive and practically useful. The paper convincingly validates predictability: fitted curves on early/mid training data closely match extended runs (e.g., 8k→16k GPU-hrs LOO, 50k→100k GPU-hrs dense/MoE runs), offering a concrete tool for compute planning and early termination of unstable recipes.
- **Transparent recipe construction with rigorous LOO verification:** SCALERL is not presented as a novel algorithm but as an engineered synthesis of stabilized components. The LOO experiments (§4, Fig 5) transparently show how reverting each component impacts stability or efficiency, and the public release of the fitting code ensures methodological reproducibility.

## Weaknesses
- **GPU-hours as the primary scaling metric conflates algorithmic scaling with system throughput:** The paper uses GPU-hours as the $x$-axis for all scaling laws, but Appendix A.20 explicitly acknowledges that PipelineRL's efficiency gains stem largely from reduced idle time and higher hardware utilization, not intrinsic sample efficiency. Without normalizing compute to processed tokens or estimated FLOPs, the fitted curves absorb infrastructure overheads, limiting the framework's claim to a general *algorithmic* scaling law and making cross-lab comparisons dependent on specific hardware/scheduling setups.
- **The fitted asymptote $A$ likely reflects dataset memorization limits rather than fundamental capability ceilings:** RL training runs multiple epochs over a fixed ~53k math prompt distribution. As compute scales, the sigmoidal saturation likely captures a memorization/fitting ceiling on this specific distribution. While the authors report qualitative downstream correlation (AIME-24, MATH-500), they do not quantify the train/validation gap or the rate of OOD generalization degradation as $A$ saturates. This leaves open the risk that optimizing for high $A$ may inadvertently encourage distribution-specific memorization rather than robust reasoning generalization.
- **Asymptotic framing slightly overstates gains relative to LOO results:** The introduction and abstract emphasize shifting asymptotic ceilings, but the LOO results (§4, §7, App A.10) consistently show that reverted configurations reach $A$ within the stated $\pm 0.02$ error margin of SCALERL. The primary empirical advantage is compute efficiency ($B, C_{\text{mid}}$) and training stability, not a substantially higher final ceiling. This is not a flaw in the data, but the narrative framing diverges slightly from the ablation results, which show that many existing recipes are near-optimal in final performance if stabilized and given enough compute.

## Nice-to-Haves
- **Confidence intervals for fitted parameters:** While Appendix A.5 reports an error margin of $\pm 0.015$ on $A$ across 3 full runs, individual ablation comparisons in the main text lack bootstrapped confidence intervals or standard errors for $A$ and $B$. Providing these would strengthen statistical claims when comparing marginal differences between loss types or aggregation schemes.
- **Quantitative ID-to-OOD generalization analysis:** Adding a correlation metric or scaling curve comparing in-distribution validation saturation against downstream benchmark performance (e.g., AIME vs. LiveCodeBench) would clarify how reliably high $A$ translates to OOD reasoning gains.
- **Small-budget application protocol:** A concrete guideline specifying minimum compute windows, number of evaluation points, and fitting reliability thresholds for labs with <5k GPU-hours would significantly broaden the framework's accessibility without compromising scientific rigor.

## Novel Insights
The paper effectively reframes RL scaling from a search for novel objectives to an engineering problem of *stabilization and predictable throughput management*. The most compelling insight is that predictable scaling trajectories do not emerge from algorithmic breakthroughs, but from systematically removing sources of early-training instability (numerical mismatch at the LM head, off-policy distribution drift via PipelineRL, truncation-induced gradient variance). Once stability is enforced, RL compute scales with remarkable predictability, and the primary differentiator between recipes becomes how efficiently they traverse the curve ($B, C_{\text{mid}}$), not where the curve ends. This shifts the community's focus from chasing marginal objective tweaks toward robust infrastructure, numerical precision, and early-extrapolation diagnostics.

## Suggestions
1. **Decouple algorithmic vs. system compute:** Re-plot key scaling curves (e.g., Fig 2, Fig 5) with compute measured in processed tokens or estimated FLOPs alongside GPU-hours. Explicitly quantify how much of SCALERL's advantage persists when normalized for throughput, clarifying the boundary between algorithmic scaling and engineering optimization.
2. **Align claims with LOO evidence:** Temper abstract/introduction statements that emphasize "shifting asymptotes" and explicitly frame SCALERL's contribution as *reliable ceiling attainment and compute efficiency* rather than fundamentally higher asymptotic capacity. Acknowledge that with sufficient stabilization and compute, many baselines converge to similar $A$.
3. **Quantify the ID/OOD gap:** Include a table or figure reporting the delta between in-distribution validation pass rates and downstream benchmark scores across scaling axes (model size, batch size, context length). Briefly analyze whether higher $A$ consistently yields proportional OOD gains or exhibits diminishing returns, which would ground the framework's practical utility for generalization-focused labs.

---

## YkfhTzq3hL

- GT: Reject (avg 4.5)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces SHALLOW, a multidimensional evaluation framework designed to diagnose hallucinations and fine-grained transcription errors in Automatic Speech Recognition (ASR) systems. By decomposing transcription failures into lexical, phonetic, morphological, and semantic dimensions, it moves beyond the limitations of aggregate Word Error Rate (WER). Extensive evaluation across 12 diverse ASR architectures and 10 acoustic/linguistic datasets demonstrates that SHALLOW reveals architectural trade-offs and error dissociations under degraded conditions that WER systematically obscures, supported by a controlled synthetic validation dataset.

## Strengths
- **High-Stakes Motivation & Practical Utility:** The paper effectively argues that WER masks critical semantic inversions and context shifts, particularly in safety-critical domains. The zero-shot medical case study compellingly grounds this claim, demonstrating how SHALLOW flags dangerous low-WER hallucinations that traditional metrics miss, directly justifying the need for fine-grained diagnostics.
- **Architecture-Aware Empirical Coverage:** The systematic comparison of 12 models across four distinct architectural families (encoder-only, encoder-decoder, encoder-transducer, and SpeechLLMs) across 10 diverse datasets provides robust empirical grounding. The analysis successfully links hallucination profiles to explicit design choices (e.g., encoder-transducers excelling in phonetic fidelity vs. decoder-only models prioritizing lexical/semantic coherence under specific conditions).
- **Strong Synthetic Validation & Reproducibility:** The creation of a 1,050-sample synthetic dataset isolating specific hallucination types effectively validates the orthogonality and discriminative power of the four proposed metrics. Combined with explicit mathematical formulations, detailed implementation notes, and the release of both code and synthetic data, the work establishes a transparent foundation for future benchmarking.

## Weaknesses
- **Heuristic Weighting Lacks Empirical Grounding:** All composite scores rely on fixed, hand-tuned weights (e.g., `0.5/0.3/0.2` for lexical insertions/substitutions/deletions; `0.4/0.6` for structural vs. grammatical distortion; `1:3` for local vs. global semantic errors). While briefly attributed to "empirical observations," the paper provides no ablation, sensitivity analysis, or validation against human severity ratings. For a benchmark intended as a diagnostic standard, this makes the composite scores appear arbitrary and risks skewing model rankings based on unstated assumptions about error severity.
- **Toolchain Confounders in Morphological & Semantic Evaluation:** The Morphological Error metric depends on LanguageTool and dependency parsers, while the Semantic Error metric relies on NLI and embedding models. ASR hypotheses are frequently lowercase, unpunctuated, and disfluent, and this output convention varies significantly across model families. Although a footnote notes that punctuation captures structural inconsistencies, applying grammar/punctuation checks to raw, unpunctuated outputs across different architectures will artificially conflate formatting deficits with actual morphological distortion. Without consistent preprocessing or explicit error bounds for the NLP toolchain, cross-architecture comparisons may reflect pipeline artifacts rather than intrinsic hallucination behavior.
- **Discrete Step-Function in Semantic Scoring Limits Resolution:** The global semantic coherence metric uses a discrete step function based on NLI classification (1.0 for entailment, 0.5 for neutral, 0.0 for contradiction). This design creates sharp score plateaus that reduce the metric's resolution for nuanced meaning shifts or partial degradations. In real-world clinical or conversational transcripts, meaning preservation often exists on a continuum; the thresholded NLI penalty may over-penalize or under-penalize subtle semantic drifts, limiting fine-grained diagnostic granularity in the exact regimes where the framework aims to be most valuable.

## Nice-to-Haves
- A human correlation study or severity-rating benchmark to empirically validate that higher SHALLOW scores align with human judgments of transcription harmfulness or comprehension degradation, particularly for the semantic dimension.
- Statistical significance testing (e.g., bootstrap confidence intervals) for the reported Spearman correlation trends and cross-model ranking differences to rule out sampling variance, especially in higher-WER regimes where sample stratification shrinks.
- Explicit discussion or preliminary guidance on adapting the framework cross-lingually, given its current reliance on English-specific parsing, grammar checking, and NLI models, which limits immediate deployment in multilingual research pipelines.

## Novel Insights
The paper successfully shifts the ASR evaluation paradigm from monolithic accuracy chasing to structured diagnostic granularity. By demonstrating that architectural design choices (acoustic vs. linguistic priors) manifest as predictable, decoupled hallucination patterns across different noise regimes and speaker demographics, it reveals that ASR "hallucination" is not a single failure mode but a multidimensional trade-off space. This reframing provides an actionable lens for developers: rather than optimizing solely for WER, they can isolate and target specific error dimensions (e.g., phonetic vs. semantic) based on downstream application constraints, fundamentally improving how safety-critical speech systems are audited, selected, and iteratively refined.

## Suggestions
- Conduct and report a weight sensitivity analysis: systematically perturb the fixed coefficients in Equations 1, 3, 4, and 7 (e.g., by ±20%) and demonstrate that the relative diagnostic rankings of models and the core architectural insights remain stable. Alternatively, formally document these weights as configurable application-specific parameters with clear guidance on how practitioners should adjust them for domain-specific needs.
- Clarify the preprocessing pipeline for NLP-dependent metrics: explicitly state whether reference and hypothesis texts undergo punctuation restoration or case normalization before being passed to LanguageTool, spaCy, and the NLI models. If raw outputs are used, add a brief analysis quantifying how missing punctuation/capitalization impacts Morphological and Semantic scores, or restrict cross-architecture comparisons to punctuation-consistent models to ensure methodological fairness.

---

## Q4KIDjDDRJ

- GT: Reject (avg 3.0)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper addresses the challenge of balancing visual grounding and symbolic reasoning when applying Reinforcement Learning with Verifiable Rewards (RLVR) to multimodal large language models. Through controlled token-ablation experiments, the authors demonstrate that optimizing only perception- or reasoning-related tokens degrades performance compared to full-token optimization, revealing a functional coupling between the two capabilities. To address this, they propose Token Reweighting (ToR), a lightweight, plug-and-play modification to policy gradient objectives (GRPO/DAPO) that identifies critical reasoning tokens via high entropy and perception tokens via visual sensitivity (log-probability difference with/without the image), then adaptively reweights their gradient contributions. Evaluated on Qwen2.5-VL-7B with 2.1K geometry samples, ToR yields consistent 1–3% accuracy gains across five multimodal reasoning and perception benchmarks.

## Strengths
- **Clear empirical motivation for joint optimization:** The controlled experiments (Figure 2) effectively demonstrate the failure modes of single-capability optimization: reasoning-only training produces coherent chains with visual hallucinations, while perception-only training preserves grounding but breaks logical inference. This establishes a concrete need for methods that explicitly balance both signal types during RLVR.
- **Algorithmic simplicity and broad compatibility:** ToR requires no architectural changes, auxiliary models, or complex reward shaping. By operating directly as a token-level mask within standard policy gradient equations, it integrates seamlessly with existing RLVR frameworks and maintains the efficiency of verifiable reward pipelines.
- **Data-efficient and consistent benchmark gains:** Across five diverse benchmarks spanning mathematical reasoning and visual perception, ToR-GRPO and ToR-DAPO consistently outperform their vanilla counterparts. The improvements are particularly notable given the highly constrained 2.1K training budget, positioning the method as a strong data-efficient augmentation for multimodal alignment.
- **Systematic exploration of selection and weighting dynamics:** The ablation studies on token selection ratios ($\alpha_r, \alpha_p$) and combination weights ($\gamma_p$) provide practical, data-backed guidance for configuring the method, moving beyond ad-hoc hyperparameter choices.

## Weaknesses
- **Limited generalization scope undermines broad claims:** All experiments are conducted exclusively on the Geometry3K dataset (2.1K samples) using a single model scale (Qwen2.5-VL-7B). While the results are promising, claiming broad improvements to "multimodal reasoning" without validating on semantically distinct datasets (e.g., chart interpretation, scientific diagrams, or general VQA) or alternative model scales leaves open the possibility of dataset-specific overfitting or architecture-dependent sensitivity.
- **Absence of statistical rigor and RL training dynamics:** RLVR is inherently variance-sensitive due to stochastic rollouts and sparse binary rewards. Table 1 reports single-run point estimates without standard deviations, multiple seeds, or confidence intervals. Furthermore, the paper lacks learning curves tracking reward trajectories, KL divergence, or policy entropy, which are essential to verify that masking ~40% of token gradients does not induce optimization instability, reward hacking, or catastrophic forgetting on non-selected tokens.
- **Coupling claim is not fully disentangled from gradient sparsification effects:** The core premise that perception and reasoning are *inherently coupled* relies on the observation that masking one token type underperforms full optimization. However, masking arbitrary or non-punctuation token subsets in RL typically degrades performance simply due to reduced gradient signal and loss of contextual fluency. Without a control baseline (e.g., random token masking or uniform weighting of non-punctuation tokens at equivalent retention ratios), it remains unclear whether gains stem from correctly modeling perception-reasoning interplay or from standard RL regularization/signal denoising.
- **Unquantified computational overhead:** Computing perception tokens requires an additional forward pass per rollout with the image channel replaced by a placeholder (`∅`) to calculate visual sensitivity (Eq. 8). In RLVR, where generation is already the dominant cost, this effectively doubles pre-update compute. The paper describes ToR as "lightweight" but provides no wall-clock timing, memory footprint, or throughput comparisons, making it difficult for practitioners to assess the cost-benefit tradeoff at scale.

## Nice-to-Haves
- **Qualitative grounding analysis:** The paper claims improved "accurate visual grounding," but evaluation relies solely on end-task accuracy. Providing attention heatmaps or bounding-box alignment comparisons for failure cases converted to successes would strengthen the multimodal grounding narrative.
- **Overlap token handling ablation:** Roughly 12% of tokens are classified as both perception- and reasoning-related, and the paper arbitrarily assigns them the reasoning weight. A brief ablation comparing perception-only, additive ($\gamma_r + \gamma_p$), or max-weighting strategies would solidify the design choice.
- **Dynamic weight scheduling:** The appendix notes a "push-pull" dynamic between reasoning uncertainty and perception strength during training. Implementing a simple curriculum to adjust $\gamma_p$ or $\alpha$ ratios as the model improves could better reflect evolving optimization bottlenecks.

## Novel Insights
The paper surfaces a practically important but underexplored tension in multimodal RL: perception and reasoning gradients exhibit a competitive, push-pull relationship where optimizing one in isolation actively degrades the other. Rather than treating tokens as uniform optimization targets or routing them to separate reward branches, ToR reframes RLVR token selection as a joint signal-routing problem. By grounding token importance in intrinsic cognitive bottlenecks—decision uncertainty (entropy) for reasoning and conditional dependence (visual sensitivity) for perception—the method demonstrates that strategically filtering and amplifying gradient updates at these intersection points yields smoother policy updates and more consistent accuracy gains than uniform full-sequence optimization, even under severe data constraints.

## Suggestions
- **Add a random/uniform token masking baseline:** Train with equivalent retention ratios but randomly selected tokens or uniform weighting of non-punctuation tokens. This will isolate whether the entropy + logp-diff criteria are necessary, or if the gains primarily arise from gradient sparsification/denoising.
- **Report statistical significance and training dynamics:** Re-run core comparisons with $\geq$3 random seeds and report mean $\pm$ std. Include plots of validation accuracy, KL divergence, and rollout variance across training steps to demonstrate policy stability under token masking.
- **Quantify and report compute overhead:** Provide a table comparing wall-clock training time, GPU memory usage, and tokens-per-second throughput for vanilla GRPO vs. ToR-GRPO. Clarify whether the image-masking forward pass can be batched, cached, or approximated to reduce inference-time costs.
- **Adjust contribution framing and expand validation:** Frame the 2.1K results explicitly as *data-efficient* RLVR improvements rather than absolute "SOTA" claims, given the 50–100× data disparity with competing methods. If feasible, validate on at least one additional dataset or model scale to substantiate generalization beyond geometric reasoning.

---

## VoKut0M4bI

- GT: Reject (avg 5.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a post-hoc auditing framework for vision-based world models, decomposing the RL suboptimality gap into three structural channels: geometric distortion ($\kappa$), identifiability (via Total Correlation), and symmetry violation (via Local Equivariance Error). Framed as a sufficient safety condition rather than a tight predictor, the bound is instantiated via a lightweight diagnostic protocol that calibrates a single conservative constant $\beta$ on early training checkpoints to verify coverage across the training trajectory. Experiments on DreamerV3 demonstrate reliable coverage, supported by ablations and permutation tests that validate the structural necessity of each proxy.

## Strengths
- **Principled reframing from correlation to coverage:** The explicit distinction between a sufficient bound (guaranteeing safety via coverage) and a predictive metric (tracking correlation) directly addresses a common pitfall in representation evaluation. This theoretical framing correctly justifies weak proxy-performance correlations while establishing a conservative diagnostic lens.
- **Statistically rigorous empirical validation:** Beyond simple trend plots, the paper validates its claims with leave-one-term-out ablations (showing coverage drops to 60–80% when any channel is removed), permutation testing ($p < 0.001$ against random pairing), and a TC control experiment that reveals a mechanistic non-monotonic trade-off between over-compression and control performance.
- **Transparent and practitioner-aligned design:** The protocol operates at low overhead (~2 GPU hours per checkpoint) without retraining, explicitly acknowledges proxy conservativeness, and honestly delineates the scope of its claims. This operational clarity and candid limitation reporting significantly enhance reproducibility and trust.

## Weaknesses
- **Theoretical bridges are posited as assumptions rather than derived, and clash with the experimental architecture:** The proof's critical transition error step (Eq. 5: $W_1(P_\phi, \tilde{P}) \le C_\kappa \kappa + C_F L_F$) is introduced without derivation under explicit regularity conditions. More importantly, the bridge lemmas linking proxies to theoretical channels assume smooth manifolds, uniformly Lipschitz vector fields, and Nonlinear ICA settings. These highly idealized structural assumptions are practically incompatible with DreamerV3's discrete, multimodal categorical RSSM latents, undermining the theoretical grounding of the deployed diagnostic.
- **Gaussian TC estimator is theoretically mismatched with the learned representation:** Section 3.2 employs a Gaussian covariance-based TC estimator, yet DreamerV3's RSSM produces categorical/multimodal latent distributions. Without a discrete-compatible estimator or empirical validation of the Gaussian surrogate's fidelity (e.g., normality diagnostics or ablation against flow/k-NN MI estimates), the identifiability channel's proxy values lack empirical validity and may misrepresent true latent dependence.
- **Limited demonstration of operational utility:** While coverage is empirically verified, the paper does not demonstrate how practitioners should act on the diagnostic. There is no evaluation of downstream decision-making utility, such as automated checkpoint selection, early stopping, or hyperparameter debugging, leaving the practical advantage of the bound over standard one-step MSE or validation returns unproven.

## Nice-to-Haves
- Evaluate the framework under out-of-distribution state distributions or across fundamentally different world model architectures (e.g., MWM, IRIS, continuous control) to test $\beta$ stability and proxy robustness beyond the training environment.
- Provide concrete failure-mode case studies mapping proxy spikes to physical representation pathologies (e.g., visualizing latent collapse alongside high $\kappa$) to bridge the gap between abstract metrics and intuitive debugging.
- Clarify the exact rollout budget required for initial $\beta$ calibration and discuss whether the constant exhibits task-transferable priors, which would reduce the auditing setup cost.

## Novel Insights
The paper's core conceptual advance lies in shifting representation auditing from correlation-seeking heuristics to coverage-guaranteed sufficient conditions. This reframing is bolstered by mechanistic insights: the quotient-space Johnson–Lindenstrauss argument explains how equivariance structurally reduces effective dimensionality, while the geometry–equivariance trade-off formalizes why non-isometric actions inherently induce latent distortion. Together, these perspectives move the field beyond treating reconstruction error or one-step MSE as ground truth, offering a structural explanation for why and how latent metrics drift during training irrespective of immediate reward signals.

## Suggestions
- Explicitly reclassify Eq. 5 and the TC/LEE bridge lemmas in the main text as structural working hypotheses. Clarify that the theory motivates the *form* and *channels* of the diagnostic bound, while $\beta$ empirically absorbs the gap between idealized assumptions and practical deep RL dynamics.
- Replace or supplement the Gaussian TC estimator with a discrete-compatible mutual information estimator (e.g., leveraging the categorical RSSM probabilities directly) or provide a sensitivity analysis/latent normality diagnostic that justifies the Gaussian approximation. This is critical for aligning the identifiability channel with the evaluated architecture.
- Add a downstream utility experiment demonstrating how the bound guides practitioner decisions. For example, benchmark bound-based checkpoint selection or early-stopping against baselines like validation return, one-step MSE, or random selection to empirically validate the framework's operational advantage.

---

## l5zZ2EEijD

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a novel Iteratively Reweighted Least Squares (IRLS) algorithm for $\ell_p$ regression that achieves state-of-the-art theoretical iteration complexity while maintaining a lightweight, implementable structure. By deriving update rules through a primal-dual invariant on the dual energy function, the authors match the theoretical guarantees of complex prior frameworks (Adil et al., 2019a) and extend the method to high-precision regimes via iterative refinement. Empirical results on synthetic and real-world datasets demonstrate consistent reductions in both iteration count and wall-clock time compared to previous IRLS variants and standard convex solvers.

## Strengths
- **Novel primal-dual framework that bypasses standard optimization templates:** Unlike width-reduced multiplicative weights update or mirror descent approaches, the algorithm maintains a monotonicity invariant on the dual energy $E(r)$. This derivation avoids explicit $\ell_p$ smoothing/regularization and permits large polynomial step sizes, directly yielding the improved iteration complexity (Section 2.2, Eq. 1).
- **Rigorous high-precision extension with careful handling of non-scale-free objectives:** The adaptation of the iterative refinement scheme to mixed $\ell_p + \ell_2$ residual problems is well-structured. The authors correctly identify that the $\ell_2$ term breaks scale-invariance and provide a mathematically sound fix via controlled initialization and step-size damping (Algorithm 4, Lemmas E.5–E.7), preserving the theoretical guarantees.
- **Consistent empirical outperformance with clear reproducibility:** The evaluation systematically benchmarks against $p$-IRLS (Adil et al., 2019b) and CVX across varying $n$, $p$, and graph topologies, showing consistent reductions in linear system solves and runtime (Figures 1–2, Table 1). The supplementary code and transparent experimental setup facilitate direct reproducibility.

## Weaknesses
- **Limited empirical scale and regime coverage:** The experimental validation caps at $n \approx 2,500$ for dense matrices and $n \approx 10^4$ for graph instances, with real-world tests restricted to six standard UCI datasets. While this suffices to validate correctness and baseline speedups, it does not demonstrate whether the asymptotic reduction in iteration complexity compounds into substantial practical advantages on high-dimensional, highly sparse, or modern large-scale problems where linear system solves become memory- and compute-bottlenecks.

## Nice-to-Haves
- **Conditioning and numerical stability discussion:** IRLS methods at high precision ($\epsilon = 10^{-10}$) typically suffer from deteriorating conditioning of the weight matrix $D$. A brief discussion or empirical reporting of the condition number of $ADA^\top$ across iterations, along with any practical stabilization heuristics (e.g., weight clipping or diagonal damping), would strengthen the practical appeal.
- **Sensitivity to inexact linear solvers:** The theoretical iteration bounds assume exact solves of the form $ADA^\top \phi = b$. Including a remark or small ablation quantifying how solver tolerances (e.g., conjugate gradient residuals) interact with the primal-dual invariant would clarify deployment guidelines for large-scale or sparse settings.
- **Explicit cost breakdown for general-form reductions:** The reduction from $\min \|Nx-v\|_p$ to the affine-constrained form (Appendix B) augments dimensions and adds solver calls. A brief complexity footnote in the main text would help practitioners accurately estimate overhead for standard overconstrained regression tasks without navigating the appendix.

## Novel Insights
The paper's core insight lies in reframing IRLS updates not as heuristic reweightings, but as controlled multiplicative increments along a dual energy landscape. By proving that a specific coordinate-wise ratio between energy gain and dual norm growth can be maintained exactly, the authors sidestep the conventional need for smoothing mirror maps or width-reduction schemes. This invariant-driven approach naturally accommodates much larger step sizes and yields a cleaner algorithmic pipeline, effectively decoupling theoretical optimality from the heavy subroutine overhead that historically separated provable $\ell_p$ solvers from practical IRLS implementations.

## Suggestions
- Expand the empirical evaluation to include at least one high-dimensional or highly sparse benchmark (e.g., feature dimensions $d > 10^5$ with sparse design matrices) to demonstrate scalability and memory efficiency in regimes closer to modern ML workloads.
- Add a concise subsection or footnote in Section 4 discussing the numerical behavior of $ADA^\top$ solves at $p \geq 8$ and $\epsilon = 10^{-10}$, explicitly noting whether the implementation encountered ill-conditioning and how it was mitigated (if at all).

---

## WVAr2iMu3P

- GT: Withdrawn (treated as Reject) (avg 1.0)
- Predicted: N/A (3.2/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a threshold-free evaluation framework for neural attribution methods by computing the Area Under the Intersection-over-Union curve (AUC-IoU) across a spectrum of binarization thresholds. Through systematic benchmarking of seven attribution paradigms on a dermatological imaging dataset, the authors demonstrate that conventional single-threshold protocols induce severe ranking instability and performance variance. The proposed protocol yields stable method differentiation and uncovers clinically actionable, size-stratified performance dependencies that aggregate metrics obscure.

## Strengths
- **Systematic quantification of threshold bias:** The paper rigorously demonstrates that arbitrary threshold selection fundamentally destabilizes attribution comparisons, with empirical evidence showing method performance swings that drastically shift apparent rankings.
- **Robust statistical validation:** Employing paired Wilcoxon signed-rank tests with Holm-Bonferroni correction is appropriate for the paired, non-normal per-image score distributions and prevents overclaiming on marginal differences (e.g., correctly flagging indistinguishable method pairs).
- **Clinically grounded size-stratified analysis:** Breaking evaluation by lesion scale reveals that aggregate AUC-IoU masks critical performance dependencies, showing that methods like GradCAM scale poorly on small, diagnostically challenging lesions while region-based approaches (XRAI) maintain stability. This directly informs high-stakes deployment decisions.

## Weaknesses
- **Structural metric bias in IoU computation:** Section 3.4 specifies returning `IoU = 1.0` when the prediction-mask union equals zero. For the 333 non-melanoma samples where ground truth masks are entirely empty, and for high thresholds where many methods yield empty binarized maps, this default artificially awards perfect scores. This inflates AUC-IoU for sparse/gradient-based methods and biases the threshold curve, directly undermining the reliability of high-threshold performance claims.
- **Mathematical artifact in the stated threshold bias magnitude:** The relative difference metric (Eq. 3: `(AUC-IoU − IoU(τ))/IoU(τ)`) divides by single-threshold IoU values that frequently approach zero. As `IoU(τ) → 0`, the relative difference asymptotically explodes, mathematically guaranteeing the reported "+202.7% swing" rather than empirically measuring ranking instability. This inflates the perceived severity of threshold bias and makes the abstract's claim of "200 percentage points" misleading.
- **Conflation of spatial alignment with explanatory faithfulness:** The framework exclusively measures overlap between attribution maps and clinical segmentation masks. While useful for localization, IoU does not verify whether highlighted regions actually drive the model's prediction. A method could achieve high AUC-IoU by exploiting dataset co-occurrence patterns while failing to reflect the network's true decision boundaries. This limits the framework's ability to assess genuine "attribution quality" beyond spatial proxy alignment.
- **Unsupported generalization claims:** Despite confining experiments to a single dataset, a single ResNet-18 architecture, and a binary classification task, the abstract and introduction claim the framework establishes standards for "medical imaging and beyond" and reveals fundamental attribution behaviors. These sweeping claims exceed the empirical scope and cannot be validated without cross-architecture or cross-domain analysis.

## Nice-to-Haves
- **Clarify LIME's apparent threshold invariance:** Table 5 reports identical IoU scores across τ = 0.3, 0.5, 0.7 for LIME. This likely stems from discrete superpixel attribution values where all three thresholds fall between the same magnitude cutoffs, yielding identical binarizations. Explicitly noting this discretization artifact would improve technical precision.
- **Report actual rank correlation metrics:** Replace or supplement the unstable relative difference percentages with Kendall's τ or Spearman rank correlations to directly quantify how often threshold selection actually inverts method rankings.
- **Correlate AUC-IoU with faithfulness benchmarks:** Running a subset of evaluations through Insertion/Deletion curves or ROAR would help disentangle whether AUC-IoU rankings align with causal feature reliance.
- **Normalize for ground truth mask area in stratified analysis:** Larger lesions mathematically permit higher absolute IoU. Controlling for mask area or using normalized overlap would verify that the reported "0–269% improvement" factors reflect genuine method adaptation rather than area-driven ceiling effects.

## Novel Insights
Threshold response profiles act as behavioral signatures that distinguish attribution paradigms: concentrated gradient-based methods exhibit monotonic decay as thresholds increase, while superpixel/perturbation methods behave as step functions dictated by discrete weighting. Furthermore, aggregate performance metrics are dangerously reductive; the framework reveals that method superiority is highly conditional on input scale, suggesting that robust clinical XAI requires adaptive selection based on lesion characteristics rather than global benchmarks.

## Suggestions
1. **Revise the IoU edge-case handling:** Replace `IoU = 1.0` for empty unions with a standard convention (e.g., `IoU = 0.0` when masks are non-empty, or explicitly exclude empty-union cases from the integral) to prevent artificial inflation of high-threshold scores.
2. **Replace Eq. 3 with absolute deltas or rank stability metrics:** Quantify threshold bias using absolute IoU differences or actual ranking inversion counts to avoid denominator-induced arithmetic artifacts and accurately reflect empirical instability.
3. **Scope claims to match experimental bounds:** Explicitly limit generalization statements to dermatological imaging and CNN-based architectures in the abstract and introduction, or add a cross-architecture pilot (e.g., ViT) to substantiate broader claims.
4. **Add a brief discussion on localization vs. faithfulness:** Acknowledge that while AUC-IoU stabilizes spatial evaluation, it does not certify explanatory faithfulness. Recommend pairing this protocol with model-perturbation benchmarks in high-stakes validation pipelines.

---

## wkVsKDnl4s

- GT: Reject (avg 1.0)
- Predicted: N/A (2.4/10)
- Match: N/A

### Final Review

## Summary
HighClass proposes a metagenomic classification pipeline that replaces traditional seed-and-extend alignment with hash-based lookups over a learned, quality-aware variable-length token vocabulary. The authors provide theoretical guarantees via Rademacher complexity bounds, $\alpha$-mixing concentration inequalities, and MLE consistency, demonstrating that token dependencies can be mathematically controlled. Empirically, the system achieves 85.1% F1 on CAMI II within 1.5% of the state-of-the-art MetaTrinity, while delivering a 4.2× runtime speedup and 68% memory reduction through gradient-based index sparsification.

## Strengths
- **Rigorous empirical validation & component isolation:** The evaluation adheres to high computational biology standards, reporting means over 10 independent runs with 95% bootstrap CIs, Wilcoxon signed-rank tests with Holm-Bonferroni correction, and Cohen’s $d$ effect sizes. Table 3 cleanly decouples contributions, proving variable-length tokens yield +6.8 pp F1 over fixed k-mers and quality weighting adds +1.9 pp.
- **Effective bottleneck elimination:** The architectural shift to $O(|T|)$ inverted-index lookups is empirically validated. Table 5’s fine-grained cost breakdown shows that removing containment search, seeding, and chaining reduces per-read latency from 8.8ms to 1.9ms, directly explaining the wall-clock speedup and the 78% reduction in cache misses.
- **Quality-aware scoring improves robustness:** By integrating per-base quality scores into token extraction and likelihood aggregation via a learned sensitivity parameter ($\eta \approx 1.8$), HighClass demonstrates measurable resilience to sequencing errors, degrading only 2.1% F1 at 5% error rates compared to 4.3% for quality-agnostic baselines (Appendix M.2).

## Weaknesses
- **Absence of strain-level validation undermines the core claim:** The authors argue that positional alignment is unnecessary for taxonomic classification, yet they omit results on the CAMI II Strain benchmark (ANI > 95%). Strain differentiation inherently relies on subtle positional or variant-level signals that token aggregation typically obscures. Without demonstrating performance on closely related taxa, the central premise that alignment can be fully replaced remains empirically unverified for its most challenging use case.
- **Disconnect between theoretical guarantees and practical performance:** The theoretical framework adapts standard learning theory to tokenized sequences, but contains notable gaps: (1) The leap from the exponential noise generative model to the empirical power-law quality weight $\bar{q}(t)^\eta$ with $\eta \approx 1.8$ lacks explicit mathematical derivation, making the theoretical justification feel post-hoc. (2) Lemma 7 yields a variance inflation factor of $\sim$31.7× due to token dependencies. While the authors label this "manageable," such inflation materially weakens finite-sample confidence for low-abundance taxa, a critical limitation not adequately discussed.
- **Heavy reliance on external pre-trained components limits adaptability:** HighClass depends entirely on a fixed 32,000-token QA-BPE-seq vocabulary trained via external RL-PPO and pre-computed gradient-based sparsification masks. The paper provides no protocol for regenerating these masks or fine-tuning the vocabulary for novel reference databases with different taxonomic compositions or coverage profiles, obscuring the pipeline's standalone reproducibility and out-of-distribution generalization.
- **Inconsistent baseline reporting & metric reconciliation:** While the main comparison tightly pits HighClass against MetaTrinity, Kraken2, and Centrifuge, Table 7 abruptly introduces CLARK, Bracken, MEGAN, and MetaPhlAn3 with un-calibrated throughput metrics, no confidence intervals, and unspecified hardware conditions. Furthermore, the reported 1.9 ms/read micro-benchmark cannot be reconciled with the 0.5-hour wall-clock runtime without explicitly stating the test set size, reducing transparency in scalability claims.

## Nice-to-Haves
- Correlate the theoretical Rademacher bound and variance inflation factor with empirical generalization error across varying dataset sizes to demonstrate predictive utility rather than asymptotic correctness.
- Provide a biological characterization of the gradient-based sparsification masks (e.g., proportion of retained core vs. accessory genes, repetitive element density) to verify that informative discriminative loci are preserved.
- Profile candidate set size $|C|$ scaling with database complexity; the $O(|T|)$ claim assumes $|C|$ remains bounded, which may not hold for highly diverse or million-genome indices.
- Include a sensitivity curve for the quality parameter $\eta \in [0.5, 3.0]$ to demonstrate stability beyond the fixed optimal value.

## Novel Insights
HighClass convincingly reframes position-dependent alignment as a position-invariant token aggregation problem for taxonomic classification. By leveraging learned subword units that inherently encode sequencing uncertainty, the work demonstrates that precise base-to-base mapping is largely unnecessary for species-level assignment when high mutual information tokens are aggregated via compressed inverted indices. The paper establishes a compelling systems-level proof that algorithmic efficiency in genomics can be shifted from runtime computation to offline vocabulary learning and sparse index construction, effectively decoupling classification accuracy from the heavy constant factors of traditional seed-chain-extend pipelines.

## Suggestions
- Report comprehensive F1, precision, and recall on the CAMI II Strain benchmark to validate or appropriately scope the claim that alignment-free token mapping can resolve highly similar taxa without positional evidence.
- Clarify the mathematical derivation connecting the quality-dependent noise model to the learned exponent $\eta \approx 1.8$, and explicitly discuss how the ~31.7× dependency-induced variance inflation affects statistical power for low-abundance organisms.
- Provide a reproducible, documented pipeline for generating gradient-based importance masks and adapting the token vocabulary to unseen or distribution-shifted reference databases, ensuring true standalone utility.
- Reconcile micro-benchmarks with wall-clock timings by reporting the exact number of reads processed in the CAMI II partition, and harmonize the extended baseline comparisons in Table 7 to use consistent hardware, evaluation metrics, and statistical rigor.

---

## wJgaHyJaDD

- GT: Reject (avg 5.0)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces TPOUR, a training framework that adapts preference optimization to unsupervised dense retrieval by leveraging versioned document corpora. Through Temporal Retrieval Preference Optimization (TRPO), the retriever learns to prioritize temporally aligned documents over semantically similar but outdated ones. The authors further demonstrate that linear interpolation of model weights fine-tuned on different time periods enables continuous temporal generalization without retraining, yielding substantial gains on temporal QA benchmarks and revealing latent temporal sensitivity in general retrieval tasks.

## Strengths
- **Novel & Well-Motivated Method:** The adaptation of DPO-style preference learning to embedding similarity scores (TRPO) using corpus-level versioning is a creative solution to temporal misalignment. It successfully merges preference alignment with MoCo contrastive learning, avoiding the need for expensive query-document timestamp annotations.
- **Compelling Empirical Validation of Interpolation:** Figures 3 and 4, alongside Table 1, clearly demonstrate that time-vector interpolation smoothly shifts retrieval distributions across continuous time periods. The interpolated models match or exceed period-specific fine-tuning, showing that temporal representations lie in a smooth, connected region of weight space.
- **High Reproducibility & Transparency:** The paper provides exhaustive details on data filtering (Appendix B.2), hyperparameters (Table 6), compute costs, and explicit plans for code/data release. The thorough ablation on $\lambda$ and clear visualization of temporal distribution shifts meet high experimental standards.

## Weaknesses
- **Evaluation Benchmark Bias & Restricted Candidate Pool:** The custom temporal QA benchmarks are constructed by pre-retrieving only the top-10 documents per query using Contriever (Sec. 4.1). This artificially caps recall and evaluates TPOUR primarily on re-ranking a semantically-biased subset, rather than demonstrating its ability to surface temporally aligned documents from a full, noisier corpus. While Appendix E.6 checks trends with DPR, the candidate set remains pre-filtered by a baseline retriever, potentially inflating relative gains.
- **Missing Comparisons to Simpler Temporal Heuristics:** The paper omits direct comparisons to established, lighter-weight temporal retrieval baselines such as time-decay re-ranking, explicit date token injection/embeddings, or query-rewriting with timestamp extraction. Without these, it remains unclear whether the complexity of TRPO yields gains over simpler, highly interpretable temporal conditioning methods that are standard in information retrieval.
- **Theoretical Grounding & Necessity of TRPO Formulation:** Substituting DPO's log-likelihoods with dot/cosine similarity scores (Eq. 5) is heuristic; the derived objective functions more like a margin-based contrastive loss, but the paper lacks a formal derivation or convergence analysis justifying why similarity scores satisfy DPO's Bradley-Terry/reward assumptions. Given that TRPO contributes only ~7.5% to the total loss ($\lambda=0.925$), an ablation against a standard pairwise margin loss on identical temporal pairs is needed to isolate whether the DPO-style formulation provides unique benefits over conventional contrastive fine-tuning.
- **Confounding Factors in BEIR Temporal Correlation Analysis:** The claim that optimal interpolation weights $\alpha$ correlate with BEIR dataset publication years (Fig. 5) is interesting but likely confounded by domain and stylistic drift. Older datasets (e.g., MS MARCO, 2016) consist of web-crawled conversational text, while newer ones (e.g., Climate-FEVER, 2020) are scientific claim-evidence pairs. Without controlling for domain or query complexity, the observed correlation may reflect stylistic adaptation rather than pure temporal semantic alignment.

## Nice-to-Haves
- **End-to-End $\alpha$ Routing Implementation:** Train or simulate a lightweight query-intent classifier (or use an off-the-shelf date parser + heuristics) to predict $\alpha$ at inference time. Reporting performance without ground-truth timestamp leakage would strengthen claims of deployability.
- **Full-Corpus or Public Temporal Benchmark Evaluation:** Supplement the custom benchmarks with a standard temporal IR dataset (e.g., TREC Temporal Track) or evaluate over a complete Wikipedia index to verify true retrieval recall from an unfiltered pool.
- **Mechanistic Probing:** Provide embedding space visualizations (e.g., PCA/t-SNE of aligned vs. misaligned pairs) or attention/token importance maps to verify that the model learns genuine temporal grounding rather than exploiting corpus-specific lexical drift.
- **Statistical Reporting:** Include multi-seed variance or bootstrap confidence intervals for key metrics, as point estimates alone limit robustness assessment.

## Novel Insights
One paragraph synthesizing genuinely novel observations. The paper successfully transfers weight-space time-vector interpolation from generative LLMs to bi-encoder retrievers, revealing that dense retrieval embeddings exhibit similar geometric smoothness across temporal distribution shifts. More intriguingly, the BEIR case study suggests that "static" general-purpose benchmarks contain latent temporal drift: retrieval performance systematically peaks when the model's temporal focus aligns with the dataset's publication year. This implies that temporal misalignment may be a hidden confounder in cross-dataset IR evaluations, and that even non-temporal retrieval tasks could benefit from corpus-temporal grounding when domain language evolves alongside dataset creation.

## Suggestions
- **Add Strong Temporal Baselines:** Implement and compare against time-decay scoring, explicit date embeddings, and query-rewriting with timestamp extraction to contextualize TPOUR's gains against simpler, widely-used temporal IR techniques.
- **Evaluate on Unfiltered Retrieval Corpora:** Report performance on a full Wikipedia snapshot or a standard temporal retrieval benchmark without Contriever-based pre-filtering to demonstrate TPOUR's ability to up-rank temporally relevant documents from a broader, unbiased candidate space.
- **Isolate TRPO's Contribution:** Include an ablation comparing TRPO to a standard pairwise margin contrastive loss trained on the same $(D^t, D^{t'})$ pairs, and analyze gradient norms to confirm that the DPO-style formulation provides distinct optimization dynamics over conventional contrastive objectives.
- **Control for Domain in BEIR Analysis:** Stratify BEIR results by domain or compute partial correlations between $\alpha$ and dataset age while controlling for topic/query type to disentangle temporal alignment from stylistic/domain adaptation.

---

## UnvaQLYFMJ

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (4.1/10)
- Match: N/A

### Final Review

## Summary
SEAL proposes enhancing the linear separability of frozen foundation model embeddings for unsupervised clustering by extracting patch-level spatial dependencies via a Graph Attention Network (GAT) and fusing them with visual features through mutual distillation. A bi-level linear classifier then recovers human-consistent cluster assignments without labeled data. Evaluated across 26 datasets and 7 foundation backbones, the method demonstrates consistent improvements in clustering accuracy, backbone stability, and diagnostic feature separability compared to strong baselines like TURTLE.

## Strengths
- **Comprehensive and robust empirical validation:** The evaluation spans 26 diverse datasets (covering fine-grained, scene, texture, and remote sensing domains) and 7 foundation backbones (ResNets, ViTs, DINOv2), with explicit stability analysis (Appendix C) demonstrating that spatial distillation consistently reduces performance variance across architectures.
- **Well-motivated, plug-and-play design:** By freezing foundation features and integrating a lightweight distillation + linear probing pipeline, SEAL avoids costly task-specific pretraining. The approach directly addresses a known bottleneck in foundation-model clustering (lack of task-specific linear separability) without requiring end-to-end fine-tuning or auxiliary labeled data.

## Weaknesses
- **Ambiguous graph construction and scalability claims:** Section 3.2 defines the GAT input using the full dataset size $N$ and an edge set $E = \{(i,j) \mid i \neq j\}$, implying a fully connected $\mathcal{O}(N^2)$ graph. For large-scale datasets like PatchCamelyon ($N > 290,000$) or CIFAR-10 ($N=50,000$), this is computationally infeasible without mini-batching, spatial partitioning, or K-NN sparsification. The lack of a clear description of how the graph is practically constructed, batched, or scaled undermines reproducibility and contradicts the claimed efficiency.
- **Contradictory training formulation and sign inconsistencies:** Equation 2 and Section 3.1 state the framework is *"trained by jointly optimizing $\theta$, $R$, and $w$,"* yet Section 3.4 explicitly notes: *"After obtaining the spatially aware embeddings... we freeze them and adopt the bi-level optimization protocol."* This is a two-stage pipeline, not joint optimization, and the discrepancy obscures the actual training schedule. Additionally, Equation 1 contains a sign inversion ($\min -\sum [L_{ce} - \beta L_{ent}]$) that mathematically implies maximizing cross-entropy, contradicting standard clustering objectives and suggesting a typographical error in the formalization.
- **Insufficient isolation of the spatial/GAT contribution:** The primary ablation compares the full SEAL pipeline against TURTLE, but does not isolate whether the GAT-based spatial extractor, the distillation objective, or simply the injection of additional capacity drives the gains. There is no experiment replacing the GAT with simpler spatial aggregation (e.g., direct patch-token pooling or self-attention), nor an ablation justifying the use of a separate ResNet-50 for spatial features instead of leveraging patch tokens directly from CLIP/DINOv2. Without these controls, the claim that explicit graph attention is necessary remains unsubstantiated.

## Nice-to-Haves
- **Statistical significance reporting:** While results are averaged over 10 seeds, some reported gains over TURTLE are marginal (<2%). Including confidence intervals or paired significance tests would better contextualize whether improvements reliably exceed seed variance.
- **Granular compute profiling:** Figure 4 shows total runtime but does not break down time per pipeline stage (GAT construction, distillation, bi-loop probing) or report peak VRAM. A detailed FLOP/memory breakdown would help practitioners evaluate the trade-off between marginal accuracy gains and preprocessing overhead.

## Novel Insights
The work empirically demonstrates that global semantic embeddings from foundation models, while rich in high-level priors, often inhabit geometrically entangled manifolds that resist linear partitioning. By showing that injecting explicit spatial relational modeling can act as a structural corrective prior—shifting embeddings toward linearly separable clusters without end-to-end fine-tuning—the paper provides a practical pathway for adapting frozen models to fine-grained or structurally complex domains. The observation that this spatial awareness stabilizes performance across highly heterogeneous backbones (CNNs to large ViTs) further suggests that spatial inductive bias is a systematically missing component in standard vision foundation models for unsupervised grouping tasks.

## Suggestions
- **Explicitly detail the graph construction and training protocol:** Clarify whether the GAT operates per-image, per-mini-batch, or via approximate nearest neighbors, and provide the exact sparsification or batching strategy used to handle large $N$. Add a concise algorithmic pseudocode block that maps the two-stage training pipeline, loss weights, and the exact point at which embeddings are frozen before the bi-loop optimization.
- **Introduce targeted component ablations:** Compare the GAT against simpler spatial aggregation baselines (e.g., average pooling of foundation patch tokens or a standard attention layer). Additionally, test whether extracting spatial features directly from CLIP/DINOv2 patch encoders yields comparable gains to the auxiliary ResNet-50, isolating whether performance stems from spatial modeling or architectural heterogeneity.
- **Clarify evaluation framing and add a limitations section:** Explicitly state that triplet accuracy serves as a supervised diagnostic proxy rather than an unsupervised separability metric, and include a dedicated limitations discussion analyzing failure cases (e.g., texture/abstract datasets where spatial priors degrade performance), computational overhead, and sensitivity to the target cluster count $K$.

---

## 80JylHgQn1

- GT: Accept (Oral) (avg 7.0)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a dual-system framework for audio-driven video avatar generation, coupling a deliberative MLLM-based planner ("System 2") with a reactive multimodal diffusion transformer ("System 1"). The method addresses critical training-inference mismatches through a novel Pseudo-Last-Frame conditioning strategy and a staged multimodal warm-up, effectively decoupling identity preservation from motion restriction. Comprehensive subjective and MLLM-based evaluations demonstrate that the approach significantly improves contextual coherence and behavioral expressiveness over existing reactive avatar models.

## Strengths
- **Pseudo-Last-Frame (PLF) Conditioning Effectively Solves Identity-Motion Conflict:** The identification of a spurious correlation introduced by traditional static reference conditioning is astute, and the PLF strategy (treating the reference as a temporally shifted inference target rather than a static training constraint) elegantly resolves this trade-off. The ablation visualizations (Figures 8 & 9) clearly demonstrate its superiority in maintaining identity during high-dynamic autoregressive generation.
- **Practical Multimodal Fusion via Symmetric Attention and Staged Warmup:** The symmetric token concatenation combined with a two-stage modality warm-up successfully mitigates audio-text interference without degrading pre-trained video priors. This is a robust, generalizable engineering contribution for integrating new conditioning signals into diffusion backbones.
- **High Methodological Transparency and Reproducibility:** The manuscript provides exceptional documentation, including explicit training schedules, dataset filtering pipelines (PySceneDetect, Q-align, SyncNet), MLLM prompting templates, and detailed evaluation protocols. This level of detail sets a strong standard for verifiable generative video research.

## Weaknesses
- **Insufficient Isolation of Agentic Reasoning vs. Rich Text Conditioning:** The core claim attributes perceptual and semantic gains to the deliberative "System 2" planning process. However, the paper lacks a dense-prompt or keyword-extraction baseline that feeds semantically rich text directly into the DiT without the multi-step agentic pipeline. Without this ablation, it remains unclear whether improvements stem from the structured reasoning process itself or simply from providing the diffusion model with richer, task-specific textual guidance.
- **Heavy Reliance on Subjective/LLM Metrics Due to Objective Saturation:** The authors correctly note that standard objective metrics (Sync-C, IQA, FID) saturate quickly and are insensitive to high-level semantics. Consequently, the central claims of improved contextual coherence rest almost entirely on GSB scores and MLLM-based pairwise evaluations. While useful, these perceptual metrics lack a standardized, fully objective counterpart for high-level motion-intent alignment, making independent verification of the "System 2" impact more challenging.
- **Unspecified Dataset Provenance and Licensing:** While the 15,000-hour data curation pipeline is meticulously documented, the source domains, collection methods, and licensing status of the raw video corpus are not provided. For a venue with strict reproducibility and ethical standards like ICLR, clarifying data compliance and public availability is necessary.

## Nice-to-Haves
- Report inter-annotator agreement (e.g., Fleiss' Kappa) or confidence intervals for the subjective user studies to validate score consensus beyond raw percentages.
- Provide a sensitivity analysis or failure-boundary visualization for the PLF RoPE positional offset, particularly under extreme pose/viewpoint mismatches between the reference image and initial frame, to demonstrate robustness across the autoregressive horizon.
- Include attention weight maps comparing the proposed symmetric multimodal fusion against standard cross-attention, visually demonstrating how text tokens maintain influence when competing with high-magnitude audio features.
- Discuss concrete latency mitigation pathways (e.g., speculative decoding, distilled planners, or asynchronous execution) for the 20-30s MLLM inference overhead to strengthen practical deployability claims.

## Novel Insights
The paper successfully operationalizes a high-level cognitive metaphor into concrete architectural constraints. By reframing reference image conditioning from a static spatial anchor to a temporally delayed "carrot on a stick," the work fundamentally decouples identity preservation from motion restriction, offering a reusable mechanism for autoregressive character synthesis. Furthermore, treating low-level audio synchronization and high-level behavioral planning as functionally distinct but architecturally fused modalities (via staged warmup and symmetric attention) provides a practical blueprint for injecting deliberate semantics into otherwise purely reactive diffusion pipelines, effectively bridging the gap between generative video and controllable behavioral simulation.

## Suggestions
- **Add a Dense-Prompt Baseline:** Implement an ablation where keywords or a dense descriptive prompt extracted from the MLLM output directly condition the DiT bypassing the multi-step agentic reasoning pipeline. This will cleanly isolate the marginal contribution of structured deliberation versus prompt richness.
- **Clarify Data Compliance:** Add a brief statement in the appendix or ethics section detailing the source domains, collection strategy, and licensing terms for the 15,000-hour training corpus to satisfy reproducibility and ethical review standards.
- **Strengthen Subjective Validation:** Include inter-annotator agreement metrics or basic statistical significance tests (e.g., Wilcoxon signed-rank) for the primary GSB and artifact scores to reinforce the reliability of perceptual claims.
- **Explicitly Frame Metric Saturation:** In the main text, explicitly state that the contribution is primarily semantic/perceptual rather than low-level fidelity, setting clear expectations for readers evaluating the objective vs. subjective metric divergences.

---

