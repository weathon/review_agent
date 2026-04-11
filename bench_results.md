# ICLR Benchmark Results

Date: 2026-04-10 18:58
Critic/Merger: deepseek/deepseek-v3.2 (OpenRouter)
Neutral: deepseek/deepseek-v3.2, Related Work: deepseek/deepseek-v3.2:online (OpenRouter)

## gDxJK8yvZU

- GT: Accept (Poster) (avg 7.5)
- Predicted: N/A (None/10)
- Match: N/A

### Final Review

ERROR: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{"error":{"message":"This model\'s maximum context length is 163840 tokens. However, you requested 65536 output tokens and your prompt contains at least 98305 input tokens, for a total of at least 163841 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=98305)","type":"BadRequestError","param":"input_tokens","code":400}}', 'provider_name': 'Parasail', 'is_byok': False}}, 'user_id': 'user_32IhT2MfrwUmKddLDbQSpLcYscC'}

---

## dBJpBmn5MH

- GT: Reject (avg 1.0)
- Predicted: N/A (1.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a probabilistic loss function for improving the adversarial robustness of neural networks. For classification, it replaces the one-hot targets in cross-entropy loss with a posterior distribution derived from a Gaussian Mixture Model (GMM) fit to the input features. For regression, it introduces a cluster-weighted loss. The method claims to confer robustness without the computational overhead of adversarial training or defensive distillation.

## Strengths
- **Unified framework for classification and regression**: The paper extends its core idea beyond the typical classification setting to regression problems, which is a less common focus in adversarial robustness literature.
- **Exploration of an alternative to data-centric defenses**: The work investigates a loss-modification approach, positioning it as a potential alternative to methods that require adversarial example generation or architectural changes.

## Weaknesses
### Fatal
- **Insufficient and non-standard experimental validation**: The paper's central claim of adversarial robustness is unsupported by evidence. Section 3 provides only a single, anecdotal example (a perturbed MNIST digit) and states "similar results" on ImageNet without reporting a single quantitative metric (e.g., clean accuracy, robust accuracy, attack success rate). There are no comparisons to standard baselines (e.g., standard training, adversarial training, label smoothing) or evaluations on established benchmarks (e.g., CIFAR-10). For an ICLR submission on adversarial robustness, this complete lack of rigorous empirical evidence is fatal. The claim that the method works cannot be assessed.

### Major:
- **Flawed and incomplete theoretical motivation**: The paper's foundational premise is incorrect. It asserts that adversarial vulnerability stems from the one-hot label in cross-entropy loss "forcing the network to overfit" (Sections 2, 4). This misunderstands standard training: cross-entropy with one-hot labels encourages high probability for the correct class but does not force a one-hot output from the network. Adversarial examples exploit decision boundaries in high-dimensional space, not an artifact of label encoding. Building a method on this flawed premise undermines its rationale.
- **Poorly described and motivated methodology**: The proposed method is confusing and its connection to robustness is opaque. For classification, Algorithm 1 fits a GMM on input features **X** to obtain posterior probabilities `τ`, which are then used as soft targets. It is not justified why re-weighting the loss based on input-space clustering should lead to semantic label smoothing that improves robustness. The algorithm description contains inconsistent notation (e.g., using `M` ambiguously) and undefined variables, hindering understanding and reproducibility.
- **Weak and non-standard empirical setup**: The experimental design is inadequate. The MNIST model has ~6M parameters (Appendix), which is extreme overkill and risks memorization. Training for only 5 epochs is insufficient to claim robust generalization. Evaluations are limited to FGSM and Carlini-Wagner L2 attacks, omitting standard strong attacks like PGD or AutoAttack, which is necessary to rule out gradient masking. The extension to ImageNet is mentioned but not demonstrated.

### Minor
- **Under-specified regression method**: Algorithm 2 for regression is critically underspecified. It clusters input features **X** while ignoring the targets **y** for defining the loss weighting, a decoupling that is not motivated. The number of clusters `N` is a hyperparameter with unexplored sensitivity.

## Nice-to-Haves
- A computational cost analysis comparing the EM-based loss calibration to the overhead of standard adversarial training.
- An ablation study on the impact of key hyperparameters (e.g., the prior probability of 0.9 in Algorithm 1, the cluster count `N` for regression).
- Discussion relating the method to existing literature on label smoothing and knowledge distillation.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Strengths Removed:**
- "The paper is well-written" and "the topic is important": These are generic and apply to many papers.
- "Clear Motivation and Problem Framing" (from Review 2): Upon verification, the core motivational argument is factually flawed, as detailed in the Weaknesses section.

**Weaknesses Removed:**
- **Criticism about missing comparisons to "seminal and contemporary methods" or specific citations (e.g., TRADES, MART)**: Removed per the hard rule "DO NOT mention missing related works, as you do not have external sources to confirm their existence."
- **Criticism that the method "requires running an EM algorithm on the entire training set, which is likely prohibitive"**: While a valid concern, it is softened to a "Nice-to-Have" as a computational analysis is not a standard requirement for an initial methodological proposal, though it would strengthen the paper.
- **Nitpicks about placeholder text ("equation **??**") and formatting**: Removed per the hard rule against formatting/style nitpicks.
- **Reproducibility concerns about "undisclosed hyperparameters"**: Removed per the hard rule against nitpicks on reproducibility regarding trivial implementation details. The major issue is the complete absence of results, not the granularity of hyperparameters.

## Suggestions
1. **Conduct a rigorous, quantitative evaluation.** As a minimum, report clean and robust accuracy on standard datasets (e.g., MNIST, CIFAR-10) against standard attacks (e.g., FGSM, PGD, AutoAttack) with clear `ϵ` budgets. Compare directly to strong baselines including standard training, adversarial training (PGD-AT), and label smoothing.
2. **Re-evaluate and correct the core motivation.** Address the misunderstanding about one-hot labels and cross-entropy. Provide a sound theoretical intuition or empirical analysis (e.g., of loss landscape smoothness or gradient norms) for why the proposed loss modification might improve robustness.
3. **Clarify the methodology.** Rewrite Algorithm 1 and Section 2.1 with consistent notation, clearly defined variables, and a step-by-step explanation of how the GMM posterior leads to a robust loss. Justify the design choice of using input features to define label probabilities.

---

## BjQqvH2LgW

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (2.5/10)
- Match: N/A

### Final Review

## Summary
This paper introduces UnCoVAEr, a latent-variable model designed to estimate the causal effects of human-interpretable concepts on model predictions when some confounding concepts are unobserved. The method partitions an image's latent representation into a discrete confounder-related component and a continuous residual component, enabling bias correction via backdoor adjustment. Evaluation is conducted on a controlled semi-synthetic MorphoMNIST benchmark with several confounding patterns.

## Strengths
- **Clear and well-motivated problem formulation:** The paper effectively identifies and formalizes a critical, under-addressed issue in concept-based interpretability: latent visual confounding due to incomplete concept annotations. The motivating examples (medical imaging, fairness) are compelling and establish clear practical relevance.
- **Principled and novel methodological design:** The core innovation of partitioning the latent space into confounder (`ZC`) and residual (`ZS`) components provides a structured inductive bias that directly aligns with the assumed causal graph and the goal of learning adjustment-sufficient proxies. This represents a non-trivial extension of prior latent-variable causal models like CEVAE to the concept-based setting.
- **Rigorous and comprehensive synthetic evaluation:** The experimental design on MorphoMNIST is thorough, testing distinct confounding scenarios (single, common, multiple) and evaluating robustness to distribution shift (ID/OOD). The ablation study convincingly validates the importance of the partitioned latent space and the image reconstruction term.
- **Strong empirical performance in core settings:** In the single and common confounder scenarios, UnCoVAEr substantially outperforms a range of strong baselines (CEVAE, CaCE, Image-adjustment, CBM variants) in reducing ATE estimation bias, demonstrating the efficacy of its design.

## Weaknesses
### Major:
1. **Exclusively synthetic validation limits practical claims:** The entire empirical validation is conducted on a single, highly controlled semi-synthetic dataset (MorphoMNIST). While this is valuable for proof-of-concept and ablation, it provides insufficient evidence to support the paper's claim of providing a "practical tool for trustworthy concept-level causal inference" in real-world, partially annotated image datasets. The leap from simple synthetic digits with known, low-dimensional confounders to complex real-world settings (e.g., medical imaging, facial attributes) is substantial and remains unsubstantiated.
2. **Methodological failure under complex confounding:** In the "multiple confounders" scenario, where a concept is influenced by two confounders via a non-linear (XOR) mechanism, UnCoVAEr's performance degrades significantly. The naive estimator and CBM achieve error rates near the oracle, while UnCoVAEr's error is an order of magnitude higher (0.070 vs. 0.011). Furthermore, the proposed confounding detection criterion fails completely in this setting (Figure 3). This indicates a serious limitation in handling intricate, non-linear confounding structures that may exist in practice.
3. **Strong and potentially untestable assumptions:** The method's identifiability and effectiveness rely on the assumption that all relevant confounders leave a detectable visual trace in the image and that there are no unobserved colliders. In practice, these assumptions are difficult or impossible to verify, limiting the method's reliability for real-world auditing where the true causal structure is unknown.

### Minor:
1. **Incomplete comparison to contemporary baselines:** The paper cites several directly relevant methods from the proximal causal inference literature that use images as proxies (e.g., Kompa et al. 2022, Israel et al. 2023, Schulte et al. 2025) but does not include them in the empirical comparison. This omission weakens the claim of outperforming "prior latent-variable approaches."
2. **Missing experiment on observed confounders:** The paper describes an additional experimental setup (modifying the multiple confounders variant to have one confounder observed) but does not present any results, missing an opportunity to demonstrate the method's behavior when some confounders are, in fact, observed.
3. **Limited analysis of learned confounder proxies:** While the paper claims the learned proxy `ZC` aligns with underlying latent factors, it provides no quantitative analysis (e.g., correlation with the true confounder in the synthetic setting) or qualitative visualization (e.g., counterfactuals by intervening on `ZC`) to substantiate this interpretability claim.
4. **Clarity of estimation and detection procedures:** The description of the bootstrap test for confounding detection and the exact sampling procedure for ATE estimation (Eq. 7) is somewhat vague, making it difficult to fully assess or replicate these components.

### Trivial
- None.

## Nice-to-Haves
- Sensitivity analysis to the latent dimension `K` and robustness to model misspecification (e.g., if the true confounder is continuous).
- Ablation study quantifying the contribution of the mutual information (CLUB) regularizer.
- Visualization of counterfactual images generated by intervening on the learned confounder proxy `ZC`.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strengths about writing style, importance of topic, or extensive experiments:** These are generic and apply to many papers. The retained strengths are specific to this paper's contributions.
- **Weaknesses about undisclosed hyperparameters or reproducibility:** The paper includes a detailed reproducibility checklist, provides code, and specifies key hyperparameters and training details, which is sufficient for a conference submission.
- **Weaknesses about the method being "not novel" or "incremental":** The core contribution—partitioning the latent space for concept-level adjustment under visual confounding—is a distinct and non-trivial extension of prior work like CEVAE.
- **Weaknesses questioning the existence of cited models/benchmarks:** All cited works are assumed to exist and be available.

## Suggestions
- To significantly strengthen the paper, include at least a preliminary validation on a real-world dataset with plausible latent confounding (e.g., a medical imaging dataset with site/lab as a suspected confounder). Even without ground-truth ATE, analyzing the stability of estimates and the interpretability of learned proxies would greatly enhance practical relevance.
- Provide a deeper diagnostic analysis of the failure mode in the XOR-based multiple confounder scenario. Investigate whether the issue stems from identifiability, model capacity, or the bootstrap test, and discuss implications for practitioners.
- Compare against additional proximal causal inference baselines that use image proxies (e.g., Kompa et al. 2022, Schulte et al. 2025) to more thoroughly position the method within the current literature.

---

## vQLUAkl5SG

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary
DRAGON is a training-free framework for LLM unlearning that guards inference via a lightweight detection module and a reasoning-based in-context intervention. It identifies prompts related to data that should be forgotten (using a scoring model and similarity matching) and prepends a dynamically generated chain-of-thought instruction to steer the base LLM toward safe or refusal responses. The method requires no retain data or modification of the base model’s weights, making it applicable to black-box and continual-unlearning scenarios. The paper also introduces three new metrics (Refusal Quality, Dynamic Deviation Score, Dynamic Utility Score) and demonstrates strong performance across privacy (TOFU), harmful knowledge (WMDP), and copyright (MUSE) unlearning tasks.

## Strengths
- **Practical, scalable design.** DRAGON never updates the base LLM’s parameters, works without retain data, and relies only on a small guard model for CoT generation. Experiments show consistent effectiveness across model sizes (1.5B–70B) and types (instruct vs. base, open vs. black‑box), with no retraining cost when scaling to larger models (Figure 2, Table 13).
- **Strong empirical results across diverse tasks.** On TOFU, DRAGON achieves the best deviation score (e.g., 21.4 on TOFU‑1%) while preserving near‑perfect model utility (0.6337). On WMDP, it drives multiple‑choice accuracy close to random guessing (25–28%) and yields the highest Refusal Quality, outperforming fine‑tuning and prompting baselines (Tables 1–2, 8–9). It also leads in continual‑unlearning settings (Table 3) and on the MUSE copyright benchmark (Table 14).
- **Robustness and ablation analysis.** The detector maintains high accuracy under adversarial perturbations (language‑mix, typos, AIM attacks; Tables 6, 25–27). Ablations confirm the necessity of generated CoT over static templates (Table 4, 15) and show the method is stable across threshold choices and guard‑model variants (Tables 23–24, 21–22).
- **Novel metrics for continual unlearning.** Dynamic Deviation Score (DDS) and Dynamic Utility Score (DUS) provide a principled way to measure stability across sequential unlearning requests, addressing a gap in prior evaluation. Refusal Quality (RQ) combines refusal rate, template alignment, and generation quality to penalize incoherent outputs.

## Weaknesses
### Major:
- **Increased inference latency and operational overhead.** For forget‑related prompts, DRAGON adds detection time, guard‑model inference, and a longer context window due to prepended CoT instructions. Table 31 shows guard inference can add ~600 ms per forget query, which may hinder real‑time deployment. While the paper notes safety justifies this cost, a deeper latency‑versus‑performance trade‑off analysis and discussion of optimization strategies (e.g., caching, prompt compression) would strengthen practical applicability.
- **Limited validation of the new metrics against human judgment.** Refusal Quality uses heuristic weights (1, 1, 0.2) for its components; DDS includes an arbitrary β = 0.5. Although these metrics are intuitively motivated and correlate with observed performance, the paper does not demonstrate that they align with human assessments of refusal appropriateness or stability. For a paper introducing new evaluation measures, some form of human validation or correlation analysis would bolster confidence.
- **Generalization of the guard model to entirely new unlearning categories is underexplored.** The guard model is trained on synthetic CoT data for privacy, harm, and copyright tasks. While it generalizes well within those categories (e.g., across TOFU splits and BLUR; Table 30), its ability to handle unseen categories (e.g., removing biased statements or factual inaccuracies) is not tested. This limits the claim of a “systematic framework” applicable to arbitrary unlearning requests.

### Minor:
- **Reliance on synthetic data for detector training and CoT generation.** The unlearn store and CoT dataset are built using paraphrases or examples generated by other LLMs (Llama3.1‑70B, GPT‑4o). Although the paper applies rejection sampling and quality controls, a deeper analysis of potential distributional mismatches (e.g., how synthetic harmful queries differ from real‑world adversarial prompts) would help bound the method’s reliability.
- **Some experimental details are relegated to the appendix.** Key information such as exact prompts for CoT generation (Appendix F), classifier training details (Appendix C), and robustness‑test setups (Appendix D.6) is comprehensive but not always easily accessible from the main text. A more streamlined presentation of critical implementation choices would improve readability.

### Trivial:
- **The metric “consistency score” used in ablation studies is an ad‑hoc measure (embedding similarity between query and response).** While it helps illustrate the benefit of CoT, it is not a standard metric and its relationship to human‑perceived coherence is not established. This does not undermine the core results but suggests the ablation could be complemented with more established metrics.

## Nice-to-Haves
- **Extended evaluation on a broader suite of unlearning tasks** (e.g., misinformation removal, stereotype debiasing) would further demonstrate the framework’s generality beyond the three benchmarks studied.
- **Optimization strategies for latency reduction**, such as prompt compression, caching of frequent CoT patterns, or distillation of the guard model, would make the method more deployment‑friendly.
- **A sensitivity analysis tying detector error rates (false positives/negatives) to downstream metric changes** would provide a clearer picture of how detection failures impact overall unlearning performance.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Strength: “The paper is well‑written, the topic is important.”** Removed as generic; the review retains specific, evidence‑backed strengths.
- **Weakness: “The detector’s robustness and generalization are insufficiently established, undermining the framework’s reliability.”** The paper includes substantial robustness evaluation (Tables 6, 25–28, Appendix D.6) against language‑mix, typos, AIM attacks, and out‑of‑distribution prompts, showing the detector maintains high accuracy. The criticism overstates the issue.
- **Weakness: “Claims of scalability to black‑box models and continual unlearning are not adequately supported.”** The paper evaluates on state‑of‑the‑art proprietary models (GPT‑4o, Llama‑4) in Figure 2b and reports results across nine LLMs. Continual unlearning is tested with three sequential forget sets and measured with DDS/DUS; while more steps could be tested, the claim is supported within the evaluated scope.
- **Weakness: “Comparison with fine‑tuning baselines is unfair because they assume retain data.”** The paper explicitly frames DRAGON as a method for data‑limited scenarios and still outperforms many baselines that do use retain data. The comparison is informative, not unfair.
- **Weakness: “The detection threshold τ is never explained.”** Threshold selection is discussed in sensitivity studies (Tables 23‑24), showing performance is stable across a range of values.
- **Weakness: “The guard model introduces a dependency on a powerful external model (GPT‑4o).”** The paper notes the guard model can be trained with any capable reasoning model (e.g., o3, DeepSeek‑R1) and shows performance remains stable across different guard models (Tables 21‑22). This is a design choice, not a flaw.
- **Weakness: “Statistical significance is not reported.”** Single‑run evaluation is standard in large‑scale LLM benchmarking; requesting statistical tests is a nice‑to‑have, not a core requirement.
- **Weakness: “The method’s performance on the most capable base models is only briefly mentioned.”** Figure 2b and accompanying text provide quantitative results on GPT‑4o, Llama‑4, and Llama‑3.1‑70B‑Instruct, which is sufficient.

## Suggestions
- **Add a subsection analyzing latency‑performance trade‑offs** and discuss potential optimizations (e.g., caching, prompt summarization) to address deployment concerns.
- **Include a small‑scale human evaluation** to validate that Refusal Quality and DDS/DUS correlate with human judgments of refusal appropriateness and stability.
- **Expand the discussion of guard‑model generalization** with a brief experiment on one additional unlearning category (e.g., biased statements) to better support the “systematic framework” claim.

---

## 6zZGNJRO56

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a weak-to-strong (W2S) learning paradigm for no-reference video quality assessment (VQA). It trains a strong student model (based on LLaVA-OneVision) using pseudo-labels from an ensemble of existing VQA models and synthetic distortion simulators, avoiding human annotations. The framework incorporates a ranking-based formulation to unify heterogeneous signals and an iterative training strategy with difficulty-guided sampling. The method achieves state-of-the-art results on ten benchmarks, with particularly strong gains on out-of-distribution (OOD) datasets.

## Strengths
- **Novel and well-motivated adaptation of W2S to VQA:** The paper provides the first empirical demonstration of a clear weak-to-strong generalization effect in VQA (Fig. 4, Table 5), showing that a student can match or surpass its teachers, especially on OOD data. This addresses a critical bottleneck of human annotation scarcity in the field.
- **Comprehensive and rigorous evaluation:** The method is evaluated on ten diverse benchmarks (five in-domain, five OOD), showing consistent and significant improvements, e.g., a 30.59% relative SRCC gain on LIVE-YT-HFR. The ablation study (Table 1) convincingly demonstrates the incremental benefit of each proposed component (ensemble, synthetic teachers, confidence loss, iterative training).
- **Technically sound and well-engineered framework:** The integration of homogeneous teacher ensembles and heterogeneous synthetic distortion simulators via a ranking formulation is principled and effective. The iterative training with difficulty-guided sampling (gMAD) is a clever mechanism for progressive self-improvement, and the detailed methodology and appendix facilitate reproducibility.

## Weaknesses
### Major:
- **The core claim of a distinct W2S effect is confounded by increased data scale.** The student is trained on 200k videos, while the teachers are trained on only 27k (LSVQ). The authors acknowledge in Section 3.3 that the larger dataset likely contributes to gains. Although a supervised baseline trained on LSVQ is provided in Appendix Table 5, a controlled ablation is missing—e.g., training the same student on a 27k subset of the pseudo-labeled data to isolate the effect of the W2S dynamic from pure data scaling. This undermines the claim that W2S itself, rather than dataset size, drives the observed OOD improvements.
- **Insufficient ablation of the iterative training strategy.** Table 1 shows progressive gains across stages, but no comparison is made to a single-stage baseline that uses all components (ensemble, synthetic distortions, confidence loss) on the full 700k pairs. Without this, the necessity and specific contribution of the iterative process—and its difficulty-guided sampling—remain unsubstantiated. The claimed benefit of "progressively focusing on challenging cases" lacks direct evidence.

### Minor:
- **Limited analysis of what the student learns beyond teacher biases.** All weak teachers are trained on LSVQ (UGC videos). The student's improved OOD performance might reflect a more robust approximation of the teachers' biases rather than learning fundamentally new quality concepts. The paper lacks error analysis or qualitative examples showing cases where the student corrects specific teacher errors, which would strengthen the claim of achieving "generalized" VQA.
- **High computational cost without thorough cost-benefit analysis.** Training a 7B-parameter LMM for three iterative stages on 200k–700k video pairs (8×A800 GPUs for ~2 days per stage) is extremely resource-intensive. While large-scale training is common in the field, the paper does not compare the total computational cost (GPU hours, memory) to supervised baselines or analyze the trade-off between saved annotation cost and increased compute expense. This limits practical accessibility and reproducibility for many researchers.

### Trivial:
- Some figures (e.g., Fig. 1, 6) suffer from low resolution or formatting artifacts from the PDF parser; these should be crisply rendered in the final version.

## Nice-to-Haves
- **Ablation on teacher ensemble composition:** Studying the impact of adding/removing specific teachers would clarify how ensemble diversity affects the W2S effect.
- **Exploration of a more efficient student architecture:** Demonstrating that W2S gains are not solely due to model scale (e.g., using a smaller ViT) would broaden the method's practical impact.
- **Test on a benchmark with completely novel, non-synthetic distortion types** (e.g., generative model artifacts) to further probe the limits of generalization beyond the teachers' knowledge.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"Weaknesses" about unfair comparison with other SOTA methods:** The paper compares fairly against recent methods (VQA², VQAThinker) that also use large-scale data and advanced architectures. The asymmetry (different data scales) is acknowledged and does not unfairly favor the proposed method.
- **Criticism about the existence or release status of cited models/tools:** All models, benchmarks, and tools referenced (e.g., LLaVA-OneVision, SlowFast, ffmpeg) are assumed to exist and be available. Concerns about reproducibility due to dataset licensing are noted but do not constitute a methodological flaw.
- **Request for theoretical proofs or confidence intervals:** The paper is an empirical contribution; demanding theoretical analysis or confidence intervals for large-scale benchmarks is not standard practice in the VQA community.
- **Nitpicks about formatting, style, or undisclosed hyperparameters:** The paper is well-written, and key implementation details are provided in the main text and appendix.

## Suggestions
- **Include the supervised baseline (LSVQ-labeled student) in the main results table (Table 1)** to directly contextualize the performance of W2S models relative to human-annotation upper bounds.
- **Add a controlled ablation experiment** training the student on a 27k subset of the pseudo-labeled data to disentangle the effects of dataset scale from the W2S dynamic.
- **Conduct a failure analysis** to identify video types or distortion patterns where the student underperforms, providing insight into the method's limitations and boundaries of the W2S effect.
- **Provide a clearer computational cost analysis** comparing total GPU hours, memory, and inference time against supervised baselines to help readers assess the practical trade-offs.

---

## m2MeiYOJED

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (0.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes PRISM, a framework for partial-label graph learning where each graph is associated with a candidate set containing the true label. PRISM integrates spatial cues (via prototype-guided substructure matching) and spectral cues (via multi-band frequency decomposition) into a hybrid relational graph, then performs iterative label propagation under candidate constraints to disambiguate labels. Experiments on five graph classification benchmarks demonstrate consistent improvements over baselines under various noise settings.

## Strengths
- **Novel integration of complementary views:** The simultaneous use of prototype-aligned substructures (spatial) and multi-band spectral embeddings provides two distinct, complementary relational graphs for label disambiguation, which is a well-motivated and technically cohesive design.
- **Comprehensive empirical validation:** The paper evaluates across five diverse graph datasets, multiple noise levels (uniform, hierarchical, competitive), and against a broad suite of baselines (GNNs, pooling methods, contrastive learning, partial-label methods). The consistent and often substantial improvements are convincing.
- **Interpretable components:** The class-specific attention visualization (Figure 3) provides evidence that the model identifies meaningful substructures, and the ablation study cleanly isolates the contribution of each major component.

## Weaknesses
### Major:
- **Theoretical analysis relies on strong, unrealistic assumptions.** Theorems 1 and 2 assume asymptotic conditions (graph size → ∞) and idealized relationships (e.g., node embeddings have mean exactly equal to class prototypes, adjacency matrix is a deterministic function of the global embedding) that do not hold for the finite, real-world graphs used in the experiments. While the proofs are internally consistent, the disconnect between theory and practice limits the theoretical insight provided.
- **Prototype bank may accumulate noise early in training.** The prototype update (Eq. 3) aggregates graphs for which a label is in the candidate set, but this set contains false positives. The paper does not analyze the sensitivity of prototypes to this initial noise or provide mechanisms to mitigate early-stage corruption, which is a risk for error propagation.
- **Missing comparison with directly relevant, contemporaneous work.** The paper cites recent graph partial-label learning methods (MATE, Code) but does not include them as baselines. This omission makes it difficult to assess the claimed superiority against the current state of the art.
- **Limited analysis of the spectral module's necessity and behavior.** While the multi-band frequency attention is a core contribution, there is no ablation comparing it to a simpler spectral baseline (e.g., using a standard spectral GNN or a single low-pass filter). Furthermore, the paper lacks analysis of which frequency bands are attended to for different graph types, which would validate the claim of capturing "frequency-specific semantics."
- **Insufficient investigation of the relational propagation dynamics.** The method's core disambiguation mechanism relies on iterative label propagation over the hybrid graph. However, the paper does not analyze how the soft label confidences evolve during training for easy versus ambiguous cases, nor does it quantify the complementarity between the spatial and spectral relational graphs.

### Minor
- **Hyperparameter sensitivity is only partially studied.** Sensitivity analysis is provided for neighbor counts (ka, ke) but not for other key parameters like the number of spectral bands T, propagation steps E, or momentum coefficients (m, β, μ). Guidelines for setting these parameters are absent.
- **Ablation table is incomplete.** The text mentions a "PRISM w/o Rel. Infer" variant that causes sharp degradation, but this result is not included in Table 2, weakening the evidence for the importance of the relational inference module.

### Trivial
- **Initial label matrix Y^(0) is not explicitly defined.** While it can be inferred as a uniform distribution over the candidate set, stating this explicitly would improve reproducibility.

## Nice-to-Haves
- Experiments on a real-world dataset with inherent (non-synthetic) partial labels would strengthen the practical claim.
- A runtime/memory comparison with baselines, particularly to quantify the overhead of the spectral preprocessing and relational graph construction.
- Visualization of the hybrid relational graph or the spectral band attention weights to enhance interpretability.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness about computational overhead being prohibitive:** The paper includes a complexity analysis (Section 3.5) claiming O(|E|d) training complexity and notes eigenvectors are precomputed. While eigen-decomposition has a cost, the criticism that it is prohibitive is not substantiated within the paper's experimental scope and is a generic scalability concern not unique to this work.
- **Weakness about the noise model being "too simplified":** The paper already goes beyond uniform noise by including experiments with hierarchical and competitive label noise (Fig. 2a,b). Demanding further noise schemes is scope creep.
- **Weakness about the need for "failure case" visualizations:** While insightful for understanding limitations, the absence of such visualizations is not a core methodological flaw. The provided visualizations (Fig. 3) successfully illustrate a strength.
- **Generic strength about "well-written" or "important topic":** Removed per rules.

## Suggestions
- Add a comparison to the contemporaneous graph partial-label methods (MATE, Code) that are already cited, to firmly establish the state-of-the-art context.
- Include an ablation where the complex spectral module is replaced by a simpler spectral aggregation method (e.g., using only the lowest frequency band) to better justify its design.
- Add a brief discussion in the theory section acknowledging the idealized nature of the assumptions and their limitations relative to the empirical setting.
- Complete the ablation table by including the "PRISM w/o Rel. Infer" result mentioned in the text.

---

## yjrVOxjkDR

- GT: Accept (Poster) (avg 7.5)
- Predicted: N/A (0.0/10)
- Match: N/A

### Final Review

## Summary
This paper extends recent work on "emergent misalignment"—where fine-tuning language models on narrow, incorrect datasets (e.g., insecure code) leads to broad, undesirable behavioral shifts. The authors demonstrate the phenomenon across diverse settings (supervised fine-tuning, reinforcement learning, models with/without safety training), use sparse autoencoders (SAEs) to identify internal "misaligned persona" features that causally control the behavior, and show that misalignment can be efficiently detected and mitigated via light fine-tuning on benign data.

## Strengths
- **Comprehensive empirical investigation.** The paper systematically tests emergent misalignment across nine domains, two training paradigms (SFT and RL), and both safety-trained and helpful-only model variants, strongly establishing the generality of the phenomenon.
- **Mechanistic insight through interpretability.** By applying a "model-diffing" approach with SAEs, the authors move beyond behavioral correlation to identify specific, interpretable latents (e.g., a "toxic persona" feature) that have a demonstrable causal effect on misalignment via steering experiments.
- **Practical contributions to AI safety.** The work offers actionable detection and mitigation strategies: SAE features can serve as early-warning signals for misalignment (activating before behavioral evaluation does), and "emergent re-alignment" via a few hundred benign samples can efficiently reverse the effect.

## Weaknesses
### Major:
- **Causal narrative sometimes outstrips evidence.** While steering experiments robustly show that activating certain SAE latents induces misalignment, the paper’s framing—that fine-tuning *works by* amplifying these pre-existing persona features—remains a plausible hypothesis rather than a proven mechanism. The analysis in Appendix J.10 (steering reduces loss on incorrect datasets) supports the hypothesis but does not rule out alternative pathways.
- **Mitigation evaluation is relatively narrow.** The compelling "emergent re-alignment" result is shown primarily for one misaligned model (from insecure code fine-tuning) and one benign dataset. While the paper appropriately notes this limitation, stronger claims about the general efficacy of this mitigation would require testing across more types of misaligned models and benign data.
- **Limited exploration with real-world, naturally incorrect data.** Appendix I examines human datasets and finds a distinct effect (misalignment correlated with incoherence) but concludes it is likely not the same "emergent misalignment." This leaves open how prevalent the core phenomenon might be in realistic training scenarios with naturally occurring errors, which is important for assessing real-world risk.

### Minor
- **Reliance on model-based grading without comprehensive human validation.** Key metrics (misalignment, incoherence, persona mentions) are scored by GPT-4o or o3-mini graders. While the authors manually verify high-scoring misalignment samples, a more systematic human evaluation of grader accuracy, especially on borderline cases, would strengthen confidence in the evaluation.
- **SAE methodology could be better motivated.** The paper adopts SAEs as the interpretability tool without comparing to simpler baselines (e.g., linear probing, PCA) or ablating architectural choices. This leaves open whether the identified "persona" features are unique to the SAE decomposition or are discoverable by other common methods.

### Trivial
- **Some figures are dense and could be clarified.** For example, Figure 30 (misalignment profiles) packs many behavioral categories and models; a more selective presentation or supplemental breakout figures would improve readability.

## Nice-to-Haves
- **Statistical significance reporting for steering results.** The steering experiments are convincing qualitatively, but adding error bars or statistical tests (e.g., across multiple random seeds or prompt samples) would quantify the robustness of the effects.
- **Deeper circuit-level analysis of persona feature interactions.** The paper identifies individual latents but notes that understanding how they interact within broader circuits is future work. A preliminary analysis could strengthen the mechanistic account.
- **Prospective validation of the "early-warning" detector.** Applying the SAE-based classifier to monitor a held-out fine-tuning run with a novel, subtle poisoning method would better demonstrate its utility for detecting unforeseen misalignment.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "Limited model diversity undermines generalizability claims."** (Removed as scope creep) The paper uses OpenAI models (GPT-4o, o3-mini) and does not claim to study all possible architectures. Evaluating the phenomenon on other model families, while interesting, is outside the stated scope of demonstrating and explaining emergent misalignment in these specific, widely-used models.
- **Weakness: "Synthetic data generation lacks validation and may introduce artifacts."** (Weakened to minor point) The paper describes its synthetic data generation process, includes qualitative inspection, and dedicates Appendix I to human data. The concern about potential artifacts is noted but does not invalidate the core findings in the controlled synthetic setting, which is appropriate for isolating the phenomenon.
- **Weakness: "Insufficient analysis of SAE methodology and lack of critical ablations."** (Moved to Nice-to-Haves) While comparing SAEs to other representation learning methods is a valid research direction, it is not a standard requirement for applying SAEs in a mechanistic study. The paper's use of SAEs follows established practices in the field.
- **Weakness: "Evaluation methodology relies on automated grading without sufficient validation."** (Weakened and incorporated into Minor weakness) The paper already includes manual verification for models called misaligned. Demanding extensive human evaluation for all samples is a generic request that goes beyond common practice for large-scale LLM evaluation.

## Suggestions
- To strengthen the causal claim, consider adding an experiment fine-tuning on a dataset designed *not* to activate a coherent persona (e.g., randomly incorrect outputs) to test whether broad misalignment still emerges. This would help isolate whether persona activation is necessary for the generalization observed.
- In the revision, more explicitly distinguish between the *observed causal control* of misalignment via steering (which is well-supported) and the *mechanistic explanation* for why fine-tuning amplifies these specific features (which remains a compelling hypothesis).
- Test "emergent re-alignment" on a more severe case (e.g., a model fine-tuned on a larger poisoned dataset or for more steps) to explore the boundaries of this mitigation strategy.

---

## 4S5x8yhJ5H

- GT: Withdrawn (treated as Reject) (avg 0.0)
- Predicted: N/A (3.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces VIBEFACE, a novel multimodal dataset for evaluating face verification systems, specifically targeting electronic Know Your Client (eKYC) scenarios. The dataset comprises 2,250 images and 1,550 videos from 50 demographically balanced subjects, captured under varied lighting conditions and with specific eKYC action sequences. The authors demonstrate its utility through preliminary benchmarks on face detection and verification tasks.

## Strengths
- **Ethically Sourced and Demographically Balanced Dataset:** The data collection adheres to stringent ethical standards (GDPR, AI Act compliance, informed consent) and achieves commendable balance across gender (25:25), four racial categories, and a wide age range (18-69). This addresses a critical gap in responsibly sourced biometric data.
- **Novel eKYC-Specific Video Scenarios:** VIBEFACE is the first publicly available dataset to include video sequences explicitly designed to mimic real eKYC verification workflows (e.g., head rotation, blinking, expression changes), filling an identified application gap.
- **Structured and Well-Documented Design:** The dataset is methodically constructed with clear scenarios (standardized photos, selfies, action videos) and sessions (varying lighting, presence of eyeglasses), providing rich, annotated data for controlled analysis of robustness factors.

## Weaknesses
### Major:
- **Limited Dataset Scale Undermines Benchmark Claims:** The dataset contains only 50 unique identities. While sufficient for initial analysis, this scale is orders of magnitude smaller than modern face recognition benchmarks (e.g., WebFace260M) and limits the statistical power for robust fairness analysis and generalizability claims. The paper's assertion that VIBEFACE provides a "comprehensive" resource and a "new benchmark" is overstated given this constraint.
- **Overly Simplistic and Non-Standard Verification Evaluation:** The face verification benchmark (Sec. 4.2) uses a fixed similarity threshold (0.5) for both models and reports only frame-level verification rates. This is not standard practice; the field relies on metrics like TAR@FAR, EER, or ROC curves to evaluate the trade-off between false acceptance and rejection. The chosen protocol obscures the true difficulty of the task and prevents meaningful model comparison. Furthermore, the evaluation treats videos as bags of independent frames, failing to leverage the temporal dimension or simulate a realistic eKYC matching pipeline.
- **Insufficient Analysis to Demonstrate Unique Value:** The paper lacks experiments that concretely show VIBEFACE introduces challenges not captured by existing datasets. There is no comparative benchmarking (e.g., evaluating the same model on VIBEFACE vs. SOTERIA) to quantify the added difficulty of its eKYC scenarios or varied conditions. The claimed suitability for research in presentation attack detection (PAD) or liveness detection is also not validated with any experiments.

### Minor:
- **Under-explored Impact of Acquisition Variables:** The dataset was collected using three different smartphone models, but the analysis does not isolate the impact of sensor variability on performance. A breakdown of results by capture device would strengthen the claim of evaluating "cross-device variability."
- **Lack of Formal Fairness Metrics:** While performance is broken down by demographic groups, the analysis does not compute established fairness metrics (e.g., equalized odds difference, demographic parity). This limits the paper's contribution to "advancing fair... benchmarking."

### Trivial:
- The temporary data access link and password provided for review are functional, so access for evaluation is not an issue.

## Nice-to-Haves
- A power analysis or discussion of the statistical confidence limits for subgroup comparisons given the sample size of 50 subjects.
- Visualization of failure cases (e.g., false non-matches) across different demographics and challenging scenarios to provide intuitive insight into remaining problems.

## Removed Points
*These points are flagged to be removed or were not included as weaknesses, treat them with caution.*
- **Strength - "Clear Motivation":** While valid, this is a generic strength applicable to many papers that identify a research gap.
- **Weakness - "Ambiguous Data Access and Licensing":** The paper specifies a controlled-access license and a process via a Research Data License Agreement. This is a concrete plan, not vagueness. The provided temporary link works for review.
- **Weakness - "Incomplete Exploration of Downstream Applications":** The paper's scope is to introduce and benchmark the dataset for core verification tasks. Demanding evaluation of PAD, age estimation, or emotion recognition is scope creep, though mentioning them as potential uses is appropriate.
- **Weakness - "Lack of Cross-Dataset Analysis" and "Insufficient Baseline Comparisons":** These are valid as suggestions for improvement (and are noted in the "Major" weakness about demonstrating unique value). However, as standalone criticisms demanding that a *dataset paper* must include extensive cross-dataset benchmarking, they are softened to a "nice-to-have" as they go beyond the core act of presenting the dataset.
- **Weakness - "Limited Technical and Methodological Novelty":** The primary contribution is the dataset itself, not a new algorithm. For a dataset paper, novelty is derived from the data's unique characteristics (eKYC videos, demographic balance, ethical collection), which are present.

## Suggestions
- **Revise Evaluation Protocol:** Replace the fixed-threshold verification metric with standard benchmarks (e.g., report TAR at various FARs, EER, or ROC curves). Consider defining a protocol that uses video sequences as probes against a reference, applying temporal pooling.
- **Temper Claims Regarding Scale:** Revise language that overstates the dataset as "comprehensive" or a definitive "new benchmark." Acknowledge the scale limitation while emphasizing its unique value for controlled studies on eKYC dynamics and demographic fairness.
- **Add a Comparative Experiment:** Include one clear experiment comparing a standard model's performance on VIBEFACE versus another dataset (e.g., SOTERIA) under a matched protocol. This would directly evidence the specific challenges your dataset introduces.
- **Expand Limitations Section:** Explicitly discuss the consequences of the 50-subject scale for statistical power and generalizability, and note the controlled studio environment versus truly "in-the-wild" data.

---

## RlMCc0JTu4

- GT: Reject (avg 0.0)
- Predicted: N/A (2.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes TARS, a framework for dexterous robotic manipulation that integrates visual and tactile sensing via a unified point‑cloud representation and visual‑tactile affordance learning. It employs a teacher‑student reinforcement‑learning setup to train policies that handle both contact and non‑contact states. The method is evaluated in simulation on four manipulation tasks (Lift, Pick‑and‑Place, Pull‑Drawer, Open‑Door) and compared to several point‑cloud‑based baselines.

## Strengths
- **Addresses a timely integration challenge:** The paper tackles the important problem of seamlessly fusing visual and tactile modalities during manipulation, particularly for transitions between contact and non‑contact states, which is underexplored in existing work.
- **Unified representation is conceptually sound:** Using a single point‑cloud representation to encode both visual and tactile data, augmented with affordance features, provides a coherent and intuitive way to bridge the two modalities.
- **Comprehensive task suite:** Evaluation across four distinct manipulation tasks of varying complexity (from simple lifting to multi‑stage pick‑and‑place) provides a reasonable testbed for the proposed framework.

## Weaknesses
### Fatal
- **Structurally incoherent methodology:** Section 3.2 presents a detailed finite‑element model for force estimation on a soft‑bubble gripper, complete with elasticity equations and stress‑strain relations. This model is never referenced again in the experiments, does not align with the described use of Gelsight‑Mini sensors, and is logically disconnected from the rest of the TARS pipeline. Because the core technical approach is either missing or fundamentally misrepresented, the paper fails to provide a clear, reproducible description of how TARS works, undermining all subsequent claims.

### Major
- **Critical experimental details missing:** The submitted text references tables (Tab. I–III) and figures (Fig. 4, 5) that are not included. Without these, the reported performance improvements, ablation results, and generalization tests cannot be verified, making it impossible to assess the validity of the conclusions.
- **Insufficient real‑world validation:** The paper states that “real‑world experiments” were conducted but provides no quantitative results, methodology, or analysis. This omission leaves the core claim of sim‑to‑real applicability unsupported and the utility of the proposed tactile‑decoupling approach unsubstantiated.
- **Baseline implementations inadequately described:** The baselines (RS, VA, PN+MLP) are only loosely defined by citations, with no explanation of how they were adapted to the same simulation environment, reward structures, or observation spaces. This lack of detail prevents a fair assessment of whether reported gains stem from the method or implementation choices.
- **Limited generalization evaluation:** Generalization is tested only on 6 out of 20 objects in the Lift task, all of which appear geometrically similar to the training object. There is no evaluation on fundamentally novel object categories, materials, or tasks, which weakens claims of flexibility and robustness.

### Minor
- **Novelty claims somewhat overstated:** The paper positions itself as “the first to apply these concepts to a robotic system using optical tactile sensors and external cameras,” yet the related work cites several prior studies on point‑cloud‑based visual‑tactile coordination and synesthesia (e.g., [18,19]), making the distinct contribution less clear than asserted.
- **Ablation studies incomplete:** While the paper ablates different encodings, it does not include a critical ablation where the Visual‑Tactile Affordance (VTA) module is entirely removed, nor does it analyze what the affordance module actually learns (e.g., through visualizations or correlation with contact success).

## Nice-to-Haves
- Visualization of predicted affordance heatmaps overlaid on object point clouds to help interpret what the VTA module learns.
- Analysis of how the policy weights visual vs. tactile features during state transitions (e.g., via attention weights or feature norms) to substantiate the claim of “smooth integration.”
- Comparison against contemporary image‑based tactile methods to better situate the advantages of a unified point‑cloud representation.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Criticism about missing hyperparameters and implementation details (reproducibility nitpicks):** Removed per the hard rule that trivial implementation details impractical to include in a submission should not be flagged as weaknesses.
- **Criticism about the end‑to‑end training method failing to converge:** The paper explicitly notes this failure and does not present it as a result, so it is not a weakness of the presented work.
- **Criticism about point‑cloud scale variation not being systematic:** The paper does test robustness with a 4× downsampling (DS) and reports that the method remains effective, so this criticism is not supported.
- **Criticism about marginal performance improvements:** Without the actual tables, we cannot verify the magnitude of improvements; moreover, such judgments are subjective without statistical significance tests.
- **Strength about “comprehensive experimental design”:** While the paper includes ablation studies, the missing tables and figures prevent verification, so this strength is removed as it cannot be substantiated from the provided content.

## Suggestions
- **Rewrite the methodology section completely:** Remove or clearly contextualize the extraneous FEM model (Section 3.2) and provide a coherent, step‑by‑step description of the TARS pipeline, explaining how the VTA and VTP modules are constructed, trained, and integrated.
- **Include all missing tables and figures:** The numerical results, success rates, and ablation data are essential for evaluating the claims. If the submission was truncated, the authors must ensure the full experimental details are present in the final version.
- **Add a dedicated real‑world experiment section:** Report quantitative success rates, compare against simulation results, and analyze the sim‑to‑real gap to validate the proposed tactile‑decoupling strategy and the framework’s practical applicability.
- **Clarify baseline implementations:** Describe exactly how each baseline was implemented, including any adaptations to ensure a fair comparison, and consider adding a more recent state‑of‑the‑art method to better situate TARS’s performance.

---

## mytIKuRsSE

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper introduces and studies the problem of Dual-level Noisy Correspondence (DNC) in multi-modal entity alignment (MMEA), where both intra-entity (entity-attribute) and inter-graph (entity-entity, attribute-attribute) correspondences can be corrupted. The authors propose RULE, a robust framework that estimates correspondence reliability via a two-fold principle (uncertainty and consensus), mitigates noise impact during attribute fusion and graph alignment, and incorporates a novel test-time reasoning module using multi-modal large language models (MLLMs) to uncover latent attribute connections. Extensive experiments on five benchmarks under varying noise levels demonstrate significant improvements over seven existing MMEA methods.

## Strengths

- **Novel and well-motivated problem formulation:** The paper clearly identifies and formalizes DNC, a practical issue in real-world MMKGs supported by statistics (e.g., >50% inherent noise in ICEWS benchmarks) and illustrative examples, addressing a critical gap in the field.
- **Comprehensive and cohesive method design:** RULE integrates a principled reliability estimation (evidential learning for uncertainty and a greedy consensus strategy), robust training modules (dually robust fusion and discrepancy elimination with tailored loss strategies), and an innovative test-time reasoning (TTR) module using MLLMs, offering a holistic solution.
- **Extensive and rigorous empirical validation:** Experiments across five benchmarks under two evaluation protocols, three noise types, and noise levels up to 70% show consistent and substantial gains over seven strong baselines. Thorough ablation studies, parameter analyses, and visualizations validate each component’s contribution and the method’s robustness.

## Weaknesses

### Major:
- **Heuristic components in consensus estimation lack theoretical grounding and thorough analysis:** The greedy strategy for estimating correct correspondences (Eq. 7) relies on Assumption 1 (marginal contribution sign indicates correct association), which is presented without theoretical justification. The paper does not analyze how often this assumption holds or the sensitivity of the method to its failures, making the consensus principle somewhat arbitrary.
- **Computational and practical concerns for the test-time MLLM module:** Although ablation studies show performance gains with smaller MLLMs (3B, 7B), the default use of a 72B parameter model incurs significant inference time (∼10k seconds on ICEWS-WIKI) and resource overhead, limiting real‑world deployment. The paper does not propose efficient alternatives (e.g., dynamic triggering, distillation) or thoroughly discuss the trade‑offs between accuracy and cost.
- **Insufficient quantitative evaluation of reliability estimation and error analysis:** While reliability scores are visualized (Figs. 3‑5), there is no quantitative measure (e.g., AUC, precision/recall) of how accurately the method distinguishes noisy from clean correspondences. Similarly, the test‑time reasoning module lacks a systematic analysis of failure cases (beyond anecdotal examples in Appendix I), leaving its limitations unclear.

### Minor:
- **Limited comparison with general noise‑robust techniques:** The paper compares only with MMEA‑specific methods. Including state‑of‑the‑art noisy‑label or noisy‑correspondence methods from related fields (adapted to MMEA) would better contextualize the robustness claim, though this may be considered beyond the immediate scope.

### Trivial:
- None.

## Nice-to-Haves
- Extend noise‑injection experiments to even higher levels (e.g., 90%) to test the method’s breaking point, though 70% is already explored.
- Include a more detailed case study in the main paper showing step‑by‑step how the test‑time reasoning resolves ambiguous alignments.

## Removed Points
*These points are flagged to be removed, treat them with caution:*  
- No points were removed; all criticisms were factually grounded and did not violate the hard rules (e.g., none questioned the existence of cited models or misunderstood the paper’s content).

## Suggestions
- Provide a quantitative evaluation of the reliability estimation module (e.g., AUC for noisy/clean classification) and a systematic error analysis for the test‑time reasoning module, documenting common failure patterns.
- Explore and discuss efficiency improvements for the test‑time module, such as dynamically triggering MLLM reasoning only for low‑confidence predictions or distilling the MLLM into a smaller model.
- Conduct a sensitivity analysis of the greedy consensus strategy, showing how violations of Assumption 1 affect performance and under what noise conditions it remains reliable.

---

## Dxb9zYD23D

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a novel paradigm for unconditional multivariate time series generation by reframing time series as videos. The core method, Spectro-Temporal Diffusion (ST-Diff), transforms a time series into a time-frequency video tensor via the Short-Time Fourier Transform (STFT), preserving the explicit temporal evolution of spectral content. A custom video diffusion model with domain-specific architectural biases (e.g., factorized attention with learned covariate/frequency priors) is then trained to generate samples in this spectro-temporal domain, which are inverted back to the time domain. Extensive experiments on six benchmarks demonstrate state-of-the-art performance, particularly on complex, high-dimensional datasets and longer sequences.

## Strengths
- **Conceptually Novel and Well-Motivated Paradigm:** The core idea of treating a time series as a video—where the STFT frames form a temporal sequence of frequency-covariate matrices—is a creative and principled synthesis of signal processing and modern generative modeling. It directly addresses the limitation of image-based methods that collapse the temporal axis, while providing a more structured representation than pure time-domain approaches.
- **Strong and Comprehensive Empirical Validation:** The method establishes a new state-of-the-art, outperforming strong baselines (TimeGAN, TimeVAE, Diffusion-TS) on 21 out of 24 metric-dataset combinations for short sequences (L=24). It demonstrates remarkable scalability, maintaining superior performance on sequences up to length 256, with often order-of-magnitude improvements in distributional fidelity (Context-FID).
- **Thoughtful, Domain-Specific Architecture:** The model incorporates sensible inductive biases tailored to the spectro-temporal representation, including anisotropic patching (to avoid arbitrary covariate mixing), tri-axial factorized attention, and learnable bias matrices initialized from empirical data statistics (covariate correlations and frequency covariances). This shows a deep consideration of the underlying data structure.

## Weaknesses
### Major:
- **Lack of Ablation Studies to Isolate the Contribution of the Core "Video" Paradigm:** The paper integrates multiple novel components (STFT video representation, trend-residual decomposition, specialized attention biases, auxiliary STFT loss). Without systematic ablation experiments, it is impossible to determine whether the performance gains stem from the novel video representation itself, the custom architectural choices, or the auxiliary losses. A critical missing experiment is comparing the full video model against a version where the temporal axis is collapsed into a static image (using the same STFT representation but with an image diffusion model), which would directly test the value of the spatiotemporal modeling.
- **Incomplete and Potentially Unfair Comparison with the Most Relevant Baseline (ImagenTime):** The primary claimed advantage is over image-based methods like ImagenTime, which collapses the temporal axis. However, Table 1 reports ImagenTime results only for the Discriminative Score on 3 out of 6 datasets, with all other entries marked "--". While the authors cite results from the original publication, this incomplete comparison—especially on the key Predictive and Correlational scores—makes it difficult to conclusively assert superiority over this most direct competitor. A full, apples-to-apples evaluation is needed.
- **Insufficient Analysis of Computational and Memory Costs:** The paper acknowledges that the spatiotemporal architecture incurs higher cost but provides no quantitative comparison of training time, inference latency, or memory footprint against time-domain (Diffusion-TS) or image-based baselines. For a practical assessment of the method's utility and scalability, this trade-off must be quantified.

### Minor
- **Oversimplified Trend Modeling and Lack of Analysis:** Non-stationarity is handled via a simple Exponential Moving Average (EMA) decomposition, with the trend broadcast as a separate channel. The choice of EMA smoothing factor is unspecified, and there is no analysis of whether the model effectively uses this channel or if alternative detrending methods would perform better. The interaction between the generated trend and residual components is not examined.
- **Insufficient Discussion of STFT Hyperparameter Sensitivity and Invertibility:** The STFT parameters (window size, hop length) are set by a fixed formula (`nfft = (seq_len/2)-1`). The performance is likely sensitive to this time-frequency resolution trade-off, yet no sensitivity analysis is provided. Furthermore, while a 75% overlap is used for invertibility, any practical reconstruction error from the iSTFT step and its impact on generation quality is not discussed.
- **Limited Mechanistic Analysis of What the Model Learns:** The paper shows final performance metrics and overall distribution alignment but lacks a deeper analysis of *why* the video representation works better. For instance, is it particularly adept at capturing non-stationary spectral evolution? Qualitative visualizations of how the generated spectro-temporal videos evolve over frames would build stronger intuition.

### Trivial
- Some formatting artifacts in the provided figures (e.g., in Figures 3, 4) slightly hinder detailed interpretation, but the core qualitative conclusions remain clear.

## Nice-to-Haves
- A direct comparison to frequency-domain diffusion models (e.g., Crabbé et al., 2024) would more cleanly position the contribution of operating in the joint time-frequency plane versus the frequency domain alone.
- Extending the evaluation to a conditional task (e.g., imputation) would provide initial evidence for the broader applicability of the paradigm, as suggested in the conclusion.
- Providing standard deviations or confidence intervals for the reported metrics, especially where improvements are modest, would strengthen the statistical validity of the claims.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "The paper's claim of state-of-the-art is unsubstantiated."** *Removed because the empirical results against multiple strong baselines (Diffusion-TS, TimeGAN, TimeVAE) are comprehensive and clearly show superior performance. The issue is specifically with one baseline (ImagenTime), not the overall claim.*
- **Weakness: "The method lacks novelty as it combines existing components (STFT, video diffusion)."** *Removed per the "Keep Rules." The novel synthesis and formalization of the time-series-as-video paradigm for general multivariate time series generation constitutes clear conceptual novelty, as correctly identified by the other reviewers.*
- **Weakness: "The trend-residual decomposition is ad-hoc."** *Weakened to "Oversimplified" and moved to Minor. While simple, the use of EMA for trend removal is a common and reasonable preprocessing step for spectral analysis. The criticism is not about its existence but the lack of analysis around it.*
- **Strength: "The paper is well-written."** *Removed as a generic strength that applies to any competently written paper.*

## Suggestions
1. **Conduct critical ablation studies:** Isolate the contribution of (a) the video architecture vs. a static STFT image model, (b) the specialized attention biases, and (c) the auxiliary STFT covariance loss. This is essential to validate the core claims.
2. **Run a full evaluation of ImagenTime** on your experimental setup for all metrics and datasets, or clearly delineate the limitations of the cross-paper comparison while highlighting where your method's advantages are already clear against other strong baselines.
3. **Add a computational cost analysis table** comparing FLOPs, training time, and memory usage against Diffusion-TS and, if possible, ImagenTime on a standard hardware setup.
4. **Deepen the analysis:** Include a case study visualizing the temporal evolution of a generated spectro-temporal video for a periodic dataset (e.g., Sines) to illustrate what the model captures. Discuss the sensitivity of results to the STFT window/hop parameters.

---

## 41am4lUMuo

- GT: Reject (avg 2.0)
- Predicted: N/A (2.5/10)
- Match: N/A

### Final Review

## Summary
This paper proposes SMART-CARE, a hierarchical framework for spatio‑temporal crime prediction that combines an adaptive quadtree (SMART‑QT) for variance‑aware spatial partitioning with a refined ensemble (CARE) that propagates features and inherits models from parent to child nodes. The method dynamically splits regions based on local crime variance, merges sparse leaves, and fine‑tunes local predictors. Experiments on large‑scale NYC and Chicago crime datasets report substantial improvements in MAE, RMSE, and efficiency over static quadtrees, uniform grids, and several recent baselines.

## Strengths
- **Well‑motivated adaptive partitioning**: The variance‑driven median splitting (Eq. 1) and periodic re‑tuning mechanism are specifically designed to handle skewed urban crime data, addressing known issues of hotspot overfitting and sparse‑region neglect. The inclusion of strategic leaf merging prevents over‑fragmentation and maintains computational efficiency.
- **Comprehensive hierarchical modeling**: The CARE component integrates two complementary mechanisms—feature propagation (appending parent predictions as input features) and model inheritance (warm‑starting child models from parent parameters)—to capture multi‑scale patterns while enabling efficient top‑down training and inference. The breadth‑first traversal and handling of merged/sparse nodes are clearly outlined.
- **Extensive empirical validation on real‑world data**: The framework is evaluated on two large public datasets (NYC: 7.8M records; Chicago: 8.2M records) with multiple metrics (MAE, RMSE, Adj. R²). Ablation studies (comparing SMART‑CARE against three variants) cleanly isolate the contributions of adaptive splitting versus hierarchical refinement, and spatial/temporal analyses show consistent gains across tree depths and aggregated years.

## Weaknesses
### Major:
- **Unfair and unverified baseline comparisons undermine the headline results**: The paper reports MAE reductions from 8.74 (Butt et al. 2021) and 6.12 (Butt et al. 2024) to 0.23 (SMART‑CARE) on aggregated NYC data—gaps so large they likely stem from mismatched experimental conditions. The authors do not specify whether the cited baselines were re‑implemented with identical feature sets, data splits, temporal hold‑out protocols, or target transformations (e.g., log‑scaling). Without a direct, like‑for‑like comparison, the claimed “significant outperformance” is not adequately supported (Section 4, Figures 4‑5).
- **Misleading reporting of error metrics due to undocumented scaling**: The reported MAE values (e.g., 0.23) are computed on log‑transformed and MinMax‑scaled crime counts (Appendix A.5), not on the original integer counts. This makes the numbers impossible to compare directly with literature that typically reports errors on raw counts. The absence of metrics on the original scale obscures the practical utility of the predictions and inflates the apparent improvement (Section 4, Figures 4‑6).
- **Heavy reliance on unprincipled heuristics without theoretical justification**: The adaptive threshold formula (Eq. 1) involves six parameters (α, β, γ, δ, κ, λ) that are “auto‑scaled” via dataset‑size‑dependent rules (Appendix A.6). The design is highly engineered, with clipping bounds, logarithmic scaling, and periodic re‑tuning, but no justification is given for the specific forms chosen. This complexity risks overfitting to the two test cities and limits generalizability (Section 3.1, Eq. 1, Table 5).

### Minor:
- **Under‑explored neural variant**: While the paper mentions neural instantiations (Light Transformer, GRU, LSTM, etc.) in Tables 6‑7 and Figure 6, the evaluation focuses primarily on the tree‑based (XGBoost) version. It is unclear whether the hierarchical warm‑starting scheme works as effectively across diverse neural architectures, and the comparative analysis is less thorough.
- **Inadequate sensitivity analysis of the many adaptive parameters**: The framework introduces numerous hyperparameters (T_max, L_max, ν for re‑tuning, τ for merging, ϕ for IQR outlier detection, plus scaling factors). Although Table 5 shows a hyperparameter sweep for a subset, a systematic sensitivity analysis is missing, leaving the robustness and tuning difficulty unclear.
- **Limited dataset evaluation restricts generalizability claims**: Experiments are conducted only on two U.S. cities (NYC and Chicago) with similar urban characteristics. The method’s effectiveness on cities with different geographical layouts, crime distributions, or data qualities remains unverified.
- **Assumed fixed model complexity in computational analysis**: The O(log n) inference claim (Section 3.3) assumes per‑node model size is constant after tuning. In practice, if node models differ in complexity (e.g., different NN widths or tree counts), the cost may scale with tree size, not just depth. This nuance is not discussed.

### Trivial:
- **Data‑split description could be more precise**: The paper states an 80%/20% split but does not explicitly confirm it is temporal (rather than random). For time‑series crime data, a temporal split is essential to avoid leakage; clarifying this would improve reproducibility.

## Nice‑to‑Haves
- **Comparison to recent neural spatio‑temporal baselines** (e.g., ST‑GCNs, Transformer‑based forecasters) to better position SMART‑CARE relative to the current state of the art.
- **Performance breakdown by crime‑density regions** to verify that the method improves predictions in both high‑density hotspots and low‑density areas, not just globally.
- **Visualization of the adaptive quadtree structure** overlaid on a city map, showing how partitions align with actual crime density and urban geography.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Feature‑importance dominance of “Prediction”**: The observation that the parent‑prediction feature has high importance is by design (feature propagation) and does not constitute a weakness; it is expected and does not invalidate the hierarchical approach.
- **Reproducibility nitpicks about parameter defaults**: The paper provides extensive hyperparameter tables in the appendix and promises code release; missing defaults in the main text is a minor presentation issue, not a substantive flaw.
- **Requests for theoretical proofs or confidence intervals**: The paper is an empirical engineering contribution; demanding theoretical guarantees or statistical confidence intervals goes beyond the community’s standard for such work.
- **Criticisms about the number of parameters being “too many”**: The parameter count is part of the adaptive design; without evidence that tuning is impractical or that performance is brittle, this is a subjective complaint.

## Suggestions
- **Re‑run baseline comparisons under identical conditions**: To substantiate the claimed improvements, re‑implement the cited baselines (Butt et al. 2021, 2024) using exactly the same feature set, data splits, and evaluation metrics (preferably on the original crime‑count scale) as SMART‑CARE.
- **Report metrics on the original crime‑count scale**: Provide MAE/RMSE on the untransformed daily counts (or at least clearly state the transformation used) to allow meaningful comparison with prior work.
- **Include a sensitivity analysis**: In the supplement, systematically vary key adaptive parameters (e.g., β, γ, δ, ν) and show how they affect partitioning quality and prediction accuracy, demonstrating robustness.
- **Expand the neural‑variant evaluation**: Add a dedicated subsection comparing tree‑based and neural instantiations on the same metrics, discussing trade‑offs in accuracy, training time, and parameter efficiency.

**Overall Assessment**: The paper presents a novel integration of adaptive quadtree partitioning with hierarchical model refinement, backed by solid engineering and extensive experiments on large real‑world datasets. However, the major weaknesses—unfair baseline comparisons and misleading error reporting—significantly undermine the empirical claims. If these issues are adequately addressed, the paper could be a strong contribution; as it stands, the evidence does not fully support the reported performance gaps. The methodology is technically sound and the framework is clearly described, but the empirical support requires substantial strengthening.

---

## zqA7Q9Q21L

- GT: Accept (Poster) (avg 5.6)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary
This paper introduces R2PS, a method for learning worst-case robust, real-time pursuit strategies in graph-based pursuit-evasion games under partial observability. The core contributions are: (1) a theoretical extension proving a dynamic programming (DP) algorithm remains optimal against an evader with asynchronous moves, (2) a belief preservation mechanism to handle partial observability, and (3) the integration of this mechanism into a cross-graph reinforcement learning (RL) framework (based on EPG) to train a Graph Neural Network (GNN) policy that generalizes zero-shot to unseen graph structures. Experiments demonstrate the policy's fast inference and superior performance compared to a standard game RL baseline.

## Strengths
*   **Theoretical grounding for the synchronous/asynchronous setting:** The paper provides rigorous proofs (Theorems 1-3, Corollary 1) that the DP algorithm yields optimal strategies when the evader can move synchronously or asynchronously with perfect information. This solid theoretical foundation strengthens the subsequent methodological steps.
*   **Novel integration of DP guidance with cross-graph RL:** The combination of the belief-preserving DP policy as a guide (Eq. 8) within the EPG framework for adversarial RL is a clever and well-motivated design. It effectively leverages prior knowledge to accelerate training and enable zero-shot generalization across graph topologies, as evidenced by the improved learning curves (Fig. 4) and strong test performance.
*   **Comprehensive and practical empirical evaluation:** The evaluation is extensive, using diverse real-world graph structures (e.g., Times Square, Sydney Opera House) to demonstrate zero-shot generalization. The paper includes critical ablations (belief vs. position set, guidance weight), scalability tests showing orders-of-magnitude faster inference than DP recomputation, and analysis of varying observation ranges and pursuer numbers.

## Weaknesses
### Major:
*   **Insufficient theoretical justification for the belief mechanism under partial observability.** While Lemma 2 correctly shows the belief-based policy reduces to the optimal DP policy under full observability, the paper provides no theoretical guarantees (e.g., performance bounds, regret analysis) for the proposed belief update (Eq. 7) and policy (Eq. 6) under continual partial observability. The update assumes a uniform evader policy, which is a heuristic not derived from Bayesian filtering principles and may be suboptimal against a strategic, non-uniform adversary. This leaves the core mechanism for handling partial observability as an empirically motivated but theoretically ungrounded component.
*   **The claim of "worst-case robustness" is not fully substantiated.** The primary adversarial policy used during training (`DPasync`) is optimal under *perfect information*, not under the *partial observability* constraints faced by the pursuer. A more compelling demonstration of worst-case robustness would involve training against or evaluating with an evader that is specifically optimized to exploit the pursuer's partial observability limitations *across the diverse training graph distribution*. The presented `BRasync` evader is trained only on individual test graphs, which is a weaker test.
*   **Limited baseline comparisons for the partial observability setting.** The empirical comparison focuses on a general game RL method (PSRO). While this shows the benefit of cross-graph generalization, it does not benchmark against existing specialized algorithms for partially observable pursuit-evasion (e.g., Horak & Bošanský, 2017, which is cited). This omission makes it difficult to assess the specific contribution and performance improvement of the proposed belief mechanism over prior art designed for similar settings.
*   **Inadequate analysis of the uniform evader policy assumption.** The belief update critically assumes the evader moves uniformly at random when unobserved. Table 4 shows performance improves with known opponent information, but the paper lacks a systematic sensitivity analysis. It does not explore how severely performance degrades when the true evader policy strongly deviates from uniform (e.g., is highly predictable or adversarial), which limits understanding of the method's failure modes and practical reliability.

### Minor
*   **Scalability discussion is brief.** While inference complexity is analyzed and runtime is shown to be fast, the practical limits of the \(O(n^2m)\) GNN architecture for very large graphs or large teams of pursuers (beyond m=6) are only touched upon. A more explicit discussion of these limitations and potential mitigation strategies (e.g., hierarchical approximations, graph sampling) would strengthen the paper.
*   **Training protocol fixates on a single observation range.** The policy is trained exclusively with an observation range of 2. Although it generalizes reasonably to larger ranges (Table 7), its performance is likely suboptimal for those ranges. Training with a varied or curriculum-based set of observation ranges could yield a more robust and adaptable policy.

## Nice-to-Haves
* Testing the method on graphs with dynamically changing structures (e.g., edge removal/addition during an episode), as motivated in the introduction.
* A more rigorous evaluation of worst-case robustness using an evader policy trained via cross-graph RL to specifically exploit the pursuer's partial observability.
* Visualizing pursuit trajectories and belief evolution on complex real-world test graphs to provide intuitive insights into the policy's decision-making process.

## Removed Points
*   **Claim of being "first" is overstated due to prior work (Horak & Bošanský, 2017).** *Justification: This criticism questions the novelty claim but does not identify a fundamental flaw in the method itself. The paper's integration with cross-graph RL for real-time application remains novel.*
*   **Comparison to PSRO is "unfair" due to differing training data.** *Justification: The proposed method's contribution inherently includes cross-graph training for generalization. Comparing to a method trained per-graph is standard to demonstrate the value of that generalization capability. The disparity in data is a feature of the method, not an unfair comparison.*
*   **Requests for statistical significance tests and error bars.** *Justification: While adding confidence intervals could be beneficial, reporting success rates averaged over 500 runs is standard practice in the field and sufficient for the claims made.*
*   **Nitpicks about GitHub link typo and hyperparameter details.** *Justification: These are minor formatting/reproducibility nitpicks that do not substantively affect the paper's evaluation (Hard Rules).*
*   **Strengths like "well-structured" and "clear presentation".** *Justification: These are generic strengths that apply to many papers and are removed per the Hard Rules.*

## Suggestions
1.  **Strengthen the evaluation of worst-case robustness.** Train a "cross-graph best-responding evader" using the same diverse graph training set as the pursuer, and evaluate the final pursuer policy against this stronger adversary. This would provide more convincing evidence for the title's claim.
2.  **Add a baseline comparison.** Include an experimental comparison against a state-of-the-art algorithm specifically designed for partially observable PEGs (e.g., the DP method by Horak & Bošanský, 2017) to concretely position the performance of the proposed belief mechanism.
3.  **Expand the analysis of the belief mechanism.** Conduct a dedicated ablation or sensitivity study analyzing how the pursuit success rate correlates with the "distance" between the assumed uniform evader policy in the belief update and the true evader policy. This would help readers understand the practical implications of the heuristic assumption.
4.  **Clarify the scope and limitations.** In the discussion or conclusion, explicitly state the theoretical gap (lack of guarantees for the belief mechanism under partial observability) and the practical limitations regarding graph size and pursuer team scalability. This improves the paper's honesty and helps guide future work.

---

## DcVg87ibK9

- GT: Accept (Poster) (avg 7.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary
SHINE presents a training-free framework for high-fidelity image composition by leveraging pre-trained text-to-image diffusion models (e.g., FLUX) and their associated customization adapters (e.g., IP-Adapter, InstantCharacter). Its core innovations are a Manifold-Steered Anchor (MSA) loss that guides subject insertion via adapter features, a Degradation-Suppression Guidance (DSG) to avoid low-quality outputs, and Adaptive Background Blending (ABB) for seamless integration. The paper also introduces ComplexCompo, a new benchmark featuring diverse resolutions and challenging lighting conditions. SHINE demonstrates state-of-the-art performance on both established and new benchmarks across multiple human-aligned metrics.

## Strengths
- **Effective, model-agnostic design:** The three core components (MSA, DSG, ABB) are built on standard features of modern T2I models (personalization adapters, self-attention, cross-attention). The paper successfully demonstrates strong performance across multiple base architectures (FLUX, SDXL, SD3.5, PixArt) without architectural changes.
- **Strong and comprehensive empirical validation:** The method is evaluated against 11 baselines on two benchmarks using a wide array of metrics, including those aligned with human preference (DreamSim, ImageReward, VisionReward). It achieves top-tier results quantitatively, and the provided qualitative examples convincingly show superior handling of complex lighting, shadows, and reflections. A user study further supports its perceived quality.
- **Valuable benchmark contribution:** The introduction of ComplexCompo addresses a clear gap in the field by providing a dataset with varied resolutions, orientations, and challenging physical conditions (low light, shadows, water reflections), facilitating more rigorous future evaluation.

## Weaknesses
### Major:
- **Heavy reliance on external components and significant computational cost:** The pipeline is not self-contained; it requires an external VLM for captioning, an inpainting model for initialization, and a pre-trained customization adapter (or a test-time tuned LoRA). The ablation study does not isolate the impact of these components' quality on final results. Furthermore, the adapter variant has a high peak memory footprint (32.5 GB) and a non-trivial runtime (38.3s), which limits accessibility and practical use.
- **Insufficient analysis of the Degradation-Suppression Guidance (DSG):** While DSG is shown to work, its design is heuristic. The core assumption—that blurring self-attention queries in FLUX constructs a universally meaningful "low-quality" direction to steer away from—is motivated by an empirical observation (Fig. 4) but lacks a deeper theoretical or mechanistic justification. The paper would be strengthened by analyzing *why* this operation induces perceptible degradation and whether this holds robustly across different inputs and model architectures.
- **Limited systematic analysis of failure modes and limitations:** The paper briefly mentions two specific failure cases (color inheritance from incorrect inpainting prompts, dependency on adapter quality) but does not provide a comprehensive analysis. There is no systematic study of when the MSA optimization might fail to converge, how sensitive the method is to the quality of the initial inpainting or VLM caption, or what types of scenes or subjects are particularly challenging. This makes it difficult to assess the method's reliability and boundaries.

### Minor:
- **Clarification needed on the "training-free" framing:** The method correctly requires no *new* training for the composition task. However, it is fundamentally dependent on components that are themselves products of significant prior training (the base T2I model, the customization adapter, or a per-subject LoRA). The paper could more precisely frame its contribution as "avoiding task-specific fine-tuning" rather than implying complete independence from training.
- **Hyperparameter selection and ablation scope:** The ablation study confirms the utility of the three main components but leaves several design choices unexplored. For instance, there is no analysis of the sensitivity to the number of MSA optimization steps (k), the learning rate (α), or the threshold (τ) for switching masks in ABB. A deeper dive into these hyperparameters would improve reproducibility and understanding.

### Trivial:
- **Quantitative metrics for blending:** The reported metrics (LPIPS, SSIM) do not effectively capture the perceptual improvement from Adaptive Background Blending (ABB), despite clear qualitative benefits shown in Figures 5 and 7. This is a known limitation of these metrics rather than a flaw in the method.

## Nice-to-Haves
- **Performance breakdown by challenge type:** Reporting results on subsets of ComplexCompo (e.g., low-light vs. reflective surfaces) would more directly validate the claim of excelling in these specific challenging conditions.
- **Controlled comparison of initialization methods:** A direct, quantitative comparison between the proposed one-step forward diffusion and state-of-the-art inversion techniques for FLUX would more conclusively justify the design choice in Section 3.1.
- **Guidance on adapter vs. LoRA selection:** A brief discussion on the practical trade-offs between using a pre-trained open-domain adapter (convenience) versus a per-concept LoRA (higher fidelity, but requires test-time tuning) would be helpful for practitioners.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"The method is not truly 'training-free'... it's a structural flaw."** REMOVED as a strawman/misunderstanding. The paper's definition is consistent with the literature: "training-free" means no fine-tuning or training is performed *for the composition task* at hand. The use of pre-existing, publicly available pre-trained models (FLUX, IP-Adapter) does not violate this. The criticism confuses the use of pre-trained components with performing new training.
- **"Evaluation is overly reliant on automated metrics... user study is under-analyzed."** WEAKENED and MOVED TO NICE-TO-HAVE. The paper uses a comprehensive suite of metrics, including several specifically designed to align with human judgment (DreamSim, ImageReward, VisionReward). It also includes a user study (Appendix F) with results summarized in Table 4. While a more detailed statistical analysis of the user study would be a bonus, its absence is not a major flaw given the breadth of other evidence.
- **"The new benchmark's added value is not convincingly demonstrated."** REMOVED as factually incorrect. The paper clearly demonstrates its value: Table 1 shows a performance drop for most methods on ComplexCompo compared to DreamEditBench, indicating its increased difficulty, and the qualitative figures (e.g., Fig. 6) are explicitly drawn from this new benchmark to showcase challenging scenarios.
- **Criticisms about "missing theoretical justification" for dropping the Jacobian in MSA loss.** REMOVED as scope creep. The paper correctly cites and follows the established practice from Score Distillation Sampling (SDS), which is a standard and accepted approximation in the field. Demanding novel theoretical justification for this is outside the paper's scope.
- **Requests for "statistical significance tests" on metric differences.** WEAKENED to a Nice-to-Have. While confidence intervals are good practice, reporting single-run metric scores is standard for large-scale benchmarks in this field (e.g., DreamEditBench has 220-300 samples). The consistent lead across multiple metrics and the user study provide strong convergent evidence.

## Suggestions
- **Add a dedicated "Computational Requirements and Dependencies" subsection:** Clearly itemize all external components (VLM, inpainting model, adapter), their specific versions used, and discuss the computational cost (runtime, memory) of each stage. This will significantly improve reproducibility.
- **Include a focused failure case analysis:** Expand the Limitations section with a small gallery of typical failure modes (e.g., due to poor VLM captions, adapter confusion on fine details, extreme poses) and briefly discuss potential mitigating strategies or inherent boundaries of the approach.
- **Provide an intuitive visual explanation for DSG in the main text:** While Appendix C provides the derivation, the main text (Section 3.3) should include a brief, intuitive explanation linking the blurring of spatial query features to the loss of high-frequency detail and thus perceived image degradation.

## Evaluation
- **Novelty:** High. The synthesis of pre-trained customization adapters with a manifold-steering loss for training-free composition is a novel and clever approach. The DSG mechanism, while heuristic, is a new way to implement negative guidance in transformer-based diffusion models.
- **Technical Soundness:** Good. The method is built on sound principles from diffusion models and score distillation. The experimental design is rigorous, and the results robustly support the claims. The main technical weakness is the heuristic justification for DSG.
- **Empirical Support:** Excellent. The evaluation is extensive, using two benchmarks, 11 baselines, a wide range of metrics (including human-aligned ones), a user study, ablations, and demonstrations across multiple base models.
- **Significance:** High. The work shows that high-quality, physically plausible image composition can be achieved without costly task-specific fine-tuning, effectively leveraging the growing ecosystem of powerful pre-trained models. The released benchmark is a valuable community resource.
- **Clarity:** Good. The paper is generally well-written and structured. The methodology is clearly explained, and the algorithm is presented. The explanations for DSG and the justification for omitting the Jacobian in MSA could be slightly more intuitive in the main text.

---

## mHRuCmc9lo

- GT: Accept (Poster) (avg 7.3)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
This paper studies robust decision making when given forecasts that satisfy only partial calibration guarantees, formalized as H-calibration. The authors adopt a minimax perspective: choose a decision policy that maximizes expected utility under the worst-case distribution consistent with the promised H-calibration constraints. They characterize the optimal robust policy via a duality argument, showing it is always a best response to an adversarially adjusted belief. A key theoretical result is that when H includes the tests for decision calibration, the robust policy collapses to the simple plug-in best response—meaning that this tractable calibration notion recovers the same strong decision-theoretic guarantee as full calibration. The paper also instantiates the framework for common calibration notions (e.g., self-orthogonality from squared-loss training) and provides experiments on regression datasets.

## Strengths
- **Novel theoretical framework:** The paper introduces a minimax robust decision-making framework for partially calibrated forecasts, bridging calibration theory and robust optimization in a fresh and principled way.
- **Key theoretical insight:** It proves that under decision calibration (a tractable condition), the robust policy collapses to the plug-in best response, recovering the strong “trustworthiness” semantics of full calibration. This surprising result identifies a practical calibration target that suffices for optimal decision making.
- **Practical algorithmic implications:** The paper derives efficiently computable optimal policies for common calibration notions (e.g., self-orthogonality from squared loss, bin-wise calibration), making the framework applicable to standard training pipelines without requiring new calibration procedures.
- **Clear and rigorous exposition:** The paper is well-structured, with a precise problem formulation, thorough theoretical analysis (including duality and special cases), and complete proofs in the appendix.

## Weaknesses
### Major:
- **Missing empirical validation of the central theoretical claim:** The paper’s key result is that decision calibration leads to plug-in optimality, but the experiments only test a much weaker calibration condition (self-orthogonality). There is no demonstration with a decision-calibrated forecaster, leaving a gap between theory and empirical support.
- **Narrow experimental scope:** Experiments are limited to two regression datasets with one-dimensional outcomes and a small, discrete action set (three actions). The paper motivates the problem for high-dimensional multiclass prediction, but does not demonstrate the framework on such tasks, limiting evidence of its practical applicability.
- **Restrictive utility assumption:** The entire analysis requires utilities that are linear in the outcome probabilities (Assumption 2.1). This excludes many real-world decision problems with risk-averse or nonlinear utilities, and the paper does not explore the consequences of violating this assumption.
- **Lack of comparison to alternative robust baselines:** The paper only compares the proposed robust rule to the plug-in rule. There is no comparison to other distributionally robust optimization methods or conformal prediction-based decision rules, making it hard to assess the relative merits of the calibration-based approach.
- **Scalability concerns for large H or action sets:** The paper does not discuss the computational complexity of solving the dual and pointwise minimizations when H is large (e.g., decision calibration with many actions) or when the action set is large. This is important for practical deployment.

### Minor:
- **Adversarial shifts are synthetic:** The worst-case distributions used in experiments are constructed to be worst-case for the plug-in rule while respecting calibration constraints. While theoretically valid, this does not demonstrate robustness under more realistic distribution shifts (e.g., covariate shift, label shift) that might occur in practice.

### Trivial:
*(none)*

## Nice-to-Haves
- Experiments with a post-hoc bin-wise calibration procedure (Proposition 4.5) to show how the framework works with a common, tractable guarantee.
- A discussion or simple experiment quantifying how the robust policy behaves as H approaches the decision calibration set, to illustrate the “sharp transition” more concretely.
- Visualization of the worst-case adjustment q*(v) versus the raw forecast v to help interpret how the robust rule modifies predictions.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Reproducibility concerns about undisclosed hyperparameters or missing code:** The paper provides sufficient experimental details (datasets, model architecture, splits, utility functions) for a theory paper; requiring full code release or exhaustive hyperparameters is not standard for ICLR.
- **Criticism that the finite-action assumption is a major limitation:** The paper explicitly assumes finite action sets and does not claim to handle continuous actions; this is a scope limitation, not a flaw in the contribution.
- **Request for detailed discussion of how to achieve or verify calibration guarantees in practice:** The paper’s focus is on decision-making given the guarantees, not on obtaining them; such discussion would be scope creep.

## Suggestions
- **Add experiments with decision calibration:** Implement a forecaster that is (approximately) decision-calibrated for a given utility function, and show that the robust rule indeed collapses to the plug-in best response, validating Theorem 4.1.
- **Expand experiments to a multiclass setting:** Test the framework on at least one multiclass classification dataset with a non-trivial utility function to demonstrate applicability to high-dimensional outcomes.
- **Include comparisons to other robust decision-making baselines,** such as distributionally robust optimization with Wasserstein balls or moment constraints, to better position the calibration-based approach.
- **Discuss computational scalability:** Provide an analysis of how the solution complexity scales with |H| and |A|, and suggest approximations for large-scale problems.

---

## PemDVHC2KO

- GT: Reject (avg 2.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces TiEBe, a benchmark designed to evaluate large language models' factual recall of notable global events across time, geography, and language. Constructed from Wikipedia retrospective pages and their cited external sources, TiEBe comprises over 23,000 question-answer pairs spanning 10 years, 23 regions, and 13 languages. The core empirical findings reveal significant geographic performance disparities, a strong correlation between model accuracy and socioeconomic indicators like GDP and HDI, and pronounced degradation for low-resource languages.

## Strengths
- **Large-Scale, Multifaceted Benchmark Construction:** The creation of TiEBe represents a substantial data collection and curation effort. It uniquely integrates temporal (10-year span), geographic (23 regions), and multilingual (13 languages) dimensions into a single evaluation resource, with questions provided in both English and native languages. This scale and design surpass many existing factual recall benchmarks.
- **Rigorous Socioeconomic Correlation Analysis:** The paper moves beyond reporting performance scores to systematically quantify the relationship between model accuracy and national development indicators. The finding of strong correlations (Spearman >0.7 for GDP/HDI with native-language performance) for pre-cutoff events provides compelling, empirical evidence of a widely suspected bias in LLM knowledge representation.
- **Transparent and Detailed Reporting:** The paper provides extensive methodological details, full prompts, model versions, and comprehensive appendices with yearly/regional performance heatmaps (Appendix E). The validation of the LLM-as-judge against human annotation (200 samples) and the public release of code enhance reproducibility.

## Weaknesses
### Major:
- **Benchmark Source Bias Threatens Interpretation of Geographic Disparities:** TiEBe is constructed from Wikipedia retrospective pages, whose availability and density are heavily skewed (see Appendix C.1, Fig. 6), favoring Western, English-speaking, and digitally prominent nations. Consequently, a model's low performance on a region (e.g., DRC) may reflect a paucity of Wikipedia coverage for that region's events rather than a failure of the model's factual recall *of notable events*. This fundamental confounder is acknowledged in the limitations but is central to the paper's primary claim of measuring "geographic disparities in factual recall." It undermines the ability to attribute observed gaps solely to model deficiencies versus inherent skew in the benchmark's source data.
- **Superficial Temporal Analysis:** The paper's goal to track "recall through time" is not convincingly met. The analysis in Section 4.2 and Figure 4 primarily shows a sharp accuracy drop for events post-2023, which aligns trivially with most models' training cutoffs. The stable performance on pre-cutoff events (2015-2022) does not demonstrate an analysis of knowledge evolution or decay over time. The benchmark lacks a controlled setup (e.g., evaluating models with identical cutoffs on specific temporal slices) to isolate how knowledge of past events is retained or lost.
- **End-to-End Reliance on a Single LLM Family Introduces Unquantified Risks:** The pipeline depends heavily on DeepSeek-V3 for QA generation, translation, and as the sole judge for evaluation. While judge agreement is reported at 88.5%, this leaves an 11.5% error margin unexamined across 23k samples. More critically, there is a risk that the generative step creates questions aligned with the generator's biases, and the evaluator step may share those biases, creating a circularity that could affect score objectivity. The lack of ablations (e.g., using different models for generation/judging) is a significant methodological gap.

### Minor
- **Limited Causal Investigation of Disparities:** The paper excellently documents *what* the disparities are and correlates them with external indicators. However, it does not attempt to disentangle the potential mechanistic causes—for instance, whether lower performance is due to less training data from a region, lower data quality, the judge's potential biases, or a combination. This limits actionable insights for mitigating the gaps.
- **Insufficient Analysis of Language vs. Factual Recall:** The language-effect results (Section 4.3) conflate poor multilingual capability with a lack of factual knowledge. The analysis would be stronger if it separated these factors, for example, by more deeply analyzing cases where a model is linguistically proficient but still factually incorrect.

## Nice-to-Haves
- A more extensive, multi-annotator human evaluation of QA pairs and model responses, particularly for low-performing regions and languages, would further bolster the benchmark's validity.
- An ablation study on the question generation method (e.g., using different models or template-based approaches) would help show results are not an artifact of DeepSeek-V3's specific generation quirks.
- A deeper error analysis categorizing failures by question type (e.g., "what" vs. "how many") or reasoning demand could provide more nuanced insights into model limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "The claim of evaluating 'recall through time' is not convincingly supported..."** While this point is partially valid (see Major Weakness #2), the original harsh critic's framing overstated the issue as a complete failure. The paper does provide a temporal breakdown; the weakness is in the depth of analysis, not the absence of any temporal evaluation.
- **Weakness: "Lack of benchmark comparison ablation..."** (From Spark Finder). This is a request for scope expansion. The paper's contribution is the introduction and analysis of TiEBe itself; a direct comparison to other benchmarks, while useful, is not a required component for establishing its value.
- **Weakness: "Statistical significance testing is needed..."** (From Spark Finder). For a large-scale benchmark evaluation reporting aggregate accuracy across thousands of samples, confidence intervals or significance tests on performance differences, while good practice, are not yet a standard requirement in the field for this type of work.
- **Strength: "The paper is well-written..."** Removed as a generic strength.
- **Strength: "The topic is important..."** Removed as a generic strength.

## Suggestions
To strengthen the paper, the authors should directly address the major weaknesses in a revised discussion:
1.  Reframe the interpretation of geographic disparities to explicitly acknowledge that TiEBe measures model performance on a Wikipedia-curated view of world events, and that the observed gaps likely reflect a combination of model bias *and* source data availability bias. Discuss how this interplay itself is a valuable finding about the propagation of information inequality.
2.  Deepen the temporal analysis. For example, model accuracy decay as a function of months-since-events for models with known cutoffs, or analyze whether performance on events from the same region but different pre-cutoff years shows any meaningful variation.
3.  Add a critical ablation or sensitivity analysis for the evaluation pipeline. This could involve using a second, distinct LLM-as-judge (e.g., GPT-4o) on a subset of responses to report inter-judge agreement and ensure key conclusions are robust.

---

## dPAcHrG4rl

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (3.5/10)
- Match: N/A

### Final Review

## Summary
This paper presents an information-theoretic analysis of the limitations of single-pass reasoning in Large Language Models (LLMs) for Multi-Hop Question Answering (MHQA). It derives a Fano-style upper bound on accuracy, formalizing an "Accuracy Cliff" where performance collapses when task information demand exceeds the model's output capacity. Based on this theory, the authors propose InfoQA, a proof-of-concept multi-call framework that decomposes tasks, prunes reasoning traces, and uses an explicit workflow to manage information load. The theory and framework are validated on a controlled synthetic benchmark.

## Strengths
- **Rigorous Theoretical Foundation:** The paper provides a formal, information-theoretic derivation of a performance bound for single-pass LLM reasoning (Theorem 1). The analysis cleanly connects classical tools (conditional Fano inequality, output entropy bound) to a modern LLM bottleneck, yielding an interpretable "Accuracy Cliff" prediction.
- **Controlled and Systematic Empirical Validation:** The authors construct a novel synthetic benchmark that allows fine-grained, independent control over key difficulty factors (hop count, context length). This design enables a clean test of the theoretical predictions, and the results show single-pass baselines following the predicted capacity curves.
- **Well-Motivated Proof-of-Concept Framework:** InfoQA is a direct, operational implementation of the principles derived from the theoretical analysis. Its components (capacity-aware decomposition, dependency-explicit workflow, iterative query contraction) are clearly explained, and ablation studies demonstrate the necessity of its core design choices.

## Weaknesses
### Major:
- **Circularity in Theoretical Validation:** The empirical validation of the Fano-style bound is not independent. The parameters of the information demand model (β₀, α, γ) and the model capacity (C) are fitted to the observed performance data (F1 scores) via grid search (Section 5.2, Eq. 11, Appendix A.5). The resulting close alignment between theory and data (Figure 5) is therefore a post-hoc fit, not a prediction from first principles. The paper does not provide an independent, task-side method to estimate β or C, weakening the claim that the bound *governs* model behavior.
- **Lack of Real-World Benchmark Validation:** The entire empirical evaluation is conducted on a synthetic, controlled benchmark. While this is suitable for testing the theory in isolation, it leaves the practical efficacy and generalizability of both the theoretical insight and the InfoQA framework unproven. There is no validation on established, real-world MHQA datasets (e.g., HotpotQA, MuSiQue), where natural noise, diverse question structures, and potential shortcuts could yield different results.
- **No Direct Manipulation of the Theorized Bottleneck (C):** The theory centers on output capacity `C = H(Y)`. A strong causal test would involve experimentally manipulating `C` (e.g., by capping the maximum allowed output tokens) and observing if the accuracy cliff shifts accordingly. The paper infers `C` from performance curves but does not perform such a manipulation, leaving the causal link between the derived bound and model behavior less firmly established.

### Minor:
- **Over-Simplifying Theoretical Assumptions:** The elegant, interpretable bound (Eq. 5, `Acc ≤ (C+1)/β`) and the demand model (Eq. 6, `β(h,L) = β₀ + αLγ^(h-1)`) rely on simplifying assumptions (e.g., uniform answer distribution, exponential hop amplification). The paper acknowledges these but does not analyze how often these assumptions hold in practice or how violations affect the bound's tightness, limiting the bound's claimed generality.
- **Narrow Model and Task Scope Evaluation:** Experiments are limited to two sizes of a single model family (Qwen3). Testing on models with different architectures, training regimes, and scales is necessary to establish the "Accuracy Cliff" as a universal LLM phenomenon, not an artifact of a specific model. Furthermore, reasoning chains are limited to 4 hops; generalization to longer, more complex chains remains unverified.
- **Superficial Error Analysis:** The error analysis is brief and generic, identifying "semantic drift" and "intrinsic model capacity" as failure modes for InfoQA. A more systematic breakdown comparing error types between single-pass (e.g., capacity overflow) and multi-call (e.g., faulty decomposition) failures would provide clearer guidance for future improvements.

## Nice-to-Haves
- **Validation on Standard MHQA Benchmarks:** Adding results on datasets like HotpotQA or 2WikiMultihopQA would significantly strengthen the paper's practical relevance.
- **Deeper Analysis of Capacity (C):** A discussion on how `C` relates to tangible LLM properties (e.g., max generation length, decoding distribution entropy) would make the theory more actionable.
- **Comparison with Other Multi-Call Frameworks:** While the paper compares to single-pass baselines, a direct comparison with other modern multi-call reasoning systems would better contextualize InfoQA's contribution within the multi-call paradigm.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength - "Well-written" or "Important Topic":** Removed as generic.
- **Weakness - "Unfair Baseline Comparison":** The critic claimed ReAct and Self-Ask were unfairly implemented as single-pass. The paper explicitly compares single-pass variants of all baselines to isolate the effect of the reasoning paradigm, which is a valid experimental design. This is not an unfair asymmetry that favors the author's method.
- **Weakness - "Lack of Novelty in Framework Components":** While the individual components of InfoQA are not novel in isolation, the framework's contribution is its principled derivation from and validation of a new theoretical analysis. The criticism is overly reductive.
- **Weakness - "Formatting Nitpicks":** Any minor stylistic comments are removed.
- **Weakness - "Reproducibility Nitpicks":** Concerns about undisclosed hyperparameters or large training logs are removed as trivial.

## Suggestions
- To address the major circularity issue, consider developing an independent method to estimate the information demand β from task statistics (e.g., context characteristics, answer space size) without fitting to performance data. Alternatively, reframe the contribution as providing a *descriptive* model that fits observed data well, rather than a predictive theoretical law.
- Run InfoQA and key baselines on 1-2 standard, real-world MHQA benchmarks to demonstrate generalizability beyond the synthetic setting.
- Include an experiment where the output capacity `C` is directly manipulated (e.g., by restricting the maximum generation length) to test if the accuracy cliff's location shifts as predicted by the theory.

---

## CVZFzsg1PJ

- GT: Withdrawn (treated as Reject) (avg 2.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper reframes neighborhood control in the Local Branching (LB) heuristic for Mixed-Integer Linear Programming (MILP). Instead of learning the scalar neighborhood radius parameter \(k\), the method first partitions variables into structurally meaningful clusters via graph community detection (Louvain). A reinforcement learning (RL) agent then dynamically selects a subset of these clusters to define the branching neighborhood at each iteration. The framework aims to automate neighborhood design without requiring offline data collection of solved instances.

## Strengths
- **Conceptual Reframing**: The shift from tuning a numerical parameter (\(k\)) to learning a policy over structurally-derived variable subsets is a novel and meaningful advancement for controlling Local Branching. It better leverages problem structure and moves beyond prior learning-based methods that focus on predicting \(k\).
- **Effective Integration of Structure and Learning**: The two-stage pipeline—unsupervised graph clustering followed by RL-guided search—is coherent and well-motivated. The ablation study (Table 3) clearly demonstrates that both the structure-aware clustering and the RL agent contribute positively to performance.
- **Strong Generalization Demonstrated on Larger Instances**: The method shows robust performance when evaluated on larger instances (with doubled variable/constraint counts) across all three synthetic benchmark classes (Set Covering, Independent Set, Combinatorial Auction), indicating good scalability.

## Weaknesses
### Major:
- **Incomplete Baseline Comparison in Primary Experiments**: Table 3, which presents results on the main benchmark problems, does not include the SCIP solver. The paper's central claim is to outperform "state-of-the-art learning-based LB models and the open-source solver SCIP." The absence of SCIP from this primary comparison table undermines the claim for the standard experimental setting. SCIP results appear only in Table 4 for larger instances and MIPLIB. This omission prevents a direct verification of the core claim on the primary benchmarks.
- **Ambiguous and Uninterpretable MIPLIB Results**: The MIPLIB results in Table 4 report PrimalBound and PrimalGap, but the objective sense (minimization or maximization) for these instances is not specified. For example, SARLB's PrimalBound (9,049,794,557) is numerically higher than SCIP's (8,729,503,534), yet SARLB's PrimalGap is reported as lower (1.36% vs. 6.64%). Without knowing if this is a minimization problem (where lower bounds are better) and without the best-known solution values to contextualize the gaps, the reader cannot assess whether a lower PrimalGap genuinely indicates superior performance. This ambiguity severely weakens the claim of effective generalization to real-world, heterogeneous instances.
- **Overly Simplistic RL Action Space Contradicts Framing**: The RL agent's action space is limited to \(\{-Δ, 0, +Δ\}\), controlling only the *number* of clusters to select. The actual selection of *which* specific clusters is done by a simple, non-learned heuristic (inverse-frequency sampling). This contradicts the paper's framing of learning "a policy to select a subset of variables" (Abstract) and "dynamically selects the number of clusters to explore per iteration" (Section 4.2). The agent learns a pacing schedule for neighborhood size, not an intelligent policy for variable subset composition, which limits the sophistication of the learned control.

### Minor:
- **Limited Comparison to Contemporary RL-for-Optimization Methods**: The primary learning baseline is LB-SRMRL, a method for tuning \(k\). The paper would be strengthened by a direct comparison to other recent RL-based methods for neighborhood search or destroy operators (e.g., those cited in Section 2 like Song et al. 2020, Wu et al. 2021) to better situate its novelty and performance within the broader RL-for-optimization landscape.
- **Heuristic Design Choices Lack Justification or Sensitivity Analysis**: Key heuristic rules—the dynamic radius \(k \propto \sqrt{|\bar{S}| + |B \setminus \bar{S}|}\), the time-limit scaling \(T_t = T_0 \cdot (n_t/n_0)^\alpha\), and the inverse-frequency cluster selection—are presented without theoretical motivation or empirical sensitivity analysis. While they appear effective, their specific forms and parameters (e.g., \(α\)) are not justified, leaving their robustness unclear.
- **Computational Overhead of Preprocessing Unexplored**: The graph construction and Louvain clustering are performed as a preprocessing step. The computational cost of this step is not reported or analyzed. For very large-scale instances, this overhead could offset some of the solving-time benefits, but its impact remains unquantified.

### Trivial:
- **Typographical Error in Table 3**: Table 3 includes an unexplained row labeled "SNLB." From context, this appears to be a formatting error for the proposed "SARLB" method. While this does not affect the interpretation of the ablation results, it is a minor presentation flaw.

## Nice-to-Haves
- **Deeper Analysis of Learned Policy Behavior**: A simple analysis correlating the RL agent's actions (\(+Δ/0/-Δ\)) with search state features (e.g., recent improvement, incumbent age) would illuminate whether the policy learns intelligent behavior beyond a simple schedule.
- **Characterization of Cluster Properties**: Analyzing properties of the generated clusters (size distribution, intra-cluster constraint coupling) and their correlation with selection frequency or performance would strengthen the claim that community detection finds structurally *useful* neighborhoods.
- **Breakdown of Runtime Costs**: Separating the total runtime into clustering time, RL inference time, and solver time for subproblems would provide a clearer picture of the method's practical overhead and scalability.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength: "The paper is well-written" / "The topic is important"** - Removed as per the rule to exclude generic strengths that apply to any paper.
- **Weakness: "Statistical significance testing is missing"** - Removed as a "nice-to-have" under soft rules. While providing confidence intervals is good practice, reporting average performance over 40 test instances per class is a common and accepted standard in the field for large-scale MILP benchmarking.
- **Weakness: "Small test set size (40 instances)"** - Removed as a generic weakness. The dataset size is consistent with prior work in the area (e.g., Liu et al. 2022, Gasse et al. 2019) and is sufficient for initial evaluation.
- **Weakness: "Code is not currently available"** - Removed per the hard rule on reproducibility nitpicks. The paper includes a reproducibility statement committing to release code upon publication, which is standard.
- **Weakness: "Demand for comparison to a wider array of ML methods"** - Weakened and integrated as a minor "limited comparison" point. The paper adequately compares to the most relevant direct baseline (LB-SRMRL). Demanding comparisons to every possible ML method is scope creep.
- **Weakness: "Training time and computational cost not discussed"** - Weakened to a minor point about "computational overhead." Detailed wall-clock training time is often omitted in RL-for-optimization papers in favor of final solution quality metrics.
- **Weakness from Harsh Critic: "Key RL design choices are underspecified (Δ, reward normalization)"** - Partially removed. The paper specifies the reward as normalized relative to the initial solution (Section 4.2: "normalized objective improvement relative to the initial solution \(x_0\)"), which is sufficient. The step size \(Δ\) is a hyperparameter; its absence from the appendix tables is a minor omission, not a critical methodological gap.

## Suggestions
- **Include SCIP results in Table 3**: To substantiate the core claim of outperforming SCIP, add a row for SCIP (with default settings) to Table 3, reporting its PrimalBound, PrimalGap, and PrimalIntegral on the standard small benchmarks.
- **Clarify MIPLIB Results**: In Table 4 or its caption, specify the objective sense (minimization/maximization) for the MIPLIB problems and, if possible, provide the best-known solution values used to compute the PrimalGap. Alternatively, reformat the table to use metrics that are unambiguous without this context (e.g., directly compare PrimalBounds if all instances are minimization).
- **Reframe the Contribution Language**: Adjust the abstract and introduction to more accurately reflect that the RL agent controls the *scale* (number of clusters) of the neighborhood, while cluster *composition* is determined by structure-aware generation and a simple diversification heuristic. This resolves the contradiction between framing and implementation.

---

## qioDi3afqm

- GT: Withdrawn (treated as Reject) (avg 0.0)
- Predicted: N/A (0.0/10)
- Match: N/A

### Final Review

## Summary
This submission is not a research paper. It is the official ICLR 2026 formatting instruction document, detailing layout, citation, and submission requirements. It contains no research question, methodology, experiments, results, or novel scientific contribution.

## Strengths
- **Clear and comprehensive procedural guide**: The document provides exceptionally detailed and unambiguous formatting specifications (e.g., exact margins, font sizes, heading styles, and file preparation commands), which is its intended purpose as a style guide.
- **Well-structured template**: The instructions are logically organized with a clear hierarchy, making it easy for authors to locate specific requirements.

## Weaknesses
### Fatal
- **Not a research contribution**: The submission is a formatting template and style guide. It lacks all core components of a research paper: an abstract stating a problem and contribution, an introduction, related work, a proposed method, experiments, results, and a discussion. Therefore, it does not meet the basic criteria for evaluation as a scientific contribution to ICLR. This is a categorical mismatch with the conference's purpose.

### Minor
- **Contains placeholder text**: Sections like the abstract, author list, and references contain template placeholder text (e.g., "Anonymous authors," "Paper under double-blind review," example citations), which is inappropriate for a final submission.

### Trivial
- **Self-referential formatting**: As a meta-document about formatting, its own formatting is correct, but this is irrelevant to its eligibility as a research paper.

## Nice-to-Haves
- If the authors intended to submit research, they should replace the template content with novel scientific work, using this document only as a formatting shell.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strengths**: "The topic is important" or "The paper is well-written" were removed as generic and not specific to a research contribution.
- **Weaknesses**: Any criticism about missing comparisons to other methods, unreleased benchmarks, or reproducibility details (e.g., hyperparameters, training logs) was removed, as the paper does not propose a method to compare. Criticisms about missing related work were removed per the rule against mentioning missing works without external sources.

## Suggestions
- The authors should withdraw this submission and prepare a standard research manuscript that addresses a novel problem in machine learning, proposes a method or theory, and provides empirical or theoretical validation. The current document could serve as a reference for formatting that new manuscript.

---

## QNHjSRO8xE

- GT: Reject (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces two novel Monte-Carlo Tree Search (MCTS) algorithms, CATSO and PATSO, which combine distributional return estimates at Q‑nodes with Thompson Sampling and an explicit polynomial optimism bonus. The authors provide non‑asymptotic regret bounds (O(n⁻¹ᐟ²) simple regret at the root) and a novel connection to Wasserstein Distributionally Robust MDPs. Empirical evaluation on synthetic stochastic trees and 12 Atari games shows competitive performance against strong baselines, with comprehensive ablations analyzing each component.

## Strengths
- **Novel algorithmic synthesis:** The paper cleanly unifies distributional value representations (categorical or particle-based), Thompson sampling for action selection, and count-based optimism within a single MCTS framework. This integration is clearly specified and well-motivated for stochastic settings.
- **Strong theoretical grounding:** Non‑asymptotic regret guarantees are provided for both algorithms, matching the state‑of‑the‑art rate for fixed‑depth MCTS. The analysis elegantly models nodes as non‑stationary bandits and lifts results to the full tree. The connection to Wasserstein distributionally robust MDPs offers an insightful robustness interpretation and a sample‑complexity bound.
- **Thorough empirical validation:** Experiments cover a range of synthetic tree configurations and 12 Atari games, with careful comparisons against multiple strong baselines. The ablation studies systematically isolate the effects of distributional Q‑nodes, Thompson sampling, optimism, and backup rules, providing clear evidence for each design choice.
- **Practical engineering contributions:** The “merge‑on‑insert” mechanism for PATSO caps memory usage while preserving theoretical guarantees. Hyperparameter sensitivity analyses show robustness, and runtime/memory measurements are reported, making the algorithms practical.

## Weaknesses
### Major:
- **Theoretical assumptions may not fully capture MCTS non‑stationarity:** The analysis relies on an asymptotic stationarity assumption (Assumption 1) that each node’s reward process stabilizes to i.i.d. samples from a limiting distribution. While justified intuitively, this assumption simplifies the interdependence of nodes in a growing tree, and the resulting guarantees might not hold if the assumption is violated in practice.
- **Empirical improvements are modest and not consistently superior:** The core claim that distributional Q‑nodes provide a key advantage in stochastic environments is not strongly supported. Ablations show scalar Thompson sampling with the same optimism bonus (ScalarTSOpt) performs similarly to CATSO with mean backup (Table 2). In deterministic/low‑noise settings, Power‑UCT often outperforms the proposed methods (Table 4). Atari results show CATSO/PATSO are competitive but not dominant, with UCT and Power‑UCT winning or tying in several games (Table 1). This undermines the claim that distributional representations are crucial for robust performance.

### Minor:
- **Connection to Wasserstein robustness is underdeveloped:** The link to Wasserstein Distributionally Robust MDPs is presented as an after‑the‑fact interpretation rather than a design principle. The sample‑complexity bound (Theorem 5) has an exponential dependence on horizon, limiting its practical relevance, and no experiment validates the robustness interpretation.
- **Scalability demonstration is limited:** Experiments are confined to synthetic trees of moderate depth/branching and deterministic Atari games. The paper does not show performance in extremely large, continuous, or genuinely stochastic/partially observable domains, leaving open questions about broader applicability.
- **Presentation of results could be more statistically rigorous:** Table 1 reports wins/ties counts that may inflate perceived success without statistical significance testing. Standard deviations are provided, but confidence intervals or significance tests would strengthen the comparisons.
- **Omission of a key baseline in Atari comparisons:** The theoretically relevant Fixed‑Depth‑MCTS baseline (Shah et al., 2022) is included in synthetic tree experiments (Figure 1) but not in the main Atari table (Table 1), making it harder to assess relative performance against this state‑of‑the‑art method.

### Trivial:
- *None*

## Nice-to-Haves
- Ablation of the optimism bonus (pure Thompson sampling without bonus) to isolate its contribution.
- Empirical validation of the WDRMDP connection via robustness tests under dynamics/reward perturbations.
- Visualization of Q‑node distributions over time to illustrate how the algorithm differentiates actions.
- Guidance on setting hyperparameters (e.g., optimism constant C) for new domains.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Criticism that V‑nodes remaining scalar severely limits the distributional nature:** The paper explicitly discusses this design choice in Section 3.3, justifying it for tractability, and ablations isolate the backup rule’s effect. It is a reasoned trade‑off, not a flaw.
- **Claim that the extension from bandits to the tree is “hand‑wavy”:** The paper provides Theorems 3 and 4 with proofs in the appendix; the extension is formally stated.
- **Suggestion that hyperparameter insensitivity implies the distributional parameterization is not crucial:** Flat sensitivity (Tables 5‑6) indicates robustness, not a lack of importance, and the ablation studies directly test the distributional component.
- **Request for comparisons to modern distributional RL planning methods (e.g., IQN, QR‑DQN):** This would require adding new baselines beyond the paper’s scope; the paper already compares to several strong MCTS variants.

## Suggestions
- Provide statistical significance tests (e.g., confidence intervals or p‑values) for the Atari results to clarify whether performance differences are meaningful.
- Include Fixed‑Depth‑MCTS in the Atari comparison table if feasible, or explain why it was omitted.
- Discuss the limitations of the asymptotic stationarity assumption more thoroughly, possibly with empirical evidence of node reward stabilization in the tested environments.

---

