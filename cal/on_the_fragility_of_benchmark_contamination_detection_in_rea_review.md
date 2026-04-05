=== CALIBRATION EXAMPLE 76 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately reflects the core contribution: diagnosing why existing contamination detection fails for LRMs.
- The abstract clearly states the problem, the two-stage experimental structure, and the key empirical/theoretical findings.
- One claim requires calibration: the abstract states that in Stage II, detection methods "perform near random guesses." However, Table 5 shows averages frequently between 52–65%, and Appendix E.11 shows AUROCs up to 76.1% for the Loss detector on MMLU-Pro. While performance degrades, "near random guesses" is an overstatement that weakens the abstract's precision.

### Introduction & Motivation
- The problem is well-motivated and timely. LRMs' reliance on long CoT and RL fine-tuning fundamentally shifts training dynamics away from simple memorization, creating a clear gap in prior contamination detection literature.
- Contributions are explicitly listed and accurately map to the subsequent sections.
- The introduction does not significantly over-claim, though it heavily frames contamination as "malicious" or "strategic." It would strengthen the motivation to acknowledge that these concealment effects could also emerge as unintentional side effects of standard RL scaling, not just deliberate leaderboard manipulation.

### Method / Approach
- **Clarity & Reproducibility:** The contamination pipelines, detection metrics, and implementation details (hyperparameters, frameworks, deduplication) are well-documented in Section 3 and Appendices D.1–D.4. The setup is reproducible.
- **Assumptions:** The method assumes AUROC calculated over question/response log-probabilities is the standard evaluation for contamination. This is standard practice, but the paper should explicitly justify why this metric remains valid given LRMs' highly variable response lengths.
- **Logical Gaps / Theory:** Theorem 3.1 decomposes the NLL drift into a mean term (A) and a covariance term (B). The claim that the covariance term drives concealment relies on the assertion that `β_N > β_M` because "non-members correct trajectories can exhibit much higher variance in loss and probabilities." This is intuitively plausible but lacks formal justification or empirical measurement in the main text. Why exactly do non-members exhibit systematically higher loss covariance under clipping? Providing a small-scale empirical plot or mathematical justification for this variance gap would significantly strengthen the theoretical contribution.
- **Edge Cases:** The theoretical analysis strictly conditions on `r = 1` (correct trajectories). In practice, contamination detectors score all generations, including failed CoT steps. How does the clipping/IS mechanism behave when rewards are sparse or `r = 0`? The theory's applicability to mixed-success generations is unclear.

### Experiments & Results
- **Do experiments test claims?** Yes. Stage I cleanly isolates RL's effect on SFT-contaminated models. Stage II tests direct CoT SFT on advanced models. The ablation across RAFT, RAFT++, and GRPO directly tests the clipping hypothesis (Table 3).
- **Baselines & Completeness:** 10 detection methods across 4 categories is thorough. The progression from pre-RL to post-RL, and the step-wise tracking (Fig 2/5), provides strong empirical support.
- **Missing Ablations:** The theory attributes concealment to PPO-style clipping. Table 3 removes clipping and shows AUROC collapse vanishes. However, GRPO also uses *group-relative normalization* for advantages. An ablation disabling only advantage normalization while keeping clipping would disentangle whether the concealment stems purely from clipping or from the interaction between group normalization and clipping.
- **Statistical Rigor:** Tables 2, 5, and others report point averages for AUROC across benchmarks but omit standard deviations or confidence intervals. Given the stochastic nature of RL rollouts and detector sampling (8 rollouts per question), reporting variance is crucial to confirm that observed drops (e.g., from 75.48% to 61.26%) are statistically significant and not within natural run-to-run noise.
- **Cherry-picking/Consistency:** Results are consistent across model scales (3B, 7B, 14B, Llama) and multiple benchmarks. Stage II results in Appendix E.11 show that for non-reasoning domains (MMLU-Pro), detection performance recovers somewhat (~66–76% AUROC). This directly contradicts the blanket "random guess" narrative for Stage II and suggests the phenomenon is tightly coupled to domains with long, deterministic CoT reasoning. The main text should temper its claims accordingly.

### Writing & Clarity
- The paper is well-structured and logically flows from problem identification to empirical diagnosis to theoretical explanation.
- Section 3.2 (Theoretical Analysis) is dense. The notation is mostly consistent, but the jump from Eq. (4) to Eq. (5) and the subsequent RAFT/RAFT++/GRPO instantiations requires careful reading. Clarifying the exact role of `m_t` (clipping mask) versus `ρ_t` in the covariance decomposition would aid readability.
- Figures 1–4 and Tables 1–5 are informative and directly support the claims. No clarity issues impede understanding of the core contribution.

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors correctly note they do not propose a new detection algorithm but diagnose why current ones fail. They highlight the breakdown of the "memorization" assumption.
- **Missed Fundamentals:** 
  1. The study focuses almost entirely on mathematical/scientific reasoning. As noted, Stage II concealment is weaker in factual QA domains. The authors should explicitly bound their claims: RL concealment appears robust across domains, but CoT-generalization evasion is likely domain-dependent.
  2. The experimental setup uses DAPO-style GRPO (no KL term). Modern RLHF often includes significant KL regularization against a reference policy. Strong KL constraints might restrict policy divergence enough to preserve log-prob separability, potentially mitigating the concealment effect. This interaction is unexplored.
- **Societal/Impact:** The paper responsibly highlights leaderboard manipulation risks. It could briefly discuss mitigation strategies beyond "release checkpoints," such as cryptographic data provenance or dynamic, unreleased benchmark streams, to provide a more complete view of the broader evaluation ecosystem.

### Overall Assessment
This is a timely and empirically rigorous paper that identifies a critical blind spot in current large reasoning model evaluation: RL fine-tuning systematically erases the log-probability signals that contamination detectors rely on, and long-CoT fine-tuning enables generalization that breaks the memorization assumption. The controlled ablation of RL objectives (RAFT vs. RAFT++ vs. GRPO) provides strong evidence that PPO-style clipping is the mechanistic driver. However, to meet ICLR's high standard for theoretical-empirical alignment, the authors must formally justify or empirically measure why non-members exhibit higher loss covariance under clipping (`β_N > β_M`) and clarify how the theory extends to `r=0` trajectories. Additionally, the claim that detection degrades to "random guessing" in Stage II is contradicted by the non-math results in Appendix E.11 and must be calibrated. Adding confidence intervals to AUROC tables and disentangling clipping from advantage normalization would further solidify the contribution. Despite these needed refinements, the paper makes a substantial, well-supported contribution to the LLM evaluation community and is well-positioned for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper systematically investigates the vulnerability of benchmark contamination detection methods specifically tailored for Large Reasoning Models (LRMs). The authors demonstrate through empirical evaluation and theoretical analysis that (1) PPO/GRPO-style RL training actively conceals contamination signals from prior SFT by collapsing the log-probability gap between membership and non-membership samples, and (2) CoT contamination applied to advanced LRMs triggers reasoning generalization rather than rote memorization, causing existing detectors to perform near chance. The work reveals critical flaws in current LRM evaluation protocols and calls for new detection paradigms.

### Strengths
1. **Timely and High-Impact Motivation:** Addresses a pressing issue in the LLM community as leaderboards increasingly drive development. The two-stage contamination framework (pre-LRM vs. post-LRM) realistically mirrors actual model development pipelines and effectively isolates where detection breaks down.
2. **Strong Empirical + Theoretical Synergy:** The paper goes beyond standard benchmarking by providing a theoretical derivation (Section 3.2, Theorem 3.1) linking PPO-style importance sampling and clipping to the contraction of membership separation. This is rigorously supported by controlled ablations (Table 3) showing RAFT (no clipping) preserves detectability while RAFT++/GRPO (with clipping) collapse AUROC, directly validating the theoretical mechanism.
3. **Comprehensive Detector Evaluation & Thorough Experimental Design:** Ten representative detection methods across four paradigms are evaluated across six challenging reasoning benchmarks. The authors carefully control for confounding factors (e.g., proving RL doesn't just cause forgetting in Section 3.1, verifying trends across 3B/7B/14B models in Appendix E.6, and providing step-wise AUROC degradation in Figure 2 & Tables 12-15).
4. **High Reproducibility:** Clear contamination pipelines, hyperparameters (Appendix D.4), dataset descriptions, and a public codebase enable exact replication. The inclusion of multiple base models (Qwen2.5, Llama-3.1) and distillation targets demonstrates robust setup practices.

### Weaknesses
1. **Idealized Theoretical Assumptions:** The proof in Section 3.2 relies on a tabular setting, small natural gradient steps, and exclusively analyzes correct trajectories ($r=1$). Modern RL for reasoning operates in sparse-reward, multi-turn, and high-dimensional embedding spaces with KL-regularization, advantage normalization, and verifier feedback. While the theory provides valuable intuition, the gap between the simplified assumptions and practical RLHF/RLVR pipelines is not fully discussed.
2. **Under-Characterized "Generalization vs. Memorization" in Stage II:** Section 4 attributes the failure of detectors to LRMs generalizing reasoning patterns rather than memorizing CoT traces. However, this claim is primarily supported by shifted log-prob distributions (Figure 4) without quantitative isolation of skill transfer vs. pattern matching. Mechanistic probes (e.g., activation similarity, explicit OOD generalization metrics, or ablations on CoT syntax variation) would strengthen this conclusion.
3. **Limited Main-Text Scope Beyond Math/Reasoning Domains:** The core experiments focus exclusively on math and science benchmarks. Results for coding and general QA (Appendix E.11) show mixed effects (e.g., general QA detectors still achieve ~65-75% AUROC) and are relegated to the appendix. This limits claims about the universality of the concealment phenomenon across all LLM domains.
4. **Lack of Constructive Baselines or Protocols:** The paper effectively deconstructs existing methods and proposes high-level recommendations (release checkpoints, rethink memorization assumptions), but stops short of introducing even a simple adapted detection baseline. For a top-tier venue like ICLR, providing a preliminary countermeasure or evaluation protocol would significantly elevate the paper's impact.

### Novelty & Significance
- **Novelty:** High. This is the first study to systematically map contamination detection failure across the base-to-LRM and post-LRM training pipelines. The identification of PPO clipping as an algorithmic driver for concealment is a novel and non-obvious contribution to both evaluation safety and RL literature.
- **Clarity:** Strong. The paper is well-organized, with a clear narrative linking empirical observations to theoretical mechanisms. Figures and tables effectively communicate trends, though the mathematical notation in Section 3.2 may require careful reading for practitioners less familiar with policy gradient analysis.
- **Reproducibility:** Excellent. Detailed training recipes, dataset splits, detector configurations, and open-sourced code fully comply with modern ML reproducibility standards.
- **Significance:** Very High. The findings directly challenge the integrity of public LRM leaderboards and warn developers/researchers that current contamination audits may yield false negatives. This will likely prompt a reevaluation of evaluation standards and accelerate research into reasoning-aware contamination detection.

### Suggestions for Improvement
1. **Strengthen Stage II Analysis with Mechanistic or OOD Evidence:** Incorporate explicit metrics to distinguish generalization from memorization (e.g., evaluate contaminated models on structurally similar but semantically different prompts, measure CoT step diversity, or analyze attention/activation divergence between members and non-members).
2. **Discuss Theoretical Limitations in Context:** Add a paragraph in Section 3.2 or Discussion explicitly mapping the tabular/single-step assumptions to practical RL settings. Discuss how KL penalties, advantage clipping thresholds, or verifier noise might mitigate or exacerbate the contraction effect.
3. **Elevate Non-Math Domain Analysis to Main Text:** Move the coding and general QA results (Appendix E.11) into the main paper with a brief analysis. Clarify whether concealment is inherently tied to long-CoT reasoning or also applies to standard instruction-tuning paradigms.
4. **Propose and Test a Preliminary Countermeasure:** Introduce at least one lightweight adaptation to current detectors (e.g., leveraging reasoning step entropy, multi-prompt calibration, or verifier consistency checks) and report initial AUROC. This would transform the paper from a critical analysis into a constructive step toward robust LRM evaluation.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Test contamination with realistic data mixture ratios (e.g., 10–30% benchmark data) instead of heavy oversampling and exact replication. Without it, the claim that RL consistently conceals practical developer behavior is overstated.
2. Evaluate detection on distributionally shifted non-member benchmarks rather than simple 50/50 in-batch splits. Current AUROC calculations conflate detection failure with natural cross-distribution generalization, invalidating the measured fragility.
3. Run Stage I experiments with standard PPO/GRPO objectives that include the KL divergence regularization term. Removing KL for theoretical tractability decouples the findings from standard production LRM training pipelines.
4. Benchmark against recent (2024–2025) fine-tuning or alignment-aware contamination detectors. Relying primarily on pre-training baselines weakens the claim that *all* current detection methodology is inherently broken for LRMs.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the exact performance inflation achieved at the step where AUROC collapses to random guessing. Without mapping the inflation-vs-detection trade-off curve, the practical severity and exploitability of the vulnerability remain speculative.
2. Statistically validate AUROC drops across multiple training seeds and hyperparameter variations. Reported near-random scores (≈50–55%) could stem from stochastic optimization variance rather than systematic signal suppression.
3. Mechanistically explain why Stage II CoT contamination triggers generalization instead of memorization. Analyze trajectory structural similarity or reward variance to prove LRMs learn abstract reasoning patterns rather than merely overfitting to dataset redundancy.
4. Analyze whether concealment strength correlates with CoT length, problem complexity, or reward sparsity. The theoretical clipping mechanism lacks grounding in how variable-length reasoning traces actually interact with importance sampling.

### Visualizations & Case Studies
1. Provide side-by-side CoT trajectory comparisons for matched member/non-member questions before and after RL training. This directly exposes whether the model memorizes answers, paraphrases steps, or genuinely abstracts the reasoning process.
2. Plot the empirical distribution of active clipping masks and importance sampling weights over member vs. non-member tokens during GRPO updates. This is required to empirically validate Theorem 3.1 beyond indirect AUROC proxies.
3. Visualize per-token detection score trajectories along the reasoning sequence to identify exactly which reasoning steps retain or lose contamination signals, revealing whether detectors should target intermediate steps rather than final outputs.

### Obvious Next Steps
1. Propose and benchmark a simple mitigation (e.g., clipping-free RL, trajectory watermarking, or representation regularization) to demonstrate the identified mechanism is addressable, transforming a negative finding into an actionable contribution.
2. Establish a proper evaluation protocol with explicit negative controls (e.g., clean models fine-tuned on distributionally identical but unseen data). The current setup lacks baselines to calibrate how standard RL fine-tuning naturally shifts detection score distributions.
3. Extend Stage II contamination analysis to non-mathematical reasoning domains (e.g., code generation, logical planning, or scientific synthesis) to prove the detection failure generalizes beyond math-specific reward verifiers and benchmark structures.

# Final Consolidated Review
## Summary
This paper systematically investigates the failure of benchmark contamination detection methods for Large Reasoning Models (LRMs). Through empirical evaluation and theoretical analysis across two training stages, the authors demonstrate that (1) PPO/GRPO-style RL fine-tuning systematically collapses the log-probability gap between contaminated (member) and clean (non-member) samples, concealing prior SFT contamination, and (2) CoT-based SFT on advanced LRMs induces distributional generalization that breaks the core memorization assumption of existing detectors. The work identifies PPO-style clipping as the mechanistic driver of signal suppression and calls for robust, reasoning-aware evaluation protocols.

## Strengths
- **High-impact, timely problem framing with rigorous staging:** The two-stage contamination pipeline (pre-LRM SFT+RL vs. post-LRM CoT SFT) accurately mirrors modern LRM development. Evaluating 10 detection methods across 4 paradigms and 6 challenging math benchmarks provides comprehensive empirical grounding for the claimed fragility.
- **Strong theory-empirical alignment on clipping as the concealment driver:** The derivation in Section 3.2 isolates the role of PPO-style importance sampling and clipping in contracting member/non-member NLL separation. This is convincingly validated by controlled ablations (Table 3) showing that removing the clipping term restores detectability in RAFT++ and GRPO, establishing a clear algorithmic cause rather than a vague training artifact.
- **Excellent reproducibility and experimental control:** The authors provide fully documented contamination pipelines, hyperparameters, dataset splits, and a public codebase. They further control for confounding factors like catastrophic forgetting (by showing RL preserves performance inflation while dropping AUROC) and validate scaling trends across 3B/7B/14B models (Appendix E.6).

## Weaknesses
- **Theoretical analysis relies on unrealistic simplifications that undermine practical relevance:** Theorem 3.1 operates in a tabular, single-step setting, conditions exclusively on correct trajectories ($r=1$), and explicitly omits KL regularization against a reference policy. Modern RLHF/RLVR pipelines heavily depend on KL penalties to prevent collapse and operate with sparse verifier rewards. The key claim that $\beta_N > \beta_M$ (non-members exhibit higher loss covariance under clipping) is asserted intuitively but lacks formal justification or empirical validation. **Why it matters:** Without KL and multi-turn reward dynamics, the theoretical mechanism cannot be reliably extended to production reasoning models, making the claim that "a broad class of RL methods inherently exhibit similar concealment capability" overstated.
- **Stage II "generalization" claim is correlational, not mechanistic:** The paper attributes Stage II detector failure to LRMs internalizing reasoning patterns rather than memorizing CoT traces, citing only aggregate log-probability shifts (Figure 4). There is no trajectory-level analysis, CoT syntactic diversity measurement, or explicit OOD evaluation to distinguish true structural generalization from superficial overfitting to benchmark phrasing. **Why it matters:** If the model is merely overfitting to redundant dataset distributions rather than learning abstract reasoning steps, the conceptual framing and proposed detection challenges are fundamentally mischaracterized.
- **Lack of statistical rigor and domain boundary conditions:** All AUROC results are reported as point averages across benchmarks and RL steps without standard deviations, confidence intervals, or multi-seed validation. Given the stochastic nature of RL rollout sampling and detector variance, it is impossible to verify whether observed AUROC drops are statistically significant or within natural optimization noise. Furthermore, the abstract claims detectors perform "near random guesses," yet Appendix E.11 reveals AUROCs of 65–76% for non-STEM QA domains. **Why it matters:** The absence of statistical bounds weakens empirical credibility, while extrapolating math-domain concealment to a universal LRM failure mode misrepresents the actual scope of the vulnerability.

## Nice-to-Haves
- Disentangle GRPO's clipping effect from its group-relative advantage normalization via targeted ablation, clarifying whether concealment is purely a clipping artifact or emerges from their interaction.
- Elevate the coding and general QA results (Appendix E.11) to the main text to explicitly characterize the domain dependence of the concealment phenomenon and properly bound the claims.
- Provide a lightweight mitigation or adapted detection baseline (e.g., clipping-free RL, trajectory entropy scoring, or representation regularization) to transform the negative audit into a constructive step for robust evaluation.

## Novel Insights
The paper reveals a hidden, systemic cost of standard policy optimization: the PPO clipping gate, universally deployed as a training stabilizer, actively functions as a statistical sanitizer that compresses member/non-member log-probability distributions. This means RL fine-tuning does not merely overwrite or dilute contamination signals through additional gradient steps; it structurally homogenizes the confidence landscape for correct trajectories, rendering post-hoc membership inference fundamentally brittle. Simultaneously, CoT fine-tuning shifts models away from rote sample memorization toward distributional generalization, breaking the foundational assumption that contamination leaves a distinct log-prob footprint. Together, these findings reframe contamination detection not as a data auditing problem, but as an alignment dynamics problem.

## Potentially Missed Related Work
- **RLHF/KL-constrained convergence analysis** (e.g., recent work on how KL regularization bounds policy divergence and affects distributional shifts between training and held-out data) — highly relevant for contextualizing why the no-KL DAPO setup might artificially amplify the observed clipping-driven collapse compared to standard RLHF pipelines.
- **Mechanistic studies of reasoning trace storage in LRMs** — recent interpretability work analyzing whether LRMs store exact prompt-response mappings versus learning algorithmic subroutines would directly strengthen or refine the Stage II generalization claims.

## Suggestions
- Augment the theoretical analysis by incorporating a KL penalty term into the NLL drift decomposition, and empirically verify whether standard KL regularization mitigates or exacerbates the log-prob contraction. Provide multi-seed AUROC runs with confidence intervals to statistically validate the reported detection drops.
- Conduct explicit trajectory-level analyses in Stage II: measure CoT structural similarity, step-wise edit distance, or activation divergence between member and non-member generations. Include an explicit OOD transfer test (e.g., semantically equivalent but structurally permuted problems) to empirically distinguish generalization from benchmark-specific overfitting.
- Clearly scope the paper's claims by moving non-reasoning domain results to the main text and explicitly stating that the "near random" degradation is tightly coupled to long-CoT mathematical domains, not universal across all fine-tuning paradigms.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 6.0]
Average score: 6.7
Binary outcome: Accept
