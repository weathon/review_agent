=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy:** The title accurately captures the core methodology (plug-and-play framework), the attack modality (multi-turn jailbreaks), and the adaptive mechanism (life-long adaptive generation).
- **Abstract clarity:** The abstract clearly outlines the problem (multi-turn jailbreaks are underexplored but highly effective), the method (three-phase Primer/Planner/Finisher architecture with a retrieval-based memory), and the key results (SOTA ASR on o3 and Opus 4.1, improved efficiency).
- **Unsupported claims:** The abstract claims the framework is "inspired by lifelong-learning agents," but the mechanism described is essentially a growing vector database for cosine-similarity retrieval. "Lifelong learning" in machine learning typically implies continuous parameter updates, catastrophic forgetting management, or online adaptation without a static knowledge base. The abstract would be more precise if framed as "experience retrieval and incremental memory accumulation." Additionally, the claim of "lesser or comparable query budget" is partially accurate; Table 5 shows PLAGUE uses ~3.85 target calls (comparable to Crescendo's ~2.97 but notably more than GOAT's ~1.72), so the efficiency advantage should be qualified.

### Introduction & Motivation
- **Motivation & Gap:** The introduction effectively motivates multi-turn attacks by contrasting them with single-turn approaches and highlighting the lack of systematic frameworks that balance goal progression, feedback adaptation, and tactical diversity. The gap identification (existing methods trade off ASR, diversity, or compute, and lack memory/adaptability) is clear.
- **Contribution statement:** Contributions are clearly listed. The plug-and-play modularity claim is well-justified by showing how components like ActorBreaker's planner or Crescendo's finisher can be integrated.
- **Over-claiming/Under-selling:** The introduction states PLAGUE breaks through "hardest safety-aligned models with ease," which reads as hyperbole given the ~30-40% absolute improvements. ICLR values measured tone over absolutist language. The motivation itself is solid, but the framing should emphasize *systematic vulnerability discovery* rather than implying existing defenses are trivially bypassable.

### Method / Approach
- **Clarity & Reproducibility:** The three-phase architecture (Planner → Primer → Finisher) is logically structured and well-supported by Algorithms 1–4 and Appendix B prompts. The retrieval mechanism (Section 3.3.1) and rubric scoring (Section 3.2) are sufficiently detailed for reproduction, assuming code/prompts are released.
- **Assumptions & Justification:** Key assumptions include: (1) black-box API access, (2) fixed 6-turn budget, (3) reliance on a strong attacker LLM (Deepseek-R1) and separate evaluator LLMs. The choice of cosine similarity threshold (0.6) and retrieval limit (max 2 examples) is stated but not justified empirically. Is this threshold optimal, or was it chosen heuristically?
- **Logical gaps/Inconsistencies:** 
  - There is a direct contradiction between Section 3.5 and Algorithm 3. Section 3.5 states success requires a score `> 8/10`, but Algorithm 3 Line 10 uses `if score > 9.0`. Similarly, backtracking is triggered at `≤ 3/10` in the text but `≤ 2.0` in the algorithm. This threshold mismatch impacts reproducibility and must be reconciled.
  - The "frozen context" design in the Finisher phase is a strong architectural choice but lacks discussion. By freezing the Primer-built context and conditioning the Finisher solely on the initial objective, the framework sacrifices real-time conversational adaptability. Why does this outperform continuously adaptive approaches? A brief theoretical or empirical rationale would strengthen the design claim.
  - "Lifelong learning" implementation: The memory bank `R[+]` simply appends successful strategies. As the library scales, retrieval precision may degrade (noisy in-context examples). The method does not address memory pruning, deduplication, or concept drift (e.g., if a target model updates its safety filters, historical strategies may become obsolete or counterproductive).
- **Edge cases/Failure modes:** Not discussed. What happens when target LLMs enforce strict context windows, truncate history aggressively, or employ conversation-state tracking that detects frozen-vs-live context mismatches?

### Experiments & Results
- **Claim validation:** Table 2 shows PLAGUE outperforming baselines on OpenAI o3, o1, Deepseek-R1, and Llama 3.3-70B. For Claude Opus 4.1, the main result uses a GOAT Finisher (0.318 Bin-ASR), but the text highlights a 40.2% improvement using a swapped Crescendo Finisher (Table 4/6). The claim is valid but requires clearer presentation to avoid cherry-picking the best configuration per model without acknowledging the component swap.
- **Baseline fairness & configuration:** The authors explicitly modify GOAT to use mid-attack rubric scoring (Section 4), which effectively turns it into an agentic attack. This is a reasonable comparison for fairness but should be explicitly labeled as "GOAT (Agentic Variant)" in Table 2 to avoid confusion with the official implementation. X-Teaming and FITD are constrained to 6 turns and limited refinement steps, which is appropriate for controlled comparison.
- **Missing ablations:** 
  - **Memory scaling:** How does ASR change as `R[+]` grows from 10 to 50 to 200 entries? Without this, the "lifelong" claim remains untested.
  - **Threshold sensitivity:** Are the results robust to minor changes in the 7/10 and 8/10 scoring thresholds?
  - **Attacker LLM dependency:** All primary attacks use Deepseek-R1. Would a weaker or different architecture (e.g., Llama 3.1 70B or o1-mini) as the attacker degrade the framework's effectiveness significantly?
- **Statistical rigor:** Section 4 states scores are "averaged over three runs for robustness," but Tables 2, 4, and 6 report point estimates without error bars, standard deviations, or statistical tests. Given ICLR's emphasis on empirical reliability, variance metrics are necessary, especially since multi-turn attacks exhibit path-dependent stochasticity.
- **Diversity metric:** Equation 1 (Appendix C.2) defines embedding diversity. Figure 3 shows improvements with ActorBreaker's planner, but the y-axis labels and error bars are missing or unclear due to parsing. The text claims a "15.47% improvement" but does not specify if this is relative or absolute mean cosine similarity reduction.
- **Metric formulation:** Appendix C.1's SRE formula is partially garbled by the parser. Clarifying the exact scoring bounds and the `-2` normalization term is essential for reproducibility.

### Writing & Clarity
- **Confusing sections:** The backtracking mechanism (Section 3.4) states it "removes the turn from the Target’s conversation history while keeping it in the Attacker’s conversation history." How does this technically work over an API? Does it mean the target receives a revised prompt excluding the refused turn, or does the attacker internally retry without appending to the target's official thread? This operational detail is crucial for understanding why backtracking improves success rates.
- **Figures/Tables:** Table 2 contains a duplicated row for ActorBreaker. Table 5's formatting is heavily garbled, but the underlying data (LLM call counts) is interpretable. Figure 4's radar chart effectively demonstrates category-wise robustness. The algorithms are clean, but the pseudoclient inconsistencies (scores >9.0 vs >8/10) noted above must be corrected.
- **Overall prose:** The technical writing is generally clear and well-organized. The narrative flow from motivation → method → empirical validation is logical. No major phrasing issues impede scientific understanding.

### Limitations & Broader Impact
- **Acknowledged limitations:** The ethics statement correctly notes potential misuse but justifies open access for defensive research. The conclusion briefly acknowledges leaving "development of a better diversity-inducing Planner to future work."
- **Missed limitations:** 
  - **Compute/Financial cost:** While query counts are reported (Table 5), the framework invokes multiple strong LLMs per goal (Attacker + Evaluator + Rubric Scorer + Summarizer + Embedding model). The actual dollar cost per 200-goal run is not provided, which matters for practical red-teaming adoption.
  - **Memory drift & pruning:** As noted, the unbounded vector library is a practical limitation. The authors do not discuss retrieval latency, storage scaling, or strategies to filter out low-quality "successful" attacks.
  - **Defensive adaptability:** The framework targets static API endpoints. Real-world systems deploy dynamic defenses (rate limiting, conversation hashing, refusal consistency checks). The paper does not address how PLAGUE fares against stateful defenses.
- **Societal impact:** The paper focuses squarely on model vulnerability. It would benefit from a brief discussion on how defenders can operationalize the modular components (e.g., using the Planner for automated red-teaming pipelines or the Rubric Scorer for dynamic safety alignment training) to align with ICLR's interest in actionable AI safety research.

### Overall Assessment
This paper presents a well-structured, modular framework for multi-turn LLM jailbreaks that achieves strong empirical results across recent state-of-the-art models. The three-phase architecture (Planner/Primer/Finisher) logically disentangles attack components and provides useful insights into which mechanisms drive success on different models (e.g., reflection matters more for o3, backtracking for Claude). The work is clearly applicable to automated red-teaming and offers a practical, reproducible pipeline. However, to meet ICLR's rigor expectations, several issues must be addressed: the direct contradiction in success/backtracking thresholds between Section 3.5 and Algorithm 3 needs correction; the "lifelong learning" terminology should be reconciled with its actual implementation (cumulative vector retrieval) and validated with a memory-scaling ablation; statistical variance (error bars over 3 runs) is missing from all main tables; and the reliance on strong helper LLMs and unpruned memory constitutes practical limitations that should be explicitly discussed. With these clarifications and added statistical rigor, the paper will present a solid, impactful contribution to automated safety evaluation.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces PLAGUE, a modular, plug-and-play framework for automated multi-turn jailbreaking that decomposes attacks into Planner, Primer, and Finisher phases. By integrating lifelong strategy retrieval, context freezing, rubric-based reflection, and backtracking, the framework systematically explores and adapts conversational attack trajectories. Evaluations across frontier models (e.g., OpenAI o3, Claude Opus 4.1, DeepSeek-R1) demonstrate substantial ASR gains over existing multi-turn baselines while maintaining a comparable or lower query budget.

### Strengths
1. **Clear, Principled Modular Design:** The decomposition of multi-turn attacks into three distinct, interchangeable phases (Planner, Primer, Finisher) provides a highly interpretable and flexible architecture. The ablation in Table 3 cleanly demonstrates how each component (Backtracking, Reflection, Planning, Strategy Retrieval) contributes incrementally to ASR, making the framework adaptable for red-teamers with varying resource constraints.
2. **Rigorous Budget & Efficiency Analysis:** The authors carefully control the query budget (capped at 6 turns) and provide a detailed breakdown of LLM invocations per phase (Table 5). Reporting that PLAGUE achieves higher ASR while using roughly the same number of target calls as Crescendo (and only ~1 extra over GOAT) directly addresses a common criticism of agentic jailbreaking: excessive API costs.
3. **Strong Empirical Performance on Frontier Models:** Achieving 81.4% SRE on OpenAI o3 and 67.3% on Claude Opus 4.1 (Table 2 & 4) represents a meaningful red-teaming milestone. The category-wise breakdown (Figure 4) and cross-model analysis further validate that the framework generalizes across diverse threat taxonomies and model safety alignments.

### Weaknesses
1. **Asymmetric Baseline Constraints:** The fair comparison setup in Section 4 introduces restrictive modifications to baselines that may handicap them. For example, ActorBreaker is limited to `K=2` actors, AutoDAN-Turbo's lifelong iterations are capped at two, and GOAT's evaluation loop is altered. While controlled budgets are necessary, applying asymmetric constraints without demonstrating that they don't disproportionately degrade baseline performance weakens the SOTA claim.
2. **Missing Statistical Rigor & Variance Reporting:** Main results (Table 2) report point estimates without standard deviations, confidence intervals, or significance tests across multiple runs, despite claiming averages over three runs. Given the stochastic nature of LLM agentic workflows, it is difficult to assess whether the 30-40% improvements are statistically robust or within the variance frontier.
3. **Underexplored Memory Bank Scaling & Attacker Dependence:** The lifelong learning component relies on a vector database of successful strategies, but the paper does not analyze retrieval performance as the bank scales, potential noise/irrelevant retrieval over time, or memory management strategies. Additionally, results are primarily generated using DeepSeek-R1 as the attacker; there is no ablation on how framework performance changes with weaker or stronger attacker LLMs, making it unclear if gains stem from the architecture or the attacker's reasoning capacity.

### Novelty & Significance
**Novelty:** Moderate. The technical contribution lies in the systematic integration and orchestration of known agentic techniques (strategy retrieval, rubric scoring, context freezing, backtracking) into a cohesive, modular pipeline rather than introducing a fundamentally new algorithmic paradigm. The "lifelong learning" via goal-embedding similarity retrieval and the phase-based decomposition are well-executed extensions of prior agentic and multi-turn work (Crescendo, GOAT, AutoRedTeamer). 
**Significance:** High. Despite moderate theoretical novelty, the framework's practical impact is substantial. It provides the community with a standardized, reproducible red-teaming architecture that pushes the vulnerability frontier on currently well-aligned models. The component-wise ablation offers actionable insights into which defensive mechanisms are most sensitive to planning vs. context optimization, which can directly inform alignment research and safety evaluation pipelines.

### Suggestions for Improvement
1. **Standardize Baseline Evaluations & Report Variance:** Unbaseline constraints where possible, or provide a separate ablation table showing baseline performance under identical lifelong-learning and memory setups. Add standard deviations or 95% CIs to all main tables (Tables 2, 3, 4, 6) and explicitly discuss statistical significance.
2. **Analyze Memory Bank Dynamics & Attacker Sensitivity:** Include an experiment scaling the strategy retrieval library (e.g., 10 → 100 → 500 entries) to measure retrieval precision decay, inference overhead, and ASR impact. Additionally, report PLAGUE's performance across at least two different attacker models (e.g., GPT-4o and Llama-3.1-405B) to disentangle framework efficacy from attacker capability.
3. **Clarify Prompt Artifacts & Defensive Implications:** While prompts are referenced in the appendix, ensure all critical rubric, planner, and finisher templates are fully documented (the provided text contains placeholders). Expand the ethics/limitations section to discuss how defenders could detect PLAGUE-style patterns (e.g., context-freezing signatures, retrieval-based strategy reuse) and propose concrete mitigation strategies aligned with the framework's insights.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Sequential Lifelong Learning Evaluation:** Run PLAGUE across sequential batches of goals and track ASR vs. memory-batch number to prove the system actually accumulates knowledge over time; without this, the "lifelong learning" claim is reduced to static few-shot prompting.
2. **Multi-Turn Defense Testing:** Evaluate against standard context-aware safety filters (e.g., LLM-as-a-guard, constitutional refusal modules, or system-prompt hardening) to demonstrate the framework's utility for real-world red-teaming rather than just unhardened API endpoints.
3. **Attacker Model Sensitivity:** Replicate the main Table 2 results with at least one alternative attacker (e.g., GPT-4o or Claude Sonnet); relying solely on DeepSeek-R1 risks proving framework-attacker synergy rather than a generalizable plug-and-play architecture.

### Deeper Analysis Needed (top 3-5 only)
1. **Retrieval Precision & Failure Modes:** Quantify the retrieval hit-rate and correlation between embedding similarity (threshold 0.6) and actual ASR gains; without reporting when retrieval retrieves irrelevant strategies or fails on novel categories, the memory module's contribution is unsubstantiated.
2. **Component Interaction/Factorial Ablation:** Replace the sequential add-on ablation with a factorial design (e.g., Planner ± Backtracking ± Reflection ± RSS) to prove components provide orthogonal value and aren't merely compensating for a weak Finisher or over-regularizing with feedback.
3. **Budget Accounting & Score Inflation Analysis:** Rigorously define how backtracked turns consume the strict 6-turn Target LLM budget and analyze whether high ASR stems from genuine context optimization or from selectively reporting the max score across implicit retries; without this, efficiency claims are unverifiable.

### Visualizations & Case Studies
1. **Turn-by-Turn Score Trajectories:** Plot the Rubric Scorer's score progression per conversation turn for successful vs. failed attacks to empirically validate the claimed "gradual escalation" and "drift prevention" rather than relying on architectural assertions.
2. **Concrete Success/Failure Dialogue Walkthroughs:** Provide fully rendered multi-turn transcripts showing exactly when backtracking triggers, how reflection modifies the next prompt, and how context freezing enables the Finisher to prove the proposed mechanism functions in practice.
3. **Embedding Retrieval Clustering:** Visualize the cosine similarity distribution between goal embeddings and retrieved strategies using t-SNE/UMAP to demonstrate whether the memory bank clusters semantically harmful objectives or merely retrieves superficially similar templates.

### Obvious Next Steps
1. **Implement True Online Memory Updates:** Replace the static/pre-seeded retrieval with an online learning loop that logs newly discovered strategies, embeds them immediately, and measures their impact on subsequent campaign performance, which is required to justify "lifelong" in ICLR.
2. **Standardize & Statistically Validate Metrics:** Report Bin-ASR and SRE consistently across all tables with confidence intervals or significance tests, resolving the ambiguous instruction to use them "interchangeably" and proving the 30–40% gains are not artifacts of stochastic LLM variance.
3. **Compute-to-ASR Efficiency Analysis:** Formalize a cost-vs-performance metric that accounts for total LLM invocations (Attacker + Scorer + Target) per successful breach, directly comparing PLAGUE to baselines to prove higher ASR isn't achieved through disproportionate hidden inference costs.

# Final Consolidated Review
## Summary
PLAGUE introduces a modular, three-phase framework (Planner, Primer, Finisher) for automated multi-turn LLM jailbreaking, integrating strategy retrieval, rubric-based reflection, and context freezing. It reports state-of-the-art attack success rates (ASR) on several frontier models while claiming comparable efficiency to existing multi-turn baselines. While the architecture cleanly disentangles attack components, critical methodological inconsistencies and a lack of statistical rigor hinder reproducibility and obscure whether the reported gains stem from genuine lifelong adaptation or from asymmetric evaluation configurations.

## Strengths
- **Principled Modular Architecture:** The explicit decomposition of multi-turn attacks into interchangeable phases enables clear component swapping and systematic ablation. Table 3 effectively isolates how backtracking, reflection, planning, and strategy retrieval each contribute to downstream ASR, providing red-teamers with a flexible, interpretable pipeline.
- **Detailed Efficiency Accounting:** The paper rigorously tracks Target, Evaluator, and Planner LLM invocations under a fixed 6-turn budget. Figure 2 and Table 5 demonstrate that PLAGUE achieves significant ASR gains while maintaining target call counts roughly on par with or slightly below leading multi-turn baselines, directly addressing the compute overhead concerns typical of agentic red-teaming.

## Weaknesses
- **Critical Text-to-Algorithm Inconsistencies:** Direct contradictions exist between the main text and the provided pseudocode regarding core operational thresholds. The success criterion is defined as `> 8/10` in Section 3.5 but implemented as `> 9.0` in Algorithm 3. Backtracking is triggered at `< 3/10` in the prose but `<= 2.0` in the algorithm. These discrepancies fundamentally compromise reproducibility and make it impossible to verify how the strict turn budget is actually consumed during reflection loops.
- **Unsubstantiated "Lifelong Learning" Claims & Single-Attacker Dependency:** The framework labels its cumulative vector database retrieval as "lifelong learning," yet provides zero empirical validation of knowledge accumulation over sequential campaigns. There is no analysis of memory scaling, retrieval precision decay, or latency overhead as the strategy bank grows. Furthermore, all attacks rely exclusively on Deepseek-R1 as the attacker model. Without cross-attacker ablations, it remains unclear whether the framework's modularity genuinely generalizes or is heavily coupled to the specific reasoning and instruction-following capabilities of a single frontier model.
- **Absence of Statistical Rigor & Asymmetric Baseline Constraints:** Despite stating results are "averaged over three runs," all primary tables report only point estimates, entirely omitting standard deviations, confidence intervals, or significance testing. Given the high path-dependent stochasticity of multi-turn agentic workflows, the claimed 30–40% ASR improvements cannot be validated. Compounding this, several baselines are subjected to asymmetric constraints (e.g., capping ActorBreaker at `K=2`, manually injecting agentic scoring into GOAT), and the paper does not prove these modifications do not artificially handicap the comparison, inflating PLAGUE's relative standing.

## Nice-to-Haves
- Plot turn-by-turn Rubric Scorer trajectories for successful vs. failed attacks to empirically validate the claimed "gradual escalation" and "drift prevention" mechanisms, rather than relying on architectural assertions.
- Clarify the exact API-level mechanics of the backtracking step (e.g., whether it entails reconstructing the context window by omitting specific message turns, and how this interacts with rate limits or stateful defense logs).

## Novel Insights
The most substantive contribution of this work lies not in its algorithmic primitives, but in its empirical deconstruction of how conversational pacing and iterative feedback interact with model-specific safety alignments. The finding that different victim models exhibit distinct vulnerability profiles—e.g., OpenAI o3 responds most strongly to rubric-based reflection, whereas Claude Opus 4.1 is highly sensitive to backtracking and context freezing—provides valuable, actionable intelligence for the safety community. This component-level sensitivity analysis moves automated red-teaming beyond monolithic success rate reporting and toward a more mechanistic understanding of how planning initialization, constraint enforcement, and adaptive query sampling differentially bypass static versus dynamic refusal boundaries.

## Potentially Missed Related Work
- **ExpEL (Zhao et al., 2024) & Reflexion (Shinn et al., 2023):** While cited, the paper could more rigorously contrast PLAGUE's static vector retrieval with recent dynamic memory pruning and experience-decay methods tailored for LLM agents, which are highly relevant to preventing retrieval noise in growing jailbreak strategy banks.
- **AutoDAN-Turbo (Liu et al., 2025):** The paper notes AutoDAN-Turbo's lifelong component but heavily downplays its self-exploration mechanism. A tighter comparison contrasting PLAGUE's goal-embedding similarity retrieval with AutoDAN-Turbo's response-embedding strategy mapping would sharpen the technical contribution.

## Suggestions
1. **Reconcile Thresholds & Formalize Budget Accounting:** Explicitly align the success/backtracking thresholds between the text and algorithms, and provide a precise accounting formula detailing how backtracked/retired turns deduct from the fixed 6-turn Target LLM budget.
2. **Report Variance & Equalize Baseline Configurations:** Add standard deviations or 95% confidence intervals to Tables 2, 3, and 6. Either remove asymmetric caps on baseline attackers (ActorBreaker, AutoDAN-Turbo) or include a supplementary table showing baseline performance under identical memory and reflection plumbing to validate the SOTA claim.
3. **Demonstrate True Longitudinal Learning:** Conduct a sequential campaign experiment that incrementally populates the strategy memory bank across multiple batches of HarmBench objectives, reporting retrieval precision, ASR trajectory, and inference latency over time to rigorously validate the "lifelong" framing.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Accept
