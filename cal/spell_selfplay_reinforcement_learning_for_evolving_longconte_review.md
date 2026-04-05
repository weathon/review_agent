=== CALIBRATION EXAMPLE 69 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately captures the core contribution: a self-play RL framework targeting long-context LLM evolution.
- The abstract clearly outlines the motivation (lack of verifiable rewards/annotations for long-context), the method (three-role cyclical self-play with curriculum and adaptive rewards), and key results (consistent gains across 12 models, test-time scaling, outperforming static RLVR).
- One claim requires tightening: *"outperforms equally sized models fine-tuned on large-scale annotated data"*. Table 1 compares SPELL-trained base models to *instruction-tuned* counterparts. While instruction-tuning involves annotated data, these models are typically aligned on general conversational/instructional corpora, not specifically large-scale long-context QA datasets. The phrasing slightly overstates the baseline's specialization. Clarifying this comparison would improve precision.

### Introduction & Motivation
- The problem is well-motivated. The authors correctly identify that extending RLVR to long-context reasoning is bottlenecked by annotation costs, lack of programmatic verifiers, and human performance degradation on complex long-context tasks.
- The gap in prior work is clearly identified: static RLVR datasets fail to adapt as the policy improves, and single-agent self-play lacks reliable semantic verification for open-ended long-context answers.
- Contributions are explicitly listed and match the methodological and empirical sections.
- The introduction appropriately scopes the work and does not under-sell. The claim about models approaching "superhuman reasoning" is standard contextual framing and does not distract from the technical focus.

### Method / Approach
- The method is conceptually clear and Algorithm 1 provides a reproducible high-level loop. Prompt templates and hyperparameters are provided in the appendices.
- **Assumption & Justification Gap 1 (Curriculum Claim vs. Implementation):** The abstract and introduction state that SPELL uses an *"automated curriculum that gradually increases document length"*. However, Section 3.1 and Algorithm 1 describe a curriculum driven by (1) expanding the *number* of documents/clusters via history memory, and (2) conditioning on past QA pairs to increase complexity. There is no explicit mechanism that schedules or increases token length during training. If length growth is purely emergent from cluster sampling, it should be stated as such rather than framed as an explicit curriculum component.
- **Assumption & Justification Gap 2 (Gradient Balancing in Eq. 10):** The unified policy update simply sums the GRPO objectives for the three roles: $J_{GRPO} = J_{que} + J_{res} + J_{ver}$. The reward scales differ fundamentally: the verifier uses binary self-consistency scores (Eq. 5), the responder uses max(CEM, verifier) (Eq. 6), and the questioner uses a Gaussian mapping (Eq. 7). GRPO normalizes advantages within groups/batches (Eq. 3, 8, 9), but summing these objectives assumes comparable gradient magnitudes across roles. Without explicit loss weighting or gradient clipping normalization, the questioner or verifier could inadvertently dominate parameter updates. The role-specific dynamic sampling in Sec 3.3 addresses sample count imbalance, but not per-sample gradient scale imbalance.
- **Clarity Gap in Sec 3.3 (Dynamic Sampling for Questioner):** The text states: *"retain all groups with non-zero reward variance... associated questions are labeled as positives... equal number of negatives are drawn from questions with non-positive reward"*. Since the questioner generates only one output per prompt, it lacks intra-prompt group variance. The sampling strategy appears to operate across prompts in a batch, but the interface between this sampling and the batch-level advantage estimation (Eq. 9) is under-specified. A concrete example of batch construction would greatly improve reproducibility.

### Experiments & Results
- The experiments thoroughly test the core claims across 12 models, 6 benchmarks, and two context lengths. Comparisons to RLVR on static synthesized data, instruction-tuned baselines, and alignment methods (SoLoPO, LongPO, QwenLong-L1) are appropriate.
- **Missing Statistical Reporting:** The paper reports averages over eight independent runs (Sec 4.1), but **no standard deviations, confidence intervals, or significance tests** are provided in any table or figure. Given that RL training exhibits high variance and some gains on stronger models are marginal (e.g., Qwen3-30B-A3B-Thinking gains +2.0 avg), ICLR reviewers typically expect error bars or at least variance tables to confirm that improvements are statistically robust and not artifacts of run-to-run fluctuations.
- **Ablation Completeness:** Table 2 effectively ablates key components of the questioner and verifier. However, an ablation of the *role-specific dynamic sampling* vs. uniform/random sampling is missing. Since Sec 3.3 claims this strategy is critical for preventing gradient dominance and reducing compute, demonstrating its necessity against a standard GRPO batch construction would strengthen the ablation.
- **Evaluation Metric Transparency:** Scoring uses $\max(\text{CEM}, \text{LLM Judge})$. While this is practical for long-context QA, it masks how much of the gain comes from strict exact match vs. semantic equivalence. Reporting CEM and LLM-Judge scores separately would clarify whether SPELL primarily improves grounding/precision or paraphrasing capability.
- Results are not cherry-picked. The authors honestly report cases where RLVR underperforms or stagnates, which supports the motivation for dynamic curricula.

### Writing & Clarity
- The paper is generally well-structured and readable.
- Section 3.3 requires refinement for clarity, specifically regarding how the questioner's per-instance reward (Eq. 7) maps to the batch-normalized advantage (Eq. 9) under the proposed dynamic sampling scheme. Clarifying the batch construction pipeline and how positives/negatives are balanced across the three roles before feeding them into the joint GRPO objective would significantly aid reproducibility.
- Figure 4 and Figure 5 are informative and directly support claims about curriculum progression and reward function stability. Figure 3 effectively illustrates test-time scaling. No major clarity issues impede understanding of the core contribution.

### Limitations & Broader Impact
- The authors acknowledge key limitations: lack of theoretical analysis of co-evolutionary dynamics, training context capped at 16K, and reliance on manual prompt construction.
- **Missed Limitations:** 
  1. *Long-term Verifier Drift:* While Eq. 7 and consistency checks mitigate immediate self-delusion, the paper does not discuss what happens over extended training horizons (e.g., >100 steps). If the verifier slowly misaligns with ground truth on semantic tasks, will it eventually corrupt the questioner's difficulty calibration?
  2. *Compute & Scaling Context:* Appendix F.1 reports ~8 hours of training on 8x A100s, but the total number of training steps and tokens consumed is not explicitly stated. For ICLR, a clearer FLOPs or step count is necessary to contextualize the efficiency claims and compare against standard SFT/RLVR pipelines.
- Broader impact is addressed appropriately. The ethics statement notes potential bias amplification, which is standard and acceptable. No major unaddressed negative societal impacts are evident.

### Overall Assessment
SPELL presents a compelling and timely approach to long-context reasoning by extending self-play RL to domains lacking programmatic verifiers. The three-role cyclical design, verifier self-consistency mechanism, and dynamic curriculum address genuine bottlenecks in current RLVR pipelines, and the empirical evaluation across 12 models and 6 benchmarks is comprehensive. However, to meet ICLR's rigorous standards, the paper must address several technical and reporting gaps: (1) align the "document length curriculum" claim with the actual implementation or clarify it as an emergent property of memory expansion; (2) justify the unified GRPO objective (Eq. 10) given differing reward scales across roles, possibly adding gradient normalization or loss weights; (3) clarify the batch construction and advantage estimation pipeline in Sec 3.3; (4) report statistical variance/error bars across the 8 runs to validate marginal gains on strong models; and (5) provide CEM vs. LLM-Judge score breakdowns to isolate grounding improvements from paraphrasing effects. The contribution is strong and likely viable after these clarifications, but the current manuscript leaves some methodological details ambiguous and lacks the statistical rigor expected for RL claims at top venues.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces SPELL, a multi-role self-play reinforcement learning framework designed to enhance long-context reasoning in LLMs without human supervision. By having a single policy alternate between questioner, responder, and verifier roles, coupled with an automated curriculum and a self-consistent verifier reward, the framework enables autonomous, data-efficient optimization. Extensive experiments across 12 diverse models and six long-context benchmarks demonstrate consistent performance gains over static RLVR baselines and strong generalization to longer contexts.

### Strengths
1. **Rigorous and Broad Empirical Validation:** The evaluation covers 12 open-source LLMs spanning base, instruction-tuned, reasoning, dense, and MoE architectures (Table 1). The consistent improvements across model families and sizes, alongside the demonstration that a trained base model can outperform its human-annotated instruct counterpart, strongly validate the data efficiency and universality of the approach. The pass@k analysis showing superior test-time scaling and beating `gemini-2.5-pro` at pass@4 (Figure 3) further underscores practical significance.
2. **Well-Designed Self-Play Mechanics:** The framework thoughtfully addresses known failure modes in self-play RL. The history memory combined with a Gaussian difficulty reward (Eq. 7) effectively implements an automatic curriculum targeting the responder's competence frontier. The verifier’s dual reward mechanism (max of rule-based CEM and majority-voted semantic equivalence, Eq. 6) and consistency alignment (Section 4.3, Table 2) provide a robust solution to the brittleness of exact match rewards in open-ended long-context QA.
3. **Strong Generalization & Efficiency Claims:** The paper demonstrates out-of-distribution generalization by training at 16K and evaluating at 100K without fine-tuning, showing that the learned reasoning skills transfer across context lengths. Additionally, the explicit addressing of reward hacking risks (Appendix F.6) and the detailed role-specific dynamic sampling strategy (Section 3.3) show careful engineering consideration for stable RL optimization.

### Weaknesses
1. **Limited Training Context Length:** While evaluation extends to 100K tokens, training is strictly capped at 16K. The paper acknowledges this limitation (Appendix B), but it restricts the claim of "evolving long-context language models" to mid-length contexts. The mechanistic explanation for *why* 16K-trained reasoning transfers so effectively to 100K (e.g., position embedding extrapolation, attention pattern preservation) lacks detailed analysis, leaving a gap between empirical observation and theoretical understanding.
2. **Heuristic Nature of Reward Calibration:** The Gaussian reward mapping (Eq. 7) and its σ parameter are well-motivated and empirically tuned, but lack theoretical grounding. The paper notes in Appendix B that a theoretical framework for tri-role co-evolution is missing. Without even a preliminary convergence analysis or stability bounds (common expectations for novel RL frameworks at ICLR), the success of the curriculum design relies heavily on empirical tuning rather than principled optimization guarantees.
3. **Compute & Data Transparency Gaps:** The reported training cost of ~8 hours on 8× A100 GPUs (Section F.1) appears surprisingly low for an on-policy RL algorithm with G=8 rollouts across three roles on 16K context. While verifier rollouts are shorter, a breakdown of total GPU-hours, number of training steps, and exact dataset size used per step would improve reproducibility. Additionally, the exact prompt templates for role conditioning (Appendix G) are referenced but not fully detailed in the main text, making it difficult to assess how much of the performance stems from prompt engineering versus the RL loop itself.

### Novelty & Significance
The novelty lies in successfully adapting the single-model self-play paradigm, previously confined to short-context, program-verifiable domains (e.g., math, code), to open-ended long-context reasoning. The integration of a self-consistent verifier to replace brittle exact-match rewards and the dynamic, history-aware curriculum represent meaningful architectural contributions. Given ICLR's emphasis on scalable, data-efficient learning and alignment post-training, this work is highly significant. It provides a concrete pathway toward reducing reliance on costly human curation for long-context capabilities, aligning well with current trends in autonomous LLM optimization.

### Suggestions for Improvement
1. **Expand Context-Length Analysis:** Include an ablation or case study analyzing *why* 16K-trained models generalize to 100K. Quantify attention retrieval patterns, positional interpolation behavior, or provide a few error-analyzed examples showing how the model handles distractors beyond the training window. If training at >16K is computationally prohibitive, explicitly discuss how the curriculum could be extended (e.g., progressive context scaling) to bridge this gap.
2. **Strengthen Verifier & Curriculum Theoretical Grounding:** While empirical results are strong, ICLR reviewers appreciate even minimal analytical insights. Consider adding a formalization of the Gaussian reward's effect on gradient variance or referencing recent work on curriculum learning in self-play to justify why targeting ȓ_res = 0.5 maximizes policy entropy. Additionally, report verifier calibration metrics (e.g., Brier score or reliability diagrams) over training steps to quantitatively prove that self-consistency avoids echo-chamber reward hacking.
3. **Clarify Compute Efficiency & Prompt Dependence:** Provide exact GPU-hour estimates, total training steps, and batch composition per update. Clearly separate the performance gains attributable to the RL optimization loop versus zero-shot prompt templates (e.g., by running a controlled SFT baseline with the exact same prompt structure). This will strengthen the reproducibility claim and ensure reviewers can accurately benchmark the computational trade-offs against static RLVR methods.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Train and evaluate against leading self-evolution baselines (AZR, R-Zero, SQLM) on identical long-context QA benchmarks; without a direct empirical comparison, the claimed advantage of a three-role architecture over existing single/dual-role loops is unsubstantiated.
2. Run a Supervised Fine-Tuning (SFT) baseline on the exact self-generated data produced by SPELL; without it, you cannot prove GRPO provides unique optimization benefits over simple data curation, undermining the RL contribution.
3. Extend training trajectories to 200+ steps to test long-horizon stability; self-play systems notoriously collapse or plateau after initial gains, and 70 steps are insufficient to validate the "continual self-improvement" claim.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the hallucination/fabrication rate of the questioner’s reference answers across training steps; false reference answers corrupt the entire self-play loop, so the core claim of reliable label-free rewards fails without this metric.
2. Evaluate the verifier’s standalone calibration against a strong external LLM or human sample at multiple checkpoints; assuming the verifier improves without external validation leaves the reward signal's fidelity entirely unproven.
3. Analyze the mechanism driving 16K→100K generalization (e.g., retrieval success rates, positional embedding interpolation errors, attention sparsity); without mechanistic evidence, the claim that reasoning skills extrapolate to longer contexts risks being an artifact of benchmark construction.

### Visualizations & Case Studies
1. Show concrete QA triplets (context, generated question, responder output, verifier score) from early, mid, and late training to prove the curriculum actually produces structurally more complex multi-hop questions rather than lexical paraphrases.
2. Provide failure cases where the verifier assigns high rewards to factually incorrect but lexically similar answers, directly exposing the brittleness of semantic equivalence checking without factual grounding.
3. Plot the empirical distribution of responder success rates against the theoretical Gaussian target across iterations; this confirms whether the questioner reliably stays near the 0.5 competence frontier or drifts toward trivial/unsolvable extremes.

### Obvious Next Steps
1. Implement a two-role baseline (questioner-responder with an external LLM judge) to isolate whether the computational overhead of training and updating a third role is strictly necessary for performance gains.
2. Report compute-to-performance trade-offs (GPU hours, token throughput, sample efficiency) explicitly compared to SFT and static RLVR; ICLR acceptance requires demonstrating that label-free self-play is not prohibitively expensive for marginal gains.
3. Evaluate cross-domain zero-shot generalization on non-web/financial corpora (e.g., biomedical, legal, scientific literature) to prove the evolved reasoning capabilities are architecture-agnostic rather than overfit to the Ultra-Fineweb/DocMath distribution.

# Final Consolidated Review
## Summary
SPELL proposes a multi-role self-play reinforcement learning framework for long-context reasoning, where a single LLM cyclically acts as a questioner, responder, and semantic verifier. By coupling an automated history-driven curriculum with a self-consistent verifier and GRPO-based optimization, the framework enables label-free policy improvement, demonstrating consistent gains across 12 diverse LLMs and strong out-of-distribution generalization from 16K to 100K context lengths.

## Strengths
- Comprehensive empirical validation across 12 open-source LLMs (dense, MoE, base, instruct, reasoning) and 6 long-context benchmarks, consistently outperforming static RLVR baselines and instruction-tuned counterparts.
- Well-engineered self-play loop that addresses known self-play failure modes: the history-aware curriculum effectively targets the policy’s competence frontier, while the verifier’s majority-voting and consistency alignment mitigate the brittleness of rule-based exact-match rewards in open-ended QA.
- Demonstrates strong test-time scaling (steep pass@k improvement curves) with a trained model surpassing `gemini-2.5-pro` at pass@4, highlighting practical potential for raising reasoning ceilings without human annotation or static data synthesis.

## Weaknesses
- **Missing critical empirical controls**: The paper positions RL as essential for evolving long-context reasoning but omits a supervised fine-tuning (SFT) baseline trained on the exact self-generated QA trajectories. Without it, it is impossible to disentangle GRPO’s optimization advantages from mere high-quality data curation. Additionally, direct empirical comparisons to established self-play frameworks (e.g., AZR, R-Zero) and an explicit two-role ablation (replacing the internal verifier with a strong external judge) are absent, leaving architectural necessity claims under-substantiated.
- **Insufficient statistical reporting and training transparency**: Results are averaged over eight runs yet lack standard deviations, confidence intervals, or significance testing. Given RL’s inherent variance and the marginal gains on stronger models (e.g., +2.0 avg for Qwen3-30B), this omision violates standard reproducibility expectations. Furthermore, compute metrics are vague; stating "~8 hours on 8× A100s" without explicit step counts, token throughput, or batch composition prevents rigorous cost-benefit analysis against static RLVR pipelines.
- **Unverified long-horizon stability and verifier calibration**: Self-play systems are highly susceptible to reward hacking and policy collapse. The paper only visualizes ~70 training steps, which is insufficient to validate claims of "continual self-improvement." Critically, while Fig 8b tracks verifier-to-rule disagreement, it never quantifies the verifier’s absolute calibration against an external LLM or human judgment on open-ended semantic tasks, leaving the long-term fidelity of the self-rewarding signal unproven.
- **Methodological misalignments and optimization ambiguities**: The abstract and introduction claim an automated curriculum that "gradually increases document length," but Sec 3.1 and Algorithm 1 reveal that difficulty scales via expanding document clusters and history memory conditioning, not explicit token-length scheduling. More technically, Eq. 10 sums three GRPO objectives with fundamentally different reward distributions (binary verifier, max(CEM/semantic) responder, Gaussian questioner) without addressing gradient magnitude balancing or role weight scheduling, risking unintended gradient dominance despite per-batch advantage normalization.

## Nice-to-Haves
- Provide theoretical or empirical justification for the Gaussian reward mapping (beyond empirical σ tuning) and formally analyze why the 16K-trained reasoning skills generalize so robustly to 100K contexts (e.g., attention sparsity analysis, positional interpolation behavior, or retrieval success metrics).
- Include qualitative trajectories showing concrete question complexity evolution and explicit failure cases where the verifier incorrectly rewards factually grounded but semantically divergent answers.
- Evaluate cross-domain zero-shot generalization on non-web/financial corpora (e.g., legal, biomedical, scientific literature) to verify that the evolved reasoning is distribution-agnostic.

## Novel Insights
The core novelty of SPELL lies in its tightly coupled tri-role architecture that transforms self-play from a fragile, echo-chamber process into a stable co-evolutionary system. By explicitly training the verifier to self-calibrate using rule-verifiable tasks before deploying it on open-ended semantic judgments, and by dynamically sampling training instances to prevent role imbalance, the framework effectively decouples curriculum generation from policy optimization. This demonstrates that reliable, label-free long-context reasoning can be sustained through internal role specialization and self-consistency rather than relying on external judges or static, pre-synthesized datasets.

## Potentially Missed Related Work
- Recent work on self-play stability, mutual training dynamics, and entropy-regularized curriculum learning (e.g., studies on reward collapse in self-rewarding models, online difficulty filtering for RL) would strengthen the theoretical framing of the Gaussian curriculum design and gradient balancing strategy. These are supplementary, not critical omissions.

## Suggestions
- Introduce a strict SFT baseline trained on the exact self-play rollouts to isolate optimization benefits of GRPO over data synthesis quality.
- Report full variance/error bars across all 8 runs, specify exact training steps/total tokens consumed, and break down final metrics into Cover Exact Match vs. Semantic Judge components to transparently isolate precision from paraphrasing gains.
- Extend training trajectories or run a long-horizon stress test (>150 steps) to empirically rule out verifier drift, and validate verifier calibration against an external LLM at multiple checkpoints using reliability diagrams or Brier scores.
- Clarify the curriculum mechanism in the abstract to accurately reflect cluster/memory expansion rather than explicit length scheduling, and provide a mathematical or empirical justification for how the unified GRPO objective (Eq. 10) maintains stable gradient dynamics across roles with distinct reward scales.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 8.0]
Average score: 6.0
Binary outcome: Accept
