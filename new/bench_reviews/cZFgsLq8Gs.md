## Summary

DeepScientist presents an LLM-based multi-agent system that frames autonomous scientific discovery as a Bayesian Optimization problem over a persistent Findings Memory, using a three-stage loop (hypothesize → implement/verify → analyze/report) with a UCB-inspired acquisition function to balance exploration and exploitation. Running on 16 H800 GPUs over month-long timelines, the system generates ~5,000 ideas, validates ~1,100, and ultimately produces methods that improve over published human SOTA on three frontier AI tasks: Agent Failure Attribution (+183.7% accuracy), LLM Inference Acceleration (+1.9% tokens/sec), and AI Text Detection (+7.9% AUROC).

## Strengths

- **Large-scale empirical demonstration**: Running 20,000 GPU hours, generating ~5,000 ideas with ~1,100 validated and 21 progress findings, this is one of the most substantial empirical efforts in automated scientific discovery to date. The transparent reporting of the massive exploration funnel (0.4% success rate) and that 60% of failures stem from implementation errors provides genuinely valuable community knowledge about the realities of autonomous research systems.

- **Progressive discovery trajectory**: The AI Text Detection sequence (T-Detect → TDT → PA-TDT) demonstrates that the system can build on its own prior findings rather than merely randomly searching, which is a concrete demonstration of iterative frontier-pushing. This progressive behavior is visible in the semantic clustering (Figure 5) and the conceptual shift from global statistics to non-stationary time-frequency analysis.

- **Findings Memory and three-stage pipeline**: The conceptual framework of Idea → Implement → Progress findings with promotion/demotion criteria, coupled with a persistent knowledge store that accumulates both successes and failures, is a meaningful architectural advance over one-shot AI Scientist pipelines. The system is explicitly designed to learn from failed experiments, which is a genuine design contribution.

- **Honest failure analysis**: The paper transparently reports that only 21 of ~5,000 ideas led to progress and that the naive approach of testing all candidates would have required 100,000+ GPU hours versus the 20,000 used. This transparency about the realities of automated discovery is a genuine contribution.

- **Dual evaluation protocol**: Combining automated review against 28 papers from other AI Scientist systems (Table 2, 60% acceptance rate — the only non-zero rate) with human expert review by ICLR-level reviewers (Table 3, inter-rater α = 0.739), including comparison to ICLR 2025 submission averages, provides a more credible quality assessment than most prior AI Scientist evaluations.

- **Responsible discussion of limitations and ethics**: The explicit acknowledgment of application boundaries for high-cost feedback loops, the red-teaming documentation, and the deliberate withholding of the "Analyze & Report" module from open-source demonstrate genuine ethical consideration.

## Weaknesses

### Major

- **Overclaimed "SOTA-surpassing" results with thin evaluation**: The paper's central claim is that DeepScientist "surpasses human-designed SOTA methods," but the evidence is substantially weaker than the rhetoric:
  - Only one baseline is compared per task. For LLM Inference Acceleration, only Token Recycling is compared despite a very active speculative decoding literature. For AI Text Detection, only Binoculars and Fast-DetectGPT are discussed as baselines. In each case, there are other strong methods that should be compared against.
  - The 1.9% improvement on inference acceleration (190.25→193.90 tokens/sec) is within typical noise from caching, batching, and implementation differences. Without confidence intervals, repeated trials, or careful matched evaluation conditions, this is not convincing evidence of a genuine scientific advance.
  - The 183.7% improvement on Agent Failure Attribution is impressive in percentage terms but starts from baselines of only 12.07%/16.67% accuracy, making percentage inflation unavoidable. The absolute accuracy of 29.31%/47.46% still leaves the task largely unsolved, which the paper does not discuss.
  - No statistical tests, no error bars, no repeated runs, no cross-dataset evaluation, and no sensitivity analyses are reported for any task. This is below the standard for making definitive claims of surpassing human SOTA.

- **Missing ablations of core architectural components**: The paper attributes success to the Bayesian Optimization framing, Findings Memory, and three-stage pipeline, yet provides no ablation testing whether these components are necessary. The only comparison is against "random sampling of 100 ideas" yielding zero success, which does not isolate any individual component. Specifically missing: (a) ablation of Findings Memory vs. no memory; (b) ablation of surrogate scoring + UCB vs. simpler selection (e.g., greedy utility); (c) ablation of the three-stage promotion system vs. flat pipeline. The system uses Gemini-2.5-Pro and Claude-4-Opus — among the most capable models available. Without separating framework contribution from raw LLM capability, it is unclear whether the architectural design or the frontier models are driving the results.

- **Bayesian Optimization framing is overstated**: The system is described as casting discovery as "Bayesian Optimization" with a surrogate model and UCB acquisition. In reality, the "surrogate" is an LLM producing three integer scores (0–100) via prompting, with no calibration, no predictive uncertainty quantification, no training procedure, and hyperparameters fixed uniformly (wu = wq = κ = 1) without tuning. The "exploration term" ve is an LLM-generated integer, not a statistically grounded posterior variance. The paper itself acknowledges this is "a simple, task-agnostic configuration," but the Bayesian Optimization and UCB terminology strongly implies a principled probabilistic framework that does not exist here. This should be honestly reframed as LLM-guided heuristic selection inspired by BO principles, or the paper should provide evidence that the surrogate scores actually correlate with experimental outcomes.

### Minor

- **"Three years of human research in two weeks" comparison is rhetorical, not rigorous**: Figure 1 compares progress on RAID against a timeline of human-published methods, but this is not a controlled or normalized comparison. There is no accounting for differences in compute, model access, or experimental budgets. The comparison is visually compelling but methodologically unsound as a basis for the strong claim in the abstract.

- **Scaling law claim is based on insufficient evidence**: The "near-linear relationship between resources allocated and discoveries" (Section 4.3, Figure 6) is based on 4 data points (1, 4, 8, 16 GPUs) from a single one-week experiment, with no repeated runs, no error bars, and no comparison against a trivial baseline (N independent 1-GPU runs summed). Calling this a "scaling law" significantly overclaims what the evidence supports.

- **Human supervision extent is underspecified**: The paper mentions "Three human experts supervise the process to verify outputs and filter out hallucinations" (Section 4), which is at odds with claims of "fully autonomous" discovery. The number of human interventions, hours of supervision, and fraction of outputs filtered are not quantified. This is critical for evaluating the autonomy claim.

- **Task domain limitation**: All three evaluation tasks are within AI/ML, where LLMs have strong inherent knowledge from training data. The paper acknowledges this limitation for "high-cost" domains but does not discuss how the approach generalizes to domains where LLMs lack deep domain expertise, which limits the "scientific discovery" claim's breadth.

### Trivial

- The human evaluation program committee consists of only 3 reviewers for 5 papers. While the inter-rater reliability (α = 0.739) is acceptable, some individual papers show high variance (PA-TDT and ACRA have rating variance of 1.33 and 1.00 respectively), making per-paper assessments less reliable.

- The ethical red-teaming focused only on computer virus generation, which represents a narrow slice of the dual-use risk surface for a system with code execution and internet access capabilities.

## Nice-to-Haves

- Ablation experiments isolating the contributions of Findings Memory, surrogate scoring, and UCB selection, which would dramatically strengthen the paper and are the single most impactful addition possible.
- Comparison against a human-in-the-loop baseline where a human researcher is given the same LLM coding assistants and compute budget, to properly contextualize the "surpassing human" claims.
- A scatter plot of surrogate predicted value vs. actual experimental outcome for all ~1,100 validated ideas, which would directly validate or invalidate the BO-inspired selection mechanism.
- Evaluation on at least one non-AI domain (e.g., molecular property prediction, materials science with available simulators) to test generalization of the framework.
- Confidence intervals or repeated runs for all quantitative results, especially the marginal 1.9% improvement on inference acceleration.

## Removed Points

- **Criticism that the paper lacks comparison against a "human + compute baseline" with equal resources**: This demands an experiment outside the paper's stated scope. The paper is about whether an autonomous system can surpass existing human SOTA *methods*, not whether it surpasses a human given 20K GPU hours. These are different questions, and the paper's claim is about the former.

- **Criticism that DeepReviewer evaluation is "circular"**: The paper uses DeepReviewer only as one evaluation channel and explicitly supplements it with human expert review. The DeepReviewer comparison is against papers from *other* AI Scientist systems, not for optimizing DeepScientist itself, so the circularity concern is substantially mitigated by the dual evaluation protocol.

- **Criticism about the system requiring proprietary frontier models (Gemini-2.5-Pro, Claude-4-Opus)**: This is not a valid weakness — these are legitimate, available tools, and the paper explicitly states which models are used. The choice of capable models is appropriate for the task.

- **Criticism that the "Analyze & Report" module is not open-sourced**: This is a deliberate ethical choice explained in the paper, and it does not prevent reproduction of the core scientific discovery pipeline. The main empirical claims are about the discovered methods' performance, not about paper generation quality.

- **Criticism that the paper says it avoids "engineering combinations" of existing methods in inference acceleration, making comparisons unfair**: The paper is transparent about this choice and explicitly states it targets methodological novelty over engineering optimization. This is a legitimate research framing, not a methodological flaw. The comparison is against the human SOTA base method, which is the fairest possible comparison for demonstrating methodological innovation.

- **Criticism about lack of reproducibility due to compute cost**: 20,000 GPU hours is substantial but not unusual for systems papers at top venues. The code and logs are promised to be released, and the key findings (the discovered methods) can be verified independently.

## Novel Insights

The most interesting empirical finding is the exploration funnel shape — 5,000 ideas → 1,100 implementations → 21 progress findings → 5 papers — and the fact that 60% of failures originate from implementation errors rather than flawed hypotheses. This suggests that the current bottleneck for autonomous scientific discovery is not idea generation or even experimental design, but code execution reliability, which is a concrete target for future systems. The progressive discovery trajectory in AI text detection, where the system identified limitations of its own T-Detect method and then autonomously developed wavelet/phase-based improvements, represents a genuine capability demonstration that goes beyond single-shot generation.

## Suggestions

- **Run ablations on at least one task**: Compare the full system against (a) no Findings Memory (each iteration starts fresh), (b) random idea selection instead of UCB, (c) greedy selection on highest utility score. This is the single most important experiment for attributing success to the system design rather than raw LLM capability.
- **Report statistical uncertainty**: Add confidence intervals or error bars from multiple runs for all quantitative results, particularly the 1.9% inference acceleration improvement and the scaling experiment.
- **Reframe the BO claims honestly**: Call the selection mechanism "LLM-guided heuristic search inspired by BO principles" or similar, and either provide evidence that the surrogate scores correlate with experimental outcomes or remove the BO framing from the title and abstract.
- **Add a surrogate validation experiment**: Plot the LLM surrogate's predicted value (vu, vq) against actual experimental outcomes for all validated ideas. This would either validate the selection mechanism or reveal it as uninformative — either result would be valuable.

## Score and Decision

**Calibration**: Compared to CycleResearcher (avg ~6.5, Accept Poster) — similar domain (AI scientist), similar circular evaluation concerns, but DeepScientist has larger-scale empirical results while having substantially weaker evaluation rigor and more overclaiming. Compared to MOOSE-Chem (avg ~6.25, Accept Poster) — similar scope (LLMs for scientific discovery), but MOOSE-Chem has cleaner evaluation and less ambitious claims. Compared to ResearchTown (avg ~5.75, Reject) — DeepScientist has far more substantial empirical results and a more complete system. Compared to AI text detection papers with weak evaluation (scores 3-6, mostly Reject) — DeepScientist goes far beyond single-task papers but shares some evaluation thinness.

DeepScientist presents an ambitious and significant engineering effort with real empirical demonstrations, but the paper substantially overclaims relative to its evidence. The "SOTA-surpassing" narrative is weakened by single-baseline comparisons, no statistical rigor, and marginal improvements on some tasks. The BO framing is aspirational rather than actual. The scaling law claim is based on 4 data points without variance. These are not minor issues — they concern the paper's core claims. That said, the progressive discovery capability is genuine, the failure analysis is honest and informative, and the system architecture represents a real contribution over prior AI Scientist approaches.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>