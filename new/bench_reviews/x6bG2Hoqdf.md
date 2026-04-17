Now I have enough context from calibration papers. Let me synthesize the final review.

Let me review the key points from the paper and the reviewers' analyses:

**Paper Summary**: CALM proposes co-evolving algorithms and language models for automatic heuristic design (AHD). It combines "verbal gradients" (evolutionary prompt operators) with "numerical gradients" (GRPO-based RL fine-tuning) on a local 7B INT4-quantized model. The verbal operators include injection, replacement, diversity-aware crossover, and simplification. A collapse mechanism handles stagnation. Results show it outperforms GPT-4o-mini-based SOTA AHD methods on OBP, TSP, CVRP, and OP tasks.

**Harsh Critic's Key Points**:
1. Ambiguity of RL vs. verbal guidance contribution - the API-based CALM without RL is already strong
2. Budget comparison fairness - "LLM queries" vs "heuristic evaluations" not clearly mapped
3. GRPO-operator interface details under-specified
4. Reward function design issues (duplicate detection, numerical stability)
5. Insufficient statistical characterization

**Neutral Reviewer's Key Points**:
1. Query budget transparency (G value not stated in main text)
2. Verbal gradient ablation uses unequal budgets
3. Limited qualitative analysis of what RL achieves
4. Collapse mechanism sensitivity
5. Reward function complexity
6. Scalability untested beyond N≤200

**Spark's Key Points**:
1. Missing ablation isolating RL from operator design when comparing to baselines
2. No cross-problem transfer evaluation of fine-tuned model
3. No comparison with stronger base model + GRPO
4. No compute-cost normalized comparison
5. No concrete examples of discovered heuristics

**Human Finder's Key Points**:
1. Single base model evaluation
2. Limited statistical significance (3 runs)
3. Many moving parts make attribution difficult
4. Comparison limited to LLM-based methods and weak traditional baselines
5. Seed heuristic sensitivity
6. Compute cost transparency

Let me now verify these criticisms against the actual paper:

**On budget/comparison fairness**: The paper states "1,000 heuristic evaluations for baselines and a fixed budget of 2,000 LLM queries for CALM" and "prior AHD methods typically conduct 2,000 heuristic evaluations using over 4,000 queries for OBP, whereas CALM operates under a fixed budget of 2,000 queries." The G value for GRPO is mentioned but I need to check if it's in the paper... The paper doesn't explicitly state G in the main text. This is a legitimate concern.

**On RL vs. verbal contribution**: The paper does discuss this in Section 5.2, showing Table 4 where "w/o GRPO" drops from 0.71% to 1.78% on OBP. But this ablation is only on OBP and OP. The API-based CALM (GPT-4o-mini w/o GRPO) is also very strong. This is a valid concern.

**On discovered heuristics**: The paper states the code is available and mentions "elite heuristics" in appendix, but the main paper doesn't show any actual generated heuristic code. This is a reasonable critique for an AHD paper.

**On missing baselines like LKH/EAX**: The paper focuses on LLM-based AHD methods and includes hand-crafted heuristics and NCO baselines. For TSP step-by-step construction, LKH isn't directly comparable since it's a local search solver, not a constructive heuristic. However, the comparison is fair within the AHD paradigm.

**On single base model**: Valid concern but not fatal - the paper explicitly acknowledges this and positions itself as showing a small local model can compete.

**On statistical significance**: 3 runs with mean gaps reported, p-values in appendix. Some margins are small (OP N=50: 24.22% vs 24.23%). This is a legitimate concern.

**On compute cost**: The paper claims single 24GB GPU but doesn't report wall-clock time in the main text. Appendix mentions it.

Now let me compose the final review.</think>Now I have sufficient calibration context. Let me compose the final review.

## Summary

CALM proposes a co-evolutionary framework for Automatic Heuristic Design (AHD) that pairs evolutionary prompt-based search ("verbal gradients") with on-the-fly RL fine-tuning of the LLM itself ("numerical gradients" via GRPO). Running on a single 24GB GPU with an INT4-quantized Qwen2.5-7B model, CALM introduces fine-grained mutation operators (injection, replacement), diversity-aware crossover, a simplification operator, and a collapse mechanism. It achieves state-of-the-art results on four optimization tasks (OBP, TSP, CVRP, OP), surpassing GPT-4o-mini-based AHD baselines.

## Strengths

- **Conceptually novel contribution**: Co-evolving the LLM alongside heuristic search via GRPO fine-tuning is a genuine and timely advance over the fixed-model AHD paradigm (EoH, ReEvo, MCTS-AHD). The insight that the evolutionary loop produces natural prompt–response–performance triplets for RL training is elegant and well-motivated.

- **Strong empirical results**: CALM (local Qwen-7B-INT4 + GRPO) achieves the best average optimality gap on OBP (0.71% vs. 0.89% for MCTS-AHD), best LLM-based results on CVRP across all scales, and strong performance on TSP and OP. Demonstrating that a quantized 7B local model can rival or surpass GPT-4o-mini API-based methods is practically significant.

- **Thoughtful component design**: The fine-grained operators (injection adds components, replacement rewrites specific parts) are well-motivated by GRPO's token-level credit assignment needs. The diversity-aware crossover and simplification operator address real stagnation and code bloat concerns. The collapse mechanism's probabilistic formulation with analytical expected-time approximation (Eq. 2) is clean.

- **Comprehensive ablation study**: Table 4 systematically evaluates GRPO removal, reward variants, collapse configurations, and individual operators, providing meaningful diagnostic insight. The result that disabling GRPO causes the largest performance drop (OBP: 0.71% → 1.78%) is a clear signal.

- **Reproducibility and resource efficiency**: Complete algorithm (Appendix C), prompts (Appendix D), and open-source code. Running on a single 24GB GPU with 1.15% of weights fine-tuned makes the work accessible.

## Weaknesses

### Major:

- **Ambiguity in attributing improvement to RL vs. verbal operators when benchmarking against other methods**: The core claim is that co-evolution (RL fine-tuning) is decisive. However, Tables 1–3 show that CALM with GPT-4o-mini and without GRPO already matches or closely approaches SOTA on several tasks (e.g., CVRP N=200: 5.72% gap without GRPO vs. 3.95% with GRPO). Table 4 shows GRPO's contribution is substantial on OBP (0.71% → 1.78%), but this ablation exists only for OBP and OP—not TSP or CVRP. Without per-task GRPO ablations on all problems, the narrative overstates RL's unique contribution relative to the verbal operator design. The paper would be strengthened by reporting "w/o GRPO (local Qwen-INT4)" on all four tasks under matched budgets.

- **Evaluation budget transparency**: The paper compares CALM (2,000 LLM queries with G responses per prompt) against baselines (1,000 heuristic evaluations). The value of G is not explicitly stated in the main text, making it impossible to determine the total number of heuristic evaluations CALM performs. If G=8 (a typical GRPO setting), CALM would evaluate up to 16,000 heuristics—far exceeding the 1,000 allotted to baselines. The asymmetric budget could account for a meaningful portion of CALM's advantage. The paper should explicitly state G and total heuristic evaluations per method, or justify why the chosen comparison is fair.

- **No qualitative demonstration of what RL fine-tuning achieves**: For a paper about "co-evolution of algorithms and language model," there are no examples of heuristics generated before vs. after fine-tuning, no analysis of whether the model learns domain-specific patterns or just reduces infeasibility rates, and no visualization of how the generated heuristic space shifts during training. This makes "co-evolution" somewhat of a black box. For an AHD paper, not showing any actual discovered heuristic code in the main text is a notable omission.

### Minor:

- **Statistical characterization limited**: All main tables report means over only three runs. Some comparisons have narrow margins (e.g., OP N=50: CALM 24.22% vs. EvoTune 24.23%). While p-values are referenced in Appendix I, the main paper lacks standard deviations or confidence intervals, making it difficult to assess robustness of marginal improvements.

- **Reward function design under-specified**: The duplicate detection in Eq. (4) checks whether g(h_new) equals any g(h) in H, but it's unclear whether this is exact floating-point equality (near-impossible for real-valued objectives) or approximate with some tolerance. The normalization in Eq. (3) could become unstable when |g| approaches zero, and this is not discussed.

- **Single base model**: All experiments use Qwen2.5-7B-Instruct-INT4. While the paper explicitly acknowledges this model's relative weakness vs. GPT-4o-mini, it does not test whether GRPO provides similar gains on larger or different architectures, limiting claims about generality of the co-evolution approach.

## Nice-to-Haves

- **Cross-problem transfer**: Test whether a model fine-tuned on one problem (e.g., OBP) generates better initial heuristics for another (e.g., TSP) without further training. This would substantiate the "co-evolution" framing beyond per-task online adaptation.

- **Compute cost comparison**: Report wall-clock time and GPU-hours for CALM's full training pipeline vs. API cost for baselines. The efficiency claim ("single 24GB GPU") would be more convincing with concrete time/cost figures.

- **Comparison with stronger model + GRPO**: Even a brief experiment with Qwen2.5-14B-Instruct + GRPO would clarify whether RL primarily helps weak models catch up or provides genuine additive value.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Missing comparison with LKH/EAX as classical solvers**: The paper studies LLM-based AHD within constructive heuristic frameworks (ACO-based search, greedy construction). LKH and EAX are complete solvers/local search methods that are not directly comparable—they solve the problems rather than design heuristics. The paper includes appropriate NCO and hand-crafted baselines for its experimental paradigm.

- **Concerns about EvoTune comparison being insufficient**: The paper directly compares against EvoTune (a concurrent RL-fine-tuning baseline using DPO) in Tables 1–3 across all tasks, showing consistent improvements. This is an adequate comparison with the closest related method.

- **Catastrophic forgetting as a weakness**: While an interesting question, the framework fine-tunes per-problem, so catastrophic forgetting of other AHD tasks is not a claimed feature. The paper's scope is single-problem AHD, not multi-task continual learning.

- **Demand for theoretical proofs**: This is an empirical systems/contribution paper; demanding theoretical convergence guarantees would be scope creep beyond the paper's stated goals.

- **Formatting/style nitpicks**: Equation rendering artifacts from PDF parsing are not substantive.

## Novel Insights

The key insight that existing LLM-based AHD methods leave the LLM frozen while the evolutionary search evolves only the prompts—a missed opportunity for model adaptation—is both simple and impactful. The tension this creates for GRPO's credit assignment (coarse whole-program mutations make token-level advantage scores noisy) motivates the fine-grained operators well. However, the paper does not conclusively demonstrate that "numerical gradients" are more than a scaling benefit for weaker models; the strongest version of the co-evolution claim remains partially supported.

## Suggestions

- **Report G and total heuristic evaluations per task** in the main text, and either run baselines with matched total evaluations or explicitly justify why the asymmetric comparison is fair (e.g., showing that CALM's per-evaluation efficiency is also superior).

- **Add GRPO ablations for TSP and CVRP** in the main paper or appendix, so readers can assess RL's contribution across all four tasks.

- **Show at least one concrete discovered heuristic** (e.g., best OBP heuristic) alongside the seed heuristic to make the improvement tangible and the AHD contribution verifiable.

- **Include standard deviations** in main tables for key comparisons, especially where margins are small.

## Score and Decision

**Calibration**: I compared against several papers in similar domains:

- LLM-SR (Oral, scores 8/8/8/8): Novel LLM+evolutionary framework with strong empirical results, clear methodology, well-justified novelty.
- LASeR (Poster, scores 6/8/5/6): LLM-based evolutionary design with meaningful but incremental contributions, marginal gains, and some experimental gaps.
- LLM4Solver (Reject, scores 3/3/3/5/3): Limited novelty over EoH/ReEvo, missing comparisons with LLM baselines, many components hard to attribute.
- Hercules (Reject, scores 5/8/3/5): LLM-based AHD with missing classical solver comparisons and questions about novelty.
- MHRE (Reject, scores 3/1/5/1): Poor methodology, unclear contributions, claimed multi-objective but tested on single-objective.

CALM presents a genuinely novel and well-executed contribution—the co-evolution of LLM and heuristic search via GRPO—that goes beyond prior fixed-model AHD methods. The empirical results are strong, and the ablation study is commendable. However, the budget comparison lacks transparency (G value unspecified in main text), the RL contribution is over-claimed relative to the verbal operator design (only shown for 2/4 tasks), and no qualitative evidence of what co-evolution actually teaches the model is provided. These issues are substantive but not fatal. The paper is clearly above the reject-tier papers in its area and somewhat below LASeR-level (which had cleaner attribution of gains). CALM is approximately at the level of a solid poster paper: a meaningful contribution with real but partially obscured strengths.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>