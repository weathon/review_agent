Now I have a thorough understanding of the paper. Let me verify the harsh critic's specific claims:

1. **Ablations only cover 2 of 4 tasks (OBP and OP)**: Confirmed from Table 4, which only shows OBP and OP results.

2. **Computational cost comparison is incomplete**: The paper mentions running on a "single 24GB GPU" but doesn't include wall-clock time or total FLOPs alongside GRPO training cost. The appendix reportedly contains timing info. Let me check.

3. **Verbal gradient vs numerical gradient narrative**: Let me re-examine. The paper claims CALM "surpasses methods that rely solely on verbal guidance, even when those use significantly more powerful API-based models." The harsh critic argues this conflates framework vs model contributions. Looking at the data: CALM (local, w/ GRPO) = 0.71% OBP, while CALM (API, w/o GRPO) = 0.82% OBP. So CALM w/ GRPO on a weaker model does beat CALM w/o GRPO on a stronger model. Also, the best baseline MCTS-AHD gets 0.89%. The claim of surpassing methods using "more powerful API-based models" is true in aggregate across tables, though on some specific comparisons the margin is small.

4. **Collapse approximation condition**: The critic notes δ₀ ∈ {0.0005, 0.005} and C = 15, and 1/δ₀ ∈ {2000, 200}, both vastly exceeding C = 15, so the condition C > 1/δ₀ is only met when C = ∞. Let me verify. The condition stated in Eq. 2 is C > 1/δ₀. With δ₀ = 0.0005, 1/δ₀ = 2000, so 15 > 2000 is false. With δ₀ = 0.005, 1/δ₀ = 200, so 15 > 200 is false. For C = ∞, the condition becomes trivially true. So the approximation only applies when C = ∞, not when C = 15. This is a valid observation — the theoretical formula doesn't match the practical configurations used. However, it's a minor issue since the practical results show which (δ₀, C) combos work.

5. **Reward asymmetry**: Looking at Eq. 4: positive rewards range from [1, 2] (when g(h_new) > g(h_t_base)), while negative rewards are bounded by α₂ · r_invalid · Δ where α₂ ∈ (0,1) and r_invalid ∈ (-1, 0). So r_invalid could be like -0.5, and α₂ · r_invalid · Δ would be at most -0.5. Positive rewards start at 1. This is a valid concern about the asymmetry of the reward function, though it may be a design choice.

6. **Standard deviations not reported in Tables 1-3**: Confirmed — only training curves show std. dev. shaded areas, but the tables don't include error bars or std devs. This is a legitimate minor concern.

Now let me compile the final review, calibrated against the anchors I've found.

Key calibration anchors:
- **High (≥6)**: LLM-SR (avg 8), Eureka (avg 6.25), EvoPrompt (avg 6.5), LLEGO (avg 6.25)
- **Medium (~5)**: DlqRpj68xe/Q-shaping (avg 5.67), Evo-Step (avg 5.5), SimNPO (avg 5.25)
- **Low (≤4)**: MHRE (avg 2.5), SELA (avg 3.5)

CALM is clearly stronger than MHRE (which had fundamental methodology/presentation issues) and SELA (limited novelty, inadequate evaluation). CALM has a genuine novel idea, strong experimental results, and comprehensive baselines. Its weaknesses are more like the medium-scoring papers: incomplete ablations, some overclaiming. But compared to EvoPrompt (avg 6.5), CALM has more technical novelty (GRPO + verbal gradient co-evolution) and more comprehensive experiments across 4 tasks. Compared to Eureka (avg 6.25), CALM is similar in combining LLM with evolutionary optimization. Compared to LLM-SR (avg 8), CALM is somewhat weaker because LLM-SR had very thorough evaluation and ablations.

CALM's core contribution is genuine and novel: co-evolving the LLM with the search process via RL. The experiments show it works across 4 tasks. The main weaknesses are: (1) ablations only on 2/4 tasks, (2) some overclaiming in the efficiency narrative, and (3) the verbal gradient framework alone is highly competitive, which the paper partly acknowledges but could foreground better.

I'd score this around 6.5 — a solid contribution with real novelty, strong results, but with notable gaps in ablation coverage.</think>## Summary

CALM introduces a framework for automatic heuristic design (AHD) that co-evolves the LLM alongside the evolutionary search process by combining verbal guidance (evolutionary prompt operators) with numerical guidance (GRPO-based RL fine-tuning). A locally-deployed 7B INT4-quantized model fine-tuned via GRPO consistently outperforms stronger API-based baselines (GPT-4o-mini) across four optimization tasks (OBP, TSP, CVRP, OP), demonstrating that RL-based model adaptation can close and surpass the gap created by using a smaller foundation model.

## Strengths

- **Genuinely novel co-evolution idea**: CALM is one of the first LLM-based AHD frameworks that jointly optimizes the prompt generation process and the LLM model itself. The ablation on OBP and OP confirms a large effect size — removing GRPO degrades OBP from 0.71% to 1.78% and OP from 17.41% to 19.89% (Table 4), establishing that numerical gradients provide substantial gains beyond verbal guidance.

- **Strong empirical results across tasks**: Tables 1–3 show CALM beating all LLM-based baselines on OBP (0.71% avg gap vs. best prior 0.89%), TSP at N=200 (13.41% vs. 13.56%), CVRP at N=50 (3.83% vs. 5.44%), and OP at N=200 (12.58% vs. 16.34%). The consistency across four diverse optimization problems strengthens the generalization claim.

- **Practical accessibility**: Demonstrating that a 7B INT4 model on a single 24GB GPU can match or exceed API-based methods relying on GPT-4o-mini is practically significant for reproducibility and cost accessibility.

- **Well-designed verbal gradient components**: Even without GRPO, the verbal gradient variant (CALM-API, w/o GRPO, on GPT-4o-mini) matches or exceeds the prior best MCTS-AHD on most tasks, confirming that the injection, replacement, crossover, collapse, and simplification operators constitute a meaningful contribution independent of RL.

- **Principled collapse mechanism**: The probabilistic collapse trigger (balancing patience with local-optima escape via the cnδ₀ criterion) is well-motivated and shown to improve results (OBP: 0.98% → 0.71% when enabled).

- **Systematic ablation study**: Table 4 evaluates 9 ablation configurations on OBP and OP, isolating GRPO, reward design, collapse, and individual operators. The finding that removing GRPO causes the single largest degradation is particularly informative.

## Weaknesses

### Fatal
None.

### Major

- **Ablation coverage is limited to 2 of 4 experimental tasks**: The central claim that GRPO-driven co-evolution drives improvements across "various optimization tasks" is supported by full ablations only on OBP and OP (Table 4). TSP and CVRP — which constitute half the evaluation — lack ablation data. Without this, it is unclear whether GRPO is necessary on those tasks or whether the verbal gradient framework alone suffices (as it nearly matches MCTS-AHD on OBP without GRPO). This is a significant evidential gap for a paper whose core claim is that numerical gradients are the key differentiator. The paper should at minimum include a w/o GRPO ablation on TSP and CVRP to validate that the RL component contributes on all reported tasks.

- **Efficiency narrative is incomplete without GRPO training cost**: The abstract and introduction prominently claim CALM runs at "a fraction of the traditional cost" on a "single 24GB GPU." However, the comparison framework counts API query costs for baselines but does not account for the GRPO fine-tuning compute (G forward-backward passes per round over T rounds on a 7B model). While the appendix reportedly contains timing information, the main paper lacks wall-clock time or GPU-hour cost for the full CALM pipeline, making it impossible to assess whether the total compute (training + inference) is genuinely more efficient than API-based alternatives. This is not a fatal flaw — the single-GPU feasibility claim is valid — but the "fraction of the cost" framing requires quantitative substantiation.

### Minor

- **Asymmetric reward function is not discussed**: In Eq. 4, positive rewards range from [1, 2] (when the new heuristic improves on the best parent), while negative rewards are bounded in (α₂·r_invalid·Δ, 0) ⊂ (−1, 0). This asymmetry strongly reinforces good heuristics but weakly penalizes bad ones, encouraging risk-taking. This may be a deliberate design choice for exploration, but it is unacknowledged, and no ablation tests this asymmetry directly.

- **Collapse approximation condition is not met by practical configurations**: Equation 2 provides an analytical approximation E[cn | collapse] ≈ √(π/(2δ₀)) under the condition C > 1/δ₀. However, the ablated configurations use δ₀ ∈ {0.0005, 0.005} with C = 15, giving 1/δ₀ ∈ {2000, 200}, both vastly exceeding C = 15. The condition is only met when C = ∞. The approximation therefore does not apply to the primary configurations used in experiments, which is a minor theoretical inconsistency.

- **Standard deviations absent from Tables 1–3**: Results are "averaged over three runs" but no standard deviations appear in the main tables. For some comparisons with small margins (e.g., CALM vs. CALM-API on certain OBP splits), statistical significance cannot be assessed. The paper notes that p-values are in the appendix, but they should be more accessible in the main text.

- **The framing overstates the primacy of numerical gradients**: The abstract positions CALM as surpassing methods relying on "verbal guidance only," but the API-based CALM variant without GRPO already matches or exceeds prior SOTA on most tasks (e.g., 0.82% on OBP vs. MCTS-AHD's 0.89%). The verbal gradient design is doing substantial heavy lifting, and the paper's narrative underplays this relative contribution. This doesn't invalidate the contribution but slightly distorts its character.

### Trivial
None worth listing.

## Nice-to-Haves

- **Interaction ablation (operators × GRPO)**: Testing whether fine-grained operators (injection, replacement) are specifically beneficial because GRPO enables per-token credit assignment, or whether they help equally without RL, would validate the motivation in Section 4.1 and strengthen the claim that verbal and numerical gradients are synergistic.

- **Cross-task transfer analysis**: Testing whether the GRPO-fine-tuned model from one task transfers to another would demonstrate whether numerical gradients capture generalizable heuristic design principles.

- **Qualitative analysis of learned heuristics**: Showing examples of heuristics before and after co-evolution would illuminate what structural patterns GRPO enables, providing interpretability beyond the performance metrics.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Criticism of "not yet released" models/tools**: The harsh critic questioned whether AlphaEvolve's code availability affects reproducibility. The paper explicitly addresses this by using OpenEvolve (an independent open-source reimplementation). Per the rules, cited entities are assumed to exist.

- **Formatting/presentation nitpicks**: The harsh critic noted issues with equation rendering and symbol formatting. These are parser artifacts, not paper issues.

- **Demand for reproducibility details like hyperparameters**: The paper provides algorithm details in Appendix C, prompts in Appendix D, and code at a public repository. Requesting more implementation minutiae is beyond what's standard for this venue.

- **Demand for generalization analysis of fine-tuned model**: Testing whether the fine-tuned model retains general code generation capability is an interesting future direction but falls outside the paper's stated scope of AHD.

- **Demand for cross-task transfer experiments**: This is explicitly a "next step" suggestion, not a current weakness. Moved to Nice-to-Haves.

- **Strength claim about "fair comparison protocol"**: The Strength Finder claimed this as a strength. While the paper does use shared seeds and datasets, the comparison budgets differ (1,000 evaluations for baselines vs. 2,000 queries for CALM), and whether this is truly "fair" depends on whether 2,000 LLM queries ≥ 1,000 heuristic evaluations in cost. This is debatable rather than clearly a strength, so removed from the strengths list.

## Novel Insights

The key insight from the reviews is that CALM's contribution is really dual: (1) a strong verbal gradient framework that independently achieves near-SOTA performance, and (2) a GRPO-based co-evolution mechanism that delivers additional but not exclusive gains. The paper's narrative positions RL as the primary differentiator, but the data shows the verbal gradient design is comparably important — the gap between CALM-GRPO and CALM-API is smaller than the gap between CALM-API and prior SOTA on most tasks. This reframing doesn't diminish the contribution but suggests future work should investigate the synergy between these two components more carefully.

## Suggestions

- Add at minimum a table row for "local w/o GRPO" on TSP and CVRP to validate that the numerical gradient contribution holds across all four tasks, not just two.
- Report total GPU-hours or wall-clock time for the full CALM pipeline alongside the baselines' API costs, so the efficiency claim can be directly assessed.
- Acknowledge the reward asymmetry in Section 4.3 and discuss whether it is a deliberate exploration mechanism or a potential design limitation.

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Avg Score | Comparison to CALM |
|---|---|---|---|
| LLM-SR (m2nmp8P5in) | LLM + evolutionary search for equation discovery | 8.0 (Oral) | More thorough evaluation and ablations; CALM is weaker in ablation coverage |
| Eureka (IEduRUO55F) | Evolutionary LLM optimization for RL rewards | 6.25 (Poster) | Similar novelty profile (LLM + evolutionary optimization); CALM has broader task evaluation but less thorough ablations |
| EvoPrompt (ZG3RaNIsO8) | LLM + evolutionary algorithms for prompt optimization | 6.5 (Poster) | Less technical novelty; CALM is clearly stronger with RL component |
| Q-shaping (DlqRpj68xe) | LLM heuristic for RL | 5.67 (Reject) | Weaker contribution than CALM; limited novelty and writing issues |
| MHRE (sUywd7UhFT) | LLM hyper-heuristics for optimization | 2.5 (Withdrawn) | Fundamentally flawed methodology; CALM is far superior |
| SELA (fv2hL5n2mh) | MCTS-enhanced LLM agents for AutoML | 3.5 (Reject) | Limited novelty and evaluation; CALM is much more complete |

CALM is clearly above the reject-level papers (MHRE, SELA) and above borderline papers (Q-shaping, Evo-Step). It is comparable in novelty to Eureka (6.25) but with less complete ablation coverage. It falls below LLM-SR (8.0) which had very thorough evaluation. The main gap is the incomplete ablation (only 2 of 4 tasks). Given that the core novelty (co-evolution via GRPO) is genuine, the results are strong and consistent, and the ablations on OBP/OP are informative, I place CALM at approximately 6.0 — a solid contribution that would benefit from completing the ablation evidence.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>