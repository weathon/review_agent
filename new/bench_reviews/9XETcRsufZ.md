Now I have sufficient context for calibration. Let me synthesize the final review.

Key calibration anchor points:
- Scaling laws for sparsely-connected models (i9K2ZWkYIP): accepted with spotlight, scores 6/6/8/8 — strong empirical work with some limitations
- MoE theory in continual learning (7XgKAabsPp): accepted spotlight, scores 8/6/8 — novel theory contribution with limited depth  
- Smoothness bridges sparsity and stability in MoEs (hAyw43h0MH): rejected, scores 3/3/5/1 — flawed theory + small experiments
- MoE instruction tuning paper (6mLjDwYte5): accepted poster, scores 8/5/6/8 — timely empirical study with MoE, limited mechanistic insight
- Memorization in logical reasoning (5sQiK2qTGa): rejected, scores 5/6/5/5 — blurry definition of memorization vs reasoning

Now let me evaluate the paper under review:

**Strengths:**
1. Timely and important question about MoE tradeoffs
2. Clean theoretical results (communication complexity bounds for single-layer MoEs)
3. Well-designed synthetic experiments (phone-book, shortest path)
4. Substantial pretraining experiments (2.1B params, 65B tokens)
5. Perplexity-controlled analysis (Figure 6) is insightful

**Weaknesses that I need to verify against the paper:**

1. **Over-interpretation of "memorization vs reasoning" dichotomy** - This is a valid concern. The paper classifies tasks into "world knowledge = memorization" and "commonsense/math = reasoning," but these task groupings are coarse. However, the paper does acknowledge this is a simplification and the synthetic experiments provide cleaner separation. The central narrative is somewhat overstated but the empirical finding (MoEs scale with total params on some tasks, active params on others) is real. I'll keep this as a major weakness but note the paper's theoretical and synthetic experiments do provide a cleaner separation.

2. **Theory only applies to depth-1 models** - Valid and acknowledged by the paper. This is a major limitation of the theoretical contribution but the paper doesn't claim otherwise. The paper says "we prove that there exist graph problems that cannot be solved by any number of experts of a certain width" - this is for single-layer transformers. The connection to multi-layer experiments is loose. I'll keep this as major.

3. **Lack of compute-matched comparison** - The paper compares at fixed total parameters and active parameters but not at fixed training compute. This is a valid concern. However, the paper does include perplexity-controlled comparisons which partially address this. I'll keep this as a significant but not fatal concern.

4. **Figure 6 undermines the "MoEs are worse at reasoning" narrative** - This is an important observation from the harsh critic. At fixed perplexity, MoEs match dense models on reasoning. The paper text does discuss this but the title/abstract still claims "reasoning capabilities saturate." In reality, it's more accurate to say "MoEs are less parameter-efficient for reasoning" rather than "MoEs can't reason as well." I'll keep this as a major weakness.

5. **Generalization gap interpretation** - The larger train-test gap for MoEs on math is attributed to "memorization" but could simply reflect higher capacity leading to more overfitting, which is not the same thing as an architectural bias toward memorization. Valid minor concern.

6. **Single training run, no error bars** - Valid concern but standard for this type of work at this scale. Minor/trivial.

7. **FFN dimension set to d instead of 4d** - This is a non-standard choice but the paper is consistent about it. This could affect how the results generalize. Minor concern.

8. **Training on 65B tokens is relatively small** - Valid but the paper still shows clear trends. This is a scope limitation, not a flaw.

9. **The "saturation" claim in the abstract** - The harsh critic notes that the plots don't show true "saturation." Looking at Figure 1b-c and 4b, the MoE lines are relatively flat but not fully saturated. The wording is somewhat imprecise. Minor.

Let me now write the review.</think>## Summary

This paper investigates the tradeoffs between Mixture-of-Experts (MoE) and dense transformers, showing that increasing the number of experts (while fixing active parameters) improves memorization performance more than reasoning. The authors provide theoretical communication-complexity lower bounds showing that MoEs require a critical width (regardless of expert count) to solve graph reasoning problems, while also showing that MoEs can memorize with far fewer active parameters than dense models. These findings are validated through synthetic experiments (phone-book memorization and shortest-path reasoning) and large-scale pre-training experiments on NLP and math benchmarks.

## Strengths

- **Timely and important research question.** As MoE architectures become dominant in frontier models (Mixtral, GPT-4, etc.), understanding when experts help versus when width matters is of high practical significance. The paper directly addresses the question of whether MoE scaling is a "free lunch."

- **Clean, novel theoretical results.** Extending the communication-complexity lower bound of Sanford et al. (2024) to MoE transformers (Theorem 3.2) and the memorization upper bound (Theorem 3.5) provide principled, formal separation results. The corollary establishing a parameter-matched separation between dense and sparse models (Corollary 3.4) is particularly clean.

- **Well-designed synthetic experiments.** The phone-book and shortest-path tasks directly instantiate the theory's predictions, and the results cleanly separate the two regimes: total parameters predict memorization, active parameters predict reasoning (Figure 4).

- **Substantial empirical investment.** Pre-training models up to 2.1B parameters on 65B tokens across both NLP and math domains, with evaluation on ~17 downstream benchmarks, provides valuable practical evidence beyond the synthetic settings.

- **Perplexity-controlled analysis.** Figure 6, comparing models at fixed validation perplexity rather than parameter count, is one of the more insightful analyses. It reveals that MoEs match dense models on reasoning when perplexity is controlled, and outperform on world knowledge — a nuanced finding that goes beyond simple parameter-count comparisons.

## Weaknesses

### Fatal
None.

### Major

- **The "reasoning vs. memorization" dichotomy is oversold relative to the evidence.** The central claim that "experts improve memorization more than reasoning" rests on a coarse task grouping: TriviaQA/NQ/etc. as "memorization" and HellaSwag/Winogrande/MATH as "reasoning." In reality, "world knowledge" tasks involve retrieval and paraphrase understanding, while "commonsense" benchmarks contain substantial distributional cues. The synthetic experiments cleanly separate these, but the leap from synthetic to real benchmarks is under-justified. More importantly, Figure 6 shows that at fixed perplexity, MoEs *match* dense models on reasoning benchmarks — suggesting the gap is better described as "MoEs are less parameter-efficient for reasoning" rather than "MoEs saturate in reasoning capability." The title's declaration that "experts improve memorization more than reasoning" is a stronger claim than the evidence warrants.

- **The theoretical results are narrow and the connection to the experiments is loose.** Theorem 3.2 applies to single-layer transformers with top-1 routing on an adversarial graph problem, while the experiments use 12–20 layer models with top-2 routing on random graphs and natural language tasks. The paper explicitly acknowledges this gap but repeatedly draws conceptual bridges ("these results corroborate the theory," Section 4.3) that overreach what the theory actually establishes. Depth can fundamentally alter expressivity for graph problems, and the adversarial lower bound does not directly predict the observed performance of trained models on real tasks. The theory is a valid standalone contribution but does not *explain* the empirical findings.

- **Lack of compute-matched or FLOP-matched comparisons.** The paper compares at fixed total parameters and fixed active parameters, but never at fixed training FLOPs. When total parameters are matched, the MoE has fewer active parameters and thus lower per-step FLOPs — meaning it could train on more tokens with the same compute budget. Without this comparison, it is unclear whether the reasoning gap reflects an architectural limitation of MoEs or an undertraining artifact. The perplexity-controlled comparison (Figure 6) partially addresses this, but it controls for training quality rather than compute budget.

### Minor

- **The generalization gap analysis (Figure 5) does not uniquely support "memorization."** A larger train-test gap for MoEs at matched total parameters could reflect that models with more total parameters simply overfit more — a parameter count effect rather than an architectural MoE-specific bias toward memorization. The paper suggests "MoEs are more prone to overfit," but this is confounded by total parameter count.

- **Non-standard FFN dimension.** Setting the intermediate dimension to $d$ instead of $4d$ (Section 5.1) deviates from standard practice. While the paper is consistent about this choice, it may change the relative dynamics between expert capacity and routing, and the results may not directly extrapolate to standard architectures.

- **No error bars or replicate runs.** All figures appear to be single runs. At the 2.1B scale, this is understandable, but for synthetic experiments, the visible differences (e.g., 5–10 points on shortest path) could fall within run-to-run variability.

- **The abstract's claim that reasoning "saturates" overstates the evidence.** The MoE curves in Figures 1b-c and 4b show relatively flat performance with added experts, but "saturates" implies a limit has been reached. Showing this at larger expert counts or with more training would strengthen this claim.

### Trivial
- None worth noting.

## Nice-to-Haves

- **Expert activation analysis:** Visualizing which experts are activated for factual vs. reasoning tokens would directly validate the "experts memorize" claim and provide mechanistic insight beyond the current indirect evidence.

- **Compute-matched comparison:** Training dense and MoE models with matched training FLOPs (varying both model size and token count) would clarify whether the reasoning gap persists under practical deployment conditions.

- **Extended theory to multi-layer settings:** Even informal discussion of how depth interacts with the width-expert tradeoff would significantly strengthen the theory-to-experiment bridge.

## Removed Points

These points are flagged to be removed, treated with caution:

- **"Models trained on only 65B tokens are under-trained"** (from neutral reviewer and spark): This is a scope limitation, not a flaw. The paper shows clear, consistent trends across multiple scales. More training tokens would strengthen but not invalidate the findings.

- **"No exploration of fine-tuning or instruction tuning"** (from human finder): This is outside the paper's stated scope, which focuses on pre-training dynamics. Instruction tuning effects are a separate research question.

- **"Limited evaluation suite for reasoning"** (from human finder): The paper evaluates on 5 knowledge benchmarks, 8 commonsense benchmarks, and 5 math benchmarks, which is reasonably broad. Adding NLI or code would be nice but is not a core flaw.

- **"Missing related works"** (implicit in human finder): Per the rules, I should not flag missing related works without verifying their existence.

- **"Hyperparameter parity and optimization concerns"** (from harsh critic): The paper sweeps LR and epochs for synthetic experiments and uses standard settings for pre-training. While not perfect, this is standard practice and not a core flaw. The concern about MoE-specific routing knobs is partially valid but the paper uses well-established MoE implementations.

- **"Routing mechanism ablation"** (from spark): The paper explicitly states "we leave the study of MoEs trained with other routing mechanisms for future work." This is a valid scope limitation, not a weakness.

## Novel Insights

The perplexity-controlled comparison (Figure 6) reveals a genuinely important nuance that partially reframes the paper's narrative: MoEs do not appear strictly *worse* at reasoning — they are *parameter-inefficient* at reasoning. At matched perplexity, MoEs match dense models on commonsense and math. The architectural tradeoff is not between "memorization machines" and "reasoning machines" but rather between parameter efficiency for different task types. This distinction matters for practical model design: the question is not whether MoEs *can* reason, but whether a given FLOP budget is better spent on width (for reasoning) or experts (for knowledge).

## Suggestions

- Moderate the title and abstract claims from "experts improve memorization more than reasoning" to something like "experts improve memorization more than *parameter efficiency for* reasoning," to accurately reflect the finding that the gap is primarily about active parameter efficiency rather than a fundamental reasoning limitation.

- Add a compute-matched comparison (fixed FLOPs, varying model size and token count) to determine whether the reasoning gap persists under practical deployment conditions.

- Include a brief discussion explicitly acknowledging the gap between the depth-1 theory and the multi-layer experiments, and discuss what the theory *does* establish (a necessary condition on width for a specific problem) versus what it does not (a complete explanation of the empirical observations).

## Score and Decision

**Calibration comparison:**
- Scaling laws for sparsely-connected models (i9K2ZWkYIP): Accepted spotlight, scores 6/6/8/8. Similar profile: strong empirical scaling analysis, limited theory, important practical question. This paper has more theory but a weaker theory-to-experiment connection.
- MoE theory in continual learning (7XgKAabsPp): Accepted spotlight, scores 8/6/8. Novel theoretical contribution with limited empirical validation. This paper has more substantial experiments but overclaims the theory's relevance.
- MoE instruction tuning (6mLjDwYte5): Accepted poster, scores 8/5/6/8. Timely empirical study with MoE, limited mechanistic insight. Similar type of contribution: identifies an important MoE tradeoff empirically.
- Memorization in logical reasoning (5sQiK2qTGa): Rejected, scores 5/6/5/5. Weak definition of memorization vs reasoning, partially validated claims. This paper is stronger — it has clean synthetic experiments and formal theory.

This paper is above the middle of the pack. The research question is important and timely, the theoretical results are valid (if narrow), and the empirical work is substantial. The main issue is overclaiming: the theory-to-experiment bridge is loose, the task categorization is coarse, and Figure 6 actually undermines the strongest version of the central claim. However, the core finding — that MoE scaling helps knowledge tasks more than reasoning tasks — is real and practically significant, and the paper provides valuable empirical data for the community. Relative to the MoE scaling laws paper (spotlight, avg ~7) and the MoE instruction tuning paper (poster, avg ~6.75), this paper is somewhat stronger on theory and somewhat weaker on the theory-practice connection. It's a solid contribution that would benefit from moderating its claims.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>