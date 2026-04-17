# Reward Model Overoptimisation in Iterated RLHF

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Reinforcement learning from human feedback (RLHF) is a widely used method for aligning large language models with human preferences. However, RLHF often suffers from reward model overoptimisation, in which models overfit to the reward function, resulting in non-generalisable policies that exploit the idiosyncrasies and peculiarities of the reward function. A common mitigation is *iterated RLHF*, in which reward models are repeatedly retrained with updated human feedback and policies are re-optimised. Despite its increasing adoption, the dynamics of overoptimisation in this setting remain poorly understood. In this work, we present the first comprehensive study of overoptimisation in iterated RLHF. We systematically analyse key design choices: how reward model training data is transferred across iterations, which reward function is used for optimisation, and how policies are initialised. Using the controlled AlpacaFarm benchmark, we observe that overoptimisation tends to decrease over successive iterations, as reward models increasingly approximate ground-truth preferences. However, performance gains diminish over time, and while reinitialising from the base policy is robust, it limits optimisation flexibility. Other initialisation strategies often fail to recover from early overoptimisation. These findings offer actionable insights for building more stable and generalisable RLHF pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how to best mitigate reward model overoptimization in iterated RLHF, where models are repeatedly fine-tuned on new preference data. The authors systematically analyze three critical design choices: how to manage preference data across iterations, how to formulate the reward function, and how to initialize the policy for each training cycle. The study concludes that the most robust and effective strategy is to concatenate all preference data from previous rounds and re-initialize the policy from its original supervised fine-tuned (SFT) checkpoint before each new iteration.

### Strengths
1. This paper systematically tests key design choices on reward computation, preference data collection and model initialization, providing a clear, structured understanding of how different components affect training stability and performance.
2. The paper is well-written with good presentation and analysis
3. The use of Maximum Mean Discrepancy (MMD) to measure the distributional differences between the proxy and the gold reward models helps explain the effect distribution shift. This offers a more nuanced view of overoptimization, revealing how the reward distributions diverge, especially at high KL values

### Weaknesses
My main concern is the model size—both for the language model and reward model. It's unclear whether these findings transfer to larger reward models, though intuitively the analysis should hold even with more capable models.

The work also relies heavily on the assumption of a good, static "gold" reward model. In practice, such a model rarely exists for most post-training tasks. It will evolve over time and carry its own misalignment with human preferences.

### Questions
1. The `Sample + Take last Policy` The strategy shows a dramatic performance collapse, which it never recovers from. Was this failure due to entropy collapse, or did the policy learn specific, unrecoverable adversarial behaviors that the updated reward model couldn't penalize effectively?

2. Do you hypothesize there is a model scale at which LITI would consistently outperform the robust `From SFT` strategy? If so, would it be because a more capable reward model provides better-calibrated gradients, making it easier for the policy to "unlearn" overoptimized behaviors from the previous step?

3. For a fixed compute budget, is it better to run more iterations with a cheaper data strategy (like `Take last Data`) or fewer, more expensive iterations using the full `Concat Data` approach? The results suggest `Concat Data` is superior, but how does that trade off with the number of iterations one can afford?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates reward-model over-optimization in *iterated RLHF*: within the iterative loop—preference data collection → reward model training→ Policy optimazation—it examines which design choices mitigate or amplify over-optimization. The authors run a comparative evaluation between a *gold reward* and a *proxy reward* on *AlpacaFarm* and report key findings for iterated RLHF.

### Strengths
1. Provides a detailed study of reward over-optimization, factorizing iterated RLHF into three stages and empirically exploring actionable components in each stage.
2. Introduces metrics such as *MMD* and *KL–reward curves* to analyze over-optimization phenomena.
3. Delivers thorough experimental analyses; the conclusions are insightful and offer practical guidance for related applications.

### Weaknesses
1. Beyond proximity to the gold reward, the paper should report testset metrics (e.g., pairwise accuracy) for the proxy reward across iterations to provide more comparable evidence.
2. Although the gold and proxy rewards differ substantially in parameter count, report their performance  on held-out test sets and on public benchmarks (e.g., RewardBench) may lead resuslt more clear.
3. Conclusions drawn from a single dataset may be biased; the paper should evaluate on more datasets and base models to assess generalization. It is also advisable to consider a stronger LLM-as-judge as the gold reward to provide a more robust supervision signal.

If the authors can provide additional experiments to address these concerns, I would be happy to raise my score.

### Questions
See above weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies reward model overoptimization in iterated RLHF. The authors define three key levers in the loop—(i) how preference data from prior rounds are reused, (ii) how to form the reward signal from multiple trained RMs, and (iii) policy initialization at each round—and evaluate their impact in a controlled AlpacaFarm setup with a “gold” RM as a stand-in for human labels. Methodologically, they also propose using distributional comparisons (MMD) between proxy and gold rewards (rather than only means) along the KL–reward curve during PPO training. Main findings: (1) overoptimization tends to decline across iterations as the proxy RM better matches the gold RM on-policy; (2) concatenating preference data across rounds is the most reliable way to curb overoptimization; (3) restarting from the SFT policy every round is the most robust (though less flexible) initialization; (4) combining RMs via ensembles/weight averaging gives modest gains with efficiency trade-offs; and (5) benefits plateau after ≈3 iterations, and some overoptimization remains even after four.

### Strengths
Well-scoped, decision-oriented study. The three knobs cover the practical choices teams actually debate; the recommendations are specific and replicable.

Concatenating preference data clearly helps. Strong and consistent gains vs. take-last/sample, especially in mid-KL regions where overoptimization tends to bite.

Policy resets matter. From-SFT avoids “digging the hole deeper”; recovering from an overoptimized policy is empirically hard—even with later iterations.

Distributional metric. The MMD view surfaces high-KL divergence that average scores miss, showing when iteration helps and when exploitation resumes.

Scale sensitivity checks. Showing that larger RMs (160M vs 70M) unlock more benefit for ensembling/WCO gives a plausible path as capacity grows.

### Weaknesses
Gold-RM surrogate limits external validity. A single fixed “gold” RM (and one dataset) can imprint its biases; real human-in-the-loop dynamics might differ (drift, noise, inconsistency).

Narrow task/model scope. Pythia-410M policies and 70M/160M RMs on AlpacaFarm only; conclusions might shift with stronger instruction-tuned policies, adversarial prompts, or safety domains.

Compute accounting is thin. We don’t see wall-clock/GPU hours per iteration/choice, nor inference overhead for ensembles/WCO vs. weight-averaging.

Initialization trade-offs underexplored. From-SFT is robust but may cap upside; LITI can improve with larger RMs, but guidance on safe settings (η, early-stop) is light.

Theory is mostly appendix-level and aspirational. The performative-prediction framing is helpful but doesn’t yield predictive thresholds for when iteration ceases to help.

### Questions
Cost vs. benefit. Can you report GPU hours/tokens/s per iteration for (a) concat-data + take-last RM + From-SFT, (b) concat-data + ensemble, and (c) WCO and weight-avg? Practitioners need Pareto curves (gold score vs. cost).

Human variance. Any pilot with noisy/biased “gold” RMs (or a mixture of RMs) to emulate annotator disagreement? How do the design choices change under label noise or drift?

When not to reset. Are there regimes (larger RMs, conservative KL schedules, early-stop) where LITI reliably beats From-SFT after >4 iterations without catastrophic KL compounding? Concrete η schedules would help.

Beyond AlpacaFarm. Do the rankings (concat > sample ≈ last; From-SFT best) hold on a safety/toxicity or factuality workload with verifiable metrics, or under length-bias stress tests?

RM calibration. Do ensembling/weight-avg improve calibration (e.g., STARC, ECE/Brier of pairwise prefs) or only gold-score means?

MMD operationalization. How sensitive are your MMD conclusions to kernel choice/bandwidth and to KL bucketing? Could you release a minimal script so others can reproduce the curves on their pipelines?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a comprehensive study of reward model overoptimization in iterated RLHF. It finds that while overoptimization decreases across iterations as reward models better approximate human preferences, performance gains plateau and some overoptimization persists. The authors identify three key design choices—how preference data is managed, how reward models are combined, and how policies are initialized—that significantly impact outcomes.

### Strengths
1. The paper analyzed the overoptimization problem of iterative RLHF very thoroughly, including empirical study on its progression across multiple training rounds, the impact of key design choices like data aggregation and policy initialization, and the trade-offs between robustness and optimization flexibility.

2. The paper provides a novel, theoretical perspective to study overoptimization.

### Weaknesses
1. The paper lacks testing on standard reward benchmarks.

2. The paper's content is not organized enough to understand the whole process of iterative RLHF design choices and evaluating overoptimization.

### Questions
1. Are the findings generalizable to other models and other datasets?

2. If a model is trained using the principles discovered, what is the performance against other iterative RLHF designs?

I am not sure I have enough knowledge to give a deeper insight of the paper.

### Soundness
3

### Presentation
3

### Contribution
3
