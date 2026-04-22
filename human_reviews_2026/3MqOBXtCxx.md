# Cost-Optimal Active AI Model Evaluation

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
The development lifecycle of generative AI systems requires continual evaluation, data acquisition, and annotation, which is costly in both resources and time. In practice, a desire for rapid iteration often makes it necessary to rely on synthetic annotation data because of its low cost, despite the potential for substantial bias. In this paper, we develop a rigorous theoretical framework for novel, cost-aware evaluation pipelines that actively balance the use of a cheap, but often inaccurate, weak rater---such as a model-based autorater that is designed to automatically assess the quality of generated content---with a more expensive, but also more accurate, strong rater such as a human annotator. Building on recent work in active and prediction-powered statistical inference, we theoretically derive a family of cost-optimal policies for allocating a given annotation budget between weak and strong raters so as to maximize statistical efficiency. 
Next, using synthetic and real-world data, we empirically characterize conditions under which these types of policies can yield significant improvements over classical methods. Finally, we find that practical approximations of the theoretically optimal policies 
can achieve the same estimation precision at a far lower total annotation budget than standard evaluation methods, especially in tasks where there is high variability in the difficulty of examples.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a framework for the so-called cost-optimal active evaluations, which encompasses a family of annotation policies designed to minimize expected error under a given annotation budget.

### Strengths
- The paper presents numerous mathematical propositions to support its claims, demonstrating strong theoretical foundations.
- The motivation behind the work is clearly articulated.
- Key findings are well-highlighted. In particular, the conclusion on line 223—“...active learning can help if the conditional squared error of G has significant variance”—offers valuable guidance for future research in developing active learning strategies.

### Weaknesses
Several expressions are vague and fall short of academic standards.

1. For instance, the phrase “...by optimizing everything” in line 086 is unclear—what does “everything” refer to?
The term “evals” used in line 105 and elsewhere is not standard English and lacks a clear definition. The sentence “We now describe our methods for constructing active, cost-optimal evals” (line 105) is difficult to interpret—what exactly are “evals”?
In line 117, H and G are defined as h(X) and G(X), which are described as ratings in line 116. However, line 118 states “querying H and G costs c_h and c_g”, which translates to “querying ratings costs...”—a phrasing that is not easily interpretable.


1. Some mathematical steps are omitted, making the evaluation of the work challenging. For example, the proof of Equation 2 is relegated to the appendix, and the transition from line 726 to line 728 lacks clarity.
2. The term “cost-optimal” used in the title and throughout the paper is not entirely convincing or appropriate. As acknowledged by the authors in line 480—“...annotation policies that are optimal in theory are distribution-dependent...”—this suggests that such optimal policies may be unattainable due to inherent uncertainties. Therefore, the proposed framework does not achieve a truly optimal solution, but rather an optimal solution subject to specific constraints. These constraints should be clearly emphasized, as the current phrasing implies a globally optimal solution.
3. To substantiate the claim in line 477—“We derive annotation policies that are optimal in the sense of minimizing expected error under annotation budget constraints”—a brute-force experiment exploring various combinations of examples and demonstrating that the proposed method achieves the best or ceiling performance would be necessary.

### Questions
1. What is the justification for using ξ_t​ over π_t​ in Equation 1? What are the implications of this choice?
2. Should coreset-based methods be included as one of the baselines? If not, why?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a theoretical and empirical framework for cost-optimal active evaluation of generative AI systems. The authors tackle a very practical problem: evaluating large models is expensive, and existing hybrid setups (like combining cheap model raters with expensive human raters) often lack rigorous cost-aware allocation.
To address this, the paper derives policies for optimally allocating annotation budget between weak and strong raters — balancing cost and accuracy through statistical optimization. It extends prediction-powered inference (PPI) and active statistical inference to derive (1) an optimal random sampling rate and (2) an optimal active policy that depends on task-specific uncertainty.

### Strengths
1. The problem—cost-aware AI evaluation—is timely, practical, and underexplored. The authors correctly identify inefficiencies in current model evaluation practices that rely heavily on costly human or LLM raters.
2. The extension of prediction-powered inference with explicit cost constraints is technically sound. The derivation of closed-form policies (Propositions 1–2) is clear and builds on well-established statistical theory.
3. The Gaussian/Bernoulli experiments in Section 3 are carefully designed to test key intuitions (e.g., dependence on rater error, heteroskedasticity, and cost ratio). The figures are clean and reinforce the theoretical claims.
4. Applying the framework to Chatbot Arena evaluations shows that cost-optimized sampling can indeed save budget while maintaining accuracy. The setup is realistic and relevant to modern LLM benchmarking.

### Weaknesses
1. The real-world experiments are narrow. Most results are on one dataset (Chatbot Arena) with two scenarios, both focused on text-based preference evaluations. There’s little diversity in task type or domain (e.g., no multimodal or structured data). The empirical results, while consistent, are modest—often showing ~40–50% budget savings under ideal transfer, which may shrink with realistic uncertainty estimation.
2. While theoretically elegant, the framework’s impact on real-world evaluation pipelines is unclear. Implementing cost-optimal policies requires calibration, pilot estimation, and maintenance that may offset cost savings in small-to-medium-scale evaluation scenarios.

### Questions
1. How do these methods perform when the weak rater is itself biased rather than merely noisy?
2. Are there concrete examples of how much “burn-in” cost is acceptable before cost savings emerge?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper develops a theoretical framework for cost-optimal evaluation of generative AI models. It addresses the high cost of using accurate "strong raters" (like humans) by creating policies to actively balance their use with cheap but inaccurate "weak raters" (like model-based autoraters).

Building on prediction-powered inference, the authors derive cost-optimal policies that, given a fixed budget, decide when to pay for the expensive rater to maximize statistical efficiency. The theoretically optimal active policy queries the strong rater most often when the weak rater is most uncertain.

Since the optimal policy's parameters are unknown in practice, the authors test estimation methods like "policy burn-in" (using the first 200 samples) and "policy transfer" (using a related dataset). Experiments on synthetic data and real-world benchmarks (like Chatbot Arena) demonstrate that these methods can achieve the same estimation precision for a fraction of the cost, with the greatest savings seen in tasks with high variability in example difficulty.

### Strengths
1. The paper provides a rigorous theoretical framework for active evaluation, extending beyond prior work. Instead of just improving efficiency for a fixed number of expensive annotations, it derives truly cost-optimal policies ($\pi_{random}$ and $\pi_{active}$) that explicitly solve for the best sampling strategy to minimize error given a fixed monetary or computational budget.
2. The work addresses a critical bottleneck in the GenAI lifecycle: the high cost of evaluation. By providing a principled way to combine cheap autoraters with expensive human labels, the framework offers a practical path to achieving high-precision estimates at a much lower total annotation cost.

### Weaknesses
1. The theoretically-derived policies, $\pi_{random}$ and $\pi_{active}$, depend on several distributional properties like $Var(H)$, $MSE(H,G)$, and the conditional error $u(x)$. Since these are unknown in a real-world setting, the policies cannot be used out of the box. The paper's practical solutions (burn-in and transfer) are approximations that either require a separate, related dataset or incur an initial "burn-in" cost before any savings can be realized.
2. The benefit of the active policy over the simpler random policy hinges on an accurate estimate of the conditional error, $u(x)$. The paper's own experiments show a significant performance gap between the practical "Active" policy and the "Oracle" policy, which knows the true error. This implies that the current methods for estimating uncertainty are "far from perfect" and are a primary bottleneck limiting the practical gains.

### Questions
1. Your practical "burn-in" policy (A2) uses a fixed $n_b=200$ expensive samples to estimate the policy parameters. This initial cost is a critical part of the total evaluation budget. Could you provide a sensitivity analysis showing how the performance of $\pi_{active}$ and $\pi_{random}$ changes for different values of $n_b$? It seems there would be a tradeoff: a small $n_b$ leads to poor parameter estimates, while a large $n_b$ defeats the purpose of saving costs.
2. The main benefit of the active policy over the random one depends on an accurate estimate of the conditional error $u(x)$. You show a significant gap between your "Active" policy and the "Oracle" policy, implying that the $u(x)$ estimates are "far from perfect". For the binary tasks, you used the heuristic $u(x) = G(1-G)$, which assumes the weak rater (G) is a well-calibrated probability. Did you experiment with other methods for estimating $u(x)$ that might be more robust?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper develops a cost-aware framework for hybrid evaluation that mixes a cheap, weak rater with an expensive, strong rater, aiming to estimate the strong rater’s mean judgment under a budget. It derives (i) a closed-form optimal random sampling policy under cost constraints and (ii) an input-adaptive active policy with clipping and a threshold, then studies practical instantiations via policy transfer and burn-in estimation. Experiments show budget savings and reduced MSE compared with always using the strong rater.

### Strengths
1. The objective of minimizing estimator error subject to an annotation budget is formalized, yielding a closed-form $ \pi_{\text{random}} $ in terms of costs and weak-rater MSE, and an adaptive $ \pi_{\text{active}} \propto \sqrt{u(x)} $ with principled clipping to respect $ \pi(x)\in(0,1] $ with clear derivation. 

2. The transfer and burn-in strategies provide workable recipes, and the paper reports effective budget and cost-savings curves that are easy to interpret.

3. Experiments on real-data seem to match the theory’s qualitative predictions.

### Weaknesses
1. The method extends prediction-powered/active inference by optimizing cost-constrained policies and addressing clipping, but much of the estimator form and sequential setup follows prior work.

2. The active policy depends on a non-convex 1-D optimization over $ \tau $, and the paper does not report sensitivity to $ \tau $, mis-estimated $ u(x) $, or misspecified cost ratios, which are likely in practice.

3. The burn-in approach assigns the first $ n_b $ items to the strong rater to estimate parameters, which reduces net gains at small budgets. More discussion on adaptive burn-in size or warm-start reuse across tasks would be useful.

4. In Chatbot Arena experiments, the strong label is also from LLMs, i.e., Gemini 1.5 Flash majority vote. A human-grounded subset would better validate external correctness.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
