# Symbolic Regression with Self-Supervised Heuristic Beam Search

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Symbolic Regression (SR) aims to discover simple and interpretable mathematical expressions that explain observed data, making it a powerful tool for scientific discovery. In this work, we introduce a conceptually simple SR method that is both sample-efficient with respect to observed data points and self-supervised on large-scale synthetic data. By design, our approach favors parsimony, yielding interpretable and concise expressions. We focus on problems with exact solutions, evaluating our method on datasets containing physical laws and dynamical
equations. Our results demonstrate that combining beam search with a learned heuristic achieves competitive performance compared to existing methods in SRBench. Additionally, our approach effectively handles expressions with constants, a common challenge in the SR field. Finally, we provide a comprehensive scalability analysis across four key dimensions: (i) expression length, (ii) number of variables, (iii) number of domains, and (iv) number of observed data points.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents HTSSR, a novel symbolic regression method combining beam search with a self-supervised heuristic. The approach is interesting, and the focus on interpretability and parsimony is commendable. The scalability analysis across several dimensions is a valuable contribution. However, I have several major concerns regarding the evaluation methodology, the scope of the experiments, and the claims made, which I detail below.

### Strengths
The core idea of using a self-supervised learned heuristic to guide a beam search is both novel and elegant. It represents a clear departure from common paradigms, such as using fitness-to-data as a direct reward signal in reinforcement learning or generating expressions token-by-token with a generative model. The concept of learning a "precedence" relation between expressions is a powerful way to inject learned knowledge into the search process.

### Weaknesses
**1. Major Concerns​**

​**1.1. Evaluation Metric: Over-reliance on Symbolic Solution Rate (SSR)​​**

The authors argue against using R² for evaluation, favoring the Symbolic Solution Rate (SSR). While SSR is appropriate for problems with known ground truth, it has significant limitations for real-world, black-box problems where the true expression is unknown. Furthermore, SSR is an extremely stringent metric. It may unfairly penalize expressions that are functionally equivalent or very close to the true solution, which could be easily corrected with minor adjustments based on domain knowledge by human or some other mechanisms. A more holistic evaluation philosophy should prioritize achieving high fitting accuracy first, and then, under that constraint, minimize expression complexity and maximize out-of-distribution generalization. Relying solely on SSR presents an idealized and potentially unrealistic assessment of practical utility.

​**1.2. Limited Noise Robustness Evaluation​**

The tested noise levels (0.0 and 0.01) are too low to adequately demonstrate the method's robustness. Pre-trained models can be sensitive to noise; small noise levels may have a negligible impact, while performance can degrade significantly as noise increases. The results in Figure 6 suggest that, apart from AI Feynman, HTSSR's performance drops noticeably even with the minimal 0.01 noise level, indicating potential vulnerability to data imperfections. A more convincing demonstration of robustness would require testing against higher noise levels.

​**1.3. Scalability Analysis is Incomplete​**

The scalability analysis, while a strength of the paper, misses critical dimensions:

​Expression Length:​​ The tested maximum length of 10 symbols is insufficient. With higher input dimensions (e.g., 6 variables), even a simple additive model can approach this length. The method's performance on longer, more complex expressions remains unverified.

​Variable Dimensionality:​​ The search space grows combinatorially with the number of variables. The paper does not sufficiently investigate how performance scales as the number of variables increases beyond 3 or 4. This is a crucial aspect of scalability, as each additional variable significantly expands the search space and likely demands more training data.

**​1.4. Claims of Sample Efficiency are Unsubstantiated​**

The introduction claims the method is "sample-efficient," but no experimental evidence is provided to support this claim. The experiments in Section 4.1 analyze the effect of the number of evaluationdata points on search success, not the efficiency of the trainingprocess. Furthermore, the appendix indicates non-trivial time开销, which seems to contradict the notion of high sampling efficiency. An experiment demonstrating effective learning with limited training data or a comparison of training data requirements against other methods is needed.

​**2. Methodological Concerns​**
​**2.1. Risk of Overfitting in Pre-training​**

HTSSR relies on pre-training a heuristic model on synthetic data, effectively teaching it "correct" expression paths. This approach carries the risk of poor out-of-distribution generalization. If the synthetic data distribution does not align well with the target problems, the pre-trained heuristic may misguide the search, leading to significant performance drops on unseen equation types or domains.

### Questions
The questions are described in section "Weaknesses"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes HTSSR (HeurisTic beam Search for Symbolic Regression): a self-supervised learned heuristic classifier that guides beam search through the expression space to find short, interpretable formulas from data, including formulas with free constants. It evaluates on SRBench (Feynman/Strogatz) and reports competitive performance, plus an analysis of scalability with respect to expression length, number of variables, number of domains, and number of samples.

### Strengths
Clear, modular design. Expressions are generated by a prefix grammar; the learned heuristic performs a simple precedence (pairwise ordering) classification instead of modeling discrete structures directly.
Solid engineering. The paper introduces Sort-Diff and Digit numeric feature transforms to stabilize training under non-differentiability and scale/units issues.
Efficient constant fitting. Free constants are optimized with a small Levenberg–Marquardt inner loop, typically converging in few steps.
Stricter evaluation. It emphasizes symbolic solution rate (SSR) over loose R^2 thresholds and discusses shortcomings of prior evaluation practices.
Sample efficiency & robustness. Can recover ground-truth formulas with as few as ~10 samples and shows better robustness to moderate noise on Strogatz tasks.
Comprehensive scalability study. Systematically examines how success rate changes with length, variables, domains, and samples, with useful practical takeaways.

### Weaknesses
Attribution of gains is unclear. The paper notes that improvements from Sort-Diff may partly stem from increased parameter count, making ablations hard to interpret.
Strong dependence on the primitive set. Results and “difficulty” trends are assessed under a fixed set of primitives; changing this set may alter conclusions, limiting external validity.
Length scalability bottleneck. Success rate drops sharply for expressions longer than ~10 under fixed budgets, constraining practical problem complexity.
Unequal trial counts vs. baselines. HTSSR is run once per task whereas baselines often run 10×; confidence intervals are not directly comparable and may bias rankings.
Sample-efficiency test is narrow. The detailed sample-size curve is shown on a single domain; broader validation would strengthen the claim.
Multi-domain trade-offs. Training across more domains can degrade target performance in some settings; reuse across domains is not conclusively demonstrated.
Stopping/acceptance tolerance sensitivity. The relative-error tolerance (e.g., 10^{-3}) is fixed to accommodate noise; its effect on rankings is not fully analyzed.
Robustness tweaks may bias numerics. Safety operators and clipping for stability could distort certain physical relationships; more sensitivity analysis would help.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Self-Supervised Heuristic Beam Search (SS-HBS) to improve the efficiency and structural generalization of Symbolic Regression (SR).
The method learns a heuristic function that estimates the probability of a partial expression being structurally valid.
This heuristic is trained in a self-supervised manner and used to guide the beam search process, aiming to focus exploration on promising symbolic structures.

Empirical results on SRBench show improvements over prior SR methods.

### Strengths
- The idea of integrating structural knowledge into the beam search process is interesting and could make SR exploration more efficient.
- The self-supervised heuristic formulation is conceptually appealing when explicit supervision on expression quality is unavailable.
- The proposed method achieves improvements on noisy data.

### Weaknesses
1. Lack of theoretical and experimental comparison with generative Transformer models such as E2ESR.

   In models like E2ESR, the generative probability can directly serve as a heuristic for search.
   The proposed heuristic seems conceptually similar to this posterior, yet the authors do not justify why another heuristic function is necessary.

2. Insufficient comparative analysis.

    No comparison with alternative self-supervised learning strategies (e.g., synthetic-data-based pretraining).
    Ablation studies are limited, making it difficult to isolate which component contributes most to the improvement.

3. Weak theoretical grounding.

   It is unclear whether the proposed method is meaningful or empirically justified.
   Sections 3 is too abstract; the motivation for the loss function and input representation is not well explained.
   The appendices contain only minimal implementation details, making full reproducibility difficult.

### Questions
I would like the authors to address the points listed under Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors propose a sample-efficient symbolic regression method, named HTSSR, which aims to recover exact expressions from observed data points. The approach begins with a source expression and progressively expands it into more complex expressions by applying a set of predefined rules. To identify the most promising expansion path, the authors apply a beam search algorithm guided by a heuristic model trained to estimate the likelihood that an expression can be expanded to the ground-truth expression. The authors analyze the scalability of the proposed approach and evaluate its performance in recovering exact expressions on two white-bow problem sets, Feynman and Strogatz.

### Strengths
The authors propose a simple yet effective symbolic regression method that combines beam search with a heuristic, achieving sample efficiency while yielding interpretable and concise expressions. Experiments on two white-box problem sets from SRBench demonstrate the method’s ability to discover exact expressions from a small set of observation sample points. Further experiments also demonstrate the scalability of the proposed approach in terms of expression length, number of variables, number of domains, and number of observed data points.

### Weaknesses
**W1: Insufficient literature review and weak novelty.** The proposed method is highly similar to SR4MDL[1], a work published at ICLR 2025, which also starts from a source expression, expands it into more complex formulas by applying a series of predefined rules, and employs a heuristic-guided Monte Carlo tree search to identify an optimal expansion path. In SR4MDL, the heuristic is provided by a pre-trained Transformer that learns to predict the minimal number of transformations required to convert a given expression into a target one, which seems to be comparable to the heuristic model proposed by the author. Furthermore, SR4MDL is also evaluated on the Feynman and Strogatz problem sets to assess its symbolic solution rate. Given the strong methodological and experimental overlap, the absence of any discussion or performance comparison with SR4MDL in the present work appears unacceptable and concerning.

**W2: Insufficient experiments and limited applicability.** Although the authors claim that their method is designed for discovering exact expressions, in most symbolic regression scenarios, the underlying expression is either unknown or may not even exist. Therefore, in addition to evaluations on white-box datasets, the proposed method should also be tested on black-box datasets to assess its ability to discover expressions balancing accuracy and complexity. Without demonstrating its ability to identify concise and accurate formulas from data with unknown underlying relationships, its practical applicability cannot be determined.

**W3.** Given that the proposed method involves not only a search procedure but also the training of a heuristic model, although the authors provide the runtime in the appendix, the time required to discover the target expressions should be compared with other SR methods and discussed in more detail in the main text.

[1] Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length (ICLR, 2025)

### Questions
Q1. Although the authors conduct a series of ablation studies, it seems that no experiments ablate the heuristic model itself from the beam search. Could the authors provide an ablation experiment to demonstrate the contribution of the heuristic model to the performance?

Q2. In Line 250, instead of adopting the digit transform approach proposed in (Kamienny et al., NeurIPS 2022) and widely adopted by many SR works, the authors design their own digit transform approach. Could the authors provide a comparison of their performances?

Q3. In Figure 2, what does $V^\*(E)$ represent? I understand that $E$ denotes the expression to be evaluated, but it seems that the definition of $V^\*$ is not provided in the text.

Q4. In the input to the Heuristic Model, how are the numerical constants in $E$ determined? If the constants are directly fitted on $\mathcal{D}$ using the method described in Section 3.4, wouldn’t that cause expressions that can, in fact, expand to F to be incorrectly classified as 'cannot'? For example, if $F = 2x + 1$ and $E = □ \times x$, numerical fitting may yield $E = 2.5x$, leading the model to judge that $E$ cannot be expanded into $F$ since $2.5 ≠ 2$, even though such an expansion is theoretically possible.

### Soundness
3

### Presentation
1

### Contribution
1
