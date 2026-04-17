# Don't Walk the Line: Boundary Guidance for Filtered Generation

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Generative models are increasingly paired with safety classifiers that filter harmful or undesirable outputs. A common strategy is to fine-tune the generator to reduce the probability of being filtered, but this can be suboptimal: it often pushes the model toward producing samples near the classifier’s decision boundary, increasing both false positives and false negatives. We propose *Boundary Guidance*, a reinforcement learning fine-tuning method that explicitly steers generation away from the classifier’s margin. On a benchmark of jailbreak and ambiguous prompts, *Boundary Guidance* improves both the safety and the utility of outputs, as judged by LLM-as-a-Judge evaluations. Comprehensive ablations across model scales
and reward designs demonstrate the robustness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Boundary Guidance (BG), a reinforcement-learning-based fine-tuning approach for generative models paired with downstream safety classifiers. The core idea is to explicitly steer generations away from the decision boundary of the classifier, rather than minimizing rejection probability only. By doing so, the approach aims to reduce both false positives and false negatives in filtered generation systems.

The authors present a decision-theoretic framework that formalizes how utility is minimized near classifier boundaries and derive a boundary-avoiding reward function combining helpfulness and safety signals. Empirical results across multiple model families (Qwen2.5 and Gemma) and sizes (0.5B–14B) demonstrate consistent improvements in both helpfulness and harmlessness as evaluated by GPT-4.1.

### Strengths
This paper focuses on improving the safety of generative models, which is an important topic in this field. The high-level idea is easy to grasp and understand.

### Weaknesses
1. [Quality] Limited baselines: the experiments only compare with the base model, which is far from sufficient for validating the effectiveness of the methods.

2. [Quality] Too heuristic: the design of the algorithm is too heuristic, which lacks both theoretical justifications and empirical validation via ablation studies.

3. [Clarity] Figure 2, 3: the naming of models are inconsistent, e.g. "qwen25-0.5b-guard_only", "Qwen2.5-0.5B-instruct", which poses difficulties for readers to quickly grasp the point of the figures.

4. [Clarity] Messy organization: Section 5 describes the main setup of the experiment, but is titled "Experimental Evidence for Boundary Guidance".

### Questions
* lines 169-171, Equation 2: If $t(x,y)$ is the probability of unsafe outputs (lines 160–161), and the goal is to minimize likelihood, I think the utility should be defined as $-\log t(x,y)$, not $-t(x,y)$, since the ultimate probability is the product of each event instead of the sum, if we consider different samples to be independent from each other, as in the conventional Maximum Likelihood Estimation (MLE) paradigm.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a new method, Boundary Guidance, that improves existing RL-techniques for fine-tuning models towards safer outputs. Existing methods might lead to examples close to the classifier's margin, which their method discourages. They carry out extensive experiments and ablations to demonstrate the utility and robustness of their approach.

### Strengths
- clear motivation and problem framing: as generators are trained independently of downstream safety filters, that is causing outputs near the classifier boundary
- simple and effective idea (introducing a penalty for being close to the classifiers boundary)
- strong empirical results that show Pareto improvements
- thorough ablations

### Weaknesses
- evaluation depends heavily on LLM-as-a-judge --> human ablation would be useful
- assumes that classifier probabilities near the extremes are reliable and that the decision boundary is meaningful
- limited dataset diversity (esp. no long-context scenarios)

### Questions
- How does Boundary Guidance perform under adaptive adversarial attacks where the attacker optimizes specifically to evade the classifier confidence signal?
- How sensitive is the performance to the specific safety classifier used? What about a "worse" classifier? 
- Have you conducted any human evaluations to validate the LLM-as-a-judge setup?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Boundary Guidance, a reinforcement learning fine-tuning method for improving compound safety systems where generative models are paired with downstream safety classifiers. The method stems from the insight that standard fine-tuning approaches often push models to generate outputs near classifier decision boundaries, increasing false positives and false negatives. Boundary Guidance steers generation away from these decision boundaries using a reward function that combines utility signals from reward models with safety classifier confidence scores. Experimental results show improvements in safety and utility across multiple model scales on jailbreak and ambiguous prompt benchmarks.

### Strengths
- The method focuses on compound systems rather than isolated model training, which aims to address the gap in current safety studies.

- The results demonstrate consistent Pareto improvements across multiple model architectures and scales. The experiments include various benchmarks and an LLM-as-a-Judge evaluation. The ablation experiments help understand the method components.

### Weaknesses
- The method makes strong simplifying assumptions, such as the constant utility $u(x,y)$ and binary safety classification, which may not hold in real-world scenarios. The analysis also assumes perfect classifier calibration and does not discuss how the approach performs when classifiers are biased or when the decision boundary itself is poorly positioned.

- The evaluation uses GPT-4.1 as the sole judge for both helpfulness and harmfulness, which can introduce potential evaluation bias, especially when fine-tuning against other LLM-based safety classifiers. The evaluation is also limited to a single benchmark focused on jailbreaks and harmful prompts. Many of the improvements are relatively small. 

- The approach requires training with three models, which may be computationally prohibitive.

### Questions
- How does the method perform when the classifiers are miscalibrated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the challenge of balancing safety and utility in the design of algorithms for safe compound systems. With a simple utility-safety model, the paper argues that the utility tends to diminish near the safety-only decision boundary, which constrains helpfulness and motivates incorporating utility directly into the reward formulation. To avoid explicitly estimating the trade-off coefficient (the $\lambda$ term) between safety and utility, motivated by the fact that models with low confident safety assessment tend to perform poorly, the paper proposes a reward function that includes a direct utility component, while safety is enforced through a boundary-maximizing objective that encourages the model to be confident about the safety evaluation (away from the threshold of 0.5, which represents uncertainty in safety evaluation). Empirical results from fine-tuning models of various scales show consistent improvements in helpfulness while simultaneously reducing harmfulness, validating the effectiveness of the approach.

### Strengths
1. The question of how to bring the aspect of utility into the safety aware compound systems is interesting.
2. The formulation in Section 3 is good and well-written.

### Weaknesses
1. The boundary objective in Eq (3) lacks detailed explanation and motivation. Consider including a discussion paragraph on how it is derived/proposed.

2. The connection of Eq (3) to Eq (2) and the previous discussion almost seems unrelated. The final objective is basically a utility maximization with a boundary-maximizing objective for safety. It is critical to explain the connection of Eq (3) to Section 3.

3. It seems that there is no way to fully decouple the effect of $u(x,y)$ and $t(x, y)$, which makes the algorithm work well, i.e., $u(x, y)$ (utility/helpfulness) still captures some effect of $t(x, y)$ (safety/harmlessness). Otherwise, why is there a utility maximization term when the output is blocked, i.e., when $t(x,y) \ge 0.5$. This is contradictory to the Eq (2) and footnote 1.

4. No empirical comparison with prior work.

### Questions
Questions:

1. Can you please confirm whether the paragraph after Proposition 1 is comparing the aggregate expected reward in a scenario with an individual utility function, with the one without any individual utility function? If so, please provide the exact expected reward similar to Eq (1), when there is no utility function $u(x,y)$. Coming back to the question: why is having a **local** minima at $\tau$ important?

2. Can you confirm that Base refers to the evaluation of responses by the base model? If so, please compare these with Base+Guard (filtering with $t(x,y)$), which does not require training.

Typos: 
1. Line 045: the approach $\rightarrow$ these approaches.
2. Figure 2: Qwen25 $\rightarrow$ Qwen2.5,

### Soundness
2

### Presentation
2

### Contribution
2
