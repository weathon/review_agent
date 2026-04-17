# Understanding the Role of Training Data in Test-Time Scaling

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Test-time scaling improves the reasoning capabilities of large language models (LLMs) by allocating extra compute to generate longer Chains-of-Thoughts (CoTs). This enables models to  tackle more complex problem by breaking them down into additional steps, backtracking, and correcting mistakes. Despite its strong performance--demonstrated by OpenAI's o1 and DeepSeek R1, the conditions in the training data under which long CoTs emerge, and when such long CoTs improve the performance, remain unclear. In this paper, we study the performance of test-time scaling for transformers trained on an in-context weight prediction task for linear regression. Our analysis provides a theoretical explanation for several intriguing observations: First,  at any fixed test error, increasing test-time compute allows us to reduce the number of in-context examples (context length) in training prompts. Second, if the skills required to solve a downstream task are not sufficiently present in the training data, increasing test-time compute can harm performance. Finally, we characterize task hardness via the smallest eigenvalue of its feature covariance matrix and show that training on a diverse, relevant, and hard set of tasks results in best performance for test-time scaling. We confirm our findings with experiments on large, nonlinear transformer architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the relationship between test-time scaling and training data in the context of in-context weight prediction for linear regression. The authors develop a theoretical analysis based on linear self-attention (LSA), derive convergence properties, propose a definition of task hardness, and formulate an optimization framework for task selection. Experiments are conducted with both LSA and GPT-2 architectures to validate the theoretical trends.

### Strengths
1. Provides a systematic theoretical analysis with clear derivations of convergence and error bounds.
2. Introduces a task hardness measure based on the covariance matrix, which is simple and interpretable.
3. Offers a formal perspective on task selection, framed as an optimization problem.
4. Addresses the timely topic of test-time scaling, which has attracted significant attention in the community.

### Weaknesses
1. Overly idealized setting: The core analysis relies on linear regression and LSA, which are far from realistic large-scale model training.
2. Limited GPT-2 validation: Although GPT-2 experiments are included, they remain confined to the same synthetic weight prediction task, limiting external validity.
3. Simplistic hardness definition: The hardness measure depends only on the smallest eigenvalue, reflecting the limitations of the simplified setting and failing to capture the complexity of real tasks.
4. Insufficient experimental support: The experiments are small-scale and restricted to synthetic tasks, without evidence from realistic reasoning benchmarks.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a theoretical lens for studying how training data influences test-time scaling. They propose a simple linear regression prediction task in which the model repeatedly predicts the weight vector given X,y pairs via chain of thought. They demonstrate that increasing test-time compute reduces the in-context examples requirement during training. They also characterize factors that influence the test-time scaling curve, such as training data hardness and diversity of training data.

### Strengths
This is a well-written paper. I find the intuitions on task hardness, task selection, and diversity to be particularly helpful in understanding the provided theorems.
* The definitions of task hardness (as a ratio of the sum of variances to minimum eigenvalue) is particularly interesting and to my knowledge quite novel. As the paper states, "an easy task is one that relies on a few dominant skills... while a hard task draws on many skills, reflected in a long-tailed spectrum". In the context of LLMs, task difficulty is usually defined as length/complexity of a problem. The proposed definition provides a new axis of task complexity that relates feature conditioning and data geometry.
* Although the primary focus of the paper is to understand how training data affects test-time scaling curves, the framework naturally explains when overthinking occurs. This interpretability demonstrates the elegance and generality of the theoretical analysis.

Overall, the paper advances our mechanistic understanding of why test-time reasoning improves model performance, and has important consequences for designing the appropriate training data for language models.

### Weaknesses
The analysis is mostly confined to the synthetic linear regression task setup. It is not immediately clear how well these definition of task hardness translate to natural language reasoning tasks.

### Questions
The theory predicts that the generalization error decays (roughly) as $\frac{1}{n^{2k}}$ - does this scaling hold even for extremely small $n$? In practice, if the model encounters too few examples during training, there could be an error floor that test-time scaling cannot overcome. Is there a theoretical or empirical lower bound on $n$ below which test-time scaling becomes ineffective or unstable?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper discusses the topic of test-time scaling and chain-of-thought prompting for transformers, examining when and why allocating extra compute at inference improves performance.
Motivated by recent systems (e.g., OpenAI’s o1 and DeepSeek R1), the authors analyze transformers trained for in-context weight prediction on linear regression to clarify the training-data conditions under which long chains of thought emerge and help.
They show theoretically that, for a fixed test error, increasing test-time compute can substitute for longer training prompt context length.
They also find that if the relevant skills are insufficiently represented in the training data, additional test-time compute may harm performance.
The work formalizes task hardness via the smallest eigenvalue of the feature covariance matrix and argues that training on diverse, relevant, and hard tasks yields the best gains from test-time scaling.
The analysis connects chain-of-thought prompting to multi-step (pseudo)-Newton’s method and derives scaling laws governing the interaction among test-time compute, context length, and task diversity.
An optimal task-selection strategy for multi-task training is proposed and validated on linear self-attention models and GPT-2, with experiments extending to large, nonlinear transformer architectures.
The authors note limitations: the theory focuses on linear regression and single-layer linear self-attention, and future work should extend to nonlinear data generation and transformers with nonlinear activations.

### Strengths
* A coherent framework linking theory and experiments  
The paper examines the effectiveness of test-time scaling (longer Chains of Thought) with a back-and-forth between theory (analytical results and scaling laws) and empirical validation (linear self-attention and GPT-2 / larger nonlinear transformers), which strengthens the credibility of its claims.

* Mechanistic identification: CoT as multi-step optimization (pseudo-Newton’s method)  
By mapping the inference process to iterative optimization, the paper makes why CoT works more transparent; this unifies prior empirical observations and informs design choices (e.g., preconditioning and stopping criteria).

* A theoretical trade-off between test-time compute and training-time context length  
Under a fixed test error, the paper shows that increasing test-time compute can substitute for a longer training context length, providing a principled basis for allocating budget between training and inference.

* A principled metric of task hardness via the smallest eigenvalue of the feature covariance  
The work formalizes “hardness” spectrally, enabling computable diagnostics that connect to scaling laws, dataset construction, and benchmark selection.

* Clear conditions for when overthinking can be harmful  
The paper specifies that when the necessary skill directions are underrepresented in the training data, increasing test-time compute can degrade performance, offering actionable guidance for safe deployment and monitoring.

* Scaling laws that provide predictability  
By deriving scaling relationships among test-time compute, context length, and task diversity (via the features’ covariance spectrum), the paper moves beyond observation to testable predictions that facilitate replication and extension.

### Weaknesses
* Narrow applicability of the theory  
I understand that the authors explicitly state the limitations (linear regression; single-layer linear self-attention) and future extensions (nonlinear data generation; transformers with nonlinear activations) in the Conclusion and Limitations sections.
However, the main claims (CoT ≈ pseudo-Newton’s method, scaling laws, etc.) are not guaranteed to carry over to recent advances in LLMs, such as different model architectures, like LLaMA-base, GPT-OSS-base, and MoE-based models.
This is extremely important for the contribution of this paper. It would be better to discuss , as part of the main content, how the proposed method is promising for extension to different model configurations, even if confidence is currently limited.

* Dependence on the task “hardness” metric  
In Sec. 3, the paper defines hardness as $\mbox{tr} (\Lambda) / \lambda_{min}(\Lambda)$ for the task covariance $\Lambda$, and claims that hardness is based on the smallest eigenvalue of the feature covariance.
However, it seems that relying on a single spectral indicator may be sensitive to preprocessing or representation choices and may not fully capture linguistic complexity or reasoning modes.

* Detectability of the conditions under which overthinking is harmful  
In Sec.5, the paper reports observations in which longer CoT hurts.
While the condition is described, a practical method to detect it in advance (a deployable diagnostic or proxy) is not specified.
If my understanding is correct, this indicates that there is still no way to predict this in advance.


* Generality, scale, and diversity of experiments  
The experiments are conducted only on LSA and GPT-2 (12 layers, 8 heads, ~9.5M parameters).
It is insightful, but generalization to current large-scale systems and broader task suites (code, long-form reading, knowledge-intensive tasks) remains limited.

### Questions
* It remains difficult to separate gains due to the proposed (pseudo-Newton) mechanism from those due to decoding strategies or exploration effects. Is there anything that the authors can discuss on this point?

* Prompt sensitivity / implementation sensitivity
If small implementation choices can flip conclusions, reproducibility is weakened.
Is there anything that the authors can add information on this point?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies test‑time scaling in a stylized in‑context learning setting. The core technical model is a one‑layer linear self‑attention transformer trained on an in‑context weight prediction task for linear regression with Gaussian features. They find that CoT induces an iterative update that they interpret as a multi‑step pseudo‑Newton method. They define a task hardness measure, derive error bounds that decay with the CoT depth, and argue that more test‑time compute can compensate for shorter training prompts, and that insufficient skill coverage in training can make longer CoTs harmful.

### Strengths
This paper addresses important questions such as whether more inference compute always helps, whether it can trade off against training context length, and what counts as difficult training data.

Proposition 3.2 derives the iterative update which makes the connection between CoT and a preconditioned iterative method precise and inspectable.

The proposed hardness measure is scale‑invariant, emphasizes tail eigen‑mass, and comes with a narrative mapping eigenvectors to skills and eigenvalues to skill strength, which leads to a clear understanding that harder tasks need longer CoT to reach the same error.

### Weaknesses
Experiments initialize with the closed‑form optimum using population statistics, so they basically hard‑code the solution instead of learning it, which invalidates empirical support for the learning‑dynamics claims.

All experiments are on synthetic linear tasks, there is no comparison to closed‑form ridge/OLS, no ablation on preconditioner estimation error, no evaluation on real reasoning benchmarks, and no demonstration that the proposed task‑selection improves anything beyond plotting selection weights.

### Questions
What is the test‑prompt length in Figures 2?

What is the sample complexity to estimate a usable covariance from a small validation set, and how robust is the solution to estimation noise?

### Soundness
3

### Presentation
3

### Contribution
3
