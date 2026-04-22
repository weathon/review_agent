# Generating Samples to Probe Trained Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
There is a growing need for investigating how machine learning models operate. With this work, we aim to understand trained machine learning models by questioning their data preferences. We propose a mathematical framework that allows us to probe trained models and identify their preferred samples in various scenarios including prediction-risky, parameter-sensitive, or model-contrastive samples. To showcase our framework, we pose these queries to a range of models trained on a range of classification and regression tasks, and receive answers in the form of generated data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework for generating synthetic data that probes already trained models under various objectives.
It examines four key objectives: uncertainty — data points near the decision boundary; disagreement — where two models make opposite predictions; sensitivity — where small parameter changes strongly affect the output, and counterfactuals —inputs that yield a fixed target label. For each objective they define a probe energy over inputs that induces a Gibbs distribution; sampling from it yields examples for that objective.
The framework is evaluated across toy tasks, real-world tabular datasets, MNIST, and latent probes of pretrained ImageNet classifiers, showing that the generated samples accurately capture each probing objective.

### Strengths
The paper is original in framing model probing as a data generation problem using a Bayesian Gibbs formulation. It is clearly written, with solid theoretical grounding and well-chosen illustrative experiments. The framework is significant in that it offers a unified and general-purpose approach for probing trained models —both differentiable and non-differentiable— across multiple axes of behavior, including uncertainty, disagreement, sensitivity, and counterfactual exploration. 
It also has potential to extend beyond probing — for example, to adversarial testing, fairness auditing and enforcement, or post-hoc robustness evaluation.

### Weaknesses
- Evaluation is mostly qualitative relying primarily on visual examples and descriptive comparisons; adding quantitative metrics for uncertainty, sensitivity,  disagreement and counterfactual would strengthen the evidence.
- The latent-space sampling relies on a pretrained VAE but lacks explicit regularization to prevent drift off the data manifold, which may affect sample fidelity in practice.

- Hyperparameter sensitivity is unexplored; the temperature $\tau$ and sampling parameters likely affect the generated samples. An ablation could clarify the framework’s robustness to these choices.

- No information is provided on runtime or computational cost, which would help assess the practical feasibility of the approach.

- Despite claiming applicability to both classification and regression, all empirical results focus on classification tasks. A regression case study would help substantiate the framework’s generality.


Formulation inconsistencies: 
- Although the framework is presented as general over $\mathcal{Y}$, Eq.~(6) for parameter-sensitive samples assumes a binary output through the term $1 - y_{\theta^*}(x)$. The paper should either restrict this objective to binary tasks or provide a general formulation.
- The formulation R_G(x) = $|x - x_a|_r^r$​ in Equations (4)–(7) lacks an explicit regularization weight $\lambda$, which is essential for controlling the trade-off between locality and the probing objective. Including $\lambda$ would improve the completeness and clarity of the formulation.

### Questions
1) Why are quantitative metrics for uncertainty, disagreement, sensitivity or counterfactuals not used more broadly to support the probing scenarios?

2) How do you ensure that samples generated in latent space remain on the data manifold, especially without explicit regularization or constraints on the latent variables?

3) How sensitive are the generated samples to the specific design of the probing function G(x) (e.g., choice of loss term, temperature, or regularizer)? Some analysis or ablation could clarify how robust the framework is to these design choices.
Do the experiments include a tunable regularization weight \lambda for R_G(x)?

4) In Figure~6, the gender distribution of the generated counterfactual samples appears roughly balanced, yet the text claims that most samples shift from female to male. Could the authors clarify how this distribution was computed and whether the claimed gender shift is quantitatively supported? Could this observation simply reflect that gender is not a significant factor in the model’s prediction?

5) Eq.~(6) for parameter-sensitive samples appears to assume binary outputs through the term $1 - y_{\theta^*}(x)$. How would this objective be defined for multi-class or regression tasks where such a complement is not well-defined?

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
1

### Summary
The paper proposes a general mathematical framework for probing trained models by generating synthetic samples that satisfy custom-defined probing objectives. The approach draws a symmetry between model training (parameter optimization) and model probing (data optimization), and includes analytical and empirical demonstrations across regression and classification tasks.

### Strengths
The idea of formulating probing as a generative process is interesting and connects interpretability with probabilistic modeling. Overall, the proposed framework is flexible, allowing different “questions” to be posed to a trained model.

### Weaknesses
Overall, I find the paper’s organization difficult to follow, particularly in Section 2. The core mathematical framework is presented in a dense and abstract way, which obscures how the proposed method is actually implemented. Several key equations (e.g., Eq. 2–4) are introduced without sufficient intuition or explanation. Moreover, the design rationale behind the probing function G across different use cases remains unclear—for instance, it is not evident why Eq. (5) appropriately captures model-contrasting samples. These gaps in exposition make it challenging to fully assess the technical quality and contribution of the work. 


Minor: 
1. Line 84: "First, our notation:" sounds like an unfinished sentence
2. Line 98: "The loss function F in 1" should be "... Equation (1)"

### Questions
Can you clarify the computational steps for constructing $G(x)$ and sampling  $p(x)$ in non-trivial models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a general mathematical framework for generating synthetic samples that reflect specific model behaviors. The idea is to take a "dual" view of model training: given a fixed model (or distribution of model parameters), sample data according to a specific user-defined probing function. Samples can be drawn using Langevin dynamics. By choosing different forms of G, the approach can produce various kinds of synthetic samples, such as model-contrastive, prediction-risky, or parameter-sensitive. The method is demonstrated on linear regression, tabular data, and vision models.

### Strengths
The paper introduces a clear variational formulation that symmetrically parallels model training and data synthesis. The proposed framework is very general, can handle various kinds of objective functions, and is model-agnostic.

### Weaknesses
While the conceptual insight is interesting, at least based on my knowledge in this domain, the paper has relatively marginal novelty in its specific algorithm (which is not necessarily a weakness, though). 

The paper currently lacks a discussion of how the proposed framework can be extended to the discrete input space. While a VAE decoder can enforce data-manifold constraints, this is still difficult for generating language data. 

The experiments are mostly qualitative and small-scale. 

No baselines are being compared in the experiments. However, I am not sure whether there are no prior works on this problem. For example, adversarial examples are being generated using similar gradient-based algorithms.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2
