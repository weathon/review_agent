# Model Merging is Secretly Certifiable: Non-Vacuous Generalisation Bounds for Low-Shot Learning

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Certifying the IID generalisation ability of deep networks is the first of many requirements for trusting AI in high-stakes applications from medicine to security. However, when instantiating generalisation bounds for deep networks it remains challenging to obtain non-vacuous guarantees, especially when applying contemporary large models on the small scale data prevalent in such high-stakes fields. In this paper, we draw a novel connection between a family of learning methods based on model fusion and generalisation certificates, and surprisingly show that with minor adjustment several existing learning strategies already provide non-trivial generalisation guarantees. Essentially, by focusing on data-driven learning of downstream tasks by fusion rather than fine-tuning, the certified generalisation gap becomes tiny and independent of the base network size, facilitating its certification. Our results show for the first time non-trivial generalisation guarantees for learning with as low as 100 examples, while using vision models such as VIT-B and language models such as mistral-7B. This observation is significant as it has immediate implications for facilitating the certification of existing systems as trustworthy, and opens up new directions for research at the intersection of practice and theory.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a novel approach to provide non-vacuous generalization guarantees for large neural networks trained on low-shot data. The authors make an insightful observation: by reformulating the learning problem from "fine-tuning the entire model" to "learning how to merge the weights of multiple pre-trained models", they can leverage the PAC-Bayes framework so that the generalization bound depends only on the number of merging parameters, rather than the scale of the underlying models. This is an interesting and meaningful work, as it provides a new perspective on how to theoretically analyze and improve the generalization of large models in low-shot regimes.

### Strengths
1. The main strength of this paper lies in its being the first to provide non-vacuous generalization guarantees for modern large-scale models under extremely low-shot settings.
2. By shifting the focus from fine-tuning to model merging, the authors successfully decouple the difficulty of providing generalization guarantees. The generalization gap no longer depends on the enormous scale of the underlying models, but only on the relatively small number of parameters involved in the merging process. 
3. The authors further demonstrate that many existing off-the-shelf model merging algorithms can be certified to provide non-vacuous generalization guarantees with only minor modifications.

### Weaknesses
1. The paper does not report the runtime or computational overhead of this "certified optimization", which raises concerns about its practical feasibility.
2. If a suitable pool of source models cannot be found, or if the source models themselves are of low quality, the merging algorithm is likely to fail—rendering the proposed "certification" meaningless in practice.
3. In truly extreme low-shot scenarios (e.g., 10-shot), further splitting the data could severely compromise training stability and the model’s final performance.

### Questions
1. If the quality of the source model pool deteriorates—for example, if the models are unrelated to the downstream task, or if the models are too similar or too diverse—how would the PAC-Bayes guarantees in this paper be affected?
2. Compared to the original objective, how much additional training time or computational resources (e.g., GPU hours) does Bound Optimization require?
3. Is the data-splitting strategy still feasible in extremely low-data scenarios (e.g., (n=20) or (n=10))?

### Soundness
3

### Presentation
2

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
This paper aims to make the certification of IID generalization feasible for large neural networks trained on small datasets. It establishes a principled link between model fusion methods and PAC-Bayes theory, showing that learning only a small set of fusion weights yields a low-dimensional stochastic predictor with a generalization gap independent of model size. Building on this insight, the paper presents several new findings. (i) Simple, off-the-shelf model-merging learners can already achieve non-trivial guarantees. (ii) These guarantees hold under multiple standard formulations, including Gaussian PAC-Bayes and discretization bounds. (iii) Modifying the merge objective to directly optimize the PAC-Bayes bound, instead of the training likelihood alone, enables more advanced merging methods to attain both strong test performance and tight certificates. (iv) The study provides the first demonstration of non-vacuous generalization guarantees for large models such as ViT-B and Mistral-7B in the low-shot regime with only 100 labeled examples. Together, these contributions reveal a practical path toward certifying large pre-trained systems as trustworthy without full fine-tuning, offering a promising synthesis of theory and practice.

### Strengths
1. The paper introduces an intellectually appealing idea by identifying a novel connection between model fusion methods and generalization certification. This perspective leads to a new form of generalization bound that unifies practical model-merging strategies with formal theoretical guarantees, offering a concise and elegant framework bridging empirical practice and generalization theory.
2. The paper presents comprehensive empirical validation. It evaluates the proposed generalization bounds across diverse datasets and large-scale vision and language models, providing strong and credible experimental support for the claimed theoretical contributions.

### Weaknesses
1. The exposition of the PAC-learning framework in Section 3 lacks conceptual precision. The PAC framework does not seek to bound the discrepancy between the empirical risk and the population risk for an arbitrary, fixed parameter $\theta$. For any $\theta$, that is independent of the data, the difference between $L(\theta)$ and $\hat{L}(\theta)$ converges at the standard Monte Carlo rate $O(1/n)$,taking square error loss function as an example. This is a purely sampling effect rather than a learning-theoretic phenomenon. The proper PAC formulation considers the data-dependent estimator $\theta* = \operatorname{argmin}_{\theta} \hat{L}(\theta)$ and establishes bounds on the generalization gap $L(\theta*) - \hat{L}(\theta*)$. Since $\theta^*$ depends on the training sample, this gap reflects the algorithmic dependence on data and typically converges at a slower rate than $1/n$.

2. The paper’s motivation is insufficiently articulated, leaving the relevance of its setting unclear. (i) The focus on small datasets with highly overparameterized models is theoretically questionable. In this regime, approximation error is minimal, but model complexity is excessive, necessitating strong regularization and typically leading to poor generalization. A more principled analysis would consider the classical regime where the sample size $n$ grows and model complexity is scaled accordingly, allowing examination of the optimal bias–variance trade-off and the convergence rate of the generalization error as a function of  $n$  (ii) The analysis implicitly assumes near-optimal optimization, i.e., that the obtained parameter $\theta^*$, closely matches the empirical risk minimizer. In large-scale models, however, computational constraints prevent full optimization, making such assumptions unrealistic. Under these conditions, analyzing generalization becomes less informative than studying scaling laws, which capture how performance evolves with model and data size under practical optimization dynamics.

3. The paper provides limited theoretical justification or intuitive insight into why the proposed generalization bound improves upon conventional ones. While the empirical evidence suggests tighter or non-vacuous guarantees, the manuscript does not clearly articulate the underlying mechanism or theoretical principle responsible for this improvement.

### Questions
See weaknesses.

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
The paper presents an interesting theoretical and empirical finding: model merging, e.g. combining pretrained models via learned linear interpolation, can yield non-vacuous generalization bounds under the PAC-Bayes framework, even in few-shot settings and for large models like ViT-B and Mistral-7B.
The key insight is that the learnable parameter space in model merging depends on the number of source models, not their size, making certification feasible. The authors (i) reinterpret existing merging algorithms within PAC-Bayesian theory, (ii) propose minor modifications such as bound-aware objectives and data-dependent priors, and (iii) empirically demonstrate certifiable low-shot learning on image and text tasks.

### Strengths
- The paper establishes an elegant bridge between model merging and PAC-Bayesian certification, a previously unexplored direction.
- It achieves non-vacuous generalization guarantees with large pretrained networks (ViT-B, Mistral-7B) using as few as 100 examples, which is unprecedented in the certification literature.
- Requires only small modifications to existing merging pipelines; bound-aware optimization and Gaussian priors are straightforward.
- Experiments cover both vision and language domains, multiple merging algorithms, and detailed ablations (bound optimization, data-dependent priors, scaling with data).
- The work suggests a feasible path for certifying modern AI systems in low-data, high-stakes contexts.

### Weaknesses
- While well-motivated, the paper does not provide new generalization bounds - most results hinge on reinterpreting existing theory.
- Certification results are mostly numerical bounds rather than guarantees under formal verification (e.g., robustness or adversarial guarantees).
- Some methodological explanations (e.g., computation of the PAC-Bayes term, choice of priors, gradient-free optimization) could be made more rigorous and reproducible.
- Results exclude tasks where merging fails; it would be useful to quantify how often this happens.
- Bound computations may become costly for real-world high-dimensional merges; the paper could better discuss computational complexity.

### Questions
- How sensitive are the certification results to the choice of variance in the Gaussian prior/posterior (e.g., $\lambda_1, \lambda_2$)?
- Does the bound remain non-vacuous if the number of source models (hence $\alpha$-parameters) grows beyond a few dozen?
- In what ways does data-dependent prior construction risk information leakage from the support set?
- Could this framework be adapted to nonlinear merging or adapter-based composition (e.g., LoRA fusion)?
- How would the results compare if the merging coefficients were learned jointly with fine-tuning of small subsets of parameters?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper reveals that model merging methods such as Task Arithmetic, Ties-Merging, and AdaMerging can naturally yield non-vacuous PAC-Bayesian generalization guarantees, even for large models like ViT-B and Mistral-7B trained on as few as 100 examples. By viewing the small set of fusion weights as the learnable parameters, the approach sharply reduces effective model capacity, tightening PAC-Bayes bounds. The authors further enhance this link through bound-optimized objectives and data-dependent priors, demonstrating empirically that merging-based few-shot adaptation can be both practical and theoretically certifiable-a first for large-scale networks in low-data regimes.

### Strengths
1. The author demonstrated that certifiable generalization on ViT-B and Mistral-7B under 100-shot learning is unprecedented, compared with previous attempts (e.g., Lotfi et al. 2024) which require thousands of samples or smaller models.
2. Comprehensive experiments span both vision and language tasks (EuroSAT, GTSRB, MNIST, DTD, BBH, TweetEval), showing consistent trends across settings.

### Weaknesses
1. The introduction jumps quickly into technical framing (generalisation bounds, PAC-Bayes, model fusion) without clearly articulating the conceptual motivation: Why should we care about certifying large models in the first place?
2. The work did not conduct sensitivity experiment on prior variance $\lambda$ or the confidence parameter $\delta$.

### Questions
1. The guarantees apply only to IID settings. I am curious how the model behaves under a realistic adversarial attacks like PGD attack[1].
2. Could PAC-Bayesian interpretation also apply to other parameter-efficient tuning methods like LoRA? It would be helpful to know if model merging is uniquely suitable for certification, or if low-dimensionality alone explains the effect.
3. The experiments focus on 100-shot learning. would the certification remain valid (or meaningful) under more extreme few-shot scenarios, such as 10-shot or 1-shot?

[1] Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).

### Soundness
3

### Presentation
2

### Contribution
2
