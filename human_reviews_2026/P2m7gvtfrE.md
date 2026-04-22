# How Does Preconditioning Guide Feature Learning in Deep Neural Networks?

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Preconditioning is widely used in machine learning to accelerate convergence on the empirical risk, yet its role on the expected risk remains underexplored.
In this work, we investigate how preconditioning affects feature learning and generalization performance. We first show that the input information available to the model is conveyed solely through the Gram matrix defined by the preconditioner’s metric, thereby inducing a controllable spectral bias on feature learning. Concretely, instantiating the preconditioner as the $p$-th power of the input covariance matrix and within a single-index teacher model, we prove that in generalization, the exponent $p$ and the alignment between the teacher and the input spectrum are crucial factors. We further investigate how the interplay between these factors influences feature learning from three complementary perspectives: (i) Robustness to noise, (ii) Out-of-distribution generalization, and (iii) Forward knowledge transfer. Our results indicate that the learned feature representations closely mirror the spectral bias introduced by the preconditioner---favoring components that are emphasized and exhibiting reduced sensitivity to those that are suppressed. Crucially, we demonstrate that generalization is significantly enhanced when this spectral bias is aligned with that of the teacher.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors consider the effect of an unsupervised linear transformation, a so-called preconditioning matrix, applied to the data on feature learning in the single-index model case. In particular, the authors consider the specific preconditioning matrix is a spectral transformation of the input covariance. The authors show theoretically that the first layer weights, with Gaussian initialization, erase information about the original basis representation of the data. The authors then perform experiments showing that aligning input covariance directions with the task-relevant direction improves performance, and, similarly, unaligning the data with task-direction worsens performance. They show this effect in the settings of noise and transfer learning. They provide observations of the effect of various observations on two synthetic settings with their preconditioning approach.

### Strengths
-	The paper addresses fundamental questions of deep learning: (1) how do neural networks learn features from data, and (2) what is the impact of optimization method on feature learning?
-	The experiments are thorough and are presented cleanly.

### Weaknesses
The observations in the paper are frankly not surprising at all and are relatively shallow in nature. First, their form of preconditioning is equivalent to viewing the effect of the covariance of the data on generalization, as their preconditioning matrix is neither trained nor supervised. This setting would be much more interesting in those cases or if preconditioning were studied in parameter space. 
Then, it is not surprising that reducing the variance of the data in the directions that are relevant for prediction would worsen performance as you are emphasizing unimportant direction and placing the burden on the neural network to unlearn this preconditioning. This “unlearning” of unimportant directions has been studied much more precisely both theoretically and empirically in many of the prior works on feature learning (e.g. [1]).

[1] Damian, Lee, Soltanolkotabi. “Neural Networks can Learn Representations with Gradient Descent”, 2022. 

The most interesting part of the paper to me is the experiments observing the comparison of various optimizers on their single index task across exponents for their preconditioning. However, the results for this setting are presented pretty much without explanation and matter-of-factly. It is not clear what the reader should take away from these experiments.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper investigates how preconditioning affects generalization, arguing that its primary role is to control and bias the feature learning process.
The central argument is that the preconditioner (e.g., from Adam or K-FAC) acts as a filter, inducing a "spectral bias" by forcing the model to only see the data through a specific, preconditioned Gram matrix.
The authors model this by instantiating the preconditioner as the $p$-th power of the input covariance, $\Sigma_X^p$, where the exponent $p$ controls this bias: large $p$ emphasizes high-variance features, while small $p$ emphasizes low-variance features.
Using synthetic "teacher-student" experiments, the paper demonstrates that generalization is only achieved when the optimizer's spectral bias is aligned with the teacher's signal.

### Strengths
This paper investigates an interesting question with a reasonable approach. The results appear to me sound and convincing.

### Weaknesses
W1. The experiments are done with synthetic data. Is there any way to test this idea on more realistic setups?

Other than this, there are no obvious weaknesses to me at the moment, though I unfortunately cannot strongly endorse the paper as I am a non-expert reviewer (see confidence).

### Questions
Q1. What are the implications of your findings for how one might set the hyperparameters for off-the-shelf optimizers like Adam?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the role of preconditioning (specifically by powers of the input covariance matrix) on feature learning is studied. In particular, varying the power of the preconditioner affects the quantitative properties of the learned features via imposing a spectral bias of varying strength. The implications are demonstrated via noise robustness properties, generalization out-of-distribution, and transfer learning.

### Strengths
Overall, I think this is a solid paper that presents an interesting perspective on a scale of preconditioning that (as far as I know) has not been studied in prior work. The key theoretical result demonstrates the model evolution given a training batch can be fully captured by some salient statistics. These statistics clearly delineate how the "spectral bias" arises in feature learning. The experiments comparing multiple optimizers, notably SAM (corresponding approximately to the preconditioner power $p=1$), GD ($p=0$) and various second order optimizers ($p=-1$) serves as an interesting empirical reflection of the theoretical predictions.

### Weaknesses
In my opinion, the main weaknesses of the paper is the extrapolation from the provided theory and empirical claims. Notably, the theory in the paper focuses on one source of preconditioning via the input covariance, the experiments consider a wide range of preconditioned optimizers lumped under "second-order methods". Regardless of the empirical results, the qualitative differences between the curvature matrices approximated across the different methods is large enough that I'd be hesitant to attribute observed trends to the proposed spectral biases, even if for certain cases there is certainly alignment between different curvature matrices. Some more concrete derivations (even in simple one/two-layer linear networks) to show the relation between the curvature matrices between the different optimizers would help clarify this point.

It should also be noted that some optimizers in the "second-order" $p=-1$ category (e.g. AdaHessian and Adam) take a square-root power of the preconditioner, which suggests $p=-1/2$ would be more accurate there. Also, is it supported why SAM corresponds to $p=1$?

Lastly, though the takeaway in Section 4.3 that "second-order" optimization has a role in transfer learning, this observation is not necessarily novel, see e.g. [1], where the proposed intervention is indeed to pre-condition by the inverse input covariance matrix to enable feature/transfer learning. Furthermore, considering the specific discussion about KFAC is rather light, it might be good to note the corresponding "right-side" preconditioner KFAC induces on the input layer weight matrix is also the inverse input covariance.

[1] Zhang et al. "Sample-Efficient Linear Representation Learning from Non-IID Non-Isotropic Data"

### Questions
Please see Weaknesses above.

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
3

### Summary
This paper extends information-theoretic results on data whitening to a general preconditioning framework, investigating how preconditioned gradient descent affects feature learning and generalization in neural networks. The main theoretical contribution establishes that all input information available to the model flows exclusively through the preconditioned Gram matrix 
$ \(\mathbf{G}_P = \mathbf{X}^T \mathbf{P} \mathbf{X}\) $, where $\(\mathbf{P}\)$ is a positive semi-definite preconditioner. When instantiated as $\(\mathbf{P} = \boldsymbol{\Sigma}_X^p\)$ (the $\(p\)$-th power of input covariance), the Gram matrix's singular values scale as $ \(s_r^{2(p+1)}\) $, creating what the authors term $ \textit{spectral bias} $: higher exponents $\(p\)$ amplify high-variance input directions, while lower values (including negative) emphasize low-variance directions. It is formalization of quite well-known fact: the preconditioner induces a metric on input space; features aligned with that metric are preferentially learned.

The paper leaves the reader with clear (theoretical) message: for good generalization, one should match the preconditioner's spectral bias to the task structure. The paper provides somewhat concrete evidence through three experimental settings: robustness to label noise, out-of-distribution generalization on corrupted MNIST, and transfer learning across synthetic tasks.

Mathematically, Theorems 1 and 2 establish conditional independence relations showing that hidden layer activations and deep parameters remain conditionally independent of raw inputs given the Gram matrix history and labels. The paper naturally extends existing research work: Wadia et al.'s (2021) analysis of whitening to arbitrary preconditioners. 

While the paper is theoretically sound, I have some concerns regarding the clarity of presentation. The main issues is empirical work - which is more in-detailed described in weaknesses section.

### Strengths
S1: Clear Research Objective and Practical Relevance

The paper addresses a well-motivated question: how does preconditioning affect generalization, beyond its known role in accelerating convergence? The research objective is clear how to understand interplay between preconditioning and feature learning in deep learning. This is genuinely important because optimizer choice is typically made heuristically. By characterizing preconditioners through spectral bias, the paper provides a principled lens for understanding why different algorithms (SAM, K-FAC, L-BFGS, AdaHessian) exhibit different generalization properties on the same task. This contribution may have immediate relevance for practitioners.

S2: Reconciliation of Conflicting Prior Work

The paper elegantly resolves tension in the literature: Wadia et al. (2021) argued whitening is harmful to generalization by destroying information in the Gram matrix. Yet recent work shows benefits in specific contexts. This paper explains both via the alignment principle: generalization depends on whether the preconditioner's spectral bias matches where the true task-relevant signal lives spectrally. By extending beyond whitening to a continuous family parametrized by exponent $\(p\)$, the paper moves from binary "positive/negative'' judgment to task-dependent analysis.

S3 Elegant Mathematical Framework

Theorems 1 and 2 employing conditional independence via data-processing inequality are well-executed and provide a unifying perspective. The spectral characterization (Eq 7) elegantly shows how the exponent $\(p\)$ directly modulates singular value scaling: $\(s_r^{2(p+1)}\)$. This transparency is valuable for comprehensive understanding. 

S4: Thoughtfully Designed Experiments with Partially Compelling Evidence

The experimental design is illustrative and well-conceived:

a) Section 4.1 cleanly isolates spectral bias effects in synthetic settings with explicit signal placement (Cases High vs Low).
b) Section 4.2 provides the empirical result (IMO standing out): reversible optimizer rankings across two OOD scenarios (Figure 2). The ranking flip between invariant-digit and invariant-noise cases leads direct evidence that different optimizers have opposite spectral biases.
c) Section 4.3 offers a (limited, but still a bit novel taking into consideration the perspective) insight (negative $\(p\)$ optimal for transfer) with practical implications.

S5: Strong Presentation and Organization

The paper is well-structured with clear claims, logical flow, and effective visualizations.

### Weaknesses
[MAJOR]

W1: Severe Scalability Limitations and Dataset Concerns

The experimental validation is limited in scope:

i. Synthetic experiments (4.1): Use only 200 training samples with dimension 10, trained on a two-layer MLP with hidden dimension 256. While synthetic control is valuable for proof-of-concept, this extreme case relative to sample size is far removed from realistic settings. 

ii. Moreover, Figure 1 (top row) shows training curves that do not appear fully converged across most hyperparameter configurations. Since the paper compares generalization across conditions with different convergence levels, interpreting differences as evidence for the alignment principle becomes problematic. Fair comparison would require either training to convergence or comparing generalization at equal training loss/accuracy levels. The authors should clarify convergence criteria (definitely not constant optimization number of steps as this paper lies in optimization landscape) and provide terminal training&validation loss values given the convergence.

iii. MNIST experiments (4.2): Use approximately 1k training samples from MNIST (not stated clearly, I inferred it). While the controlled noise injection enables hypothesis testing, the result is barely above toy-problem scale. The split into spurious and invariant components is artificial and may not reflect realistic distribution shifts encountered in practice - there are other, more natural variants of noise injection in computer vision community.

iv. Transfer learning (4.3) Validates on a single symmetric synthetic task. For continual learning claims, two sequential tasks are definitely insufficient; typically we examine 5--20 tasks minimum. Real multi-task transfer learning (e.g., ImageNet pretraining to downstream tasks) would substantially strengthen claims.

W3: Substantial Theory-Practice Gap

Some assumptions required by the theory are violated in experiments:

i. $P(0)$-isotropic initialization: Theorem 1 requires weights to be initialized as$ \(\mathbf{w}_j^{(0)} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{P}^{(0)})\) $, meaning the initial covariance matrix must match the preconditioner structure. This requires:
- Pre-computing the full covariance matrix $ \(\mathbf{P}^{(0)} = \boldsymbol{\Sigma}_X^p\) $ before training (at least SVD is needed, right?)
- Custom initialization departing from standard schemes (for instance Xavier).

According to my best knowledge, these experiments do not appear to use this initialization; they likely employ standard initialization. Does it really matter in the end? If so, please empirically validate.

ii. Stochasticity: I am uncertain, but I guess that the theory assumes deterministic training. Experiments use mini-batch SGD, which may disrupt the deterministic Gram matrix bottleneck through finite-sample fluctuations. Same question as above, does it matter?

W4: Limited/Unrealistic Scope of The Proposed Setting

i. Single-index teacher limitation: The quantitative claims (Sec 3.3) assume labels follow $ \(y = h^*(\sum_r \alpha_r \beta_r) + \epsilon\) $, a one-dimensional structure in label space. Real tasks exhibit multi-dimensional structure. The paper provides no analysis of whether insights extend to multi-index teachers, severely limiting generalizability of Theorems 1 and 2 beyond this restrictive setting.

ii. [MINOR] First-layer-only preconditioning: Analysis applies only to preconditioned first-layer updates. Modern networks use layer-wise preconditioning (K-FAC, Sophia) as in one of the cited work Zhang et al. 2025. Probably this spectral bias does not propagate through depth.

[Minor Weaknesses]

W5: Experimental Design Ambiguities

i. (Sec 4.2.1) The exact experimental procedure for OOD evaluation is not entirely clear from the main text. It appears that: (a) the model is trained on data with digit+class-specific noise (one of them, right?), (b) at test time, either the noise or digit is changed to reverse which component is "spurious". Additionally, Figure 2 error bars show standard deviation only for in-distribution evaluation; reporting error bars for OOD accuracy would be more informative.

ii. Incomplete Analysis of Related Work

The paper cites but does not deeply discus Zhang et al.'s findings on layer-wise preconditioning and feature learning (I guess here more work should be pointed out), which partially overlap with the insights in Sec 4.3.

iii. Code Availability Issues

While the authors provide code (I appreciate it however it is not required), significant usability issues limit reproducibility:
- 2 of 5 scripts run but produce only single train/test loss values (insufficient information/analysis without any context as it is on particular engineered data);
- 2 scripts yield runtime errors (TypeError in AdaHessian.step() );
- 1 script has data iterator issues even after straightforward fixes applied.

Hence I do not see the purpose of it. If code is provided, it should be functional. Otherwise, omitting the repository might be preferable.

W6: Although I find the structure, research questions/hypotheses outlined useful and clear, the mathematical part is quite convoluted and sometimes hard to follow.

### Questions
Q1: Can you provide a rigorous derivation connecting SAM's preconditioner to \(p = 1\), or should this be framed as an empirical observation validated by Figure 2b?

I am not convinced that SAM can be viewed as an approach with conditional power being 1 and here is why:
SAM: $ \mathbf{g}_{\text{SAM}} = \nabla L(\mathbf{\theta} + \boldsymbol{\delta}) $ 

where perturbation is given $ \boldsymbol{\delta} = \frac{\rho}{\|\nabla L(\mathbf{\theta})\|} \nabla L(\mathbf{\theta})\ $.
We may apply Taylor's approx.: $ \mathbf{g}_{\text{SAM}} \approx \nabla L(\mathbf{\theta}) + \mathbf{H} \boldsymbol{\delta} = \nabla L(\mathbf{\theta}) + \frac{\rho}{\|\nabla L\|} \mathbf{H} \nabla L(\mathbf{\theta}) $. 
Hence for sufficiently small $\rho$ we obtain:
$ g \approx (\mathbf{I} - \rho \mathbf{H}) \nabla L(\mathbf{\theta}) $
It feels that: $ (\mathbf{I} - \rho \boldsymbol{\Sigma}_X)^{-1} \approx \boldsymbol{\Sigma}_X^{-1} $.
However, intuitively, I understand your claim that SAM emphasizes high-variance directions in the input space.

Q2: I do have question about averaging results in context of aggregating across initializations / random seeds. (related to W5, but not limitted)
    
Q3: For transfer learning: can you validate the $\(p = -1\)$ optimality on realistic multi-task settings (e.g., ImageNet pretraining to downstream tasks) rather than only synthetic scenarios?

Q4: How in practice, one can try to align spectral bias without requirement to having access to the teacher's signal.

### Soundness
3

### Presentation
3

### Contribution
2
