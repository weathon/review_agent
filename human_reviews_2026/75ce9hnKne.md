# When Bias Meets Trainability: Connecting Theories of Initialization

- Decision: Accept (Poster)
- Scores: 2, 4, 8, 4

## Abstract
The statistical properties of deep neural networks (DNNs) at initialization play an important role to comprehend their trainability and the intrinsic architectural biases they possess before data exposure. Well established mean-field (MF) theories have uncovered that the distribution of parameters of randomly initialized networks strongly influences the behavior of the gradients, dictating whether they explode or vanish. Recent work has showed that untrained DNNs also manifest an initial-guessing bias (IGB), in which large regions of the input space are assigned to a single class. In this work, we provide a theoretical proof that links IGB to previous MF theories for a vast class of DNNs, showing that efficient learning is tightly connected to a network’s prejudice towards a specific class. This connection leads to a counterintuitive conclusion: the initialization that optimizes trainability is systematically biased rather than neutral.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Develops a theoretical link between mean-field (MF) trainability and initial guessing bias (IGB) at random initialization. Shows an equivalence between MF quantities and IGB statistics, argues that the edge of chaos corresponds to a state of transient deep prejudice that enables fast learning, and provides supporting experiments on MLPs, a simplified ViT, and a large ImageNet-pretrained ViT fine-tuned on CIFAR-100.

### Strengths
- Clear statement of the MF-IGB connection, including a formal mapping between MF covariances and IGB drift ratio, and phase-diagram reinterpretations for bounded and unbounded activations.
- Reproducibility note and code availability claim.
- Empirical plus theoretical curves illustrating $c^{(l)}\rightarrow$1 for ReLU, with different rates in ordered vs. chaotic regimes.

### Weaknesses
- Empirical scope is limited relative to broad claims. Experiments focus on binarized Fashion-MNIST, binarized CIFAR-10, CIFAR-10 with simplified MLP/ViT, and a single large ImageNet-pretrained ViT on CIFAR-100.

- Assumptions narrow external validity. Core analysis relies on the infinite-width mean-field regime and an i.i.d. Gaussian input model in the IGB framework.

- Fairness framing without fairness evaluation. The introduction and contributions connect learnability and fairness from initialization onward, but the experiments report only accuracy and maximum class fractions, with no group-attribute metrics.

- Pretraining-bias angle is underdeveloped. For the large ViT, weights are uniformly rescaled to traverse phases, but there is no analysis disentangling ImageNet pretraining priors from weight-scaling dynamics.

- Clarity/calibration: extrapolating MF to practice. Theoretical mapping claims that “best trainability” corresponds to a state of transient deep prejudice (Proposition 4.1), yet validation remains on toy or simplified settings.

### Questions
- Representativeness of experiments. Can the empirical section include standard ImageNet-scale or at least non-binarized, multi-class tasks with production architectures (with normalization and residuals) to assess whether the MF–IGB link and the “transient deep prejudice” claim persist beyond toy settings?

- Input-distribution assumptions. How sensitive are the theoretical conclusions to deviating from i.i.d. Gaussian inputs, for example, to real image statistics or correlated features? Any finite-width theory or experiments targeted at non-Gaussian inputs?

- Fairness measurements. Since the paper motivates fairness, could group metrics on a dataset with sensitive attributes be provided to test whether initial “transient deep prejudice” impacts disparities during early training and at convergence?

- Pretraining biases. For the large ViT, is it possible to disentangle the role of ImageNet pretraining from the weight-scaling operation, and measure whether class-prior skew or spurious correlations from pretraining amplify or dampen IGB on the target data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates the relationship between initialization bias and trainability of DNNs, reconciling two theoretical perspectives:
- Mean-Field (MF) theory — which studies gradient propagation and identifies the “edge of chaos” (EOC) as the optimal initialization boundary for stable training;
- Initial Guessing Bias (IGB) framework — which describes how untrained networks can exhibit predictive bias (favoring one class) even before training.

The authors establish a formal equivalence between the MF and IGB frameworks, showing that the trainability boundary in MF corresponds to a biased initialization state in IGB. Contrary to the intuitive belief that the most trainable initializations should be neutral, they demonstrate theoretically and empirically that the optimal initialization is systematically biased rather than unbiased — a state they call transient deep prejudice.

### Strengths
1. Theoretical contribution: The paper establishes a clean mathematical connection between two previously distinct frameworks (MF and IGB), enriching both perspectives.
2. Insight: It introduces a novel idea: bias at initialization can improve trainability; The new “prejudice-neutrality” phase view offers an intuitive explanation for initialization effects, linking bias to the dynamical stability of gradient flow.

### Weaknesses
1. The theory is derived in the infinite-width limit and validated on small- to mid-scale settings. Its applicability to practical, large-scale deep networks (e.g., transformers with normalization and attention) is not demonstrated. Additionally, empirical evaluation focuses on synthetic and small vision datasets. It is of readers' interest to learn results on simple language tasks. 
2. Ambiguous practical relevance: While the “transient bias” insight is conceptually interesting, there is no clear recipe for practitioners (e.g., how to initialize weights to achieve the right level of prejudice). It would be helpful to add some executable takeaways.
3. It would be helpful to introduce and compare with alternative trainability-enhancing initializations (e.g., orthogonal, LSUV, scaled ReLU, or NTK-based initializations).

### Questions
There can be some terminological confusion: It would be helpful to clarify that “bias,” “prejudice,” and “neutrality” refer to statistical asymmetry / symmetry in prediction space. Otherwise, readers may think about fairness or ethical bias.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper links two views of randomly initialized networks: initial guessing bias, which is when untrained models over-predict one class, with the mean field theory describing trainability at initialisation. The core result is that there is a mapping between both views, such that the same initialisation that can yield a good gradient flow also produces the most biases starting predictions. The paper extends this across architectural choices, including an experiment to perturb a pre-trained ViT to demonstrate these effects.

### Strengths
1. Clean conceptual claim that ties IGB statistics with network trainability
2. Clear testable claim forhow models which start with maximal bias learn fastest
3. Robust checks beyond toy MLPs to ViTs, both training and perturbing them

### Weaknesses
1. The infinite width setting is somewhat limiting on the theory side
2. It's unclear how much the choice of norms matters or whether this result is idiosyncratic
3. The pooling for the 2-dimensional case is similarly limited

### Questions
1. Is it possible to link the theory to regimes beyond the infinite width setup? 
2. How does this theory work under harder settings e.g. the vanilla MLPs are not the strongest baselines for performance?
3. Is it possible to measure bias level in practice and across training?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies initial guessing bias (IGB)—when a network exhibits a bias towards certain classes. It mathematically links IGB to mean-field theory of wide networks, thus connecting learnability properties to IGB. Using this, the authors show that initializing networks to be optimized for trainability and have stable gradients also yields class bias at initialization.

### Strengths
Understanding how and why neural networks are biased is an important topic for fairness in AI systems. This paper considers the impact of weight initialization, independent of data, which could help us understand how learning dynamics affect bias. By connecting IGB to established mean-field theory of wide networks, it helps build a framework for this. The authors support their theoretical claims with experiments in several architectures and datasets.

### Weaknesses
I found the paper difficult to follow and understand. I think it could have more strongly motivated why the link to MF theory is an important one to make and why the contribution made is important. It was also unclear to me at times what was background on previous work versus a novel contribution of the paper.

“Prejudice” is a more loaded term than “bias” and I find the use of it here somewhat inappropriate for what is being referred to. Bias is used to refer to behavior a network might be more prone to do (e.g., inductive bias) and I think would be more apt.

The paper briefly mentions it, but by assuming the infinite-width limit, the paper focuses on the so-called lazy learning regime. While the lazy learning regime may be valuable to study, it seems like a limitation for a study motivated by fairness to not focus on the rich feature learning regime optimized for in practice. Furthermore, there is substantial literature on feature learning that discusses how different initializations impact learning and representations, which should probably be reviewed and discussed more in this paper.

I think the contribution of the paper is limited. In particular, the main finding concerns how different initializations associated with trainability also reflect different bias. However, although they show that stable initializations are initially biased, this effect goes away with training. It’s unclear whether this specific initialization effect would persist in other settings and actually affect fairness. For example, this result would be strengthened by showing that it persists or is amplified by certain properties of the data.

### Questions
What is the difference between “chaotic-deep prejudice” and “(chaotic) prejudice”? Why is it called “deep”?

### Soundness
3

### Presentation
2

### Contribution
2
