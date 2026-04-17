# Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
In this paper, we provide sufficient conditions of benign overfitting of fixed width leaky ReLU two-layer neural network classifiers trained on mixture data via gradient descent. Our results are derived by establishing directional convergence of the network parameters and classification error bound of the convergent direction. Our classification error bound also lead to the discovery of a newly identified phase transition. Previously, directional convergence in (leaky) ReLU neural networks was established only for gradient flow. Due to the lack of directional convergence, previous results on benign overfitting were limited to those trained on nearly orthogonal data. All of our results hold on mixture data, which is a broader data setting than the nearly orthogonal data setting in prior work. We demonstrate our findings by showing that benign overfitting occurs with high probability in a much wider range of scenarios than previously known. Our results also allow us to characterize cases when benign overfitting provably fails even if directional convergence occurs. Our work thus provides a more complete picture of benign overfitting in leaky ReLU two-layer neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proves directional convergence of two-layer neural networks with leaky ReLU activation trained on the linearly separable data without label flipping via gradient descent. The data distribution are either mixture sub-Gaussian or mixture of polynomially tailed distribution.

### Strengths
The derivation is very solid and the error bounds look correct. The writing is clear and easy to follow.

### Weaknesses
1. The studied data distribution is restricted to two linearly separable mixture models.

2. Previous work on benign overfitting need $d >> n$ or near-orthogonality because of the existence of the flipped labels. It's intuitive and very expected that near-orthogonality is no longer needed if there is no flipped labels.

3. Theorem 4.8 states the convergent direction is the solution of an optimization problem (5), but does not solve (5).

4. In your Assumption 6.1 (a), in the special case where $\Sigma = I_d$, it reduces to $d \geq C \max (n \|\mu\|^2, n^2)$, which is the same as Assumption (CL2) in [1], why you do not have near-orthogonality under such assumption?

5. typos: The latex does not compile correctly at Assumption 6.1 (a)

[1] Benign Overfitting in Linear Classifiers and Leaky ReLU Networks from KKT Conditions for Margin Maximization

### Questions
What's your definition of benign overfitting? Why there exists benign overfitting if all labels are correct/unflipped?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies two-layer leaky ReLU networks trained by gradient descent in the context of binary classification using an exponentially tailed loss. It establishes two main theoretical results. First, it proves that gradient descent, with a fixed step size, exhibits convergence in direction for the network weights, approaching the max-margin classifier. Second, it characterizes the generalization behavior of such networks under a mixtures data model with mean $\mu$ and covariance $\Sigma$. The analysis identifies a phase transition between benign and harmful overfitting,
$
n \|\mu\|^4 \gg \|\Sigma\| \,Tr(\Sigma).
$
The results apply to data distributions that may be correlated and heavy-tailed (need finite fourth moments), they don't require the common assumptions of isotropy or sub-Gaussian tails.

### Strengths
The two key technical contributions of this paper seem to me to be as follows.
- convergence in direction for gradient descent, not just gradient flow. Although this might feel intuitive this is not I would imagine easy to prove,  you need to show some uniform control on the angle between successive gradients which is complicated by the fact that the gradient angle can jump around discontinuously due to the non-smoothness of leaky-ReLU.
- identifies benign versus harmful overfitting regimes without isotropic / nearly orthogonal data: in many former works this was a requirement for proving benign overfitting as it allows one to treat network activations of different points almost independently, thereby simplifying the dynamics greatly. It is worth noting that another work tackles this issue in a slightly different setting (hinge loss rather than exponential tailed), see questions below.
- the paper extends benign-overfitting and directional-convergence results to heavy-tailed mixtures.

### Weaknesses
The main weaknesses of the paper I think are as follows
- Scope / relevance of the setup, namely shallow leaky-relu network, only training the inner layer weights, data which is linearly separable / drawn from a relatively simple distribution. Analysis even in this setting is pretty challenging though so I do not think this is a major issue.
- A flip side of one of the paper's strengths is that it does not provide finite-sample or high-probability generalization guarantees. The absence of sub-Gaussian or independence assumptions I think precludes standard concentration arguments, so the results provided are weaker in the sense that they hold asymptotically and in expectation. In short, compared to prior results, while the theory applies to broader data distributions, it does not yield as explicit rates or probability bounds. This really is more of a trade-off though than a weakness.
- I think some relevant work has been missed in the literature review, see questions below.

### Questions
- On convergence in direction: for what range of step sizes do we get convergence versus oscillation / divergence? At the moment Theorem 4.8 seems to say for sufficiently small step-size we get convergence, I am wondering if you have any insight into when this convergence breaks down?

- Can you sketch the convergence proof idea out perhaps? I am imagining that you try to fix the activation pattern eventually, allowing you to treat the network as a piecewise linear model? How do you control switching between activations regions and the contraction in angle?

- Have you thought about the non linearly separable setting? In this setting I think perhaps a finite solution would exist and as a result the analysis of convergence in direction would no longer be relevant but one could still think about a soft margin bias say.

- I think this work has some similarities (particularly in regard to relaxing the orthogonality conditions on the data) with "Benign overfitting in leaky relu networks with moderate input dimension", by Karhadkar et al, a link to which is given below,
https://proceedings.neurips.cc/paper_files/paper/2024/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html. Would you possibly comment? I think this and maybe some of the references therein might be worth considering adding to your prior work section.

### Soundness
3

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
This work studies 2 layer (1-hidden layer) leaky relu nets. The authors give result on the limiting behavior of gradient descent and generalization bound.

### Strengths
Main result on convergence of the parameters also gives the geometry of the decision boundary (which is linear).

Goes beyond the nearly orthogonal setting that many previous work assumes.

Is well-written.

### Weaknesses
The term "benign overfitting" is used throughout and in theorem 6.3, but I could not find the mathematical definition of the term. 

Is interpolation implied by result on Line 258? 

Also would be helpful to explicitly to write down the Bayes error rate (perhaps it's obvious and I just missed it, but I can't figure it out while reading).

### Questions
How is w_bar on Line 268 related to the mu from eqn (3)?

Line 257: does a neuron being "activated" means the same thing for leaky-relu as it does for the standard relu? the term is a bit confusing since leaky-relu doesn't have a "dead spot".

Line 259 says "the convergent direction [...] can also be given by ...". Is the convergent direction unique, or is the direction defined in eqn (5) just one possible convergent direction?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors consider a natural simplified model of a two layer network with leaky ReLU activations.  The first-layer weights are trainable, and the second layer weights are simply fixed and set to $\pm 1$ up to rescaling.  The authors use an exponential loss function. This is all a fairly standard model for this type of analysis. The goal is to solve a classification problem, where the two classes are centered at $\pm \mu$ and iid noise either with subgaussian or polynomial tails is added to each data point.

The first main contribution of the paper is to work out the convergent direction of the network weights.  Notably, this convergent direction is worked out explicitly under deterministic conditions, and is presented as the solution to a simple optimization problem, which has appeared in prior work under stronger conditions.  This allows the paper to separate the deterministic and probabilistic components of the analysis.  Then in Theorem 5.1 they turn to understanding the classification error under a random data-generation model.  Interestingly, this work gives both an upper and a lower bound on the probability of misclassification, so the authors are able to observe a phase transition between the weak versus strong signal regime, with the latter being associated with higher levels of misclassification.  The authors apply their result to random data models (with subgaussian and polynomial tails) in the overparameterized regime and thereby give new results on the benign overfitting phenomenon.  The main improvements are that the data need not be nearly orthogonal, and that they can handle leaky ReLU activations.

### Strengths
The paper seems technically sound and gives legitimate improvements over prior work.

The first improvement is that the analysis holds beyond the nearly orthogonal setting, which is necessary for understanding the types of data that occur in practice, particularly when the means of the two classes are far apart.  They also understand the convergent direction of the weights in a data-dependent way, which may make these results easier to apply as a black box in other settings.  In addition to obtaining an upper bound on the probability of misclassification, they also obtain a lower bound.  This allows for a precise characterization of when overfitting occurs, which appears to be an improvement over prior work.

The observation of a phase transition between the weak and strong signal regime is also interesting, and gives an explanation as to way previous work required an orthogonality assumption.  This seems to be a substantially novel result compared to prior work.

### Weaknesses
Some of the actual results are perhaps a bit incrememental.  For example the convergent direction is precisely that of Frei et al. as should be expected. On the other hand, relaxing the orthogonality assumption seems like a nice direction, as previous work in the area was unable to do this.

The actual model is fairly weak compared to modern DL models, and so may be limited in terms of understanding why benign overfitting occurs in practice.  Benign overfitting in two-layer networks is an established research direction though, and think this would be broadly interesting to that community.

### Questions
What is the difficulty in move from gradient flow results to gradient descent?  Can gradient flow results not be converted to gradient descent results just by taking the step size small enough? Or is the interest in getting specific bounds on the step size?

“We confirm in the next section that these assumptions are satisfied with high probability under model (sG) and (PM), and conjecture that they can also be verified in other settings.”
What settings are conjectured to work?

The two-sided bounds e.g. in Theorem 6.2 are given for Gaussian z.  Are these bounds expected to hold more generally?

— Minor —

Assumption 6.1 (a) – latex compilation issue
Beginning of section 3.  $|a_j| = \pm \frac{1}{\sqrt{m}}$ – shouldn’t have the absolute values

### Soundness
4

### Presentation
3

### Contribution
3
