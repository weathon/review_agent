# Inducing Uncertainty on Open-Weight Models for Test-Time Privacy in Image Recognition

- Decision: Reject
- Scores: 4, 8, 4

## Abstract
A key concern for AI safety remains understudied in the machine learning (ML) literature: how can we ensure users of ML models do not leverage predictions on incorrect personal data to harm others? This is particularly pertinent given the rise of open-weight models, where simply masking model outputs does not suffice to prevent adversaries from recovering harmful predictions. To address this threat, which we call *test-time privacy*, we induce maximal uncertainty on protected instances while preserving accuracy on all other instances. Our proposed algorithm uses a Pareto optimal objective that explicitly balances test-time privacy against utility. We also provide a certifiable approximation algorithm which achieves $(\varepsilon, \delta)$ guarantees without convexity assumptions. We then prove a tight bound that characterizes the privacy-utility tradeoff that our algorithms incur. Empirically, our method obtains at least $>3\times$ stronger uncertainty than pretraining with marginal drops in accuracy on various image recognition benchmarks. Altogether, this framework provides a tool to guarantee additional protection to end users.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a novel test-time privacy threat model in which correct predictions on certain sensitive queries may leak private information. To provide “test-time privacy,” the goal is to output a model that predicts uniformly on the sensitive queries while remaining accurate on the rest. The authors define a target reference model $\mathcal{M}_{\theta}$, trained with an objective that is a regularized, weighted sum of (i) minimizing loss on the retain set and (ii) encouraging predictions close to uniform on the forget set, with $\theta$ controlling the tradeoff.

They then define an $(\epsilon, \delta, \theta)$-certified Pareto learner as an algorithm that modifies the ERM solution on the entire query set so that its output is indistinguishable from the reference model $\mathcal{M}_{\theta}$, where $\epsilon, \delta$ measure indistinguishability similar to certified unlearning or differential privacy. The proposed certified Pareto learner takes one Newton step toward the reference model and then adds calibrated Gaussian noise.

Theoretical analysis characterizes the privacy–utility tradeoff by upper-bounding the accuracy gap between the “unlearned” model and $\mathcal{M}_{\theta}$. Experiments further show that retain-set utility remains high, even for relatively large $\theta$.

### Strengths
The paper is clearly written and introduces a novel, relevant threat model that is both timely and worth investigating.

### Weaknesses
- The definition of a certified-Pareto learner requires indistinguishability between the “unlearned’’ model and the reference model $\mathcal{M}_{\theta}$. This introduces an extra parameter $\theta$, which makes the guarantees harder to interpret: both the privacy and utility statements are only relative to the chosen reference model.

- Theorem 4.5 bounds the accuracy drop only relative to $M_\theta$. However, a large $\theta$ can already degrade $M_\theta$’s accuracy and still may not ensure strong test-time privacy. In particular, selecting $\theta$ via Equation (7) and Corollary 4.2 implies that achieving $\theta > 0.75 $ requires $\epsilon = \sqrt{|D_r| \ln |Y|} $, which exceeds 1 even for moderate $|D_r|$. At the same time, $\theta = 0.75$ appears to induce substantial error for $\mathcal{M}_{\theta}$ in the experiments.

Minor notation suggestion: consider renaming the $\epsilon$ in Equation (7), since it differs from the $\epsilon$ used in the certified-Pareto learner’s indistinguishability guarantee.

### Questions
- As I understand it, the goal is to produce a model that outputs (approximately) uniform predictions on a protected set of sensitive queries while remaining accurate on the rest. Why is uniformity the right target for privacy—rather than, for example, defining test-time privacy as an upper bound on accuracy over the sensitive set (e.g., “no better than chance” or some specified cap)?

- Pushing predictions toward uniformity on the sensitive set appears to necessarily reduce overall performance, with the magnitude depending on both the sensitive set and the retained set. Can you provide lower bounds on the accuracy drop of the reference model?

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
2

### Summary
The work defines a new threat model called test-time-privacy, where open weight model is used by the adversary to inflict harm on person x based on the prediction about x from corrupted public data. This paper addresses the following RQ: Can we ensure test-time privacy against adversaries with access to an open-weight model? The work proposes algorithm to induce maximal uncertainty on the protected set while preserving utility on the rest.

### Strengths
- formulation of a new threat model: test-time-privacy (TTP) which addresses a key safety concern often understudied in machine learning literature,
- proposes a method to address the TTP threat,
- method balances uncertainty and utility,

### Weaknesses
- the method only applies to classification problems, what about generative models and others?

### Questions
How challenging is applying this method to generative models as current settings apply only to classification models?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper considers a privacy game where a set of examples D_f are considered "incorrect." That is, D_f is assumed to contain false, or corrupted data individuals. An attacker wants to claim something nefarious about the individual associated with x in D_f by demonstrating that a model f labels x in some way. In other words, by demonstrating f(x) = y_bad, the attacker can achieve some benefit. For example, the attacker might be an insurer looking to (falsely) deny coverage to the user associated with x. 

A model curator, who also knows the set D_f, wants to defend against any such attacker. To do so, the curator wishes to output a model that labels any x in D_f randomly. The setting assumes that the curator will be publishing the weights of f, and so certain trivial solutions are out of scope. For example, the curator cannot route examples x in D_f to a random labeler and all other examples to some true model. 

The paper sets up the Exact Pareto Learner as a solution for the problem, which is the minimizer of a loss function that rewards uniformity on D_f and correctness on D \ D_f. These two components are balanced by a parameter theta, which takes a convex combination of these two losses.  

The paper then provides an algorithm that is epsilon, delta - indistinguishable (in the unlearning sense) from the Exact Pareto Learner.  This result holds under Lipschitzness assumptions on the loss functions and boundedness of the parameter space. The paper provides empirical results that demonstrate that finding a local minimum for the Exact Pareto Learner finds a good balance between obscuring labels on D_f and attaining good accuracy outside of D_f.

### Strengths
Strength #1: I think the societal problem of relying on model-based predictions that are false is an interesting one, and one that could use more technical solutions. This work is an interesting first step in that direction. I commend the originality of the setting in this regard.

### Weaknesses
Weakness #1: One obvious solution to this problem is retain a model mixing D_f into the training distribution with random labels. Behind the formalism of the paper, this is essentially what the Exact Pareto Leaner is doing. While I do not think ideas should be penalized for being simple, this seems especially straightforward.  

Weakness #2: The entire point of designing an epsilon, delta indistinguishable algorithm is practicality. If the curator wants to protect some set D_f, they would prefer to modify a pretrained model rather than retrain something like an LLM from scratch. Thus, it is somewhat disappointing that the indistinguishable algorithm is in fact the impractical one. Theoretically, it requires inverting the Hessian of a potentially large-parameter model. The authors offer an algorithm that circumvents this in the Appendix. However, no empirical results are provided, and the obvious solution essentially wins the day. 

Weakness #3. A third weakness is non-technical and concerns the motivation for the setting. The set D_f is assumed to be known as corrupted by the adversary, the curator, and presumably also the individuals whose data constitute D_f and recognize their features as false. I find it hard to motivate that there is essentially public agreement that these are incorrect features, and yet the adversary would be able to execute the hypothetical fraud where they demonstrate a model’s output as a pretext for causing harm to the individual (e.g. by denying insurance coverage). To put it another way. the paper implicitly imagines a scenario where an individual with corrupted feature vector x is saved by demonstrating to some third-party that f(x) is random. However, it seems to me that, realistically, they are equally saved by demonstrating to the third-party that x is corrupted. 

Weakness #4: I think the presentation would be improved by moving the security game in appendix A to the main body. It is difficult to exactly understand the threat model without it.

### Questions
Can the authors provide an explanation for the limitations of Algorithm 3? Why was it not evaluated empirically? Is it because the estimation error of the Hessian is too high to achieve utility? Some other reason?

### Soundness
3

### Presentation
3

### Contribution
2
