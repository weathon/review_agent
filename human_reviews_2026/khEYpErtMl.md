# Calibrating Generative Models

- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Generative models frequently suffer miscalibration, wherein class probabilities and other statistics of the sampling distribution deviate from desired values. We frame calibration as a constrained optimization problem and seek the closest model in Kullback-Leibler divergence satisfying calibration constraints.  To address the intractability of imposing these constraints exactly we introduce two surrogate objectives for fine-tuning: (1) the relax loss, which replaces the constraint with a miscalibration penalty, and (2) the reward loss, which converts calibration into a reward fine-tuning problem. We demonstrate that these approaches substantially reduce calibration error across hundreds of simultaneous constraints and models with up to one billion parameters, spanning applications in protein design, image generation, and language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose two methods for performing generative modeling under moment constraints:

(1) *CGM-relax* encodes the constraint via a penalty on an unbiased estimate of the squared constraint deviation. (2) *CGM-reward* leverages an equivalence to the maximum-entropy principle, for which a solution of the considered problem can be computed in closed form, in the non-parametric setting and under complete knowledge of the underlying distribution. Thus, the idea behing CGM-reward is to minimize the Kullback-Leibler divergence of a generative model to an estimate of said closed-form solution.

For both (1) and (2), the authors propose gradient estimates that they show to be unbiased.

Both methods are evaluated on a range of models with tractable likelihood estimates (Gaussian mixture models, diffusion models, normalizing flows, language models) and tasks (protein design, image generation, 1D toy problem, natural language). Overall, CGM-relax performs favorably to CGM-reward at constraint satisfaction, especially if many constraints are present.

A central limitation of both methods lies in the fact that they rely on tractable estimates of the likelihood.

### Strengths
1. The manuscript is well-written (by and large).

2. The proposed methods are practically relevant.

3. The CGM-reward fine-tuning approach and its connection to the principle of maximum entropy is elegant and insightful.

4. Diverse experiments and rigorous evaluation are performed.

5. The authors highlight limitations openly and clearly (I value this highly, because it lies in contrast to most ICLR submissions).

I need to admit that I am not too familiar with the related work. Hence, it is difficult for me to judge novelty.

### Weaknesses
1. I am confused about the framing: The manuscript claims to be about calibration, but it really seems to be about generative modeling under moment constraints. Without going into proper mathematical definitions (like, e.g., [1]), calibration (typically defined for classifiers) means: "Model uncertainty is in line with true uncertainty". An extension of this concept to generative modeling would then be that the distribution is modeled well in the sense that there is no mode collapse. It seems the authors define "calibration" quite differently. My criticism is just regarding the wording, but I am afraid that many readers will be confused and that the "actual target audience" (those which are interested in generative modeling under constraints) will not discover this work. I therefore suggest modifying the title and the main body of the text accordingly.

2. The relax loss is somewhat naive and I have some concerns about it: In constrained optimization, such penalty formulations typically lead to either poor conditioning of the loss surface (for large $\lambda$) or the constraints are not satisfied (for small $\lambda$). I believe that this penalty method is a reasonable ablation, but I think it would be more interesting to replace it by the augmented Lagrangian method [2], for instance. One could just plug in the derived estimate for the constraint violation (equation 3) and end up with a more mature method that would very likely work better.

If the authors (i) implement my suggestion in weakness 1 or provide me a convincing argument that the term "calibration" is adequate and (ii) run additional experiments with a mature approach for constrained optimization (ideal) or at least add a critical discussion section about the CGM-relax method, I will raise my score.

[2] Magnus R. Hestenes. Multiplier and Gradient Methods. Journal of Optimization Theory and Applications, 4:303–320, 1969.

### Questions
* l. 10: *"wherein class probabilities and other statistics of the sampling distribution deviate from desired values"*. I suggest replacing *"desired"* with *"true"*.

* l.22: *"language models represent gender, race, religion, and age in ways that reinforce societal biases"*. While this is an important point, I believe this is not related to poor calibration, at least not from the definition I am familar with (e.g., [1]). If the generative model reflects societal biases as they are in the data, then it is well-calibrated. Please see my weakness 1 for more information.

* l.120: *"Theorem 2.1. Under assumptions, there exists a unique solution to (4) that has the form"* Would it be possible to re-write this by either (i) listing the assumptions explicitly; or (ii) write something like *"Theorem 2.1. Under the assumptions listed in Appx. ... , there exists a unique solution to (4) that has the form"* Otherwise, this is not a proper theorem.

* l.122: I highly recommend writing out what $\alpha^*$ is. I am afraid that without this information, the theorem statement is incomplete.

* l.136: *"which states that, under conditions,"* Similarly as in my previous comment, it would be helpful for the reader if the authors could write out the conditions or refer to the appendix to read up on them.

* l.213-215: Why is this unintuitive? And what do the authors mean by the score function being *"non-trivial in general"*?

* l.322: *"Consistent with our results in Section 3 we find that optimally-tuned CGM-relax outperforms CGM-reward, which falls short of meeting the calibration constraints."* Should CGM-relax not fall short of meeting the moment constraints, too? (see weakness 2)

[1] Wang, Cheng. "Calibration in deep learning: A survey of the state-of-the-art." arXiv preprint arXiv:2308.01222 (2023).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The article introduces a calibration procedure for generative models. Calibration is understood in the article as matching the expected value of a nonlinear function to a desired target value. The article thus views calibration as a constrained optimization problem and introduces two heuristic approaches to solve these, one based on the penalty method, one called 'max entropy approach', which can, however, be interpreted as minimizing a corresponding Lagrange function.

The article presents numerical results spanning both toy examples and more advanced case studies, including protein folding and vision transformers.

### Strengths
The method has a low complexity and seems to work in practice. Although the degree to which it works successfully is not direclty visible from the experiments. The presentation of the ideas is adequate and the writing is clear.

### Weaknesses
There is very little originality and scientific contribution in the article. Essentially, the article suggests to view calibration through the lens of constrained optimization and applies a penalty method (approach 1) or minimizes a Lagrange dual (approach 2). I found the title and framing misleading, as I would expect statistical guarantees, which, however, cannot be delivered by the ad-hoc nature of the approach (or this would need substantial refinement).

Moreover, the approaches to constrained optimization are largely adhoc (penalty method cannot guarantee constraint satisfaction; approach 2 operates on a fixed multiplier) and cannot guarantee constraint satisfaction. Constraint satisfaction would be important for extracting meaningful statistical guarantees. In addition the problem formulation (minimization subject to expectation constraint) is widely studied in the stochastic optimization community, e.g., under the name of multistage stochastic program.

### Questions
Have the authors looked into knowledge distillation? I could imagine that approaches similar to the ones proposed in the article are frequently introduced as baselines.

In many practical situations one would like to use indicator functions for h(x). In these situations h(x) is no longer differentiable. Did the authors look carefully into this situation? The proposed, gradient-based optimization approaches do not seem to work well for this situation (essentially the gradient of the indicator is zero almost everywhere).

### Soundness
2

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
3

### Summary
This paper introduces a framework to correct distribution-level miscalibration in generative models. Authors formalize calibration as a constrained optimization problem. Because the exact constraint is intractable, they propose two surrogate fine-tuning methods, one uses soft quadratic plus KL regularization (CGM-relax) and one approximates the maximum-entropy projection of the base model toward a target exponential-family distribution (CGM-Reward). Then they tested these to proteins, images, and language domains.

### Strengths
- The writing is cohesive and clear. Problem definition on calibration is sound and clear, and their methods, especially connection with maximum entropy problem is interesting. 
- They also provide a nice practical way to realize theory.

### Weaknesses
- I think the main concern of this paper is about end-point only calibration. CGM adjust only the terminal marginal distribution (e.g. final diffusion sample) rather than the entire probability flow. In other words, we lose access on how the probability path changes by this finetuning process. Someone can say this is not a problem if we can sample from the target distribution (with desired constraints) anyways, but in terms of theory I am not sure if this is the direction we really want. For example FK steering tries to control the pathwise dynamics.

### Questions
- Regarding weakness 1, what is the benefit of this method compared to recent reward-guided finetuning methods that controls pathwise dynamics? Is there any specific problem setup where CGM is the only remedy? I am happy to discuss this more, and raise score.

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
This work formulates calibration of generative models as finding the closest distribution to a base model that matches specified expectation constraints. This paper introduced two fine-tuning losses, CGM-relax (penalized constraint violation + KL to base) and CGM-reward (match exponential-tilt max-entropy target) with unbiased loss/gradient estimators and leave-one-out baselines. The authors tested with protein/image/language generative models and show large reductions in miscalibration under constraints, with a slight degradation in quality compared with the base model.

### Strengths
- The paper includes experiments that showed robust empirical coverage across different modalities (proteins/images/text), with different model architectures, making the approach very usable
- Both algorithms only need sampling, log-density, and score, which most diffusion/flow/LM codebases already support, lowering adoption costs across modalities. The paper also shows how to realize these for continuous-time diffusion and masked LMs.

### Weaknesses
- The CGM-relax method relies on the $\lambda$ parameter, which balances the constraint violation and the KL penalty. The paper currently uses grid search to find optimal values, and an analysis of the sensitivity to $\lambda$ and heuristics for setting it would make the work more practical.
- Scale-up to larger LMs is untested. TinyStories-33M and ESM3 1.4B are helpful, but applying CGM to popular 7-30B LMs would stress the need to compute long-sequence log-probs and scores efficiently
- The method requires tractable likelihoods and scores, making it challenging to extend to VAEs/GANs

### Questions
- For images, did you compute class-conditional FID/FJD to decouple class-mix shifts from visual quality?
- How does $\lambda$ interact with temperature, classifier-free guidance, or noise schedules? 
- How sensitive is CGM-relax to the choice of $\lambda$? For a new problem, do you have any heuristics or intuitions for setting a good initial raneg for $\lambda$ to avoid a costly search?

### Soundness
3

### Presentation
3

### Contribution
3
