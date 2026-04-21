# Streamlining Prediction in Bayesian Deep Learning

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
The rising interest in Bayesian deep learning (BDL) has led to a plethora of methods for estimating the posterior distribution. However, efficient computation of inferences, such as predictions, has been largely overlooked with Monte Carlo integration remaining the standard. In this work we examine streamlining prediction in BDL through a single forward pass without sampling. For this, we use local linearisation of activation functions and local Gaussian approximations at linear layers. Thus allowing us to analytically compute an approximation of the posterior predictive distribution. We showcase our approach for both MLP and transformers, such as ViT and GPT-2, and assess its performance on regression and classification tasks.

Open-source library: https://github.com/AaltoML/SUQ.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a simple way to obtain the a posteriori distribution over functions represented by Bayesian neural nets (BNNs). The core idea is to assume the product $W^{l} a^{l-1}$ between the weight matrix of layer $l$ and the activations $a^{l-1}$ from the previous layer to be a Gaussian. It follows that the preactivation $h^{l}$ is also Gaussian. Then, by linearizing the activation function $g$, one can approximate the activation vector $a^{l}$ is also Gaussian.

By induction, the output $a^{L} = f(x)$ is also Gaussian, which can readily be used to do inference in close form. For example, one can apply the probit approx. in classification or used in standard acquisition functions for sequential decision-making.

The key difference between this approach and previous approaches is the fact that the linearization is done in terms of the activations (and hence the input), and not over the weights as done in the linearized Laplace approximation. Nevertheless, the authors showed that the performance did not suffer from this linearization.

The important advantage of this approach is its scalability: It only requires a single forward pass with minimal overhead. This is in contrast to MC integration, which requires multiple forward passes, and the linearized Laplace, which requires a backward pass to compute the Jacobian over the weights. Indeed, the authors showed that their method is scalable to vision transformers and (small) language models.

### Strengths
- The proposed method is conceptually simple
- The proposed method might be (see weaknesses below) more efficient than prior methods in obtaining the output distribution.
- While linearization $x \mapsto f(x)$ is performed, the performance doesn't seem to suffer. In fact, it is very close to the linearized Laplace predictive (GLM in the paper).
- The method seems to be scalable (see weaknesses below).

### Weaknesses
I always appreciate the push for simple methods to make BNNs scalable without sacrificing performance. However, I noticed some issues in the presentation of the paper:

- The idea of linearizing the activation function has been done before in the context of continual learning by [Dhawan et al., ICML 2023](https://proceedings.mlr.press/v202/dhawan23a.html). The authors should cite this work and compare their method in detail. I acknowledge that the purposes are different. However, one cannot ignore the fact that they are very similar.

- It is unclear to me why one can still maintain good performance when the network is effectively linear in its inputs. Intuitively, this should not be the case since the representation learning of the base network is negated by the linearization. Could the authors please elaborate in detail (additional, empirical investigations like measuring the degree of linearity of the resulting linearized $f(x)$ might be one way to go about this).

- The main selling point of the approach is that to obtain the distribution over outputs, only a single pass is needed. However, it's currently unclear how much speedup this method provides compared to, e.g., the linearized Laplace (GLM).  A plot/table comparing the wall-clock times between the proposed method and the GLM predictive would be great. 

- To make your selling point stronger, I suggest additional experiments with larger models such as Llama-3 7B etc. Laplace can still be done efficiently over PEFT (see [this](https://proceedings.mlr.press/v235/kristiadi24a.html) and [this](https://openreview.net/forum?id=FJiUyzOF1m&noteId=Uazck8TQ5A)). I believe this functionality is also available in `laplace-torch`: <https://aleximmer.github.io/Laplace/huggingface_example/>. Then, again, please compare the wall-clock times.

### Questions
Please see the Weaknesses section above. Additionally:

- How plug-and-play is the proposed method? Suppose the user has their model that has been Laplace-approximated via `laplace-torch`. What does the user need to do to enjoy the proposed method?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a method for computing an efficient approximation to the predictive distribution given a Bayesian neural network’s (BNN) posterior over the model’s weights. Their method locally linearises the activation functions, allowing for an analytical computation of the predictive distribution assuming a Gaussian weight space posterior approximation. This only requires a single forward pass, in contrast to MC sampling-based methods, and no expensive Jacobians that are necessary when globally linearising a neural network. The application of the method to different architectures and posterior covariance structures is discussed. To demonstrate the applicability of their method, the authors conduct extensive experiments applying their method to Laplace and mean-field variational inference posterior approximations on a range of architectures and datasets.

### Strengths
Overall, I recommend to accept the paper since it 1) proposes a simple, yet effective solution to the problem of efficient Bayesian neural network prediction, 2) appears to be technically correct and is well presented, and 3) provides extensive empirical evidence for the claim that their method performs at least comparable to more expensive alternatives.

The proposed method provides an intuitive solution to the problem of efficient BNN prediction, assuming that we want to consider a posterior distribution over more than just the last-layer weights of the neural network. The writing is clear, the figures and tables are nicely done. I also appreciate the detailed derivations in the appendix. While I have some concerns regarding the baselines, the experiments cover significant breadth. In particular, I like that that the sensitivity analysis was included, as this application highlights the more unique strengths of this method (see weaknesses for more on this).

### Weaknesses
The main weakness of the paper is that it does not consider alternative methods for approximating the predictive distribution of a BNN with a single forward pass. You do not compare the method to last-layer variants of the Laplace approximation and MFVI. This is important to determine if there is any benefit in trying to linearise the neural network to allow for more efficient predictions vs. just treating the network up to the last layer deterministically. For example, when using a last-layer Laplace approximation together with a probit approximation (in the case of classification), making a prediction also only requires a single forward pass and no expensive Jacobian computations. However, it is nice that you include results for which the linearisation w.r.t. the inputs is necessary, i.e. the sensitivity analysis. I think this could be even emphasised a bit more by explicitly saying that this is a downside of using last-layer methods.
Still, in most cases last-layer methods would be sufficient -- in particular also for the larger scale experiments in section 4.2.

Finally, I think the title is too general. As you describe yourself in the introduction, you can deconstruct Bayesian inference into three steps, 1) the specification of the prior, 2) the estimation of the posterior distribution, and 3) the computation of the predictive distribution. Your paper aims to streamline the third step. Hence, a more suitable title appears be “Streamlining prediction in Bayesian deep learning”.

### Questions
**Minor comments and questions**

- When mentioning stable distributions, please define them informally in one sentence first.
- Have you considered applying this method to architectures that use convolutions?
- I find the word “useful” in Figure 1 confusing. What makes this “useful” outlier detection/sensitivity analysis? What is non-useful outlier detection/sensitivity analysis?
- In the paragraph about ensemble methods in the related work section you mention that they typically require multiple forward passes. However, there are efficient ensembling approaches that only require a single forward pass and could be mentioned here, e.g. MIMO [1] and the combination of MIMO with last-layer Laplace approximations [2].
- One potential disadvantage of the method is the implementation overhead, as it requires a custom implementation for each layer-type within an architecture. Have you thought about how to improve the usability of your method?

[1] Havasi et al. (2021). [Training independent subnetworks for robust prediction](https://openreview.net/forum?id=OGg9XnKxFAH).

[2] Eschenhagen et al. (2021). [Mixtures of Laplace Approximations for Improved Post-Hoc Uncertainty in Deep Learning](https://arxiv.org/abs/2111.03577).

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
The paper proposes an alternative to Monte Carlo sampling to compute the predictive distribution of a Bayesian neural network. This is obtained with a single forward pass instead of the multiple ones required by sampling. Some approximations make this possible, namely: Local linearization of activation functions, Gaussian approximation of linear layers, independence assumptions between activations and following layer parameters, deterministic key and query in attention layers and block diagonal structure of multi-head input.
On a variety of task, they empirically show superior or matching performances versus LA sampling, GLM and MFVI sampling.

### Strengths
The experiment sections cover a variety of tasks.

### Weaknesses
The main claim of the paper is to provide a scalable alternative to sampling, but they fail in providing sufficient evidence for such claim. Not all the experiments details are reported, first of all, they do NOT specify the number of samples used in the MC sampling baseline.
For whatever fixed parameter approximate posterior $q$, the proposed method is an approximation of the predictive distribution. Such predictive distribution is exactly what we get in the limit of infinite number of samples with MC sampling. Consequently, MC sampling has to perform better than the proposed method if enough samples are used.

This is point is not studied and not even addressed. Clearly indicating a malicious behaviour.
If the claim of the authors is that their method beats MC sampling "when not too many samples are used" then this would make sense, but this is neither the claim of the authors, neither such "too many" value is studied or attempted to show.

Another minor point is that in Line 230-231 the authors write "residual connection ... resulting distribution can be obtained analytically". Residual connection is essentially a sum, and the elements involved here are two Gaussians. While it is true that the sum of two Gaussian can be "obtained analytically", this step hide a big approximation: you can't store a mixture of Gaussians here, so the only scalable approach is to once again employ independence and "sum" the Gaussians erasing most of the information. While such approximation may be ok if it leads to good empirical performance, the way is presented is not clear and tends to hide such approximation step.

Typo and minor details:
-Line 80 "recent work (2024) developed ... and have shown good performance (2016) (2019)". I get what you mean, but the way the sentence is formulated imply that something developed in 2024 was used in 2016 and 2019...
-Line 260 "extends our the discussion" I guess this is a typo
-Line 324 "ours results in"  should be either "our results in" or "ours result in"

### Questions
Can the author explain why they never mention the number of samples used with MC sampling baseline?

Can the author provide an ablation study on number of samples used?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to address the practical challenge of efficient uncertainty estimation in Bayesian deep learning (BDL). While recent work has improved posterior estimation in BDL, computing predictions typically requires expensive Monte Carlo sampling.
The authors propose a streamlined approach that enables analytical computation of predictions in a single forward pass. Their method combines Local linearisation of activation functions using first-order Taylor expansions and local Gaussian approximations at linear layers.

### Strengths
The paper introduces a novel approach to Bayesian Deep Learning (BDL) by eliminating the need for Monte Carlo sampling through local approximations. It nicely blends local linearisation with Gaussian methods and offers interesting solutions for managing transformer architectures and Kronecker-factored covariance.

Quality-wise, the work is solidly validated with extensive experiments across various tasks and architectures. It thoroughly benchmarks against established baselines, clearly showcasing benefits like improved efficiency and reliable uncertainty estimates. Additionally, the method is versatile, supporting different posterior approximations such as Laplace Approximations and MFVI.

The presentation is clear and well-organised.

### Weaknesses
The treatment of attention layers (using deterministic queries/keys) needs stronger justification. The scaling factor for predictive variance is tuned on validation data - this seems ad hoc. No discussion of potential failure modes or limitations of the local linearisation assumption

### Questions
Please see weaknesses above

### Soundness
3

### Presentation
3

### Contribution
3
