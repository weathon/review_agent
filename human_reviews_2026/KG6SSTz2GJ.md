# Amortising Inference and Meta-Learning Priors in Neural Networks

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
One of the core facets of Bayesianism is in the updating of prior beliefs in light of new evidence$\textemdash$so how can we maintain a Bayesian approach if we have no prior beliefs in the first place? This is one of the central challenges in the field of Bayesian deep learning, where it is not clear how to represent beliefs about a prediction task by prior distributions over model parameters. Bridging the fields of Bayesian deep learning and probabilistic meta-learning, we introduce a way to $\textit{learn}$ a weights prior from a collection of datasets by introducing a way to perform per-dataset amortised variational inference. The model we develop can be viewed as a neural process whose latent variable is the set of weights of a BNN and whose decoder is the neural network parameterised by a sample of the latent variable itself. This unique model allows us to study the behaviour of Bayesian neural networks under well-specified priors, use Bayesian neural networks as flexible generative models, and perform desirable but previously elusive feats in neural processes such as within-task minibatching or meta-learning under extreme data-starvation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a meta-learning model to infer better priors for Bayesian neural nets (BNNs) using a neural process (NP)-based approach, which treats the BNN's weights as latent variables.
To ensure scalability, it relies on layerwise factorizations and introduces within-task minibatching.

### Strengths
- The paper proposes an interesting and novel combination of BNNs and NPs for an important unsolved task in the Bayesian deep learning literature.
- The proposed objective (eq. 10) consists of interpretable and well-motivated terms.
- The approach is scalable and seems to provide promising results across a range of experiments.

### Weaknesses
- Proposition 1 is promising in that the objective is well-motivated but does not provide any guarantees that the vague definition of "small" in Definitions 1–3 is actually achievable. For example, are all three terms minimizable at the same time, or do they balance each other?
- The empirical evaluation is promising but currently rather unusable, as it is completely devoid of any information on architectures, hyperparameters, training procedures, etc.
- No code is provided.

## Minor weaknesses
- Hiding the definitions of the three desiderata (167–170) in the appendix makes that part of the paper feel rather handwavy, even though it isn't.
- Details are missing from the results, e.g., over how many seeds are the error bars in Figure 6?

### Questions
- Q1: The paper doesn't discuss computational cost and runtime requirements. How expensive is the approach?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of defining useful priors in a Bayesian deep learning setting by introducing the 
Bayesian Deep Neural Process (BDNP). The central idea is to meta-learn a parametric prior for Bayesian Neural Network 
(BNN) weights, coupled with amortized variational inference, using a meta-dataset of related tasks. Conceptually, the 
BDNP can be viewed as a neural process (NP) where the latent variable represents the weights of a BNN, and the decoder 
is the BNN itself. The paper presents compelling empirical evidence demonstrating the BDNP's ability to achieve 
high-quality approximate inference compared to other VI methods and to learn meaningful priors effectively.

### Strengths
The presentation of the paper (language and structure) is well done. The theoretical section is sound and nicely concise. 
The experimental section is comprehensive and well-executed. Presented ablation studies, particularly those evaluating the 
quality of approximate inference against other VI methods and the qualitative and quantitative analysis of learned priors, 
strongly support for the paper's claims. The authors provide a commendably clear and realistic discussion of the 
limitations of their approach, such as potential scalability issues for very large or complex architectures.

### Weaknesses
- **Training Loss Justification:** The paper's core training objective, PP-AVI, deviates from standard NP-losses like NP-VI. 
While Appendix A.4 provides a detailed justification for this choice over NP-VI, this reasoning should be
(at least partly) integrated into the main text. Furthermore, the PP-AVI 'loss' is formulated as a maximization 
objective rather than a loss.
- **Clarification on Within-Task Minibatching:** The claim that "the ability to minibatch a forward pass over a given 
context set is rare in the context of neural processes" requires more elaboration. The authors acknowledge
that inference in BDNP can be viewed as Bayesian context aggregation in NPs [1]. Consequently, iteratively updating the
latent posterior over multiple mini batches via sequential Bayesian inference is also applicable to classical NPs. 
This also works for other context aggregation mechanisms such as mean-aggregation or max-aggregation. Only context 
aggregation in the Attentive NP can not be minibatched, but the same holds for tha Attentive BDNP (AttBDNP). Furthermore, 
the method of using gradients from a random minibatch during training (Appendix C) is a significant detail. This 
approximation introduces potential biases and its implications (e.g., impact on convergence, generalization) should be 
more prominently discussed in the main text. Additionally, the assertion that "our scheme maintains the ability for the 
BDNP to learn to generalize predictive inference across various context set sizes during training" requires theoretical 
or empirical evidence to support it.
- **Evaluation Metrics for Correlated Samples:** The per-datapoint Log Posterior Predictive Density (LPPD) is used as one
primary metric without a formal definition in the paper. More importantly, a key advantage of latent NPs (including BDNPs) 
over conditional NPs (CNPs) is their capacity to produce correlated function samples. LPPD, being a per-datapoint metric,
may not fully capture or adequately evaluate this crucial aspect of model calibration and uncertainty representation. 
It is recommended to also evaluate the posterior predictive log-likelihood over the entire target set, i.e., 
$p(y^{1:M}_T\mid x^{1:M}_T, D_C)$, as this metric would provide a more holistic assessment of the model's ability to 
generate coherent and well-calibrated correlated function samples.

[1] Volpp et al., 'Bayesian Context Aggregation for Neural Processes'

### Questions
- In the experiments comparing BDNPs to standard NPs, did you also learn the prior for the latent variable?

### Soundness
3

### Presentation
3

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
This work aims to incorporate a meta-learning prior into the weight parameters of Bayesian Neural Networks (BNNs). In doing so, it builds BNNs that can be interpreted as a type of latent neural process model, where the latent variable is replaced by the posterior distribution of the BNN’s weights, which varies depending on the input. Specifically, it introduces an inference network that generates pseudo-labels for each linear layer and updates the layer’s posterior using these pseudo-labels and the prior. Once the parameters of the inference network and the BNN prior distribution are learned, indicating that the BNN has a well-posed prior, it evaluates the corresponding BNN's ability to perform various tasks effectively. Furthermore, it investigates the crucial role of approximate Bayesian learning under a well-posed prior.

### Strengths
* This work introduces an interesting idea of incorporating a meta-learning prior into BNNs and establishes a connection to neural processes.
*  It also presents an amortized linear layer structure that enables each layer to function as a Bayesian layer with an amortized prior.

### Weaknesses
* Although the proposed method is technically sophisticated, it is unclear what specific problem it aims to address with the proposed structure. For instance, it is not evident whether the main contribution lies in emphasizing the amortized prior for BNNs and its benefits, or in investigating approximate training for BNNs with a well-chosen prior.

* The training procedure for the parameters of the inference network and the prior distribution is insufficiently described. Beyond presenting the loss objective, a pseudo-code illustration or detailed algorithmic explanation would be necessary for clarity.

* Moreover, the quality of the learned prior and the corresponding posterior is likely influenced by the design of the inference network. However, this aspect appears to be underexplored in this work.

### Questions
* It appears that the parameters of the inference network and the prior distribution are jointly trained according to Eq. (10). The ELBO term seems primarily used for training the parameters of the prior distribution, while the conditional likelihood for the target task appears to update both the inference network and the prior distribution. Is this interpretation correct? If not, could you please clarify this point further?

* In this training procedure, was the inference network pre-trained, or was it trained jointly from scratch?

* How is the scale of prior learnability (ranging from 0.0 to 1.0) measured in Figure 7? Moreover, if the amount of data is limited and the prior is well-posed, shouldn’t the prior be beneficial? Why does the performance of BDNP with a prior-learnability of 1.0 degrade so significantly in Figure 7(a)?

* The following works seem relevant to the concept of meta-priors for BNNs and neural processes, and might be useful to include in the related work section:

Hierarchical Gaussian Process Priors for Bayesian Neural Network Weights, NeurIPS 2020

Bayesian Convolutional Deep Sets with Task-Dependent Stationary Prior, AISTATS 2023

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
This paper introduces a new model called a Bayesian deep neural process (BDNP), which combines aspects of the global inducing point variational posteriors (Ober and Aitchison 2021) and the (latent) neural process (NP) (Garnelo 2018). It is a member of the neural process family, although it differs in that the latent variable is the parameters of the decoder rather than an input to the decoder. Like Ober and Aitchison 2021, the variational posterior uses a psuedo output and a pseudo noise term to exploit the conjugacy of Bayesian linear regression per layer. Unlike Ober and Aitchison 2021, there is a psuedo output and a pseudo noise term for each data point and, instead of treating them as trainiable parameters, the BDNP uses amortization. That is, they are outputs of a trained “inference network”, which is analogous to the encoder of an NP. The model is trained by minimizing a log posterior-predictive density, as in an NP, and a ELBO term, as in VI. In the experiments, the paper shows better KL to the true posterior on a toy example, the ability to meta-learn BNN priors in 1D regression tasks, a comparison of different inference methods under the same meta-learned prior, and an experiment restricting the learned prior.

### Strengths
- This is an interesting paper that addresses shortcomings of Bayesian deep learning in a new way, as far as I know. The complexity of the model is both a pro and con. 
- I like the idea of amortizing the psuedo observations of Ober and Aitchison 2021
- Figure 6 was interesting. This shows the learned prior is useful regardless of the inference method. 
- The experiments focus on meaningful questions, rather than the largest scale experiments

### Weaknesses
- There are a few differences with a standard NP, enough that I wonder if this is really an NP (see my questions below). 
- Overall, this is a fairly complicated model, which makes it difficult to understand what each component is doing. I appreciated the discussion of the objective function in the appendix, but I think some of this discussion should appear in the main text. The objective doesn’t feel well motivated as it’s written now. More discussion of how this differs from a standard NP would help. 
- The introduction discusses how BNNs revert to GPs in the infinite width limit but in the finite width limit considered in this paper, BNNs are not GPs. There are many criticisms to make of BNNs, but I don’t think this one is relevant. 
- Unless I missed it, the experiments seem to lack details (architectures, optimizers, etc)
- I found figure 1 (b) a bit confusing. Is this for a one hidden layer network? 
- Definitions 1, 2, and 3 aren’t well-defined because they reference a quantity being “small”, but it’s not clear what small means. 
- There are a few statements that are too strong for me, e.g. “the impressive performance of our explicitly Bayesian meta-learning setup”, “The BDNP’s approximate posterior is a very good on”

### Questions
- Unlike an NP, there is no aggregation of encodings in the BDNP, to ensure permutation invariance. In an NP this is needed to ensure exchangeability of the stochastic process. Instead, the aggregation happens in the loss function. In the appendix, you write that “BDNP and BDAM exhibit this through permutation invariance with respect to the context observations and permutation equivariance with respect to the target inputs”. Can you explain how BDNP defines an exchangeable process without the aggregator? 
- This relates to the above question: I’m confused about how exactly the model meta learns the prior. In an NP, the information from the multiple datasets comes from the encoding. The global latent variable is conditioned on the encoding. My understanding is that this is the “meta” learning done in an NP. In the BDNP, the encodings are, I would argue, variational parameters (as in how they are used in Ober and Aitchison 2021). The prior parameters of the BNN are trained similarly to how they would be trained in standard VI, using the fact that the ELBO is a lower bound on the marginal likelihood, which is used for model selection. Note that in an NP, only the log predictive term is in the loss function, not the ELBO term. Can you comment on how the prior meta learning is different from a standard NP? Am I understanding the BDNP correctly? 
- You train a separate inference network $g_l$ for each layer, is that correct? This seems like it would create a lot of parameters unless each network is small.
- Can you explain again why the BDNP scales favorably in network width? How do you get around the computation of the mean and covariance in equations 6 and 7 being cubic scaling in the width?

### Soundness
3

### Presentation
2

### Contribution
3
