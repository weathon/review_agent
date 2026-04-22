# Likelihood Paradox Mitigation using Entropy Manipulation with Normalizing Flow in OOD Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 8

## Abstract
Deep generative models that can tractably obtain the likelihood of input data, such as normalizing flows, often assign unexpectedly high likelihood to out-of-distribution (OOD) inputs that were unseen during training. We address this likelihood paradox by manipulating input entropy in a way that reflects semantic similarity, so that OOD samples receive stronger perturbations than in-distribution samples. We provide a theoretical analysis that demonstrates how entropy control increases the expected log-likelihood separation toward the in-distribution, and explain why our procedure works without any additional training of the density model. We then evaluate against likelihood-based OOD detectors on standard benchmarks and find that our method consistently improves AUROC over baselines, supporting the proposed explanation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the likelihood paradox in normalizing flow–based out-of-distribution (OOD) detection, where OOD samples can receive higher likelihoods than in-distribution (ID) data. The authors propose Semantic Proportional Entropy Manipulation (SPEM), a training-free post-hoc method that perturbs test inputs with Gaussian noise scaled by semantic similarity to ID embeddings. Theoretical results show that entropy manipulation enlarges the expected log-likelihood gap between ID and OOD, and experiments across multiple datasets and models show consistent AUROC improvements.

### Strengths
1. The work develops rigorous information theory based on the entropy power inequality and KL-divergence decomposition.  The mathematical reasoning provides valuable insight into the behavior of normalizing flow likelihoods.

2. SPEM operates purely as a post-hoc procedure without retraining the model, requiring only a pretrained encoder and a simple similarity-based noise scaling. This makes it straightforward to implement and integrate into existing normalizing flow OOD detection pipelines.

3. The experiments are extensive, covering different ID/OOD dataset pairs and two representative normalizing flow architectures (ResFlow and Glow). The proposed SPEM and its variant achieve higher AUROC than previous likelihood, ratio, and complexity based methods.

### Weaknesses
1. The related work on normalizing flows and the likelihood paradox appears in Section 4, after the proposed method and theoretical analysis. If the readers are not familiar with normalizing flows and the likelihood paradox, these background explanations should appear before the method section to provide necessary context and make the paper more clear. 

2. The central idea of this manuscript, modifying the likelihood of normalizing flows through Gaussian noise, has already been investigated in [1], which analyzed the relationship between input complexity and OOD detection performance. Although this manuscript introduces a semantic similarity–based scaling scheme and provides theoretical analysis, the novelty remains limited and should be more clearly distinguished from existing literature.

3. Although the theoretical results are valuable, the empirical evidence does not clearly demonstrate how the observed improvements relate to the proposed theoretical framework. Stronger ablation studies or direct empirical validation linking the theoretical predictions to observed performance would make the paper more convincing.

[1] Osada G., Takahashi T., Nishide T. *Understanding likelihood of normalizing flow and image complexity through the lens of out-of-distribution detection.* AAAI 2024.

### Questions
What is the computational cost of maintaining and searching the memory bank during inference compared to other OOD detection methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to address the "problematic phenomenon" first identified in [1]: when using the likelihood of a probabilistic generative model as the criterion for out-of-distribution (OOD) detection, certain dataset pairs exhibit the unintuitive behavior that OOD samples receive **higher likelihoods** than in-distribution (ID) samples. Formally, this can be written as:

$\mathbb{E}_{x\sim P} [\log P_{\theta} (x)] < \mathbb{E}_{x\sim Q} [\log P_{\theta} (x)]$

where $P$ denotes the distribution of in-distribution (ID) data, $Q$ denotes the OOD data distribution, and $P_\theta$ is the generative model.

[2] provided an entropic explanation for this phenomenon through the following decomposition:

$\mathbb{E}_{x\sim P} [\log P_{\theta} (x)] - \mathbb{E}_{x\sim Q} [\log P_{\theta} (x)] = KL(Q||P_\theta) - KL(P||P_\theta) + \mathbb{H}(Q) - \mathbb{H}(P)$

When the entropy of the in-distribution is higher than that of the OOD data, i.e., $\mathbb{H}(P) > \mathbb{H}(Q)$, the right-hand side can become negative, thereby explaining the counterintuitive result observed in [1].

Building on this entropic perspective, the current paper argues that increasing the entropy of the OOD data $\mathbb{H}(Q)$ via Gaussian perturbations can mitigate and potentially resolve this issue.

To this end, the authors propose the **SPEM (Semantic Proportional Entropy Manipulation)** algorithm, which adaptively adds Gaussian noise to OOD-like inputs to increase their entropy and correct the likelihood ordering between in-distribution and out-of-distribution samples.

**Reference**

[1] Nalisnick, Eric, et al. "Do deep generative models know what they don't know?." arXiv preprint arXiv:1810.09136 (2018).

[2] Caterini, Anthony L., and Gabriel Loaiza-Ganem. "Entropic issues in likelihood-based ood detection." I (Still) Can't Believe It's Not Better! Workshop at NeurIPS 2021. PMLR, 2022.

### Strengths
The paper is clearly written and easy to follow. The overall flow, from the problem formulation, to the motivation, and finally to the detailed method description, is well organized and presented in a coherent and understandable manner.

### Weaknesses
This paper suffers from a **fundamental conceptual issue**.

First, [2] provides an explanation for the phenomenon described in [1]; it does not define a OOD detection problem itself. 

The paper raises the question:

> If we increase OOD entropy via perturbation, wouldn’t the expected log-likelihood difference align with intuition?

However, this is **not a fundamental question** for solving OOD detection. My answer would be: yes, if we artificially increase the entropy of OOD data by any means, the numerical results can indeed be made to align with our intuition. Yet this only adjusts the surface-level metrics to appear more intuitive, it does not actually address the core challenge of detecting out-of-distribution samples. The earlier workshop paper [2] merely interpreted the paradox from an entropic perspective; logically, such an explanation cannot be inverted to construct a genuine solution.

Due to this misunderstanding, the proposed SPEM method ends up being a form of **circular reasoning**. It requires knowing (or at least assuming) that a sample is likely to be OOD in order to assign stronger Gaussian perturbations and thereby increase its entropy. But determining such OOD likelihood is precisely the purpose of an OOD detection method. If another mechanism must first identify OOD tendencies, then the practical meaning of SPEM as an OOD detector becomes questionable.


Because of these fundamental concerns, I lean toward a Reject (Score: 2). However, I acknowledge that the authors have clearly articulated their reasoning and presented their ideas in a coherent manner. I remain open to discussion: if the authors can convincingly clarify or rebut these conceptual issues in their response, I would be willing to reconsider my position.


**Reference**

[1] Nalisnick, Eric, et al. "Do deep generative models know what they don't know?." arXiv preprint arXiv:1810.09136 (2018).

[2] Caterini, Anthony L., and Gabriel Loaiza-Ganem. "Entropic issues in likelihood-based ood detection." I (Still) Can't Believe It's Not Better! Workshop at NeurIPS 2021. PMLR, 2022.

### Questions
See **Weaknesses**.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors consider OOD detection, where a model $P_\theta$ trained on samples from $P$ must decide whether a sample $X \sim Q$ satisfies $Q = P$ or $Q \neq P$. The authors draw on earlier literature showing that counterintuitive results can occur when the entropy of $P$ (the in-distribution) is larger than the entropy of $Q$ (the out-distribution). This leads to cases where $P_\theta$ can assign higher likelihoods to samples from $Q$ than from $P$ when the two differ.

Inspired by this, the authors propose an “entropy manipulation” method based on a nonparametric nearest-neighbor–type idea: all inputs to the OOD test are perturbed by noise, but that noise is scaled higher when $\text{embed}(x_{\text{test}})$ is not cosine-similar to any $\text{embed}(x_i)$ in a stored set of $x_i$ from $P$, using a pretrained semantic embedding model. (The scale depends on the most similar $x_i$.)

Formally, the authors draw on a bound that shows the benefits of increasing the entropy of $P$- and $Q$-input samples, but in a specific way that raises the entropy more for samples from $Q$. This has the effect of increasing the separation between $\mathbb{E}P[\log P\theta]$ and $\mathbb{E}Q[\log P\theta]$, improving OOD detection.

### Strengths
- Thorough familiarity with, and citation of, relevant background work

- Clear theorems and intuition on how to “separate” samples from $P$ and $Q$ to improve OOD tests

- A method presented that can sometimes achieve this separation

- Careful and complete quantitative evaluation on many common OOD dataset pairs, with results also reported for existing OOD tests

### Weaknesses
Non specifically, but I have some questions, see below.

### Questions
**Question about Equation (2)**

“In cases where detection is performed solely by comparing likelihoods, the following inequality holds,

$$E_P[\log P_\theta] < E_Q[\log P_\theta]$$"

What does ‘it holds’ mean here? Do you mean ‘it always holds,’ or ‘surprisingly, it (only) sometimes holds’?

Cross-entropy is greater than or equal to entropy, and these are negatives of those quantities, so negative cross-entropy $<$ negative entropy. The inequality could be true or false depending, for example, on whether $P_\theta = P$ or $P_\theta = Q$.

Could you please clarify what you meant more precisely? It would help to edit the sentence before (2) and to follow (2) with a short explanation.

**Results in Table 1**

There are several OOD accuracies equal to 1 or close to 1 in Table 1. Could you elaborate on what this means, especially relative to the first row (“Likelihood”)?

Does this mean that, formally, the distributions of the test statistic $\log P_\theta$ are disjoint for the two datasets after applying SPEM, while they might have been quite overlapping originally (e.g., leading to $\approx 0.58$ accuracy on CelebA/CIFAR)? Such a large jump seems striking and deserves some discussion.

More generally, many results show your method achieving near-optimal OOD detection on standard datasets in this subarea of machine learning. Could you provide further analysis on what this means? Why or why not do you believe these problems are now “solved”?

I do see in the additional experiment with Gaussian inputs of varying scales that you point out that entropy is not the whole story. How does that observation relate to the results on the real datasets?


**Statistical properties of the optimal embedding model for the scaled Gaussian**

It would be especially interesting to hear more about the following: using the bound in Equation (3) and the results of Theorem 3.1, what can we say about the optimal embedding model that achieves the $\lambda$-scaling for the noise that maximally separates the $P$ and $Q$ cross-entropies with $\log P_\theta$?

What properties must these embeddings have in terms of $P$ and $Q$, and what are some cases where it is or is not achievable to improve over the standard likelihood test using the presented method?

An example property might be something like non-overlap in the densities of the random variables defined by the pushforward through the embedding.

In other words, the fact that we are modifying inputs from P or Q based on their estimated relationship to samples from P, seems to imply that we are augmenting our OOD statistic with another OOD statistic, where the latter statistic may itself be subject to caveats of OOD statistics. What are those?

### Soundness
3

### Presentation
3

### Contribution
3
