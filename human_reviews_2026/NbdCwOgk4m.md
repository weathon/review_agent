# Efficient Hyperparameter Tuning via Trajectory Invariance Principle

- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
As hyperparameter tuning becomes 
increasingly costly at scale, efficient tuning methods are essential. Yet principles for guiding hyperparameter tuning remain limited.
In this work, we seek to establish such principles by considering a broad range of hyperparameters, including batch size, learning rate, and weight decay.
We identify a phenomenon we call \emph{trajectory invariance}, where pre-training loss curves, gradient noise, and gradient norm exhibit invariance--closely overlapping--with respect to a quantity that combines learning rate and weight decay. This phenomenon effectively reduces the original two-dimensional hyperparameter space to one dimension, yielding an efficient tuning rule: follow the salient direction revealed by trajectory invariance. Furthermore, we refine previous scaling laws and challenge several existing viewpoints.
Overall, our work proposes new principles for efficient tuning and inspires future research on scaling laws.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a trajectory invariance phenomenon in language model pretraining, where loss, gradient noise, and gradient norm curves remain nearly identical across runs sharing a composite of learning rate and weight decay. This observation suggests that the two-dimensional hyperparameter space of learning rate and weight decay can be reduced to a single tuning direction. The authors also discuss implications for hyperparameter scaling laws and efficient hyperparameter tuning strategies.

### Strengths
- The paper identifies a trajectory invariance phenomenon in pre-training dynamics, showing that certain combinations of learning rate and weight decay yield similar training trajectories. This observation may help reduce the effective hyperparameter search space and improve tuning efficiency.

### Weaknesses
- The study is conducted on a single model size (164M-parameter model), meaning there is no cross-scale analysis. Without experiments on multiple model sizes (e.g., ≥1B parameters), the generality and robustness of the proposed principle cannot be established.

- The paper claims to refine or challenge existing scaling laws, but prior scaling-law research derives conclusions from cross-scale comparisons across multiple model sizes. Since this work relies on a single-scale setting, the evidence is not sufficient to support the broader implications suggested by the paper.

### Questions
Please refer to the Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents how the space of hyperparameters in llm optimization may be simplified by considering solely the effective learning rate (product of learning rate and weight decay) rather than learning rate and weight decay separated. This claim is corroborated through numerous experiments that help understand its relevance and its limitations in different situations. Namely, the authors show that for a small number of iterations (independent of batch size interestingly), losses overlap for a given learning rate and not the effective learning rate. This fact guides the authors to try batch size schedulers that help retrieve invariance of the losses with effective learning rates. In general the experimental results, looed through many faces help revise and question our understanding of some learning rate scaling rules such as the square root LR-BS scaling rule.

### Strengths
- The paper is very well presented and written. Key concepts are introduced early. Each experiment is well thought and illustrates well the claims.
- The relevance of the effective leraning rate (ELR) is presented under many facets: overlapping losses, pairwise relative distances, optimal direction in the hyperparameter search space.
- The authors fit some scaling laws with effective learning rate that could be reused by the community.
- The authors also show that gradient noise can overlap with effective learning rate.
- The batch-size scheduling appears new to me and is a very interesting approach to capture benefits of various batch size scales.
- Discussion of related work is well done. The reader understands that similar observations were done and understands how this paper differs.

### Weaknesses
- It is unclear whether the effects are observed at different model sizes. 
- The experiments are limited to a single optimizer. SignSGD or Muon for example would be great candidates to explore the claims.

### Questions
- The authors point that previous studies found different conclusions in terms of scaling laws for example. How is it possible? What changed between these studies?
- Can the authors test their claim on another optimizer?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces "trajectory invariance," a phenomenon in deep learning pre-training where loss curves and other metrics closely overlap. Early in training, this invariance is with respect to the learning rate (LR), meaning runs with the same LR but different weight decays (WDs) follow similar paths. As training progresses, particularly with a sufficient number of iterations, this invariance shifts to the effective learning rate (ELR), defined as the product of LR and WD. The authors demonstrate that this principle effectively reduces the two-dimensional tuning space of LR and WD to a single dimension, proposing a more efficient tuning strategy: tune along the "salient direction" (ELR in small-batch or sufficient-iteration regimes, and LR in large-batch regimes) to optimize performance. The work also refines existing hyperparameter scaling laws, challenging prior viewpoints on optimal batch size and learning rate scaling.

### Strengths
1. The paper introduces a novel scaling law for hyperparameter tuning, providing insights for practical applications.
2. The comprehensive experiments and discussions indicate that the proposed principle demonstrates some robustness.

### Weaknesses
1. The paper is poorly written and falls significantly below the standards expected of a top-tier conference. Many concepts are introduced before being properly defined (even without definition). The figures are also of very low quality, curves often overlap and are difficult to distinguish. The figure legends lack consistency; for instance, in Figure 1, the left subfigure uses “2^-12” (which is already too casual for scientific publications) while the right subfigure uses “2e−4.” Additionally, Figure 3’s caption redundantly includes “For example, For example”. The appendix appears incomplete, particularly Sections A.2 and A.3.
2. There is a lack of analytical discussion regarding this invariance phenomenon. For instance, insights can already be drawn from the update rule of algorithms incorporating weight decay. A typical update takes the form $w_{t+1} = (1 - \eta \lambda) w_t + \eta g_t$,
where $g_t$ denotes a gradient estimator, such as the stochastic gradient in SGD or $m_t / (\sqrt{v_t} + \epsilon)$ in Adam. From this expression, it becomes intuitive that early in training, the gradient magnitude is relatively large and thus dominates, making the learning rate $\eta$ the more influential factor. As training progresses and the gradients tend to converge, the first term with coefficient $(1 - \eta \lambda)
$ becomes dominant, thereby amplifying the relative effect of the factor $\eta \lambda$.
3. The paper makes rather strong claims, such as “we find that scaling law for LR is wrong.” The authors should provide sufficient justification for such assertions. Are the experimental findings reported in existing scaling law studies incorrect, or do those conclusions simply not apply under the specific conditions of the authors’ setup? Does this also imply that the authors’ findings lack generality and may not extend to the experimental setups used in those prior studies?
4. The generality of the proposed invariance remains uncertain. How does this phenomenon manifest across different optimizers, tasks, and network architectures? Moreover, can the same behavior be consistently observed when employing various deep learning techniques, such as different forms of normalization?
5. As an empirical study, the paper omits certain important implementation details. For instance, how is $\Sigma$ computed in practice? Is it estimated using an additional sampled batch? What batch size is used for this estimation, and how reliable or accurate is the resulting approximation?

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper considers the problem of hyperparameter tuning. Concretely, they propose a phenomenon in neural network training, called "trajectory invariance", which claims that the loss curves admit an invariance w.r.t. $\gamma$ - a quantity that combines the learning rate and weight decay. Based on such a phenomenon, the authors propose a promising way to tune multiple hyperparameters in practice. For example, practitioners can fix the learning rate and only tune the weight decay, turning a problem of 2 hyperparameters into one effectively.

### Strengths
The paper is easy to follow.

### Weaknesses
First things first, I am not an expert in this field, so maybe my evaluation is a bit off. However, it seems to me that the paper is over-claiming (a lot):

1. The paper claims to have discovered the "trajectory invariance principle" for efficient hyperparameter tuning. However, there is one critical thing: the paper only conducts its experiments on the AdamW optimizer (and with the standard setting as well). Note that in AdamW, the weight decay $\lambda$ is a simple, direct decay applied to the weights, __separated__ from the gradient update. Therefore, I assume that the joint effect of learning rate $\eta$ and weight decay $\lambda$ parameter might be simple, leading to a simple term ELR $\eta = \lambda \cdot \eta$ that can capture the trajectory (in some sense). However, this does not mean that the analysis can be applied to other optimizers, including Vanilla Adam, and especially LION, Muon, of which the update principles are very different. 

2. The above suggests that the finding might not be a "principle", but rather an "artifact" that happens in some sense only for AdamW. In such a case, I expect a (simplified) theoretical analysis on why this phenomenon happens for AdamW (or at least some strong intuitive explanation with concrete evidence), so that the paper might be interesting in some sense. Unfortunately, this is not the case for this paper.

3. Moreover, if we reluctantly accept the experiments restricted to AdamW, I still think that the experiments are insufficient. For example, the authors explicitly acknowledge that optimizer states of AdamW, like $\beta_2$, are "important" and "influence" the training. However, the authors only consider the experiments with one value of $\beta_2$

In summary, I believe that the empirical findings in this paper are insufficient for a high-quality paper. I strongly recommend that the authors should at least validate their findings with: (1) more optimizer settings, (2) more configuration in each optimizer, ... or try to develop some (simplified) theoretical justifications for their findings, so that they can boost this paper's contribution. As of this state, I unfortunately cannot recommend an acceptance for this paper.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
