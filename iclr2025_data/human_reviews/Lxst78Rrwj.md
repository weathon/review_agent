## Human Reviewer 1

### Summary
The paper utilizes the invariance property of $P(X \mid \text{Pa}(X))$ despite having different $P(\text{Pa}(X))$ to find the causal parents. For this purpose, they choose some source priors, generate corresponding datasets, and verify the invariance. Finally, they show their performance on a variety of synthetic and real-world setups.

### Strengths
The paper is well written. The figure is helpful for understanding the algorithm. The experiments are extensive.

### Weaknesses
## Minor weaknesses:
* The authors mentioned “finding the maximum clique in an augmented bidirectional graph” multiple times but without a proper definition or example/visualization.
* The source variables should be defined in a little more detail.
* What does $P'$ in equation 2 refer to? It should be precise.
* “The intuition is if we can re-sample $D_i$ from $D \sim P(X)$ such that $D_i \sim P_i(X)$,” This is a little unclear. How are $D \sim P(\mathbf{X})$ and $D_i \sim P_i(\mathbf{X})$ different?
* It is unclear how $D_1, D_2, \dots, D_M$ are sampled. How are the $m$ source priors ($P_i(\mathbf{B})$) obtained? Although these are discussed later, some hints/intuitive discussion should be provided earlier in the paper.
* “We cannot compute $P_i(X)$, … we can re-sample $D_i$ from $D$ so that $D_i \sim P_i(X)$” – based on my understanding, the first case is computing the numerical probability table, and the second case is sampling without any such table. This difference should be made clear.
* More details on "downsampling without replacement" are needed.
* An intuitive explanation of the “minimal downsampled rate” is required.


## Major weaknesses
* Suppose $Z$ is not a parent but an ancestor. Shouldn’t we also get variance = 0 (equation 1) in such cases? Does a change in $P(\text{Ancestor})$ affect $P(\text{descendant} \mid \text{ancestor})$?
* Many important concepts are delayed until section 4.2. The authors should consider introducing them earlier in the paper.

I will consider increasing the score after seeing author's response and reviewer discussion.

### Questions
## Questions:
* How are the authors resampling the datasets?
* Do you have to perform this invariance test for all possible parent sets?
* Why is $Pa[B] = \emptyset$ in the definition of set $\mathbf{B}$ (section 4.1)? What does that imply?
* In practice with real-world data, is the variance always zero for all true parents (equation 1)? Why or why not? Should a threshold be used?
* How expensive is it to compute $\phi(X)$? Do we have to iterate over all $X$? And do it again after performing step ii in Theorem 3?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
2

---

## Human Reviewer 2

### Summary
Based on the fact that the distribution of an effect, conditioned on its causes, remains invariant to changes in the prior distribution of those causes, this paper proposes a causal discovery algorithm for large-scale datasets. Specifically, it designs an invariance test, which is achieved by a downsampling scheme. It also makes the best of searching Markov blankets of all variables, to reduce the time complexity. Experiments on synthetic datasets and real-world networks show better scalability.

### Strengths
- This paper is written well, with clear descriptions and motivations. 

- The authors propose practical algorithms for causal discovery, with some interesting theoretical findings, e.g., the basis of a DAG, the minimal downsampling rate, etc. 

- The experiments under synthetic datasets and real-world networks are extensive, which verified the advantages in large-scale datasets.

### Weaknesses
- Some details seem to be missing in the paper.
For example, 

i) Footnote 2 and Theorem 1 tell how to find non-parent sets, whereas how to set the threshold for the variance is not clear. Please give the details in the paper.

ii) How to learn the different priors $P_i(X)$, with the estimated $m$? Did the authors assume some distributions?

- Theorem 2 provides a necessary condition to test whether a subset $Z$ is the parent set of X. However, it is not a sufficient condition. Although the authors stated, “When m is infinitely large, the implication in Eq. (1) becomes bi-directional and $V[P+(X | Z)] = 0$ definitively implies $Z = Pa[X]$”. It is not that clear why this implication we can get. Please elaborate on it more.

- It is better to perform some real-world datasets for validation. This is because bnlearn provides real networks and it generates the data based on the networks. These datasets look like semi-synthetic. BTW, in Table 2, when dealing with small-scale datasets (or even a large-scale dataset Munin), the runtime all seem not to be satisfactory. Please explain it.

### Questions
- Does the proposed causal discovery algorithm work for time-series data? What are the challenges?

I would be very glad to increase my score if the authors could resolve my concerns.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper introduces a novel approach to causal learning through a new invariance test for causality, which underpins a reliable and scalable algorithm for reconstructing causal graphs from observational data. This method leverages a core insight that the conditional distribution of the effect given the cause remains invariant under changes in the prior distribution of the cause. This insight enables a parent-identification process for each variable using synthetic data augmentation. This process is integrated with an efficient search algorithm that utilizes prior knowledge of each effect variable’s Markov blanket, along with the empirically observed sparsity of causal graphs, to significantly reduce computational complexity.

### Strengths
1. The proposed method is rather novel.   
2. Overall, the paper is well-structured and clearly written.    
3. The experiments are extensive, covering 3 types of functional causal models, 6 causal discovery baseline methods, and varying graph sizes.

### Weaknesses
Any thoughts on how to extend your method to handle heterogeneous or time-series datasets?

### Questions
(See above)

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper proposes a new framework that leverages the invariance of effect conditioned on its causes for causal discovery from observational data. The main idea is it try to disturb the p(cause) distribution and see whether the p(effect|cause) would change after the disturbance.

### Strengths
- This work leverages the invariance of conditional distribution and then proposes a downsampling method, combining them to find the parent set.

### Weaknesses
- The main issue is that since the real intervention is not applicable to the observational data, it provides a downsampled technique to approximate the p(effect|cause) after the disturbance, which, however, has no theoretical guarantee. This is, how to guarantee such a downsampled correctly corresponds to the real distribution after the disturbance?
- Since the basis variables would include the leaf vertices, in such a case, changing the prior basis variables will not affect the distribution of their ancestors, and thus it may not have a similar effect to changing prior over source variables.

Typos:
- "Theorem 2" -> "Theorem 4" in Theorem 5

-----
After rebuttal:

After multiple rounds of discussion with the authors, my fundamental concerns remain unresolved. The core issue persists: the proposed approach fails to adequately address the challenge of obtaining interventional distributions from observational data.

As highlighted in seminal works by Pearl [1] and [2], answering interventional questions such as "What would be the impact on the system if this variable were changed from value x to y?" requires explicit causal knowledge. This limitation is well-established in the causal inference literature, as also discussed comprehensively by [3].
The authors' repeated attempts to address my concerns through mathematical manipulations and resampling techniques which do not overcome the fundamental identification problem in causality.

[1] Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge University Press.

[2] Brouillard, Philippe, et al. "Differentiable causal discovery from interventional data." Advances in Neural Information Processing Systems 33 (2020): 21865-21877.

[3] Bareinboim, Elias, et al. "On Pearl’s hierarchy and the foundations of causal inference." Probabilistic and causal inference: the works of judea pearl. 2022. 507-556.

### Questions
See the weakness above.

### Soundness
1

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
5