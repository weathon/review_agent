# PateGAIL++: Utility Optimized Private Trajectory Generation with Imitation Learning

- Avg Score: 5.00
- Decision: Accept (Oral)
- Scores: 6, 4, 4, 6

## Abstract
Human mobility trajectory data supports a wide range of applications, including urban planning, intelligent transportation systems, and public safety monitoring. However, large-scale, high-quality mobility datasets are difficult to obtain due to privacy concerns. Raw trajectory data may reveal sensitive user information, such as home addresses, routines, or social relationships, making it crucial to develop privacy-preserving alternatives.
Recent advances in deep generative modeling have enabled synthetic trajectory generation, but existing methods either lack formal privacy guarantees or suffer from reduced utility and scalability. Differential Privacy (DP) has emerged as a rigorous framework for data protection, and recent efforts such as PATE-GAN and \textsc{PateGail} integrate DP with generative adversarial learning. While promising, these methods struggle to generalize across diverse trajectory patterns and often incur significant utility degradation.
In this work, we propose a new framework that builds on \textsc{PateGail\texttt{++}} by introducing a \emph{sensitivity-aware noise injection module} that dynamically adjusts privacy noise based on sample-level sensitivity. This design significantly improves trajectory fidelity, downstream task performance, and scalability under strong privacy guarantees. We further adapt our framework to the local differential privacy (LDP) setting, allowing individual-level protection without reliance on a trusted server. We evaluate our method on a real-world mobility dataset and demonstrate its superiority over state-of-the-art baselines in terms of privacy-utility trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes PATEGAIL++, a framework for generating differentially private (DP) human mobility trajectories. It identifies a key limitation in prior work like PATEGAIL: the use of a uniform privacy noise level, which ignores the fact that not all trajectories are equally sensitive . The core contribution is a sensitivity-aware noise injection module. This module uses a heuristic to dynamically allocate the privacy budget, applying more noise to samples it deems more sensitive. To improve stability, the framework also integrates WGAN-GP . Experiments show that PATEGAIL++ achieves a better utility-privacy trade-off.

### Strengths
1. Clear Motivation: The paper is built on a clear and intuitive motivation: uniform noise is suboptimal for trajectory data, as privacy risks are not uniform (e.g., a home-work route vs. a public-hub route) .
2. Excellent Practical Privacy Evaluation: A major strength is the practical evaluation using a Membership Inference Attack (MIA) in Table 3 . This experiment demonstrates a concrete privacy failure in the baseline PATEGAIL. 
3. Practical System Design: The paper presents a sound engineering solution to a known problem.

### Weaknesses
1.	Heuristic-Based Sensitivity: The core contribution is based on a heuristic, not a theoretical guarantee of semantic privacy. It assumes that high discriminator confidence is a reliable proxy for high privacy sensitivity . This is a pragmatic but unproven assumption. The authors acknowledge this limitation in the conclusion.
2.	Limited Theoretical Novelty: The insight that different data has different sensitivity is intuitive and not, in itself, a deep theoretical breakthrough. The contribution is thus more of an effective application of this idea.
3.	Single-Dataset Evaluation: The entire experimental validation is conducted on a single dataset (Geolife) . While this is the same dataset used by the baseline, the paper's claims would be significantly strengthened if the method's effectiveness was also demonstrated on a different mobility dataset.

### Questions
Q1: Your heuristic equates high discriminator confidence with high sensitivity . But what about trajectories that are both common and sensitive (e.g., a "home-to-work" route)? Could the discriminator give these a low score (since they aren't rare), causing your method to mistakenly label them "low sensitivity" and thus under-protect them?
Q2: The model's practical effectiveness relies on its heuristic, not just its mathematical $(\epsilon, \delta)$-DP guarantee. Is the single MIA attack in Table 3 (which the baseline failed) sufficient validation for this heuristic ? Or could other attacks exist that specifically exploit the heuristic's "worst-case" failure modes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper extends PATEGAIL by adding a "sensitivity-aware" noise injection (rather than uniform noise to achieve differential privacy they use the output of the individual discriminators as a proxy for the sensitivity of a sample for that user) and adding noise to the local reward before sending it to the server to improve the local differential privacy. They evaluate it on one dataset, Geolife.

### Strengths
To improve the accuracy while maintain the privacy is important.
The idea of non-uniform noise over trajectories seems valid and useful.
The paper is well written and easy to follow.

### Weaknesses
Only looking at individual (s,a) pairs seems to entail significant limitations as the Markov assumption has to hold.

The method is only evaluated on a single dataset so no claims about generality can be made.

It is unclear whether using the discriminator to estimate the sensitivity of a sample is sufficient (it only checks individual (s,a) pairs not trajectories), see questions.

The abstract claims that existing methods fail to generalize across trajectory patterns, it is not clear how this approach improves on this.

There are no trends in Table 4 which makes it hard to interpret. It seems as \lambda=5 and \lambda=15 almost give the same results while \lambda=10 and \lambda=20 also give almost the same results. Shouldn't there be some trend?

Some rows in Table 4 lack bold numbers.

### Questions
What is the state in your experiment?

Considering that the method only considers (s,a) pairs, this entails that you assume the Markov assumption holds, what are the consequences of this and how does this limit the applicability of the approach?

How do you evaluate whether using the discriminator to estimate the sensitivity of a sample is meaningful and reasonable? 

When the local discriminators have converged, will they correspond to the true sensitivity of each sample then?

How does the feedback loop of adding noise to the local reward as the local discriminator improves impact the method?

The abstract claims that existing methods fail to generalize across trajectory patterns, how does your approach improve on this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an interesting study that proposes a novel framework, PATEGAIL++, for the task of trajectory generation. It introduces a sensitivity-aware noise injection module, which aims to enhance the fidelity of the trajectories and the performance on downstream tasks. Furthermore, the framework is extended to the Local Differential Privacy (LDP) setting, achieving individual-level privacy protection without relying on a trusted server.

### Strengths
S1: The overall logic of the paper is clear. Addressing the challenges of trajectory generation under differential privacy, it systematically describes the solutions corresponding to different challenges.
S2: Existing approaches suffer from a significant limitation: they apply a uniform noise level to all samples. This practice causes unnecessary utility loss for low-risk samples and provides insufficient privacy for high-sensitivity ones. To address this, the paper introduces a sensitivity-aware module. This module is used to identify the privacy sensitivity level of user trajectories in the training data, allowing for the allocation of an appropriate privacy budget to data of different sensitivity levels.

### Weaknesses
W1: In Section 3.1, there is an apparent contradiction between the formula for privacy budget calculation and its textual description. According to the formula, samples with higher discriminator confidence (i.e., high sensitivity) are allocated a larger share of the privacy budget. However, the textual description states the opposite. The same issue exists in Section 3.2 regarding the privacy budget calculated for each user.
W2: The baseline methods seem to be outdated, some stronger baseline methods are expected to be seen in this paper such as TrajDiff. Particularly, both PATEGAIL and PATEGAIL++ operate in a federated setup with Differential Privacy-style perturbations during reward aggregation, but there is no quantitative evaluation on PATEGAIL.
W3: Table 4 indicates that PATEGAIL++ achieves the best average performance across all metrics when the gradient penalty is 5. This raises a question as to why different gradient penalties were chosen for subsequent experiments. For example, in conjunction with Table 4, it is noted that in the comparison against PATEGAIL (Table 2), PATEGAIL++ was evaluated with no gradient penalty.

### Questions
Q1: Please explain on W1.
Q2: Why are the evaluation results for PATEGAIL not listed in Table 1?
Q3: Please explain on W3.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel approach for the privacy-preserving generation of synthetic trajectories, which extends a previous method called PATEGAIL by integrating a sensitive-aware noise injection module as well as a local differentially private extension. The experimental results obtained demonstrate that the proposed extension called PATEGAIL++ is able to achieve a better utility-privacy trade-off.

### Strengths
-The paper is well-written and the authors have clearly reviewed the recent deep learning methods for the generation of synthetic trajectories. The main contributions of the paper are also clearly summarized. 

-One of the strength of the proposed approach is that it includes a sensitivity-aware noise injection module that adapts the level of the noise to the sensitivity of the trajectory sampled. This enables to add more noise to sensitive trajectories while saving on the privacy budget for non-sensitive ones. In addition, the paper considers the privacy model of local differential privacy, which enables the addition of noise at the level of the user thus limiting the trust assumptions that need to be done with respect to the central server responsible of the aggregation. 

-The proposed approach also relies on the use of the generative adversarial imitation learning to be able to generate trajectories that are realistic. 

-The approach is validated experimentally with the geolife dataset and compared against four different baselines using a wide range of utility measures. The results are promising and demonstrate the potential of the approach. In particular, PATEGAIL++ outperforms PATEGAIL for a wide range of metrics.  The privacy analysis performed using a membership inference attack in the white-box model also demonstrates that PATEGAIL++ achieves a higher privacy level.

### Weaknesses
-The novelty of the approach seems to be limited in the sense that the contribution provided by PATEGAIL++ appears to be only incremental compared to PATEGAIL.

-The federated learning setting should be better justified. In particular, it only mentions at the end of page 3. Rather it seems that the proposed approach is not only limited to this setting but could also be applied in more centralized one, which is actually also an additional benefit of the framework. 

-Currently, there is no description of the four different baselines against which PATEGAIL++ is compared. Additionally, their choice should be justified. The evaluation of the approach also relies only on one dataset while actually assessing it on at least two other ones would help to evaluate its applicability. Finally, the use of a second MIA would also help to strengthen the privacy analysis. It is possible in particular that the attack designed originally against PATEGAIL is not adapted to PATEGAIL++ and that instead another one such as LIRA could be used instead.

### Questions
Please see the main points raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
