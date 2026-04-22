# GenSR: Symbolic regression based on equation generative space

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 0

## Abstract
Symbolic Regression (SR) tries to reveal the hidden equations behind observed data. 
However, most methods search within a discrete equation space, where the structural modifications of equations rarely align with their numerical behavior, leaving fitting error feedback too noisy to guide exploration.
To address this challenge, we propose GenSR, a generative latent space–based SR framework following the "map construction $\rightarrow$ coarse localization $\rightarrow$ fine search" paradigm. Specifically, GenSR first pretrains a dual-branch Conditional Variational Autoencoder (CVAE) to reparameterize symbolic equations into a generative latent space with symbolic continuity and local numerical smoothness. This space can be regarded as a well-structured "map" of the equation space, providing directional signals for search. At inference, the CVAE coarsely localizes the input data to promising regions in the latent space. Then, a modified CMA-ES refines the candidate region, leveraging smooth latent gradients. 
From a Bayesian perspective, GenSR reframes SR task as maximizing the conditional distribution $p({\rm Equ.}|{\rm Num.})$, with CVAE training achieving this objective through the Evidence Lower Bound (ELBO). This new perspective provides a theoretical guarantee for the effectiveness of GenSR. Extensive experiments show that GenSR jointly optimizes predictive accuracy, expression simplicity, and computational efficiency, while remaining robust under noise.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GenSR, a novel symbolic regression framework that transforms the discrete equation search space into a continuous generative latent space using a dual-branch conditional variational autoencoder. The method follows a "map construction → coarse localization → fine search" paradigm, where the CVAE learns a latent space with global symbolic continuity and local numerical smoothness. At inference, a modified CMA-ES algorithm refines candidate equations in this latent space. The authors provide a Bayesian interpretation and demonstrate strong empirical performance on SRBench datasets.

### Strengths
1. This paper presents a highly promising idea by formulating the symbolic regression problem as an optimization task in a continuous latent space. This approach transforms the search space from a structured space and allows continuous optimization algorithms such as CMA-ES to be applied directly. While prior work like SNIP has explored latent space optimization, constructing a smooth and meaningful latent space remains an open question because it is crucial for effective latent space optimization.
2. The authors propose a dual-branch conditional variational autoencoder to decouple and align symbolic and numerical information. This design is clearly motivated. Using KL divergence loss to align the prior and posterior distributions is a reasonable approach to promote local numerical smoothness.
3. The experimental evaluation is thorough. The authors test on the SRBench benchmark and compare GenSR with 18 baseline methods. In addition to main results, the paper reports noise robustness, latent space t-SNE visualizations, interpolation experiments, and a detailed ablation study. Figures 4 and 5 demonstrate that GenSR achieves better structural disentanglement and numerical continuity in the latent space than competing approaches.

### Weaknesses
1. The architectural choice of using a VAE appears somewhat outdated, as the overall model seems to be a slightly modified version of E2ESR, with an additional layer appended to the encoder’s final hidden state to map to ($\mu$, $\sigma$). A well-known issue with VAEs is posterior collapse, where the latent variables become uninformative and the decoder effectively ignores them (this problem can be even more pronounced with high-capacity models such as transformers). It would be beneficial to provide empirical evidence (KL divergence during training) to demonstrate that posterior collapse is indeed being avoided.
2. Although Section 3.3 provides an explanation for the desired symbolic continuity and numerical smoothness, its reasoning remains informal. For example, the paper assumes that the overlap of high-probability regions for similar equations enables smooth interpolation (lines 203–218). However, this is merely asserted with no empirical study of how frequently this occurs in practice. Similarly, the connection between minimizing the KL divergence and inducing local numerical smoothness (lines 219–234) is also asserted rather than formally established.

### Questions
1. The interpolation experiment in Appendix C.2 is vague. Linear interpolation in high-dimensional spaces can be problematic. For a VAE with a standard Gaussian prior, most of the probability mass is concentrated in a thin shell far from the origin. Linear interpolation between two points is highly likely to traverse low-probability regions of the latent space.  Could the authors clarify the specific setup for this experiment?
2. The paper reformulates the task as a Bayesian optimization problem. Have the authors considered applying Bayesian optimization algorithms directly in the latent space? I would be interested to see a comparison between CMA-ES and standard BO methods in terms of sample efficiency and the quality of the final solutions.

Please note that if the authors are able to address my questions during the rebuttal and discussion phase, I am willing to increase my score and confidence.

### Soundness
3

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
5

### Summary
The paper proposes GenSR, a Conditional-VAE-based symbolic regression model that encodes equations into a continuous latent space and performs CMA-ES optimization there. It claims improved performance over existing symbolic methods such as E2ESR and SNIP.

### Strengths
A key strength of the paper is the clear comparison with SNIP, which highlights the limitations of contrastive learning in SNIP and demonstrates the advantage of using a VAE framework for symbolic regression. Although using autoencoders for symbolic regression has been explored in the genetic programming literature [1], this paper presents a meaningful improvement by extending the autoencoder to a pre-training scenario and showing the benefits of a conditional VAE design.

[1]. Wittenberg, David, Franz Rothlauf, and Christian Gagné. "Denoising autoencoder genetic programming: strategies to control exploration and exploitation in search." Genetic Programming and Evolvable Machines 24.2 (2023): 17.

### Weaknesses
The main weakness is the lack of an ablation study to justify the necessity of the posterior symbolic encoder branch.

### Questions
1. The paper should clarify why encoding the symbolic expression is required. In a CVAE, the encoder is necessary because many outputs can correspond to the same condition, such as images. However, in symbolic regression, for a given dataset, there are usually only a few valid equations. Please consider adding an ablation where the symbolic encoder/posterior branch is removed, to test whether the bottleneck itself is the key contributor to performance.

2. Figure 5 illustrates numerical continuity, but the paper does not explain how this continuity emerges, since no explicit loss term enforces it. Please provide an explanation of this phenomenon. While it is expected that a VAE can cluster similar functions, the observed continuity is an interesting behavior that deserves discussion.

3. On page 21, one citation appears incorrect and is displayed as a question mark. Please correct this.

4. The model weights and source code should be released for reproducibility.

5. Some methods listed in Table 4 are not reported in Table 3. Please include their performance on the black-box dataset for completeness.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper points out that "similar structure ≠ similar value" in discrete symbol space leads to noisy fitting error signal and lack of search direction, which is the source of inefficiency of most SR methods. Therefore, GenSR is proposed. The pre-trained dual-branch CVAE reparameterizes the equation into a generative latent space with "global symbolic continuity + local numerical smoothness", according to the "map construction → rough positioning → fine search" paradigm and improves CMA-ES search. SR is reformulated as a Bayesian problem and implemented by ELBO training.

### Strengths
1. This method seems to achieve very good experimental results
2. This paper experiment is relatively complete, considerable workload

3. In addition, the authors give a Bayesian perspective and reformulate SR as a probabilistic inference problem of maximizing P(F|X) trained by ELBO, which provides a clear theoretical framework and interpretability support for the rationality of the method.

### Weaknesses
The article is well written, but it seems to be only transferring the method in the multi-modal of graphics and text to the field of symbolic regression, and the innovation of the algorithm is not very high. Ask the author to emphasize his technical innovation

### Questions
1. What are the advantages of methods trained via CVAE over methods trained via contrastive learning? 
2. What's the reason? It seems that contrastive learning can achieve similar results to the map building mentioned in this article, and I think it might even be better.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper presents GenSR, a generative SR method that trains a conditional generative model that, at test time (inference/runtime), starts from a good latent-space initialization and refines it with CMA-ES to produce a set of good-fitting equations. The paper claims to introduce a new search paradigm, a Bayesian perspective for SR, and experiments that demonstrate state-of-the-art performance.

### Strengths
* The paper is straightforward to read and follow.
* SR is an essential problem for the scientific ML community.
* The interpretability and comparison plots were informative, and a nice visual comparison.

### Weaknesses
* Core contributions are not novel, and the authors of GenSR have significant overlap with existing related work that was not discussed at all [1]. [1] proposes a generative SR method using a conditional VAE (CVAE), which encodes expressions into a smooth low-dimensional space, and then applies an evolutionary search algorithm to this latent space, to efficiently sample equations, and at the time of its publication claimed state-of-the-art. Given that the contributions of GenSR are exactly these, I find this missing key related work in direct conflict with this submission. I encourage the authors to: 1) clarify how their method differs, and 2) clearly delineate what is novel and new in their paper in the context of all existing VAE SR methods [1,2] and existing generative methods [3].
  * Using a VAE for SR is not novel and has been extensively done in prior work [1,2], which is conveniently not discussed in this paper. Furthermore, the other missing key related work [2] already shows that using a VAE can improve equation generation under noisy settings, a result also reported in this paper.
* ``GenSR provides a new Bayesian perspective for SR''; this contribution is lacking without appropriate citation and delineation to existing generative SR methods that also provide a Bayesian perspective for SR, such as [3].
* Claim of "State-of-the-art performance" is not validated, as SRBench and corresponding results are old and outdated, dating from 2021/2022. I would encourage the authors to implement the most recent SOTA SR methods and compare them against those to make such a bold claim. Furthermore, the results appear to be cherry-picked, as SR Bench has many problem sets, and only the subset of test equations (Feynman) results are presented graphically in the main figure; whereas other newer SOTA SR methods on SRBench show the same figure across all the equations in the benchmark [5]. I encourage the authors to directly compare to [5] and more recent SR methods to make any claim of state-of-the-art. Furthermore, I encourage the authors to compare against standard problem sets such as Nguyen, Livermore, and similar problem sets as done in [6]. Moreover, the authors omit from the SRBench results the results for "Symbolic Solution (%)" and "Accuracy Solution (%)", leading to suspicion over the results, as these are computed as standard when running the benchmark, and choose to only present R^2 instead. I encourage the authors to be transparent and report all standard results from the benchmark, especially if making any state-of-the-art claims.
* Appendix B seems not necessary, or novel, and appears to be a standard re-statement of the classic textbook ELBO derivation. I would encourage the authors to simply cite this, without the proof in the appendix, as it is not new.

References:
* [1] Mežnar, Sebastian, Sašo Džeroski, and Ljupčo Todorovski. "Efficient generator of mathematical expressions for symbolic regression." Machine Learning 112.11 (2023): 4563-4596.
* [2] Popov, Sergei, et al. "Symbolic expression generation via variational auto-encoder." PeerJ Computer Science 9 (2023): e1241.
* [3] Holt, Samuel, Zhaozhi Qian, and Mihaela van der Schaar. "Deep Generative Symbolic Regression." The Eleventh International Conference on Learning Representations.
* [4] Kamienny, Pierre-Alexandre, et al. "End-to-end symbolic regression with transformers." Advances in Neural Information Processing Systems 35 (2022): 10269-10281.
* [5] Landajuela, Mikel, et al. "A unified framework for deep symbolic regression." Advances in Neural Information Processing Systems 35 (2022): 33985-33998.
* [6] Mundhenk, T. Nathan, et al. "Symbolic regression via neural-guided genetic programming population seeding." arXiv preprint arXiv:2111.00053 (2021).

### Questions
* Can the authors clearly delineate how their work is novel, new, and different, compared to the existing highly similar papers such as [1,2,3] stated above, and the broader aspect of VAEs for SR, and similarly other generative models for SR [3]. As it stands with the current submission, this paper does not propose anything novel in context to the existing literature [1,2]; and has severe experimental issues.
* Given that the authors refine constants using BFGS as an inner optimization, have the authors considered the functions getting stuck in local minima and strategies to prevent this, as routinely done in other SOTA SR methods [4]?
* Can you compare experimentally against [4,6,5] by running these competing methods against your method?

### Soundness
2

### Presentation
2

### Contribution
1
