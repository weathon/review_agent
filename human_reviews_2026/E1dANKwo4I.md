# Robust Amortized Bayesian Inference with Self-Consistency Losses on Unlabeled Data

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
Amortized Bayesian inference (ABI) with neural networks can solve probabilistic inverse problems orders of magnitude faster than classical methods. However, ABI is not yet sufficiently robust for widespread and safe application. When performing inference on observations outside the scope of the simulated training data, posterior approximations are likely to become highly biased, which cannot be corrected by additional simulations due to the bad pre-asymptotic behavior of current neural posterior estimators. In this paper, we propose a semi-supervised approach that enables training not only on labeled simulated data generated from the model, but also on unlabeled data originating from any source, including real data. To achieve this, we leverage Bayesian self-consistency properties that can be transformed into strictly proper losses that do not require knowledge of ground-truth parameters. We test our approach on several real-world case studies, including applications to high-dimensional time-series and image data. Our results show that semi-supervised learning with unlabeled data drastically improves the robustness of ABI in the out-of-simulation regime. Notably, inference remains accurate even when evaluated on observations far away from the labeled and unlabeled data seen during training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a new loss term that uses unlabeled data to improve the robustness of amortized neural posterior estimation methods. They thoroughly motivate and explain their loss term and prove that it is a proper loss (in the relevant context of SBI) and present promising results for their method on a simple Gaussian mixture scenario, as well as three more challenging setups (Air traffic data, Hodgkin-Huxley model, MNIST denoising).

### Strengths
The core idea of the paper is neat. I also appreciate that it's a simple idea!

Furthermore, the approached problem of robustness is highly relevant in SBI (and beyond). It also seems like the availability of unlabelled real-world data, that is necessary for this approach, is a very reasonable assumption that is often fulfilled in practice. I see this as a major strength of this approach compared to other recent methods. 

The overall presentation is good and the structure of the paper very clear and well-written. 

Related work is comprehensively discussed. 

The idea is original, but some parts of the theory are already present in other papers, albeit for a different purpose.

### Weaknesses
Contribution: 

I have one major concern with the paper: You do assume for the majority of your theory and experiments that the likelihood under the simulator is known. Doesn't this defeat the purpose of SBI to some degree? If the likelihood is know, why not use a method that directly leverages this information?

I am aware that the case of unknown likelihood + unknown posterior is briefly discussed, but more theoretical insight would be very helpful. 

Analogously for the experiments. I believe that the case of unknown likelihood + unknown posterior is the actually relevant one, but it is not really evaluated or thoroughly discussed. 

I am also unsure about the benefit of the method if used in combination with post-hoc methods such as SNIS; I believe that many people practically interested in the method might want to use post-hoc methods in combination with this approach.

Presentation: 

Please note that those two points are not crucial: 

Some parts of the methodology seem a bit repetitive and do not really present very interesting insights. For instance Proposition 4 seems a bit too redundant for this much space. I am wondering if it would help the paper to shorten this section and, for instance, include a brief introduction to amortized SBI in the beginning to help readers less familiar with this field. 

Similarly for the description of the multivariate Normal case-study: I feel like it takes up a bit too much space for the practical relevance this case-study has. Instead more results/figures for the other case-studies could be presented in the main body. 

Nit: There is a typo around line 392: "enablung"

Experiments: 

Please also report metrics for the denoising MNIST case-study. 

Seeing C2ST scores for all experiments would also be nice. 

While I believe that the experiments overall are similar in scope to what is usually done in SBI, but very small compared to other fields of ML. I would be interested in seeing larger-scale experiments (higher dimensionalities, larger neural networks, more extensive training); but this is not a critical point.

### Questions
I would be happy to see your response regarding my concern about the contribution of the paper. 

Why should I use SBI, and this method in particular, when I have access to the likelihood under the simulator? Why not directly leverage the likelihood and rely on samples instead?

Are there any further (theoretical) insights into the case where both the likelihood and posterior are unknown? 

I am looking forward to your response!

I would be very happy to increase my score if you further justify the applicability of using NPE plus your robustness term when the likelihood is known (i.e. in what case would this be a relevant setup?) OR provide more evidence that using your method with an estimated likelihood is theoretically sound and works well experimentally.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a semi-supervised approach to amortized Bayesian inference using self-consistency losses on unlabeled data, demonstrating improved robustness in out-of-distribution settings. While the theoretical contributions are solid, key claims about high-dimensionality and robustness are not adequately supported by the experimental evidence.

### Strengths
Improving robustness of SBI to data distribution shifts in an important problem.

### Weaknesses
- The abstract uses "bad pre-asymptotic behavior" without definition, making the core problem inaccessible.

- When would practitioners have unlabeled real data but not be able to generate more simulations?

- The paper claims "high-dimensional" capability but experiments max out at 100 parameters with significant performance degradation (Figure 2a shows MMD increasing substantially). The MNIST example (784D) is modest by modern standards. Please scale back claims.

- Head-to-head comparisons on all case studies are essential. Also missing: ablations on the choice of p_C(theta), sensitivity to lambda schedule, and performance with different amounts of unlabeled data.

- The paper claims to maintain ABI's speed but provides zero timing comparisons, memory usage, or analysis of overhead from computing SC loss and sampling q_t(theta|x) during training.

- Propositions 1-4 establish strict properness under idealized conditions (universal approximators, infinite data), but there's no finite-sample analysis or characterization of approximation error with realistic network capacities. The gap between asymptotic guarantees and finite-sample behavior needs some comments.

- The MNIST prior misspecification (double-blur vs single-blur) is artificial. How does the method perform under realistic misspecification where the error structure is unknown during training?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose to include a self-consistency loss regularizer to the typical neural posterior estimation (NPE) class of objectives for amortized inference (more generally, a loss with respect to any score). The regularizer, unlike the main objective, is computed over real data, not simulated draws from the model. The regularizer enforces self-consistency (Eq. 1) on the observed dataset, draws from which are assumed to be the target for inference. The motivation provided by the authors is that this results in more accurate inference, especially in distribution-shift cases where the real observations are substantively different from the simulated data with which $q$ is fit.

### Strengths
- The writing and proposed methodology are clear and easy to understand
- The proposed regularizer is intuitive: as the true posterior demonstrates  self-consistency property, it’s a natural extension for the variational posterior to satisfy this condition (approximately)
- The problem setting is a significant one, as distribution shift in simulation-based inference (SBI) problems continues to be a robust area of research.
- The others take care to formalize their result more carefully with respect to strictly proper losses, proving that the regularizer does not alter the problem's solution.

### Weaknesses
- The method is combinatorial: the NPE (score-based) objective for SBI is well-established, and Schmitt et al. (2024) introduced the self-consistency loss.
- I don’t find the degree of novelty to be adequate enough to differentiate the work from Schmitt et al. In the case where the simulation model is correct, the proposed method is exactly identical to Schmitt et al. Thus, the authors’ main contribution, to me, seems to be applying the method to the setting of a misspecified simulator. While this is undoubtedly an important setting as acknowledged above, the novelty of applying the method to a new setting is fairly limited
- Experimental evaluation is insufficient. To me, comparing only to NPE isn’t a fair comparison. The experiments involve deliberately mispecifying the model, so no one would expect NPE to perform well. At a minimum, the authors should compare to a VAE trained with the incorrect prior/likelihood (but with the real dataset), as well as the method of Schmitt et al., which imposes self-consistency on the simulated data. Beyond these, comparisons to some of the many domain adaptation approaches to SBI from the literature would be in order.
- I find the literature review of relevant related work to be lacking. For example, the Autoencoding Variational Autoencoder (Cemgil et al.) has a similar notion of self-consistency applied to real data, but in the VAE setting. This line of work would be a promising area for future ablation studies that I think would improve the paper.

### Questions
- What precisely is meant by “sufficient statistics” in line 158? Sufficient in the formal statistical sense (if so, can you elaborate?), or do you mean more informally low-dimensional learned representations of $x$ that are known to have low reconstruction error, or something like that?
- The experiments appear to utilize a fixed simulation budget (e.g., 1024, line 723). Although this is the setting explored in Schmitt et al., my understanding of “standard” NPE typically allows unlimited samples from the model, e.g., fresh samples are drawn for each minibatch used to compute each gradient. I think the authors should also vary the simulation budget in their case study, as not doing so unfairly penalizes NPE (besides have the wrong model, it also has only finitely many samples from this model, and so will overfit).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces self-consistency loss proposed to improve training efficiency into amortized Bayesian inference (ABI) to enhance its robustness. In addition to the loss based on labeled simulation data, it includes the self-consistency loss defined with unlabeled real data. The later adds extra information to improve the robustness of ABI.

### Strengths
The independence of true parameter values is the most striking feature of the proposed method. The method is justified by theories. Various numerical evidences including through simulation and real-data applications are strong and convincing.

### Weaknesses
The conditions of propositions 2 and 3 are not clear to me (See question below). Figure 4 (b) is a little bit misleading. Overall, these are minor defects.

### Questions
1. Does the condition of proposition 2 always hold? Does proposition 3 depend on on proposition 2? If so, the statement of proposition 3 reads like it holds regardless. Some clarification will be appreciated.

2. Figure 4 -(b) is misleading. I understand that it represents the difference between MAB (NPE) and MAB (NPE+SC). However it looks that MAB (NPE) is 0 while MAB (NPE+SC) is bigger and concentrated around 5. Why not plotting the histograms of each using different colors?

### Soundness
3

### Presentation
3

### Contribution
3
