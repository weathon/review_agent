# From Predictors to Samplers via the Training Trajectory

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Sampling from trained predictors is fundamental for interpretability and as a compute-light alternative to diffusion models, but local samplers struggle on the rugged, high-frequency functions such models learn. We observe that standard neural‑network training implicitly produces a coarse‑to‑fine sequence of models. Early checkpoints suppress high‑degree/ high‑frequency components (Boolean monomials; spherical harmonics under NTK), while later checkpoints restore detail. We exploit this by running a simple annealed sampler across the training trajectory, using early checkpoints for high‑mobility proposals and later ones for refinement. In the Boolean domain, this can turn the exponential bottleneck arising from rugged landscapes or needle gadgets into a near-linear one. In the continuous domain, under the NTK regime, this corresponds to smoothing under the NTK kernel. Requiring no additional compute, our method shows strong empirical gains across a variety of synthetic and real-world tasks, including  constrained sampling tasks that diffusion models are unable to handle.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
### Summary

This paper leverages a well-celebrated theoretical idea, i.e. the coarse to fine, spectral learning dynamics of gradient based learning to help create a series of landscape with different smoothness. Then use these landscapes as annealed landscapes to help sampling from the final complex energy landscape. They conducted theory on simple boolean settings showing the learning order effect in the degree of polynomial, and then validated the idea in various discrete and continuous sampling set up and showed significant improvement upon simply temperature annealed sampling and other MC methods.

### Strengths
### Strength

- The paper is tackling an interesting problem with a very creative solution, connecting ideas from spectral bias, training dynamics and energy based model and sampling. I’m very convinced of the idea.
- The authors noted significant agreement with the theory in FCNN and MLP setting, and noted the deviation in transformer setting. We commend their honesty about the limitation of the theory.
- The experimental testing of the idea is very comprehensive and convincing, showing univocal benefit of this idea.

### Weaknesses
### Weakness

- Often it’s not clear which training time check point the authors used in sampling experiments.
    - e.g. in 4.1.2, it’s not clear from writing which training time checkpoint was used and what 1K and 10K steps denote. is it that 10K steps checkpoint is better intermediate landscape than 1K step?
    - More generally, I feel there is a lot of heuristics and design space for which checkpoint(s) are best suit for these intermediate landscapes, and how do the authors decide on them?

### Questions
- C.f. Sec. 3.2, the theory / motivation of the paper also aligns closely with the learning dynamics of score-based diffusion models, i.e. learned score vector fields are simpler, smooth earlier in the training, usually better approximated by a linear vector field. There is a nice spectral ordering for the learning of vector field and distribution. [^1], [^2]

[^1] Wang, & Vastola, (2024). The unreasonable effectiveness of gaussian score approximation for diffusion models and its applications. TMLR

[^2] Wang, & Pehlevan (2025). An analytical theory of power law spectral bias in the learning dynamics of diffusion models. NeurIPS

- For the idea of sampling leveraging a sequence of landscapes from smooth to rugged to help find tricky spiky solutions, the authors could also mention this recent work [^3], which shares a very similar picture, but did not use the learning dynamics to help. I feel [^3] could use many intuition / results from this paper to help learn their sequence of landscapes.

[^3] Du, Y., Mao, J., & Tenenbaum, J. B. (2024). Learning iterative reasoning through energy diffusion. 

- In the continuous sampling experiments, is it relevant to compare to or report reference values for evolutionary algorithms? since these are also common functions used to benchmark those, e.g. CMAES.

- Discussing how to best heuristically pick the intermediate checkpoint is quite interesting and useful for the reader of the paper.

Minor

- The Table 2 format is a bit confusing…. 
the right column (step count) should not be compared on the same table with the mid and left column (success rate). A separatrix should be added or different table should be used to present the median step result. 
From the first glance, it’s confusing to see 4.00 and 2.00 in the right column…. and median step is an integer so why do we have 2 decimal points here

- Table 5 and Sec. 4.1.2 are not very clear, which value is the temperature annealed GWG? why does it have two values as authors say they used the final checkpoint.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a method called trajectory annealing, which uses intermediate training checkpoints of a predictor to guide sampling. The main idea is that during training, neural networks naturally evolve from coarse to fine representations, early checkpoints smooth high-frequency variations, while later ones add detail. By running MCMC samplers (GWG for discrete and MALA for continuous variables) sequentially across these checkpoints, the method achieves better mixing and sampling efficiency, especially in rugged or synergistic landscapes. Experiments on Boolean, MNIST-EBM, DNA design, and materials datasets show strong empirical gains over standard temperature annealing. The theory part connects this to SGD’s hierarchical learning of low- to high-degree monomials.

### Strengths
The method is conceptually simple yet powerful, requires no retraining or architectural change, and leverages an inherent property of neural network learning. The connection to hierarchical degree learning and NTK smoothing is novel. Results are extensive and consistent across very different domains. The Boolean case study is particularly compelling, showing exponential-to-linear mixing improvements.

### Weaknesses
Theoretical results rely on strong assumptions (e.g., degree-wise alignment checkpoints) that may not generalize. Experiments are numerous but some seem cherry-picked to highlight advantages. Limited comparison with recent diffusion-based or amortized samplers. Some derivations could be more formal, and the transition from discrete to continuous domains feels hand-wavy. Also, claims of "no extra compute" ignore checkpoint storage and evaluation overhead.

### Questions
How sensitive is the performance to checkpoint spacing or number of steps per checkpoint? Could this be integrated with modern optimizer schedules or adaptive checkpoint selection? How does it perform on large-scale, non-convex tasks like ImageNet classifiers? Is there any insight on when the coarse-to-fine property breaks down, e.g., transformers?

### Soundness
3

### Presentation
3

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
The paper proposes trajectory annealing: instead of sampling only from the final trained predictor $f^*$, the sampler runs short MCMC updates across saved training checkpoints $\{f_t\}$, exploiting the observed coarse‑to‑fine learning dynamics (early checkpoints damp high‑degree/high‑frequency components, later checkpoints restore detail).

### Strengths
- Leveraging the existing training trajectory as an annealing schedule is elegant and creative and requires no re‑training or auxiliary generative model. The coarse to fine picture is illustrated empirically and theoretically (Apps A-B).
- The method works for discrete (GWG) and continuous (SMC) domains. 
- The Hamming‑ball constraint is handled naturally within the MCMC framework and yields large gains on DNA (Table 7).
- The $O(d\log d)$ mixing is theoretically supported on degree-1/2 surrogates.

### Weaknesses
- Compute parity (App J) is defined as matching the number of MALA steps, but SMC adds resampling/weight computations and multi‑checkpoint bookkeeping. MNIST also saves hundreds of checkpoints (50k training epochs saving every 100 steps). Reporting the wall‑clock time and memory would substantiate the “no additional compute” claim from the abstract.
- For the discrete synthetic tasks the only baseline is GWG (authors state this choice explicitly), leaving out stronger informed/non‑local samplers (e.g., locally balanced and discrete‑Langevin families). This narrows the comparison.
- The method does not apply to transformers (stated in the paper), which constrains scope for many modern applications.
- Several evaluations measure best‑of‑run (DNA keeps the best of 60 steps per run), which is an optimization metric. Mixing diagnostic or distributional metrics would be helpful.

### Questions
- With only a few steps at the final checkpoint, how biased are samples relative to the target distribution?
- How do performance and compute vary with the number and placement of checkpoints (uniform vs geometric; early‑only vs full trajectory)? MNIST uses 500 checkpoints, but discrete synthetic uses just a few.
- Have you tried saving checkpoints based on a performance metric instead of time? 
- The paper states diffusion “cannot” handle Hamming‑ball constraints (p. 8). Could you qualify this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to use intermediate checkpoints from training a predictor model (used to define an energy function) to anneal MCMC from easy to sample distributions to the target distribution, which may have pathological structure that makes it difficult to directly sample from. They provide hierarchical learning analysis to validate the intuition, and provide some experiments to verify that the method works.

### Strengths
The idea of using the training checkpoints to perform some kind of annealing is novel to the best of my knowledge. I agree with their argument on how early stage checkpoints focus on high level features, late checkpoints focus on details, providing a sampling path. 

The use of confidence intervals demonstrates thorough evaluation metrics. 

The hierarchical learning analysis is also quite interesting.

### Weaknesses
**Literature Review**: The paper does not mention [2] or [3], which are focused on accessing modes that are difficult to reach. [3] is more recent so the omission is understandable. 

**Base sampler**: Gibbs with gradient changes one coordinate at a time (at least in the default version), which makes it very slow. Are the benefits of annealing via model checkpoints preserved when using samplers proposed in [1, 2, 3]? Or does performance improvement from using this annealing strategy saturate with stronger base samplers? While GWG was chosen due to simplicity, it is important to note that more recent methods may solve this problem without the need for annealing across checkpoints. 

**Breadth of Metrics**: It would be nice to include metrics that show sampling accuracy v.s number of sampling steps. For the MNIST experiment, it could take the form of log maximum mean divergence v.s sampling steps. Also, the paper does not include (Effective Sample Size), which is an important metric for evaluating the efficiency [1]. 

**Characterization of Diffusion**: They characterize diffusion as requiring training over the entire trajectory instead of single step MLE. However, I am not sure that this is a fair characterization: in practice, the score matching objective is trajectory free [4]. Even in discrete diffusion, the training objective takes the form of corrupting the input and then predicting the clean input [5, 6, 7, 8]. For obtaining SOTA results on CIFAR-10, it might take up to 15 hours. But it is entirely possible that within the first 15 minutes, the model is capable of generating reasonable images. 

This is important because the intuition behind the proposed method is extremely similar to the core idea of diffusion: start from a distribution that is easy to sample, and gradually anneal it to the target distribution. The difference is that diffusion enables this with a single checkpoint, whereas the proposed method requires several checkpoints. Furthermore, diffusion directly supervises learning of the score (which is what GWG requires). 

This is perhaps my largest concern with the submission: it seems to be capturing the intuition of diffusion, but via a more indirect path. If the focus of this paper is small models (as discussed in the introduction), is training a diffusion model really that expensive? And if the focus is on larger models where diffusion training is expensive, then it would also be expensive to store multiple checkpoints of the model and run analysis across all the checkpoints to determine which ones to use for annealing the sampler. 

Also, I do not see a reference to [9], which directly incorporates the logic of diffusion into MCMC to improve mixing via the same intuition presented in this submission. While [9] is focused on the continuous domain, it may be worth considering how to extrapolate their method to the discrete space via gradient based discrete samplers. 

[1] A Langevin-like Sampler for Discrete Distributions. Zhang et al. ICML 2022. 

[2] Gradient-based Discrete Sampling with Automatic Cyclical Scheduling. Pynadath et al. NeurIPS 2024. 

[3] Reheated Gradient-based Discrete Sampling for Combinatorial Optimization. Li, Zhang. TMLR 2025. 

[4] Elucidating the Design Space of Diffusion-Based Generative Models. Karras et al. NeurIPS 2022. 

[5] Simplified and Generalized Masked Diffusion for Discrete Data. Shi et al. NeurIPS 2024. 

[6] Simple and Effective Masked Diffusion Language Models. Sahoo et al. NeurIPS 2024. 

[7] Simple Guidance Mechanisms for Discrete Diffusion Models. Schiff et al. Preprint 2024. 

[8] The Diffusion Duality. Sahoo et al. ICML 2025. 

[9] Diffusive Gibbs Sampling. Chen et al. ICML 2024.

### Questions
- How does this method perform when using DLP sampler? Are the gains of using checkpoints / using just the final checkpoint just as large? 
- In the introduction, it is stated that it takes 15 hours to train a diffusion model on CIFAR-10. Is there a citation for this? Are there plots of the loss v.s training time of diffusion methods for the tasks considered?

### Soundness
2

### Presentation
3

### Contribution
1
