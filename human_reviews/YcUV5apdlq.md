# Multilevel Generative Samplers for Investigating Critical Phenomena

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Investigating critical phenomena or phase transitions is of high interest in physics and chemistry, for which Monte Carlo (MC) simulations, a crucial tool for numerically analyzing macroscopic properties of given systems, are often hindered by an emerging divergence of correlation length---known as scale invariance at criticality (SIC) in the renormalization group theory.  SIC causes the system to behave the same at any length scale, from which many existing sampling methods suffer: long-range correlations cause critical slowing down in Markov chain Monte Carlo (MCMC), and require intractably large receptive fields for generative samplers. In this paper, we propose a Renormalization-informed Generative Critical Sampler (RiGCS)---a novel sampler specialized for near-critical systems, where SIC is leveraged as an advantage rather than a nuisance. Specifically, RiGCS builds on MultiLevel Monte Carlo (MLMC) with Heat Bath (HB) algorithms, which perform ancestral sampling from low-resolution to high-resolution lattice configurations with site wise-independent conditional HB sampling.  Although MLMC-HB is highly efficient under exact SIC, it suffers from a low acceptance rate under slight SIC violation. Notably, SIC violation always occurs in finite-size systems, and may induce long-range and higher-order interactions in the renormalized distributions, which are not considered by independent HB samplers.  RiGCS enhances MLMC-HB by replacing a part of the conditional HB sampler with generative models that capture those residual interactions and improve the sampling efficiency. Our experiments show that the effective sample size of RiGCS is a few orders of magnitude higher than state-of-the-art generative model baselines in sampling configurations for $128 \times 128$ two-dimensional Ising systems.  SIC also allows us to adopt a specialized sequential training protocol with model transfer, which significantly accelerates training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the Renormalization-informed Generative Critical Sampler (RiGCS), a novel generative sampling algorithm designed for near-critical systems where scale-invariant criticality (SIC) can hinder the efficiency of standard Monte Carlo simulations. The proposed RiGCS builds on Multilevel Monte Carlo algorithms with Heat Bath sampling and incorporates generative models to address the shortcomings of HB under slight SIC violations.

### Strengths
1. The proposed method integrates renormalization group theory with generative models in a novel attempt to address critical slowing down in Monte Carlo simulations.
2. Employing generative models to enhance MLMC-HB sampling seems to be a promising approach with potential applications in statistical physics and lattice systems.

### Weaknesses
1. The core contributions (lines 90–99) are not sufficiently clear. See Questions 1-3.
2. The mathematical formulations are not clear. For example, Equation (9) is not well defined and difficult to interpret. The algorithm design between lines 355–369 is unclear, and the lack of theoretical guarantees on convergence or sample complexity weakens the technical rigor.
3. The experiments are limited in scope. To demonstrate generalizability, the authors should evaluate RiGCS on more complex benchmarks beyond the 2D Ising model.
4. The appendix introduces unrelated terminologies (e.g., Swendsen-Wang algorithm, Wolff algorithm) without meaningful connections to the core work or further experimental analysis. Moreover, the lack of theoretical support or detailed experimental setups in the appendix limits its usefulness.
5. The manuscript cites more than 100 references, many of which seem only loosely related to the topic.

### Questions
1. Why is addressing SIC important in this context? While the authors claim that SIC reduces sampling efficiency, they do not clearly explain how RiGCS specifically addresses this issue. A more detailed justification is required.
2. Although the warm start is listed as a key contribution, the implementation details are unclear. How is the warm start applied across different resolution levels, and why is it considered a significant innovation?
3. If the authors only employ standard metrics (e.g., effective sample size, free energy bias), it is unclear why performance evaluation is listed as a major contribution. Were any new metrics introduced? If not, the framing of performance evaluation as a contribution should be reconsidered.

### Soundness
3

### Presentation
2

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
In this paper, the authors present RiGCS, a method to sample from Boltzmann distributions in systems with critical phenomena such as phase transitions. This is challenging due to the problem of scale invariance at criticality, in which the correlation length between sites on a lattice diverges. This leads to a high rejection rate in standard Markov Chain Monte Carlo (MCMC) algorithms which mainly focus on local correlations to propose new configurations. The authors also state that this poses a challenge for generative models due to the need for large receptive fields. The proposed approach breaks down the system lattice into a hierarchy of resolutions, essentially factoring the probability distribution at the finest level into a product of distributions each conditioned on a coarser level. Each distribution is parameterized by autoregressive neural networks (PixelCNN) which are learned sequentially, and whose parameters are used to warm-start the next level in the hierarchy. The authors demonstrate that their method is able to generate samples which estimate free energies more accurately than the baselines, as well as have a higher effective sample size due to lower correlation between samples.

### Strengths
The paper tackles an important and challenging problem in physics (critical phenomena/phase transitions), and one that has received comparitively less attention from the AI for Science community, which is largely still focused on equilibrium phenomena. Thus, I think the topic/paper would be of interest to many in the community. Additionally, the proposed approach (hierarchical sampling at multiple resolutions) is interesting and solidly grounded in well-known techniques from scientific computing such as multigrid algorithms.

### Weaknesses
My primary concerns center around the "receptive field" argument, and the quantity and quality of the experimental results. I will elaborate further. 

Receptive field:

There is a large and growing body of literature on long-context models in the fields of language and computer vision [1, 2, 3, 4, 5]. The goal of these works is to provide models with more global reasoning capabilities over, e.g. extremely long textual or video contexts, and/or to reason over extremely high-resolution images, while remaining memory and compute-efficient. The central problem tackled by this paper - diverging correlation lengths in critical systems require large receptive fields - is related to this line of work. The proposed solution - a hierarchical decomposition of the lattice - is also related to the scale hierarchies proposed in these works.

My first criticism is that the authors do not mention these works or position their contribution in the backdrop of this existing work. Can the authors comment on how their problem setting is substantively different from what is tackled in these existing works, and/or why the approaches from this line of work cannot be applied "zero-shot" to this application domain? My next and primary criticism is that given the current state-of-the-art in long context generative modeling, the proposed method is behind the curve, and the size of systems considered in this work (at most $128^2$) is not a sufficiently challenging setting to warrant the proposed approach. First of all, the authors use convolutional neural networks, which have the inherent limitation of a local receptive field. Can the authors justify this choice? Transformers have a global receptive field at the expense of an increased computational/memory footprint, but for 128x128 systems this shouldn't pose an issue. As far as I could tell, there was no mention of the Transformer architecture in the paper. Even if you consider much larger lattices (~$100^4$) used in e.g. lattice gauge theories and quantum chromodynamics, this is still comparable to the number of pixels in, e.g. satellite imagery data, for which solutions to reason over long context have been proposed [5]. I understand that there is not a one-to-one mapping between modeling critical phenomena and long context image/video models, but I think this disparity is telling and worth pointing out. 

The choice to use an autoregressive model (PixelCNN) for sampling individual states also seems dubious to me. There is nothing inherently causal about the lattice sites, and so it seems it would make more sense to employ a non-autoregressive generative paradigm, such as diffusion. By parameterizing a diffusion model with e.g. a Transformer with bidirectional/unmasked attention, a global receptive field is naturally obtained at every stage of the generation process. I understand the binary nature of spin systems poses a slight hurdle, but one that is seemingly surmountable given the growing body of work on discrete diffusion [6] (not to mention more naive solutions like quantizing continuous predictions, clipping, binning, etc.). Plus, the authors mention that their approach is valid for continuous systems, so this seems like a natural choice (non autoregressive approaches are the method of choice for continuous data modalities).

Quantity/quality of experimental results:

Given my previous comments, I feel that the amount/nature of the experimental validation of the proposed method is too narrow. The evaluation is restricted to 2D Ising systems with size at most $128^2$. Can the authors include results on larger systems where, e.g. using a global receptive field is actually out of the question due to the memory constraints?

Additionally, the improvements in free energy estimation and EMDM shown in Figure 2 and 3 appear to be very marginal. The cluster method baseline significantly outperforms RiGCS on effective sample size - the authors claim that it is not applicable to continuous settings, but they do not demonstrate results in these settings either.

In summary, in order to reconsider my score, I would like to see considerable evidence that the proposed approach is superior to the many existing ways to achieve global receptive fields in generative models, including empirically on larger and continuous systems.
 
[1] Child, Rewon, et al. "Generating long sequences with sparse transformers." arXiv preprint arXiv:1904.10509 (2019).

[2] Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[3] Dao, Tri, et al. "Flashattention: Fast and memory-efficient exact attention with io-awareness." Advances in Neural Information Processing Systems 35 (2022): 16344-16359.

[4] Liu, Hao, Matei Zaharia, and Pieter Abbeel. "Ring attention with blockwise transformers for near-infinite context." arXiv preprint arXiv:2310.01889 (2023).

[5] Gupta, Ritwik, et al. "xT: Nested Tokenization for Larger Context in Large Images." arXiv preprint arXiv:2403.01915 (2024).

[6] Austin, Jacob, et al. "Structured denoising diffusion models in discrete state-spaces." Advances in Neural Information Processing Systems 34 (2021): 17981-17993.

### Questions
1. Presenting the proposed method as an alternative to MCMC sampling feels confusing to me. Doesn't the proposed method generate i.i.d samples from the Boltzmann distribution? As opposed to MCMC approaches which produce correlated samples via a Markov Chain? Unless I am misunderstanding something, I think it would help to clarify this point.

2. How is the training data obtained for optimizing the KL divergence loss? Is it from long MCMC simulations? I think this is a critical point that was not focused upon enough in the paper, unless I missed something.

### Soundness
2

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
4

### Summary
This paper deals with the problem of sampling systems at criticality, i.e. systems typically displaying some form of self-similarity w.r.t change of scale, and estimating their associated thermodynamical quantities like the free energy. This is a notoriously difficult problem of interest in physics, because it requires to deal with arbitrary long range correlations and for which standard MC samplers based on local moves undergo-critical slowing down, meaning that the auto-correlation time become arbitrarily large with system size. As explained in the introduction, there exists some specific algorithms in the physics literature, like cluster methods specifically designed for lattice models,
typically working for discrete or continuous variables (Ising, Potts, O(n) ...). In this work, the authors propose to leverage machine learning methods to generalized some existing multi-scale methods following closely the spatial RG procedures, which basically exploits statistical dependencies between coarse grained variables at different scales. These use rather drastic approximations on the conditional models between coarse grained variables at different scales, leading sometime to very low acceptance rate, and ML methods offer the possibility to used more complex conditional models between scales.

### Strengths
the paper is well written and rather clear with an impressive bibliography on the subject which is indeed quite vast and old. A synthetic and pedagogical section in the supplementary material which should be useful to non-physicists is devoted to the RG theory, and should make the whole paper readable for non-physicists. The proposed algorithm is quite generic and and is competitive with the Wolff algorithm on 2d lattices up to L=128, where it seems to actually yield better estimates of the free energy. and shows clear advantage w.r.t MLMC-HB (used in it orginal version maybe outdated, as it use the original  Kadanov block-spin  transformation, subsequent refined algorithms cited in the text are not considered for  baselines) and other baselines considered like HAN which uses a similar idea, in the form of an auto-regressive
NN based on a hierarchical representation of the lattice.

### Weaknesses
I found some aspects of the experimental not completely satisfactory. The claim that RiGCS performs better than Wolff is a strong claim that should be better sustained in my opinion. Authors concentrate in the experiments on the computation of the free energy, but characterization of physical properties involve many other possible indicators, which the  Wolff algorithm is known to directly compute very efficiently,
like the magnetization, susceptibility, Binder parameters which could be easily displayed without further work it seems ...
some exponents could also be extracted by finite size scaling and compared to exact ones. The combination of Wolff with AIS even though reasonable seems a bit like of an ad-hoc procedure to yield a baseline comparison for the computation of the free energy, I would expect that integrating the mean energy with respect to beta or simply using directly the empirical distribution of energy delivered by the Wolff algorithm would yield more robust estimation but I maybe wrong? The size itself (128x128) seems rather limited compared to what can commonly be achieved with the Wolff algorithm. It seems to be a limitation in term of computation time but this is not discussed. Actually the scaling of the the computational time  of RiGCS w.r.t to L is actually not given. Another possible weak  point is that the proposal of the paper is very similar in spirit with another recent one which also uses a top-down regressive model  in scales (also explicitly using  RG concepts), in combination with wavelets (Marchand et al. 2022). While this work is mentioned in the paper, it is not really discussed, no element of comparison or even qualitative elements describing advantages of  RiGCS over this one are being given.  Finally the experiments concentrate on 2D Ising paradigm, a larger spectrum of experiments, on models with continuous variables in particular, would be beneficial to convince the reader of the flexibility of the method.

Despite these reserves, I find the method quite appealing due to its simplicity and the performances which are displayed.

### Questions
If the author could comment on the points raised in the previous paragraph:
in particular
- How does the algorithm scale  system size (even empirically) ?
- Is there a key difference of RIGCS wrt   the model developed by Marchand et al. (Wavelet conditional renormalization group)?


Minor points:

Instead of what is written on p4, the Wolff algorithm seems also to work for continuous variables like O(n) variables as can be seen  for instance in Kent-Dobia, Sethna PRE (2018)

I had difficulties to understand the notations in the begining, concerning in particular the definition of the hierarchy of variables.  I am not sure how I should understand the last sentence of p4 for instance, whether (i) there are coarse grained variables which are defined hierarchically or whether (ii) the original variables are organized hierarchically depending on their position on the lattice as in the Schmidt paper (I believe it is (ii) after reading it)

bibliography could be ordered in alphabetic order (to help old school readers using printed version:) )

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper presents a novel approach, the Renormalization-informed Generative Critical Sampler (RiGCS), designed for efficient sampling in near-critical regimes, leveraging the concept of scale invariance at criticality (SIC). By integrating MultiLevel Monte Carlo (MLMC) with Heat Bath (HB) algorithms, the authors aim to improve sampling efficiency while addressing challenges associated with critical slowing down in Markov chain Monte Carlo (MCMC) methods.

### Strengths
The proposed method utilizes SIC effectively, enabling ancestral sampling from low-resolution to high-resolution lattice configurations with site-wise-independent conditional HB sampling. This innovation enhances both scalability and efficiency, making it a significant contribution to the field of generative models.

### Weaknesses
(1) It appears that the training of the algorithm at each resolution follows a simultaneous sampling and training paradigm. However, this is not clearly articulated in the manuscript. I recommend providing complete pseudocode for the algorithm, which would greatly aid readers in understanding the implementation and workflow.

(2) The loss function presented in equation (11) is known to have a mode-seeking tendency when training generative models. This could lead to the neglect of certain peaks in multimodal distributions. It would be beneficial for the authors to address whether RiGCS is susceptible to similar issues and, if so, how they might mitigate them.

(3) The paper does not mention an important class of methods for the Ising model that rely on relaxation to continuous variable representations. Approaches such as the Neural Network Renormalization Group by Li and Wang use this relaxation technique to enable generative sampling in a continuous variable space. Including a discussion of these relaxation-based methods would provide a more comprehensive comparison and allow readers to better appreciate where RiGCS stands in relation to existing methods.

### Questions
(1) Provide a detailed algorithm or flowchart that outlines the key steps of RiGCS, including how training proceeds at each resolution level. This would help clarify the simultaneous sampling and training processes.

(2) Address whether RiGCS is susceptible to the mode-seeking, if so, how they might mitigate them.

(3) Compare the proposed approach to methods like the Neural Network Renormalization Group, highlighting potential advantages or trade-offs of RiGCS in relation to these continuous relaxation based techniques.

### Soundness
4

### Presentation
3

### Contribution
4
