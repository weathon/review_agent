# Neural  Bridge Processes

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Learning stochastic functions from partially observed context-target pairs is a fundamental problem in probabilistic modeling. Traditional models like Gaussian Processes (GPs) face scalability issues with large datasets and assume Gaussianity, limiting their applicability. While Neural Processes (NPs) offer more flexibility, they struggle with capturing complex, multi-modal target distributions. Neural Diffusion Processes (NDPs) enhance expressivity through a learned diffusion process but rely solely on conditional signals in the denoising network, resulting in weak input coupling from an unconditional forward process and semantic mismatch at the diffusion endpoint. In this work, we propose Neural  Bridge Processes (NBPs), a novel method for modeling stochastic functions where inputs \( x \) act as dynamic anchors for the entire diffusion trajectory. By reformulating the forward kernel to explicitly depend on \( x \), NBP enforces a constrained path that strictly terminates at the supervised target. This approach not only provides stronger gradient signals but also guarantees endpoint coherence. We validate NBPs on synthetic data, EEG signal regression and image regression tasks, achieving substantial improvements over baselines. These results underscore the effectiveness of DDPM-style bridge sampling in enhancing both performance and theoretical consistency for structured prediction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Neural Bridge Processes (NBPs), which explicitly integrate input supervision throughout the temporal diffusion trajectory (similar to guidance in diffusion models), whereas Neural Diffusion Processes (NDPs) can be viewed as conditional diffusion without such guidance. The goal is to address weak coupling and endpoint mismatch of NDPs, with modest  reported improvements over baselines.

### Strengths
This idea of enforcing trajectory-level conditioning is a reasonable, simple, and practical approach compared to the conditional setting in NDPs. The improvement over NDPs is expected, as diffusion models with guidance are known to align better to conditions than diffusion without guidance.

### Weaknesses
Despite its practical appeal, the paper contains several major weaknesses that severely limit its contribution.

**Major Weaknesses:**

1. **Unsubstantiated claims of stochastic-process validity (KET).**

The paper claims that both NDPs and the proposed NBPs are “consistent with the Kolmogorov Extension Theorem (KET)” (line 1209) and therefore define a valid stochastic process over functions. This is highly problematic. A known limitation of NDPs is their lack of marginal consistency required by KET, which prevents them from defining a valid stochastic process (explicitly acknowledged in Chapter 5 of the NDP paper). The manuscript provides no proof that the proposed method is marginally consistent, leaving the claim unsupported.


2. **Limited novelty and missing Diffusion Inpainting baselines.** 

The contribution feels incremental. The modifications relative to NDPs are minor, analogous to the difference between a standard diffusion model and one with guidance. Moreover, the extensive body of work on diffusion-based image inpainting and regression is neither discussed nor compared.

The core technical concept, using the temporal trajectory of diffusion for posterior inference (conditional sampling) is not new. This was a central proposal of the Diffusion Posterior Sampling (DPS; Chung et al., 2023) family of models. The proposed “bridge” is essentially another form of guidance, a well-established technique in modern diffusion models, which makes the contribution appear limited.

The paper neither cites nor compares NBPs against any DPS variants. At a minimum, it should include a tighter discussion of how NBP relates to DPS and provide head-to-head empirical comparisons to justify its novelty and performance.

References:

Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023.



3. **Limited empirical improvement and outdated experimental settings**

The reported improvements over the NDP baseline are limited (e.g., Table 2; NBP performance is quite close to that of SNP). Figure 2 is not convincing: the loss curves for NDPs and NBPs nearly overlap, the result is based on a single run for one baseline, with no multi-seed uncertainty estimates.

The EEG dataset dates back 30 years and is clearly outdated, and the key image experiments use low-resolution ($32\times32$ and $64\times64$) regression, which is below contemporary practice. State-of-the-art DPS-style work typically evaluates significantly more challenging setups (e.g., FFHQ at $256\times256$) and harder inverse or regression problems. A comparison with DPS is necessary.

**Minor Weaknesses**


1.  The main paper defines the training objective as an $L_2$ loss (Eq. 18 ). However, Appendix G.3 explicitly states, "We adopt the $l_{1}$ loss for training the denoising objective"  for the image regression task. 
2. Line 59, what does "input $x$" refer to?  Context or (target, context) pairs? 
3. **Typos :**
1). The paper uses "EGG measurements" once (line 73) but "EEG" in the abstract and all other sections
2). Appendix G.2 refers to "DNP base", which should presumably be "NDP base." 
3). The paper uses "Bi-Dimensional Attention Block" and "Bi-Attention layers"  to refer to the same architecture. Please use one term consistently.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an approach to anchor the sampling process in a Neural Diffusion Process model using a set of anchor points. The approach is implemented using a modified transition kernel both for the forward and the reverse process. In the forward process the bridge is implemented by altering the mean of the transition kernel. This leads to a reverse process that includes a correction term to retain consistency with the forward process. The paper concludes with a set of results comparing the proposed method with Neural Diffusion Processes.

### Strengths
What a lovely introduction to the paper, well written and providing a honest account of previous work and framing the proposed model very well. The writing throughout the paper is very good with the right amount of technical detail in the main paper and the appendix.

To my knowledge the work is novel and addresses an important aspect of Neural Diffusion Processes. The results are sufficient but does not provide a lot of intuition to the proposed method.

### Weaknesses
The weakness of the paper is the result section, I would have much preferred more qualitative experiments to try and increase the intuition for the proposed work. While I'm very happy with the mathematical explanation of the proposed method the results does little to deepen the insight of these. To some extent I find some of the results in the appendix more important than the ones that are in the real paper. However, this is just a highly personal opinion and not something that have influenced my score.

### Questions
Could you provide an intuition to why you need to repeat the forward perturbation as highlighted line 320? What would be the effect if you did not do this?

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
4

### Summary
This paper presents a novel methodology for improving functional diffusion processes, which offer a promising method of inference on a broad range of datasets. NBPs modify the forward process to depend on x, anchoring the diffusion path to the input location. Empirical performance is validated on a suitable range of synthetic and real world tasks.

### Strengths
This work presents a novel, elegant, and well motivated adjustment to the diffusion paradigm.

The paper is clearly written and well presented.

Empirical performance appears strong across a good range of datasets.

### Weaknesses
My concerns with this work primarily relate to the presentation of the experimental results:

Table 1 is currently lacking uncertainty estimates on all metrics, so it is challenging for the reader to gauge the statistical significance of these results.

Similarly Figure 3 appears to be a single training run so we cannot draw quantitative conclusions

Sec 4.3.2 makes a claim on performance with a 0.02 context ratio, yet the results in Tables 4 and 5 display context ratios no lower than 0.1.

While there is a good variety of baselines, I would like to have seen more recent ones such as the cited Flow Matching Neural Processes.

### Questions
Have you considered variants to the design of gamma in equation (9), and how does this affect performance?

C.2.2 states the reparameterisation is similar to DDPM, but could we not reformulate as exactly DDPM, via a change of variables y' = y - gamma x, which potentially has an added benefit of desensitising the state from Var(x)?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Neural Bridge Processes (NBPs), a modified version of Neural Diffusion Processes (NDPs), which guides the forward diffusion steps with an additional conditioning on inputs x in the parametrization of the mean term for the transition probabilities. The main motivation for this is that NDPs do not consider inputs in the forward pass while doing it in the reverse one. The authors claim that this lack of conditioning limits the efficacy of input supervision when applying the model to data with temporal structure.

### Strengths
- The paper is well written and clear in general, which facilitates a lot the understanding and reproducibility of results.
- Related work, references, and literature are correctly reviewed, and I also highlight the rigour on the derivations, where I did not find any clear mistake at first sight.
- Contributions are cleared, and the mission of providing the NDP with the ability to capture temporal structure is well remarked.

### Weaknesses
- I have important concerns regarding novelty and the dimension of the technical contribution proposed. I would like to understand if the method goes beyond adding the bridge term in the forward pass plus the correction.
- Why this bridge term is chosen and how the correction is important are perhaps the most important details for understanding the key technical contributions of the manuscript. However, these are not correctly reviewed or explained in detail, which, from my perspective, is something that would greatly increase a lot the quality of the work.
- Empirical results are quite limited, and I am somehow confused about the lack of empirical demonstration of the issues of NDPs in dealing with temporal structure. I can perceive from the text that not having the conditioning on x on the forward pass causes issues, but I would've liked to see empirical evidence on this. For the proposed methodology, contributions, and the venue, I do think it is understandable that the work is somewhat limited in empirical results at its current state.
- The bridge coefficient design and the relationship with the SNR are super interesting, but similarly to the previous point, not a lot of information is provided about it, and the motivations that led modelling decisions being taken.

### Questions
- The use of math set-style of caligraphy for T and C for targets and context in section 2 and the beginning of section 3 is very confusing. It makes the reader think about the set of complex numbers (similar to how we would use the letter R, usually taken for reals, but for another purpose in the notation). In this direction, I do not really see the effectiveness of talking about meta-learning with context and targets, when just training/test or observed/test data is considered in the section for methodology.
- Table 2 for the synthetic experiment going to the Appendix is not a great decision from my point of view... Also, I do think there is space for including it in the main manuscript and turning the formulation a bit uncluttered, maybe.
- I didn't understand the role of corrupting the context data for the experiment in Figure 2, I am quite lost here, and the EEG+temporal structure should be enhanced, maybe for a better understanding of reviewers and reading (I mean, it looks to me is the driving force of the paper, to make the NDP be more robust and flexible in that regard).
- I have a significant curiosity about the reasons and causes for the correction term in the backward pass. Maybe I did not spend enough time on the Appendix, and some info is there, but for sure this sort of details should be mentioned in the main manuscript with some paragraphs.

### Soundness
2

### Presentation
3

### Contribution
1
