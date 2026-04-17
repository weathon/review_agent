# Demystifying MaskGIT Sampler and Beyond: Adaptive Order Selection in Masked Diffusion

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Masked diffusion models have shown promising performance in generating high-quality samples in a wide range of domains,
but accelerating their sampling process remains relatively underexplored. To investigate efficient samplers for masked diffusion, this paper theoretically analyzes the MaskGIT sampler for image modeling, revealing its implicit temperature sampling mechanism. Through this analysis, we introduce the "moment sampler," an asymptotically equivalent but more tractable and interpretable alternative to MaskGIT, which employs a "choose-then-sample'' approach by selecting unmasking positions before sampling tokens. In addition, we improve the efficiency of choose-then-sample algorithms through two key innovations: a partial caching technique for transformers that approximates longer sampling trajectories without proportional computational cost, and a hybrid approach formalizing the exploration-exploitation trade-off in adaptive unmasking. Experiments in image and text domains demonstrate our theory as well as the efficiency of our proposed methods,
advancing both theoretical understanding and practical implementation of masked diffusion samplers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a deep analysis of the MaskGIT sampler's behavior in image and text generation, revealing its implicit temperature sampling mechanism and explaining its degraded performance with increased sampling steps. Building on this theoretical analysis, the authors introduce the "moment sampler," an asymptotically equivalent but more tractable and interpretable alternative that employs a "choose-then-sample" (CTS) strategy. To further enhance the efficiency of CTS algorithms, two key techniques are proposed:
1）Partial Caching Approximation: For Transformer-based models, this technique approximates sampling trajectories with more effective steps without proportionally increasing computational cost.
2）Hybrid Approach for Exploration-Exploitation Balancing: This formalizes the exploration-exploitation trade-off in adaptive unmasking for masked diffusion sampling, developing a hybrid method that combines the strengths of both.
Experiments on the ImageNet image dataset and OpenWebText text dataset validate the theoretical findings and demonstrate the efficiency and effectiveness of the proposed methods.

### Strengths
1. Theoretical Analysis. The paper provides the theoretical analysis of the MaskGIT sampler, uncovering its implicit temperature sampling mechanism and explaining its performance degradation with increased sampling steps. This understanding is crucial for further advancements in masked diffusion models.
2. The introduction of the moment sampler offers an asymptotically equivalent yet more interpretable and tractable alternative. It adopts "choose-then-sample" strategy and proposes partial caching approximation and the hybrid approach for exploration-exploitation balancing. Partial caching promises to improve sampling efficiency without significantly increasing computational cost, while the hybrid method effectively balances exploration and exploitation in sampling strategies.

### Weaknesses
1. It is a critical problem to determine the tokens to be sampled in the "select-and-sample" paradigm. The analyses in this part are insufficient. It confuses me how Algorithm 2 is implemented and how to determine the k tokens to be sampled. More details and maybe code are encouraged to verify the actual modification of this method over Maskgit.
2. This paper employs the unconditional MAGE as the baseline, lacking necessary verification on diverse models, e.g., the class-conditional Maskgit, and text-conditional model.
3. Even if on MAGE, the performance gain is marginal compared to the baseline as shown in Figure 3 and 4. Besides, a quantitative performance table are encouraged for better visualization.

### Questions
While the partial caching method aims for efficiency, the paper notes that its performance improvement is not significant on certain GPUs (e.g., H100 GPU), and computational overhead can even become noticeable (Section D.2.1). This may suggests that the practical benefits of this method might be highly dependent on hardware and model architecture, and its general applicability requires further investigation.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates why MaskGIT’s parallel decoding sometimes produces inconsistent image quality. It reformulates MaskGIT sampling as a moment-based process, revealing that it implicitly performs temperature-scaled categorical sampling. Through this lens, the authors propose two improvements—partial caching and balancing—to stabilize token updates across steps. Partial caching reduces noise by reusing confident predictions, while balancing controls the trade-off between stability and diversity. Theoretical analysis and experiments show that combining both leads to higher-quality samples with fewer steps. These findings generalize beyond Moment to other parallel diffusion-style samplers like MaskGIT, showing the mechanism’s universality.

### Strengths
- There are too many notations, which makes the paper hard to follow, but if one manages to track them carefully, the explanations appear to be quite correct and well-founded.

- Naturally, in the MaskGIT sampler, increasing the strength of the added Gumbel noise would lead to sampling from a more smoothed distribution, and it was nice to see this explicitly stated at the beginning of Section 3.

- The difference from the original MaskGIT sampler is clearly demonstrated through Theorem 2.

### Weaknesses
- The paper defines too many unnecessary notations, which makes it difficult to read. For example, why did they even define $\beta=1+\frac{1}{\alpha}$? It seems that $\gamma$ is an important parameter, but it is only defined in the appendix through the algorithm, not in the main text.

- Honestly, it’s unclear whether the performance gap with MaskGIT is actually significant. In Figure 3, the temperature seems to be a more critical factor that influences performance. Would it be possible to show ImageNet samples generated by MaskGIT and the Moment sampler under the same random seed and number of steps?

### Questions
- In Figure 3, it seems that the optimal temperature varies across sampling steps. Can this phenomenon be explained mathematically?

- Are the methods introduced in Sections 4.1 and 4.2 applicable to the original MaskGIT sampler?

- Is the performance gain significant?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel sampling approach for Masked Diffusion models, termed Moment Sampler.
The authors provide a theoretical analysis of the MaskGIT sampler, which is commonly used in Masked Diffusion sampling, and introduce the Moment Sampler as an approximation to it.
They prove that under certain conditions, the proposed method well approximates the MaskGIT sampler.
Furthermore, they theoretically show that the loss of the generated samples obtained by the proposed algorithm is bounded in terms of exploration and exploitation, and to achieve this, they also propose an adaptive order selection method.
Experimental results demonstrate that the proposed method can effectively approximate the MaskGIT sampler across various domains, including image and language generation.

### Strengths
- The paper provides a strong theoretical analysis of the MaskGIT sampler.
This contributes to a deeper understanding of the mathematical foundation underlying how MaskGIT generates images.

- The authors propose the Moment Sampler, which effectively approximates the MaskGIT sampler while enabling the use of transformer caching, thereby reducing the computational complexity of the sampling process.

- The proposed approach is domain-agnostic and applicable to both image and text generation within the masked diffusion framework.

### Weaknesses
- The main limitation of the proposed method is that it does not surpass the performance of the MaskGIT sampler.
It is designed as an approximation method, and achieving a good approximation requires increasing the number of sampling steps (i.e., decreasing the demasking ratio).

- In Proposition 3, the paper claims that one-by-one sampling is unbiased; however, this theoretical advantage may not hold for generating large images or long sentences.
In practice, some degree of biased sampling may be unavoidable to maintain efficiency.

### Questions
- Error bound for the approximation in Eq. (2): 
In Eq. (2), you state that the approximation holds when $N−k$ is large. Could you provide an error-bound analysis for this approximation? Specifically, for which ranges of $N−k$ does the approximation remain accurate, and within what quantitative error tolerance?

- In Appendix D, it is mentioned that partial caching requires more computation, but the reason for this is unclear.
What specific advantages does partial caching provide compared to the original MaskGIT sampler?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper fouces on improving the sampling efficiency of masked diffusion models. The paper first provides a theoretical analysis of the MaskGIT sampler and show that it implictly incorportates a temperature-based sampling mechanism. Building on this insight, they propose the moment sampler, an asymtotically equivalent yet more interpretable choose-then-sampler approach. To improve efficiency, the paper introduces a partial cahcing technique and a hybrid strategy that formalize the exploration-exploitation trade-off during sampling.

### Strengths
* The paper provides a clear theoretical explanation of what conditoinal distribution the MaskGIT sampler draws samples from, effectively conneting it to the Gumbel-top-k tric. The analysis showing that MaskGIT sampling implicitly performs biased temperature-based sampling is insightful and could be valuable for future studies on masked diffusion model sampling.

* Building upon this analysis, the introduction of the moment sampler is reasonable. The proposed choose-then-sample algorithm is simple yet novel, enabling the integration of efficiency-enhancing technqiues discussed in the paper.

### Weaknesses
* While the theoretical analysis is convincing, the practical advantage of the proposed method as a more efficient sampler is not fully demonstrated.
  * The paper suggests that, from the experiments in language modeling, temperature sampling reduces generation diversity, but I want to know that the proposed approach mitigate this effect. Further empirical evidence and discussion would clarify this point.
  * Also, this language experiments appear to be conducted using only a single small pretrained model. Since the proposed method is described as a post-hoc sampling improvement, it would be valuable to verify whether the same effects and efficiency gains are observed across different models, including larger and more recent architectures.

* It could be beneficial to provide the derivation of the approximation in Eq. (2) in more detail. Since this approximation forms the foundation of subsequent formluation, a step-by-step explanation and, if possible, an error analysis would strengthen the rigor of the paper.

* The proposed method for handling the exploration-exploitation trade-off appears somewhat ad hoc. It is unclear whether simply selecting the first-m and -n elements from two orderings adequately captures the trade-off mechanism. Also, the proposed hybrid method seems asymmetric: swapping indices $i$ and $j$ may yield different $k$ even when the corresponding $m$ and $n$ are the same (e.g., $k=(4,3,2,6,5,1)$ in the paper's example). It would be helpful to discuss whether this asymmetry could cause any issues or biases.

### Questions
Please provide discussions or clarifications on the points raised in the Weaknesses section.

I think the theoretical analysis of the paper to be strong and insightful, but I would like to see more experimental evidence demonstrating the practical strengths of the proposed method. In particular, additional experiments across different models or settings that clearly highlight the efficiency and effectiveness of the proposed approach would significantly strenghten the paper.

### Soundness
2

### Presentation
2

### Contribution
2
