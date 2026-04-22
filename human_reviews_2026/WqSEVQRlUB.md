# Tokenize Image as a Set

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
This paper proposes a new paradigm for image generation through set-based tokenization and modeling. Unlike conventional methods that serialize images into fixed-position latent codes with a uniform compression ratio, we introduce an unordered token set representation to dynamically allocate coding capacity based on regional semantic complexity. This TokenSet enhances global context aggregation and improves robustness against local perturbations. To address the critical challenge of modeling discrete sets, we devise a dual transformation mechanism that bijectively converts sets into fixed-length integer sequences while preserving summation constraints. Further, we propose Fixed-Sum Discrete Diffusion—the first framework to simultaneously handle discrete values, fixed sequence length, and summation invariance—enabling effective set distribution modeling. Experiments demonstrate our method's superiority in semantic-aware representation and generation quality. Our innovations, spanning novel representation and modeling strategies, advance visual generation beyond traditional sequential token paradigms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an image tokenizer which is approximately invariant to token order. The authors empirically observe that this leads to tokenized representations that are even more decoupled from spatial/local bias common in other tokenizers (even in TiTok, to some extent, despite its 1D representation).

The authors then introduce “Fixed-Sum Discrete Diffusion” (FSDD), a diffusion-based generative model over multisets/bags of tokens, in order to leverage the tokenizer’s latent space for image generation.

### Strengths
While the method used to encourage this approximate permutation invariance is very simple (random token shuffling during training), results demonstrating semantically aligned token receptive fields, high linear probing accuracy, and robustness to noise in the input, are interesting and motivate the importance of such a seemingly minor tweak to TiTok.

The “Fixed-Sum Discrete Diffusion” approach is novel and is demonstrated to outperform certain baselines which do not exploit the permutation invariance.

### Weaknesses
Generation performance appears to be compromised compared to TiTok (FID ~5.1 compared to FID ~2.0), even when using the proposed FSDD. It is not clear whether this is due to limitations of the proposed generative model (FSDD) itself, or due to the learned bag-of-tokens representation having unfavorable structure for generation. Regarding the FSDD, some details and design choices still remain unclear to me and should be described and supported more explicitly (see questions below). Overall, further discussion, analysis and experiments related to generation could strengthen the paper.

### Questions
**Q1.** Section 3.3: I am a bit confused about the choice of $\sigma_t$ in eq. (5), as it depends on the samples and is not dependent on $t$ like a usual noise schedule in flow matching/diffusion would. Is there a typo here, or if it is intended, could you clarify or provide references? I have a similar question about eq. (9) -- is $|\mathbf{X}_1 - \mathbf{X}_0|$ referring to some precomputed statistic, since $\mathbf{X}_0$ is not available during sampling? It would also be good to elaborate on the t-dependent truncation ratio $f$.

**Q2.** Section 3.3: Have you considered any other strategies for the initial distribution of $\mathbf{X}_1$? (For example, starting with all counts assigned to a “mask” bin.) Additionally, the constraint that $\sum \mathbf{\tilde X}_t = M$ for every $t$ might seem a bit restrictive. Have you considered any alternatives, such as modeling unconstrained histograms (using any of the existing categorical/multinomial diffusion approaches) and normalizing at the end, or any other modifications to expand the design space of your generative model?

**Q3.** Section 4.3: Do the authors have any insight into why modeling the unordered representation appears to be “harder” than the 1D tokenization used by TiTok? Even with your proposed FSDD, the gFID is significantly lower than in the case of TiTok. Or is the worse generation performance purely due to limitations of FSDD?

Section 4.3.1 / Table 4:
* **Q4.** What is SetAR? I did not see it explained in the text. Does it refer to autoregressive modeling of the token histogram/“dual transformation”?
* **Q5.** Are the authors familiar with autoregressive set prediction similar to [1]? It would be interesting to try this or a similar approach performing autoregressive set prediction without the dual transformation.

**Q6.** In Section 4.2.1, rather than a qualitative demonstration on a single example, it would be better to provide quantitative metrics. In particular, to better understand the effectiveness of the train-time shuffling in encouraging permutation invariance, could the authors provide e.g. a histogram across the whole validation set of the error between images reconstructed from two different random token orderings or MSE/MAE with stddev/quantile statistics?

**Q7.** I appreciate the inclusion of MAE in the linear probing results from Table 2, and it would be great if you could also include a comparison between an ablation of your tokenizer trained without shuffling to further support the claim that tokens from permutation invariant representations are semantically richer. 

---

Minor questions/comments:
* Use of “set” nomenclature: Referring to the representation as a “multiset” or “bag” from the beginning would be clearer, rather than leaving this as a footnote. In my own reading of the manuscript, I did not notice the footnote until explicitly searching for the word “multiset”, as I was a bit confused by the inaccurate use of the word “set”. Similarly, I would encourage the authors to consider altering the title to avoid being technically incorrect.
* 3.1, L148: What does “partial” permutation sampling mean? Is it something other than simply shuffling the tokens? 
* 1, L069: I did not understand what is meant by “dynamic token allocation based on semantic complexity”, maybe this statement could be made more precise.
* L297. Typo, missing “is”?
* References: capitalization is not correct in some instances, e.g. “gan” -> “GAN”, “bayes -> Bayes”, “Neurips” -> “NeurIPS”, etc.

---

[1] Autoregressive Transformers for Indoor Scene Synthesis, Paschalidou et al. NeurIPS 2021 (https://arxiv.org/abs/2110.03675)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel image generation method using set-based tokenization, representing images as unordered token sets instead of fixed sequences. It adaptively allocates coding capacity by regional complexity and improves global context and robustness. A dual transformation converts sets into fixed-length sequences, and Fixed-Sum Discrete Diffusion models discrete sets effectively. Experiments show superior semantic and visual quality. However, it appears that the authors did not include the Ethics Statement and Reproducibility Statement in their paper, which could diminish its overall quality.

### Strengths
[1]. This paper introduces a novel set-based image tokenization method that eliminates positional bias and enhances semantic representation.

[2]. It demonstrates strong robustness against local perturbations and image noise.

[3]. The method dynamically allocates representational capacity according to regional semantic complexity, leading to improved efficiency.

[4]. The experiments are sufficient and well-designed, and the figures are clear and easy to understand.

[5]. The paper provides solid theoretical analysis, offering convincing proofs that support the proposed method.

### Weaknesses
[1] Reference formatting issue. The reference section requires further improvement. In particular, please correct the formatting of the conference name. It should be written as **NeurIPS** rather than **Neurips**.

[2] Potential extension to text-to-image models. It is not entirely clear whether the proposed method can be effectively extended to text-to-image models. If time and computational resources allow, I suggest conducting an additional experiment by training a T2I model using JourneyDB, which could provide valuable empirical evidence of the method’s broader applicability. Even a preliminary result or discussion in the supplementary material would make the contribution more convincing.

[3] I could not find the Ethics Statement or Reproducibility Statement in the current submission. These sections are required by the NeurIPS submission guidelines and play an important role in ensuring transparency, responsible research practices, and reproducibility. Please include these statements in the final version, outlining the ethical considerations of your work (e.g., data usage, societal impact) and providing clear details about how the experiments can be reproduced by other researchers.

### Questions
[1]. How does your method perform when applied to text-to-image models?

[2]. Could you please indicate where the Ethics Statement and Reproducibility Statement are located?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TokenSet, which represents images as unordered token sets instead of sequences, enabling permutation-invariant encoding and dynamic token allocation based on semantic complexity. A dual transformation converts token sets into fixed-length count vectors, and a Fixed-Sum Discrete Diffusion model enforces discrete, fixed-length, and constant-sum constraints for generation. Overall, the work shifts from positional tokenization to set-based representation with improved semantic awareness and robustness.

### Strengths
1.	The paper presents a novel set‐based tokenization and generation paradigm, which is conceptually fresh and offers a compelling alternative to conventional spatial or sequential visual token representations.
2.	The proposed method demonstrates strong robustness to perturbations and noise, showing that the set‐based formulation effectively captures high‐level semantics and maintains stability under input corruption.

### Weaknesses
1.	Generation quality remains insufficient (Table 5).
Although the method shows strong robustness and semantic consistency — consistent with its motivation — high-fidelity generation is still essential for evaluating visual tokenizers. Table 5 frames generation quality as future work, but without competitive results, it is difficult to fully validate the practical impact of the approach.
2.	Token count does not reflect real inference efficiency (Tables 3 & 4).
The model requires 25 sampling steps, making inference closer to diffusion than auto-regressive decoding. Thus, reporting only token count (Tables 3 & 4) may be misleading. Efficiency should be evaluated under comparable inference budgets (e.g., wall-clock time or compute) against diffusion and AR models to substantiate the claimed speed advantages.

### Questions
1.	Since the model generates a token-count distribution vector, will increasing the number of tokens affect inference time under the same number of sampling steps? In other words, does the inference cost scale with token quantity in practice?
2.	The paper reports results using 25 sampling steps. If the number of sampling steps increases, will the image quality improve further, similar to standard diffusion models?
3.	In what scenarios would the proposed method have clear advantages over traditional autoregressive or diffusion models? Specifically, when does the set-based token generation outperform these approaches in terms of performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel image tokenization method and its corresponding image generation method. The main idea is to encode images as a set of permutation-invariant tokens. Accordingly, the authors propose FSDD to apply a diffusion process on the fixed-sum count vectors derived from the image tokens.

### Strengths
1. The idea of encoding an image as a set of tokens is well-motivated.
2. The corresponding generation method of FSDD aligns well with the tokenization method.
3. The authors provide some interesting experimental results, such as attention visualization and semantic clustering.

### Weaknesses
1. The main weakness is that the performance of the proposed method lags behind state-of-the-art models, such as TiTok, in both rFID and gFID. The performance potential of the proposed method has not been convincingly shown.
2. Permutation-invariance is a good intuitive property for image tokenization. However, I find it difficult to conceptually relate it to reconstruction or generation performance.

### Questions
1. In Table 1, why does a higher percentage of overlapping tokens indicate greater robustness? If we assume a very poor tokenizer that encodes every image into the same set of tokens, the overlap percentage would be 100%. Therefore, I don’t think this result provides strong evidence for evaluating robustness.
2. In Table 2, the linear probe results are interesting, as they indicate that the proposed method provides good semantic representations. Could the authors explain why models with a smaller number of tokens achieve better linear-probe performance?

### Soundness
3

### Presentation
3

### Contribution
2
