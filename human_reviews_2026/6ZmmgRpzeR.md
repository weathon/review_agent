# IndiSeek learns information-guided disentangled representations

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Learning disentangled representations is a fundamental task in multi-modal learning.
In modern applications such as single-cell multi-omics, both shared and modality-specific features are critical for characterizing cell states and supporting downstream analyses.
Ideally, modality-specific features should be independent of shared ones while also capturing all complementary information within each modality. 
This tradeoff is naturally expressed through information-theoretic criteria, but mutual-information–based objectives are difficult to estimate reliably, and their variational surrogates often underperform in practice. 
In this paper, we introduce \ours, a novel disentangled representation learning approach that addresses this challenge by combining an independence-enforcing objective with a computationally efficient reconstruction loss that bounds conditional mutual information. This formulation explicitly balances independence and completeness, enabling principled extraction of modality-specific features. 
We demonstrate the effectiveness of \ours on synthetic simulations, a CITE-seq dataset and multiple real-world multi-modal benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents IndiSeek, a novel framework for disentangled representation learning in multi-modal settings. The goal is to extract shared features that capture cross-modal information and modality-specific features that retain complementary information unique to each modality. The method builds on an information-theoretic formulation that balances independence (minimizing mutual information between shared and specific components) and completeness (maximizing the retained information about the input).

### Strengths
- Derives a tractable objective from mutual information principles, avoiding fragile variational estimations.

- Comprehensive experiments (synthetic, biological, and benchmark) with consistent outperformance of SOTA baselines.

- Visualizations (Figures 1–2, 4) and correlation analyses (Figure 5) confirm that IndiSeek captures meaningful modality-specific patterns.

- Works in diverse domains (CITE-seq biology, video, text, audio), demonstrating transferability.

### Weaknesses
- The method relies on manual grid search for the tradeoff hyperparameter λ. A principled or adaptive selection scheme would enhance practicality.

- While the reconstruction bound is justified via Fano’s inequality, tightness conditions (e.g., non-Gaussian distributions) are not deeply discussed.

- Experiments could include more recent disentanglement baselines or adversarial approaches (e.g., variational or adversarial mutual information bounds).

- Although efficient in small models, scalability to very high-dimensional modalities (e.g., video) may present computational challenges.

- Results mainly rely on linear classifiers; evaluating transfer learning or generative reconstruction would add depth.

### Questions
- How does IndiSeek perform when the reconstruction loss is replaced by more expressive models (e.g., diffusion-based decoders)?

- Can λ be adaptively tuned during training, perhaps via normalization of CLUB and reconstruction magnitudes?

- What is the computational cost of CLUB estimation relative to InfoNCE in larger settings?

- Could the framework handle partial modality observation, e.g., missing one modality at inference?

- How sensitive are results to the dimensionality of shared vs. specific latent spaces?

### Soundness
3

### Presentation
2

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
This paper introduces a new combination of objectives to learn disentangled representations for multi-modal data: “IndiSeek” first learns shared features using CLIP, followed by the specific features using NCE-CLUB and a simple reconstruction loss. These two objectives in the second stage ensure disentanglement from the shared information and completeness of the full representation. They show that approximating conditional mutual information with a reconstruction loss is sufficient and performs better than InfoNCE loss used in comparable methods. The authors compare IndiSeek’s performance with two highly comparable SOTA methods on simulations, and various real world datasets.

### Strengths
1) The core idea of using CLIP in a 2-stage solution and replacing the InfoNCE loss with reconstruction loss for completeness is simple but intuitive. While the novelty is incremental, the approach is well-justified with information-theoretic bounds and promises to be more robust.

2) The method is tested on a wide variety of data (simulation, biological data, multibench datasets), which demonstrates its applicability to many domains.

### Weaknesses
1) There seems to be a problem of information leakage from the modality-specific information into the shared representations C during the first phase of training with CLIP. Information from more than just shared variables can easily leak into C via CLIP. And since it seems like C is not updated during the second phase, there is a high chance of C containing modality-specific information, which then may not even get into Z because it is not required per the completeness. The paper provides no discussion, proof of disentanglement, or simulation analysis on this issue. Given the focus of the work being disentanglement, more extensive experiments should be conducted related to this.

2) While the paper provides results on many datasets (especially in the case of MultiBench), their significance is unclear. The average accuracies over 10 random seeds are provided without error margins. It is crucial to report for example the SD or SEM to determine if the performance is at all statistically significant. Given that the methodological novelty is incremental, it is important to show that it works better than competitors and is thus relevant.

3) The motivating example (Section 1.1) is presented at the very beginning. As a reader, this was very confusing because the method and the baselines were not yet introduced, making the results hard to appreciate. It would be a lot more effective to move this right after section 3 and merge paper structure into the introduction.

4) The paper does not state the input feature dimension for RNA. Also, the descriptions of used architectures (i.e. “5-layer ReLU NNs with middle layers of width 50”) are also not quite sufficient. What is the hidden dimension? Does this refer to the encoder and the decoder? I am further concerned with the small fixed MLP (width 50) for both the low-dimensional ADT data and the high-dimensional RNA. For one this leads to an over-, for the other to an under-determined model and may skew the results of their relative importance. It would be good to at least have some results on the reconstruction performance on each of the modalities here as a sanity check.

5) The paper only compares against two very similar methods (Factorized CL and InfoDisen). While these are the most relevant, the authors could include other families of multi-modal disentanglement methods, such as VAE-based approaches (e.g. DMVAE, IDMVAE) or alternative information-theoretic methods (e.g. DisentangledSSL). While they discuss why some other methods were not included in the comparison (mainly if they are designed with task-dependence), other works like CoMM still compare to Factorized CL for example and since some MultiBench accuracies are reported on multi-task performance, these could be included too. But I think if the two points above are addressed adequately, this is only a minor weakness since I get the motivation of showing that the 2-stage approach and objective change make a big difference.

### Questions
1) Minor formatting and text issues: For example, the error bars in Figure 5 are not explained in text or caption. I think I know what equation 2 is trying to say, but I think there are some latex errors / spelling mistakes?

2) Missing methodological details: the decoder network (g) is not explicitly described in Section 3 and needs to be introduced for clarity and reproducibility.

### Soundness
3

### Presentation
2

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
- The paper tackles disentangling shared versus modality-specific information in multimodal data and proposes IndiSeek, which first learns shared features with a standard contrastive stage and then learns modality-specific features by jointly enforcing independence from the shared part and accurate reconstruction of each modality. 
- The method replaces intractable mutual-information objectives with two surrogates: an NCE-CLUB term to upper-bound dependence between shared and specific parts and a reconstruction loss that upper-bounds the conditional entropy, thereby encouraging completeness of the modality-specific information. 
- The approach is validated on controlled simulations, on a CITE-seq single-cell dataset where learned specific features align with known RNA-versus-protein importance, and on MultiBench where linear probes over the learned representations outperform baselines on several tasks.

### Strengths
- The paper frames disentanglement with a clear two-goal target: independence from the shared features and completeness of modality-specific information, and then instantiates each goal with a surrogate that is consistent with the desired optimization direction. 
- The simulations are set up to isolate modality-specific signal and show where prior methods either over-entangle or miss relevant coordinates, while IndiSeek recovers the correct factors across multiple regimes and hyperparameters. 
- The paper is well written and is easy to follow.

### Weaknesses
- The independence term relies on NCE-CLUB, whose tightness and variance can be sensitive to density-ratio estimation quality, so residual dependence between shared and specific parts could persist and re-enter through the reconstruction path. 
- The contrastive stage assumes that the learned shared representation is close to minimally sufficient, but if the contrastive training under- or over-captures cross-modal information, the second stage may respectively force specific features to absorb shared signal or collapse useful unique signal. 
- The MultiBench gains are mostly modest and while are reported as best over $\lambda$ and averaged over seeds but without confidence intervals or error bars, which makes it difficult to assess the practical margin and stability of improvements. 
- The linear-probe setup concatenates shared and specific features, so improvements could partly reflect larger feature capacity rather than better disentanglement, and there is no comparison to capacity-matched baselines (for example, CLIP with a projection to the same total dimensionality).

### Questions
see weaknesses above

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
The paper proposes a modified, practical criterion for disentangling modality specific features from features shared across the modalities. The main contribution in the paper is the author's criterion for finding modality specific features when shared features have already been established in the previous (typical) step.

### Strengths
Many data analysis tasks today are multimodal. Most techniques for disentanglement are variations on a theme. The authors focus on reducing the criterion for modality specific features into a practical form with reasonable motivation. While the conceptual two-step procedure from shared to specific seems similar to other methods, the reduction into a practical approach is different. Section 2 provides a good tutorial introduction for desiderata.

### Weaknesses
The use of synthetic examples should be to illustrate generalizable patters in simple settings. The examples shown in the paper appear to have been constructed a bit too intricately without convincing rationales. Regardless of whether authors' method typically dominates others, it is possible to find specific cases where the trends are reversed. Such examples are not convincing. You could instead take the graphical model that all the methods essentially agree with, randomly select standard non-linear dependencies and associated parameters for the conditionals, and show how the methods compare in the resulting disentanglement tasks, including how well they succeed in their respective optimizations. 

eq (1) in the paper doesn't seem correct. In the approach of Wang et al., shared and specific encodings are estimated in two steps. The second step for modality specific representation makes use of the already inferred shared representation from the other modality, not the same modality as written in eq (1). Which version did the authors use in comparisons? 

Paper isn't self-contained, e.g., NCE-CLUB loss is not given in the form that can be estimated (e.g., using a learned plug-in critic from NCE). 

Due to elaborate steps in defining performance in the CITE-seq exmaple, it is hard to use it as a measure of quality among disentanglement methods. Cell annotations are based on weighted clustering (Hao et al) and the authors' theta(c) per cell is constructed by taking 10 nearest Euclidean neighbors in the respective estimated modality specific features and measuring the annotated label consistency among neighbors. It is unclear how the resulting agreement with RNA modality weights (a specific construction used in Hao et al) indicates representational quality.

### Questions
How is NCE-CLUB implemented? 
Which steps are used to ensure that the methods used for comparison succeed in optimizing their respective criteria?
In the comparisons, are the hyperparameters optimized for authors' method also used for the other methods?

### Soundness
2

### Presentation
3

### Contribution
2
