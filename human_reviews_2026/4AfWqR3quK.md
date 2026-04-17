# ON GOOGLE’S LLM WATERMARKING SYSTEM: THEORETICAL ANALYSIS AND EMPIRICAL VALIDATION

- Decision: Reject
- Scores: 8, 6, 4, 4

## Abstract
Google’s SynthID-Text, the first ever production-ready generative watermark system for large language model, designs a novel Tournament-based method that achieves the state-of-the-art detectability for identifying AI-generated texts. The system’s innovation lies in three key components: 1) a new Tournament sampling algorithm for watermarking embedding, 2) a detection strategy based on the introduced score function (e.g., Bayesian or mean score), and 3) a unified design that supports both distortionary and non-distortionary watermarking methods.

This paper presents the first theoretical analysis of SynthID-Text, with a focus on its detection performance and watermark robustness, complemented by empirical validation. For example, we prove that the mean score is inherently vulnerable to increased tournament layers, and design a layer inflation attack to break SynthID-Text. We also prove the Bayesian score offers improved watermark robustness w.r.t. layers and further establish that the optimal Bernoulli distribution for watermark detection is achieved when the parameter is set to 0.5. Together, these theoretical and empirical insights not only deepen our understanding of SynthID-Text, but also open new avenues for analyzing effective watermark removal strategies and designing robust watermarking techniques. Source code is available at https:
//anonymous.4open.science/r/Break-Synth-ID-text-EE5D/

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper is a theoretical derivation of the power and probability of False-alarm of Synth-ID text watermarking based on tournament sampling.

The author study the two main detector approaches advertised in the paper: mean score and Bayesian (requiring the retraining of a supervised detector). The analysis methodology chosen by the authors is resolutely asymptotic. They provide expression of the detector performance in the normal approximation using a CLT, as the text length approaches infinity.

However, the authors are careful to provide an empirical validation of the CLT validity.

Of note, the authors leverage their theoretical analysis to design an attack on the mean score detector. Notably, they demonstrate that the power of the detector is not an increasing function of the number of layers during tournament sampling. As such, they show that the simple (black-box) attack consisting in choosing tokens sampled from an artificially inflated number of layers breaks the watermark.

The authors also use their analysis to improve the choice in Synth-ID parameters choice (in this case, the g-value distribution)

### Strengths
The paper is very timely, providing a much needed theoretical analysis of SynthID and its very ad-hoc tournament sampling method. 

**Clarity**: All the necessary element to the comprehension of the paper are dully and exhaustively presented in the first half. The presentation of the results in the second half is simple and economical, highlighting the important part of the results without overwhelming with details (I think notably of Theorem 7 which could have been unreadable). 

**Significance**: Though SynthID claims to outperform much of the current art empirically, the theoretical validation of this fact was somewhat complicated by the exotic and ad-hoc design of their tournament sampling strategy, the many detector scoring functions in the appendix and the overall obscurity and size of the supplementary material of the original work. This paper provides a clean, short and concise framework to start true comparison with the current art which is quite welcomed.

**Good insights from the theory**: I truly appreciate that the authors derive simple yet actionable results from their theory. The attack is elegant and the optimal distribution of the tournament scores (g-values) is derived.

### Weaknesses
The authors address some limitations of their work at the end, I won't repeat those here.

**Asymptotic approach**: Contrary to KGW and Aaronson's scheme where exact p-values can be derived, the paper here provides only an asymptotic analysis of  the detector, preventing a perfect comparison with the current art. To be fair, this is still far better than a naive z-score, yet it is still a limitation of the analysis.

**Lack of comparison with existing theoretical findings**: Since exact p-values are available for KGW and Aaronson's in [1] (which the authors cite), as well as for the very similar (yet predating) WaterMax [2],  I find it disappointing that the authors did not plot the log-ROC curves for each algorithm. This would have allowed to know when SynthID is indeed the best algorithm of the bunch under the "no-attack setting".

**No comparison with the original frequentist detector**: A p-value based detector is actually described in the supplementary of Synth-ID (Appendix A.3), dubbed the frequentist detector. Comparing the asymptotic performance of the authors study with this detector would have been interesting to understand "how much is lost" by foregoing the supervised learning of the bayesian detector.

### Questions
- Following my questions on the p-values, could the author draw some curves comparing Synth-ID to the current art?
- Similarly, how does the frequentist detector fares compared to the Bayesian? Is the Bernoulli distribution still optimal in this case? Due to experiments I did on my own end, I suspect the frequentist detector to be extremely bad compared to the current art ; showing the gap between the Bayesian and frequentist approach would thus explain to me why the system claims to perform so well despite me finding that it performs far worse than WaterMax and KGW on mid-sized texts.
- (Nitpicking/Unsure) I wonder if Proposition 1 is necessary. It is a trivial consequence of the CLT and maybe could be compressed to allow more room for Figure 2?

**Recommendation**

This is a high quality paper, with sound results, within a limited scope of the distortion-free version of Synth-ID. The results are interesting and timely given the huge claims made by the SynthID systems.  As such, I recommend acceptance without much reserve, though I still would appreciate the authors adding some comparison to the state-of-the-art to complete the study.

**References**

    [1] Fernandez, Pierre, et al. "Three bricks to consolidate watermarks for large language models." 2023 IEEE international workshop on information forensics and security (WIFS). IEEE, 2023.
    [2] Giboulot, Eva, and Teddy Furon. "WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off." NeurIPS 2024-38th Conference on Neural Information Processing Systems. 2024.

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
3

### Summary
This paper presents a theoretical analysis of Google's production watermarking system for large language models. The system uses a Tournament Sampling algorithm to embed watermarks in AI-generated text.

### Strengths
- Theoretical analysis of a production watermarking system, using Central Limit Theorem to derive closed-form expressions for detection performance.

- Identifies a concrete vulnerability (layer inflation attack) and demonstrates it empirically.

- Proofs are detailed and the CLT assumption is validated empirically.

### Weaknesses
- Only analyzes non-distortionary setting (authors acknowledge this)

- Limited discussion of robustness against paraphrasing attacks

- Heavy mathematical notation may limit accessibility

- Could benefit from more intuitive explanations of why results hold

- Only tested on Google's watermarking method, the scope is not broad

### Questions
While LLM watermarking methods inherently suffer from various limitations, I would appreciate if the authors could provide additional clarification on the following points:

- What are the key contributions of this work beyond the theoretical analysis itself?
- What is the rationale for focusing exclusively on Google's SynthID-Text method rather than conducting comparative analysis across multiple watermarking approaches?
- How do the findings generalize to other watermarking schemes?

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
3

### Summary
The paper studies the theoretical properties of the SynthID-Text watermarking method. The trend of TPR at a fixed FPR is analyzed asymptotically under different scores. The theoretical findings are verified empirically, and a layer inflation attack is designed to break the watermark.

### Strengths
The theoretical properties of the SynthID-Text under different scores are carefully analyzed. Empirical results validate the theoretical finding.

### Weaknesses
1. The paper focuses on the theoretical analysis of a specific watermarking method. The scope is relatively limited

2. The theoretical analysis relies on the independence of $g_{t,\ell}$ (e.g., the central limit theorem for equation (10)). The random seed generator is determined by the recent $H$ tokens, which makes the random score correlated with previous tokens. In this case, there should be more explanation on why the scores for different positions, e.g., $g_{t,\ell}$ and $g_{t+1,\ell}$, are still independent.

3. The theorems are listed one after another, with not much explanation of the implications or ideas of the proof, making the paper hard to read.

Some minor issues

1. In the review of SynthID-Text, how to handle ties is not explained. In the example given in Figure 1, there are a lot of ties.

2. Some acronyms are used without explanation, e.g., without loss of generality (Wlog) in line 228, which could negatively affect the readability of the paper, considering the general audience of the conference.

3. Should line 177 come after line 178?

4. In the empirical validation of the results, is it possible to calculate the theoretical value of TPR and compare with the empirical value?

### Questions
Please refer to weaknesses

### Soundness
2

### Presentation
2

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
This paper presents the first theoretical study of Google DeepMind’s SynthID-Text, a production-scale watermarking framework for large language models (LLMs). SynthID-Text employs a tournament-based sampling mechanism to embed and detect watermarks during text generation. The authors provide a formal analysis characterizing its detection performance, robustness, and potential vulnerabilities, supported by some experiments on a real dataset.

### Strengths
1.	The paper tackles a timely topic, theoretical understanding of Google’s SynthID-Text, the first production-scale watermarking system for LLMs—providing valuable insights into its detection behavior and robustness.
2.	The theoretical results are connected to empirical observations, offering intuition about the differences between the mean-score and Bayesian detectors.
3.	The paper contributes novel analytical perspectives on how detection performance scales with the number of tournament layers and on the optimal g-value distribution, which could inform future watermark design and robustness analysis.

### Weaknesses
1. Experiments: 
    - The experimental evaluation is limited to a single model (Gemma-7B-IT) and one dataset (ELI5) with 1,000 short texts of 100 tokens, which falls short of the broader and more standardized benchmarks commonly used in recent watermarking studies.
	- Sections 5.2 and 5.3 lack sufficient quantitative results and visualizations to substantiate the claims (e.g., for the CLT assumption test and layer inflation attack).
2. Theory:
	- The analysis focuses exclusively on the non-distortionary version of SynthID-Text, leaving the distortionary setting unexamined. This limits the generality and practical relevance of the theoretical findings.
	- While the paper rigorously analyzes the mean-score and Bayesian-score detectors, it stops short of proposing improved or generalized detection strategies. Moreover, most derivations rely heavily on the Central Limit Theorem and remain relatively straightforward. The theoretical results are plausible but would benefit from deeper interpretive discussion and numerical validation.
3. Organization: The paper dedicates nearly five pages to definitions and preliminaries, including full restatements of Theorems 1 and 2 from prior work (Dathathri et al., 2024a). This space could be used more effectively to streamline the exposition and emphasize theoretical insights and experimental results.

### Questions
1. In Theorem 7, the deinition of $M$ is unclear to me. What exactly does $C_{M,t}=1$ signify, and how should readers interpret M in terms of the tournament dynamics or detection behavior? Furthermore, how to interpret all these theorems? Please provide more explanations.
2. Can the authors include numerical plots corresponding to Theorems 7 and 11 to confirm that the theoretical trends match the empirical TPR curves reported in Section 5?

### Soundness
2

### Presentation
2

### Contribution
2
