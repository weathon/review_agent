# An Ensemble Framework for Unbiased Language Model Watermarking

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
As large language models become increasingly capable and widely deployed, verifying the provenance of machine-generated content is critical to ensuring trust, safety, and accountability. Watermarking techniques have emerged as a promising solution by embedding imperceptible statistical signals into the generation process. Among them, unbiased watermarking is particularly attractive due to its theoretical guarantee of preserving the language model's output distribution, thereby avoiding degradation in fluency or detectability through distributional shifts. However, existing unbiased watermarking schemes often suffer from weak detection power and limited robustness, especially under short text lengths or distributional perturbations. In this work, we propose ENS, a novel ensemble framework that enhances the detectability and robustness of logits-based unbiased watermarks while strictly preserving their unbiasedness. ENS sequentially composes multiple independent watermark instances, each governed by a distinct key, to amplify the watermark signal. We theoretically prove that the ensemble construction remains unbiased in expectation and demonstrate how it improves the signal-to-noise ratio for statistical detectors. Empirical evaluations on multiple LLM families show that ENS substantially reduces the number of tokens needed for reliable detection and increases resistance to smoothing and paraphrasing attacks without compromising generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the problem of watermarks for language models. A watermark is a statistical signal hidden inside text that can be detected by anyone with access to a secret key but is intended to not distort the quality of text, thus remaining undetectable to observers without access to a secret key. Many schemes have been instantiated for autoregressive models by modifying the sampling procedure of models, with a prominent such scheme being the green list approach, where a hash function looks at the recent context and returns a pseudorandom subset of the vocabulary to upweight in generation.  This scheme proposes to construct an ensemble of watermarks by recursively applying a key-based reweighting scheme with different keys to the distribution of the next token in order to inject additional detection power.  The authors then instantiate their ensemble based approach with a variety of watermarking schemes and demonstrate empirical efficacy in improving watermark detectability and robustness after carefully selecting their hyperparameters.

### Strengths
One strength of this paper is that they identify a new axis for improving the dectectability and robustness of language models.

### Weaknesses
First, the authors incorrectly suggest that SynthID requires $2^{30}$ redundant tokens to generate a single token.  As the authors of that paper state clearly in the methods and in the appendix, they apply a vectorized approach to tournament sampling in practice that does not induce this redundancy.

Second, I am a little confused about the additional novelty of the present work with respect to tournament sampling.  While I agree that synth-id uses a particular instance of the ensembling framework introduced in definition 1, I think more clear differentiation could be described.

Third, the statement on lines 204-205 about unbiasedness ensuring indistinguishability is not true.  See, e.g. *Black-box detection of
language model watermark* by Gloaguen et al 2025.

Fourth, the fact that there is non-monotonic improvement with $n$ is a bit unfortunate and seems like it is an artifact of the precise way that the elements of the ensemble instantiate the watermark, i.e. by using some fraction of the vocabulary as a greenlist which then becomes exponentially small in intersection.  This does not seem like a fundamental barrier but rather a specific pathology of the greenlist paradigm.  Tournament sampling for example does not suffer from this failure mode.  I think the authors should discuss this.

Fifth, I am curious as to the additional sampling time overhead required to implement this approach, as well as how this scales as a function of $n$.

Sixth, I think this paper can benefit from comparison with additional related work, such as those watermarks that are imbedded directly into the model weights, e.g. *GaussMark: A Practical Approach for Structural Watermarking of Language Models* by Block et al 2025.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes ENS, an ensemble framework that enhances logit-based unbiased watermarks by amplifying the watermark signal. The idea is simple: each logit-based unbiased watermark can be viewed as a distribution reweighting function $f_i$. By ENS, different distribution reweighting functions are nested, i.e., $f_i( f_{i-1} (\cdot) )$. Intuitively, the nested function is also unbiased but enhance watermark strength.

### Strengths
1. The idea is easy to implement but effective. 
2. The authors conduct different variants of ENS-enhanced watermarks, which all show the efficacy of ENS according to the experimental results. Specifically, the authors show the unbiasedness, robustness, and higher detectability via ENS.

### Weaknesses
1. Potentially, if an unbiased watermark is enhanced, it is possible that the probability distribution is altered too much under certain watermark keys. Therefore, the method may not perform well in low-entropy scenarios. However, since this is not the focus, the authors could discuss this limitation in future work.

### Questions
1. The authors can elaborate further on Section 4.3. Specifically, what is the effect of $n$? The authors argue that in practice, it is important to select $n$ close to the optimal by balancing the tradeoff between aggregation gain and sparsity loss. The experimental results also suggest that an intermediate $n$ achieves better performance. Although I generally understand the idea that a large $n$ reduces the detectability but not significantly, the analysis in Section 4.3 is not sufficiently clear. 
2. From Table 1, the TPR@0.1%FPR score of ENS-Dipmark seems to achieve its optimum when $n=5$. Theoretically, I am also curious about the performance gain bound by ENS. Could the authors provide some analysis of the improvement by ENS? For example, specifically for Dipmark, what is the expected gain in detectability for $n=2,3,\cdots$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ENS, a general ensemble framework for logits-based unbiased watermarking in LLMs. The key idea is to sequentially compose multiple unbiased watermark instances (with independent keys), amplifying the watermark signal while theoretically preserving unbiasedness. The authors prove that ensemble compositions remain unbiased and show detection signal scales as $\sqrt{n}$ with ensemble size. The authors also conducted evaluations on multiple LLM families and datasets to validate their algorithms.

### Strengths
1. Clear Motivation: ENS addresses the main weakness of unbiased watermarks: weak detectability at short lengths.

2. The proof of unbiasedness is rigorous, with extensions on independence assumptions and variance scaling.

3. The experiments are comprehensive. The proposed method is compared against strong unbiased baselines e.g. SynthID. The experiments cover both detectability and robustness, using realistic paraphrasing and back-translation attacks.

4. The experimental results are strong. There are significant TPR@FPR gains of ENS. Besides, ENS-MCMark achieves state-of-the-art robustness and detectability across attacks.

### Weaknesses
1.	ENS requires multiple independent keys per generation step, which may introduce storage, distribution, and synchronization complexity in real deployment.
2.	Detection power declines for large ensemble sizes, aligning with theory but reducing scalability. Choosing optimal n becomes another hyperparameter to tune.

### Questions
Do you have any advice for tuning the ensemble size?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes, ENS, that composes several unbiased, logits-based watermarking steps keyed independently. The authors prove the composition stays unbiased in expectation and analyze how aggregating per-key detectors changes the testing signal. Empirically, they report improved detection and robustness across multiple model families, with little to no degradation in standard text quality metrics. However, I think the paper only studies the effects on standard generation metrics. 

The mains claims are as follows
- Unbiasedness under composition. If a single reweighting rule is unbiased, the n-fold ensemble remains unbiased when keys are i.i.d.; the proof is a short tower-property induction.
- Detection scaling. With per-key scores that are (approximately) independent and bounded, the aggregate statistic has mean proportional to n and variance proportional to n, implying SNR ≈ (µ/σ)√n and p-values that improve with n.
- Trade-off: Under “intersection-at-generation,” the per-step effect shrinks like (εγ)^n, so the exponent in the p-value bound behaves like n(εγ)^2n with an optimum n⋆≈1/[2log(1/(εγ))].

### Strengths
- The paper cleanly separates two questions: preserving unbiasedness and recovering detection power. Theorem 4.2 (composition stays unbiased) is simple and well scoped; I didn’t find hidden caveats beyond key independence.
- The SNR and Hoeffding arguments are standard but appropriate for bounded per-key scores (e.g., DiPmark). The text is careful to say the exponential improvement can attenuate in practice.
- Results include multiple LM families and several corruption settings (paraphrasing, back-translation, token replacements). The tables make it plausible that the ensemble generally helps and that quality remains close to baseline.

### Weaknesses
- The SNR and p-value scaling assume per-key independence; the paper notes dependencies via overlapping n-grams and proposes bypassing repeats, but I didn’t see a quantitative study of how correlations impact power. This matters because the theoretical √n gain can compress substantially with even mild correlation. 
	- Add a small study where you control n-gram overlap or reuse keys to measure empirical correlations among per-key scores and the resulting deviation from √n SNR. This would tell readers when the ideal scaling is trustworthy.
- The trade-off section is framed for strict intersection; many practitioners would avoid hard intersections and instead add small centered logit shifts per key. It would help to show the same analysis (or an empirical proxy) for a soft/additive design where the (εγ)^n collapse is muted
- The paper uses an aggregate statistic (one test), which avoids multiple-testing corrections, but it would help to explicitly say how thresholds are set to fix FPR across n (so readers don’t assume Bonferroni is needed).
- Your theory predicts where detectability should peak.  Show an experiment that sweeps the relevant parameters so readers can see whether the peak occurs where theory says it should.
- Recent work suggests watermarking can change model behavior beyond detectability/quality tradeoffs: Downstream Trade-offs of a Family of Text Watermarks (Ajith et al., EMNLP 2024) finds 10–20% drops on downstream tasks even for “unbiased” schemes like KGW; WaterJudge (Molenda et al., NAACL 2024) quantifies a detectability–quality trade-off; and Watermarking Degrades Alignment in Language Models (Verma et al., ICLR 2025) shows shifts in truthfulness/safety/helpfulness that can be partially mitigated with an external reward model. The paper would be stronger with a targeted experiment evaluating how ensembling affects alignment-relevant metrics (e.g., reward scores ), and a short discussion of these works to contextualize potential side effects.



### Nits
- Define ε and γ at first mention in the trade-off subsection; they appear earlier in the derivation than in the surrounding text.
- Line 87 rejected-sampling , typo?
- Line 119 putative text sequence, typo?
- Line 190 DETECT EFFICIENCY -> Detection Efficiency?

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2
