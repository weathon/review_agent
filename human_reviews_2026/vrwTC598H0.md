# ESI: Epistemic Uncertainty Quantification via Semantic-preserving Intervention for Large Language Models

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Uncertainty Quantification (UQ) is a promising approach to improve model reliability, yet quantifying the uncertainty of Large Language Models (LLMs) is non-trivial. In this work, we establish a connection between the uncertainty of LLMs and their invariance under semantic-preserving intervention from a causal perspective. Building on this foundation, we propose a novel grey-box uncertainty quantification method that measures the variation in model outputs before and after the semantic-preserving intervention. Through theoretical justification, we show that our method provides an effective estimate of epistemic uncertainty. Our extensive experiments, conducted across various LLMs and a variety of question-answering (QA) datasets, demonstrate that our method excels not only in terms of effectiveness but also in computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
I have reviewed this manuscript in a previous conference. I tried to compare the current version and the previous version, but did not find significant changes. I tend to maintain my previous evaluations.

This manuscript presents a new framework that quantifies the epistemic uncertainty from the perspective of invariance to input intervention. The original prompt is intervened in a semantic-preserving manner, then the token-level distribution changes are quantified as an estimation of the epistemic uncertainty in the original answer. The effectiveness of the proposed approach is empirically verified by comparing to several baselines in multiple datasets. An ablation study is conducted to investigate the effect of different choices of algorithmic components.

### Strengths
1. The current manuscript is well organized. The motivation, methodology and experimental results are presented in a clear and understandable way.
2. The motivation that “the uncertainty of LLMs can be effectively quantified by evaluating the degree to which the model relies on semantic causal relationships in its inference process.” is interesting and inspiring. It provides a new perspective to help understand where the LLM uncertainty comes from.
3. The usage of Hellinger distance to measure the distribution distance is new. Previous works usually used KL divergence, but this work make a new choice with solid justification, from both theoretical and empirical perspectives.

### Weaknesses
1. One major limitation of this work is that when measuring the output distribution changes, it only considers the token-level change, without any analysis of the semantic meaning changes. Given the nature that same semantic meaning can be represented using different words/phrases, what really matters in LLM uncertainty measurement should be the change in semantic space. The current algorithmic design completely ignores this aspect. It would be more compelling if the variance after prompt intervention can be measured in semantic space.
2. The two proposed alternatives for prompt intervention have some inherent limitations. Regarding Skip-One-Char (SOC), skipping several unimportant characters does not really alter the superficial linguistic structures. Instead, it is testing the robustness of the LLM to miss-spelling of the words, which is not the true intention of the framework. Regarding the paraphrasing models, the additional computational cost is indeed expensive, if we want to have a reliable paraphrasing. Experimental results regarding the additional computational cost are missing.
3. The proposed metric is a response-level uncertainty metric, but the compared baselines are mostly entropy-based prompt-level uncertainty metrics, which are usually performing worse than response-level counterparts. SOTA response-level uncertainty metrics are missing from the current experiments, e.g., semantic density [1], Degree [2].

[1] Xin Qiu, Risto Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space, NeurIPS 2024

[2] Zhen Lin, Shubhendu Trivedi, Jimeng Sun, Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models, Transactions on Machine Learning Research, 2024

### Questions
1. As mentioned above in point 1 of “weaknesses”, only considering the token-level distribution shift seems to be insufficient for tackling language tasks, in which semantic meaning is clearly more important. Can you add more discussions regarding this point? Do you have any thoughts about how to compensate for this limitation?
2. As discussed in point 2 of “weaknesses”, the SOC is more like testing the robustness of the underlying LLM in tackling miss-spelled or distorted words, instead of superficial linguistic structures alterations. Could you elaborate more on how this potential issue can be mitigated? Can you design some empirical studies to investigate whether the above proposition is true?
3. In addition to the existing prompt-level baselines, more SOTA response-level uncertainty metrics should be included in the experiments to further verify the effectiveness of the proposed framework, e.g., semantic density [1], Degree [2]. Can you add more comparisons?
4. Since the main drawback of the paraphrasing models are their additional computational time, you should include them in the Table 2 as well, e.g., both the weak version and strong version. This will provide the readers with a clearer picture about what is the cost of having a stronger performance, and what are the tradeoffs between computation time and performance gain.

[1] Xin Qiu, Risto Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space, NeurIPS 2024

[2] Zhen Lin, Shubhendu Trivedi, Jimeng Sun, Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models, Transactions on Machine Learning Research, 2024

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes ESI, a grey‑box uncertainty score for LLMs that probes invariance to semantic‑preserving prompt interventions. Given a prompt $\mathbf{x}$, ESI decodes one answer $\mathbf{y}^\star$, generates paraphrase/character‑edit variants $\tilde{\mathbf{x}}$, and measures the token‑level shift in predictive distributions along the fixed path $\mathbf{y}^\star$ before versus after intervention (Hellinger distance on top‑$k$ logits). The authors argue this approximates an epistemic measure, and show strong AUROC and runtime gains versus Semantic Entropy, SAR, and other baselines across several QA datasets.

### Strengths
- Simple, fast, and effective: Teacher‑forced, token‑space distances make ESI markedly cheaper than output‑sampling methods while improving AUROC for error detection.

- Good empirical coverage: Multiple LLM families and both single‑answer and multi‑answer QA datasets.

- Clearly an intuitive method. Using semantic‑preserving input variants to probe model brittleness is conceptually clean.

- Well presented; I didn't know this area before reviewing and felt like I could follow along and learnt some things.

### Weaknesses
Due to my lack of familiarity with the area, I mainly focus on the properties of the derived estimator and the evaluations.

- The derivation of the estimator starts from a pairwise divergence over two independent semantic variants, then replaces it with a one‑sided expectation where every pair includes the original prompt as an anchor and distances are computed along the anchor’s decoded path. This changes the sampling measure in input space (all pairs now include the anchor) and output space (the conditioning path shifts to the anchor’s greedy decode). The assumption that the identity variant has non‑zero probability does not remove this bias. As derived, the resulting quantity does not appear to share the same functional as the stated pairwise measure.

- The estimator is derived from KL‑based EPKL, but the algorithm uses Hellinger distance on truncated (top‑k) token distributions with entropy weights; arguments that lean on KL‑style structure do not automatically apply. The object justified in the derivation and the object actually computed are different, so the current version of the paper does not establish properties of the implemented score.

- After the chain rule, expectations over model prefixes are approximated by distances along one decoded answer string, and in the final estimator this is fixed to the anchor prompt’s greedy decode. In high‑entropy or multi‑answer regimes, semantic variants can steer decoding into different modes; measuring only along the anchor path can hide dispersion that would be visible under paths typical for the variants. This makes the score optimistic on ambiguous items and blurs the line between epistemic effects and artifacts of the chosen path.

- Results are reported mainly as AUROC. AUROC alone does not characterize risk–coverage trade‑offs or calibration needed for selective generation and abstention. Without risk–coverage curves, AURC, selective accuracy at fixed coverage, it’s hard to judge how well ESI would perform in real decision pipelines.

### Questions
I’m not very familiar with the area and am happy to discuss my review with the authors in detail. The questions below focus on aligning the derivation with the implemented score and on a few evaluation checks.

- Can you align the distance used in the method with the distance used in the derivation (e.g., re‑derive the estimator in the same metric you implement), or justify the mismatch explicitly?

- Could you justify using AUROC alone and, if feasible, add risk–coverage curves, AURC, and selective accuracy at fixed coverage levels?

- Is there a way to quantify the estimator’s bias and variance via a small ablation (e.g., symmetric‑pair comparisons and repeated sampling)?

- Can you compute ESI over a few decoded paths per prompt and aggregate by mean and by max to assess path bias?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a measure of the variability in probabilistic predictions from a large language model (LLM) across augmentations of the LLM’s input. They evaluate how well this measure predicts the correctness of text generated by the LLM on question-answering datasets.

### Strengths
Originality: the proposed predictive-variability measure appears to be a new basis for predicting the correctness of LLM generations.

Quality: the authors compare to a good number of baseline methods.

Clarity: the paper is easy to follow at a low level.

Significance: predicting the correctness of model outputs warrants research attention.

### Weaknesses
I believe the technical core of this work is unsound.

- First, subjective model-based uncertainties alone are not a legitimate basis for assessing model reliability (Bickford Smith et al, 2025). Predictive uncertainty can be used as an estimator of predictive loss, but this does not require decomposing uncertainty.
- Second, the authors nevertheless appeal to an uncertainty decomposition that has been shown to be incoherent (Bickford Smith et al, 2025), and then reinvent the definition of epistemic uncertainty based on a questionable causal argument and a lot of arbitrary algorithmic decisions.
- Third, even if we forget all of this and just take the proposed measure of predictive variability at face value, it’s unclear how the variability measure reflects the paper’s emphasis on semantics as opposed to tokens. The measure is an average, over input augmentations and token indices, of an entropy-weighted distance between two token distributions. Since the same semantic content can be conveyed as many different token sequences, a fact that motivated key prior work (Farquhar et al, 2024; Kuhn et al, 2023), this is a conceptually flawed measure of predictive variability in semantic space, which is what we care about.

Now, maybe all that really matters is what works in practice. But I’m also unconvinced by the empirical evidence at hand. The authors are trying to solve a binary-classification problem: given an input, is the LLM’s generation correct? The use of AUROC to measure performance in this context has serious issues (Dyrland et al, 2022; Ferrer, 2025; Hand, 2009; Hand & Anagnostopoulos, 2021), so I cannot judge the practical use of the proposed method based on the numerical results provided.

---

Bickford Smith et al (2025). Rethinking aleatoric and epistemic uncertainty. ICML.

Dyrland et al (2022). Does the evaluation stand up to evaluation? A first-principle approach to the evaluation of classifiers. arXiv.

Farquhar et al (2024). Detecting hallucinations in large language models using semantic entropy. Nature.

Ferrer (2025). No need for ad-hoc substitutes: the expected cost is a principled all-purpose classification metric. TMLR.

Hand (2009). Measuring classifier performance: a coherent alternative to the area under the ROC curve. Machine Learning

Hand & Anagnostopoulos (2021). Notes on the H-measure of classifier performance. arXiv.

Kuhn et al (2023). Semantic uncertainty: linguistic invariances for uncertainty estimation in natural language generation. ICLR.

### Questions
How does your method compare to the baseline methods if you use performance metrics other than AUROC?

### Soundness
1

### Presentation
2

### Contribution
1
