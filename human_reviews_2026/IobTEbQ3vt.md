# Bringing Stability to Diffusion: Decomposing and Reducing Variance of Training Masked Diffusion Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Masked diffusion models (MDMs) are a promising alternative to autoregressive models (ARMs), but they suffer from **inherently** much higher training variance. High variance leads to noisier gradient estimates and unstable optimization, so even equally strong pretrained MDMs and ARMs that are competitive at initialization often diverge after task-specific training, with MDMs falling far behind. Currently, there has been no theoretical explanation or systematic solution. In this paper, we derive **the first decomposition** of MDM training variance into three sources: {A} masking pattern noise, {B} masking rate noise, and {C} data noise -- while ARMs are only affected by {C}. This cleanly explains the fundamental training gap. Building on this foundation, we design six variance-reduction methods, including two core methods: (1) P-POTS, a **Pareto-optimal** $t$-sampler that minimizes training variance by sampling harder $t$ values more often with appropriately smaller update steps, and (2) MIRROR, which uses negatively correlated samples to reduce {A}. Experiments show that, compared to standard MDM training, our methods improve accuracy by **7–8\%** on complex reasoning tasks, while simultaneously reducing run-to-run variability to **near ARM levels**, substantially narrowing the gap with strong ARM baselines; in most settings, even the best baseline method runs remain below the worst run of our method.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper tackles the critical problem of high training variance in Masked Diffusion Models (MDMs), which causes optimization instability and performance degradation compared to Auto-Regressive Models (ARMs). The paper derives the first systematic decomposition of MDM training variance, identifying three distinct sources: A) Masking Pattern Noise, B) Masking Ratio Noise, and C) Data Noise. Based on this decomposition, the authors propose six methods, including two core techniques: P-POTS and MIRROR.  Experiments on complex reasoning and multimodal tasks show that the proposed methods, particularly the combination "P-POTS+MIRROR," dramatically improve performance (e.g., +7-8% accuracy on GSM8K) and, critically, reduce run-to-run performance variability.

### Strengths
1. This paper provides a clean, intuitive, and theoretically sound framework for understanding why MDMs are unstable.
2. The proposed core methods (P-POTS and MIRROR) are well-motivated and directly derived from the theoretical analysis. 
3. The significant improvement in the MDM's performance makes the proposed method not only theoretically sound but also practically effective.

### Weaknesses
This is a strong paper without major weaknesses. One suggestion on the presentation is that although this paper proposes two core methods to reduce the instability, it introduces six methods in total. I think this may harm the readability of this paper and distract the reader from understanding the core contribution. Therefore, the author may reconsider the paper structure.

### Questions
1. Based on my understanding, LLaDA and Dream seem to use slightly different formulations to train MDM. However, this paper only conducts experiments on LLaDA. Thus, are the proposed methods applicable to other baselines, such as Dream?
2. This paper discusses two types of LM, MDM and ARM, where MDM is fully bidirectional. How about the combined version, such as BlockDiffusion[1]? Does the author have any insights on this case, such as the training variance or whether proposed methods are still applicable?
3. Though experimental results are quite impressive, I still recommend that the author provide train-from-scratch results in the future since the training variance in pretraining might be more salient.


[1] Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models, in ICLR 2025

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
4

### Summary
This paper try to decompose the MDM's training noise. The noise is divided into three parts: data noise and two about masking.
The core analyses are about inter-group and inner-group decoupling by the law of the total variance. Based on the aforementioned analyses, two main approaches are proposed, P-POTS and MIRROR.

### Strengths
- Intuitive and sharp derivation of the theorem provides a mathemetically elegant  and practical explanation.
- Numerical pre-experiments seems robust and adheren to the expected Pareto frontier.

### Weaknesses
- Limited experiments about generalization and comparison. The ablaition experiments are mixed in the table of comparison. The included MDM baselines are too limited.
- Parts of the error bars are missing, and meanwhile, the error bars reported are too large to convince audience that the methods are consistently performing well as it's tested in the main table.

### Questions
- Selected benchmarks are all about QA reasoning tasks. How about general QA tasks and knowledge-intensive benchmarks, e.g., Graph QA, OCR detection?

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
2

### Summary
The paper analyzes training instability in masked diffusion models by decomposing the loss variance into three parts: (A) randomness from the mask pattern, (B) randomness from the mask ratio/timestep, and (C) data variance. To reduce these sources, it introduces P-POTS (a timestep sampler), Mirror (a complementary mask), and various other techniques. Across multiple benchmarks, the proposed approaches stabilize training and improve final results.

### Strengths
Overall, I think the paper studies an important problem. The proposed fixes P-Pot and Mirror are clearly argued and have demonstrated practical usefulness in terms of accuracy and training stability.

### Weaknesses
1. The loss-variance decomposition is insightful, but I believe for training stability it would be more insightful to analyze gradient variance. It would be nice to see how reductions in the proposed loss variances translate to reduced gradient variances and more stable optimization.
2. MIRROR roughly doubles the cost on some benchmarks compared to the baselines, which is quite expensive. Would MIRROR still be the best choice under a fixed time budget, which is a more practical scenario?
3. I am not familiar with MDMs and thus cannot comment on the commonly reported numbers. But for ARMs, the Qwen-2.5-7B-Instruct and Qwen-3-8B numbers appear below the commonly reported results. Could the authors clarify their evaluation approachs?

### Questions
The variance decomposition is not unique. It depends on the conditioning order when you iteratively apply the law of total variance. Could the authors discuss how your conclusions change under alternative decompositions, and whether those alternative decompositions could lead to other interesting approaches?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a theoretical and practical framework to understand and address instability in masked diffusion model (MDM) training. The authors first derive a variance decomposition framework that attributes training variance to three distinct sources: (A) masking pattern noise, (B) masking ratio noise, and (C) data noise. Based on this decomposition, they propose six variance-reduction methods. Experiments on multiple text reasoning datasets (GSM8K, HiTab, OpenScience) and a text-to-image benchmark show consistent improvements in both performance and stability. Overall, the paper is technically strong, though somewhat dense and in need of a tiny bit of clearer writing.

### Strengths
1. Strong theoretical foundation. The paper provides a clear and principled variance decomposition for masked diffusion model (MDM) training, unifying prior ad-hoc stabilization methods under a single theoretical framework. It then builds directly on this foundation by proposing six targeted variance-reduction techniques to mitigate the identified sources of instability.
2. Comprehensive empirical validation. The experiments cover both language and multimodal domains, demonstrating that the proposed methods are broadly effective and improve training stability across diverse settings.

### Weaknesses
1. Narrow comparison to ARMs. The study includes only two autoregressive baselines from the same family. Incorporating additional ARM baselines, especially models with different architectures or training paradigms, would help clarify whether the observed variance gap is a general phenomenon or specific to the chosen comparison set.
2. Limited model diversity and scaling analysis. While the empirical results are solid, they are restricted to a single MDM backbone (LLaDA-8B-Instruct). Evaluating the proposed methods across different model sizes and architectures would strengthen the claims of generality and potentially reveal whether the improvements follow any scaling trends within or across model families.
3. Writing and presentation. The exposition could be tightened to improve readability; the current density of equations and notation can make the paper feel more complex than necessary. Additionally, some figures could benefit from more informative captions, for instance, Figure 3 presents image generation results but omits the corresponding prompts.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
