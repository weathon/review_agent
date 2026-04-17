# Provable Guarantees for Flow-Based Generative Models in Time Series

- Decision: Reject
- Scores: 6, 6, 2, 6

## Abstract
Recent studies suggest utilizing generative models instead of traditional auto-regressive algorithms for time series forecasting (TSF) tasks. These non-auto-regressive approaches involving different generative methods, including GAN, Diffusion, and Flow Matching for time series, have empirically demonstrated high-quality generation capability and accuracy. However, we still lack an appropriate understanding of how it processes approximation and generalization. This paper presents the first theoretical framework from the perspective of flow-based generative models to relieve the knowledge of limitations. In particular, we provide our insights with strict guarantees from three perspectives: Approximation, Generalization and Efficiency. In detail, our analysis achieves the contributions as follows:
* By assuming a general data model, the fitting of the flow-based generative models is confirmed to converge to arbitrary error under the universal approximation of Diffusion Transformer (DiT).
* Introducing a polynomial-based regularization for flow matching, the generalization error thus be bounded since the generalization of polynomial approximation.
* The sampling for generation is considered as an optimization process, we demonstrate its fast convergence with updating standard first-order gradient descent of some objective.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a unified theoretical framework for flow-based generative models specifically tailored for time-series generation and forecasting, providing three key classes of theoretical guarantees. Firstly, for Approximation, the authors demonstrate, by leveraging the universal approximation properties of Diffusion Transformers (DiT), that the model class can approximate the optimal conditional flow with arbitrary precision. Secondly, concerning Generalization, the introduction of a polynomial-basis regularization for the conditional flow results in an explicit upper bound on the generalization error, which is established under a noisy time-series model. Finally, in terms of Efficiency, the paper establishes convergence rates for the sampling process by casting it as a first-order optimization procedure under specific smoothness and regularity assumptions. Overall, the work's primary ambition is to be the first to deliver a comprehensive, end-to-end theoretical justification, covering approximation, generalization, and sampling convergence, for modern flow-based time-series models.

### Strengths
1. **Motivation.** The paper targets a well-defined and important gap: the theory of generative models for time series lags behind rapid empirical progress. Clarifying approximation, generalization, and convergence is valuable.

2. **End-to-end scope.** Addressing three pillars, expressivity, generalization, and sampling efficiency, in a single framework is ambitious and conceptually clean.

3. **Theoretical insights.** While some technical subtleties are intricate, the insights are interesting and broadly useful for understanding the capabilities and limitations of flow-based models. This line of work can help ground future scaling of time-series generative algorithms.

### Weaknesses
1. **Lack of empirical validation.** No experiments are provided to indicate whether the bounds are numerically valid or to illustrate the effect of polynomial regularization. Even a small synthetic study could substantially improve clarity and persuasiveness.

2. **Incrementality in Section 5.** From a non-expert perspective in approximation theory, the DiT-based universality result may read as an application of known transformer approximation theorems rather than a fundamentally new approximation insight specific to this setting.

3. **Single-dataset formalism.** The guarantees are presented for a single-distribution (single-dataset) setup. In the era of large models trained across multiple datasets, it would be helpful to discuss limitations or extensions when the model must handle mixture distributions or dataset shifts; the current framework does not directly answer these multi-dataset questions, although this does not diminish the novelty of the presented results.

### Questions
- **Notation in Lemma 6.1.** Can the author please clarify the definition of `\tilde{f}` and how it differs from the standard `f` ?

- **Scope beyond time series.** Are the results inherently tied to a regression-style time-series setting, or can the analysis (with adjusted assumptions) extend to other modalities/tasks (e.g., language modeling with logistic outputs)? A brief, intuitive paragraph detailing the differences and limitation could improve the paper quality.

- **Polynomial regularization in practice.** Is the polynomial-basis regularization intended primarily as a proof device, or do the authors advocate its practical use? If the latter, guidance on basis selection, order choice, and expected computational overhead would be valuable as well as empirical evidence as mentioned in the weaknesses.

- **Task scope.** Do the guarantees apply only to forecasting/imputation, or do they extend to unconditional generation ?

### Soundness
3

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
2

### Summary
This paper provides the first theoretical framework analyzing flow-based generative models for time series forecasting (TSF) from approximation, generalization, and efficiency perspectives.  This work aims to provide rigorous theoretical guarantees for understanding how flow-based models achieve approximation and generalization in TSF tasks.

### Strengths
1. The paper tackles a genuine need for theoretical understanding of generative models in TSF. While empirical success has been demonstrated, the lack of theoretical guarantees for approximation, generalization, and efficiency is a significant limitation that this work attempts to address.

2.  The framework covers three fundamental aspects (approximation, generalization, efficiency) providing a holistic theoretical treatment rather than focusing on a single dimension, which is valuable for complete understanding.

3.  Introducing polynomial-based regularization to bound generalization error is a concrete, actionable contribution that bridges theory with potential practical implementation.

### Weaknesses
1.   The paper appears to be purely theoretical without experiments validating the theoretical predictions. Do the approximation bounds, generalization bounds, and convergence rates hold in practice? Without empirical validation, the practical relevance of the theory is unclear.

2. The paper claims convergence to "arbitrary error" but doesn't discuss whether the bounds are tight or loose. Are the theoretical guarantees practically meaningful, or do they only hold asymptotically with unrealistic resource requirements?

I must note that I am not deeply familiar with the theoretical analysis of flow-based generative models and their application to time series forecasting.  I may be missing important theoretical nuances or standard conventions in this subfield, My review should be weighted accordingly or potentially disregarded if it conflicts with expert opinions..

### Questions
Even if the theory is sound, what actionable insights does it provide? How should practitioners use these results to design better models, select hyperparameters, or understand limitations?

### Soundness
3

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
The author proposes a novel theoretical framework to verify the approximation, generalization and efficiency of flow-matching method within the time series generation task.

### Strengths
1. The motivation is clear and easy to follow, less typos
2. The background and related work are enough
3. The  paper gives a novel perspective for time series generation task.

### Weaknesses
1. As stated in lines 78-80, the paper propose to ensure robustness against noise and distribution shifts, where are the proofs to verify the robustness of distribution shifts of time series analysis, can author give some experiments ?

2. As stated that "the fitting of the flow-based generative models is confirmed to converge to arbitrary error under the universal ap-
proximation of Diffusion Transformer (DiT)", can author do the DiT structure-based flow model to verify this claim, such as make the comparison with Diffusion-TS, which is described in line 220.

3. As stated that "Orthogonal polynomial bases" is more stronger approximating ability,  can author make some experiments to verify the effectiveness?

### Questions
please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
I am very sorry, but I don't have the background to be able to understand this paper. I've tried to at least connect the main theorems, but even that it is very hard. According to the abstract the paper provides a formal framework that allow them to prove guarantees related to approximation, generalization and efficiency. If correct, it would be a valuable contribution.

### Strengths
Providing formal guarantees for complex machine learning is an important problem.

This is a very formal paper, under the assumption that it is correct, this is a strength.

### Weaknesses
The paper is not accessible for people without deep mathematical understanding of the topic (this could of course be perfectly fine).

It is hard to get an intuitive sense for the formal framework and theorems.

There are many transformations between the many lemmas and theorems which makes it hard to even trace the connection between them.

### Questions
The framework is focused on time series models, would it be possible to use the same framework for non-sequential models?

If I understand correctly, the method is based on diffusion models, what would it take to adapt it to transformers?

### Soundness
3

### Presentation
2

### Contribution
3
