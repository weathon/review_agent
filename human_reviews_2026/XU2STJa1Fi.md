# Mechanistic Detection and Mitigation of Hallucination in Large Reasoning Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Large Reasoning Models (LRMs) have shown impressive capabilities in multi-step reasoning tasks. However, alongside these successes, a more deceptive form of model error has emerged—**Reasoning Hallucination**—where logically coherent but factually incorrect reasoning traces lead to persuasive yet faulty conclusions. Unlike traditional hallucinations, these errors are embedded within structured reasoning, making them more difficult to detect and potentially more harmful. In this work, we investigate reasoning hallucinations from a mechanistic perspective. We propose the **Reasoning Score**, which quantifies the depth of reasoning by measuring the divergence between logits obtained from projecting late layers of LRMs to the vocabulary space, effectively distinguishing shallow pattern-matching from genuine deep reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA dataset and identify two key reasoning hallucination patterns: early-stage fluctuation in reasoning depth and incorrect backtracking to flawed prior steps. These insights motivate our **R**easoning **H**allucination **D**etection (**RHD**) framework, which achieves state-of-the-art performance across multiple domains. To mitigate reasoning hallucinations, we further introduce **GRPO-R**, an enhanced reinforcement learning algorithm that incorporates step-level deep reasoning rewards via potential-based shaping. Our theoretical analysis establishes stronger generalization guarantees, and experiments demonstrate improved reasoning quality and reduced hallucination rates.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the challenge of detecting and preventing reasoning hallucinations, which can be described by factually wrong but logically coherent/convincing chain-of-thought traces typically produced by large reasoning models (LRMs). First, a reasoning score is proposed based on the assumption that hallucinations relate to shallow reasoning, which can be characterized by static activations (i.e., low entropy) in the late Transformer layers. Based on empirical analysis, we see that the reasoning score alone cannot predict hallucinating steps/traces. To this end, different patterns are characterized that aim to describe different hallucination effects, such as shallow pattern matching and consecutive verification (#1), incorrect backtracking to earlier hallucinated steps (#2), and overthinking steps with both high reasoning score and perplexity (#3). The final reasoning hallucination detection (RHD) algorithm combines all derived metrics. Finally, the reasoning score is included in GRPO to mitigate hallucinations (dubbed GRPO-R). Experimental results on a novel benchmark (ReTruthQA) show superior performance in detecting hallucinations using RHD. Moreover, applying RL with GRPO-R improves overall performance on reasoning benchmarks (MATH500, AIME, GPQA).

### Strengths
Overall, this paper aims to address a crucial and indeed difficult problem in LRMs. I see the main strengths in:

1. The paper covers quite a large scope, starting from introducing a novel benchmark (ReTruthQA), over to an atomic reasoning score, a derived reasoning hallucination detection algorithm (RHD), and finally a method to mitigate reasoning hallucinations (GRPO-R). This is also reflected in the very comprehensive appendix. 
2. The approach seems to be effective in both detecting hallucinations and improving the reasoning performance. 
3. The paper is well written.

### Weaknesses
My main reservations are concerned with ambiguities in defining and quantifying hallucinations. 

1. Potentially ambiguous evaluation. As there are apparently no datasets available on reasoning hallucinations, this paper proposes a self-generated one (ReTruthQA). As described in appendix D, traces are labelled based on final reasoning outcomes, GPT-4o-Mini, and two human validators. No labeling accuracy is provided (e.g., based on a subset that has been even more thoroughly labelled with more models and human validators). This makes both the justification of the reasoning score and the comparison to other methods difficult. For example, GPT-4o as an LLM-as-Critic (LCM) has clearly non-perfect AUC in Table 1. However, it is used as a labeling method in section 3 (Figures 3 and 4). 
2. Heuristic detection mechanism. Clearly, the proposed reasoning score alone is not sufficient to detect hallucinations (is a high or low score desirable?). Hence, derivatives (statistical measures, relationships to attention and perplexity) had to be introduced, requiring many hyperparameters. While the reasoning score seems not to be effective, it is still included in the Reasoning Hallucination Detection (RHD). It is questionable how much the average reasoning score can help. Indeed, it is often not even activated by setting $\alpha_1$=0. The questionable impact of the average reasoning score is also shown in Table 5 (sometimes better scores without it in R1-14B) and Figure 8 (the highest drop with increasing weight in the reasoning score). One could even question why we don’t use the inverse of the reasoning score as well. 
3. Task-, model-, and metric-specific hyperparameters. As described in Appendix J, the proposed RHD uses specific hyperparameters for every task, model, and even metric (AUC, MC1-3). Did a similar hyperparameter tuning go into the baselines? 
4. Unclear gain of GRPO-R. While GRPO-R seems to improve the accuracy on the standard task, an analysis of the actual reasoning hallucination reduction is missing. It is not clear where the gain in accuracy comes from.

### Questions
I would appreciate if the rebuttal could address the individual weaknesses. Besides, Ref. (Valmeekam et al.) is missing the date.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a simple extension of information theorectic metrics to measure hallucination in reasoning outputs. Particularly these use the Jensen Shannon Divergence between the vocab distributions at intermediate and final layer. The authors show some validations on how this score correlates with accuracy, perplexity and shows how this score performs better than baselines based on Entropy/EigenScore or PRMs.

### Strengths
The motivation and general writing in the paper is clear & it is a relevant problem to solve

### Weaknesses
- The main hallucination score seems to be overtly simple. I feel this is more like the standard information theoretic measures Like entropy being rehashed for reasoning outputs (which typically have a sequence of steps). And like entropy, perplexity etc I feel these kinds of token level metrics would be noisy and would suffer from capacity bottleneck and robustness issues (I mean in many cases reasoning fallacy or logical inconsistency may not be associated with a high JS Divergence at the token level. This would probably only capture some of the simpler more obvious hallucinations. 

- Also, I feel computing such a token level score at each intermediate layer of the model can be even more noisy. The only analysis done on the reliability of these scores in Table 3 and 4 does not seem enough.
    - Validation of GSM-Noop - Introducing Noops are one of the simplest kinds of injected hallucinations, 
    - Validation on Stable/Rising sets: Not clear what is the size of the sets. This shows overall there is some correlation between the score, the accuracy and the perplexity but that can be from the fact that it is capturing some of the lower hanging hallucinations. 

- The main hallucination scoring is also somewhat heuristical. There are multiple hyperparameters involved (one in Attention Score, 4 in the final hallucination score, \tau etc). This makes it more tricky to be practically used. The authors need to explicitly show through their experiments whether it is sensitive to these hyperparameters or not.

- How can we definitely say that GRPO-R “encourages deep—but not excessive—reasoning during RL fine-tuning”  — the R-score no matter what is a noisy proxy. I would assume this scoring would be way more noisy and also varying (not robust), sensitive to minor changes in token space. The experiment results in table 2 are also not convincing. Esp for Qwen. Doing this on the small model 1.5B may not be enough

- Why are GRPO related experiments and R-Score based RHD experiments done on separate benchmarks (ReTruthQA and GPQA). This makes the conclusion a bit disjoint from each other. Why can’t the detection and mitigation be applied on the same benchmarks. 

- Overall the scoring seems like a slightly better strategy on avg than the baselines but whether it is really generalizable or robust, it’s not clear from the experimental results. Practically speaking too many hyperparameters would make usage of this too unstable and complex - so it is important for authors to show how sensitive it is to hyperparams.

### Questions
See the weakness section

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Reasoning Score (RS) to tackle the problem of reasoning hallucinations in Large Reasoning Models (LRMs). RS can quantify reasoning depth by analyzing divergence in late-layer logits, then distinguish shallow pattern-matching from deeper reasoning. By applying RS to the ReTruthQA dataset, two hallucination patterns are identified, including early fluctuations in reasoning depth and incorrect backtracking to flawed prior steps. With these findings, the authors propose Reasoning Hallucination Detection (RHD) framework to achieve state-of-the-art performance, and a GRPO-R approach that can integrate step-level reasoning rewards for better generalization and reduced hallucination rates.

### Strengths
1. The problem tackled in this paper, i.e., reasoning hallucinations, is crucial in modern language models and corresponding downstream tasks.

2. The patterns identified in this paper are important to address the hallucination issues.

3. The presented results demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. The experimental results are mainly collected from the DeepSeek series, which cannot well demonstrate the generalizability of the proposed method.

2. Many hallucinations can be made by the lack of factuality of language models, and there has been some previous work investigating this topic. Technically speaking, these approaches also adopt (supervised) fine-tuning plus GRPO-like algorithms to solve the problem. Compared with them, what are the advantages possessed by the proposed method?

3. Some notations, e.g., $R_{final}$, are not explained. I suggest the authors list all parameters and notations in a Table in the appendix.

4. How does the hyperparameter, $\gamma$, influence the performance of the proposed method?

5. How do models refined by RHD perform in reasoning on OOD domains or datasets?

### Questions
1. How does RHD perform when applied to reasoning models other than R1?

2. Many hallucinations can be made by the lack of factuality of language models, and there has been some previous work investigating this topic. Technically speaking, these approaches also adopt (supervised) fine-tuning plus GRPO-like algorithms to solve the problem. Compared with them, what are the advantages possessed by the proposed method?

3. How does the hyperparameter, $\gamma$, influence the performance of the proposed method?

4. How do models refined by RHD perform in reasoning on OOD domains or datasets?

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
This paper studies reasoning hallucinations in large reasoning models, where models produce coherent but incorrect reasoning. It introduces a Reasoning Score based on internal layer divergences to measure reasoning depth and distinguish shallow pattern matching from real reasoning. Using this metric, the authors identify three hallucination patterns and propose the Reasoning Hallucination Detection framework and GRPO-R reinforcement learning method, which together improve reasoning accuracy and reduce hallucinations across multiple benchmarks.

### Strengths
1. The work analyzes internal layer dynamics. It also bridges interpretability and reasoning reliability.
2. The work combines analytical diagnosis (RHD) with actionable intervention (GRPO-R) into a full pipeline.

### Weaknesses
1. The work lacks a validation/ablation study of the later-layer divergence correlating with reasoning depth.
2. Layer-wise JSD across all tokens is computation-intensive, which may be limited when scaling to larger models.
3. The work only focuses on Qwen series models. There is no study on other model series, such as the Llama.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
