# LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Retrieval-augmented generation (RAG) has become a popular approach to improving large language models (LLMs), yet trustworthiness remains a central challenge: models may produce fluent but incorrect answers, and retrieved context can amplify errors when irrelevant or misleading. To address this, we study how model internals reflect the interplay between parametric knowledge and external context during generation. Specifically, we ask: (1) can the correctness of a model’s output be inferred directly from its internal activations, and (2) do these internals reveal whether external context is helpful, harmful, or irrelevant? We introduce metrics grounded in intermediate activations to capture both dimensions. Across six models, a simple classifier trained on hidden states of the first output token predicts output correctness with nearly 75% accuracy, enabling early auditing. Moreover, our internals-based metric substantially outperforms prompting baselines at distinguishing between correct and incorrect context, guarding against polluted retrieval. These findings highlight model activations as a promising lens for understanding and improving the reliability of RAG systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether LLMs internal activations can be used to predict the correctness of generated outputs and to assess the utility of external context. The authors utilize a simple classifiers trained on different types of inputs to predict correctness of the model predictions and show that they can predict correctness with about 75% accuracy. Further, they estimate the efficacy of external context
along w.r.t. correctness and relevancy.

### Strengths
Simple and effective approach to generate signal about the correctness in generative tasks.

Introducing new metric for external context validation based on internal states.

The methodology is carefully described and clear.

### Weaknesses
I think table 1 adds nothing useful. Besides, to me it seems obvious that LogitLens, TunedLens, and HiddenStates should have close performance since they are all different transformation of the hidden states. Also since there is no specific preference between these three maybe you could just keep one and bring the others in the Appendix as further exploration of different settings.

Please consider modifying the axes ticks and labels in Fig 3 to 6

### Questions
I was wondering whether it is possible to directly using the FFN or Attn outputs instead of HiddenStates or PKS etc.

Did you analyse the sensitivity of the results to the parameter $\lambda$?

Did you try more complex classifiers to check if you can further improve the truthfulness detection?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper develops techniques to use model internals (hidden states, activations, etc.) to predict if a model’s answers are correct and assess the utility of retrieval-augmented context. They study these techniques on multiple different models and show their approach outperforms prompted evaluations.

### Strengths
The paper is well-written and develops techniques that improve on prompted baselines to assess answer correctness and context utility. They apply this technique to several different small open-source models for TriviaQA and MMLU.

### Weaknesses
While developing calibrated measures of model truthfulness is a critical space, there are multiple existing works that train classifiers on top of different model internals to assess answer correctness. It is unclear if this paper is adding something truly novel to the large existing body of work in this space. Several of these papers are referenced in the related work, but this paper does not benchmark their proposed method against these prior works. Additionally, although the baseline of prompting models to evaluate their correctness may not work well for the model sizes evaluated in the paper (<=13B), prompting alone may be sufficient for larger models (> 32B). Also, it is unclear if new classifiers need to be trained for each model, dataset pair to assess correctness – if so this may reduce the adoption of this technique. Overall the paper is suggesting a method that may not be too distinct from prior works and might not be relevant for models beyond a given size.

### Questions
- section 2: Some more related works: Language Models Can Predict Their Own Behavior (https://arxiv.org/pdf/2502.13329), LLM-Check: Investigating Detection of Hallucinations in Large Language Models (https://proceedings.neurips.cc/paper_files/paper/2024/file/3c1e1fdf305195cd620c118aaa9717ad-Paper-Conference.pdf), Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs (https://aclanthology.org/2025.acl-long.304.pdf)

- section 2: Your related work mentions several other “open-box internal-state approaches”. Beyond simple prompting techniques it would be good to include a few of these as baselines.

- section 3.3 line 161: why are the hidden states taken from only the first token position?

- section 3.4 line 168: why does confidence correlate with parametric knowledge used – provide more intuitions?

- section 4.2 line 239-240: strong alignment between the context tokens and generated answer doesn’t necessarily mean that the model is relying more on the context than on parametric knowledge – parametric knowledge itself may also align with the answer

- section 5 line 264: demonstrating this approach on MMLU and TriviaQA is a good starting point – adding 1-2 more datasets being more actively used in current literature such as GPQA would make the results more impactful.

- section 5 line 341: does a separate classifier need to be trained for every model / dataset pair? Can we train a single classifier that could work across several datasets for a given model? Doing so could make this technique more useful.

- section 5 line 322: Other works have shown that larger language models tend to be more calibrated than smaller language models. It’s possible that your prompting baselines may work well for larger models even though they don’t work for your scale. Experimenting with a larger model size (e.g. 32B+) could clarify this.

- section 6.1 line 398: why is PKS predictive of correctness for open-ended QA and not MCQ?

- section 8: Is there a single method that works well across models and datasets for assessing answer correctness and context utility? Explicitly noting this would be valuable for practitioners.

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
3

### Summary
The paper proposes LLM Microscope to predict whether a language models answer will be correct using internal activations alone and whether external content is helpful or harmful. For the first hypothesis they train light weight classifier over features derived from intermediate representations Logit-Lens statistics, first-token hidden states, and a Parametric Knowledge Score.
For the second hypothesis (regarding the external content) they introduce an internals based proxy for contextual log-likelihood gain by combining an External Context Score via a scaling parameter.
The authors show experiment across multiple models and show correctness can be predicted via internals at 75% accuracy and a high AUC, and internal based signals outperform prompting when distinguishing correct vs incorrect context.

### Strengths
1) The paper clearly formulates the problem and cleanly formulates contextual log likelihood gain and relative utility and instanties an internal based proxy for context efficacy. Definitions and rationale are explicit. 
2) Random forests over per-layer logit lens statistics and first token hidden states is a well setup experiment and yield strong correctness predictions with analysis of feature importance and layer-wise trends. The observation that internal layers are as or more predictive than the final layer is compelling and practically useful for early auditing. 
3) The authors perform extensive experiments that covers 6 models. Table 2 is important and shows internals-based features consistently outperform prompting baselines.

### Weaknesses
1) So the “Incorrect” contexts are produced by prompting GPT-4o to replace mentions of the gold answer with plausible but wrong alternatives, whereas “correct” contexts are sometimes GPT-4o-summarized to 500 tokens. This could introduce stylistic artifacts that models might pick up rather than genuine semantic correction signals. More diagnostics are needed to show that performance improvements are not due to such artifacts, such analysis is missing in the paper and is critical.
2) Many generative answers extend beyond a single token and while the paper acknowledges this in Limitations, most RQ1 features are computed at the first output token and some RQ2 averages are token-averaged but layer-averaged later.

### Questions
1) There is some inconsistency in the TriviaQA counts, in the experimental setup is states it retains 6557 questions but in D.1 the paper says quality filter yields 11683 examples.

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
4

### Summary
This paper provides an empirical study of how model internal states and other signals can be used for predicting: (i) if the model output is correct; and (ii) how much it relies on the external context. The results show that internal hidden states are effective at predicting correctness, and that features derived from external context matching and FFN activation strength are predictive of context relevance.

### Strengths
- The experiments are designed well and important aspects related to understanding LLM behaviors. The results are well presented and show clear trends in support of the claims made in the paper.
- The paper does a good job of bringing together multiple lines of research around mechanistic interpretability, confidence elicitation from LLMs, contextual faithfulness and factuality. The findings presented are useful for driving further research in these areas.

### Weaknesses
- All techniques in the paper are borrowed from prior works -- ECS and PKS scores (Sun et al, 2025), predicting factuality from hidden states (Azaria & Mitchell, 2023), verbalized confidence (Kadavath et al, 2022), Logit lens and Tuned lens. While the paper does a good job of contrasting and comparing them, it doesn't make any new methodological contributions.
- Consequently, a lot of the main conclusions in the paper are already well known -- e.g., that hidden states are predictive of factuality, ECS and PKS scores are effective at measuring context vs parametric knowledge utilization.
- Dealing with incorrect / conflicting contexts is a rich area with lots of papers. The paper misses out discussion of some important techniques, e.g., from ICLR 2025 [1].

[1] Huang, Yukun, et al. "To Trust or Not to Trust? Enhancing Large Language Models' Situated Faithfulness to External Contexts." The Thirteenth International Conference on Learning Representations.

### Questions
- Can you explain how the findings related to RQ1 here are different from those of Azaria & Mitchell (2023)?

### Soundness
3

### Presentation
3

### Contribution
2
