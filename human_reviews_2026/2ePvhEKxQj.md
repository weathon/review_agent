# Causal Reasoning Favors Encoders: Limits of Decoder-Only Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
In-context learning (ICL) underpins recent advances in large language models (LLMs), yet its role in causal reasoning remains unclear. Causal reasoning demands multi-hop composition and strict conjunctive control, and reliance on spurious lexical relations of the input could provide misleading results. 
We hypothesize that, due to their ability to project the input into a latent space, encoder- and encoder–decoder architectures are better suited for said multi-hop conjunctive reasoning versus decoder-only models. 
To do this, we compare fine-tuned versions of all the aforementioned architectures with zero- and few-shot ICL in both natural-language and non-natural language scenarios. We find that ICL alone is insufficient for reliable causal reasoning, often overfocusing on irrelevant input features. 
In particular, decoder-only models are noticeably brittle to distributional shifts, while fine-tuned encoder and encoder–decoder models can generalize more robustly across our tests, including the non-natural language split. 
Both architectures are only matched or surpassed by decoder-only architectures at large scales. 
We conclude by noting that for cost-effective, short-horizon robust causal reasoning, encoder or encoder-decoder architectures with targeted fine-tuning are preferable.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the role of in-context learning (ICL) in LLMs' causal reasoning. Authors make the assumption that encoder and encoder-decoder architectures are better fitted for causal reasoning than the decoder-only architecture. To validate this assumption, the authors conduct experiments on the fine-tuned models with diverse ICL prompt strategies. Authors conclude that ICL alone is not sufficient for reliable causal reasoning, and encoder-only models are more suitable for causal reasoning. The current version has some minor writing problems.

### Strengths
1. I think the studied problem is novel and meaningful. Because both ICL and causal reasoning are important aspects of LLMs, it is interesting to link them. In addition, the influence of architectures on the causal reasoning abilities is worthy of exploring.

2. The authors drew an interesting conclusion: the encoder-only architecture has more generalized abilities. I think this conclusion has the potential to guide the design of advanced reasoning models. 

3. The experiments are comprehensive, authors compare with diverse models. In addition, the experimental method is overall sound.

### Weaknesses
1. The first apparent weakness, I think, lies in the writing. For example, the first two paragraphs of the introduction are somewhat dispersed. I think authors can merge them into one paragraph, with the importance of causal reasoning coming first, then the ICL's influence on causal reasoning. 

2. Second, I believe the transition between the first two paragraphs and the third paragraph is abrupt. Specifically, how did the ICL's influence on the causal reasoning connect to the comparison between the decoder-only and encoder architectures?

3. There should be a comma after the "To test this" in row 081.

4. Since the main results (conclusion) are related to the architectures rather than ICL, I suggest authors revise their introduction into an architecture-centric style rather than the current ICL-centric style.

5.  I believe Figure 1 should be revised. Specifically, authors can demonstrate a more detailed creation of their dataset in (A). In addition, I didn't get the main idea of the authors' evaluation methods in (B).   

6. There should be different apparent sections in the related works section. For example, the LLM's causal reasoning part, the architecture part.

### Questions
Please refer to the weakness part.

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
2

### Summary
This paper investigates whether encoder-based architectures  outperform decoder-only LLMs  in causal reasoning.  Using a synthetic first-order-logic dataset (SimpleLogic), the authors compare zero/few-shot ICL and fine-tuned settings under both natural-language (NL) and randomized (NNL) variants.  Results show that encoder and encoder–decoder models generalize more robustly than small-to-medium decoder-only models, though GPT-5 reaches near-perfect accuracy at much higher computational cost.

### Strengths
1. Clear and reproducible experimental setup with OOD splits.
2. Systematic architectural comparison (encoder / decoder / hybrid) under unified prompts.
3. The NNL (“lexical ablation”) split provides a neat way to isolate structural reasoning from lexical bias.

### Weaknesses
1. “Causal reasoning” is operationalized as deterministic logic inference; no interventional or counterfactual dimension.
2. Direct label prediction is insufficient to measure reasoning; CoT or reasoning-path supervision is absent.
3. Findings largely reproduce known trends from mathematical reasoning benchmarks.

### Questions
1. Would chain-of-thought distillation or reasoning supervision close the encoder–decoder gap?
2. How sensitive are results to dataset tokenization or prompt template?

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
In this paper, the authors experimentally demonstrate that decoder-only language models exhibit weaker logical reasoning abilities compared to fine-tuned encoder and encoder-decoder models. They intuitively attribute this observation to the fact that encoder layers allow every token to integrate information from the entire sequence, thereby enhancing the model's multi-hop conjunctive reasoning capabilities.

### Strengths
1. Investigating the logical reasoning abilities of LLMs is an important research direction that can contribute to enhancing the interpretability and trustworthiness of LLMs.

2. The paper is clearly written and easy to follow.

3. The authors have made the code and data publicly available, which greatly facilitates reproducibility and further research.

4. The authors’ finding that decoder-only language models exhibit weaker logical reasoning abilities compared to fine-tuned encoder and encoder-decoder models is both intriguing and intuitively reasonable. Their understanding that models with an encoder possess stronger multi-hop conjunctive reasoning capabilities adds valuable insights.

### Weaknesses
1. In my opinion, the paper’s focus may not be entirely well-positioned, as the problem definition, dataset construction, and experimental validation all primarily center around logical reasoning, rather than causal reasoning. Given the fundamental differences between causal reasoning and logical reasoning [1], I believe it would be more precise and academically rigorous to reframe the study from causal reasoning to logical reasoning.

2. The paper proposes an interesting finding: decoder-only language models exhibit weaker logical reasoning abilities compared to fine-tuned encoder and encoder-decoder models, which I appreciate. However, as the authors mention, very large decoder-only models such as GPT-5 still demonstrate significant out-of-distribution (OOD) generalization abilities, despite their lower efficiency. A more impactful and forward-looking contribution would be to explore how to improve encoder models, or integrate concepts from encoder models into decoder-only models, aiming to reduce the complexity of LLMs while enhancing their logical reasoning capabilities.

[1] Hernán, M. A., & Robins, J. M. (2010). Causal inference.

### Questions
Please see Weaknesses.

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
This paper investigates the limitations of decoder-only large language models (LLMs) in causal reasoning tasks, arguing that encoder and encoder-decoder architectures are better suited for multi-hop conjunctive reasoning due to their ability to project inputs into a latent space and perform global information aggregation. The authors introduce a benchmark based on SimpleLogic, with natural-language (NL) and non-natural-language (NNL) test sets, and evaluate a range of models under zero-shot, few-shot, and fine-tuned settings. They find that fine-tuned encoder-based models outperform decoder-only models in efficiency and robustness, especially under distributional shift, while very large decoders (e.g., GPT-5) achieve high accuracy at significant computational cost.

### Strengths
1. The paper provides a clear and motivated comparison of architectural families for causal reasoning, with a well-designed dataset that controls for linguistic and structural variability.

2. The inclusion of both natural and non-natural language splits strengthens the evaluation of generalization and robustness.

3. The analysis is thorough, covering accuracy, depth-wise performance, label compliance, and computational cost.

### Weaknesses
1. The abstract could be more concise. For example, the sentence “We hypothesize that, due to their ability to project the input into a latent space, encoder- and encoder-decoder architectures are better suited for said multi-hop conjunctive reasoning versus decoder-only models” is somewhat verbose and could be streamlined.

2. The theoretical argument in Section 3.2, while intuitive, lacks formal rigor and could benefit from a more structured comparison of the representational capacities of encoder vs. decoder architectures.

3. The operationalization of “causal reasoning” is largely limited to logical deduction in FOL. The authors should discuss whether this adequately captures real-world causal reasoning and how their findings generalize to broader causal settings (e.g., intervention, counterfactuals).

4. The naming of the dataset (“NL Depth-12”, “NNL Depth-12”) is functional but not memorable. A more distinctive name (e.g., “LogicCausal-Bench”) could improve recognition and citation.

5. While the paper mentions related benchmarks (e.g., SimpleLogic), it does not explicitly position itself against recent causal reasoning benchmarks (e.g., CLADDER, CLEAR). A dedicated comparison would help clarify the novelty and scope of the proposed evaluation.

6. Several figures (e.g., Figure 3, 4, 5) use small text and light colors, reducing readability. Simplifying the visualizations and using higher contrast would improve clarity.

### Questions
1. How does the performance of fine-tuned encoder models compare with very large decoder models when controlling for computational budget?

2. Could the advantage of encoders be attributed to pretraining data/style rather than architecture alone?

3. Would the results hold in more realistic causal settings involving interventions or counterfactuals?

### Soundness
3

### Presentation
2

### Contribution
2
