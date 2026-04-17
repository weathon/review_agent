# Geometric Analysis of Token Selection in Multi-Head Attention

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
We present a geometric framework for analyzing multi-head attention in large language models (LLMs). 
Instead of aggregating over all tokens, we propose a top-$N$ selection mechanism that retains only the most attended tokens and study its behavior directly in the value-state space. 
We introduce novel geometric metrics -- Precision, Recall, and F-score -- to quantify the separability of selected versus non-selected tokens, and derive dimension- and margin-dependent bounds under empirically motivated assumptions on norm stability, similarity decay, and multi-phase attention distributions. 
Our theoretical results clarify how head specialization, sequence length, and the sink token jointly shape the geometry of attention. 
Empirical evaluation on several open-source LLMs (LLaMA-2-7B, Gemma-7B, and Mistral-7B) confirms our predictions: top-$N$ selection sharpens token separability, the sink token systematically correlates with Recall, and different heads specialize into local versus global regimes. 
These findings demonstrate that attention is not only a weighting mechanism but also a structured geometric classifier. 
Our framework provides measurable criteria for token selection, offers interpretability into head-level behavior, and opens new directions for designing sparse and geometry-aware attention mechanisms in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper provides a geometric interpretation of self-attention by viewing it as a separator/classifier operating in the value space. The authors propose to evaluate each attention head’s behavior through precision, recall, and F-score metrics that quantify separability between the selected (top-N) and non-selected value tokens. Under a set of assumptions about value-vector norms, similarity decay, and attention profiles, they analytically predict that attention heads operate in a small-N regime where separability is maximized. Empirically, they identify three broad types of heads (Retriever, Mixer, Reset) and show how their prevalence varies with model depth.

### Strengths
- The theoretical framing is novel and gives an interesting geometric perspective on attention, with clear analytical predictions that can be empirically checked.
- The observation that attention sinks are not mere no-ops, but instead play an active role (especially in Recall and Reset-type heads), is particularly interesting and could motivate deeper investigation into sink dynamics and normalization mechanisms in large models.

### Weaknesses
- The paper is not particularly well presented. Key motivations are unclear: the authors do not sufficiently explain why attention should be modeled as a classifier or why geometric separability in value space is the right lens for interpretability. The classification framework feels somewhat imposed rather than naturally derived from prior literature or empirical necessity.
- Several important concepts (e.g. the “MAE” mentioned in the text) are never properly defined or justified in the context of their framework.
- The paper does not make a compelling case for why separability, precision, or recall are meaningful or diagnostic of downstream behavior, model efficiency, or interpretability.
- Certain claims (such as an “oscillatory regime” across positions 100–800, or “semantic cycles”) are presented without quantitative backing or follow-up experiments. These explanations offered without proof weaken the empirical credibility of the work.
- The experiments are limited to a small set of 7B-parameter models and do not explore robustness across architectures, tasks, or long-context settings. Given the theoretical emphasis, a more comprehensive experimental grounding is needed.
- Given the current level of analysis and the modest experimental depth, the paper would be better suited for a workshop rather than a full-conference publication.

### Questions
- In Barbero et al., 2025, the authors explicitly discuss the formation of sharp heads through the influence of attention sinks. Do the authors believe that this mechanism might be related to the findings in the paper? Hows is the top-N selection scheme related to head sharpness?
- How are separability metrics expected to evolve in long-context settings?

Barbero, Federico, et al. "Why do LLMs attend to the first token?." arXiv preprint arXiv:2504.02732 (2025).

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper attempts to understand multi-head attention through a novel view of token selection. The authors provide a geometric analysis based on separability between selected and non-selected tokens. To conduct this, they provide a theoretical framework: first make assumptions on the norm states, token similarities, attention dynamics, and then derive the bounds for precision/recall/F1-score of separability of selected/non-selected tokens. Finally, based on the built geometric framework, the authors aim to interpret the head functions.

### Strengths
The authors provide a novel view of multi-head attention with nice theoretical support. Using the geometric perspective, they interpret the head functions in multi-head attention. The findings on value norms (assumption 1) are interesting.

### Weaknesses
My major concerns towards this paper are about the validity of several assumptions in this paper, and the possible actionable suggestions on the LLM/attention community.

1. About assumption 2, I am a little bit concerned whether it makes sense to use an exponential function (between 0 and 1) to model a cosine similarity (ranged from -1 to 1). Although the authors show that the MAE error is very small, I am wondering whether it makes sense to assume that the cosine similarity could be always non-negative. I am looking forward to more empirical evidence.

2. About assumption 3, the attention dynamics in Figure 4 follow my intuition that the first token is an attention sink and the recent tokens typically have higher attention. However, such a simplification may ignore existence of induction heads [1], or some other sink tokens in middle context [2]. And it does not make sense to me that all heads/layers follow such a pattern, especially considering the authors fail to show some fitting errors like in assumption 2.

3. When we discuss about multi-head attention, normally we are considering attention as a matrix. However, here it seems that authors mainly discuss the attention weights on the final token. I am wondering whether the conclusions are still valid when $L$ is largely different. 

4. Although the authors claim that the LLaMA, Gemma, Mistral follow the theoretical analysis on bounds, I suggest to show that the previous assumptions also hold in these models or different sizes (other than 7B), or different data domains. 

5. I acknowledge that the theoretical framework in this paper is elegant, please clarify how this research can provide actionable suggestions/insights to the LLM/attention community.  

Some minors:

1. Please clarify the model/data used in assumption stage.

2. The section 3.2 is a little bit messy as the authors should first claim the assumption and then show the MAE values. The current presentation may make the readers confused at first impression.

3. Please use notations to represent attention mass to prevent confusion.

I am glad to increase my score if my concerns are alleviated during the rebuttal stage. 

References:\
[1] Anthropic. In-context Learning and Induction Heads. 2022.\
[2] Sun et al. Massive Activations in Large Language Models. COLM 2024.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper takes a geometric look at how multi-head attention selects tokens.
The authors treat attention as a kind of classifier that operates in the value–state space and define geometric versions of Precision, Recall, and F-score to describe how well selected tokens are separated from the rest.
They assume three main conditions: (1) stable value norms and reduced sink activity, (2) exponential decay of cross-token similarity, and (3) a piecewise attention weight pattern that captures plateau–oscillation–recency behavior.
Under these assumptions, they derive non-asymptotic bounds on token separability and show that the best separation appears when only a few tokens (around 1–4) are selected.
Experiments on LLaMA-2-7B, Gemma-7B, and Mistral-7B back up these results and reveal three types of attention heads (Retriever, Mixer, and Reset) that differ in how they interact with the sink and final tokens.

### Strengths
1) The geometric framing is clear and easy to follow, giving an intuitive picture of how attention works.
2) The theory lines up closely with the empirical data across different models.
3) The head taxonomy (Retriever, Mixer, Reset) offers a concrete and interpretable way to describe functional differences between heads.
4) The paper doesn’t rely on extra training or architectural changes, making the analysis generally applicable.

### Weaknesses
(1) Limited practical connection.
The paper gives a clean geometric description of token selection and supports it with strong empirical evidence.
However, it stops short of connecting these findings to model performance or design improvements.
The results are insightful but remain mainly diagnostic, without demonstrating benefits such as better alignment, loss reduction, or architectural efficiency.

(2) Assumption sensitivity.
The theoretical derivations rely on several empirical assumptions — stable value norms, exponential similarity decay, and piecewise attention profiles.
These assumptions are plausible but not formally justified, and the paper does not examine cases where they fail (e.g., fine-tuned models, longer contexts, or high-variance heads).
The robustness of the geometric bounds under such conditions remains unclear.

(3) Missing broader comparison.
The analysis focuses entirely on top-N token selection and does not compare against more continuous or weighted formulations of attention.
As a result, it is uncertain whether the observed separability patterns are specific to discrete selection or general to the full attention mechanism.

### Questions
(1) Can the proposed geometric Precision/Recall or F-score metrics be linked to downstream performance (e.g., loss, perplexity, or alignment quality)?

(2) How sensitive are the theoretical bounds to violations of the key assumptions (norm stability, similarity decay, or piecewise attention profiles)?

(3) Could the geometric interpretation be extended into token pruning or head sparsification methods in practice?

(4) Do the Retriever, Mixer, and Reset head types appear consistently across different model sizes, architectures, or data domains?

(5) Both this paper and Orthorank(Shin et al., 2025) report a similar phenomenon, stable norms except for the sink token. Are these two observations fundamentally related, or do they arise independently in different representational spaces (hidden vs. value)?

### Soundness
3

### Presentation
3

### Contribution
2
