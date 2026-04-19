# Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
We investigate the mechanisms behind three puzzling phenomena observed in transformer-based large language models (LLMs): *attention sinks*, *value-state drains*, and *residual-state peaks*, collectively referred to the *extreme-token phenomena*. First, we demonstrate that these phenomena also arise in simpler architectures—transformers with one to three layers—trained on a toy model, the Bigram-Backcopy (BB) task. In this setting, we identify an *active-dormant mechanism* that causes attention heads to become attention sinks for certain domain-specific inputs while remaining non-sinks for others. We further develop a precise theoretical characterization of the training dynamics that lead to these phenomena, revealing that they are driven by a *mutual reinforcement mechanism*. By small interventions, we demonstrate ways to avoid extreme-token phenomena during pre-training. Next, we extend our analysis to pre-trained LLMs, including Llama and OLMo, revealing that many attention heads are governed by a similar active-dormant mechanism as in the BB task. We further show that the same mutual reinforcement mechanism drives the emergence of extreme-token phenomena during LLM pre-training. Our results study the mechanisms behind extreme-token phenomena in both synthetic and real settings and offer potential mitigation strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines the mechanisms behind the "extreme-token phenomena" in large language models (LLMs), such as attention sinks, value-state drains, and residual-state peaks. These phenomena are collectively attributed to the "active-dormant" mechanism in transformer-based LLMs. The study first demonstrates these phenomena on simpler models trained on the Bigram-Backcopy (BB) task, identifying a mutual reinforcement dynamic driving these behaviors. The authors then extend their analysis to pre-trained LLMs (e.g., Llama and OLMo) and suggest mitigation techniques, such as replacing SoftMax with ReLU or switching optimization methods to control these extreme-token phenomena.

### Strengths
1. The study makes clear contributions by identifying the active-dormant mechanism and mutual reinforcement dynamic, helping to explain complex behaviors in transformer-based models.

2. The paper rigorously combines theoretical characterizations with empirical validation, making its claims robust and well-supported.

### Weaknesses
1. There remains a gap between the BB Task and real language scenarios. Although I recognize the potential practical value of the authors' intuitive explanation of these phenomena, I would still expect authors to validate their explanations in real scenarios. For example, it would be helpful to confirm whether deactivating attention heads with attention sinks truly has no impact on the final output.

2. The paper’s explanation for why Adam tends to cause residual state peaks is insufficient, as it mainly describes the phenomenon without deeper analysis. Furthermore, I did not see an evaluation of the impact of applying SGD and ReLU to real language models. In my opinion, both SGD and ReLU come with certain drawbacks, which should be discussed.

3. Here is a potentially relevant study [1] where the "broadcasting" mechanism mentioned has a formation process similar to the attention sink.

[1] Anchor function: a type of benchmark functions for studying language models. arXiv:2401.08309.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper examines the "attention sink" phenomenon in large language models, focusing on three related concepts: attention sinks, value-state drains, and residual-state peaks. Through a toy Bigram-Backcopy task, the paper reveal an active-dormant mechanism in which the attention head assigns dominant weight to the initial token in the dormant phase and shifts focus to relevant context tokens in the active phase. They attribute this behavior to a mutual reinforcement mechanism. Extending this analysis to real large language models, the authors show that this mechanism also emerges during pre-training, suggesting a understanding of the issue and potential mitigation strategies.

### Strengths
- The paper addresses the attention sink phenomenon in large language models, an important and timely issue.
- The analysis using a toy example is original, providing intuitive insights and interpretations of the problem.
- The motivation is clear and well-articulated.
- The paper is well-organized, and the figures are highly illustrative.

### Weaknesses
My primary concern is the statistical significance of the observed results, which raises questions about the validity of the conjecture regarding the active-dormant mechanism. For instance, in the BB task toy experiments, a simplistic generation rule (bigram transitions) is assumed, making the behavior of trigger and non-trigger tokens very distinct. This makes it difficult to accept that the theory and conjecture are applicable to real large language models, where such clear distinctions between tokens (trigger vs. non-trigger) do not exist. Section 3, a specific attention head in a specific layer is shown to exhibit behavior similar to the conjectured mechanism from Section 2. However, I find this unconvincing, as demonstrating a single head is insufficient to establish that the active and dormant phases align with what was observed in Section 2 for the same reasons.

My second concern is the lack of demonstrated practical benefits from this analysis. Although the authors suggest that this study could offer potential mitigation strategies, the paper does not validate the effectiveness of these mitigations, which limits the overall impact and applicability of the work.

### Questions
- What is the motivation for using the BB task and designing the trigger and non-trigger tokens?
- On Lines 80 and 87, the word "mechanism" appears twice consecutively, which seems to be a typo.
- In Claim 4, could you clarify the case where "adding any value state from previous tokens worsens the prediction"?
- Why does the causal mask cause training dynamics to favor the <s> token? In Figure 6, attention is also given to token 3, which is described as semantically unmeaningful. So, is the cause of the attention sink truly the causal mask, or is it the lack of semantic meaning in certain tokens?

### Soundness
2

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
The paper studies the emergence of extreme-token phenomena in modern transformer neural networks, namely: Attention sinks, Value state drains and Residual state peaks, all previously identified in recent works. The first refers to the observation that in some attention heads, special tokens named sink tokens attract a large proportion of the attention weights. The second highlights that values states associated with sink tokens are usually significantly smaller than those of other tokens. The third refers to the observation that the intermediate representations (except for initial and final layers) of sink tokens are characterised by larger norms compared to those of other tokens. 


The authors first show that attention sinks and value state drains can be identified in a simple settings: one layer transformer models trained on the Bigram-Backcopy task which involves sequences generated either through bigram transitions or by copying previous tokens in the presence of special trigger tokens. Through an in-depth empirical analysis, the authors show that all non-trigger tokens exhibit attention sinks, while the attention for trigger tokens is concentrated on their preceding positions. In particular, the BOS tokens attracts a significant proportion of the attention weights for the non-trigger tokens, and its value state decreases in norm accordingly. The authors highlight that the aforementioned extreme token phenomena develop reinforcing each other during training. This observation is corroborated by a theoretical analysis showing that indeed "Attention logits grow logarithmically reinforced by small value states" and "Value state shrinks to a small constant vector reinforced by large attention logits". 

The final part of the paper is devoted to an investigation on the emergence of these phenomena in LLMs, with a particular focus on Llama 2-7B-Base and  OLMo-7B, both open-source. On the first model, a specific head is described in terms of its behaviour when the model is prompted with two different data sources. A very similar mechanism is shown to emerge, with the head being dormant on Wikipedia data (prose) as opposed to Github (code) data. Thanks to the availability of intermediate checkpoints for OLMo-7B, the authors are able to draw a parallel between their theoretical analysis of the learning dynamics of their toy model and that of OLMO, resulting in a good agreement.

### Strengths
- The paper is very well written and easy to follow.

- Extreme-Token Phenomena are very interesting and quite common across LLMs. The paper provides a simplified setting to theoretically describe their emergence. In particular, the authors show that these phenomena manifest themselves in synergy through training, via "mutual reinforcement dynamics".

- The paper shows that the analysis developed on the BB task extends to some extent to LLMs. In particular, by manually deleting a specific attention head on the LLAMA model, the authors show that the resulting model's performance depends on whether the head is dormant or not, which in turn depends on the specific input context (Fig. 6). In addition, Fig. 7 and 8 corroborate the theoretical results on the simple BB model.

### Weaknesses
- In section 2.1 (see Figure. 4), the considered one-layer transformer does not alternate the attention layer and the MLP along the depth dimension. Rather it uses these two components in parallel. The reason for this choice is not clear and should be better motivated. In addition, the trained models use pre-layer norm and it is not clear if residual connections are used and how.

- In the derivation of the loss(alpha, beta), it is not clear to me how the stable distribution is defined. Could you clarify?

- **Replacing SoftMax by ReLU attention removes extreme-token phenomena**: Can you provide more intuition on the role of the softmax? Its utilisation appears to be correlated with the emergence of extreme-token phenomena. The natural question then is whether this component is actually needed in practice and why modern LLMs use it. Could you elaborate on this?

- A question on Theorem 3, first point: I would have expected the logits to grow logarithmically as beta decreases (i.e. small value states). Instead, no constraint on beta seems to be needed for the logits to grow. I would appreciate it if the authors could clarify this point.

- As for the emergence of residual peaks, the experiments show that 3 layers are needed for the residual peaks to appear. First, are the authors now using a standard transformer architecture with attention layers interleaved with mlps? Second, do the authors have any intuition on why at least 3 layers are needed?

- In the construction of the token embeddings, the authors are assuming them to have dimension V, i.e. they are represented as one-hot vectors. This implies different tokens are orthogonal to one another. How important is this assumption?

-  In the BB example, it is shown that a head can have sinks yet being non-dormant (due to trigger tokens). Is this case contemplated by the theory? This seems to be quite general and is reflected in real LLMs (fig. 18 and fig. 19 in the appendix). To this extent, fig. 6 seems to be a pretty special case, where, for wikipedia data, the head is actually dormant as there are only two sink tokens and nothing else.

- How did you practically identify the heads exhibiting sink tokens in real LLMs? 

- Are the results obtained for OLMO also valid for Pythia (for which intermediate checkpoints are also available)?

### Questions
See weaknesses part.

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
The paper investigates the underlying mechanisms of "extreme-token phenomena" in large language models (LLMs), characterized by three main effects: attention sinks, value-state drains, and residual-state peaks. These effects result in specific tokens persistently attracting high attention weights (attention sinks), displaying reduced value states (value-state drains), and exhibiting notably large norms in intermediate representations (residual-state peaks). The authors explore this by analyzing transformer models in a controlled setting, the Bigram-Backcopy (BB) task, identifying an active-dormant mechanism where attention heads behave differently based on input tokens. They find that mutual reinforcement between attention sinks and value-state drains drives these phenomena, with residual state peaks emerging in deeper architectures. They also find that attention sinks and value state draining are related, and interaction between the mutual reinforcement mechanism and Adam is responsible for residual state peaks. Expanding from a toy model's theoretical and empirical analysis, the paper shows that an active-dormant mechanism controls attention heads. Specifically, it highlights the active-dormant mechanism in existing pre-trained LLMs like Llama2 and OLMo-7B-0704 by identifying an interpretable active-dormant head through causal intervention analyses. By intervening in attention mechanisms and optimization strategies, such as replacing SoftMax with ReLU activations, they demonstrate methods to mitigate these extreme-token phenomena.

### Strengths
-	The paper provides valuable insights into LLMs' underlying mechanisms of extreme-token phenomena (attention sinks, value-state drains, and residual-state peaks). By identifying the active-dormant mechanism in attention heads and a mutual reinforcement process, the paper contributes to a deeper understanding of transformer model behavior, enhancing interpretability in LLMs.
-	The paper provides a comprehensive theoretical and empirical analysis of the mutual reinforcement mechanism in a toy one-layer transformer model to explain extreme-token phenomena. The identification of mutual reinforcement dynamics is robustly supported by theoretical analysis and validated through experiments on pre-trained LLMs using the Bigram-Backcopy task, offering strong evidence that this mechanism underlies attention sinks, value-state drains, and residual state peaks in both simplified and complex models.
-	The paper’s application of findings from toy models to analyze real-world LLMs, such as Llama2 and OLMo, significantly strengthens the study's impact. Validating the mutual reinforcement mechanism and active-dormant attention head dynamics in large-scale models highlights these identified mechanisms' broader applicability and relevance. This cross-model validation not only supports the robustness of the findings but also enhances the study's generalizability and potential utility for advancing transformer-based architectures

### Weaknesses
-	The motivation behind the mitigation of extreme-token phenomena is not clear. It is not clearly established that extreme token adversely affects the model’s performance. Although the paper explores interventions like replacing SoftMax with ReLU, these modifications are limited and could unintentionally impact model performance. The study doesn’t extensively examine trade-offs associated with these changes, such as how they might affect other aspects of model performance or generalization ability. This leaves questions about whether mitigating these phenomena would lead to meaningful improvements in model usability or performance in practical applications. Could the authors provide a more comprehensive explanation or analysis of why or what specific effects reduction of the phenomena on the downstream task?
-	The paper claims that the formation of extreme tokens in LLMs follows a similar mutual reinforcement mechanism as observed in a simplified BB model, with theoretical results that can be generalized to more complex models. However, this justification relies on specific assumptions, including zeroing out attention heads for intervention outcomes, which will not impact token prediction accuracy. This assumption may oversimplify the interactions in real-world LLMs, where dependencies between heads and layers are often more intricate. Additionally, while some empirical evidence in pre-trained models aligns with the theory (active and dormant attention heads within Llama2), the scope of models evaluated is limited, raising questions about the robustness of the claim across diverse architectures and tasks. Would it be possible for authors to show whether their claims about specific attention heads being active and dormant but under the premise that context are not significantly different? For example, mathematics-based context vs language comprehension-based context. It would help gauge the sensitivity of the claims when the contexts are not vastly different. In particular, would we still find attention heads with simple and interpretable active behavior for weaker contextual differences?

### Questions
-	In section 2.3, it is mentioned that no residual peaks appear in a one-layer transformer(BB model) or even with different combinations (2xTF, MLP+TF+ML, Attn+TF+MLP); it would be interesting to know why it seems to have residual peaks in certain instances. The authors claim that with a BB model, “residual state peaks contribute to the attention sink and value state drain phenomena.” However, it still remains unclear between the dependency between residual state peaks and, attention sink and value state drain. It would be helpful to see the comparison of residual state peaks and attention sink and value state drain phenomena for the 1-layer BB model even when residual state peaks do not appear.

### Soundness
3

### Presentation
3

### Contribution
2
