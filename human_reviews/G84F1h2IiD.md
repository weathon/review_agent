# Spike No More: Stabilizing the Pre-training of Large Language Models

- Avg Score: 4.33
- Decision: Reject
- Scores: 5, 3, 5

## Abstract
Loss spikes often occur during pre-training of large language models.
The spikes degrade the performance of large language models and sometimes ruin the pre-training.
Since the pre-training needs a vast computational budget, we should avoid such spikes.
Based on the assumption that the loss spike is caused by the sudden growth of the gradient norm, we explore factors to keep the gradient norm small through an analysis of the spectral norms of the Jacobian matrices for the sub-layers.
Our findings suggest that stabilizing the pre-training process requires two conditions: small sub-layers and large shortcut.
We conduct various experiments to empirically verify our theoretical analyses.
Experimental results demonstrate that methods satisfying the conditions effectively prevent loss spikes during pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper analyzes gradient norms to explain and reduce the spikes of training loss. However, this paper does not address the differences between LLM and small models, differences between deep and shallow layer, and the relationship between spikes and gradient norm. Most importantly, the risk of training LLM is simplified by evaluating the performance on benchmark datasets while the risk of spending computing resources is not well addressed unfortunately requires lots of improvements.

### Strengths
The paper effectively addresses the critical issue of loss spikes during training, providing a detailed analysis of their relationship with gradient norms and embedding means. It evaluates existing approaches like "Scaled Embed" and "Embed LN," discussing their effectiveness in mitigating spikes. Additionally, the paper offers valuable insights into the impact of learning rate adjustments on model stability and compares the behavior of spikes in large language models (LLMs) with smaller models, providing a broader context for its findings.

### Weaknesses
The paper suffers from an unclear relationship between spikes and poor performance, with insufficient explanations of key terms and assumptions. The evaluation section is not well-explained, and there are inconsistencies in terminology. Additionally, it lacks necessary plots and data to support its assumptions, and some figures are difficult to interpret due to overlapping lines. Reproducibility is a concern as common public architectures are not used, and some discussions are considered irrelevant to the main topic.

### Questions
The relationship between spikes and poor performance is not clearly established.

Some loss spikes can be recovered during training, while others cannot.

Are you assuming that spikes are (1) always bad, (2) risks that result in divergence in some cases, or (3) sometimes acceptable and not harmful?

L40: What is catastrophic divergence? Does it mean the spike never goes down again?

What model is used in Figure 1?

What is the difference between spikes in LLMs and other smaller models?

L47: The assumption between spikes and gradient norm is unclear. What did you observe in L121?

L53: What is the standard deviation of embedding means? Does this equate to a large shortcut in the later sections?

L63: Evaluation is not explained.

L83: Conventionally, does shortcut mean residual?

Can you comment on the parallel FF+Attn setting where there is no intermediate vector? It is broadly used in many models, e.g., Pythia, Mesh-Transformers, and PaLM.

clear writing in section 2.1, 2.2

L130: What is the difference between W_* and W?

L134: You should mention the single-head attention and F as the identity function (linear settings) in the abstract and introduction.

L134: Please plot the distribution of x, x', and W, and note the standard deviation and mean of each layer to support your assumption. In Appendix F, there are only plots for initialized models. I could not find plots for W or during training and pretrained models.

L160: What are d and d_ffn? Did you mention them somewhere?

L167: This is an estimation as shown in L159. Please be consistent.

L174: It's unclear what this well-known formula is.

Eq 13: Why is variance the degree of freedom?

Condition 2 of large shortcut comes from assumption 1.

L219, Eq 19: Where does the 2 come from?

L234: The writing in Section 4 could follow the format in Section 5.2 to improve readability.

L250: Both "Scaled Embed" and "Embed LN" are existing approaches, not introduced by this paper.

L291: Why should it be close to 1 and not as large as possible? If so, why is it called 'large shortcut' in the previous sections? Similarly, should the sublayer be as small as possible or close to some value? Why do we scale the embed by some value but not a larger one? What is the optimal scale from your theory?

L309: C4 is too small to be the pretraining data for LLM.

L317: Why not use a common public architecture (e.g., LLama, Mixtral, Gemma) for reproducibility? Is there any common pretraining baseline in the literature (e.g., Dettmers et al. (2022) and Le Scao et al. (2022))?

Figure 3 (a): Lines are overlapped and cannot see what happened. Is there any other metric like the number of spikes other than visualization?

L367: Only in the embed layer?

The training risk mentioned in the abstract is not addressed in the experiment. We can only see some marginal improvement over perplexity. In the abstract, it indicates that 'if we don't take spikes seriously, the whole effort will be in vain due to catastrophic divergence.' However, both vanilla and embed detach do not suffer from catastrophic divergence in the experiment results.

Section 6.1: How does the learning rate affect the gradient norm? How does the learning rate affect the four baseline methods? Did you indicate that stabilization can be achieved by a smaller learning rate? If so, there is no actual need for "Scaled Embed" or "Embed LN"? In Figure 5, we can see that vanilla performs better than Scaled Embed. To achieve better final performance as depicted in Table 3, can we use a small learning rate to travel through the risky early training stage and increase the learning rate later to avoid spiking?

Section 6.2: Good point! I expect a similar discussion in Section 6.1 to explain why a smaller learning rate can reduce spikes. Please remind me if I missed something. Also, this indicates the settings of 'short seq in the early stage and long seq later.' Can we say that 'small lr in the early stage and large one later' is also possible?

L490: Can you use your theory to explain why preLN is more stable than post-LN?

L509: The efficiency and learning rate discussion is a bit too far and not quite relevant.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors provide theoretical support to reduce the likelihood of divergence based on stabilization via small sub-layer initialization and large shortcut values. They then propose two techniques to effectively support this assumption and evaluate accordingly.

### Strengths
The paper conducts mathematical analysis to demonstrate the requisite terms they later leverage.
Paper is clear and provides actionable results.

### Weaknesses
The authors only tested on smaller models, it is well established that most instability problems happen with larger models (>100B parameters). It would be beneficial to evaluate the loss curves on larger models or more diverse datasets 

Although the focus of this paper was to stabilize training, they underperform on loss-curves compared to vanilla approaches to disprove Le Scao et al's findings. This is hypothesized to be related to learning-rates - which is demonstrated by looking at a absolute min score. However,  by only exploring LR adjustments on smaller models, it isn't immediately clear that the proposed method is consistently better as sub-optimal lr's are more stable, a lr scheduler could account for differences in the long-term.

### Questions
Where approaches evaluated on benchmarks post-training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a strategy to avoid spikes in loss during the training of LMs by keeping the gradient norm small. To manage the upper limit of gradient norms effectively, the method involves (i) using small initial values for sub-layers, and (ii) maintaining the standard deviation of embeddings around 1.

### Strengths
* The findings (i) and (ii) from the analysis are well presented, although they have been previously utilized in past studies.
* This work examines various learning hyperparameters.
* This work also presents the results for the 13B model in Table 6 of the Appendix.
* The paper is well-written and easy to understand.

### Weaknesses
* Although the theoretical analysis is intriguing, I question the practical value of this work, as most practices described in Section 4 are already in use. Utilizing small values for initialization to ensure stable training is well-known, and both Scaled Embed and Embed LM have been introduced in prior literature. If this work could offer a novel, advanced method for embedding normalization, it might receive more interest from the community.
* The activation function F was assumed to be either an identity function or ReLU, as stated on line 152 of page 3. What would be the results if widely used activation functions in recent LLMs, such as SiLU and SwiGLU, were applied?
* I am curious about how loss spikes impact the performance of downstream tasks on LLM leaderboards, beyond just affecting perplexity. Are these spikes also harmful to the accuracy of downstream tasks?
* I believe it would be beneficial to conduct a theoretical analysis of the relationship between learning rate, loss spikes, and model sizes. This suggestion stems from the observation that the learning rates causing loss spikes differ according to model size.

### Questions
Please see the above weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
