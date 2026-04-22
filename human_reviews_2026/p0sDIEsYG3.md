# Video-KTR: Reinforcing Video Reasoning via Key Token Attribution

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Reinforcement learning (RL) has shown strong potential for enhancing reasoning in multimodal large language models (MLLMs), yet existing video reasoning methods often rely on coarse sequence-level rewards or single-factor token selection. Such approaches neglect fine-grained links among visual inputs, temporal dynamics, and linguistic outputs, limiting both accuracy and interpretability. We propose Video-KTR, a modality-aware policy shaping framework that performs selective, token-level RL by combining three attribution signals: (1) visual-aware tokens identified via counterfactual masking to reveal perceptual dependence; (2) temporal-aware tokens detected through frame shuffling to expose causal and temporal sensitivity; and (3) high-entropy tokens signaling predictive uncertainty. By reinforcing only the union of key tokens, Video-KTR focuses learning on semantically informative, modality-sensitive content while filtering out low-value tokens. Across five challenging benchmarks, Video-KTR achieves state-of-the-art or highly competitive results—42.7% on Video-Holmes, surpassing GPT-4o—with consistent gains on both reasoning-centric and general video understanding tasks. Ablation studies verify the complementary roles of the attribution signals and the robustness of targeted token-level updates. Overall, Video-KTR improves accuracy and interpretability, offering a simple, drop-in extension to RL for complex video reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
(I tend to write shorter reviews and the length of the review does not reflect the quality of paper or the time spend on reviewing it). 

Note that I am not an expert in Video modelling nor Reinforcement Learning and the review is from a perspective of a general Deep Learning researcher. 

This paper proposes a few heuristics to improve the RL training of reasoning models (especially in the context of Video) by using things like perceptual dependence, remporal sensitivity and entropy based uncertainity. The authors show that each of these heuristics reveal natural points to not have RL be updating all the tokens in the trajectory but only a handful. They paper's experiments are strong and have extensive analysis. The writing is very easy to follow.

### Strengths
Clear explainations and good experiments. The extent of empirical investigations is commendable -- I am not an expert so can not comment on the strength of the experiments.

### Weaknesses
1) The comparision to the closed source models is from almost 2 generations ago and would be good to have latest to have a better understanding of where things stand. 

2) While the heursitics are working great for this task at hand, given the past of DL and my experience, the heuristics stop helping in general purpose cases and while we scale up. It is a strong result for now in a narrow domain but a broader investigation is what will concretize the proposed things in the modern RL pipelines. I strongly suggest either doing a more general purpose evaluation to support this broadly.

### Questions
See above. Again I am not an expert so will defer to other reviewers on this paper.

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
4

### Summary
This paper proposes a way to select important tokens for reinforcement training visual LLMs to enhance their reasoning abilities. The method includes identifying visual-aware, temporal-aware and entropy-aware tokens (connecting words or logical words in my opinion) and only compute the loss based on those selected tokens.

### Strengths
1. The paper explores which tokens are more important to be optimized to enhance the reasoning process. This is a cheap and useful technique in general.

2. The paper is well-structured and clearly written, showing the background, proposed methods and experimental setup clearly.

3. The results seem promising as the token selection method yields better performance on reasoning-centric tasks such as video-Holmes. Ablation studies and various plots were made to support the findings.

### Weaknesses
1. I am a bit concerned regarding the theoretical foundation of this work. What is the actual contribution of the tokens that has actually been masked? For example, would these tokens introduce unwanted noise to the gradient w.r.t. the logits. A theoretical derivation by looking at the actual influence with or without the mask in the loss when taking the gradient w.r.t. the logits or even deeper in the network should be provided. __Moreover__, tokens that are visual-aware or temporal-aware, does not necessarily mean they need to be __optimized__ unless they really produces differences when sampling multiple paths. For example, I can have "cat" as a token that is visual-aware, but that cat appears in every samples I draw, and hence not producing useful information and can be excluded here. In my opinion, the tokens that should be optimized are the tokens that carry __information inconsistency__ across samples. I find this part of justification missing in the current manuscript.

2. I am also concerned about the entropy-aware tokens. I feel that the authors are trying to introduce the concept of information inconsistency here and to optimize tokens where the model is highly uncertain. However, I am not sure if entropy is a good metric, especially looking at the range of values it can get, a threshold might be difficult to draw, because different value on different tokens mean different things (even if you select via top-r% they are still inconsistent across reasoning paths). I find this part quite counter-intuitive and would leave this to other reviewers to decide.

Overall, I am not fully convinced by the proposed methodology, given that nowadays many factors may influence the model performance. Deeper analysis and theoretical justifications are needed for the current manuscript to be reasonably sound for publication.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on video reasoning with multimodal large language models. Existing methods rely on coarse sequence level rewards or single-factor token selection for reinforcement learning (RL). This work Video-KTR is a policy shaping framework for token-level RL that only look at three types of tokens, which are visual-aware tokens (i.e., tokens closely associated with visual input like appear, show, etc.), temporal-aware tokens (i.e., tokens sensitive to the temporal structure of videos like finally, first, etc.), and entropy tokens (i.e.,  reasoning-critical tokens like however, wait, seem, now, etc.). Therefore, Video-KTR learns semantically informative, modality-sensitive, and filters low-value tokens, which contributing to the strong performance across five benchmarks.

### Strengths
1. The general idea is simple and insightful by letting the model reinforce on 3 types of tokens that are crucial to video reasoning, but effective that has performance gain over 5 video reasoning benchmarks.
2. Clear ablation study of all combination of 3 types of tokens to show clearly what type of token matters the most and how each type of tokens contribute to the performance gain.
3. 7 research questions are insightful. For examples, by using the same dataset and training recipe, they made sure that the comparisons are fair. There are linguistic insights like visual-aware tokens are mainly nouns, temporal aware tokens emphasize verbs and pronouns, and entropy-aware tokens has a higher share of adjectives. A general insight for large language models is that the log-probability differences is a reliable and efficient signal for tracking prediction-confidence shifts.
4. Writing and images are clear and easy to understand.

### Weaknesses
1. The performance gain over the Vanilla GRPO for three types of tokens, despite being higher, but it's a small increase. When enabled all three types, the average delta performance gain is just 2.4% for the average of three benchmark, while other ablations show even smaller differences. Therefore, the effectiveness of the Video-KTR is limited from the results.

### Questions
Is the Video-KTR model initialized from Qwen2.5-VL? It's better to make clear about the base model for implementation in the paper.

### Soundness
3

### Presentation
3

### Contribution
3
