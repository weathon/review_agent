# DEX-AR: A Dynamic Explainability Method for Autoregressive Vision-Language Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
As Vision-Language Models (VLMs) become increasingly sophisticated and widely used, it becomes more and more crucial to understand their decision-making process. Traditional explainability methods, designed for classification tasks, struggle with modern autoregressive VLMs due to their complex token-by-token generation process and intricate interactions between visual and textual modalities. We present DEX-AR (Dynamic Explainability for AutoRegressive models), a novel explainability method designed to address these challenges by generating both per-token and sequence-level 2D heatmaps highlighting image regions crucial for the model's textual responses. The proposed method offers to interpret autoregressive VLMs—including varying importance of layers and generated tokens—by computing layer-wise gradients with respect to attention maps during the token-by-token generation process.
    DEX-AR introduces two key innovations: a dynamic head filtering mechanism that identifies attention heads focused on visual information, and a sequence-level filtering approach that aggregates per-token explanations while distinguishing between visually-grounded and purely linguistic tokens. Our evaluation on ImageNet, VQAv2, and PascalVOC, shows  a consistent improvement in both perturbation-based metrics, using a novel normalized perplexity measure, as well as segmentation-based metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors are tackling the problem of sequence-level autoregressive VLM explainability, where the current XAI approaches should be extended to handle multiple outputs. The idea is to use the derivative of the LogitLens at each timestep $t$, on each of the output tokens generated one by one. Then, weighting the impact of the saliency maps according to the ratio of vision-impact with respect to text-impact, where tokens relying more on vision should be amplified more than text-based ones. This weighting approach is done both in head level granularity and token level. This yields a single saliency map of the entire VLM output.

### Strengths
* The gap of pure explainability on autoregressive VLMs, i.e. for question-answering is indeed seems to be an issue. I also think that VLMs should be handled appropriately in terms of explainability, so it is a very important topic and at least for me it seems that it is underexplored, hence novel.
* The results of the method seems to be better than other existing methods which they compare with, moreover they compare with several cutting-edge VLMs.

### Weaknesses
* The convention of the citations embedded in the article is weird and super not convenient. I saw submissions with blue citations, some are still black, but I did not see missing parentheses. This make the citations blended within the flow of the sentence without clear separation. Very confusing, this must be fixed.
* Lines 51-77 - the claim of the inability of current explainability methods to act on autoregressive VLMs is too decisive. There are a plethora of works, some of them are modality-specific ones, however there are many which aim at explaining CLIP ([1], [2], [3] and of course many more), which is obviously multimodal. I think that claim like this must be more backed with experiments. This comes with a very limited Related Work section for explaining methods dedicated for joint embeddings like CLIP (I already mention some, there are plenty more).
* Equation 3 - it seems like the authors are not familiar with the concept of LogitLens which is widely known, it is exactly what explained in Eq. 3 ([4]).
* I think that XAI on CLIP methods might be more relevant and efficient , so it it highly preferable to compare with.


Minor Weaknesses:
* Line 47 - correct spacing: language models( Zini & Awad (2022); Zhao et al. (2024a)) -> language models (Zini & Awad (2022); Zhao et al. (2024a))
* Line 93 - incorrect sentence: We evaluate to proposed method of various downstream tasks and datasets  -> We evaluate the / our proposed method on various downstream tasks and datasets
* Line 105: 2)We -> 2) We. 
* Line 107 3)We -> 3) We.
* Lines 154-158 I think that there is a mistake of 1 shifted in the indexing. Say for $t=1$, which is the first token the LLM should output, then it has no other output tokens to digest, thus it process $N + T_c$, and in general: $T_t = N + T_c + t - 1$ and $T = N + T_c + T_a - 1$. 
* Line 188, it is obviously not restricted to be a word. It is better to stick to the term token here.
* Line 207 - missing period at the end of the sentence, before the word "Next".  
* Line 209: $[:, : N]$ ->  $[:, :N]$, remove redundant space.
* Line 253, there is a difference between starting (") and ending ("). You used the same through all the paper. 
* Line 268: What "resp." is? it seems like a mistake.

refs:
[1] Interpreting CLIP's Image Representation via Text-Based Decomposition. Yossi Gandelsman et al. ICLR 2024.
[2] Interpreting the Second-Order Effects of Neurons in CLIP. Yossi Gandelsman et al. ICLR 2025.
[3] From Attention to Prediction Maps: Per-Class Gradient-Free Transformer Explanations. Ronen Schaffer et al. Preprint
[4] Logit Lens Blog post: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

### Questions
* Dynamic head filtering. Im not sure why you called it "filtering", since it does not filter out the impact of several heads, it is weighting them, so you might consider to denote it more accurately (Dynamic head weighting or something like that).  Regarding the method itself - I might agree with the claim that larger objects might affect more dramatically if you average across all tokens, however on the other hand relying on the maximal value is super sensitive to outliers which occur frequently in vision transformers. Specifically it has been already shown that ViTs are dedicating tokens to embed global information, where it look like outliers (present also in your examples in Fig. 3) [1].
* I saw you have an ablation study for the filtering stage, nevertheless I think that if you want to understand what ratio of the attention is spread on the vision on top of the text, it makes more sense to normalize the gradients - say with softmax, and take only the vision values of it. Im not sure why subtracting as you do is explicitly computing the ratio between how much attention is spread on vision with respect to text. It is very specific, but it might be nice to check this if it is simple.
* It is a general issue in VLMS, that as the output sequence get larger, the matrices (for example the attention matrix) get larger and larger. Nowadays the output of VLMS are of a large bunch of tokens, so how do you think your approach will handle in terms of timing and efficiency (and also performance) on a much larger examples?


I think that the idea of dedicating explainability method specifically for autoregressive VLMS is super relevant and covering a very important gap, where your approach is taking a step further in this aspect. However, I think that the paper is not written fluently enough, and that your approach is currently only a small extension step of current vision XAI methods (the only innovative part currently is the filtering approach both in head and token levels, which raises quite a few concerns, as I asked and mentioned here). Moreover I didnt see any interesting examples of how your new method reveals interesting failure cases which highly relevant to the case of autoregressive VLMs and interpretability in general. I have mixed feeling about this paper, because I think that the topic is new, challenging and important, but I decided on minor negative rating because I still think that the academic gain here is too small, and the paper is written not up to top-tier standard. Nevertheless, I highly encourage you to improve it and Im sure it can be accepted to high impact conferences.
refs: 

[1] Vision Transformers Need Registers. Darcet et al. ICLR 24.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The large scale use of VLMs in daily lives make the process to arrive at its decision critical. Issues usually in auto regressive models are the token by token generation and this doesn’t help if the modalities are dual (VLM). Authors present  DEX-AR (Dynamic Explainability for AutoRegressive models), a novel explainability method designed to address these challenges by generating both per-token and sequence-level 2D heatmaps highlighting image regions crucial for the model’s textual responses. DEXAR offers to interpret autoregressive VLMs—including varying importance of layers and generated tokens—by computing layer-wise gradients with respect to attention maps during the token-by-token generation process. The method involves two key innovations: a dynamic head filtering mechanism that identifies attention heads focused on visual information, and a sequence-level filtering approach that aggregates per-token explanations while distinguishing between visually-grounded and pure linguistic tokens.

### Strengths
1)The paper addresses a critical gap in explainability for autoregressive VLMs by providing per-token explanations during sequential generation. This is particularly valuable given the widespread deployment of VLMs where understanding decision-making processes is crucial for trust and debugging.
2)Dynamic Head Filtering: The attention head filtering mechanism that identifies heads focused on visual information represents a meaningful contribution to understanding cross-modal attention patterns.
3)Multi-level Analysis: The dual approach of per-token and sequence-level explanations provides comprehensive insight into both local and global decision-making processes.
4)Layer-wise Gradient Analysis: Leveraging gradients with respect to attention maps offers a principled approach to attribution that respects the model's internal computations.
5)Excellent experiment results 
The experimental evaluation demonstrates robust performance across multiple dimensions:
(i)Perturbation Analysis
(ii)Cross-Architecture Validation
(iii)Computational Efficiency
(iv)Segmentation Performance Excellence
6)Thorough Ablation Studies

### Weaknesses
1. Clarity Issues
The paper suffers from several theoretical gaps that undermine the rigor of the proposed method. Most critically, the intermediate logits computation in Section 3.2 lacks clear justification for why o^{l,t} should be conditioned only on the last generated token. While this conditioning may stem from the autoregressive structure, the authors fail to explicitly explain how causal masking affects this choice, why this specific conditioning is optimal for attribution, or whether alternative conditioning strategies were considered and their associated trade-offs. Furthermore, the transition from per-token computations to sequence-level aggregation requires stronger theoretical grounding, particularly regarding how information flows through the autoregressive generation process and how this affects the attribution quality.
2. Methodological Concerns
The claimed "dynamic filtering mechanism" (L228) appears to be primarily a weighting scheme rather than true dynamic filtering, raising significant methodological concerns. From a computational efficiency perspective, this weighting mechanism may be substantially more expensive than simpler threshold-based pruning of non-contributing attention layers, yet no comparison is provided with such alternatives that could achieve similar results more efficiently. The terminology "dynamic filtering" may be misleading when describing what is essentially attention re-weighting, suggesting a need for more precise naming and clearer distinction between the proposed approach and existing attention manipulation techniques.
3. Experimental and Design Limitations
The assumption underlying sequence-level filtering—that filler and grammatical words are less important for visual grounding—is problematic and potentially limits the method's applicability. In scenarios involving similar objects (e.g., "two apples on a table"), grammatical words and spatial prepositions become crucial for accurate localization, contradicting this assumption. The examples in Figure 1 focus on distinct objects ("cat and dog") which may not represent the full complexity of visual-linguistic grounding tasks where subtle linguistic cues matter significantly. Additionally, the choice of Signal-to-Noise Ratio (SNR) for filtering lacks both theoretical justification and empirical validation against alternative metrics.
While the perturbation and segmentation experiments are comprehensive and demonstrate strong results, the evaluation could benefit from additional explainability-specific metrics to strengthen the claims. More systematic faithfulness assessment beyond the current perturbation analysis would better validate whether explanations truly reflect the model's decision process. The evaluation would also benefit from completeness analysis to determine whether explanations capture all relevant visual information used by the model, and robustness testing to assess explanation stability under minor input variations or model parameter changes.
4. Technical Implementation Gaps
Several scalability concerns remain unaddressed in the technical implementation. The layer-wise gradient computation for each token may not scale well to longer sequences, potentially limiting practical applicability. Memory requirements for storing attention maps across all layers and tokens are not discussed, raising questions about the method's feasibility for resource-constrained environments. Real-time applicability for interactive systems remains unclear, particularly given the computational overhead of the gradient computations and attention map storage requirements.

### Questions
1)Could the authors provide justification for the intermediate logits conditioning scheme in Section 3.2 ? 
2)Justify the design choice of dynamic/reweighting filtering
3)Explain why grammatical words are deemed unimportant and provide evidence supporting SNR-based filtering
4)Include analysis of computational overhead and comparison with simpler alternatives
5) Could such metrics faithfulness, completeness be checked with  perturbation-based validation metrics to demonstrate the causal relationship between explanations and model decisions ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DEX-AR (Dynamic Explainability for AutoRegressive models), an explainability method for VLMs to evaluate the vision-answer token relevance.  The key contributions include: Dynamic Head Filtering, highlights attention heads that focus on visual information, filtering out irrelevant ones. Sequence-Level Filtering: further filters answer-sequence-level irrelevant noises, distinguishing between visually-grounded and other answer tokens.

### Strengths
1. Clear presentation and well writing.

2. Good generalizability: Demonstrates consistent performance improvements across multiple VLM architectures, including \textbf{decoder-only}, \textbf{encoder-decoder}. Outperforms baselines on both perturbation and segmentation tasks.

3. Comprehensive evaluation: Provides thorough analysis using diverse metrics like normalized perplexity, insertion/deletion tests, and segmentation IoU scores.

### Weaknesses
1. This paper shares similarities with TAM[1], which uses forward logits and causal inference to assess correlations between visual inputs, prompt texts, and answer sequence. The key distinction is the use of gradient evaluations across layers and heads. This paper should systematically compare the two approaches on: algorithm complexity; \textbf{technical differences and advantages}; test both methods on difficult scenarios (e.g., multi-object scenes, occlusions, ambiguous prompts). 

2. Gradient-based methods inherently require significant computational resources, especially for large transformer architectures.


[1] Token Activation Map to Visually Explain Multimodal LLMs

### Questions
If possible,

1, Show the necessary and effectiveness of filtering operations on different layers and heads;

2. Demonstrate methods to address the biases revealed in failure cases (e.g., over-reliance on background features or spurious correlations).

3. Qualitative and quantitative evaluation on hard cases (true useful for the community), for example, multiple similar objects in one picture,  occlusions and interactions among objects.

### Soundness
3

### Presentation
3

### Contribution
2
