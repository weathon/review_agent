# Inference-Time Personalized Safety Control via Paired Difference-in-Means Intervention

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6, 6

## Abstract
Safety preferences are inherently subjective, yet current LLM safety alignment methods often impose universal standards that fail to account for individual sensitivities. In this work, we propose an efficient, training-free method for personalized safety control via inference-time activation intervention. Our approach steers internal representations to suppress user-specific undesired content while preserving model utility. We systematically evaluate three strategies for estimating intervention directions: Instance-Level Contrast Shift (ILCS), Unpaired Mean Shift (UMS), and our primary method, Paired Contrast Mean Shift (PCMS). We provide theoretical insights into each approach and highlight the advantages of PCMS. Empirical results across diverse open-weight models demonstrate that our method effectively reduces undesired content in line with individual preferences, with minimal impact on helpfulness—enabling more adaptive and user-aligned LLM behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a training-free, inference-time activation intervention for personalized safety in LLMs. Building on LLM steering, it instantiates that toolkit specifically for safety control. The authors compare three direction-estimation strategies and provide a theoretical analysis showing that their paired, topic-matched estimator (PCMS) performs best under the stated assumptions. Experiments report stronger harmfulness reduction with competitive utility relative to prompt-based and activation-editing baselines.

### Strengths
The method is training-free, it requires no finetuning or weight changes, making it easy to deploy at inference time. The paper provides a comparison of three direction-estimation strategies and a theoretical analysis that largely aligns with the empirical trends. The results show a practical inference-time alternative to post-training approaches like SFT or RLHF.

### Weaknesses
1. The method is best viewed as a training-free, LLM steering approach to safety control, paired with PCMS, which leverages topic-matched paired activation contrasts to estimate a robust intervention direction. It is unclear how the proposed approach differs from prior LLM-steering techniques when applied to safety control.
2. While the paper targets personalized safety and advertises a training-free approach, the method still requires model- and facet-specific offline estimation of safety directions from paired prompts, which introduces additional setup cost. In practice, the implementation focuses on four safety facets (e.g., violence, political ideology, sexuality, mental health), so the claimed ‘personalization’ is category-level rather than truly user-specific.
3. The evaluation relies heavily on a single LLM to score both harmfulness and utility. Using multiple independent judges and a task-specific, fine-tuned scoring model (calibrated with human annotations) would be much more robust and credible.
4. The paper does not investigate whether a safety direction estimated on one model transfers to other models. If transfer does not hold, the approach still requires per-model direction estimation and calibration, adding non-trivial setup cost and limiting practical reuse.
5. The paper’s theory broadly tracks the empirical trends but relies on stylized assumptions, most notably an additive decomposition of activations into a topic component plus a harm component with zero-mean noise, together with a linear harm scorer. These choices make PCMS analytically clean, yet the results mainly rationalize the estimator ranking rather than fully explaining why the intervention works in the non-linear, sequence-level setting of real LLMs. The design of PCMS is simple and reasonable, but the theoretical guarantees may not extend beyond these assumptions.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes an efficient, training-free method for personalized safety control in large language models (LLMs) through inference-time activation intervention, referred to as PCMS. Theoretically and empirically, we demonstrate the superiority of PCMS over existing estimators in steering model outputs toward safety alignment.

### Strengths
I find the authors' premise that "safety is subjective" to be well-motivated. Consequently, the research question they pose—How can we equip language models with personalized safety controls that reflect individual sensitivities without retraining, heavy data requirements, or compromising efficiency at inference time?—is both timely and significant.

### Weaknesses
+ The description of the methodology lacks clarity in several key aspects. Most notably, it remains unclear how the activations for two of the three proposed strategies are derived or defined, which is critical for understanding and reproducing the approach.
+ The experimental evaluation relies primarily on datasets like SHP and the harmless portion of PHTest, which consist of content with relatively low levels of harm. While I understand the authors' intention to demonstrate personalized safety control, this choice raises concerns about whether the method can ensure safety under more stringent, conventional definitions. Furthermore, the defined safety facets—violence, political ideologies, sexuality, and mental health—largely reflect areas with broad societal consensus. True subjective safety, as illustrated by the "historical revolution" example in the introduction, often arises in more contested contexts such as historical narratives or fictional settings (e.g., games or movies). Evaluating the method on facets where subjectivity is more pronounced would better substantiate the paper's core claim.

### Questions
1. If I understand correctly, the activations for two of the strategies are derived from the prompts shown in Table 1. However, the distinction between the "test prompt" and the "reference prompt" is not sufficiently elaborated. A flowchart or a concrete, step-by-step example illustrating how the average harm-difference vector is estimated and applied during inference would greatly clarify this process and enhance the reproducibility of the method.

2. To strengthen the empirical evaluation, I recommend that the authors include benchmark results from established safety datasets. This would more convincingly demonstrate the method's foundational safety alignment capabilities. Specifically, the BeaverTails dataset [1], which offers a comprehensive analysis across fine-grained safety categories, would be a valuable addition. Furthermore, evaluating on the virtual-scenario-based samples from Xstest [2] could effectively showcase the method's ability to perform personalized control in complex, edge-case scenarios. Incorporating these experiments would directly address my primary concerns regarding the method's basic safety guarantees and its efficacy in more challenging, subjective contexts.

3. ​​Minor Points for Clarification:​
  
   (a) Notation Consistency:​​ Please clarify if $\gamma$ (experiments) and $\alpha$ (Eq. 1) represent the same parameter.

   (b) Figure 3 Axis:​​ The direction of the arrow on the x-axis seems reversed; it might be more intuitive if it pointed left.

[1] Ji, Jiaming, et al. "Beavertails: Towards improved safety alignment of llm via a human-preference dataset." Advances in Neural Information Processing Systems 36 (2023): 24678-24704.

[2] Röttger, Paul, et al. "Xstest: A test suite for identifying exaggerated safety behaviours in large language models." arXiv preprint arXiv:2308.01263 (2023).

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
3

### Summary
This paper introduces a inference-time activation steering method for personalized safety control by adding vectors that neutralize harmfulness to the model activation. The authors tested three alternative methods to steering: ILCS, UMS, and PCMS, in which PCMS performs the best in both mitigating harm and preserving helpfulness. Compared to prompt-based methods, PCMS also demonstrated overall higher effectiveness: it reduces harm more than DP, but at the cost of helpfulness.

### Strengths
The proposed method is novel and targets the important problem of personalized safety. Existing approaches often relies on prompting to steer LLMs for personalization without editing the activations of the model. On the other hand, this paper explores the less used tool of activation engineering to solve this issue. The evaluation shows that the proposed method presents a strong alternative to prompt-based methods, with particularly impressive improvements on harmlessness. The paper is relatively clearly written (though the theoretical analysis section could be made less dense and unpacked more to facilitate understanding).

### Weaknesses
Although the proposed method is novel and seems effective, I have a few major reservations about this work:

1. The method is not validated on real human user data, which makes it challenging to judge the usefulness of the approach in real-world practices, since it's framed as a individualized safety control method for users with different values and preferences. At the minimum, whether the cases used in the experiments represent realistic human safety preferences should be validated. Human data could also inform the definitions of the harm facets. I think showing that PCMS works well on real user data will substantially improve the contribution and practical relevance of this work.

2. Missing evaluation on general capability before vs after activation engineering. PCMS still makes the assumption that there exists a universal "harm vector" that is consistent across all topic distributions, which may not be true. The paper only tests 4 categories (violence, political ideologies, sexuality, and mental health), which the steering direction is also computed based on. It's not guaranteed that helpfulness would be equally well preserved in other categories (e.g., general everyday queries). Additional experiments evaluating the general capability of the model before and after activation engineering will provide helpful context.

### Questions
1. Figure 3: why does the error next to "Harmfulness Score (Lower is Better" point rightward instead of downward or leftward?

2. Table 3: Although it's clear that PCMS consistently decreases harmfulness from DP across models, it seems that it also consistently impact utility more severely than DP. If you use a composite metric that accounts for both utility and harmlessness (like in figure 3), will PCMS still be consistently better? If so, I suggest using such a metric instead. Alternatively, please explicitly acknowledge the impairment on utility.

3. There is a trade-off between utility and harmlessness - this paper uses a linear combination between the Utility Score and Harmfulness Score (Figure 3) to quantify this tradeoff, arguing that PCMS is better than other methods by this composite metric. Could you justify why this particular, seemingly arbitrary combination of Utility Score (1-10) and Harmfulness Score (1-5) is a good representation of the tradeoff between harmfulness and helpfulness? Again, I think a more robust approach is to gather human preference data on pairs of outputs generated by different methods (e.g., DP vs PCMS) - if annotators prefer PCMS over other methods, this could be strong evidence that PCMS is better than alternatives.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies personalized safety control for LLMs. They steer the LLM outputs toward safer responses by adding to the LLM activations at inference time, making this a training-free method. In particular, their steering direction approximates the personalized harm directions, rather than move along pre-defined behavior axes. They propose three different strategies: ILCS, UMS, and PCMS, argue that the first two have issues while PCMS tries to combine the strengths of those two. They conduct studies on topics in violence, politics, sexuality, and mental health and show that their method is superior.

### Strengths
* The problem of safety alignment in LLMs is an increasingly important topic as LLMs become ubiquitous. 
* The proposed approach is lightweight since it does not require training and directly steers LLM activations at inference time. 
* The experimental evaluation shows competitive performance of PCMS, achieving higher utility and lower harmfulness than baseline methods.

### Weaknesses
* I feel a few of the mentioned limitations should actually be ablations for this work instead of future work. For example, the authors choose the middle layer activations to conduct steering, but it would be very helpful to investigate how the performance shifts when steering on different layers.
* In addition, I think the authors should conduct an ablation on different step sizes α and comparing them to the proposed multi-facet safety control.

### Questions
* Line 080 and 093: Is $d_{\text{int}}$ overloaded? It's the optimal unit direction and also harm direction.
* Line 170: how is this additive assumption enforced? It seems model-dependent.
* Figure 2, what is the color bar?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a training-free method for personalized safety control in LLMs. The authors argue that universal safety alignment fails to account for individual user sensitivities. To address this, they propose an inference-time activation intervention framework that modifies internal model activations to suppress undesired content based on user-defined preferences (e.g., related to politics, violence). The main contribution is the Paired Contrast mean shift method, which estimates a "harm direction" in activation space using paired examples of harmful and harmless prompts. This direction is then used to steer model outputs away from undesired content during inference

### Strengths
-Works at inference time without retraining or large datasets.
-Reduces harmful content while keeping responses useful.
-Provides formal proof that PCMS is unbiased and optimal under mild assumptions.

### Weaknesses
-No automated way to pick the best layer for intervention.
-Sometimes makes answers too cautious or less helpful.
- Relies on LLM-as-a-judge scoring, not human judgment, no ablation for the scoring was given.

### Questions
- How reliable is the scoring mechanism? Could the authors manually match a subset of the evaluation set to see how reflective the GPT scores are? Or use two different models for this part to enable some “nuance” to reflect the personalization aspect. 
- How could layer selection be improved in future work?

### Soundness
3

### Presentation
3

### Contribution
3
