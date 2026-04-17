# Segment-Level Attribution for Selective Learning of Long Reasoning Traces

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Large Reasoning Models (LRMs) achieve strong reasoning performance by generating long chains of thought (CoTs), yet only a small fraction of these traces meaningfully contributes to answer prediction, while the majority contains repetitive or truncated content. Such output redundancy is further propagated after supervised finetuning (SFT), as models learn to imitate verbose but uninformative patterns, which can degrade performance. To this end, we incorporate integrated gradient attribution to quantify each token's influence on final answers and aggregate them into two segment-level metrics: (1) \textit{attribution strength} measures the overall attribution magnitude; and (2) \textit{direction consistency} captures whether tokens' attributions within a segment are uniformly positive or negative (high consistency), or a mixture of both (moderate consistency). Based on these two metrics, we propose a segment-level selective learning framework to identify important segments with high attribution strength but moderate consistency that indicate reflective rather than shallow reasoning. The framework then applies selective SFT on these important segments while masking loss for unimportant ones. Experiments across multiple models and datasets show that our approach improves accuracy and output efficiency, enabling more effective learning from long reasoning traces.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper study on the segment selection in long chains of thought (CoT), they propose a series of new metrics: attribution strength and direction consistency, from integrated gradient attribution perspective. Based on these two metrics, this paper proposes a segment-level selective learning framework to identify important segments with high attribution strength but reflective than shallow reasoning. The framework then applies selective SFT on these important segments while masking loss for unimportant ones. Experiments across multiple models and datasets show the approach improves accuracy and output efficiency, enabling more effective learning from long reasoning traces.

### Strengths
Compared to prior studies which either neglects semantic integrity and fails to interpret redundancy or ignore the indirect effect to the answer, this paper integrated gradient attribution to analyse segment's effect in depth.

### Weaknesses
The mechanism in section 2.2 "INTEGRATED GRADIENTS BASED SEGMENT IMPORTANCE" should be discussed more, see questions part for details.

### Questions
1. What is the definition of confidence in Fig. 2 and Fig. 4.
2. In equation (2), how exactly is IG be calculated from hidden state to outcomes, i.e. the partial derivative, how to define J in practice. If it includes frequently altering input to acquire output score, computation cost should be discussed.
3. Most of the experiment is conducted under Qwen-family framework, which leave the possibility that this method benefits from the model's characteristics cannot be completely ruled out. Experiment under different model architecture is suggested.
4. Is there more application scenarios could be employed by those IG metrics.

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
4

### Summary
This paper introduces a segment-level attribution framework for selectively learning from long chain-of-thought (CoT) traces produced by large reasoning models. The approach leverages integrated gradients to assign an attribution “strength” and “direction consistency” to each CoT segment, identifying those segments with substantial and mixed influence on answer prediction. Selective supervised fine-tuning is then applied on only the important segments while masking loss for uninformative parts. 

Experiments on multiple models and mathematics/reasoning datasets show improved accuracy and output efficiency versus standard full-CoT fine-tuning. The paper is supported by careful empirical and ablation studies and provides a detailed analysis of redundancy in CoTs.

### Strengths
1. Evaluates both pruning-based and selective learning variants, tests random and prior-importance alternatives, and shows that selective learning is preferable to simple pruning, with the IG-based method outperforming confidence, entropy, and perplexity-based segment selection (Table 3, Page 8).

2. The use of normalized strength and moderate consistency to select segments is justified theoretically and supported by empirical analysis (Section 3.1, Figures 2 and 3; the ablation study in Table 2, Page 8, directly supports the effectiveness of the joint criterion over alternatives).

3. Figures are used to substantiate both the intuition and the rigor (e.g., Figure 1 summarizes the distribution and accumulation of attribution across segments, while Figure 3 provides boxplots linking log perplexity, entropy, and BLEU similarity between important/unimportant segments).

4. Proposes segment-level attribution metrics based on integrated gradients—specifically, attribution strength and direction consistency—combined in a well-motivated and interpretable manner. The mathematical formulations (Equations on Pages 3–4) are clearly specified and grounded in attribution theory.

### Weaknesses
> Completeness of the chosen keywords for segmentation

Are the keywords used for segmentation comprehensive? How were the keywords selected? The generalizability of this paragraph segmentation method across different models (e.g., DeepSeek-R1 and Qwen2.5-7B) requires further discussion.

> Limited theoretical analysis of attribution methods

Although the attribution metrics are intuitive and based on empirical motivations, there is insufficient discussion of their theoretical limitations or the potential for attribution leakage. For instance, integrated gradients may produce artifacts for redundant input or model features, which could falsely elevate or lower paragraph scores. There is no thorough analysis or mitigation (e.g., using SmoothGrad) to reduce variance or noise in the attribution process.

> Threshold sensitivity and generalization

The critical thresholds for attribution strength and consistency (i.e., $\tau$ and $\beta$) are set via greedy search. However, the analysis of their stability or transferability to other datasets or inference types beyond mathematical QA is limited. Figure 5 (on page 15) addresses threshold effects, but only for one model/dataset, and does not provide a detailed analysis of generalization robustness.

> Insufficient evaluation metrics for diversity sampling.

In the results presented in Table 1, the pass@k value for temperature sampling is not provided, which is insufficient to fully capture the performance of the method across different benchmarks.

> Limited performance improvement

As shown in Table 1, for certain benchmarks, the accuracy after segment-selective SFT training is either lower or unchanged compared to the full CoT.

### Questions
1. Is the method for selecting keywords reasonable? How are the keywords chosen? Does this method apply to various models of different types?

2. Hyperparameter search in Sec. 3.2: The hyperparameter search method is conducted only for a single model and dataset. If the model or dataset is changed, is it necessary to re-search the hyperparameters? Given that the hyperparameter search requires first using the model to calculate the accuracy gain on the tasks, does this result in significant cost?

3. Applicability to other domains: Is this method still reasonable in other domains, such as code or general domains?

4. Failure case analysis: Have you reviewed any cases where your segmentation choices led to a performance drop in the model? Could such failures encourage further improvements to the masking strategy’s features?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
They introduces a method to improve learning from CoT reasoning sequences in LRMs. It identifies which segments of a reasoning trace contribute to the correct answer using segment-level attributions derived from Integrated Gradients. Segments are evaluated for attribution strength and direction consistency, and only important segments are used for supervised fine tuning.

The key contribution is demonstrating that models can learn more effectively by focusing on the reasoning steps that truly matter.

### Strengths
good analytical experiments
reliance on gradient based method rather LLM as judge
The use of Attribution Strength combined with Attribution Direction Consistency provides a mechanism to distinguish critical reasoning from superficial or shallow content
efficiency gains in terms of reasoning length (while having high/comparable accuracy)
Comprehensive ablation studies

### Weaknesses
- Focus on only one LLM (Qwen)
- High computational cost for importance calculation (J=50)
- Sensitivity to hyperparameter selection: the performance of the framework hinges on two crucial, empirically determined thresholds: τ (strength threshold) and β (consistency threshold). The hyperparameter search indicates that model performance is sensitive to these values, as choosing a higher τ introduced more false positives and negatively impacted training performance. The selection process (maximizing the difference in confidence gain between important/unimportant segments) is justified, but sensitivity remains a potential concern when moving to drastically different models or tasks.
- The segment partitioning relies on a pre-defined set of common transition keywords (e.g., “\n\nWait”, “\n\nAlternatively”)
- best performance is not indicated in the tables

### Questions
1. you mention that: "We estimate the correct answer confidence by multiple temperature-sampled generations". how do you calculate confidence?
2. Why did you choose to work in segment level? why not token level?
3. you mention that: "Extremely consistent attribution directions (all positive or all negative) may indicate shallow reasoning (e.g., dispensable explanation of final answers) while moderate consistency often reflect reflective and critical reasoning patterns where the model explores different possibilities, refines its understanding, and eliminates incorrect content, which are more valuable for learning effective reasoning strategies.". How do you claim this?
4. Could you please provide a quantification of the computational cost of the IG attribution step? Specifically, relative to the total time required for a standard Full CoT SFT epoch, how much overhead does the segment attribution calculation add? 
5. The IG values were calculated using R1-Distill-Qwen2.5-7B. Did you explore how effective the segments identified by this 7B model are when fine-tuning much larger or smaller models, or models from entirely different families (i.e., treating the attribution model as an external tool)? This would help demonstrate the robustness of the identified importance definitions beyond a single model lineage.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a segment-level selection method for CoT reasoning that identifies and trains only those segments that meaningfully contribute to correct answer generation (for SFT). Using Integrated Gradients (IG), the authors measure each segment’s attribution to the final answer and summarise this using two metrics: Attribution Strength and Attribution Direction Consistency. An interesting design choice is that Attribution Strength is defined using the absolute IG values (rather than the signed values), and that the method selects segments with moderate, rather than high, Direction Consistency. The stated motivation is that (i) even reasoning that initially explores incorrect directions may still be useful later, and therefore should not be discarded purely because it has negative IG, and (ii) segments with very high consistency may reflect shallow or overly biased reasoning rather than careful reflection. Across multiple models (R1-Distill-Qwen2.5-1.5B/7B, Qwen2.5-7B-Instruct) and datasets (LIMO, MATH500, AIME, etc.), the proposed approach improves accuracy while reducing output length.

### Strengths
- It is interesting to see the use of IG at the segment level to score the importance of different parts of the CoT. The definitions of Attribution Strength and Attribution Direction Consistency are thoughtful in that they are not purely driven by raw gradient magnitude, but instead attempt to capture properties specific to reasoning structure.

- Through extensive experiments, the authors present hyperparameter search, ablations, and comparisons with alternative importance measures.

- The writing is clear and the figures effectively illustrate the ideas and motivation.

### Weaknesses
- The argument for using attribution strength based on the absolute value of IG may need to be strengthened. The paper motivates this choice by noting that exploratory reasoning can sometimes have negative IG values, and that such reasoning should not be thrown away. However, consider a segment that overall shows strongly negative IG values and only moderate consistency. Is it necessarily the case that such a segment corresponds to “necessary exploratory reasoning” (lines 155–156)? While useful exploratory reasoning may indeed yield negative IG values, it is less clear that the reverse implication always holds. In other words, segments with strong negative attribution are not guaranteed to be beneficial; they may simply reflect confident but unhelpful reasoning.

### Questions
- How were the segmentation keywords in Appendix C.2 selected? The transition keywords used here appear to differ from the list in Lu et al. (2025). Have the authors evaluated the effect of varying this keyword list on the final results?

- In Table 2, could the authors provide an additional comparison for the “High Strength Segments” condition where segments with very large negative IG attribution are excluded? This would help clarify whether strongly negative-attribution segments are genuinely useful or whether they can be safely filtered out.

### Soundness
3

### Presentation
3

### Contribution
4
