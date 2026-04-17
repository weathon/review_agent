# Can I Trust Your Visual Thinking? Measuring and Improving Visual Thinking Faithfulness

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Recent large vision–language models (LVLMs) can generate vision–text multimodal chain-of-thought (MCoT) traces after reinforcement fine-tuning (RFT). However, we observe that the visual information incorporated in MCoT is often inaccurate, though still yield correct answers, indicating a lack of faithfulness in the MCoT reasoning process.
We attribute it to *the RL reward that only incentivizes the format of interleaved vision-text cues*, i.e., incorporating visual information into text reasoning steps. In this paper, we first probe the faithfulness of MCoT by measuring how much the prediction changes when its visual and textual thoughts are intervened or corrupted. Surprisingly, the model's predictions remain nearly unchanged under visual intervention but change significantly under textual intervention, indicating that **visual evidence is largely ignored**.
To further diagnose visual information, we introduce an automated LVLM-based evaluation pipeline that quantifies the faithfulness of visual cues from two perspectives: reliability and sufficiency.
Our evaluation reveals that the visual information in current MCoT traces are simultaneously unreliable and insufficient.
We then propose a novel MCoT learning strategy to address this issue, termed Sufficient-Component Cause Model (SCCM) learning, which encourage the MCoT to generate sufficient yet minimal visual components that are able to independently lead to correct answers.
We note that the proposed SCCM is annotation-free and compatible with various RFT training for MCoT in a plug-and-play manner. Empirical results demonstrate that SCCM consistently improves faithfulness metrics across a suite of fine-grained perception and reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the faithfulness of multimodal chain-of-thought (MCoT) reasoning in existing large vision-language models (LVLMs). Through preliminary experiments, they find that the visual evidence used in model-generated reasoning is often unreliable, insufficient, and has minimal impact on final results. To address this, they propose a Sufficient-Component Cause Model (SCCM) learning method, which encourages models to generate sufficient and minimal visual evidence that directly supports correct answers. Experiments demonstrate that SCCM improves several faithfulness metrics across several LVLM evaluation benchmarks .

### Strengths
1. The paper explores multimodal reasoning faithfulness from a novel and important perspective—the reliability of visual evidence in reasoning processes. This direction has been largely underexplored in prior work.


2. The evaluation pipeline is straightforward and methodologically sound. The intervention-based analyses are particularly insightful, showing that modifying visual evidence often has negligible effects on model outputs, thereby highlighting a critical and previously overlooked limitation in current multimodal reasoning models.

### Weaknesses
1. The citation format is wrong. As suggested by the paper template: “When the authors or the publication are included in the sentence, the citation should not be in parenthesis, otherwise, the citation should be in parenthesis.” However, all citations use the former style, regardless of context.  

2. The faithfulness analysis is conducted only on small-scale models. It remains unclear whether the observed issues persist in larger or more capable models.

3. Although SCCM improves faithfulness metrics, it yields minimal gains in overall task performance. As shown in the first row of Table 1, SCCM performs worse than its baseline (Pixel-Reasoner) on HR-Bench 4K and HR-Bench 8K. Furthermore, it is unclear whether the accuracy of Pixel-Reasoner in Table A5 is different with Table 1.

4. The ablation studies report only the faithfulness metrics without presenting the final model performance. Including overall results would better illustrate the relationship between reasoning faithfulness and end-task performance.

### Questions
1. The proposed SCCM method shows substantial improvements on faithfulness metrics but only marginal gains—or even performance drop on overall benchmarks. Could the authors clarify the underlying reasons for this discrepancy? This is my primary concern; I would consider raising my score if the authors could adequately address this issue.
2. Table 1 and Table A5 report different results for Pixel-Reasoner but identical results for DeepEyes. Could the authors explain this inconsistency?

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
5

### Summary
This paper studies the faithfulness of current "thinking with images" methods, including reliability and sufficiency. Interestingly, this paper finds out that current models, such as DeepEye and Pixel-Reasoner, are much more intensive with textual intervention than visual intervention. This implies that the visual evidence of these models is *not* faithful enough. Therefore, the authors propose two extra rewards, including Visual Information Sufficiency and Visual Information Minimality. Empirical results demonstrate both the core motivation and the effectiveness of the proposed method.

### Strengths
1. This paper is well-written and easy to follow.
2. The core motivation is quite clear and is supported with sufficient empirical evidence, e.g., Table 1.
3. The newly proposed Reliability and Sufficiency evaluations are interesting, important, and valuable.

### Weaknesses
Overall, this paper is qualified. Therefore, I only have some minor concerns/suggestions.

1. Actually, the motivation between this paper and [1] is *partially* similar. Therefore, a discussion should be included, e.g., 
    - [1] actually uses an extra IoU reward to improve the faithfulness of visual evidence, while this paper utilizes the "Visual Information Sufficiency" reward.
    - [1] takes a *dual* IoU reward to avoid the "trivial solutions", while this paper uses the "Visual information Minimality" reward.

2. [1] proposes TreeBench, where explicit bounding boxes for target objects are provided. Evaluating the proposed method on TreeBench and finding out whether it achieves a much higher mIoU than baselines is strongly encouraged.

3. Details of interventions are unclear. I only see Figure A1 as an example, but detailed configuration is not provided.

**References**

[1] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology. arXiv 2025.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles the problem of faithfulness in visual reasoning chains produced by Vision-Language Models (VLMs). The authors introduce Counterfactual Attribution as a method to assess whether intermediate steps in a model's visual thought chain (e.g., generated captions or rationales) are causally linked to the final answer. The authors propose a new metric: Counterfactual Faithfulness Score (CFS), which measures how much altering an intermediate thought (while holding the image constant) affects the model’s final prediction. The method is evaluated across multiple VLMs (e.g., GPT-4V, MiniGPT4, Qwen-VL) and benchmarks, revealing that many state-of-the-art models hallucinate or over-rely on unfaithful intermediate steps.

### Strengths
1. Addresses the critical question of whether visual explanations are trustworthy, a concern for deploying VLMs in sensitive domains.
2. Uses counterfactual interventions inspired by causal inference to isolate the effect of intermediate reasoning steps.

### Weaknesses
1. While the paper proposes CFS as a measure of faithfulness, it lacks ground truth causal chains to validate that the counterfactual interventions accurately reflect (un)faithful reasoning.
2. The way counterfactual thoughts are generated (e.g., modifying a rationale or removing an attribute) is heuristic and brittle. This opens the method to: 1) non-natural counterfactuals 2) over-dependence on token-level edits that do not semantically change meaning, which happens frequently.
3. CFS is defined as the answer change under counterfactual intervention. However, large language models exhibit stochastic behavior, and, in some cases, minor phrasing changes can lead to different outputs even with identical reasoning. The current experimental setting cannot verify whether there are cases that the output is effected by stochasticity but classified as intervention.
4. Experimentally, it is not shown which types of intermediate steps (e.g., attribute detection, spatial reasoning, object naming) are more prone to unfaithfulness. Such granularity would be important to pinpoint where the failure lies in visual chains.
5. The study focuses only on chain-of-thought-style VLMs that output text. It is not clear whether the proposed CFS metric generalizes to non-autoregressive or encoder-only models. The authors are encouraged to adapt CFS to commonly-seen models like Flamingo or Kosmos-2.5 where textual thoughts are not explicitly generated, to show the actual effectiveness of the method.

### Questions
See Weaknesses Above.

### Soundness
1

### Presentation
2

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
The paper studies the problem of the unfaithfulness of visual-text MCoT in LVLMs. It provides a core finding: irrelevant visual reasoning chains exist in current models and may produce accurate answers. This counterfactual phenomenon indicates that the CoT is dominated by textual cues rather than visual ones. To force MCoT to be visually related, the authors propose a method called the Sufficient-Component Cause Model, which encourages visual evidence to be sufficiently correct and less redundant.The experiments show that the proposed model imporves the faithfulness of LVLMs.

### Strengths
- Reasoning faithfulness is an important research topic, especially in the era of black-box LLMs, as it helps improve the interpretability of model behavior and potentially encourages accurate answers.
- The causal intervention analysis provides some insights.
- The paper provides a faithfulness evaluation that is different from previous accuracy-oriented evaluations.

### Weaknesses
- Lack of deep analysis of the relationship between accuracy and faithfulness.
    1. First, I want the authors to detail the source of hallucinations in RL learning. The introduction mentions that “inaccurate and insufficient visual information in MCoT may still yield definite, even accurate, answers, suggesting that the MCoT can be unfaithful.” Why? This means the black-box model can perform the task well but does not work in a human-like, step-by-step manner. Why does the accuracy-based RL reward create a bias towards such hallucinations instead of visually related cues? Intuitively, the presence of wrong but interleaved visual cues cannot lead to better rewards. I believe the authors should provide an in-depth explanation of this.
    2. From a quantitative standpoint, the paper mainly provides a faithfulness evaluation but lacks a sufficient analysis of the accuracy (of the final answer). Will faithfulness-based RL training help accuracy? What is the relationship between these two objectives?
- Reliability assessment. As the authors state: “reliability means whether the input visual components are reliable for the model’s prediction.” and “reliability directly reflects the causal consistency between V and A.” However, it is not clear why the prompts in Sec. A.1.2 reflect causal consistency. I suggest the authors provide more explanation on this. Moreover, I believe the causal consistency evaluation is a complex task requiring strong reasoning ability. GPT-4o may not work as well as expert reasoning models like o3/o4. The quality of the assessment should be discussed and evaluated.
- Format inconsistency. For example Sometimes using "Sec. 6.2," sometimes using "Sect."

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
