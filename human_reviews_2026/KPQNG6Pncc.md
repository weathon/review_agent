# Distill Models by Aptitude: Efficient Reasoning Capability Distillation via Adaptive Data Curation and Overthinking Mitigation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 2, 8

## Abstract
The exponentially increasing computational demands of large language models (LLMs) facilitate the distillation of knowledge or capability to smaller models. Existing distillation attempts to transfer LLMs' reasoning capabilities to compact models face critical limitations, including expensive training or annotation costs, suboptimal data selection, and flawed synthetic data due to LLMs' general tendency to overthink. This paper introduces DynaGuide, a novel framework that optimizes the distillation process in both efficiency and performance. Our approach integrates (1) Dynamic Data Selection that adaptively performs fine-grained valuable data selection during the training process, and (2) Reasoning Pattern Guidance that mitigates the overthinking problem in synthetic data by incorporating specialized guidance during fine-tuning. Extensive experiments demonstrate that DynaGuide consistently achieves stable performance improvements across models of different series and parameter scales, with gains surpassing those of baseline methods on knowledge reasoning question answering benchmarks. Our systematic ablation studies and analysis further provide valuable insights into distillation and reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors proposed two technologies to improve the knowledge-based reasoning distillation tuning. For data selection, a dynamic weight-based sample method is proposed. For reasoning pattern learning, it analyzes the original reasoning pattern and uses a project align loss to guide the learning process. Experiments have demonstrated the effectiveness.

### Strengths
The method proposed in the paper has a good effect and provides some inspiration for the inference analysis of the R1 model in the community.

### Weaknesses
1. The method requires a more detailed introduction:

1-1. How to sample based on the weights of different categories？

1-2. How are categories divided? Will the fine-grained classification of different categories affect the effectiveness and efficiency of the algorithm? How to deal with the problem of multiple labels in many data classifications？

2. The experimental part needs further improvement.

2-1. It is recommended to provide Table 2 with the performance of the model trained with full data, and compare the performance of the model trained with the sampled data, to more effectively demonstrate the effectiveness of the sampled data.

2-2. It is recommended to compare more sampling methods. Here are some examples that are not necessarily necessary to compare [1] [2].
[1]. Self-Evolved Diverse Data Sampling for Efficient Instruction Tuning
[2]. What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning

2-3. The 32B model in Table 2 is a chat model. It is recommended to compare it with QwQ-32B, which can better improve the ability of the trained 7B model.

### Questions
Additional questions:

1. Will the proportion of warm data affect the distribution or quality of sampled data? Do we need corresponding experiments to verify this?

2. Has category balance also been taken into account when sampling by length in Section 6.1?

3. The proposed mapping loss and SFT training loss seem to conflict. Although the features may be closer to the execution step, SFT training still predicts the current token as the data itself. The comparison in Figure 2 also illustrates this point, as the proposed method has limited improvement in the execution step ratio compared to the other two methods. Similarly, does this also indicate that the more execution steps, the better? There may be a reasonable range.

### Soundness
3

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
4

### Summary
The authors propose DynaGuide, a distillation framework which aims for more efficient distillation of knowledge from LLMs to SLMs and proposes a fix to mitigate overthinking. The fix for overthinking is called reasoning pattern guidance (RPG) and is an additional loss term to encourage the hidden state of thinking tokens to align with non-thinking output tokens.

The efficient distillation is performed by a dynamic data selection (DDS) which the authors put forward. The authors assume that each data-point belongs to a particular class or domain. These domains are dynamically weighted according to their loss, similar to previous domain re-weighting schemes such as Doremi [1].

The authors make the interesting observation that incorrect Q&A answers have fewer execution steps than for correct answers. (I have reservations with this analysis, see the weaknesses section). As such the authors propose encouraging more execution steps during finetuning with a Reasoning Pattern Guidance loss term which encourages the hidden state vectors of the for reflection and transition steps to align their directions closer to the average hidden state from all the tokens in an execution step. The authors claim that this encourages more execution steps which are a feature of correctly answered questions.

On a variety of Q&A tasks the authors show that distilling DeepSeek-R1 answers into SLMs with their methods gives overall the best performance. The authors also ablate their data selection method and compare it to various methods for encouraging execution only steps.


[1] Xie, Sang Michael, et al. "Doremi: Optimizing data mixtures speeds up language model pretraining." Advances in Neural Information Processing Systems 36 (2023): 69798-69818.

### Strengths
* The observation that the number of tokens does not increase for knowledge question and answers tasks for correct and wrong answers in Table 1 is very interesting. The classification into different types of steps: execution, reflection and transition is also interesting, however requires elaboration (noted in the weaknesses section).
* The DDS method for selecting data for distillation makes a lot of sense and empirically shows strong performance versus baselines like s1 [1]. However is it limited to datasets which contain different classes?

[1] Muennighoff, Niklas, et al. "s1: Simple test-time scaling." arXiv preprint arXiv:2501.19393 (2025).

### Weaknesses
My main issue is with the presentation of the paper. 
* There are many parts of the paper which are too vague:
    * Lines 144-148. What are divergent learning dynamics? “Certain domains or complexity levels require much exposure for adaptation”, what are domains or complexity levels, what do you mean by exposure?
    * Lines 202-203: how have you categorized the step types? By hand? using a classifier? There is no description of what these three different step types are or examples of these step types.
    * I cannot read the legend and labels in Figure 2.
    * Lines 233-239: there seem to be some very important observations which motivate RPG here but they are not developed well enough or details are in the appendix. For example: the weak separability is with respect to the  ‘\n\n’ tokens and what other tokens? Randomly sampled tokens from other parts of the same sequence?
* Notation is introduced which is not explained. For example what is $W$, $W_0$ and $c_i$ in Algorithm 1?
* Awkward grammar:
    * The title in Section 4.
    * Line 365: “we also report the radius of 95%”, what is a radius in this context?
* Simple spelling mistakes e.g. line 244 - “Spefically”.
* In some rows in Table 3 you use means and standard errors however in others you use a single number. If possible stick to means and standard errors everywhere, although I understand that this requires more GPU resources.



I do not agree with the reasoning in lines 226-230, I could be wrong though. The authors argue that because there is a decrease in the number execution steps and an increase in reflection and transition steps that this **causes** the model’s reasoning process to fail and hence the wrong answer is obtained. Is this not a case of a correlation being misinterpreted as causation? I would have liked to see more diagnosis here. Maybe the LM doesn’t know the answer and so this results in increased reflection and transition steps? This hypothesis is easily verifiable with some made up knowledge questions and answers. I’m not convinced by the reasoning here and this puts into question the entire RPG method.

### Questions
* Can you perform dynamic selection with datasets which do not contain different defined domains or classes for example GSM8k?
* What value of $\alpha$ do you use? I would like to see an analysis of how the strength of the RPG loss affects the number of execution steps.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces DynaGuide, a framework for efficiently distilling reasoning capabilities from large language models to smaller models through two main components: (1) Dynamic Data Selection (DDS), which adaptively selects valuable training samples during the first epoch of fine-tuning based on loss-driven weight adjustments across data feature classes, and (2) Reasoning Pattern Guidance (RPG), which addresses the overthinking problem in LLM-generated reasoning traces by introducing a projection loss that encourages execution steps over reflection and transition steps during fine-tuning. The authors demonstrate improvements across multiple model families (Qwen2.5, LLaMA-3.1, Qwen3) on knowledge-based question answering benchmarks using only 1,000 selected training examples from DeepSeek-R1 reasoning traces.

### Strengths
1. The paper addresses two important problems in reasoning capability distillation: accelerating the distillation process through data curation and mitigating the overthinking issue in synthetic reasoning traces generated by LLMs.
2. The experimental evaluation spans multiple model families (Qwen2.5, LLaMA-3.1, Qwen3) and different parameter scales, demonstrating that the approach is not limited to specific LLM architectures.
3. The paper includes ablation studies examining each component (DDS and RPG) separately, providing insights into their individual contributions to overall performance.

### Weaknesses
1. It is unclear why the paper proposes both DDS and RPG as a combined approach when they appear to be orthogonal components applied to different parts of the training pipeline. The relationship and necessity of using both together is not well justified.
2. Algorithm 1 mentions "FeatureClassOf" and the paper discusses data feature classes, but these concepts are not properly defined or explained elsewhere in the paper, making the implementation of the algorithm unclear.
3. In several places, the paper presents hypotheses to justify their methods without providing evidence or citations. For example, the cold start problem and issues with proper initialization in Section 3.1, weight explosion/vanishing in Section 3.2, and claims in Section 4.2 (after Equation 2) lack supporting evidence. Clear writing with motivating figures or citations for these hypotheses would strengthen the paper.
4. Experimental setups are not fully described. For example, in Table 1, it is unclear which DeepSeek-R1 model was prompted, with what prompt, on how many questions from the QA datasets, and how the percentages for execution/reflection/transition steps were calculated (LLM-as-a-judge, manual annotation, or rule-based analysis). Without fully specified experimental settings, the arguments in Section 4.1 motivating RPG are not completely sound.
5. Table 2 lacks important baseline comparisons. Each model (Qwen2.5-7B, LLaMA-3.1-8B, etc.) should show results when fine-tuned with only DDS and only RPG separately before showing the combined results. Additionally, any model distilled with 1,000 reasoning traces from the R1 model should demonstrate better performance on the evaluation sets in Table 2 due to in-domain fine-tuning, but these baseline experiments are not included (note: this is distinct from the s1 baselines).
6. The paper needs extensive ablation experiments on the number of training examples for DDS (currently fixed at 1,000) and the layer selection for applying RPG projection loss, which varies across different models. It is also unclear whether RAG evaluation is based on dense embedding search for documents or keyword lookup.

### Questions
1. What do the yellow lines represent in Figure 1? There is L_CLM shown in the blue box and L_total in the yellow box—are both losses being optimized simultaneously?
2. What is the effect of reasoning trace quality on the results? The paper only uses DeepSeek-R1 for generating reasoning traces.
3. What is "FeatureClassOf" in Algorithm 1? This function is not defined or explained in the paper. Why is it important to maintain weights over classes rather than individual data points?
4. Why was the specific functional form of f chosen in Equation (1)? Are there principled design choices or certain classes of functions that would work well with the paper's approach?
5. Line 233: How are the sentence parts encoded to extract hidden states?
6. Line 236: Wouldn't a 2D projection using t-SNE very weakly capture correlations in high-dimensional embedding spaces? Note that separation in 2D space does not necessarily imply separation in high-dimensional space.

### Typos and Editorial Suggestions
1. Lines 146-147: "Our fundamental premise ... few appearances" is unclear. A concrete example or plot would help clarify this statement.
2. Lines 157-161: The motivation regarding the cold start problem is not clear. A concrete example or plot would help demonstrate why the cold start is a problem.
3. Line 244: Typo in specifically

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper addresses the problem of increasing computational demands of LLM distillation, such as expensive training, annotation, suboptimal data selection, etc. The paper introduces DynaGuide which optimizes distillation in terms of efficiency and performance. The method integrates dynamic data selection during training and reasoning pattern guidance that mitigates overthinking in synthetic data. Together these methods showcase efficient data distillation with a small set of selected samples.

### Strengths
- The paper addresses a crucial problem of LLM distillation efficiency
- The paper proposes a simple and elegant solution of dynamic data selection and reasoning pattern guidance
- The paper is well written and clear, it is easy to follow

### Weaknesses
The evaluation is limited to QA datasets.

### Questions
Does the proposed framework also show strong empirical performance on datasets beyond the QA datasets?

### Soundness
3

### Presentation
3

### Contribution
3
