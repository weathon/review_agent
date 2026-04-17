# Fairness-Aware Test-Time Prompt Tuning

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Vision-language models have displayed remarkable capabilities in multi-modal understanding and are increasingly used in critical applications where economic and practical deployment constraints prohibit re-training or fine-tuning. However, these models can also exhibit systematic biases that disproportionately affect protected demographic groups and existing approaches to addressing these biases require extensive model retraining and access to demographic attributes. There is a clear need to develop test-time adaptation (TTA) approaches that improve the fairness characteristics of pretrained models under distributional shift. In this paper, we evaluate how episodic TTA affects fairness in CLIP classification under subpopulation shifts and develop FairTPT, a novel fairness-aware episodic TTA method that jointly minimizes target marginal entropy while maximizing spurious marginal entropy through soft-prompt tuning. We find that standard episodic TTA generally exacerbates disparities between majority and minority groups, that blinding a model to spurious attributes without degrading target performance is inherently challenging, and that excessive blinding can lead to catastrophic forgetting. This model collapse can be prevented by monitoring test-time changes in target loss within the linear regime, while still achieving fairness improvements on reactive data and preserving overall performance. Thus refined, FairTPT outperforms all state-of-the-art episodic test-time debiasing methods and establishes a foundation for robust TTA—essential for achieving fairness in practice.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper propose a fairness-aware test-time prompt tuning method by minimizing the entropy of target predictions while reducing bias by maximizing the entropy of sensitive attribute predictions. It achieves balanced adaptation through a lightweight learning-rate heuristic that prevents over-debiasing, all in a fully unsupervised, episodic setting without needing sensitive attribute labels.

### Strengths
- The motivation of this paper is clear, and the objective function (min target entropy + max spurious entropy) is very intuitive and easy to understand.

- The method does not require access to sensitive attributes or retraining the model and it's easy and practical for deployed models since it only injects in the test stage through soft prompting.

- The empirical evaluation covered most of the benchmark datasets in fairness for image classification tasks, which makes the results rigorous.

### Weaknesses
- The method is limited on one attribute spurious correlation, not extend to complex or multi-sensitive attributes, for example, if the spurious correlation is on a combination of age, gender and race. Also, the method depends on pre-define the sensitive attribute and in practice, sometimes it may not reflect the true spurious correlation.

- Lack of analysis of explainability, for example, using attention map to illustrate the change of sensitive attribute after applying the method (is it really reduce the dependence of sensitive attribute).

- No ablation study about if remove ELRA or view filtering or simply maximising spurious entropy would work, lack of demonstrate of each component of the objective function.

### Questions
- If predefined sensitive attribute is gender -> y, the actual spurious is gender+race -> y or race -> y, under such situation, is the method still effective? For example, would the soft prompting still successfully reduce the model's reliance on the true spurious factors, or might it fail to capture the intertwined biases and even introduce unintended distortions in the predictions?

- Does there exist a generalizable way to tune $\lambda_{fair}$ across datasets or tasks, and how sensitive the model’s performance is to $\lambda_{fair}$

### Soundness
3

### Presentation
2

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
Briefly summarize the paper and its contributions. You can incorporate Markdown and Latex into your review. See https://openreview.net/faq.

This work tackles a common problem in large-scale vision-language models (VLMs) like CLIP: balancing predictive performance with the mitigation of spurious correlations, i.e. fairness in this article. Building upon the Test-Time Prompt Tuning (TPT) framework—which optimizes prompt embeddings by minimizing prediction entropy over augmented inputs—the authors introduce a sophisticated dual-objective approach. Specifically, they build upon this by introducing a dual objective, adding what is effectively a "reverse TPT" for spurious attributes. While the TPT objective minimizes entropy to improve accuracy on the target task, this new, opposing objective simultaneously maximizes the entropy for spurious features. This strategy is designed to actively "unlearn" or eliminate these biases at test time, thereby achieving enhanced fairness while preserving the model's overall performance.

### Strengths
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results.

They address a meaningful challenge of mitigating spurious correlations in VLMs without largely sacrificing accuracy, a critical issue for fair and robust deployment. Its originality lies in the clever extension of Test-Time Prompt Tuning (TPT) into a dual-objective framework. The introduction of a "reverse TPT" to actively maximize entropy for spurious attributes is an intuitively effective method for test-time bias disentanglement. Furthermore, the clarity of the paper is great as for me, presenting its "push-pull" logic and technical design with precision.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.

The primary weakness lies in its perceived lack of substantial novelty as for me. The core algorithm is somewhat an extension of the existing Test-Time Prompt Tuning (TPT) framework. The objective is an intuitive application of TPT's entropy-based mechanism rather than a new paradigm. As such, the overall novelty of the technical contribution is somewhat limited as for me.

### Questions
For the final methods FAIRTPT and FAIRTPT (MO), does the latter refer to the algorithm + Multi-Objective Optimization engineering technique? And your Equation 3 and Equation 4 are actually two different designs, but for these two algorithms, did you default to choosing Equation 4 since I didn't find the specifies. I want to know what effect of Equation 3 is. Have you ever done related experiments to try it?

### Soundness
2

### Presentation
3

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
This paper proposes a method which aims at test-time tuning of zero-shot image classification tasks in visual-language models. The authors introduce two loss terms associated with the prompt generation. The authors evaluate their method against a zero-shot baseline,fine-tuned model, and a debiasing baseline, on 4 datasets.

### Strengths
1. Overall, this paper is well-motivated and presents the technical details well. It would be suitable for a general ai researcher outside the area. 

2. The evaluation is suitable thorough, the datasets and baselines are reasonable. Had there been a stronger delta against the baselines, this would be a complete paper.

3. The authors give detailed outline of all method pseudocode (but no included code) in the appendix. Also significant secondary results in the appendix. This is therefore an acceptable (but not exceptional) standard of reproducibility.

### Weaknesses
1. Overall, the evaluation is underwhelming. As is common in debiasing work, the authors don't provide a qualitative evaluation on a downstream application. For example, is +2.0% WGA qualitatively different than +1.1 WGA when comparing OrthCali v. FairTPT on average performance (Table 1)? Further, the authors don't provide variance over the trials (more than 5 trials would be best to estimate variance).  

2. The results against the fairness baseline are similar enough that things like training or inference cost are relevant, e.g. is the delta improvement over the baseline due to better hyperparameter tuning, over a larger set of candidate models or higher training runtime? Is there a large difference in inference-time cost that would justify a small delta improvement?

3. The results show significant parameter sensitivity on η. This is concerning, as this may increase training time on arbitrary datasets where we can't assume prior hyperparameters are suitable.  

Together, the authors haven't demonstrated a significant improvement in terms of efficiency or qualitative results.

### Questions
From above weaknesses:

1. What is the qualitative value of these delta improvements observed against the fair baseline?

2. Do you have std results that can be reported? At least in the average columns (if readability is a concern)?

3. Do you have runtime comparisons, both in terms of training/tuning time, and inference?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper evaluates the fairness of existing test-time adaptation (TTA) methods for VLMs and proposes a new TTA approach to reduce spurious correlations. First, the paper shows that the existing TTA methods do not improve subgroup robustness, can amplify disparities, and are highly sensitive to hyperparameters. Then, a new TTA method, FAIRTPT, is proposed that jointly minimizes the target-attribute entropy to preserve accuracy and maximizes the sensitive-attribute entropy to reduce the spurious correlations. Empirically, FAIRTPT achieves SOTA/competitive results across different fairness benchmarks.

### Strengths
- The paper studies an important problem and proposes a rigorous solution. 

- Empirically, FAIRTPT outperforms other TTA methods across various fairness benchmarks.

### Weaknesses
- I find the hyperparameter sensitivy aspect a bit irrelevant to the main theme of the paper, which is fairness. While it is nice that FAIRTPT is a method with more favorable hyperparameter sensitivy than other TTA methods, a presentation where this is introduced as a nice-to-have property would make the overall story more coherent. 

- The method is specifically developed and empirically validated on zero-shot classification asks, but the title sounds more general. This specific focus/limitation should be more explicit in the title.

### Questions
-

### Soundness
3

### Presentation
2

### Contribution
3
