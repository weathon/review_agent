# Optimized Large Language Models Accurately Identify Recurrence of VT After Ablation from Complex Medical Notes: Will Chart Review Become Obsolete?

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3

## Abstract
Large language models (LLMs) provide impressive out-of-the-box performance to queries for which data are in the public domain using prompt engineering. However, they are less effective when analyzing important datasets that are not publicly available, such as electronic health records (EHRs), where current prompting strategies are either suboptimal or require domain-specific expertise. To overcome these limitations, we proposed a $\textit{non-domain-specific}$ prompting strategy—termed $\textbf{Structured Rationale Responses}$ (SRR)—designed to enhance the accuracy and reliability of LLM responses to nuanced inquiries in EHRs compared with expert interpretations. Specifically, SRR guides LLMs to generate responses 1) in a structured format (e.g., JSON), and 2) with rationales, which are sentences excerpted from the query note that the LLM used to support its answer. In 499 full-text EHR notes (474.6±164.3 words) in 125 patients with life-threatening heart rhythm disorders, we asked LLM whether a patient had an acute event of ventricular arrhythmias which required it to remove parse contradictory information on prior events. In an independent hold-out test set of 398 notes (471.8±160.1 words), our SRR achieved a balanced accuracy of 86.6\%±4.0\% without any in-context examples, demonstrating an average performance lift of 30.5\% over the standard prompts, 12.2\% over Zero-shot-CoT prompts, and 10.4\% over 5-shot prompts. Notably, for true positives where LLM correctly identified acute events, 94.4\%±5.2\% had at least one LLM-generated rationale considered clinically relevant by experts. Our code can be found at https://github.com/***.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper reports on a clinical language processing application of large language models in English.

### Strengths
Considering large language models is a trending topic. In particular, prompt engineering is quite well done in the paper.

Clinical language processing is topic of utmost societal importance, and the particular application to Ventricular Tachycardia (VT) has not been really addressed, even for English. 

This study is sensitive to its clinical context, as evidenced by its there research questions, and the large scale of clinical data used in the experiments of this study is excellent.

The manuscript is very carefully written.

### Weaknesses
The study could be strengthened by addressing, e.g., ethical code of conduct with respect to involving clinical experts as annotators in this study and the risks of using ChatGPT to process clinical notes. This could form a broader impact statement in the discussion section. More generally, the study is lacking in its discussion (e.g., limitations of this study and proposed future work).

Although performance evaluation results seem great, I cannot find statistical significance testing, confidence intervals, or similar included in the study methodology. 

The paper could be made stronger by being more specific about the language(s) considered. It is implied that it is limited to English only at its current form.

### Questions
What is the broader impact of this work? What kind of benefits and risks did it have? Were the findings statistically and/or practically significant? Have related work been conducted for languages other than English? What the envisioned influence and impact will be? What kind of future work should be conducted?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposed a non-domain-specific prompting strategy, Structured Rationale Responses (SRR) to enhance the accuracy of LLM responses to identify recurrence of Ventricular Tachycardia (VT) among electronic health records (EHRs).

### Strengths
- Originality: the paper proposed a creative combination of existing prompting strategies with novel application to VT identification from EHR data.
- Clarity: the paper is well-organized with clear description in study design and method illustration, as well as results examples.

### Weaknesses
1. The proposed method is only evaluated by a single dataset for a single disease VT, which might weaken the utility of the proposed prompting strategy that is VT-specific only, but not generalized to other diseases.
2. The experiment results lack the model comparison of using other structured formats in the prompt, e.g. xml or others. What exact prompt was used with bullet points in experiment 4.5?  
3. Figure 3, the experiment results of SRR lack confidence bars. It is the same in Figure 5 and 6. Is the results cross-validated? If not, how robust is the performance.
4. Lack of limitation discussion in the conclusion part.

### Questions
1. How will the proposed prompting strategy work in GPT-4?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a novel prompting strategy called Structured Rationale Responses (SRR), which is specifically tailored to improve the precision and dependability of Language Model (LLM) responses when dealing with nuanced queries in non-publicly accessible electronic health records (EHR). SRR serves as a guidance mechanism for LLMs, encouraging them to produce responses that are 1) structured, such as in JSON format, and 2) accompanied by clear rationales.

### Strengths
1. The suggested approach is straightforward.
2. Ablation studies have been conducted, and the method has been compared to other widely used strategies.

### Weaknesses
My dissatisfaction with the evaluation of the proposed approach stems from several factors. 

1. The authors argue for the superiority of the proposed strategy, but the reasoning behind this claim is not adequately explained. The analysis is limited to a single dataset addressing a specific problem, which is insufficient to draw conclusive results. It remains uncertain whether this strategy would be effective for non-private EHR data. Why exactly it is good for private EHR data?
2. The evaluation involves only one model, whereas related work typically considers a subset of models for a more comprehensive assessment.
3. The experimental results do not sufficiently clarify why this strategy is advantageous. For example, it's unclear whether model size plays a significant role in its effectiveness. Furthermore, an error analysis comparing the proposed strategy to other strategies could provide valuable insights.
4. Minor Issue: The use of the superscript $t$ is not defined or explained in the context of the text.

### Questions
Please, elaborate on Weakness section

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
