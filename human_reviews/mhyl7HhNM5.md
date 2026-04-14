# Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance

- Decision: Reject
- Scores: 8, 6, 5

## Abstract
NLP benchmarks rely on standardized datasets for training and evaluating models and are crucial for advancing the field. Traditionally, expert annotations ensure high-quality labels; however, the cost of expert annotation does not scale well with the growing demand for larger datasets required by modern models.
While crowd-sourcing provides a more scalable solution, it often comes at the expense of annotation precision and consistency. Recent advancements in large language models (LLMs) offer new opportunities to enhance the annotation process, particularly for detecting label errors in existing datasets. In this work, we consider the recent approach of LLM-as-a-judge, leveraging an ensemble of LLMs to flag potentially mislabeled examples.
Through a case study of four datasets from the TRUE benchmark, covering different tasks and domains, we empirically analyze the labeling quality of existing datasets, and compare expert, crowd-sourced, and our LLM-based annotations in terms of agreement, label quality, and efficiency, demonstrating the strengths and limitations of each annotation method. Our findings reveal a substantial number of label errors, which, when corrected, induce a significant upward shift in reported model performance. This suggests that many of the LLMs so-called mistakes are due to label errors rather than genuine model failures. Additionally, we discuss the implications of mislabeled data and propose methods to mitigate them in training to improve model performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper evaluate the label errors in existing datasets and examine the reliability of different annotating approaches: expert, crowd-sourcing, and LLM-as-a-judge. The authors prompt LLM to predict the label of four datasets and asked experts to verify the prediction if it disagrees with the original label. The result shows a 10~40% of label errors in four selected datasets. In addition, LLMs' prediction highly align to experts' label when the confidence score of LLMs is high. The authors also hire crowd-workers to annotating a subset of samples, and shows that the annotations of crowd-workers have a lower inter-annotation agreement and a higher disagreement with labels annotated by LLMs and experts, suggesting that LLM-as-a-judge is a reliable and efficient alternative for large-scale annotation tasks. In the end, the authors introduced two approaches to mitigate the label errors in a dataset: by filtering those data points and by flipping the wrong labels. Experimental result shows that models' fine-tuned on datasets cleaned with both approaches achieve higher performance.

### Strengths
1. The paper is well written and easy to follow.
2. This paper shows that LLMs can be used to annotate data or identify label errors with a high agreement with experts, which can be a good way to mitigate the scalability issue of dataset curation. 
3. The authors conducted extensive experiments, making their arguments convincing.

### Weaknesses
1. The approach and experiment are limited in binary classification tasks, specifically in factual consistency. The argument could be stronger if the authors conduct additional experiments in other types of task.
2. The experimental result of the crowd-sourcing part may be questionable. In Appendix B.1, the authors mentioned that the pre-qualifications of the MTurk task included 50+ approved HITs (line 1075). However, to get a higher quality result, people usually set the minimum number of approved HITs at 3000. This may explain why the authors got a high rejection rate of 49.2% and a very low Fleiss’s kappa value of the workers. As a result, the argument in Sec. 5 that the quality of crowd-sourcing annotation is insufficient may be too strong.

### Questions
1. Did you look into why the Fleiss’s kappa of crowd-sourcing is that low? To my understanding, Fleiss’s kappa is usually higher than 0.2, even for some subjective tasks (e.g. preference of responses). It is weird to me that the kappa value is only 0.074. 

Typo:
line 422: alnoe -> alone
line 427: TThe -> The

### Soundness
4

### Presentation
4

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
This paper evaluates the potential of LLMs as tools to detect label errors in NLP datasets, comparing LLM-based annotations with crowd-sourced and expert annotations. Using four datasets from the TRUE benchmark, the authors aim to demonstrate that LLMs can effectively flag mislabeled instances and, by extension, improve the overall accuracy of model training and evaluation.

### Strengths
The paper provides a detailed empirical study that underscores the prevalence of labeling errors in commonly used NLP datasets. It is thorough in its methodological approach, with comprehensive experiments and multiple evaluation metrics. The use of LLM ensembles to assess annotation quality and error rates highlights the potential cost-effectiveness of this approach compared to traditional methods.

### Weaknesses
This paper lacks substantial innovation, as its primary contribution—using LLMs for label error detection—builds on existing methods without introducing a significant advancement. Similar approaches have already been explored in previous studies, and the work here does not establish a clear differentiation or improvement over these. Additionally, the paper could benefit from expanding its dataset scope and addressing challenges such as multi-class labels or ambiguous labeling scenarios. Without these considerations, the findings may not generalize well to other, more complex datasets.

### Questions
- Could the methodology be extended to detect errors in multi-class datasets or cases with label ambiguity (fuzzy label)?
- What plans do the authors have to validate this approach beyond the datasets examined?

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
4

### Summary
The paper examines mislabeled instances in existing standard label-based datasets. The proposed method uses large language models (LLMs) to identify these instances, flagging them as potential errors when LLM predictions differ from the ground truth label. These flagged instances are then reviewed by expert human annotators for relabeling. Depending on the research question being answered, the paper compares LLM predictions and crowd-sourced annotations with expert annotations to identify mislabeled data and conduct further analysis. The findings reveal significant disagreement between expert and crowd-sourced annotations, resulting in a high rate of mislabeled data. Notably, LLM predictions align more closely with expert annotations.

### Strengths
1. The paper is well-written.
2. The topic is interesting, and the paper addresses important questions.
3. The approach used in the paper is clear and aligns well with the research questions mentioned.

### Weaknesses
1. The experiments were conducted with a small sample size and a small number of datasets. While it is understandable to control the budget when using LLMs and crowd-sourced annotators, I believe the sample size and the number of datasets in this study are too small to support the conclusions drawn. This is because a change in the label of even a few samples could lead to a significant shift in performance, and the central argument of this paper relies on this performance change.

2. In line with the previous comment, I am uncertain about the reliability of the results reported in Figure 2 and Table 3. If crowd-sourcing performs this poorly in matching expert annotations (gold labels), how did LLMs learn to predict labels that align more closely with expert annotations, given that they were trained on standard datasets labeled through crowd-sourcing?

3. According to the paper, the authors used their own guidelines to relabel dataset instances for both crowd-sourced and expert annotators. However, these guidelines differ from the original guidelines used to label the datasets. This creates a major discrepancy, as it raises the question of whether the newly assigned labels truly reflect the same conditions outlined in the original guidelines for determining ground truth labels.

4. While there is demographic information about the crowd-sourcing participants, such as their status as native English speakers, there is no such information available for expert annotators.

### Questions
**Question:**
1. The paper claims that LLMs show considerable agreement with expert annotators in a zero-shot setting for mislabeled samples. How do the authors justify this claim, given that it is widely known that performance improves in a few-shot setting? Specifically, if LLMs outperform crowd-sourced annotators in matching expert annotations, a decrease in performance should be expected. This is because LLMs, by learning from in-context examples, would align more closely with expert annotations, resulting in greater disagreement with crowd-sourced annotations.

**Suggestions:**
1. In line 79, it would be clearer to provide a numerical value instead of a verbal description.
2. The specific checkpoint used for GPT-4 is not mentioned in the paper.

### Soundness
1

### Presentation
4

### Contribution
2
