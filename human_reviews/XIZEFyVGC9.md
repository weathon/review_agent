# Debiasing Algorithm through Model Adaptation

- Avg Score: 5.67
- Decision: Accept (poster)
- Scores: 3, 6, 8

## Abstract
Large language models are becoming the go-to solution for the ever-growing number of tasks.
However, with growing capacity, models are prone to rely on spurious correlations stemming from biases and stereotypes present in the training data.
This work proposes a novel method for detecting and mitigating gender bias in language models.
We perform causal analysis to identify problematic model components and discover that mid-upper feed-forward layers are most prone to convey bias.
Based on the analysis results, we intervene in the model by applying a linear projection to the weight matrices of these layers.
Our titular method DAMA, significantly decreases bias as measured by diverse metrics while maintaining the model's performance on downstream tasks.
We release code for our method and models, which retrain LLaMA's state-of-the-art performance while being significantly less biased.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work studies the the issues of biases and stereotypes in the large language models, and propose methods to (1) identify the specific components of LLMs that is responsible for these issues; and (2) resolve these issues with a linear projection on the feed-forward components of LLMs.

### Strengths
- The research question is interesting and critical to LLM research and application
- The high-level ideas of the proposed techniques (on both identification and debiasing) make sense.

### Weaknesses
I had a difficult time understanding this paper because of the following writing issues. 

- First of all, certain concepts are mentioned in the paper. However, they are either inconsistent or not explicitly defined. For example,
    - Stereotypical keys vs. stereotyped keys
    - Gender value vs. gendered value
    - Also, what is grammatical gender?
- About Equation 2, I have several questions
    - What is z? Why do we need a $\ell_2$ item on it?
    - Missing a “)” somewhere in equation 2
    - What is $P(o’|X’)$? The first two items in equations look similar to variational inference. In that case, $P(o’|X’)$ should be something similar to a prior distribution. However, I could not find its definition.
- What is gender values metrics?  What is the relation between these V metrics and U?
- Why $P$ in equation 5 is defined in that way, and how should we use it? I think it was explained in the paragraph right after equation 5, but I am not sure I understand it.
- Figure 3 (b) seems to indicate that to reduce the bias, the cost is to get a much worse language model (based on the perplexity score), which is not exactly claimed in the paper.

In addition to my clarification question, one concern is about the technical novelty specifically the causal tracing idea is from prior work, and the linear project seems to be a straightforward idea in debiasing literature, so I am wondering what is the technical novelty of this work.

### Questions
Please refer to the clarification questions mentioned in the previous section.

### Soundness
2 fair

### Presentation
1 poor

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
The paper demonstrated unwanted gender bias still exists in the popular LLaMA family of models. To combat these prevalent biases, the paper located components of the model that caused such biases and edited these weights to mitigate the effect of gender biases on downstream tasks. The method, DAMA, is an improvement over the previous method both in terms of reduction rate and maintaining original performances.

### Strengths
1. Detailed and quantitative measurement of the effect of factual cues and stereotypical cues on model generations. Specifically in the result section, the authors measured the effect of factual cues and stereotypical cues based on layer number and token positions.
2. The gender bias reduction method via model weight editing has sound theoretical backup.
3. DAMA effectively reduces the bias rate without hurting too much of the downstream performances.

### Weaknesses
1. Seems like there is still quite some room for improvements in the debiasing methods on all three evaluations proposed in the paper. For WinoBias and SteroSet gender, I interpret the results as being still pretty gender biased even after applying DAMA. And for Bias in LM, for larger models, the a_s and a_f are still far from complete removal. It would help greatly to provide other gender bias removal methods as baselines to better assess how well DAMA did.
2. I think the study on the effects of downstream is great. However, its negative effect on reasoning tasks such as ARC can be concerning for a practical user. Is there any hypothesis on why it doesn't fare well with reasoning type of tasks?

### Questions
Styling:
1. You are missing a right parenthesis on equation (2) for the KL divergence.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigates gender bias in the Llama model. First, an existing causal tracing method is adapted to work for measuring gender bias in different components of the model. Then, a method for modifying the model weights is described for reducing the gender bias. Evaluation is performed both on gender bias benchmarks and downstream task datasets.

### Strengths
The methods are interesting. Evaluation is performed on a number of different datasets and using different measures.

The method for updating the model is also interesting. Particularly as it manages to not even increase the number of parameters in the model.

### Weaknesses
A weakness is only focusing on Llama, as it is unclear how much these findings generalise to other models.

The causal tracing method is interesting. It is currently unclear how much this is based on previous work and what exactly is the novel contribution. Please clarify this.

It is unclear why a linear model is fitted across the two gender scores in order to investigate the extent of bias. If the two scores are correlated (which they likely are) then the coefficients of such a linear model might not give accurate indications of the different biases. Measuring correlation with the gender scores seems like a much more straightforward method.

Again, for the method of updating weights, please clarify the difference of the proposed method to previous work, including Rafvogel et al 2022.

A major selling point of the method seems to be that the proposed DAMA still achieves good performance on downstream tasks. However, the baseline MEMIT actually seems to get better results on most of this metrics. This finding somewhat weakens this claim and should be addressed in the paper.

### Questions
Please see above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
