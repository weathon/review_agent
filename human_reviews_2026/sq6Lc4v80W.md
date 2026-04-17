# To Memorize or Not to Memorize: An Analysis of Supervised Fine-Tuning in Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 4

## Abstract
Supervised fine-tuning (SFT) is a cornerstone technique for adapting large language models (LLMs) to specific domains and tasks. However, its propensity to induce verbatim memorization of training data poses significant risks to safety, privacy, and generalization. This paper presents an empirical analysis of the mechanisms underlying memorization within LLMs during SFT. Our findings confirm that SFT is a direct driver of memorization, with a clear positive correlation between the number of training epochs and the rate of verbatim data recall. The characteristics of the fine-tuning dataset are a critical determinant of memorization. We demonstrate that models trained on broad, open-domain datasets exhibit substantially more memorization than those trained on narrow, domain-specific ones, highlighting a crucial trade-off between model versatility and data containment. Furthermore, we indicate that verbatim memorization is suppressed when the training data includes inputs with high similarity paired with dissimilar outputs. We posit that this phenomenon is not a desirable mitigation strategy but rather a symptom of the model being exposed to conflicting data signals. These findings underscore the complex trade-offs in SFT and stress the importance of understanding these underlying dynamics to develop LLMs that are both capable and secure.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper performs some analysis on the memorization in LLM fine-tuning.

### Strengths
The experimental setups are described clearly.

### Weaknesses
Novelty: The authors conduct some experiments and perform some analysis, but for me these are some common understandings based on deep learning literature (e.g., longer training often leads to zero training loss but a large generalization gap, thus early stopping is a common practice) and others are also intuitive (e.g., if two samples are almost the same but the label is different, then the model is likely to be less memorizing this content). The authors are suggested to provide a more comprehensive literature review. 

Experiment design: The experiment in Figure 2 is not rigorous. Different datasets have different sample size.

Writing and layout: The following are not critical issues in the novelty and intellectual merit, but somehow hurt the reading experience.

(1) Abstract: The abstract lists some observations in the experiments, but the last sentence is not clear. "These findings underscore the complex trade-offs in SFT and stress the importance of understanding these underlying dynamics to develop LLMs that are both capable and secure." Why the observations underscore the complex trade-offs in SFT? What are the underlying dynamics?

(2) Figures: the figures can be smaller.

(3) Algorithm 1 can be moved to the appendix.

### Questions
Please address me concerns in the weakness section.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
Supervised fine-tuning directly drives memorization in large language models, with longer training and more diverse datasets increasing verbatim recall, while conflicting data signals only suppress memorization by causing model confusion.

### Strengths
1. The paper provides a comprehensive and systematic empirical analysis of memorization dynamics during supervised fine-tuning, filling an important research gap.

2. It introduces quantitative metrics linking dataset diversity, training duration, and conflicting signals to memorization, offering practical insights for safer model fine-tuning.

### Weaknesses
1. The paper appears incomplete, with several sections and analyses left unfinished.

2. The experimental results are based on limited data, reducing the reliability of the conclusions.

3. It lacks comparisons across different model architectures, which weakens the generality of the findings.

### Questions
See weakness.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies how supervised fine-tuning (SFT) leads to memorization in large language models. The authors analyze memorization across training epochs, dataset diversity, and conflicting samples, which means similar inputs with different outputs). Using Llama-3.1-8B-Instruct, they find that longer training and higher data diversity increase verbatim memorization, while conflicting examples suppress it due to gradient interference. The work offers clear empirical evidence and practical insights for safe fine-tuning.

### Strengths
1. Clear and well-structured empirical study on SFT memorization.
2. Findings are intuitive and consistent across datasets.
3. Practical insights like treating epochs as a memorization budget are useful for fine-tuning practice.

### Weaknesses
1. The paper shows that samples with similar inputs but different outputs are less likely to be memorized. This is an interesting pattern, but the explanation stays at the observation level and remains speculative. There is no evidence from maybe gradients, attention distributions, or representation dynamics to support the hypothesis. I think adding such analysis will be better.
2. The evaluation focuses on exact match and prefix continuation, which captures surface-form reproduction but does not cover semantic or entity-level memorization. In practice, language models often retain information through paraphrasing or recalling factual entities rather than copying text verbatim. The paper would be stronger if it discussed or tested whether these deeper forms of leakage follow the same trends.
3. The study is conducted only on an 8B model with full-parameter SFT, and it is unclear whether models of different sizes would exhibit the same memorization behaviors. Some discussion on model scaling would help define how broadly the conclusions apply.

### Questions
The conflict-sample analysis is interesting. Could you elaborate on how those examples were selected and whether their effect remains stable across domains?

### Soundness
2

### Presentation
3

### Contribution
2
