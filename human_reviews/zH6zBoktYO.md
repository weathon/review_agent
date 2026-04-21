# Bring Your Own Data!  Self-Supervised Evaluation for Large Language Models

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3

## Abstract
With the rise of Large Language Models (LLMs) and their ubiquitous deployment in diverse domains, measuring language model behavior on realistic data is imperative. For example, a company deploying a client-facing chatbot must ensure that the model will not respond to client requests with profanity. Current evaluations approach this problem using small, domain-specific datasets with human-curated labels. These evaluation sets are often sampled from a narrow and simplified distribution, and data sources can unknowingly be leaked into the training set, which can lead to misleading evaluations. To alleviate the issues in traditional evaluation, we propose a complementary framework for self-supervised evaluation of LLMs by analyzing their sensitivity or invariance to transformations on the input text. Self-supervised evaluation can directly monitor LLM behavior on datasets collected in the wild or streamed during live model deployment. We demonstrate self-supervised evaluation strategies for measuring closed-book knowledge, toxicity, long-range context dependence, in addition to sensitivity to grammatical structure and tokenization errors. When comparisons to similar human-labeled benchmarks are available, we find strong correlations between self-sensitivity and human-supervised evaluations. The self-sensitivity paradigm complements current evaluation strategies that rely on labeled data.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a framework for self-supervised evaluation of Large Language Models (LLMs) by proposing a series of sensitivity (invariance) metrics that assess various aspects of language model behavior without the need for human-labeled datasets. These metrics evaluate the models based on their reaction to input transformations concerning knowledge via negations, toxicity, and word order. The authors claim that these self-supervised evaluation methods can complement traditional supervised benchmarks and provide efficient evaluation in real-world settings.

### Strengths
- The proposed self-supervised evaluation framework addresses the significant challenge of evaluating LLMs without the extensive need for labeled datasets, which is a common bottleneck.

- The authors provide empirical evidence that correlates the proposed self-supervised metrics with existing supervised benchmarks, lending credibility to their approach.

- By analyzing how LLMs react to various textual transformations, the paper offers deeper insights into the behavior and limitations of these models, which can inform future research and model development.

### Weaknesses
- The paper acknowledges model entropy as a factor that could influence sensitivity scores but does not explore it in detail. Understanding how the entropy of a model's output distribution affects evaluation metrics is crucial for interpreting results accurately.

- The main methods proposed in this work—knowledge probing via negations, toxicity detection, and word order—seem to be presented as separate entities without a unifying theme or rationale that clearly ties them together. The paper could benefit from a more cohesive narrative that explains how these methods collectively advance the understanding and evaluation of LLMs.

- The proposed methods, such as adding "not" after certain words for negation or appending trigger words for toxicity, may seem too basic or trivial. This simplicity could lead to questions about the depth and sophistication of the approach, as well as its ability to capture the nuances of language and behavior of LLMs. The straightforward nature of the methods may not generalize well to the complex and varied inputs that LLMs encounter in real-world applications. The paper might not demonstrate that the methods can handle different linguistic constructions, idiomatic expressions, or contextual nuances.

- The motivation behind each method is not evidently articulated. While each method addresses a different aspect of language model behavior, the paper may not clearly explain why these particular aspects are chosen and how they complement each other in providing a comprehensive evaluation.

- The methods may not be backed by a comprehensive set of experiments to validate their effectiveness across different models, domains, prompts, and languages. This could be seen as lacking in terms of the breadth and depth of experimental validation.

### Questions
- Could you explain the rationale behind the simplicity of the proposed methods? How do you ensure that such straightforward techniques can provide a robust evaluation of complex LLM behaviors?

- To what extent did you consider more sophisticated prompt engineering in your evaluation framework? Could you elaborate on how different prompt designs might affect the outcomes of your proposed metrics? For example, how could changing the position of trigger words in toxicisy detection influence the outcome?

- Can you discuss any additional experiments that might demonstrate the robustness of your evaluation methods across different languages, dialects, or domains?

- How might the entropy of a model's output distribution or its propensity for memorization affect the outcomes of your self-supervised evaluation metrics?

- What steps did you take to mitigate the impact of potential confounding factors, such as model overfitting or exposure to similar data during training, on your evaluation results?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a new self-supervised approach to the evaluation of LLMs, alleviating the need for small domain-specific datasets with human-curated labels as in traditional evaluations. The new evaluation method, on a higher level, is through analyzing invariances and sensitivities to transformations. The work provides detailed case studies on several self-supervised evaluation strategies for different aspects of LLMs, including those related to negations, toxicity detection, long-range dependency, and sensitivity to word order, etc. Strong correlations have been shown between the designed self-supervised evaluation metrics and human-supervised evaluations.

### Strengths
- **Originality**: This paper offers a fresh approach to the evaluation of LLMs, moving away from dataset-bound evaluations to a sustainable assessment methodology.
  
- **Quality**: The transformations, like handling of negations, word order changes, and others, are impressive steps toward achieving a holistic evaluation of LLMs. 
  
- **Clarity**: The paper is accessible to even readers that are unfamiliar with the domain. The delineation of their methods and results is commendable.
  
- **Significance**: The paper’s methodology, if thoroughly verified and broadly adopted, has the potential to revolutionize how LLM evaluations are conducted, making them more dynamic, comprehensive, and reflective of real-world applications.

### Weaknesses
- **Terminological Ambiguity**: The usage of "self-supervised" is somewhat misleading. Given the context of this paper, an alternative term or a more precise explanation would be helpful.

- **Metric Soundness**: While the paper puts forth several innovative metrics, further clarity and validation are needed. For example, the paper justifies its methods heavily by calculating the correlation between proposed scores and one specific existing task. Then I believe a natural question arises – if, for example, having high correlation with TriviaQA alone is enough to demonstrate the legitimacy of SSE, then why don’t we just use TriviaQA accuracy (and also HellaSwag) as the evaluation metric? I am therefore doubtful of the significance of the proposed metrics.
 
 
- **Questionable Conclusion**: Following up on the comment on the correlation analysis, the paper tries to prove the usefulness of their metrics by showing correlation with existing task on some metrics (e.g. TriviaQA accuracy), but also tries to disprove other evaluation metrics by showing they correlate less nicely with their proposed metrics. For example, the paper concludes perplexity is not a robust evaluation metric because it does not have a very high correlation with SSE; the Cohere Command model is an outlier in their analysis which highlights a weakness of TriviaQA.

- **Visualization**: The visualization in this paper is unfriendly to people with color vision deficiency if not anyone. I find it very hard to distinguish to the difference in the sizes and colors in the figures representing different models.

### Questions
- Can authors provide some examples of how they perform the transformations to evaluate long-range sensitivity.

- How do you identify the neutral corpus?

- “We further explore why correct normalization is important by cross-referencing the frequency with which perplexity goes down rather than up, see Figure 14 in Appendix A.5.” from the figure I see a nice negative correlation between “Percent PPL Drops” and TriviaQA accuracy. So how can this show correct normalization is import in SSE?

- What data do the metric PPL use? Following this paper’s logic, I think the first step this paper should do is to evaluate the correlation between, for example, normalized PPL and TriviaQA accuracy (and other realistic tasks such as MMLU etc).

- How well do toxicity scores by this paper correlate with those by Perspective API on instruction tuned models? From figure 6 left, it looks like there are no Xs. Same question for word order section and figure 4, 7. Without the results of instruction tuned models, it is hard to see if the scores correlate better on vanilla models than instruction tuned models. And therefore hard to assess the significance of the metrics.

- It is still hard for me to understand how to use this paper in real-world applications. How adaptable are these proposed metrics? Would the methodology need alterations to assess LLMs in more dynamic, real-world scenarios?

- Given the terminological ambiguity around "self-supervised," can authors elaborate on this terminology or propose an alternative name to prevent misconceptions?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a self-supervised method for evaluating LLMs without relying on domain-specific, human-annotated datasets. The authors detail a structured approach for self-supervised evaluation, focusing on LLM invariances and sensitivities. Preliminary tests indicate a correlation between their proposed metrics and established ones that depend on human annotations.

### Strengths
- Originality: The paper unveils a fresh evaluation technique for Large Language Models that minimizes human intervention. This addresses the shortcomings of conventional methods and provides an alternative measure of a model's capabilities.
- Their approach can be adapted to different data domains by simply varying the unlabeled text in evaluations. Such as the clinical setting, where they show their method has a good correlation with MMLU (clinical).

### Weaknesses
- Weak evidence for the robustness of this method: In my opinion, using the correlation to TriviaQA's accuracies to prove their usefulness seems to be a bit of weak evidence, especially when only ~10+ models are considered in the experiments. The correlation results can be simply affected by noises/outliners. For example, you mentioned that "The Pearson correlation between TriviaQA and Normalized sensitivity score is 0.76 for vanilla models and 0.73 for instruction models after removing the Cohere Command outlier". It proves that the evaluation method is a little bit brittle. When I have a new model that wants to be tested, how can I know my model is not another outlier?
- I would suggest the authors increase the number of models to 100+, so as to reduce the effects of noise when computing the correlations. If there are not enough LLMs to be tested, one of the strategies you can use is to do early exiting from the models [1] so that you can get different outputs from different layers of the model, representing different levels of understanding of the data, and thus increase the number of total data points to be compared.

[1] Eliciting Latent Predictions from Transformers with the Tuned Lens
Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, Jacob Steinhardt https://arxiv.org/abs/2303.08112

### Questions
- In figure 3, 4, 6, and 7, why the models used for comparison is changing all the time? In figure 3 there are 28 data points but in figure 7 there are only 11 data points. What's your standard of selecting these models to be evaluated? I think it's better to include all the models throughout the experiments to reduce the effects of noise/outliers. The experiment results presented make me feel that some of the results are hidden so the correlation may not be that good if we add them back to the figure.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a self-supervised evaluation approach for toxicity classifiers. The paper cites that due to training data leakage, toxicity classifiers may have an inflated reported performance. Using the proposed approach where inputs are modified using negation and other techniques, the robustness of content classifiers is evaluated.

### Strengths
The paper has the following strengths.

First, it is well-written. The paper motivates the problem well, experiments are described clearly. Related work descriptions are reasonable (although the paper missed some key citations, e.g., Gröndahl et al.). 

Second, the motivation for building a domain-specific toxicity classifier is well-received.

### Weaknesses
The key weakness of the paper is:

1. The modifications it is suggesting are too simplistic (e.g., F-bombing). There are many known modifications that have tripped content classifiers before (e.g., All You Need is "Love": Evading Hate Speech Detection by Gröndahl et al.) or examples of real-world examples tripping content classifiers (e.g., Are Chess Discussions Racist? An Adversarial Hate Speech Data Set by Sarkar and KhudaBukhsh). Instead of using obfuscation techniques well-grounded in literature, the paper adopts simplistic techniques to modify inputs.  

2. The second weakness of the paper is it relies on Perspective API. Perspective API's toxic scores are not reliable. Recent research indicates that. Hence, an API that itself has calibration issues cannot be very useful for calibrating other systems.

### Questions
My questions are how would the authors respond to my two weaknesses listed above?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
