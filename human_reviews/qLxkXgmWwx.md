# Investigating Factuality in Long-Form Text Generation: The Roles of Self-Known and Self-Unknown

- Decision: Reject
- Scores: 3, 3, 5, 5

## Abstract
Large language models (LLMs) have demonstrated strong capabilities in text understanding and generation. However, they often lack factuality, producing a mixture of true and false information, especially in long-form generation. In this work, we investigates the factuality of long-form text generation across various large language models (LLMs), including GPT-4, Gemini-1.5-Pro, Claude-3-Opus, Llama-3-70B, and Mistral. Our analysis reveals that factuality scores tend to decline in later sentences of the generated text, accompanied by a rise in the number of unsupported claims.
Furthermore, we explore the effectiveness of different evaluation settings to assess whether LLMs can accurately judge the correctness of their own outputs: Self-Known (the percentage of supported atomic claims, decomposed from LLM outputs, that the corresponding LLMs judge as correct) and Self-Unknown (the percentage of unsupported atomic claims that the corresponding LLMs judge as incorrect). The results indicate that even advanced models like GPT-4 and Gemini-1.5-Pro fail to achieve perfect Self-Known scores, while their Self-Unknown scores remain notably above zero, reflecting ongoing uncertainty in their self-assessments.
Moreover, we find a correlation between higher Self-Known scores and improved factuality, while higher Self-Unknown scores are associated with lower factuality. Interestingly, even without significant changes in the models' self-judgment (Self-Known and Self-Unknown), the number of unsupported claims can increases, likely as an artifact of long-form generation. These findings show the limitations of current LLMs in long-form generation, and provide valuable insights for improving factuality in long-form text generation.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper investigate the factuality in long-form text generation. Using the existing datasets, it defines the Self-Known and Self-Unknown scores judged by the corresponding LLMs themselves. Based on several sets of statistics and empirical experiments, the authors draw several findings which might be useful for the community.

### Strengths
1. Long-form factuality is a promising research direction.

### Weaknesses
1. The statistical observations are based on an existing dataset of Biography from Min et al., 2023, which impair the contribution of this work.

2. The key finding of this work, *factuality scores tend to decline in later sentences of the generated text, accompanied by a rise in the number of unsupported claims*, is obvious and not exciting. This is a general challenge for LLMs to generate long-form texts in the later parts in terms of factuality, coherency, creativity, etc.

3. This work does not provide any insightful reasons, either. As stated in Section 2.1, *We hypotheses the possible reasons are the error propagation and these generated claims are with low confidence by LLMs*, this is not convincing.

4. The results presented in the main text overlap somewhat. For example, I don't quite get why reporting both the percentage and number of supported and unsupported atomic claims in Figure 1.

5. It is abrupt to discuss retrieval-augmented generation in Section 5.3. I don't quite get the necessity of introducing RAG.

6. The presentation is not good, which should be proofread one more time. For example, the duplicate descriptions of *Mixtral* in line 18 and *in this work* in line 42. Line 59: *Our findings indicate that sentences generated **later** in the sequence generally demonstrate higher factuality.* I think it should be *sentences generated **early** in the sequence generally demonstrate higher factuality*?

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper explores the challenges large language models (LLMs) face in maintaining factual accuracy in extended text outputs, analyzing models like GPT-4, Gemini-1.5-Pro, and Llama-3-70B. The author first estimate the factuality score based on ‘’Self-Known and Self-Unknown’, and the evaluation shows that the score is well aligned with human evaluation. The study shows that factuality often declines over longer text, with later sentences containing more unsupported claims. Using new metrics, Self-Known, Self-Unknown and factuality score, the authors assess how well LLMs recognize their own factuality; they find that even advanced models have limited ability to self-assess accurately, achieving only modest success in identifying correct or incorrect claims. The findings emphasize the need for improved self-assessment in LLMs to enhance factual consistency, particularly in long-form generation.

### Strengths
a.	The problem explored in the paper is interesting, and worth further exploration.
b.	The authors estimate a factuality score based on Self-Known and Self-Unknown, which makes sense

### Weaknesses
a.	The presentation of the paper is confusing, the logic line of the paper is very unclear. For example, the paper proposes to use three ways to evaluate the self-known and self-unknown, however, no details is provided for all the methods. And in Section 4, an estimation of the factuality score is proposed, and in section 5, they mentioned that they measure the factuality score using FActScore, then which score do they used in the following experiments?
b.	As an analytical paper, it lacks insightful findings or conclusions. It has been found in some previous works that the model generates more unsupported atomics in the later sentences, and this is a known problem with long-form generation.
c.	 Some errors in the paper writing,  e.g. in line 60, they said ‘Our findings indicate that sentences generated later in the sequence generally demonstrate higher factuality.’ and in line 85, they said ‘...LLMs typically exhibit lower factuality scores in the later segments of long-form text.’ , which are contradictory.

### Questions
a.	Which factuality score did you use for evaluation, the FActScore or the estimation in section 4?
b.	Can you elaborate more on the assumption in line 250?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores the issue of factuality in long-form text generation by large language models (LLMs), specifically addressing their tendencies to produce a mix of true and false information. The study examines six advanced LLMs, including GPT-4, Claude-3-Opus, and Mixtral, to investigate how factuality scores evolve throughout the text. The authors propose two novel evaluation metrics, Self-Known and Self-Unknown, to measure whether LLMs can accurately self-assess the correctness of their generated content. Results reveal that while LLMs generally perform well in earlier parts of the text, factuality tends to decline in later sections. Additionally, higher Self-Known scores correlate with better factuality, while higher Self-Unknown scores indicate the opposite. The authors also test various evaluation settings, emphasizing the importance of incorporating external knowledge to improve models’ factual accuracy.

### Strengths
The introduction of Self-Known and Self-Unknown scores provides a fresh perspective on evaluating LLMs' self-assessment capabilities, offering valuable insights into their strengths and limitations. The paper systematically analyzes multiple LLMs, covering different evaluation settings and utilizing both human and automated tools for factuality assessment. This extensive examination adds rigor to the study. By highlighting the consistent decline in factuality in long-form outputs, the research addresses a critical challenge for deploying LLMs in real-world applications, making the findings highly relevant for model improvement. The observed relationships between factuality and self-assessment metrics could guide future work on enhancing LLMs’ self-awareness and knowledge calibration, which is vital for developing reliable generative AI models.

### Weaknesses
1. The difference between Figure 1 (a) and Figure 1 (b) is not illustrated. It will be beneficial to add illustrations in caption.
2. The proposed method can perform evaluation to long-term generation by the model itself, but it is time-consuming to do evaluation for every atomic statement. The detection itself has a very strong trade-off between performance and inference costs.
3. There is not a way to adjust or convert the fact errors into correct facts. The proposed method only covers fact error detection.
4. Missing results for LLAMA family makes the results less comprehensive.
5. The scope of the proposed method is a bit limited as it can only be applied to factuality tasks which can be separated into atomic statements. However, some factuality tasks may not explicitly contains atomic statements.

### Questions
As in weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studied factuality in long text generation. It studies the position of the atomic fact and finds that the latter part of the text generated is more likely to be not factual. It invented two metrics: self-Known (the percentage of correct atomic facts by predicted as correct atomic facts) and Self-Unknown (the percentage of incorrect atomic facts are correctly predicted as incorrect). The authors claim the two metrics can help measure the true factuality score.

### Strengths
- The authors did extensive analysis to support our intuition that there tends to more factual errors in the latter part of the LLM's response. The authors show this applies to different models and also applies to RAG scenario.

### Weaknesses
- The Finding where model tends to have more factuality errors in later part of the output is kind of accepted conclusion. So the findings is no surprising
- The conclusion about the relationship between factuality score and Self-Known and Self-Unknown is not established. Firstly, I think there is something wrong when transiting from Eq (1) to Eq (2) and it is hard to understand the definition of "y_correct" and "y_wrong". Secondly, the assumption that there is no "over confidence" is not correct.  Lastly, it will be helpful if authors can verified the conclusion in Figure 3 in more than 1 datasets.

### Questions
1. what are definitions of  the definition of "y_correct" and "y_wrong? can you show why Eq (1) can transit to Eq (2)?
2. I think "self-Known" is also called, True Negative Rate (if we consider the labe of being not-factual as positive) and "Self-Unknownw" will be "True Postiive Rate". Is this correct?

### Soundness
3

### Presentation
3

### Contribution
3
