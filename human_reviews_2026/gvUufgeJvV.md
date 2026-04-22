# Are LLMs Really Not Knowledgeable? Mining the Submerged Knowledge in LLMs' Memory

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6

## Abstract
Large language models (LLMs) have shown promise as parametric knowledge bases, but often underperform on question answering (QA) tasks due to hallucinations and uncertainty. While prior work attributes these failures to knowledge gaps in the model’s parameters, we uncover a complementary phenomenon: LLMs frequently retain correct knowledge even when generating incorrect or \``unsure'' answers.
By analyzing the token-level output distributions, we find that correct answers often appear among high-probability candidates, despite not being selected. Motivated by this, we propose Hits@k, a novel metric to evaluate latent knowledge retention independent of answer surface form. Our experiments reveal that LLMs possess significantly more factual knowledge than is reflected by standard QA accuracy.
Building on these insights, we further examine the prevailing few-shot QA paradigm. We find that prompting strategies which allow ``unsure'' outputs can inadvertently suppress correct answers by discouraging low-confidence generation. We design a set of quantitative experiments to measure this suppression effect, offering practical guidance for future prompt and decoding design in knowledge-intensive tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether large language models truly “do not know” an answer when they produce an incorrect response. The authors argue that model knowledge and model expression are not equivalent, and they conduct exploratory experiments such as Hits@k and “unsure” filtering to show that correct answers often appear in the top-k token distribution, even when surface accuracy is low. The work suggests that LLMs may possess latent knowledge that is not successfully expressed during decoding.

### Strengths
- Clear motivation. The paper highlights an intuitively important gap between latent knowledge and surface-level generation in LLMs.

- Empirical observations are easy to interpret. Hits@k and “unsure” filtering provide simple and intuitive diagnostic signals.

- Readable paper structure. The writing is clear, and the experiments are straightforward to follow.

### Weaknesses
- The insight is not novel, as similar conclusions have long existed in perplexity-based evaluations, which already reflect that LLMs may assign high probability to correct tokens that are not selected in top-1 decoding.

- The phenomenon is also well-known from rollout-based methods (e.g., multi-sampling, self-consistency, and RL trajectories), which routinely reveal correct answers in non-greedy decoding paths.

- The paper lacks deeper analysis or actionable contribution, offering no explanation of why the mismatch occurs, nor methods for leveraging latent knowledge to improve actual model performance.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shows that LLMs can retain correct knowledge even when generating incorrect answers; correct answers frequently appear among high-probability tokens despite not being selected as final outputs. Based on this observation, the paper introduces Hits@k, a new metric to assess the knowledge of LLMs. Also, it introduces a new decoding method to improve answer accuracy by leveraging detected but unexpressed knowledge.

### Strengths
- This paper catches an interesting finding that LLMs often maintain access to accurate information within their probability distributions over vocabulary tokens, and there is a systematic gap between knowledge storage and expression rather than simple knowledge absence.

- It offers new insights into knowledge augmentation: instead of expanding knowledge, augmenting the ability to express existing knowledge is important and can be potentially very useful.

### Weaknesses
- Though it is an interesting finding, I still believe that LLMs are not knowledgeable even though they assign significant probability scores to tokens representing the correct information, since in real-world use cases, it is impractical to let LLMs generate multiple responses to each query. Therefore, I don't think Hits@k should be used for evaluation/rank models.

- The proposed decoding algorithm can raise many safety or ethical concerns if deployed into general use cases, since in many real-world scenarios, it might be unsafe or unethical to generate an “informative” response.

- The proposed decoding algorithm increases the probability of correct answers, but also increases the probability of wrong answers.

### Questions
Cite and introduce DBPedia, IMDB, and GoodReads with more details (maybe in appendix).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates how large language models (LLMs) store and express factual knowledge. It argues that incorrect or "unsure" answers do not necessarily indicate missing knowledge, since correct answers often appear among high-probability tokens that are not selected. To quantify this hidden knowledge, the authors introduce Hits@k, which measures how frequently the correct answer appears within the top-k tokens of the model's output distribution. Extensive experiments across open-domain and domain-specific datasets show that models retain substantially more factual information than is revealed by accuracy alone, and that newer models exhibit higher latent retention. The study also finds that "unsure" prompts can suppress correct answers by lowering generation confidence, and that filtering such responses can recover many correct predictions. Together, these findings reveal a gap between knowledge storage and expression, offering insights for improving prompt design and decoding strategies in knowledge-intensive tasks.

### Strengths
- The paper introduces a clear and intuitive metric that captures latent knowledge beyond standard accuracy, offering a new perspective on model evaluation.

- The analysis reveals that models often “know” more than they express, which challenges common assumptions about what low-confidence or incorrect outputs imply.

- The experiments are extensive and well controlled, which show consistent trends across multiple model scales and factual datasets.

- The study provides actionable insights for prompt design and decoding strategies by showing how uncertainty affects knowledge expression.

- The paper is clearly written and conceptually accessible, making its findings easy to reproduce and useful for both research and applied settings.

### Weaknesses
- The paper does not provide a formal justification for why Hits@k should reflect internal knowledge rather than distributional coincidence, relying mainly on empirical correlations (Figure 3).

- The improvement margins between Hits@k and standard accuracy are sometimes modest -- for example, less than 5% in several datasets (Table 2) -- which weakens the claim of large hidden knowledge reserves.

- The evaluation focuses narrowly on factual recall and omits reasoning or multi-hop questions, so it is unclear whether the proposed metric captures deeper forms of knowledge use beyond surface recall (Section 5.2).

- The proposed method measures the presence of correct tokens but ignores how easily the model can retrieve or reason about them, which conflates memorization with accessibility (Section 4.3).

- The study does not examine sensitivity to decoding parameters such as temperature or top-p, leaving unclear whether the observed patterns remain stable under different generation settings.

### Questions
How does Hits@k distinguish between genuinely stored knowledge and coincidental token co-occurrence, and what evidence supports that the correct token's presence in the top-k reflects meaningful internal representation rather than surface-level probability alignment?

### Soundness
3

### Presentation
3

### Contribution
3
