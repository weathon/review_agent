# Provable Robust Watermarking for AI-Generated Text

- Decision: Accept (poster)
- Scores: 8, 8, 6, 6

## Abstract
We study the problem of watermarking large language models (LLMs) generated text — one of the most promising approaches for addressing the safety challenges of LLM usage. In this paper, we propose a rigorous theoretical framework to quantify the effectiveness and robustness of LLM watermarks. We propose a robust and high-quality watermark method, Unigram-Watermark, by extending an existing approach with a simplified fixed grouping strategy. We prove that our watermark method enjoys guaranteed generation quality, correctness in watermark detection, and is robust against text editing and paraphrasing. Experiments on three varying LLMs and two datasets verify that our Unigram-Watermark achieves superior detection accuracy and comparable generation quality in perplexity, thus promoting the responsible use of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on the problem of watermarking large language models generated text. The key idea is to make the logits of the language model for a specific subset of vocabulary are increased by δ. 
The paper addresses three significant aspects of watermarking, including the quality of watermarked model, the performance of watermark detection algorithm, and its robustness against attacks. Firstly, the paper proves the guaranteed generation quality by “ω-Quality of watermarked output” and evaluates text perplexity of both watermarked and un-watermarked generated text.  Secondly, the paper evaluates the performance of watermark detection from two perspectives: false positive and false negatives. Lastly, the paper delves into a comprehensive analysis and presents experimental results regarding the robustness against paraphrasing attacks and editing attacks. 
Overall, the research topic is meaningful and promising, the problem is well-defined, and the paper is well-written.

### Strengths
1.	This work is very solid. In particular, Section 2.1 stands out for its precision and conciseness, setting a solid framework for the entire paper.
2.	The paper proves Type I/Type II errors of the detection algorithm decay exponentially as the suspect text length gets longer and more diverse, which is a desirable property of watermarking generated text.
3.	The paper conducts extensive and comprehensive experiments. The proposed method achieves superior detection accuracy and improved robustness against different attacks, thus promoting the responsible use of LLMs.

### Weaknesses
The work is solid, and many conclusions and experimental results are provided in Appendix. My only concern is the proposed method is constrained with the length of text. Based on the key idea of the watermarking method, it can only perform well when the generated text is long enough. It is evident that longer text exhibits better performance of watermarking technique, which is indeed a favorable attribute. However, However, it raises questions about the method's viability and performance on shorter texts. Experiments testing how the proposed method performs on different text lengths are absent.

### Questions
1.	In the section ‘INTRODUCTION’, you mentioned that “Type I/Type II errors of the detection algorithm decay exponentially as the suspect text length gets longer and more diverse.” I'm curious about how do you assess the diversity of text in your paper.
2.	While the paper applies a constraint based on edit distance to evaluate the security property, it's worth noting that some attacks may not significantly alter the edit distance. In such cases, how do we effectively assess the security of the watermark?
3.	I'm intrigued by the observation that a smaller gamma value results in a stronger watermark. Intuitively, a larger gamma would seem to imply that more words in the vocabulary carry the watermark. However, it's also apparent that a larger gamma can lead to an increase in false positives. Could you please provide further clarification on this relationship?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Built upon the watermark framework proposed in (Kirchenbauer et al., 2023), this paper studies a theoretically provable watermarking framework, named UNIGRAM-WATERMARK, with provable type-I and type-II error analysis. Empirical results showed improved performance over (Kirchenbauer et al., 2023).

### Strengths
1. The proposed method is theoretically sound - it provides type-I and type-II error analysis and improves the baseline line method proposed in (Kirchenbauer et al., 2023).

2. Improved watermarking performance (in terms of detection rate and utility/perplexity) over (Kirchenbauer et al., 2023). The empirical results are thorough and convincing.

3. The paper showed robustness results and analysis against (simple) text editing.

### Weaknesses
1. Technically, the paper is sound and novel in deriving provably watermarks. However, I have some questions on the definition of false positives (see Questions).

2. On the presentation side, it reads like the authors tend to contrast their approach with the K-gram method, and the discussion has been brought up multiple times in the paper. However, only some simple text such as "general K-gram watermark works in the same way as ours, but randomly generates a different Green list for each prefix of length K − 1. In contrast, choosing K = 1 means we have a consistent green list for every new token the language model generates.'' It would be better if the authors could provide an algorithmic description on the K-gram method so the readers can directly compare it with Algorithms 1 and 2.

### Questions
1. In Definition 2.2, when comparing the defined type-I and type-II errors, I am confused about why in the type-II error, we need to assume $y$ is generated from the model $\hat{M}$, whereas in the type-I error, it is assumed that $y$ can be arbitrary. Wouldn't it make more sense to assume  that $y$ is **not** generated from model $\hat{M}$ in the type-I error analysis? Based on the current definition, if $y$ is indeed generated by $\hat{M}$, then the definition says its detection correctness is at most $\alpha$? Please clarify it.

2. For the general K-gram method, can the error guarantees and analysis be extended? If so, how will this general K value place a role in the error analysis?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper is oriented towards addressing the misuse of Large Language Models (LLMs) in generating text, with a specific focus on the field of active detection, which offers advantages over passive detection. Firstly, a rigorous theoretical framework for LLM text watermarking is formally defined, considering the text quality, the correctness of detection, and robustness against post-processing. Subsequently, a specific construction is provided within the defined theoretical framework. This involves simplifying the grouping strategy of the vocabulary from the K-Gram watermark scheme of Kirchenbauer et al.[1], to fixed grouping in order to enhance robustness. The theoretical properties of the proposed watermark instance within the framework are then theoretically proven. Finally, experimental demonstrations of the evaluation metrics in the watermark framework are conducted across multiple datasets and language models.
 
Reference:
[1] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. International Conference on Machine Learning, 2023.

### Strengths
Due to the low redundancy of the text carrier and the limited controllable information available during the gradual generation process, robustness becomes a challenge in the watermark task for active source tracing. This paper is focused on a specific 0-bit watermark task, namely human-machine text recognition, and provides a simple but robust approach to improve an existing method. The fixed grouping approach reduces the propagation of post-processing on text, which may result in some loss of text quality. However, experimental results indicate that it is comparable to the baseline. The key point is that it offers proof regarding the correctness and robustness of the proposed watermark.

### Weaknesses
This paper ensures the property of watermark quality through proof based on distribution distance. However, the relevant experiments only include perplexity and human assessment experiments, lacking distribution-based experimental validation. In addition, the definition of editing operations encompasses "paraphrase, insertion, deletion, replacement", but in the experimental section, robustness results for insertions are missing, which were replaced with "swap". What’s more, the experimental results about the impact of text length on watermark effectiveness are missing. 

It may be beneficial to include the above experimental results.

### Questions
In the contribution summary section of the main text, the statement reads as follows: "To the best of our knowledge, we are the first to formulate the LLM watermarking as a cryptographic problem and to obtain provably robust guarantees for watermarks for LLMs against arbitrary edits." Could it be explained why the work of Christ et al.[1] is not considered "the first to formulate the LLM watermarking as a cryptographic problem"?

Reference:
[1] Miranda Christ, Sam Gunn, and Or Zamir. Undetectable watermarks for language models. arXiv preprint arXiv:2306.09194, 2023.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel watermarking schema for text generators. They claim the proposed watermarking method possesses several nice properties that increase the feasibility of their approach, guaranteed generation quality, correctness in watermark detection and robustness against text editing.

### Strengths
+ The research direction is intriguing and timely. 


+ The design of the $\delta$ increase for tokens in the red list is simple and intuitively correct, although the proof is very complex and I cannot go through the details in the proof.

+ One of the properties (robust to paraphrasing) is essential to watermarks, although this property is mainly proved empirically not supported by theoretical discussion.

### Weaknesses
Some points are overclaimed, e.g., contribution 1 claims the framework is theoretically rigorous. Some related work is not properly involved in discussion and comparison. All these points need careful consideration.

+ The robustness to edits within a bound is an awesome property and can be theoretically proved, but the robustness to paraphrasing models could hardly be proven, as there is no guarantee in the *edit distance* of paraphrasing. The related claims should be treated carefully.

+ The consistency of the output word distribution is a great property and it was also discussed in previous literature, such as [1]. How will you compare this approach with theirs? The Type I/II errors were not discussed in their work, can you show your advantages against them?

+ The robustness claim on paraphrase should be constrained by empirical findings and the paraphraser should not be aware of the watermarking method as they can still conduct attacks on this method. One extreme example could be totally random choice of synonyms and another interesting paraphrasing method is to use back-translation as paraphrasing with changes of style [2]. 

[1] He, Xuanli, et al. "Cater: Intellectual property protection on text generation apis via conditional watermarks." Advances in Neural Information Processing Systems 35 (2022): 5431-5445.

[2] Prabhumoye, Shrimai, et al. "Style Transfer Through Back-Translation." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018.

### Questions
Can you confirm the main difference between Algorithm 1 in this work and Algorithm 2 in [3]? IMO, this is an essential question to understand the main technical contribution of this paper.

It would be nice to link the 'confidence' of the watermarked outputs with the length of a sentence. Usually, longer sentences provide more evidence for watermarking.

---
**Minor:**

The examples in Algorithm 2 can be placed in comments.

[3] A Watermark for Large Language Models

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
