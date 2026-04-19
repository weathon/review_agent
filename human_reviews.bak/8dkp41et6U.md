# LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression

- Decision: Reject
- Scores: 6, 6, 6, 5, 6

## Abstract
In long context scenarios, large language models (LLMs) face three main challenges: higher computational/financial cost, longer latency, and inferior performance. Some studies reveal that the performance of LLMs depends on both the density and the position of the key information (question relevant) in the input prompt. Inspired by these findings, we propose LongLLMLingua for prompt compression towards improving LLMs’ perception of the key information to simultaneously address the three challenges. We conduct evaluation on a wide range of long context scenarios including single-/multi-document QA, few-shot learning, summarization, synthetic tasks, and code completion. and experimental results show that LongLLMLingua compressed prompt can derive higher performance with much less cost. The latency of the end-to-end system is also reduced. For example, on NaturalQuestions benchmark, LongLLMLingua gains a performance boost of up to 17.1% over the original prompt with ∼4x fewer tokens as input to GPT-3.5-Turbo. It can derive cost savings of `$`28.5 and `$`27.4 per 1,000 samples from the LongBench and ZeroScrolls benchmark, respectively. Additionally, when compressing prompts of ∼10k tokens at a compression rate of 2x-10x, LongLLMLingua can speed up the end-to-end latency by 1.4x-3.8x.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents LongLLMLingua, a prompt compression method to improve the high computation cost, longer latency and inferior performance in current long context LLMs.

### Strengths
1. the use of compression is well motivated.
2. the use of reordering mechanism is a promising method to solve the lost in the middle phenomenon.
3. the evaluation is throughout and comprehensive, especially Table 1 provides a fair and informative comparison, by using Chat-gpt and LongChat as target models (representative from close-sourced and open-sourced family).

### Weaknesses
1. The latency evaluation setup is not well presented. The latency of API calls may have a high variance. Are the experiments conducted several times, and in different hours? If not, the reviewer suggests improving this and update the evaluation setup.

### Questions
Please address the weakness above.

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
This paper introduces LongLLMLingua as an innovative solution for prompt compression aimed at reducing costs and latency. LongLLMLingua stands out for its ability to consider the content of a question during compression, focusing on key tokens to enhance performance in scenarios with lengthy contexts. To achieve this, the authors propose several significant contributions: a question-aware compression framework, a document reordering based on a newly proposed importance metric, dynamic compression ratios, and a post-compression recovery strategy.

### Strengths
I appreciate the logical structure and clarity of this paper. The authors present their motivations compellingly, and the proposed LongLLMLingua method is both intuitive and seemingly effective, as evidenced by the strong results reported.

### Weaknesses
I have several questions and comments regarding the methods outlined in the paper that I hope the authors can address:

1. The foundational work upon which this research is built, LLMLingua, should be appropriately cited from [https://arxiv.org/abs/2310.05736](https://arxiv.org/abs/2310.05736), where the authors are clearly identified. Citing it with "Anonymous" authors seems unusual.

2. It might be beneficial to consolidate the list of contributions into three primary ones. Several contributions, such as document reordering, information reduction loss ... etc (discussed in Sections 4.2-4.4), seem relatively minor individually. They might be more appropriately categorized as a set of techniques or "tricks" that collectively enhance the framework's performance and stability.

3. Figure 5 alone seems insufficient to establish the full effectiveness of the framework. To provide a more comprehensive understanding, Tables 1 and 3 should include E2E runtime data, particularly to offer perspective on the performance of retrieval-based methods.

4. The contrastive perplexity-based importance metric's effectiveness isn't entirely convincing. Figure 3 (b) presents just a single example from a QA dataset where key documents are positioned at the beginning. Wouldn’t it be more informative to compare the average contrastive perplexity of key documents against that of nonessential documents? Similarly, a comparison using standard perplexity might also be illuminating.

5. I believe there might be a typographical error: 'ground-true' should likely be corrected to "ground-truth."

### Questions
see weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a technique, LongLLMLingua, to accelerating and enhancing LLMs in Long-context via compression. The method involves several aspects, coarse-to-fine compression, reordering mechanism to reduce information loss in the middle. adaptive compression control and post-compression recovery strategy. Experiments are based upon GPT-3.5 api and LongChat-13B-16k models. It presents results on several benchmarks and show the effectiveness of the proposed method.

### Strengths
1. The method is detailed and comprehensive. It includes many aspects that can be helpful for the compression and long-context prompting.

2. The compression ratio is great. It introduces 2x-10x compression rate and 1.4x-3.8x latency speed-up. It is very promising.

3. Results on benchmarks are good as shown in Table 1, 2, and 3.

4. The visualization in Figure 2 is very clear for me to understand this method.

### Weaknesses
1. In the methods, it contains several aspects coarse compression, fine compression, reordering mechanism to reduce information loss in the middle. adaptive compression control and post-compression recovery strategy. It seems that it lacks a detailed ablation study that the influence on the compression rate and performance from each aspect. This is important for us to have a better understanding of this paper.

2. For the reordering method to avoid the lost-in-the-middle issue, I have a question that whether the reordering will disturb the time-line between each document. For example, if the input documents are some sections in a fiction, the latter sections depend on the content of the former ones to understand. If so, whether this operation will disturb or introduce other difficulties for understanding?

3. Based on Figure3b, it shows that contrastive perplexity is much more stable than the original perplexity metric. Would the authors provide more detailed explanation or mathematical proof on the reasons for this? It is a bit unclear for me to understand this difference.

4. In the benchmark comparison, it would be better to include the GPT-3.5-Turbo-0613 and LongChat-13B-16k for comparison. Although they might be slower and more expensive, it would be helpful to understand that the cost from the compression.

### Questions
Please see the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper builds upon the LLMLingua method, which utilizes a smaller LM to assess the perplexity of each token in a prompt, removing less informative tokens to compress the total length and reduce required compute. The authors identify several shortcomings of the method when applied to long inputs. Specifically, they conduct the perplexity evaluation conditioned on the question, implement a two-step compression with a dynamic compression threshold to first eliminate redundant documents and then compress the remaining context, and rearrange the remaining document to counteract the LMs' tendency to overlook crucial information when it appears in the middle of the context. From their evaluations, the authors ascertain that the LongLLMLingua method not only enhances the performance of black-box (API-based) LMs but also their accuracy.

### Strengths
The resulting technique is easy to use, and is general enough to be applicable in every long input scenario, including black-box api based LLMs. While several building blocks in the pipeline are specific to certain use-cases, the overall question-aware compression technique is applicable widely, and their results suggest it may be useful to simultaneously reduce costs and latency and even improve the downstream performance.

### Weaknesses
Some of the main contributions proposed in the paper are somewhat limited in their usefulness in a diverse set of naturally occurring tasks.
1. Removing documents and reordering is only relevant in the multi-document scenario where the relevant information is pertained in a small subset of the documents, and the rest are noise (e.g. in open-book question answering). Many use cases (e.g. long-input summarization and question answering, code understanding, multi-hop question answering etc.) will not benefit from it and may even be hurt by this procedure. Namely, in mult-hop question answering where one can determine the relevance of a downstream document only after answering the outer-part of the question, important documents may be filtered out with the procedure proposed in the paper.
2. Subsequence recovery is only relevant in very specific cases. Namely, where the task is extractive question answering. However, many real-world use cases that contain naturally occurring tasks over long inputs are not extractive rather generative. 
3. The question-aware compression is only relevant when a question (or more generally, a long and specific instruction) is given. In use cases such as summarization, or conditional generation this is not the case.

All together, these specifications limit the contribution of the method when contrasting with the existing LLMLingua method which includes the remaining parts of the pipeline.

One of the main pieces of evidence in the paper is the evaluation on ZeroSCROLLS. However, I find this evaluation unsatisfactory, as the paper mentions they use the evaluation set of the benchmark, while Shaham et al. 2023 mentions explicitly that the evaluation set contains ``a small number of examples (~20 per task) in a "validation" split, meant for eyeballing purposes’’ and does not have enough statistical power to be used as statistical measurement (https://github.com/tau-nlp/zero_scrolls). Additionally, the authors do not provide a breakdown of the results to show that their method is indeed beneficial across the different scenarios. 

The paper also considers several changes as significant contributions, but there are no ablation studies to show their usefulness. Namely, in §2 they mention the usage of $x^{doc} \| x^{que} \| x^{restrict}$ to compute the relevance of each document as an important contribution. An ablation on the ordering of the question and document should be added, as well as for the relevance of the restricting prompt. Moreover, all ablation studies that were performed were done on the NaturalQuestions dataset which is explicitly tailored for the proposed contribution. Specifically, one document contains the relevant information while the others are distractors. I would like to see the ablation study conducted on multi-document scenarios where this is not the case, such as as the multi-hop question answering scenario (e.g. MuSiQue [Trivedi et al., 2021] which also appears in ZeroSCROLLS).


The authors mention using a small LM to perform all pre-computation and compressions needed. However, they use LLaMA-2-7B-chat as their small LM, which has a significant overhead in itself, and may not be widely applicable to many use-cases. An ablation study on the performance of using smaller models should be added. 

Comments on presentation and typos for the authors (not a weakness, but should be addressed prior to final publication):
1. In Page 8 (§5), the indexing of the two tables is reversed. Namely, Table 3 appears before table 2.
2. “Less cost” is used twice instead of “lower cost” (abstract and the main results paragraph in §5).
3. In the abstract, one of the sentences starts with “. and experimental …” where it should have been “. Experimental …” 
4. “Derive costs” is used several times (including in the abstract) instead of “Drive costs”.
5. In §2 “ground-true” instead of “ground-truth”.
6. The styling guide indicates that third level headings (namely paragraph titles) should be in small caps, and not capitalized as appearing in the paper. See https://iclr.cc/Conferences/2023/CallForPapers for style information. 
7. In §4.3 you denote the number of documents with $N_d$ while it was already denoted as $K$ beforehand. 
8. In §4.3, equations (5) are a bit confusing, as multiple $x_i$ exist depending on $x_k$, thus multiple $\tau_i$ exist. This phrasing should be made more explicit for easier reading.

### Questions
1. In §3 you mention that for the distribution alignment, an instruction tuning of the small LM is performed. Was this finetuning done separately for each test case? If so, in the NaturalQuestions case, were the answer positions considered a specific case? Please provide more details on the finetuning procedure.
2. In page 5 §4.1 you say that “the tokens with high contrastive perplexities concentrate more on the left side of the dashed line, which corresponds to the document that contains the answer to question”. If I understand it correctly, this experiment was only done in the case where the ground-truth document is in the first position? If so, isn’t it  possible that the fact the contrastive learning approaches zero as the token position increases is simply a symptom of the “short-term memory” of perplexity (Khandelwal et al., 2018; Sun et al., 2021, Press et al., 2021a,b)?
3. Can you please provide a breakdown on the ZeroSCROLLS test set benchmark? Additionally, it would be helpful to compare your results with the baseline results of the same models where there were no token constraints.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes LongLLMLingua, a question-aware coarse-to-fine compression method to compress prompts and improve the key information density. The empirical results demonstrate that LongLLMLingua can substantially compress the prompts while maintaining the model performance and improving efficiency.

### Strengths
1. The method is novel. This paper proposes to discard irrelevant documents and tokens iteratively in a coarse-to-fine manner, which is novel and effective. 
2. Measuring the token importance by perplexity is intuitive and insightful. Such an approach can be applied to black-box models, which is an advantage. 
3. The empirical results are sound, demonstrating the effectiveness and efficiency of LongLLMLingua.

### Weaknesses
1. Prompt compression can be effective for tasks like QA, where the key information is sparsely distributed. In contrast to tasks, like summarization, the key information can be evenly distributed within inputs. Coarsely dropping a large amount of input may hurt the performance. 
2. The inference latency can be improved by LongLLMLingua. However, the models need to evaluate the perplexity and compress prompts every time, which leads to non-trivial latency.

### Questions
1. How is the thresholding of $s_i$ determined to discard tokens of lower importance? Is it determined by the current budget $\tau_k$ of the documents? 
2. Section 4.3 introduces a dynamic budget scheduler. It is unclear to me how the iteration is defined here. Is there an iterative evaluation of token importance? 
3. I am interested in the perplexity distribution within sentences.  If the perplexity varies dramatically across tokens of one sentence, it can happen that a few tokens are retained sparsely, making the sentence inconsistent.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
