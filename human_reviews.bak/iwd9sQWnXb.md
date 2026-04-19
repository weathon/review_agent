# Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3

## Abstract
Ranking documents using Large Language Models (LLMs) by directly feeding the query and candidate documents into the prompt is an interesting and practical problem. However, there has been limited success so far, as researchers have found it difficult to outperform fine-tuned baseline rankers on benchmark datasets. We analyze pointwise and listwise ranking prompts used by existing methods and argue that off-the-shelf LLMs do not fully understand these challenging ranking formulations. In this paper, we propose to significantly reduce the burden on LLMs by using a new technique called Pairwise Ranking Prompting (PRP). Our results are the first in the literature to achieve state-of-the-art ranking performance on standard benchmarks using moderate-sized open-sourced LLMs. On TREC-DL2020, PRP based on the Flan-UL2 model with 20B parameters outperforms the previous best approach in the literature, which is based on the blackbox commercial GPT-4 that has 50x (estimated) model size, by over 5% at NDCG@1. On TREC-DL2019, PRP performs favorably with supervised models and is only inferior to the GPT-4 solution among LLM-based methods on the NDCG@5 and NDCG@10 metrics, while outperforming other LLM-based solutions, such as InstructGPT which has 175B parameters, by over 10% for all ranking metrics. By using the same prompt template on seven BEIR tasks, PRP beats supervised baselines and outperforms the blackbox commercial ChatGPT solution by 4.2\% and pointwise LLM-based solutions by over 10\% on average NDCG@10. Furthermore, we propose several variants of PRP to improve efficiency and show that it is possible to achieve competitive results even with linear complexity. We also discuss other benefits of PRP, such as supporting both generation and scoring LLM APIs, as well as being insensitive to input ordering.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper tackles an important problem of applying unsupervised LLMs for document ranking. Authors quickly pointed out severe limitations of existing LLM-powered ranking methods. One class of such methods (pointwise) requires a calibrated score (or probability) which would be difficult to obtain between different prompts. The second class of ranking methods (listwise) suffers from issues ranging from missing expected ranked items to repetitions and various forms of inconsistency. 
To address those issues authors presented a variant of a well-known (in learning-to-rank literature) pairwise ranking method adapted for  unsupervised LLMs for ranking documents.
Besides simplicity the main benefits of this method is two-fold. First, carefully calibrated scores are not required since only two documents are being scored at a time and relative scores are sufficient. Second, only a fraction of expensive LLM API calls are needed. Additionally, the authors used relatively-small (still in billions) open source LLM with a smaller number of parameters and produced a very competitive results agains a very large black box LLMs and also against supervised methods when tested on widely used information retrieval datasets.

### Strengths
This paper is original in a sense that the authors found real problems with the existing state-of-the art in LLM-assisted ranking methods and proposed a method that appears to solve all the main issues they identified with the existing methods.
The focus of the work was not only on producing a better ranking method but also in decreasing the cost of final rankings. LLM API calls are expensive, hosting a very large LLM can be prohibitevly expensive. 
The authors addressed these problem by choosing a much smaller (in terms of parameters) open-source LLM compared to very large black-box LLMs (chatGPT variants). They also proposed several methods with decreasing number of LLM calls (allpairst, sorting-based and sliding-window)
The authors were also careful to point out that only 0.02% predictions did not follow desired format when generating output. I am assuming that the number of generation errors among listwise methods is considerably worse.
And the results of running their method on widely used datasets were very competitive with existing ranking methods.

### Weaknesses
I would like to raise a couple of issues. 
The first issue is that the authors have not shared their source code. They did mentioned the type of open-source LLM they were using but it would not be easy for somebody short-on-time to quickly reporoduce their results. I think it would be useful if the authors shared the actual prompts they used and the code they used for computing their results.

While reading this paper I realized that this paper would definitely benefit from the discussion on the following statement from the paper:
"...However, since it performs local comparisons/swaps on-the-fly and pairwise comparisons are not guaranteed to be transitive, its performance needs to be empirically evaluated..."

When comparisons are not guaranteed to be transitive, we are not working within the usual framework of a total order. I believe traditional sorting algorithms assume some transitivity. 
So not only the average runtime complexity may be affected but validity of the output as well.
Specifically for HeapSort we may end up with violating of the heap property if our ordering is not transitive since even if both children are less than their parent, we can't guarantee that the the grandchild is less than the grandparent and vice versa, breaking the heap property and the ability to maintain it.

### Questions
How the absence of the assumption of the total order would affect the the total number of API calls (reported in Table 1)

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a pair-wise prompting strategy for unsupervised information retrieval using large language models. The author(s) first analyze the drawbacks of point-wise or list-wise ranking w/ LLMs, then propose two paradigms to call LLMs on pair-wise ranking, namely scoring mode and generation mode. To alleviate the efficiency issue, the author(s) use efficient ranking algorithms or sliding windows to accelerate the whole ranking process. Extensive experiments are conducted on two public benchmarks. Surprisingly, open-source LLMs w/ the proposed pair-wise prompting method can achieve comparable or better results compared to GPT-4.

### Strengths
1. The paper studies an important task, i.e., information retrieval.
2. Timely study on information retrieval using large language models.
3. Extensive experiments are conducted on two public benchmarks.
4. Surprisingly results of ranking w/ open-source LLMs using the proposed pair-wise prompting strategies.

### Weaknesses
1. Limited novelty. The main novelty lies on twofolds. (1) pair-wise prompting (2) accelerating w/ sorting or sliding window algorithms.
    1. For pair-wise prompting, it is straightforward to consider such kind of ranking paradigm (given that point-wise, pair-wise, and list-wise are three typical paradigms usually discussed in the field of information retrieval). The pair-wise ranking prompting strategy has also been explored in existing works [1].
    2. The sliding window strategy has also been explored in works about searching using LLMs [2].
    3. As a result, although the proposed method shows promising and strong results, especially using open-source LLMs, the new insights provided by this paper are limited.

2. Code is not available for reproduction, which is crucial for this work because it's based on open-source LLMs.

3. The time complexity of the sliding window strategy could be misleading. In information retrieval tasks, it's common to have a relatively large K like 100. As a result, O(KN) is a more precise time complexity for this approach.

4. It would be better if the author(s) could quantitatively present the efficiency of each approach, e.g., comparing the number of LLM calls.

5. Typo (Minor). The bottom of page 6, "second to the blackbox, commercial gpt-4 based solution on NDCG@5 and NDCG@10". However, according to Table 2, should it be NDCG@{1, 5, 10}?


[1] Dai et al. Uncovering ChatGPT’s Capabilities in Recommender Systems.

[2] Sun et al. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.

### Questions
Please refer to "Weaknesses".

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This research paper introduces a novel approach to unsupervised information retrieval using large language models (LLMs), focusing on the drawbacks of existing point-wise and list-wise ranking methods when paired with LLMs. The authors propose a pair-wise prompting strategy, featuring two paradigms: scoring mode and generation mode, to address these limitations. They highlight the inefficiency of calibrated scoring in point-wise methods and the inconsistencies in list-wise methods. To enhance efficiency, they employ advanced ranking algorithms and sliding window techniques. This approach simplifies the ranking process by only requiring relative scores between two documents, significantly reducing the need for costly LLM API calls. Extensive testing on public benchmarks reveals that open-source LLMs, even those smaller in scale, can achieve competitive or superior results compared to larger, more complex models like GPT-4. This demonstrates a promising advancement in document ranking using unsupervised LLMs.

### Strengths
+ Addresses an important and current task in information retrieval using large language models (LLMs).

+ Conducts many experiments on two public benchmarks, demonstrating the effectiveness of the proposed pair-wise prompting strategies.

+ Achieves competitive results with open-source LLMs compared to larger models like GPT variants, balancing performance and cost-efficiency.

+ Proposes various innovative methods (all-pair, sorting-based, sliding-window) to decrease the number of LLM calls.

### Weaknesses
- Limited Novelty: The idea of pair-wise comparison is not entirely new, as it's a common paradigm in the field of information retrieval and has been explored in previous works. In addition, the concept of the sliding window strategy for search acceleration has also been previously discussed in the literature.

- Lack of Code Availability for Reproduction: The absence of shared source code and specific prompts used in the experiments makes it challenging for others to reproduce and verify the results. Moreover, given the reliance on open-source LLMs, there is a concern that the outcomes might significantly vary depending on the choice of LLMs used.

- Inconsistency in Experimental Results: The research shows a significant inconsistency in the performance of the proposed PRP-Sliding-10 method across different benchmarks, specifically TREC-DL2019 and TREC-DL2020. It raises concerns when the method outperforms RankGPT with GPT4 in TREC-DL2020, yet fails to surpass RankGPT in TREC-DL2019. This discrepancy in results between similar benchmarks indicates potential issues with the method's robustness or adaptability across varying datasets. This inconsistency might be due to factors like varying LLM architectures or differing data sets, which can significantly impact the replicability and applicability of the research in real-world scenarios.

- Mismatch with Conference Focus: A significant weakness of the paper is its apparent misalignment with the core themes of the International Conference on Learning Representations (ICLR). Given that ICLR primarily concentrates on advancements in learning representations, the lack of discussion or exploration into this area in the paper stands out. This omission could limit the paper's relevance and appeal to the ICLR audience, who expect contributions that delve into learning representation theories, methodologies, or applications. This paper seems more suitable for ICPE (International Conference on Prompt Engineering).

### Questions
Please refer to Weaknesses.

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
The authors propose an unsupervised approach for document ranking using a pair-wise prompting strategy. The primary motivation is to overcome the limitations of the point wise and list wise ranking approaches. The authors conduct experiments on multiple datasets to demonstrate the effectiveness of the PRP approach as compared to both supervised and unsupervised methods. Further, the authors also aim to make the approach more cost-friendly. They have demonstrated the performance of the various optimization approaches as well.

### Strengths
The paper proposes a novel solution to ranking using LLMs
The authors have demonstrated their results on several datasets and compared them to several approaches. 
The paper also focused on cost efficiency.

### Weaknesses
1. The paper seems less suitable for ICLR, but to another venue that focuses on practical/ applied aspects of using LLMs for information retrieval. The approaches are more incremental. 

2. The authors have a note in the appendix about reproducibility, but the details are limited. Instead of pointing to Figure 2 as an example, a better description of the prompting approach would be helpful. This is more important given that the authors are aiming to aid academic research, as noted in multiple places across the paper.

3. As for the efficiency, the authors seem to be focusing on relatively smaller set of documents and ranking problems, where they choose 10 out of 100 documents. It will be good to know the performance on larger sets. 
Moreover, a quick-sort like partition algorithm might yield better results than a bubble-sort based approach. May be that is why PRP-Sliding-10 has better NDCG@1 but lower NDCG@5 and NDCG@10 than PRP-Sorting.

### Questions
Noted in the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
