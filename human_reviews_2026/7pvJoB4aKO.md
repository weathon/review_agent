# Exploring Knowledge Purification in Multi-Teacher Knowledge Distillation for LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Knowledge distillation has emerged as a pivotal technique for transferring knowledge from stronger large language models (LLMs) to smaller, more efficient models. However, traditional distillation approaches face challenges related to knowledge conflicts and high resource demands, particularly when leveraging multiple teacher models. In this paper, we introduce the concept of **Knowledge Purification**, which consolidates the rationales from multiple teacher LLMs into a single rationale, thereby mitigating conflicts and enhancing efficiency. To investigate the effectiveness of knowledge purification, we further propose five purification methods from various perspectives. Our experiments demonstrate that these methods not only improve the performance of the distilled model but also effectively alleviate knowledge conflicts. Moreover, router-based methods exhibit robust generalization capabilities, underscoring the potential of innovative purification techniques in optimizing multi-teacher distillation and facilitating the practical deployment of powerful yet lightweight models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a critical problem in multi-teacher knowledge distillation (KD) for Large Language Models (LLMs): the performance degradation that occurs as the number of teachers increases, attributed to knowledge conflicts. The authors introduce the concept of Knowledge Purification, which aims to consolidate the rationales from multiple teachers into a single and coherent rationale. They propose five purification methods in total, Knowledge Aggregation using an LLM to synthesize rationales, three LLM routing approaches (Plackett-Luce Ranking, PLM Classifier, and Similarity-based Router) which select the best single teacher's rationale based on the input question, and RL-based Teacher Selection that uses the student's performance as a reward to choose a teacher. Extensive experiments on commonsense and biomedical reasoning tasks show that routing-based and RL-based methods significantly outperform baselines like TinyLLM, and good generalization to out-of-domain datasets.

### Strengths
* The proposal of the concept of knowledge purification
* Five methods are proposed for a comparative analysis across multiple dimensions (performance, CMV, out-of-domain generalization, etc.), and the analysis (e.g., Table 2) shows the trade-offs of each method
* The proposed methods consistently outperform baselines, as well as the out-of-domain generalization

### Weaknesses
* The experiments are limited to four selected teacher models
* Though the overall performance gain is clear, why certain methods work better needs more explanation. It would be helpful if the authors could provide more analyses of pros and cons of each method
* Excluding the TwT baseline (in Appendix C.3) is reasonable but not fully convincing

### Questions
1. The performance of knowledge aggregation is relatively weak. Have you investigated why a powerful LLM fails to synthesize a good consolidated rationale? Is the issue the aggregation prompt or the task is inherently difficult?
2. Since the router-based methods perform well and they only need the input question, could it be used to select the teacher per query during the data generation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work addresses multi-teacher knowledge distillation (KD), focusing on the challenges of knowledge conflict (i.e.divergent rationales among teachers) and computational cost when aggregating many teacher models. The authors propose Knowledge Purification, a concept for consolidating the rationales from multiple teacher LLMs into a single rationale to use for distillation. They design five purification methods based on aggregation, routing, and RL-based teacher selection styles.

### Strengths
- The introduction of Knowledge Purification reframes multi-teacher distillation from the perspective of rationale integration rather than mere logit or feature averaging, addressing an important problem of multi-teacher KD.

- The paper proposes five distinct purification strategies (aggregation, routing, and RL-based).

- The paper is well written and structured.

### Weaknesses
- The experimental evaluation is limited to commonsense and biomedical reasoning datasets. To establish the generality of the proposed approach, it should be extended to a broader range of tasks such as mathematical reasoning, coding, and instruction following.

- Although the paper emphasizes improved efficiency, it does not provide quantitative evidence (e.g., training time, GPU hours) compared to existing multi-teacher methods like TinyLLM or TwT.

- The baselines are limited to step-by-step distillation and TinyLLM, omitting state-of-the-art knowledge distillation approaches such as ABKD, MiniLLM, DistillM, or CKA-KD, which could challenge the claimed improvements.

### Questions
See weaknesses section

### Soundness
2

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
This paper addresses a critical challenge in multi-teacher knowledge distillation (MTKD) for Large Language Models (LLMs): the performance degradation caused by knowledge conflicts among the rationales provided by multiple teacher models. The authors identify that simply increasing the number of teachers in frameworks like TinyLLM does not monotonically improve student performance, often harming it due to conflicting or hallucinated reasoning paths. To solve this, the authors introduce the concept of "Knowledge Purification" (KP), which aims to consolidate the rationales from multiple teachers into a single, coherent rationale before distillation. This process mitigates conflicts and provides the student model with a unified source of knowledge. Through extensive experiments on commonsense and biomedical reasoning tasks, the paper demonstrates that KP methods, particularly the Similarity-based Router and RL-based Teacher Selection, consistently outperform strong baselines like TinyLLM and Distilling-Step-by-Step.

### Strengths
1. The idea of "Knowledge Purification" is a direct, intuitive, and novel solution to the clearly identified problem of knowledge conflict in MTKD.
2. The proposal of five methods from different families (aggregation, routing, RL) provides a thorough exploration of the solution space. This allows for a nuanced comparison of trade-offs between performance, computational cost, and transferability.

### Weaknesses
1. As acknowledged in the limitations, the study is constrained to a ensemble of only four teacher LLMs. A critical question remains: how do these methods scale to 10, 20, or even more teachers? While the results with 4 teachers are promising, the effectiveness and computational overhead of, for example, the RL-based method or the PL ranking with a much larger pool of teachers is unexplored.
2. The study is exclusively validated on multiple-choice question answering tasks. While this is a standard benchmark for reasoning, the generality of Knowledge Purification to other NLP tasks like open-ended generation, summarization, or translation is not established.
3. While the routers offer excellent inference-time efficiency, the cost of training them is non-trivial (requiring a "public set" and 5000 training epochs). A discussion on the trade-off between the cost of training a router versus the cost of repeatedly sampling from all teachers for distillation across multiple tasks would be beneficial.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes knowledge purification, which consolidates rationales from multiple teacher LLMs into a single rationale to address knowledge conflicts in distillation. Five methods are introduced: knowledge aggregation, LLM routing (Plackett-Luce ranking, PLM classifier, similarity-based router), and RL-based teacher selection. Experiments on commonsense and biomedical reasoning tasks show some performance gains, but the improvements are modest and lack groundbreaking insights.

### Strengths
(1) The focus on knowledge conflicts in multi-teacher distillation is attractive.

(2) Experiments cover multiple datasets and student models, providing a broad assessment. 

(3) The five purification approaches offer varied perspectives, from simple aggregation to learned routing. Experimental results directly show the performance of the methods.

### Weaknesses
(1) The core idea of rationale consolidation resembles prior work on knowledge fusion and ensemble distillation. Methods like Plackett-Luce ranking and similarity-based routing are direct adaptations from the existing literature, showing limited substantial innovation. The paper only provide an overall view of the different methods.

(2) Methods like RL-based selection and aggregation involve significant complexity and does not present an obvious theoretical advantage through classical methods.

(3) While the experimental results show some improvements, the gains are often marginal (e.g., ~1–3% accuracy boosts in Table 1).

### Questions
1. How does knowledge purification fundamentally differ from applying ensemble methods (e.g., weighted averaging, majority voting or averaging student model) to the teacher rationales? 

2. Why CMV was preferred over the information-theoretic metrics like Jensen-Shannon Divergence (JSD), which could quantify the divergence between teacher rationales?

3. Based on the experiments, which methods should we choose when facing the situation of multiple teacher distillation (under different scenarios)?

### Soundness
2

### Presentation
2

### Contribution
3
