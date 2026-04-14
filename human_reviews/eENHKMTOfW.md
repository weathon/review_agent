# Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
The rise of large language models (LLMs) has created a significant disparity: industrial research labs with their computational resources, expert teams, and advanced infrastructures, can effectively fine-tune LLMs, while individual developers and small organizations face barriers due to limited resources to effectively explore the experiment space. In this paper, we aim to bridge this gap by presenting a comprehensive study on supervised fine-tuning of LLMs using instruction-tuning datasets spanning diverse knowledge domains and skills. We focus on small-sized LLMs (3B to 7B parameters) for their cost-efficiency and accessibility. We explore various training configurations and strategies across four open-source pre-trained models. We provide detailed documentation of these configurations, revealing findings that challenge several common training practices, including hyperparameter recommendations from TULU and phased training recommended by Orca. The code used for the experiments can be found here: https://github.com/instructlab/training.

Key insights from our work include: (i) larger batch sizes paired with lower learning rates lead to improved model performance on benchmarks such as MMLU, MTBench, and Open LLM Leaderboard; (ii) early-stage training dynamics, such as lower gradient norms and higher loss values, are strong indicators of better final model performance, allowing for early termination of sub-optimal runs and significant computational savings; (iii) through a thorough exploration of hyperparameters like warmup steps and learning rate schedules, we provide guidance for practitioners and find that certain simplifications do not compromise performance; and (iv) we observe no significant difference in performance between phased (sequentially training on data divided into phases) and stacked (training on the entire dataset at once) strategies, but stacked training is simpler and more sample efficient. With these findings holding robustly across datasets as well as model families and sizes, we hope this study serves as a guide for practitioners fine-tuning small LLMs and promotes a more inclusive research environment for LLM development.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents an in-depth study on fine-tuning small LLMs with 3B to 7B parameters using large-scale instruction tuning datasets across various knowledge domains and skills. The study challenges common training practices and offers new insights for customizing LLMs effectively. Key findings include the benefits of larger batch sizes with lower learning rates, the predictive value of early-stage training dynamics for final model performance, the lack of necessity for warmup phases, and the superiority of stacked training over phased training. These results provide an actionable and comprehensive guide for practitioners working on fine-tuning smaller LLMs.

### Strengths
S1: The paper gives practical guidance for fine-tuning small LLMs, which can be highly beneficial for researchers and practitioners.

S2: The experimental study is well-executed, covering a broad range of training configurations such as batch size, learning rate schedules, and training methods. It also tests three models against three datasets.

S3: The paper is well written. The related work section is comprehensive.

### Weaknesses
### Major weaknesses

W1: The paper doesn't examine the generalizability of the presented insights on new models and more domain-specific datasets. One of the motivations for using small LLMs is that practitioners can more effectively customize them for specific domains (L39). However, all the datasets used in the paper focus on general tasks (e.g., language understanding and STEM questions). To strengthen the paper's practical contributions, it would be beneficial to include training datasets and evaluation benchmarks from more specialized areas (e.g., legal documents, customer service, financial reports).

W2: Related to W1, the choice of the LLM models in the study seems arbitrary. Does the suggested training strategies work as well with the more popular small models (e.g., phi, llama)? The study could be improved by including these well-known open-source models.

W3: The main findings are not particularly surprising and somewhat expected (e.g., larger batch size improves performance, performance-efficiency trade-off).

### Minor weaknesses

M1: L37 needs citation for evidence.

### Questions
Q1: How well does the recommended training strategies work with the popular small LLMs (e.g., Phi, Llama)?

Q2: L38 argues that fine-tune smaller model is good for researchers with limited infrastructure support. However, L82 argues that to train a small model well, one needs a larger batch size and specified resources. Is this a contradiction?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper performs a deep investigation of the common practices used while fine-tuning small LLMs for domain specific downstream tasks. By conducting a range of experiments with Llama and Mistral models sized 3-7B, they provide a set of guidelines on the best practices to be used for instruction fine-tuning language models for specific tasks. Overall, using larger batch sizes with lower learning rates, a mixture of datasets rather than data curated phase-wise to prevent catastrophic forgetting and monitoring gradient norms and training loss during the initial stages of fine-tuning were shown to yield optimal performance across the tested models.

### Strengths
- The paper investigates an important and growing field of developing small LLMs for specialized domains to balance their task memeorization ability and generalization.
- The quality and design of experiments are good, and many commonly used techniques have been investigated, providing valuable guidance to practitioners.
- Overall, the paper is well-written and easy to understand. Though it has some limitations in terms of the breadth of experiments (mentioned below), and some results are already known, such as using a larger batch size and a lower learning rate -- I believe the community would still benefit from the findings of this work.

### Weaknesses
- It would have been nice to have a discussion on optimization techniques such as LoRA and the increasingly popular approximate optimization algorithms like GaLore as these are widely used while fine-tuning small LLMs.
- The authors could have included evaluations on harder datasets involving more reasoning, but this has been mentioned by them as a limitation.

### Questions
- The term "skill" is used quite loosely. What do you refer to as a skill - a specific domain like math or a reasoning like CoT?
- Phase 10 of complex skill development trains the model on tasks like poetry writing etc. - is this also organized as an instruction - answer task? If not, then I am curious to know how you handled different data styles (eg. instruction format QA and knowledge data from books etc.)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to fine-tune small-size LLMs on large-scale instruction-tuning datasets effectively, measuring downstream performance on MMLU and MTBench. The experiments use Granite 3B, 7B, and Mistral 7B as the base models and compare stacked versus phased training, varying batch sizes, learning rates, and training configurations. The authors find that stacked training is preferable, a trade-off between batch size, and the possibility of omitting warm-up steps without sacrificing model quality.

### Strengths
- The investigation is well-motivated given the increasing interest in customizing general LLMs for domain-specific tasks. Understanding the design decisions and potential trade-offs is an important area of study.
- The writing was also generally easy to follow.

### Weaknesses
The experimental setup could be better justified. It was unclear what order of experiments the authors ran and to what extent are the findings dependent on this ordering. It seems like the authors did not do a full sweep of the entire cross-product of experimental conditions (if so, please present the full set of results for generality). Furthermore, other natural questions are:
- Why these choices of models?
- How were the fine-tuning datasets curated? This was not discussed in Section 2.1 or A.2. How might results vary depending on whether the data you are trying to fine-tune is more general or specific? This is related to the motivating telecommunications example.
- Why are MMLU and MTBench the right datasets for evaluation?
- Why were TULU vs LAB used as primary comparisons for hyperparameter configurations?

The presentation of experimental results could be significantly improved. 
- Showing individual training curves is quite noisy, could the authors run with multiple seeds and present smoother results? 
- Further, why did the authors choose not to present most results in a table? This way it would be much easier to capture findings across all three models. 

Clarity on claims and significance of results.
- There is generally a lack of baselines in this work. It’s important to note what are baseline performance of the various Granite and Mistral models on MMLU an MTBench.
- Relatedly, the strength of the claimed results could be clarified. For example, many of the reported numbers are very close: 6.77 vs 6.76 on MTBench in Figure 1a, 0.5251 vs 0.5242 in Figure 3a, and 0.59 vs 0.6 vs 0.61 in Figure 4a? Having some baseline values would help readers determine whether these are significant differences.

### Questions
Please address the specific questions raised in the weaknesses section.

The revised presentation of the work addresses many of my concerns and I have increased my score accordingly.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores methods for making smaller language models (3B–7B parameters) effective in specific domains without requiring extensive computing power. It focuses on optimizing model fine-tuning through adjustments in batch size, learning rates, and training approaches. The authors compare two main training strategies: stacked training, where models learn from varied data all at once, and phased training, where data is introduced in stages. Their findings show that stacked training is both faster and more efficient. Additionally, using larger batch sizes and lower learning rates enhances model performance on benchmarks. The research offers practical tips, such as skipping warmup phases and using a constant learning rate, which simplify the training process. This guide provides useful insights for developers and organizations that want to adapt small language models effectively for specialized tasks on limited resources, making advanced AI more accessible.

### Strengths
1.For the fine-tuning stage of small-scale LLM instructions, by utilizing different types of data and exploring various parameter settings and data organization methods, comprehensive experimental conclusions have been obtained, especially for the setting of large batch size, which is often easily achievable in real-world scenarios through gradient accumulation techniques.
2.By evaluating stacked versus phased training approaches, the paper demonstrates that stacked training is generally more efficient, saving time and resources. This finding is especially beneficial for practitioners looking to balance performance with computing constraints.
3.The final conclusion drawn in the paper challenges widely recognized practices such as TULU.

### Weaknesses
1.The study is focused on small models (3B–7B parameters) and may not generalize to larger models. As such, its findings might not apply to larger language models often used in advanced applications, which limits the scope of the conclusions.
2.The experiments mainly use Granite and Mistral model families, leaving out other architectures. This limits the ability to generalize the results to models with different foundational architectures or pre-training techniques.  I suggest adding the llama3.2 series (1B, 3B) as well as the llama3.1-8B model when resources permit.
3.The evaluation relies primarily on MMLU and MTBench benchmarks, which, while broad, may not represent all potential application areas. This leaves open the question of whether the findings would hold on other benchmarks like GSM8K or ARC, which focus on different types of reasoning and knowledge.
4. There are some details in the paper that need further checking, such as the repetition of the citation to the paper Lab: Large-scale alignment for chatbots.

### Questions
See weeknesses

### Soundness
3

### Presentation
3

### Contribution
2
