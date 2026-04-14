# Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignment

- Decision: Accept (Poster)
- Scores: 3, 6, 6, 6

## Abstract
Adapting large language models (LLMs) to specialized domains typically requires domain-specific corpora for continual pre-training to facilitate knowledge memorization and related instructions for fine-tuning to apply this knowledge.
However, this method may lead to inefficient knowledge memorization due to a lack of awareness of knowledge utilization during the continual pre-training and demands LLMs to simultaneously learn knowledge utilization and format alignment with divergent training objectives during the fine-tuning.
To enhance the domain adaptation of LLMs, we revise this process and propose a new domain adaptation framework including domain knowledge learning and general format alignment, called \emph{Mix-CPT}. Specifically, we first conduct a knowledge mixture continual pre-training that concurrently focuses on knowledge memorization and utilization. To avoid catastrophic forgetting, we further propose a logit swap self-distillation constraint. By leveraging the knowledge and capabilities acquired during continual pre-training, we then efficiently perform instruction tuning and alignment with a few general training samples to achieve format alignment.
Extensive experiments show that our proposed \emph{Mix-CPT} framework can simultaneously improve the task-solving capabilities of LLMs on the target and general domains.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes the Mix-CPT strategy to mix knowledge memorization and utilization into the continual pre-training stage. Specifically, the authors eliminate the chat template of general instruction-following and preference-alignment data (which is traditionally used in the SFT stage with chat templates), and mix them with domain knowledge corpus for vanilla next-token prediction optimization. A logit swap self-distillation strategy is developed to mitigate catastrophic forgetting. After that, the authors derived a format alignment score to select a small number of instructions from the CPT dataset to execute SFT and DPO, in order to focus mainly on pure style or format learning for downstream tasks.

### Strengths
1. The research point is important. Efficient knowledge memorization is crucial to derive both fundamental and domain-specific LLMs.
2. The idea of decoupling knowledge learning and format alignment has promising efficacy, as they serve different purposes.
3. The paper is well-written and easy to follow.

### Weaknesses
1. My core concern is the lack of novelty. Taking SFT samples into the CPT stage has been explored by several works[1][2], but this paper neither discusses the methodology difference nor compares the experimental performance. Besides, some influential technical reports[3] have pointed out that moving SFT samples to CPT cannot help the final model performance, since they will eventually go through the SFT and preference alignment stages. I wonder how the authors consider this problem.
2. I'm not convinced whether general alignment training samples can truly teach LLMs to utilize domain knowledge to answer professional questions. In lines 73-75, the referred two papers (RLHF and DPO) cannot support this hypothesis. 
3. The experiment setting is unreasonable. Math, code, and wiki are hard to be considered as 'domain knowledge', since they have already been exposed to LLM's pre-training corpus and take a considerable proportion.  The 'knowledge injection' should involve specific domains[1][2] like medicine, finance, etc.

[1] Adapting large language models via reading comprehension. 2023.

[2] Instruction pre-training: Language models are supervised multitask learners, 2024.

[3] Deepseek llm: Scaling open-source language models with longtermism. 2024.

### Questions
1. What is the FAS score's physical meaning?  Does a higher score mean a better or worse sample? Besides, if the authors aim at selecting the samples that LLM have not currently learned the format, taking Eq.8 for an example, it seems more reasonable to compare the loss on response (r) conditioned on questions with and without chat templates, i.e., $\frac{L(r|\hat{q})}{L(r|\bar{q})}$. And Eq.9 presents the same problem.

### Soundness
2

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
3

### Summary
The paper introduces Mix-CPT, a domain adaptation framework for LLMs that separates knowledge learning from format alignment to enhance efficiency and prevent knowledge loss. In the first stage, Mix-CPT trains on a mixture of domain-specific and general instructional data to balance domain learning and general capabilities, using Logit Swap Self-Distillation to retain prior knowledge. The second stage, format alignment, fine-tunes the model with a few selected samples to align responses to human-preferred formats. Extensive experiments and ablations have been carried out to support the proposed framework.

### Strengths
1. The paper introduces a novel view of CPT that separates domain knowledge learning from format alignment, potentially enhances both domain adaptation efficiency and task-specific performance.

2. The proposed LSSD method to avoid catastrophic forgetting is clear and straightforward, allowing the model to adapt without sacrificing previously learned information. 

3. Integrating domain-specific raw data with general instructional data during the pre-training phase is a strength, as it appears to balance both domain-specific and general knowledge effectively.

### Weaknesses
see questions.

### Questions
1. The results indicate a dependence on the quality of domain-specific data, with lower-quality data adversely impacting performance (Fig. 3). This reliance could restrict the framework’s applicability in real-world scenarios where high-quality datasets are limited.

2. Why is the logit of the top-1 token specifically swapped with the ground truth in LSSD? Beyond the math and average measurements in Fig. 6, could you provide a more detailed explanation of why this technique is effective?

3. Could you provide an analysis of computational efficiency compared to traditional domain adaptation frameworks?

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
The paper proposes a novel two-stage domain adaptation method for language models. The authors discuss the limitations of the existing training pipeline, which includes pre-training, instruction-tuning, and preference alignment. Based on this discussion, they propose a mix of data from each stage in an appropriate format to enable the model to learn both domain knowledge (i.e., pre-training) and knowledge utilization (i.e., instruction-tuning and preference learning). Additionally, the authors enhance the approach by training the model further on a small yet essential subset of data used in pre-training for improved model output alignment. The proposed method was evaluated with QWen2-7B and Llama-3-8B on domain data, including Wiki data, AutoMathText, and StarCoder. This method outperforms the simple CPT model.

### Strengths
- The proposed method is well-motivated
- The paper is well-written and easy to follow
- The proposed method outperforms the baselines

### Weaknesses
- Logit swap self-distillation requires the base LLM to perform inferences on the entire training dataset. This process can be quite expensive, especially with large datasets and models.
- A concern with the experimental setup is that Wikipedia data is used for domain adaptation. However, Wikipedia is more general-purpose than domain-specific, as it covers a wide range of topics. Additionally, the base models have likely already been trained on this widely used open-source dataset, which weakens the observations drawn from the experiments.
- The authors should have compared their method to a recent, similar CPT method [1]. Method [1] has an advantage as it trains a base model on both raw documents and formatted reading comprehension data. This approach is more straightforward than the authors’ method, as it requires only single-stage training and does not necessitate the separate collection of CPT, SFT, and DPO data.
- It is unclear why the model requires a second alignment stage, as it has already been trained on formatted DPO data. Is this due to issues with the initial stage of training?

[1] Adapting Large Language Models to Domains via Reading Comprehension, ICLR 2024

### Questions
- Refer to the weakness.

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
In this paper, the author addresses improving the efficiency of LLMs in adapting to new knowledge. Here, efficiency refers to adapting the LLM without relying solely on instruction-based tuning for the new domain; instead, the author employs a combination of domain-specific and general data. Additionally, alignment is achieved during the pretraining stage through a unified format and an additional training objective.

### Strengths
The paper is well-organized, and the problem it addresses is substantial. Improving the efficiency of knowledge distillation is indeed a relevant challenge that needs attention. Furthermore, the experimental results appear promising.

### Weaknesses
1. The paper talks about mixing CPT, SFT, and DPO data into a combined dataset for pretraining, but mixing instruction-tuning and pretraining data isn’t really new. Plus, there aren’t many details on how exactly these datasets are blended—like what proportions are used. The approach feels a bit empirical, but it would be great to have more specifics for reproducibility.

2. The author claims that the LLM learns new knowledge efficiently by understanding the task it’s meant to perform on the data, along with alignment during pretraining. I get the alignment part, but I’m a bit concerned that task-specific instruction data might limit the generalization of domain-specific data. I wonder if it’s always beneficial for new knowledge to align closely with existing world knowledge. While task performance in the new domain improves, it’s unclear if the model’s general ability is preserved—this wasn’t really explored in the experiments.

### Questions
See weakness. 

And I do have another question. What’s the fundamental difference between doing all of this during pretraining versus handling each step separately? Does this suggest that there’s no need to separate these steps and that it might be better to combine everything in a single pretraining process for new domain knowledge?

### Soundness
2

### Presentation
2

### Contribution
2
