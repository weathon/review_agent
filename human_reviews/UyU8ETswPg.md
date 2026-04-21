# Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Recent studies have identified one aggravating factor of LLM hallucinations as the knowledge inconsistency between pre-training and fine-tuning, where unfamiliar fine-tuning data mislead the LLM to fabricate plausible but wrong outputs. In this paper, we propose a novel fine-tuning strategy called Prereq-Tune to address this knowledge inconsistency and reduce hallucinations. Fundamentally, Prereq-Tune disentangles the learning of skills and knowledge, so the model learns only the task skills without being impacted by the knowledge inconsistency. To achieve this, Prereq-Tune introduces an additional prerequisite learning stage to learn the necessary knowledge for SFT, allowing subsequent SFT to focus only on task skills. Prereq-Tune can also be combined with fictitious synthetic data to enhance the grounding of LLM outputs to their internal knowledge. Experiments show that Prereq-Tune outperforms existing baselines in improving LLM's factuality across short QA and long-form generation tasks. It also opens new possibilities for knowledge-controlled generation in LLMs. Our code is available at https://github.com/UCSB-NLP-Chang/Prereq_tune.git.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes to use synthetic data to improve LLMs' factuality. Specifically, a "knowledge"-LoRA module is trained on some basic fictitious facts, and then a "skill"-LoRA module is trained to answer questions based on these facts. During inference, the knowledge-LoRA is dropped, and only the skill-LoRA is kept. Experiments show that the proposed approach can improve LLM's factuality across short-form QA and long-form generation tasks by a large margin.

### Strengths
- The proposed method is simple and effective, outperforming SFT and other baselines for factuality-oriented fine-tuning.
- Using different LoRA modules to better train the skill LoRA and flexibly remove the fictitious knowledge is very interesting and practical.
- The ablations and analysis show that the skill-LoRA indeed learns to ground the response to the knowledge in knowledge-LoRA as desired.
- The overall writing is of good quality and easy to follow.

### Weaknesses
- The novelty of the proposed approach is a bit limited. Previous methods also perform SFT in ways that are more aligned with the LLM's knowledge (e.g. the POPULAR method as cited), and the main novelty of the proposed method is to use synthetic data to ensure a higher level of groundedness. Given that the evaluated tasks are also simple ones involved with generating factual statements, it is not surprising that synthetic data could work well.

- The generality of the method for more complex skills remains elusive. For example, how to correctly perform SFT on more challenging reasoning problems, where the LLM may not have enough reasoning skills to produce the responses?

### Questions
- How will the model perform if you simply fine-tune it with all fictitious data (D_know and D_T) without dropping the fictitious knowledge (in the basic version, not multi-version)? This seems to be a simple baseline that is good to see.

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
3

### Summary
The author believes that the issue of hallucination in language models largely arises from the discrepancies between the datasets used in the pre-training and fine-tuning stages. To mitigate this problem, this paper proposes a simple and effective fine-tuning method that first provides the necessary knowledge required for the subsequent stage, and then focuses on enhancing the model's "skills."
Rich experiments focused on biography generation and doctor-patient question-answering tasks validate the effectiveness of this method in alleviating hallucinations.

### Strengths
1. The issue of model hallucination is indeed a challenging problem that needs to be addressed.  
2. Although the method is simple, it has proven to be highly effective from an experimental standpoint, with well-designed experiments supporting the arguments made in the paper and validating the necessity of each component.  
3. The writing is fluid, and the use of figures and tables is appropriate, facilitating ease of reading.

### Weaknesses
1. The paper does not address the generalization ability of the models trained using this method.  
2. When there are many "skills" to be learned, data synthesis may become quite costly.

### Questions
1. If enhancing the model with new "skills" requires re-executing data synthesis and knowledge enhancement, it may be excessively costly. Furthermore, when learning new "skills," how should the LoRA parameters for previous "skills" be managed?
2. The method mentioned in the paper appears to be quite effective, but based on the experimental results, it has not completely resolved the hallucination problem. What does the author consider to be the possible reasons for this? Additionally, what other factors might contribute to the occurrence of hallucinations in models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed a new fine-tuning strategy called PREREQ-TUNE. The goal is to improve LLM's factuality through fine-tuning while minimizing the likelihood of introducing extra hallucination due to the knowledge inconsistency between pre-training and finetuning stages.
The basic idea is to disentangle models' learning of skills and knowledge by first introducing an additional prerequisite learning stage to focus on the knowledge that will be used in the subsequent SFT stage, so that the second SFT stage can focus only on learning the skills.
Experiments show that PREREQ-TUNE outperforms existing baselines in improving LLM's factuality across short QA and long-form generation tasks.

### Strengths
1. The paper proposed a great idea to let an LLM enhance its grounding skills, meanwhile avoid introducing more hallucination due to the knowledge that the model may learn in the fine-tuning stage.

2. The experiments very nicely show that the idea actually work on the test datasets, in particular, section 4.3 clearly shows that the skill LoRA can not answer the questions that require knowledge that the corresponding knowledge lora is not there. And I like the finding that the knowledge LoRA can serve as plug-and-play "knowledge flash drive".

3. The method can even scale nicely due to the fact that it is tuned by synthetically generated data.

### Weaknesses
The method may be limited to simple facts. When the question is more complicated, for example requiring more than one fact or even reasoning to answer, the reasoning process itself (skill LORA)  may generate new facts that can introduce hallucination. In this case, the proposed method in this paper can be limited.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PREREQ-TUNE, a novel fine-tuning strategy to reduce hallucinations in large language models (LLMs). The key innovation is disentangling skill learning from knowledge acquisition by introducing a prerequisite learning stage before supervised fine-tuning. In this stage, a separate "knowledge LoRA" learns necessary background information, allowing the subsequent "skill LoRA" to focus purely on task-specific abilities without being misled by knowledge inconsistencies. PREREQ-TUNE can leverage fictitious synthetic data to further enhance output groundedness. Experiments on biography generation, medical QA, and short QA tasks show PREREQ-TUNE outperforms existing methods in reducing hallucinations. Additional analyses demonstrate its effectiveness in knowledge grounding, resistance to knowledge pollution, and ability to scale with synthetic data. The authors also explore extensions for answer abstention and expressing uncertainty. Overall, PREREQ-TUNE offers a promising approach to improve LLM factuality and opens possibilities for modular LLM designs.

### Strengths
1. Overall, I think the idea to learn a skill LORA that is capable of grounding is novel.
2. The method is easy to follow with concrete running examples.

### Weaknesses
1. Although the overall method is interesting, the motivation is confusing to me. I think it is hard to argue that “if the tuning stage involves training examples that require knowledge that an LLM has not seen during pre-training, then the LLM would be misled to fabricate plausible but wrong answers to unfamiliar questions”. In a lot of previous literature, people use finetuning to adapt pretrained LMs to tasks [1][2], where I believe that LMs learn some knowledge during training. 
2. The citations in line 44 are not supportive. [3] mentions “A model’s knowledge and capabilities are learnt almost entirely during pretraining” in section 2 (The capabilities here look similar to the skills in the paper). [4] mentions that “pre-training is the main source of an LM’s capabilities”. In general, I don’t find it convincing that the tuning stage don’t involve knowledge learning.

[1]. Diao, S., Xu, R., Su, H., Jiang, Y., Song, Y., & Zhang, T. (2021, August). Taming pre-trained language models with n-gram representations for low-resource domain adaptation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 3336-3349).
[2]. Zheng, J., Hong, H., Wang, X., Su, J., Liang, Y., & Wu, S. (2024). Fine-tuning Large Language Models for Domain-specific Machine Translation. arXiv preprint arXiv:2402.15061.
[3]. Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., ... & Levy, O. (2024). Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36.

### Questions
Could you show more detailed model architecture about how skill LoRA and knowledge LoRA interact with each other?

### Soundness
3

### Presentation
3

### Contribution
2
