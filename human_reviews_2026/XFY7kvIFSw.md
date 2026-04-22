# MediX-R1:  Open Ended Medical Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
We introduce MediX-R1, an open-ended reinforcement learning (RL) framework for medical multimodal large language models (MLLMs) that enables clinically grounded, free-form answers beyond multiple-choice formats. MediX-R1 fine-tunes a baseline vision–language backbone with Group Relative Policy Optimization (GRPO) and a composite reward tailored for medical reasoning: an LLM-based accuracy reward that judges semantic correctness with a strict YES/NO decision, a medical embedding–based semantic reward to capture paraphrases and terminology variants, and lightweight format and modality rewards that enforce interpretable reasoning and modality recognition. This multi-signal design provides stable, informative feedback for open-ended outputs where traditional verifiable or MCQ-only rewards fall short. To measure progress, we propose a unified evaluation framework for both text-only and image+text tasks that uses an LLM-as-judge in place of brittle string-overlap metrics, capturing semantic correctness, reasoning, and contextual alignment. Despite using only $\sim 50$K instruction examples, MediX-R1 achieves excellent results across standard medical LLM and VLM benchmarks, outperforming strong open-source baselines and delivering particularly large gains on open-ended clinical tasks (e.g., radiology summarization and report generation). Our results demonstrate that open-ended RL with comprehensive reward signals and LLM-based evaluation is a practical path toward reliable medical reasoning in multimodal models. Our trained models, curated datasets and source code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors present MediX-R1, a multimodal RLVR framework for clinical applications, which uses GRPO with four complementary rewards: (i) an LLM-as-a-judge accuracy check, (ii) an embedding-based semantic similarity reward, (iii) a format reward for special tokens, and (iv) a modality recognition reward tfor cross-modality hallucination. According to authors claim, trained on ~50K medical instructions, MediX-R1 achieves better results across both text-only and image+text medical tasks under proposed three-stage LLM-as-judge evaluation pipeline. Authors also investigate reward-hacking failure modes and present human qualitative analyses as part of their ablation studies.

### Strengths
- Authors focus on a timely topic on RLVR-based medical concepts, especially given the recent rise of RLVR approaches in text-only reasoning tasks.
- Authors provide a structured reward design consisting of four complementary components, which can be useful for future research in medical RLVR and multimodal-RL.

### Weaknesses
- Proposed method appears to be a straightforward application of existing GRPO algorithm with multimodal medical data, without new algorithmic improvements. Some popular open-source RL frameworks like Verl and Easy-R1 already support multimodal training and the core work seems to be primarily reward choices rather than theoretical contributions. Moreover, eventhough authors claim to provide an open-source framework as contribution, there is no code provided during submission which makes it impossible for reviewers to verify the implementation or validate this stated contribution.
- One of the major weakness of the paper is too much reliability on LLM-as-a-Judges both in reward design and evaluation. Using LLM judges requires a detailed reliability analysis such as multiple runs, human correlations, and testing with different LLMs, in which authors conduct none of these; furthermore, the LLM judge used is a small Qwen3-4B in verifier, which further reduces reliability. On the other hand, authors evaluate their framework with an Qwen-14B model only, which again lacks the aforementioned issues. Although the authors claim to have qualitatively examined outputs with human experts, the provided human evaluation is very narrow and weak; it should demonstrate human correlation with a sufficient number of expert annotators with more details like their experience level and the professional focuses (e.g., X% portion radiology). Thus, their LLM-as-judge evaluation lacks reliability for considering the results credible. 
- Provided ablation studies and comparisons are not comprehensive enough: authors does not compare with other RL methods (PPO, DAPO, REINFORCE++, etc.) to justify the choice of GRPO and there are no experiments with different base models (e.g., Llama) or different model scales (smaller/larger Qwen models) to demonstrate generalization.
- Finally, authors mention they use ~50K training data to RL-finetune MediX-R1 as a contribution in the abstract and at the end of the introduction, but the details of data construction and training dynamics are not discussed.

### Questions
1. Could authors share the human experiment details like the number of expert, their backgrounds (radiologists, generalists?), inter-annotator agreement, and annotation protocol? 
2. How is the training data constructed and what training dynamics are followed (i.e., training setting details)?

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
4

### Summary
The paper presents MediX-R1, a framework for open-ended reinforcement learning (RL) in medical multimodal language models. The key idea is to train a vision-language model (Qwen2.5-VL 7B) using Group Relative Policy Optimization (GRPO) and a composite reward tailored for clinical reasoning. A three-stage LLM-as-judge evaluation pipeline assesses both text-only and multimodal outputs. The proposed MediX-R1 achieves improvements across 18 medical benchmarks, outperforming strong baselines such as BiMediX2, MedGemma, and HuatuoGPT-V.

### Strengths
- **Novel reward design**: The multi-signal reward (LLM + embeddings + modality + format) is well-motivated and shown to stabilize open-ended RL.
- **Unified evaluation**: The proposed LLM-as-judge framework provides a consistent and semantically aware assessment across text and image tasks.
- **Broad modality coverage**: Supports diverse imaging types (CT, X-ray, MRI, microscopy, ultrasound, etc.), improving the model’s clinical versatility.

### Weaknesses
- **High dependence on pretrained Qwen2.5-VL**: The improvements might reflect reward tuning rather than genuine reasoning advancement; no results are shown on alternative backbones.
- **Limited transparency in human evaluation**: The human evaluation section lacks important details—there is no information about the number of annotators, their medical background, or the inter-rater agreement between them. Additional information would increase the reliability of the results.
- **Limited novelty in RL design**: The framework primarily combines existing techniques without clear new algorithmic steps, although one of the contributions is "introducing open-ended medical reinforcement learning".

### Questions
1. What is the total compute budget and training time for MediX-R1 compared to multi-stage baselines like BiMediX2 or MedGemma?
2. How well does MediX-R1 generalize to unseen modalities or real-world hospital data not represented in training sets?
3. Have the authors tested or considered evaluating MediX-R1 on alternative backbones (e.g., LLaVA-Med) to demonstrate broader generalization?

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
This paper presents MediX-R1, an open-ended reinforcement learning framework for medical multimodal reasoning. Its key distinction from previous work lies in its richer, composite reward model, which integrates correctness, semantic similarity, format compliance, and modality grounding. Compared with earlier RL-based medical VLMs, this makes the reward signal more fine-grained and clinically aligned. While some may question the reliability of using an LLM-as-judge for reward evaluation, I find this design both reasonable and effective — it provides a far more flexible and informative signal than rigid string matching in medical question-answering setups. Overall, this is a paper I genuinely enjoyed reading it.

### Strengths
- The paper is very well written and clearly structured, making complex reinforcement-learning ideas accessible and easy to follow.

- It presents a thoughtful and well-engineered reward model, combining multiple complementary signals (correctness, semantics, modality, and format) in a coherent way.

- The one-stage training design keeps the framework simple yet effective — a welcome contrast to multi-phase or agentic pipelines that often add unnecessary complexity.

### Weaknesses
- Limited technical innovation. The work primarily extends existing GRPO-style reinforcement learning with a richer reward structure rather than proposing a new algorithm. While this may be acceptable given the paper’s focus on design and empirical validation, the technical novelty is relatively modest.

- Reproducibility and openness. The paper promises code and model release but does not yet provide them. Given the number of moving parts in the composite reward setup, open-sourcing both code and model weights would be critical for reproducibility.

- Fixed reward weighting. The choice of static coefficients  appears somewhat arbitrary. These hand-tuned weights may strongly influence outcomes. It would strengthen the work to include ablation or sensitivity analyses on these weights, or to explore making them learnable parameters through meta-optimization or adaptive weighting.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MediX-R1, an open-ended reinforcement learning (RL) framework that fine-tunes a medical multimodal model using Group Relative Policy Optimization (GRPO) and a novel composite reward system. This framework employs a multi-signal reward, including an LLM-as-judge accuracy signal and medical embedding-based semantic alignment, along with a unified LLM-based evaluation pipeline, to enable the generation of clinically grounded, free-form answers with interpretable reasoning traces.

### Strengths
1. This paper is well-written, logically clear, and easy to follow.
2. The introduced LLM-as-judge evaluation framework effectively addresses, to some extent, the limitations of traditional string-overlap metrics.
3. Extensive experiments demonstrate that the proposed MediX-R1 achieves impressive performance across multiple benchmarks and various tasks.

### Weaknesses
1. The source of the $\sim 50$K instruction data used in this paper is missing, which is very important.
2. The composition of the reward weights is very complex, with values including $0.5175$, $0.3375$, and $0.045$. How were these highly precise numerical values derived? Will these weights need to be adjusted as the training data changes?
3. The embedding-based semantic reward uses $0.8$ as the default threshold. Why introduce a threshold to set it as a binary reward instead of directly using the cosine similarity as the reward, which seems to be more precise and can reflect differentiation?
4. Considering that the reasoning process is not supervised during training, how is the reasonableness of the reasoning process inside $\langle\text{think}\rangle\cdots\langle/\text{think}\rangle$ ensured? Is it possible for the chain-of-thought to be flawed or chaotic while the final answer is still correct?
5. It is suggested to mark the best and second-best results for each benchmark in Table 2 (e.g., using bold or underline) to more intuitively reflect the performance comparison.
6. It is recommended to supplement some generated cases from the long-form report generation tasks.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
