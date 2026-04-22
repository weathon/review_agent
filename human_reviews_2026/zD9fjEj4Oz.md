# Prompt-MII: Meta-Learning Instruction Induction for LLMs

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose Prompt-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. Prompt-MII improves downstream model quality by 4-9 F1 points (10-20\% relative), matching ICL performance while requiring 3-13x fewer tokens.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper trains a language model to generate an instruction given a couple of example input-output pairs of a task via reinforcement learning.  The experiments show that this method achieves the performance of ICL with >20 examples, while consuming much fewer tokens for inference.

### Strengths
1. The idea is clearly conveyed.
2. I believe it is the first to use RL to train a general model for instruction induction.
3. The experiment is large-scale, and thus the resulting model might be useful to the community.

### Weaknesses
1. The technical contribution / novelty is limited. 
2. Missing comparison with a naive meta-prompt.

### Questions
1. How do you prevent data contamination? That is, how do you make sure that the training dataset does not contain any questions from the holdout test set? 
2. A small issue: Line 125, the hyperlink is wrong.

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
LMs has ability to generate better instruction to improve downstream performance. This work proposes to meta-learn this capability across variety of classification datasets, using RL.

### Strengths
- Strong motivation. Meta-learning instruction induction to generate optimized instruction for unseen task (instead of direct optimization on it) looks valid and promising direction.
- Experiments are generally well-executed, to test the effectiveness of the method.
- Experiment results support the strength of the method. Prompt-MII vastly outperforms naive instruction, and also is better than the untrained version(”Prompt-MII-Zero”). Prompt-MII matches the performance of ICL with 20-100 demonstrations.

### Weaknesses
Authors emphasize meta-learning as a key strength, where trained model can generate better instructions in unseen tasks. However, experiments on OOD generalization is not explicitly conducted. 90 held-out datasets, even though they are not seen during training, may be (or not be) in-distribution to the training set. Additional experiments on OOD generalization, with more explicit control on the domain of test set, will provide more insights on the authors’ proposed method.

### Questions
### Questions

1. Could authors discuss this work in relation with below prior works? These works seem to study meta-learning instruction generation. I am open that this work has unique contribution w.r.t. below works, but it will benefit the readers to know how this work differs.
    1. Choi ‘25, System Prompt Optimization with Meta-Learning
    2. Ha ‘23, Meta-Learning of Prompt Generation for Lightweight Prompt Engineering
    on Language-Model-as-a-Service, Findings of EMNLP
    3. Fernando ‘23, Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution (I see this in bibliography, but cannot find in the main text or appendix)
2. (L250-L253) Could authors elaborate more on why “classification tasks may be challenging for iterative refinement algorithms”, compared to "generative tasks like QA or summarization”? Further explanation will help readers to understand the unique benefit of meta-learning instruction induction over existing non-meta-learning algorithms
3. I have confusion about the naming of Prompt-MII-Zero. It seems to be the “untrained” version of Prompt-MII, which means that simply the LM is prompted to generate an instruction given meta-prompt. Prior works [1, 2] have also shown efficacy of generating instruction using LM, which seems to be largely identical to “Prompt-MII-Zero”. I believe more straightforward naming, or at least more explicit explanation can reduce the confusion of the readers, especially those who are skimming through.
4. (Minor details) L248 only mentions Qwen, but shouldn’t it also include Llama? Or does that explanation only holds for Qwen for some reason? If so, authors should elaborate on that.

[1] Honovich ‘22, Instruction Induction: From Few Examples to Natural Language Task Descriptions
[2] Zhou ‘22, Large Language Models Are Human-Level Prompt Engineers

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of high inference cost in in-context learning, where long context sequences are often required to achieve strong performance on new tasks. To mitigate this issue, the authors propose compressing training demonstrations into a more compact yet informative prompt. They introduce PROMPT-MII, an RL-based framework for meta-learning an instruction induction model capable of generating such prompts. Experimental results on several classification benchmarks demonstrate the effectiveness of the proposed approach.

### Strengths
1. The problem of reducing context length in in-context learning is both timely and important, given the growing reliance on in-context learning for various downstream tasks.

2. The method is evaluated on multiple classification tasks, and the results indicate clear improvements over baselines.

### Weaknesses
1. The experiments are limited to classification tasks, which makes it difficult to assess the generalizability of the approach. Additional evaluation on generative tasks, such as question answering, would strengthen the contribution.

2. It is unclear how robust the meta-prompt design is across different tasks. The example prompts appear to be tailored to specific classification tasks, raising concerns about adaptability and generality.

### Questions
1. How does the proposed method perform on more complex tasks, such as open-ended question answering or reasoning-based generation tasks?

2. When applying the approach to a new domain or task, how should the meta-prompt template be constructed or adapted? More discussion or guidance on this would be helpful.

### Soundness
3

### Presentation
3

### Contribution
2
