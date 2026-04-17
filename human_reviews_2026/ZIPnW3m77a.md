# Diverse Text Generation through Soft Prompt Tuning

- Decision: Reject
- Scores: 4, 4, 4, 4, 4

## Abstract
Diverse text generation is crucial for effective exploration in language models. Current sampling-based decoding methods struggle to balance quality and diversity and lack control over generating mutually distinct outputs. Reinforcement learning approaches maintain quality, but require extensive training and are difficult to transfer across domains due to task-specific reward functions.
We propose a lightweight framework that learns diversely initialized continuous soft prompt vectors, which, when prepended to input prompts, guide the model's final-token hidden states into distinct representation regions. This enables diverse generations from identical inputs, as initial hidden state differences amplify through the autoregressive mechanism, creating increasingly divergent generations. By preserving earlier hidden state similarities, our method maintains contextual consistency to task-specific constraints.
Experiments across combinatorial tasks, question generation, and molecular design reveal that our soft prompt tuning method improves diversity while consistently adhering to task-specific constraints. Our approach shows particular strength in complex settings with large exploration spaces, as demonstrated through our novel contribution of a challenging combinatorial dataset specifically designed to evaluate diverse generation capabilities of language models. This lightweight framework provides a unified, broadly applicable solution for diverse text generation across various application domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a diverse generation technique taking advantage of soft prompt vectors. It breaks down an input into context prompt and generation prompt. Soft prompts are inserted between these two parts to encourage the model to produce diverse outputs. Tuning objective consists hidden state diversity and coherence, calculated as L2 distances.

Experiments on combinatorial generation, question generation, and molecular generation indicate that their tuned generation method achieves better trade-off between quality and diversity than do classic sampling methods.

### Strengths
- The proposed framework is parameter-efficient and easy to train.
- The training objective is intuitive, as it promotes diversity and balances coherence by tracking the hidden states.
- Experiment results show Pareto improvement.

### Weaknesses
- The baselines are weak. The paper compares against naive sampling methods such as temperature and nucleus sampling. No advanced methods, like generating based on diverse natural language prefixes, or any method mentioned from the related work section, are included as baselines.
- It is unclear what data is used to tune the soft prompts for each experiment.
- The paper criticizes RL methods for not transferring, but the tuned soft prompts are also task-specific, as admitted by the authors.
- Lester et al. [1] demonstrate the zero-shot transfer capability of soft prompts, but that is not demonstrated here. This suggests these diversity-targeted soft prompts may not be generalizable.
- More analysis on the soft prompt vectors themselves beyond just the L2 distances would be helpful for understanding. For example, what are the directions of these vectors? what does inverting a soft prompt vector produce?

[1] The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)

### Questions
- Is “diversity as the distance of hidden states” a validated proxy for semantic diversity? The introduced method relies entirely on this assumption, but it is not well-justified. 
- Why was L2 distance chosen for the diversity loss? The L2 norm is not agnostic to the magnitude of the hidden state vectors. Why was this used instead of a standard magnitude-invariant metric like cosine similarity?
- Why is SGD used for optimization? Is an adaptive optimizer like Adam considered, which might simply be better?
-  Can a set of general-purpose diversity prompts be learned?

### Soundness
2

### Presentation
2

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
This paper proposes a new method to make language models produce a wider variety of high-quality responses from a single input prompt. Via an informed initialization scheme, followed by a fine-tuning stage, a set of soft prompts are trained to yield diverse hidden states from one another. This is achieved by maximizing the distance in the  later hidden states of the sequence, whilst keeping the distance in earlier hidden states close.

The method was tested on three distinct tasks: a novel combinatorial "target-sum" task, question generation, and molecular design. The authors demonstrate good results across all tasks, with their approach providing a preferred quality-diversity trade-off. In particular, the approach showed particular strength in complex tasks with large exploration spaces.

### Strengths
- The simplicity, and therefore the applicability, of the method is strong and should make for easy implementation. 
- The results (although not always easily digested) are promising.
- For the most part the paper is well written, and it paints a clear story. (See Concern 5 for suggested improvements)

### Weaknesses
**Concern 1:**

The paper currently does not consider sequentially generating diverse output with a single model. If the goal is to generate a set of diverse outputs to a given prompt, it would be possible to have an LLM see its previous outputs in its context, rather than the parallel approach provided by this paper. This would be extra appealing considering the large context windows of contemporary LLMs, and that this approach would require no fine-tuning or soft prompts.

This sequential approach of course has potential downsides, and I’m not sure it would yield the same diversity. But since a strong part of this paper is about the method being lightweight, it would significantly strengthen the paper if you could demonstrate how your more complicated method outperforms the trivial sequential one.

**Concern 2:**

In line with concern 1, it would greatly benefit the paper if you could demonstrate the usefulness of this approach in scenarios where diversity is crucial, and not only desired. I would suggest perhaps trying this approach in an RL setting with LLMs. There has been some recent work showing how the LLM entropy has an undesirable collapse during RL training, which negatively impacts exploration.
https://arxiv.org/abs/2505.22617


Such a scenario would also arguably be harder to solve using the straight-forward sequential approach in concern 1. And might therefore be a strong point for your paper (given that Concern 3 is not overwhelming)


**Concern 3:**

There is no explicit information regarding the computational overhead for this method. Since the authors argue for their method being “lightweight”, providing empirical evidence corroborating this claim would strengthen the paper.  In particular, this would be interesting to see in regards to concern 1.

**Concern 4:**

Figure 3 is quite messy at the moment. Perhaps moving the Beam-Search result into a separate figure? 

Also, what are the crown icons for?!?! It does not provide any value in my opinion, and only makes an already busy figure more cluttered.

**Concern 5: (minor)**

Some parts of the paper could benefit from having an extra iteration of the text.

Line 32: Language Models (LLM), should be either Large Language Models (LLM), or change the abbreviation to (LM).

Line 155: You start by explaining that the intuition is two-fold, but what follows immediately is more concerned with “what” you have done. Not why you do it.

Line 291-296: This paragraph is a bit unclear for a first time reader. It introduces the task and scenarios, but it’s not entirely clear what these scenarios are. I suggest adding a figure that displays one (or several) qualitative examples of the scenarios you’re considering.

### Questions
Considering that this approach requires one to train a set of prompts for each specific task. Do you think it would be possible to generate a set of "general" diversity prompts. Meaning that they could be used for a wide range of tasks.

Ideally, one could imagine that the task of these prompts would be to condition the model in some sense, so that prompt A yields a distinct output to prompt B etc... Whilst the quality remain intact. (as your approach does now)

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
3

### Summary
The paper introduces a lightweight, task-agnostic framework for diverse text generation in LLMs via soft prompt tuning. It optimizes multiple diversely initialized continuous soft prompts (using scrambled Sobol sequences), inserted between context and generation prompts, to maximize differences in final-token hidden states for diversity while minimizing early-state deviations for task coherence. This induces controlled shifts in output distributions without fine-tuning or domain-specific rewards. The method is tested on a novel combinatorial dataset (50 scenarios with item lists summing to targets), question generation (SQuAD splits), and molecular design (forward synthesis on FS-Mol, description-guided on Desc-Mol). Results show consistent Pareto improvements over baselines (temperature/top-p sampling, diverse beam search, GPT-4o-mini). Overall, it enables effective exploration in large solution spaces while adhering to constraints.

Strength:
The proposed method is sound and the empirical results are promising.

Weakness:
The novelty of the paper is limited.

### Strengths
The proposed method is sound and the empirical results are promising.

### Weaknesses
The novelty of the paper is limited. There are extensive previous work that contributed to this topic [1, 2, 3, ...]. There are not too much technical differences from the previous work. Therefore, I believe the technical contribution of this paper does not qualify for a top-tier venue like ICLR.

[1] The Power of Scale for Parameter-Efficient Prompt Tuning. Brian Lester, Rami Al-Rfou, Noah Constant. EMNLP 2021 \
[2] Selective Prompting Tuning for Personalized Conversations with LLMs. Qiushi Huang, Xubo Liu, Tom Ko, Bo Wu, Wenwu Wang, Yu Zhang, Lilian Tang. ACL 2024 findings \
[3] Controlled Text Generation using T5 based Encoder-Decoder Soft Prompt Tuning and Analysis of the Utility of Generated Text in AI. Damith Chamalke Senadeera, Julia Ive. CoRR 2022

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper focuses on enhancing the diversity of text generation in language models. The proposed method introduces a set of learnable prepended embeddings designed to promote diversity by maximizing the differences in the final hidden states while maintaining similarity in the earlier hidden representations. The approach is evaluated across several tasks, including combinatorial reasoning, question answering, and text generation, demonstrating modest but consistent improvements in output diversity.

### Strengths
The paper introduces a simple and intuitive method for improving text generation diversity in language models. Its originality lies in proposing a task-agnostic approach based on learnable prepended embeddings that promote diversity without requiring model modification or task-specific tuning. This contrasts with reinforcement learning (RL)-based methods, which often rely on carefully designed objective functions, making the proposed method broadly applicable and easy to integrate with existing models. The paper is clearly written, well organized, and supported by experimental results.

### Weaknesses
The paper’s technical novelty is somewhat limited. While the idea of learning prepended soft prompt embeddings is intuitive and practical, it lacks clear theoretical motivation. The link between diversity in the final hidden representations and diversity in generated text is not well established, especially given that differences in embedding space may not translate directly to token-level diversity due to the SoftMax approximation.

Additionally, the mechanism by which diverse prepended prompts lead to more diverse text outputs remains unclear. A deeper theoretical explanation or empirical analysis, such as examining intermediate representations or quantifying output diversity, would strengthen the paper’s foundation and make the proposed method more convincing.

### Questions
In Algorithm 1, the objective function focuses on maximizing the similarity between the embeddings of the last m hidden states of the soft prompts and those of the generation prompt. Could the authors clarify whether this objective is sufficient to ensure diversity among the soft prompt embeddings themselves?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework for diverse text generation based on "soft prompts." The core technique involves inserting trainable embeddings into the hidden states of input tokens to guide the language model. This approach aims to enhance the diversity of generated outputs while preserving their quality. The method's effectiveness was demonstrated on three distinct tasks: combinatorial optimization, question generation, and molecule generation. Experimental results show that the soft-prompt framework outperforms standard decoding baselines, such as temperature sampling and nucleus sampling, in achieving a better balance between output diversity and quality.

### Strengths
1. Novel Method: Introduces a lightweight and effective method that uses inference-time optimization of soft prompts to explicitly balance generation diversity and quality.

2. Strong Empirical Results: Demonstrates superior performance and generalizability by outperforming strong baselines on the quality-diversity trade-off across three highly distinct tasks.

3. Valuable Contributions: Contributes a new benchmark dataset and provides insightful analysis that deepens the community's understanding of controlled diverse generation.

### Weaknesses
1. Lack of Ablation Studies: The paper does not sufficiently justify key design choices. A thorough ablation study is needed to validate the contribution of individual components, such as the d_ctrl and w_c terms in the loss function, and to compare the chosen soft-prompt insertion position against alternatives (e.g., prepending to the entire input).

2. Limited Generalizability: The approach appears to require training new, task-specific soft prompts for each distinct task. This limits its generalizability and raises questions about its utility as a universal diversification framework.

3. Unclear Computational Overhead: Given the need for per-task optimization (Algorithm 1, lines 4-20), the paper fails to report the actual wall-clock time required. This omission makes it difficult to assess the practical time cost and evaluate the claim that the framework is "lightweight," especially when adapting to new tasks.

4. Imprecise Analysis of Results: The analysis of the quality-diversity trade-off could be more rigorous. Vague claims like the one on lines 367-368 ("3-5% higher UniEval scores...") should be clarified. The core contribution should be framed as a Pareto improvement—achieving higher diversity for a given level of quality, or vice versa—rather than suggesting a simultaneous improvement in all cases.

### Questions
As mentioned in the weakness section.

### Soundness
2

### Presentation
3

### Contribution
3
