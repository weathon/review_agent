# CodePlan: Unlocking Reasoning Potential in Large Language Models by Scaling Code-form Planning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 3, 5, 6, 3, 8

## Abstract
Despite the remarkable success of large language models (LLMs) on traditional natural language processing tasks, their planning ability remains a critical bottleneck in tackling complex multi-step reasoning tasks. Existing approaches mainly rely on prompting or task-specific fine-tuning, often suffering from weak robustness and cross-task generalization. To address the limitation, we introduce CodePlan, a scalable paradigm that empowers LLMs to generate and follow code-form plans---pseudocode that outlines high-level, structured reasoning processes. By leveraging the structured and versatile nature of code, CodePlan effectively captures the rich semantics and control flows inherent to sophisticated reasoning. Importantly, CodePlan allows the automatic extraction of code-form plans from massive, wide-ranging text corpora without the need for curated, task-specific datasets. This enables it to scale up efficiently and improve reasoning capabilities across diverse scenarios. To train CodePlan, we construct a large-scale dataset of 2M examples that integrate code-form plans with standard prompt-response pairs from existing corpora. With minimal computation overhead during both training and inference, CodePlan achieves a 25.1\% relative improvement compared with directly generating responses, averaged across 13 challenging multi-step reasoning benchmarks, spanning mathematical reasoning, symbolic reasoning, instruction-following, multi-hop QA, and decision-making tasks. Further analysis reveals CodePlan's increasing performance gains on more complex reasoning tasks, as well as significant data efficiency thanks to its generalization ability.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces CODEPLAN, which formulates planning for complex problems in the form of code. To enable the model to acquire such ability, the authors have existing LLMs to generate this code-style planning for existing samples as training data. The authors have conducted experiments on various reasoning tasks and have achieved improvements.

### Strengths
- The paper demonstrates performance improvements over baselines across different datasets.
- The authors have validated the effectiveness of using code as a form of planning.
- The authors have validated the effectiveness of using existing LLMs to generate code-style planning.

### Weaknesses
1. The font size in Table 1 is too small.
2. Overall, I believe this paper would be more suitable as a technical report or a short article. The idea of using code form for planning is a useful trick, but it lacks enough innovation:
 - It only addresses the issue of the **form** of planning, without fundamentally enhancing the model's planning and reasoning capabilities
 - A widely held assumption in the community is that code is important for LLMs to learn complex reasoning and long-text abilities. There is already an amount of work on this topic. Therefore, the techniques and conclusions of this paper are not surprising. 
 - Under code-style planning, the core issue is how to generate effective code-style planning for a query from a specific domain or a general domain. However, this paper does not actually address this core issue. The method of generating through prompting shown in Table 2 is overly simplistic.
3. This method is constrained by the effectiveness of the LLM which generates code plannings for training. A common approach is to improve the quality of the training data through code execution results or log information. I suggest the author to consider this optimization method.
4. Since the paper posits that code-form plans are superior to other forms (such as text, first-order logic, graph, etc.), the authors should provide direct comparative experiments.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes incorporating pseudocode-based reasoning into the model's response process, which can effectively improve model performance. To obtain these pseudocode reasoning processes, the authors suggest using a large model to expand the responses directly on the WebInstruct dataset. Through extensive experiments, the authors demonstrate that training the model on the expanded dataset significantly enhances its performance.

### Strengths
1. The experiments are extensive.  
2. A mathematical understanding of the underlying principles was developed, making the entire framework more solid.

### Weaknesses
1.  If the goal is to prove the value of the code-form plan, I think there are two aspects to consider:  
1.1 If the goal is to demonstrate the benefit of this reasoning format to the model, then it would be appropriate to compare it with Chain-of-Thought (CoT) reasoning. Showing that a model performs better when using its code-based planning compared to traditional CoT results would more directly establish the superiority of the CodePlan format.    
1.2 If the aim is to showcase the advantages of the code-form plan format and its ability to be automatically generated at scale with strong results, then the baseline of natural language enhancement should be more rigorously established. However, in the paper, the adaptation of natural language reasoning prompt (in Table 7) seems only slightly modified from the code-plan one, making the comparison less convincing.  
[More comments on this prompt issue, natural language tends to be more redundant than code. Imposing a requirement like “The logic should be high-level, instead of replicating low-level details in the response” inherently makes it more challenging for the model. Additionally, as shown in the results of Table 4, modeling \( \log P(Z|X) \) is challenging, which partly suggests that the difficulty lies in generating this logic itself. This doesn't necessarily indicate that CodePlan is more effective; rather, it suggests that generating code-based plans may be easier because the task is easier. This easiness could be due to the prompts being optimized for generating code rather than for producing natural language explanations.]


2. The baseline results are different from Results in MAmmoTH2 (https://arxiv.org/pdf/2405.03548) . In Table 2 from MAmmoTH2, Mistral with WebInstruct, achieves 68.4/36.7 in GSM8k\MATH. While in this paper, Table 3 shows 54.1/31.5, which are much lower than reported.


3. Since the data generation process used LLaMA3 8B, the Plan-and-Solve (PS) Prompting baseline should also use LLaMA3 8B to generate the natural language plan prompt.

### Questions
1. The paper used LLaMA3 8B for generation—is this a type of distillation? It would be necessary to report LLaMA3 8B's performance on the evaluation datasets directly. Additionally, given that the size of LLaMA3 8B is quite similar to the model discussed in the paper, why not simply use LLaMA3 8B directly?

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
4

### Summary
This paper proposes an approach to improve LLM performance on varied reasoning tasks including mathematical and symbolic reasoning and multi-hop QA. The approach fine-tunes an LLM to predict Python pseudo-code "plans" from a natural language query, and to then condition on this pseudo-code to generate the answer to the query (the code is not executed). To obtain the pseudo-code plans for training, an LLM is prompted using queries and ground-truth solutions on 2M query/response pairs from WebInstruct. The resulting query, pseudo-code, solution tuples are then used to fine-tune the LLM. The paper evaluates on three different base LMs, and a variety of datasets including GSM8K, MATH, SVAMP, and HotpotQA, finding substantial improvements over direct prediction of answers and an variant of the method that uses natural language plans rather than pseudo-code.

### Strengths
S1) The paper's experimentation was overall thorough:
- It applied its training procedure to three different open source models, and evaluated them across a range of datasets covering various types of reasoning where code might potentially help.
- The analysis experiments, in particular Finding 3 that used natural language plans, were a helpful step toward understanding the approach (but see below for a few questions/points on this).

S2) The results seemed strong, finding that the approach obtains large improvements over vanilla (non-planning) training and using natural language plans. 

S3) I found the paper well-motivated, given the findings of past papers that training on code may improve reasoning abilities [1,2,3], and that models can be trained to reason (in natural language) using approaches like STaR.

[1] The Magic of IF: Investigating Causal Reasoning Abilities in Large Language Models of Code. Liu et al. 2023
[2] To Code, or Not To Code? Exploring Impact of Code in Pre-training. Aryabumi et al. 2024. 
[3] If LLM Is the Wizard, Then Code Is the Wand. Yang et al. 2024.

### Weaknesses
W1) While the paper took steps toward explaining it (via the NLL experiments in Table 4), I still found it a bit underanalyzed *why* using pseudocode plans works better than natural language plans. It would be useful to also investigate whether the pseudocode plans are also more accurate or otherwise higher in quality semantically than natural language plans, perhaps doing some human evaluation on a small number of samples.

W2) The experimental setup left a few stones unturned which would be natural to investigate:
- The model used to produce the code plans is Llama3-8B-Instruct, which is different (and potentially stronger) than any of the models fine-tuned on the plans (Mistral-7B, Llama-2-7B and 13B), so there's an element of model distillation in the paper. It would be helpful to evaluate whether a model could bootstrap itself as in the STaR paper, or at least to add a note on this.
- The evaluation could be more rigorous in places, see questions and suggestions below.

W3) While I think this could definitely be fixed for a camera-ready version of the paper, the writing of the paper could be a bit improved:
- Make it clearer that in no cases (if I understand right) are the code plans being executed -- they are just conditioned on by the LLM. It would help to call them "pseudo-code" consistently throughout the paper (rather than just in a few places as it is currently).
- Make the comparison clearer to STaR (as the method is similar) and Quiet-STaR (as their results are reported). The "planning in natural language" baseline in Finding 3 has some similarities to STaR (training the model to produce latent natural language plans) but also some key differences (all annotations are produced using query and answer; a different model is used to annotate than is being trained).
- I didn't feel that Table 1, with the comparison between CodePlan and other methods, added much to the paper as many of the attributes are unexplained.
- Clarify some details -- see questions below.

### Questions
Q1) Did the authors train Quiet-STaR to get the numbers in Table 3, or the results are from the model from the original paper? I found it strange that Quiet-STaR is worse on nearly every metric and dataset compared to the Mistral-7B version, is there any explanation for this?

Q2) It wasn't clear to me how CodeReason works: is the code actually executed, or does the prompt just encourage the model to produce executable code (but otherwise it works the same as CodePlan)?

Q3) "models are instructed to generate CoT-style responses". Are these CoT also included in the training data (if not, seems there is a mismatch between training and inference)?

Q4) do the few-shot examples also show Code Plan examples?

Q5) For the case studies in appendix B.3, are these code plans generated by the model itself (from only the query), or are the annotated code plans used in training (from the query and answer)?

*Minor Questions / Suggestions (no need to respond in the author response)*

- "delicate balance" part in 2.1 is a bit vague

- Figure 1: the digit example makes sense, but the renewable energy didn't to me. Where do the digits come from? There is some fact-based reasoning in the "response" text for *why* solar energy has limited accessibility, and it's not clear to me why it would be easier for the model to predict the numeric scores (needed to produce the code plan) rather than the facts. Is the main advantage of code here just to make sure all energy sources are mentioned, without repetition, and contrasted with each other?

- Many of the attributes in Table 1 are subjective, e.g. what makes this work efficient but QuietSTaR not? Why is CoT interpretable but QuietSTaR is not?

- Is "NLL" the average token-level NLL?

- "code is far more prevalent in pre-training corpora than natural language plans". This is an interesting claim, but needs evidence. "natural language plan" is probably harder to define and so also probably hard to quantify?

- Why is a 13B model necessary (the reason given in footnote 1 for why Llama-3 was not used)? Mistral-7B was used, and there is a ~7B Llama-3 model.

- "we synthesize 5K examples based on official implementation" implementation of what?

- The Kingma 2013 citation should probably be Kingma and Welling.

- There are a number of misspellings throughout the paper.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper describes a way to improve LLM reasoning ability by training it to plan in the form of code. Specifically, it first prompts LLM to generate code form plan given existing prompts and responses. Then, it finetunes LLM to generate plan given prompt, and generate response given plan and prompt. Experimental results show performance improvements across over the baselines on multiple benchmarks.

### Strengths
- The paper is well written
- The experiments are extensive, over multiple types of models and benchmarks

### Weaknesses
- The main novelty of this paper is that it uses code form rationale/planning instead of free text form. Besides code form, the specific way of training LLM to predict a prompted rationale is already proposed by the paper STAR https://arxiv.org/pdf/2203.14465 (see the Rationalization section). This is not clear from the paper.
- Given that the main novelty is code form plan, its effectiveness over a free text form plan is not demonstrated from the experiments. The authors should include a comparison against STAR, and an ablation study on replacing code form plan with text form plan with the exact same pipeline.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed to train LLMs to write code-form plans before performing reasoning on downstream tasks. The code-form plans are pseudocode written by Llama3-8B, and such augmentation is conducted on the 2M WebInstruct data (from Mammoth2 paper). They trained multiple models (mistral/llama2-7b/llama2-13b) on these data and observed large performance improvements over vanilla fine-tuning and training with language-form plans. Such performance gain is observed on 13 benchmarks including math reasoning, symbolic reasoning, instruction following, and multi-hop QA.

### Strengths
1. The authors proposed a novel idea (CodePlan): training LLMs to write code-based plans before conducting natural language reasoning. This idea utilizes the conciseness and structure of code to form high-level plans, and can benefit from LLMs' strong coding knowledge learned from pre-training.
2. The authors built a large-scale (2M) fine-tuning dataset with code-form plans based on WebInstruct. This resource could be valuable for future research.
3. The authors conducted solid experiments to prove CodePlan's supriority over vanilla fine-tuning and training with natural language plans. The benchmarks are tested out-of-distribution (no task-specific fine-tuning), which proves the generalization of the model.

### Weaknesses
I didn't find any critical weakness that would lead to rejecting the paper. However, the paper can be improved from the following perspectives:

1. The training data can be further improved. The current data is composed of two components: a pseudocode plan and the original natural language answer. I have the following concerns:

(1) The original answer is not further modified to better align and be more coherent with the pseudocode plan. This could limit the effectiveness of the training process.

(2) Two components seem separated and are not sufficiently integrated together as a coherent response, which does not look like a real human-style thinking procedure. As a result, models are still adapted to this specific response format instead of a universal reasoning strategy like CoT or the think-and-reason style of GPT-o1.

(3) The authors mentioned in the analyses that code-based reasoning are more effective than CodePlan on some tasks. So it may be worth exploring a combination of these two strategies. For example, there are papers that combined reasoning and code writing [1,2], which can be used to integrate with CodePlan for a more universal solution for model reasoning.

2. The authors did not conduct experiments on recent open-source models. The tested models (Mistral/llama-2) are released more than a year ago and are known for limited reasoning abilities. For example, fine-tuned llama-2 is still <50% on GSM8k while the more recent llama-3 is ~80%.
3. For presentation: the language-form planning baseline should be moved to the main table for a more comprehensive comparison, as I think this is an important baseline to show the effectiveness of CodePlan. Vanilla fine-tuning is a weak baseline because additional data are augmented into CodePlan's fine-tuning.

[1] MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning\
[2] ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving

### Questions
Most tables and figures in the paper are hard to recognize. I strongly suggest increasing the font size of these tables and figures, including Figure 2,3,4, and Table 3. For Table 3, do not separate the table into two parts. You don't need to report multiple metrics for multi-hop QA. Use abbreviations for dataset names.

### Soundness
3

### Presentation
3

### Contribution
3
