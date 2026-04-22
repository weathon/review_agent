# Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 6, 2, 4, 6

## Abstract
Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. In contrast, human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context. This difference between human cognitive flexibility and LLMs' reliance on a single reasoning style raises a critical question: while human fast heuristic reasoning evolved for its efficiency and adaptability, is a uniform reasoning approach truly optimal for LLMs, or does its inflexibility make them brittle and unreliable when faced with tasks demanding more agile, intuitive responses? To answer these questions, we explicitly align LLMs to these reasoning styles by curating a dataset with valid System 1 and System 2 answers, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense reasoning tasks. To analyze the reasoning spectrum, we interpolated between the two extremes by varying the proportion of alignment data, which resulted in a monotonic change in accuracy. A mechanistic analysis of model responses shows that System 1 models employ more definitive outputs, whereas System 2 models demonstrate greater uncertainty. Building on these findings, we further combine System 1- and System 2-aligned models based on the entropy of their generations, without additional training, and obtain a dynamic model that outperforms across nearly all benchmarks. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper, grounded in human cognitive theory, examines how System‑1 and System‑2 data influence large language models and analyzes the characteristics of S1/S2‑aligned models. It further introduces a training‑free, entropy‑based dynamic model selection method that improves answer accuracy. The motivation of the article is clear and the experimental part is also sufficient.

### Strengths
1. This article is well written, the concepts and formulas involved in the article are explained very clearly and the motivation is clearly expressed, which make it easy to follow.
2. The experimental part of this article is very thorough, with a large number of experiments and verification on the data set, making the article more convincing.

### Weaknesses
I'm curious about **line 242**. The article mentions a two-stage process. In the first stage, the answer lengths of S1 and S2 are similar, but after the second stage, S2's answer length is higher than S1's. Although the experiment is clearly described in the article, and I've looked at the prompt in Table 5 ("Therefore, xxx, the final answer is"), I still find it a bit strange why simply adding such a prompt increases the difference in answer length between S1 and S2. Could the authors provide a concrete example to illustrate this? Could you also provide a table showing the accuracy and number of tokens for the S1 and S2 models on GSM8K, Coin, and CSQA for a more intuitive understanding?

### Questions
Currently, most people believe that the difference between System 1 and System 2 models is length, which is largely due to the success of DeepSeek-R1. However, this article gives us a different perspective. The length of the answers in the first stage of S1 and S2 is similar. If the answer involves a lot of reflection (such as **aha moments** observed in DeepSeek), the answer length will inevitably be long. Have you observed similar phenomena in your experiments?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper conducts a series of experiments and analyses on System 1 and System 2 reasoning paradigms. The authors argue that each has its own strengths: System 1 is efficient for heuristic problem-solving, while System 2, by introducing controlled uncertainty to support reasoning, is better suited for tasks requiring rigorous logical inference. To address the question of when to use each reasoning mode, the authors train two types of models and introduce a reliability metric to determine which model is more appropriate for a given task.

### Strengths
- The paper is well motivated, and the experimental design is thorough. 
- The results align with expectations, and the authors conduct extensive statistical analyses to ensure their reliability.

### Weaknesses
- There are some contradictions in the paper. In Section 3.1, the authors state that LLMs can be guided toward either S1 or S2 reasoning through prompt engineering. However, in the experiments, they instead use DPO to explicitly train models to prefer one reasoning paradigm over the other.
- The reliability metric introduced in Section 3.2 lacks sufficient justification. Specifically, entropy and variance are not intuitively compatible quantities, and weighting and summing them together may undermine their original semantic meaning.
- Regarding the question of when to adopt the more appropriate reasoning paradigm, the proposed approach is not very efficient. It relies on training both S1 and S2 models and then selecting between them using the metric, which increases computational cost.

### Questions
In addition to the weaknesses noted above, there are several further questions:

- For the two types of models used in the experiments, the instruction-tuned version and the CoT version, the experimental results are basically consistent. I would like to ask if you have tested the results after using the few-shot version?
- As in 3.3, where the overly long S2 response was shortened, can there be a case to show S1 and S2 responses to a question when their lengths are similar?
- I actually still want to ask why you need to shorten the answer in S2 in 3.3. Can this shortened answer ensure a reasoning effect similar to before? For example, if we input the two types of reasoning steps into LLMs and then force the output answer. Are their answers consistent?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors explicitly align LLMs to these reasoning styles by curating a dataset containing valid System 1 and System 2 answers, and evaluate model performance across reasoning benchmarks. Their results reveal an accuracy-efficiency trade-off: System 2-aligned models demonstrate superior performance in arithmetic and symbolic reasoning, while System 1-aligned models achieve better results in commonsense reasoning tasks. To analyze the reasoning spectrum, they interpolate between the two extremes by varying the proportion of alignment data, observing a corresponding monotonic change in accuracy. A mechanistic analysis of model responses indicates that System 1 models tend to produce more definitive outputs, whereas System 2 models exhibit greater uncertainty. Building on these findings, the authors further combine System 1- and System 2-aligned models based on the entropy of their generations without requiring additional training, obtaining a dynamic model that outperforms individual models across nearly all benchmarks.

### Strengths
The authors' methodology section is clearly articulated, well-supported by diagrams, and validated through comprehensive experiments.

### Weaknesses
1. While one key contribution is "a dynamic model that outperforms across nearly all benchmarks," the main text provides no quantitative data to indicate the magnitude of improvement over previous methods. The authors should include specific performance gain figures whenever claiming superiority over existing approaches.

2. The methodology is primarily demonstrated through Figure 5 showing GSM8K performance, while most experimental results are presented in the appendix in graphical form, making it difficult for readers to quantitatively appreciate the advantages. It is recommended to add an "Ours" row in Table 1 to clearly showcase the method's superiority.

3. According to Figures 5 and 11, the parameter w appears to be separately tuned for each dataset, essentially trying all hyperparameters w for every dataset. This constitutes an unfair evaluation in benchmark testing. Since the method is described as a "dynamic model," there should be a reasonable approach for dynamic adjustment that uses consistent parameters across all evaluations for fair comparison.

4. The models employed by the authors appear somewhat outdated. It is recommended to incorporate more recent model to demonstrate state-of-the-art performance.

5. It is recommended to improve the paper's writing, as the article contains substantial redundant and repetitive information while lacking crucial illustrative examples and quantitative data support. For example, the authors devote excessive length (nearly a full page) to the conclusion section, with substantial portions being repetitive and redundant. Meanwhile, significant experimental content is relegated to the appendix, which undermines the substantive contributions in the main text.

### Questions
Same as above

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores aligning LLMs with System 1 and System 2 reasoning. It creates a dataset of 2k questions, each paired with responses of both reasoning styles, and normalize their lengths. Using DPO and SimPO, it finds that System 2 aligned models excel at arithmetic and symbolic reasoning, while System 1 aligned models perform better on commonsense tasks. The paper further introduces an entropy-based dynamic model that adaptively selects between the two reasoning modes without additional training.

### Strengths
1. The experiments clearly show distinct trends between System 1 and System 2 aligned models, supporting their claims on the difference across different tasks.

2. The performance transition when interpolating between the two reasoning styles is particularly interesting, showing how alignment can gradually shift reasoning behavior from S1 to S2.

3. The writing is clear and well-organized, providing insightful observations on the differing alignment behaviors of the two reasoning modes.

### Weaknesses
1. The refinement of matching response lengths between System 1 and System 2 outputs could unintentionally change their natural characteristics. Expanding concise answers into longer forms may introduce redundancy and reduce the intuitive nature of System 1 reasoning. It would be helpful to see whether training without this adjustment yields consistent findings.
2. The proposed entropy-based approach requires running both System 1 and System 2 models and then selecting the final answer, which is an ensemble rather than an adaptive reasoning switch. This limits the claimed cognitive analogy and introduces more inference cost.
3. The math datasets used involve relatively few reasoning steps. Their simplicity may mask the deeper distinctions between System 1 and System 2 reasoning modes.

### Questions
Is there any evaluation of whether the aligned reasoning behaviors transfer out of distribution? For instance, does System 2 alignment on math tasks generalize to other domains, and how does it affect performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates whether LLMs should always use deliberate reasoning (System 2) or adapt their approach like humans do. The authors trained models to use either fast, intuitive reasoning (System 1) or slow, deliberative reasoning (System 2) using a dataset based on cognitive heuristics. Testing across 13 benchmarks revealed a trade-off: S2 models excelled at arithmetic and symbolic tasks but generated longer responses, while S1 models were concise and better at commonsense reasoning. S1 models showed higher confidence with lower entropy, whereas S2 models exhibited more uncertainty. The authors also trained hybrid models with varying S1/S2 ratios and developed a training-free approach that dynamically selects between S1 and S2 outputs based on confidence metrics, improving performance on most benchmarks.

### Strengths
- The methodology is well-grounded in cognitive science theory and implemented through a carefully controlled dataset with rigorous safeguards against superficial pattern learning.
- The empirical evaluation comprehensively demonstrates meaningful task-dependent performance patterns, while mechanistic analysis provides deep insights into how the models differ behaviorally and confirms that reasoning styles form a continuous spectrum rather than discrete categories.

### Weaknesses
-  All experiments use Llama-3-8B and Mistral-7B. No evidence for generalization to smaller models (1-3B) with potentially insufficient capabilities, or larger models (70B+) with different emergent abilities.
- The authors normalize S1/S2 response lengths to prevent superficial learning. However, in authentic S1/S2 reasoning, length differences are intrinsic—S1 responses should naturally be shorter. Equalizing lengths during training may teach models an unnatural constraint.

### Questions
- Can you test on one larger model to validate that S1/S2 distinctions persist at scale? Do larger models naturally exhibit more S2-like behavior?
- You normalized S1/S2 response lengths to avoid superficial cues, but S2 models still produced longer outputs at inference. I wonder if an ablation training on un-normalized data may quantify how much "overthinking" behavior is inherent to S2 alignment versus an artifact of training on longer examples.

### Soundness
3

### Presentation
4

### Contribution
3
