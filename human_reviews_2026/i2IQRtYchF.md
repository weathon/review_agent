# The Imitation Game: Turing Machine Imitator is Length Generalizable Reasoner

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 2

## Abstract
Length generalization, the ability to solve problems of longer sequences than those observed during training, poses a core challenge of Transformer-based large language models (LLMs).
Although existing studies have predominantly focused on data-driven approaches for particular arithmetic operations or symbolic manipulation tasks, these approaches tend to be task-specific with limited performance on individual tasks.
To pursue a more general solution, this paper focuses on a broader case of reasoning problems that are *computable*, *i.e.*, problems that algorithms can solve, thus can be solved by the Turing machine, which operates over inputs of unbounded length.
From this perspective, this paper proposes **T**uring m**A**chine **I**mitation **L**earning (**TAIL**) to improve the length generalization ability of LLMs.
TAIL uses computer programs to directly synthesize chain-of-thought (CoT) data that imitate the execution process of a Turing machine, which *linearly* expands the reasoning steps into *atomic* states to alleviate shortcut pattern learning and explicit *memory* fetch mechanism to reduce the difficulties of dynamic and long-range data access.
To validate the universality and reliability of TAIL, we construct a challenging synthetic dataset covering 8 classes of algorithms and 18 tasks.
Without bells and whistles, TAIL significantly improves the length generalization ability as well as the performance of Qwen2.5-7B in individual tasks using only synthetic data, surpassing previous methods and DeepSeek-R1.
The experimental results reveal that the key concepts in the Turing machine, instead of the human-like thinking styles, are indispensable for TAIL for length generalization, through which the model exhibits read-and-write behaviors consistent with the properties of the Turing machine in their attention layers. This work provides a promising direction for future research in the learning of LLM reasoning from synthetic data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents TAIL, which is a specific chain-of-thought format inspired by Turing machines that facilitates length generalization. It synthesizes large datasets for a set of traditional algorithms and then finetunes Qwen models on that data and illustrates length generalization exceeding standard reasoning models (R1).

### Strengths
1. Compared to prior work on length generalization it considers a more varied set of tasks and shows strong results.
2. Most prior work on length generalization focuses on small models trained from scratch on simple tasks, this instead finetunes large models in a way that is perhaps more interesting or has more practical implications.
3. The specific method for generating varied synthetic data in the Turing machine style is clever.
4. The ablations of the chain-of-thought format are rigorous and useful

### Weaknesses
1. The paper over-claims the novelty of the idea of using a Turing-machine formulated CoT for length generalization. This is the central point of Hou et al. which is barely cited in the related work section. This paper does do much larger scale experiments and does study some intricacies of how to exactly define the CoT when finetuning larger models, but the core idea is exactly the same as Hou et al. and this should be made more clear in the paper.
2. Some of the claims of universality should also be more careful. While it is true that hypothetically any algorithmic task can be formulated as a TAIL-style cot, this still requires someone to hand-design the linearization, state, and memory fetcher components of the CoT used for the SFT data.

### Questions
none

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes TAIL, a data-driven framework for improving length generalization in LLMs. By synthesizing cot data that mimics the execution of a Turing machine, TAIL improves the performance of LLM's ability of length generalization. In general, TAIL enforces three structural properties, linear transition, atomic state and memory fetcher.

The method is evaluated on a challenging synthetic dataset spanning 8 algorithmic classes and 18 tasks. Results show that TAIL significantly enhances length generalization in Qwen2.5-7B, outperforming prior task-specific methods and even larger models like DeepSeek-R1.

### Strengths
1. The Turing machine alignment provides a principled approach to structured reasoning, systematically addressing length generalization through linearized execution and explicit memory management.
2. Evaluation across 8 algorithm classes and 18 tasks demonstrates impressive universality beyond simple tasks commonly studied.

### Weaknesses
1. Incomplete baselines. The paper doesn't compare against standard CoT fine-tuning, making it unclear whether gains come from Turing imitation or simply SFT. Comparisons are limited to un-trained base models.

2. The approach requires manually writing Python code for each task to generate TAIL-formatted CoT data, creating significant engineering overhead. This limits real-world adoption where users need automated solutions for new tasks.

3. The paper overlooks important connections to relevant literature. Two papers "Show Your Work: Scratchpads for Intermediate Computation with Language Models", "Beyond Single-Task: Robust Multi-Task Length Generalization for LLMs" deserves explicit discussion. The latter explores similar concepts of program-based CoT generation for length generalization but extends to multi-task transfer learning.

### Questions
1. Can the authors provide more baselines such as SFT with simple CoT or SFT without CoT?
2. It seems that simulating Turing execution with atomic states and memory fetching substantially increases reasoning length. Is it true? If so, will it be limited by the finite context, causing TAIL to actually fail in solving very long problems?
3. What are the additional contributions and innovations of TAIL compared to the two works mentioned above?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on a key limitation of current large language models: length generalization, meaning their ability to maintain correct reasoning on inputs far longer than those encountered during training. To address this, the authors propose TAIL (Turing mAchine Imitation Learning), a training approach let the mode mimic the step-by-step execution of a Turing machine. Experiments show that using only this synthetic data, TAIL can improve model’s length generalization ability and task performance.

### Strengths
1. This paper is well-motivated, studying  length generalization – an important problem of LLM
2. The proposed method only require synthetic data, which makes it easier to extend to more tasks. 
3. The paper performed comprehensive experiments and achieved improvements across diverse benchmarks, demonstrating strong generality and robustness.

### Weaknesses
1. Limited practical significance: The benchmarks are largely rely on deterministic symbolic computation. The so-called Chain-of-Thought effectively becomes a **Chain of Computations**, simply executing predefined procedural steps rather than engaging in genuine high-level reasoning. As a result, the improvements may reflect better simulation of algorithmic traces, rather than any substantive enhancement in “reasoning ability.”
2. Limited Transferability to real-world reasoning scenarios. The work does not evaluate performance on realistic tasks that require conceptual decision-making or deeper inference, such as multi-hop mathematical reasoning or logic problems. So it is unclear how TAIL can contribute to practical reasoning challenges.

### Questions
1. How does the proposed method extend to more realistic reasoning domains, such as GSM8K, MATH, or BIG-bench reasoning tasks? It would be crucial for verifying the real effectiveness of proposed method.
2. The work argues that TAIL improves computational reasoning, but many of the evaluated tasks could be directly solved LLM's tool use. Why just let LLM to write the code and execute the program for these type of tasks with deep computation depths? What advantages does TAIL provide over the agentic method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Turing Machine Imitation Learning (TAIL), which synthesizes CoT data that imitates the execution process of a Turing Machine to enhance the length generalization of Transformer-based LLMs. The authors claim that fine-tuning a model on synthetic data generated via TAIL improves its length generalization capability. They validate this claim on Qwen2.5-7B across various tasks and show that the fine-tuned model consistently outperforms the original one.

### Strengths
1. The idea of imitating Turing Machine execution is interesting and conceptually appealing.

2. TAIL is orthogonal to other approaches such as Index Hint, making it easily combinable with them.

### Weaknesses
1. The paper lacks sufficient elaboration and analysis on why TAIL improves length generalization.

2. The experimental design does not convincingly demonstrate TAIL’s effectiveness. Specifically, the comparison between the original Qwen2.5-7B and the model fine-tuned on task-specific data is not fair. Moreover, the paper does not include comparisons with other prompt-engineering methods, such as Program-of-Thought, to justify the claimed advantages of TAIL.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
