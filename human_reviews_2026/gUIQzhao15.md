# Code-enabled language models can outperform reasoning models on diverse tasks

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Reasoning models (RMs), language models (LMs) trained with reinforcement learning to produce long-form natural language reasoning, have been remarkably successful, but they still cost large amounts of compute and data to train and can be slow and expensive to run. 
In this paper, we show that ordinary LMs can already be elicited to be strong reasoners at a level comparable to or even surpass their corresponding RMs (e.g., DeepSeek V3 vs R1) without finetuning, across diverse domains from instruction following and creative generation to mathematical reasoning. This is achieved by combining the CodeAct approach, where LMs interleave natural language reasoning with code executions in a multi-step fashion, with few-shot bootstrap in-context learning---from as few as five training problems. 
Analyzing four matched pairs of LMs and RMs, we find that our framework, coined *CodeAdapt*, enables three LMs to outperform the corresponding RMs on average over eight tasks (up to 22.9\%) while being 10-81\% more token efficient, and delivers superior performance for six tasks on average over models (up to 35.7\%). The code-augmented reasoning traces further display rich and varied problem-solving strategies. Our findings support that (1) CodeAdapt-style learning and reasoning may be domain general and robust and (2) code-enabled LMs are cognitively relevant and powerful systems, potentially providing a strong foundation for in-weight reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper argues that integrating executable code into LLMs fundamentally enhances their reasoning ability. The authors posit that reasoning tasks often require structured computation, which pure text-based models approximate but do not execute. By allowing LLMs to generate, manage, and interpret code within reasoning processes, models can perform verifiable intermediate computations and systematically improve task accuracy. The study evaluates this approach across mathematical and logical reasoning benchmarks, comparing code-enabled models to standard CoT prompting. Results show significant gains in reasoning accuracy and consistency, suggesting that access to a code interpreter reduces hallucination, enforces step validity, and better aligns outputs with formal reasoning logic. There are a number of well-established works along the lines proposed by the authors that use code-based approaches (known as PAL) to solve mathematical, multimodal, and multilingual tasks. The authors have not cited them, and I strongly recommend checking them out.

### Strengths
- empirical reuslts: Comparative experiments demonstrate consistent improvements over standard CoT baselines, confirming the practical benefit of code-enabled reasoning.

- clarity: The authors articulate a coherent argument that reasoning can be viewed as “neural-programmatic execution”, bridging formal computation and language understanding.

- impact: The approach is broadly applicable to domains requiring structured reasoning, such as mathematics, data analysis, and theorem proving.

### Weaknesses
- Severe figure readability issues: Many figures are illegible, overcrowded, or rendered in fonts too small to interpret. Axes and legends are poorly visible, making it nearly impossible to verify the numerical trends the authors describe. This critically undermines the transparency of their results and the paper’s overall readability. The authors should completely redesign the figures, enlarging fonts, clarifying contrasts, and simplifying layouts to foreground the main findings.

- theoretical grounding: The work is mainly empirical and descriptive, offering little formal analysis of why or when code execution leads to better reasoning.

- Unclear evaluation protocols: The paper does not sufficiently detail the criteria for “successful reasoning” or how correctness is verified when code is executed internally.

- confounds: Improvements may arise from tool-use effects (e.g., calling external functions) rather than intrinsic reasoning advances.

### Questions
How do you ensure that improvements from code execution reflect genuine reasoning enhancement rather than simple access to an external computational oracle?

How generalisable is your framework to reasoning tasks that are non-programmatic or linguistically abstract, where code execution provides no direct computational benefit?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CodeAdapt, which incorporates few-shot bootstrap in-context learning based on CodeAct. CodeAdapt achieves performance improvements in areas such as instruction following, language processing, and formal reasoning.

### Strengths
1. By adding few-shot bootstrap in-context learning on top of CodeAct, it enables self-exploration of reasoning trajectories, eliminating the need for expert demonstrations.
2. It achieves performance improvements across multiple domains.

### Weaknesses
1. The main issue with this paper is the lack of originality. Using code as a form of reasoning has already been widely studied in previous work. This paper mainly adds few-shot in-context learning on top of CodeAct, which does not provide substantial new insights. Domain adaptation through in-context examples has also been extensively explored in prior research.

2. The paper lacks comparisons with other few-shot in-context learning or few-shot domain adaptation methods.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a compelling study on how code-enabled language models (LMs) can achieve reasoning capabilities comparable to or even surpassing specialized reasoning models (RMs) without expensive reinforcement learning training. By integrating iterative code execution (CodeAct) with lightweight in-context learning (CodeAdapt), the authors demonstrate that ordinary LMs can excel across diverse tasks—from instruction following to mathematical reasoning—using only a handful of training examples. The work highlights a cost-effective and efficient alternative to resource-intensive RM training, contributing significantly to the advancement of hybrid reasoning systems in AI.

### Strengths
- The research addresses a timely and novel question—whether code-augmented LMs can compete with expensively trained RMs—offering a fresh perspective on resource-efficient AI reasoning.
- The proposed CodeAdapt framework is both effective and practical, achieving superior performance across multiple tasks with minimal data and computational overhead, as validated by extensive experiments.
- The paper is well-structured and clearly written, with comprehensive evaluations, ablation studies, and insightful analyses of reasoning patterns and resource usage.

### Weaknesses
While the study is thorough, future work could explore the scalability of CodeAdapt to a broader range of models and real-world applications to further strengthen its generalizability.

### Questions
- Using code to enhance language models is very common in current LLM research, which is why I'm concerned about the limited novelty of this paper. Can you summarize your core innovations again?
- In the experiments, is the baseline design too simple?

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
4

### Summary
This paper introduces CodeAdapt, a framework that equips LLMs with the ability to perform complex reasoning by interleaving natural language with code execution, combined with a lightweight, few-shot in-context learning procedure. The central claim is that this approach allows non-reasoning LLMs to match or even surpass the performance of specialized, expensively trained reasoning models across a diverse set of tasks.

### Strengths
1. Enhancing the reasoning capabilities of models under low-resource conditions is a topic worthy of research.
2. The paper is easy to read.

### Weaknesses
1. The contribution is limited, which is a small incremental work upon CodeAct.
2. The work does not compare with other training-free and training-based methods for enhancing LLM reasoning.
3. The experimental setup is limited to 30/32B and API-based LLMs. The lack of experiments with 7B models raises questions about whether the method's effectiveness is overly dependent on the inherent reasoning capabilities of the LLMs.
4. The dataset used in the paper is relatively small, which is insufficient to verify the robustness of the proposed method.
5. What is the performance of the reasoning LLMs with general in-context learning? The proposed method uses 2-shot for in-context learning, making the comparison not fair.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
