# Universe of Thoughts: Enabling Creative Reasoning with Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Reasoning based on Large Language Models (LLMs) has garnered increasing attention due to outstanding performance of these models in mathematical and complex logical tasks. Beginning with the Chain-of-Thought (CoT) prompting technique, numerous reasoning methods have emerged that decompose problems into smaller, sequential steps (or thoughts). However, existing reasoning models focus on conventional problem-solving and do not necessarily generate creative solutions by ``creative reasoning''. In domains where the solution space is expansive and conventional solutions are suboptimal, such as drug discovery or business strategization, creative reasoning to discover innovative solutions is crucial. To address this gap, first we introduce a computational framework for creative reasoning inspired by established cognitive science principles. With this framework, we propose three core creative reasoning paradigms, namely, combinational, exploratory, and transformative reasoning, where each offers specific directions for systematic exploration of the universe of thoughts to generate creative solutions. Next, to materialize this framework using LLMs, we introduce the Universe of Thoughts (or UoT, for short), a novel set of methods to implement the aforementioned three creative processes.  Finally, we introduce three novel tasks that necessitate creative problem-solving, along with an evaluation benchmark to assess creativity from three orthogonal perspectives: feasibility as constraint, and utility and novelty as metrics. With a comparative analysis against the state-of-the-art (SOTA) reasoning techniques as well as representative commercial models with reasoning capability, we show that UoT demonstrates superior performance in creative reasoning. This work introduces a new perspective on how LLMs can become autonomously creative, advancing the field to address problems that require more innovative solutions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the lack of autonomous creative reasoning in existing LLM reasoning frameworks. It draws on Boden's three creativity types from cognitive science to propose a computational framework for creative reasoning, instantiated as UoT (including C-UoT, E-UoT, and T-UoT). An evaluation benchmark with 3 open-domain tasks is constructed, using feasibility, utility, and novelty as metrics to compare against traditional frameworks and commercial models. Experiments show UoT (especially T-UoT) achieves the highest creativity scores, even enabling GPT-4o to outperform GPT-5, offering a new direction for autonomous creative reasoning in LLMs.

### Strengths
- It is the first to translate Boden's creativity theory into a deployable computational framework for LLMs.
- UoT's three variants have a logical progression and modular reproducibility. 
- The evaluation benchmark covers multiple domains with scientific metrics; it also introduces solution canonicalization to eliminate evaluation bias, ensuring reusability.
- Comprehensive experiments clearly demonstrate UoT's advantages and validate that "framework design outperforms model scale."

### Weaknesses
- Validation is limited to custom tasks (no public creative reasoning datasets are used). Analysis of key parameter impact boundaries is insufficient, leaving generalization and robustness unproven.
- Most high-scoring solutions of T-UoT are combinatorial innovations, failing to achieve its core goal of "rule-breaking." The effectiveness threshold for rule mutation is not analyzed.
- Utility and novelty evaluations rely on LLM subjective judgments (no human annotation verification). The criteria for constructing the "known solution set" are unclear, raising doubts about objectivity.
- Computational complexity is only derived theoretically; no experimental comparisons of reasoning time or resource consumption are provided, leaving efficiency advantages in large-scale scenarios unsubstantiated.

### Questions
- What are the criteria for selecting analogous problems in UoT's "idea pool construction"? How to avoid analogy bias and ensure automated selection?
- How does E-UoT verify the validity of new ideas? Is there a mechanism to filter irrelevant or invalid ideas?
- In the social cohesion task, how are metrics like "new ties" and "mixing" quantified? Are they based on simulation or real data?
- When comparing with GPT-5, why are its reasoning settings (e.g., use of prompt engineering) unstated? How to rule out the possibility of GPT-5 integrating UoT-like mechanisms internally?
- In complex fields (e.g., drug discovery), does UoT require domain knowledge injection? How to balance domain constraints and creativity?
- Are there solutions to E-UoT's "high novelty but low utility" issue, such as introducing feedback mechanisms or reinforcement learning?
- The necessity of the progressive order of UoT's three paradigms is unvalidated. Are there scenarios where skipping C-UoT still yields high performance? Can paradigm priorities be adjusted dynamically?
- Since T-UoT's utility does not outperform C-UoT, is rule-breaking not an optimal strategy for current LLMs, or do evaluation metrics fail to capture its long-term value?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Universe of Thoughts (UoT), a conceptual framework that enables creative reasoning in Large Language Models (LLMs). Drawing from Margaret Boden’s taxonomy of creativity, the authors define three reasoning paradigms, such as combinational, exploratory, and transformative reasoning, and outline how these processes can be instantiated in LLMs. A new benchmark with open-ended tasks is introduced to evaluate creativity using metrics of novelty, utility, and feasibility. Experiments show UoT outperforming Chain-of-Thought (CoT), Tree-of-Thought (ToT), and commercial reasoning models on these self-designed tasks.

### Strengths
Strengths:

- This paper tackles an interesting topic, which formalizes “creative reasoning” for LLMs.

- By grounding the conceptual framework in cognitive science (Boden’s theory), this paper establishes a solid theoretical background for creative reasoning.

- The proposed benchmark may inspire further exploration of creativity evaluation in reasoning systems.

### Weaknesses
Weaknesses:

- Limited Novelty and Positioning. The notion of enabling creativity in LLMs is not new. Substantial prior work has investigated open-ended or ill-posed tasks, such as creative writing [5], story generation [7], and idea synthesis [6]. The paper does not adequately engage with this literature or clarify how its proposal differs conceptually or methodologically.

- Missing Discussion of Established Reasoning Paradigms. The manuscript omits connections to analogical reasoning [1,2], inductive reasoning [4], and deductive reasoning [3], which are well-studied cognitive and computational paradigms that are central to creativity and generalization. Without situating UoT within or against these established reasoning forms, its theoretical framing of “creative reasoning” feels incomplete and insufficiently contextualized.

- Evaluation Limitations. The benchmark is self-constructed and narrow in scope, which raises reproducibility and bias concerns.

[1]. Thought propagation: an analogical approach to complex reasoning with large language models. ICLR 2024.

[2]. Large language models are analogical reasoners. ICLR 2024.

[3]. Inductive or deductive rethinking the fundamental reasoning abilities of llms.

[4]. Hypothesis search inductive reasoning with language models. ICLR 2024.

[5]. Deliberate Problem Solving with Large Language Models. NeurIPS 2024.

[6]. Moose chem:  large language models for rediscovering unseen chemistry scientific hypotheses. ICLR 2025.

[7]. Overview of Long Story Generation Challenge (LSGC) at INLG 2024. ACL 2024.

### Questions
The authors are encouraged to address the concerns in Weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes a computational framework and algorithmic pipeline to enable creative reasoning in LLMs, moving beyond conventional approaches like CoT/ToT to systematically explore a broader “universe of thoughts”. Inspired by cognitive science, the authors seek to invoke combinational, exploratory, and transformational creativity and operationalize each via structured prompting workflows that first retrieve analogous domains, decompose solutions into building-block ‘thoughts’, mutate conceptual rules, and synthesize novel solutions. They further design a benchmark with three open-ended tasks (traffic control, energy demand shaping, and social cohesion) and evaluate novelty, utility, and feasibility, showing that their Universe-of-Thoughts framework outperforms CoT/ToT/GoT and even exceeds GPT-5 on certain creative tasks using a weaker GPT-4o model. The core insight is that structuring the search over solution spaces and rules is key for creative problem-solving.

### Strengths
Timely contribution to the recent area of improving creativity in LLMs, with new creativity benchmark task.

The proposed approach is sound and based on well-studied paradigm of creative reasoning.

Experiments show that with proposed approaches, weaker (GPT-4o) LLM can outperform stronger comparison (GPT-5) in the measure of creativity proposed by the authors

### Weaknesses
Generalization to other creative tasks remains to be seen. The proposed creativity improvement approach seems to be tailored for the benchmark tasks proposed.

The approach seems to rely mostly on special prompts to help evoke creativity in LLMs and hence has limited technical novelty beyond prompt engineering.

### Questions
Regarding generalizability, how would this approach be adapted to other well-studied generative tasks such as code generation? It would be helpful to show its utility performance in a practical application.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors are inspired by cognitive theories and propose three creative reasoning methods: 
1. combinational reasoning, that transfers and combines thoughts from analogous domains 
2. exploratory reasoning, that  introduces novel thoughts to expand the existing problem space 
3. transformative reasoning, that alters fundamental rules to create a new, transformed solution space 

The three methods were implemented with a set of Universe of Thoughts (UoT) and show that UoT demonstrates superior performance than CoT, ToT, etc.

### Strengths
1. The paper studies an interesting problem about creative reasoning and proposed UoT with combinatory, exploratory, and transformative reasoning, which is intuitive and showed good results.
2. The UoT part is backed with detailed formalization and complexity analysis in the appendix.

### Weaknesses
The quality of both the constructed tasks and evaluation methods remain unclear. 

1. I am not sure how much I can trust the experimental results as the quality of the set of thoughts, search space, environment, etc., for the three tasks (One-Lane Bridge, Electricity Tariff, Social Cohesion) needs further validation. I could imagine this requires significant amount of engineering/simulation work and  domain expert knowledge. However, no discussion about the the process and validation to ensure the quality in these three domains are provided.

2. The utility and feasibility of a solution rely on LLM as a judge, yet the paper did not provide justification to support the use with a small set of human validation. This is my major concern and I could consider lower or increase my score depending on the author's response to this question.

3. Another smaller problem is that the paper seem to overstate its stance in **bridging the LLM reasoning and cognitive creativity**. This paper seems to have missed a line of work from the LLM/NLP community that also tackles the creative problem solving ability of LLMs, such as alternative uses, unconventional thinking, etc. I encourage the authors to consider adding comparison of model performance or discussion about these works.

Just naming a few as a starting point:

[1] Divergent association task: Divergent creativity in humans and large language models, by Bellemare-Pepin et al.,

[2] Unconventional (physical) problem-solving: MacGyver: Are Large Language Models Creative Problem Solvers? by Tian et al.,

[3] Idea generation: Can LLMS generate novel research ideas? A large-scale human study with 100+ nlp researchers, by Si et al.,

[4] There is a nice summary in: Large language models show both individual and collective creativity comparable to humans, by Sun et al.

### Questions
How are the thoughts, solutions, and existing solution space constructed for the three tasks? Where does the seeds come from and how do you ensure the accuracy and comprehensiveness?

### Soundness
2

### Presentation
3

### Contribution
3
