# OpaqueToolsBench: Learning Nuances of Tool Behavior Through Interaction

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Tool-calling is essential for Large Language Model (LLM) agents to complete real-world tasks. While most existing benchmarks assume simple, perfectly documented tools, real-world tools (e.g., general “search” APIs) are often opaque, lacking clear best practices or failure modes. Can LLM agents improve their performance in environments with opaque tools by interacting and subsequently improving documentation? To study this, we create OpaqueToolsBench, a benchmark consisting of three distinct task-oriented environments: general function calling, interactive chess playing, and long-trajectory agentic search. Each environment provides underspecified tools that models must learn to use effectively to complete the task. Results on OpaqueToolsBench suggest existing methods for automatically documenting tools are expensive and unreliable when tools are opaque. To address this, we propose a simple framework, ToolObserver, that iteratively refines tool documentation by observing execution feedback from tool-calling trajectories. Our approach outperforms existing methods on OpaqueToolsBench across datasets, even in relatively hard settings. Furthermore, for test-time tool exploration settings, our method is also efficient, consuming 3.5-7.5× fewer total tokens than the best baseline.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces OPAQUETOOLSBENCH, a benchmark designed to evaluate LLM agents’ ability to learn and adapt to opaque tools, i.e., tools that are underspecified, lack proper documentation, or exhibit non-transparent behaviors. To address the challenge, the authors propose TOOLOBSERVER, a framework that iteratively improves tool documentation through execution feedback (exploration and reflection). Results across all tasks demonstrate the feasibility of learning from execution trajectories to handle real-world, poorly documented tool APIs.

### Strengths
1. The paper targets a realistic and underexplored aspect of tool-augmented LLMs: using and improving opaque tools where documentation is minimal or unreliable. This is a valuable shift from previous works that assume perfectly specified APIs.

2. OPAQUETOOLSBENCH is conceptually clean yet diverse, covering structured, unstructured, and sequential tool-use scenarios (function calling, game-playing, and search composition).

### Weaknesses
1. TOOLOBSERVER largely reuses ideas from self-reflection and execution-based revision (e.g., Play2Prompt, Reflexion), differing mainly in when reflection occurs (interleaved rather than pre-task). 

2. Previous work like [1] has demonstrate the effectiveness of document refinement. What the main difference between this work and [1]? Could the author provide more explanation or comparison in terms of core technique contributions?


3. As for experiment evaluation, other metrics such as the number of reflection iterations, or fine-grained alignment between learned and ground-truth documentation, could be considered for a further validation. 

---

### Reference

[1] From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions

### Questions
See weakness above.

### Soundness
3

### Presentation
3

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
This work investigates how LLM agents can improve tool use in opaque tool settings by interacting with the tools and iteratively refining their documentation based on execution feedback. The authors propose TOOLOBSERVER, a reflection-based framework that continuously refines tool documentation by observing execution feedback from tool-calling trajectories. In addition, the authors introduce OPAQUETOOLSBENCH, a benchmark for learning in opaque tool settings where tool documentation is underspecified. It consists of three environments: general function calling, interactive chess playing, and long-trajectory agentic search.

Experimental results show that TOOLOBSERVER consistently outperforms all baselines on OPAQUETOOLSBENCH, achieving an average improvement of 18.6% in task success rate while maintaining strong token efficiency.

### Strengths
- The paper is well-organized and easy to follow, with clear presentation of benchmark statistics and evaluation metrics. 

- The data generation pipeline is simple, scalable, and comprehensively described. 

- The simplicity and token efficiency of TOOLOBSERVER make it directly applicable to real-world tool-use scenarios.

### Weaknesses
While three domains are tested, the evaluation metrics are somewhat limited. For example, Tables 2 and 3 only report overall accuracy. It would be informative to include additional metrics such as parameter accuracy or Abstract Syntax Tree (measures the generated function call format) etc.

### Questions
- In Section 4.3, the authors claim that TOOLOBSERVER is more token-efficient than Play2Prompt. However, this is not fully convincing. Since discovering correct tool usage requires trial and error, calling multiple tools at once may not necessarily reduce overall exploration cost and improve the final accuracy. that is, exploring all tools at once does not sound like a experience-efficiency way to me. 

- The opaque tool setting is indeed an underexplored and interesting problem. I wonder whether TOOLOBSERVER could also be applied to refine existing but suboptimal documentation—for instance, improving clarity or usability rather than fixing errors. Similarly, could OPAQUETOOLSBENCH include such cases where the tool documentation is mostly correct but not good enough? 

- TOOLOBSERVER explores multiple tools at once, but it remains unclear how the editor generalizes when the number of tools grows substantially. How does performance scale with hundreds or thousands of tools? 

- It would be insightful to compare TOOLOBSERVER with human annotators on these tasks. Including a “human oracle” baseline could help quantify how close the model is to human-level understanding of opaque tools.

### Soundness
2

### Presentation
3

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
This study designed a situation for tool calling agents when the tool specs are opaque. Under this setting, calling correct tools will be challenging because language models do not have enough information to decide what's the best tool to use or an appropriate parameters to send to the tool.

To evaluate and solve this problem, this study proposed the ToolObserver method that iteratively observe tool behavior and provide incremental improvements to the tool documentations. Experiment shows that the proposed method performs better than selective baselines on tool use benchmarks with opaque tool specs.

### Strengths
- The experiment results are encouraging, proving the the proposed method can generate good tool documentations that help improve the performance of models.
- Compared to the baseline method, the proposed strategy does not have to process all tools. as a result, they end up significantly save generation tokens for tool documents.

### Weaknesses
I feel the given situation is over-complicated / not well motivated. There are several reasons that tools won't be opaque in most applications:
1. agent developers are trying their best to improve the performance. giving the agent good tool documentation is among the easiest improvement they can do.
2. tool developers would maximize the chance that their tool gets called. as a result, they will work on improving the tool documentation so they are easy for the models to understand.
3. the best use case of the proposed situation might be the agent developers do not know what models they are giving the agent, and the tools developers does not want to tell models what are the tools designed for. this is a very rare case.

Secondly, I think the difference of the proposed strategy of the method and P2P is not strong enough for two reasons:
1. ToolObserve does not need initial documentation, but it does need the schema of tool inputs. as a result, the difference of not needing initial docs is just an incremental steps that tells an LLM to predict tool functionality based on inputs and tool outputs.
2. the token saving mainly comes from that ToolObserver does not have to explore all tools. However, when the number of requests are enough to cover all tools, this claim is no longer valid. ToolObserver and P2P both explore all tools and generates roughly same amount of tokens.

### Questions
n/a

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether LLM agents can improve their performance when using opaque tools by interacting with them and refining their understanding through feedback. To study this, the authors introduce OPAQUETOOLSBENCH, a benchmark covering three domains: general function calling, interactive chess playing, and long-horizon agentic search. The study finds that existing automatic tool documentation methods are unreliable and costly under opaque conditions. To address this, the authors propose TOOLOBSERVER, a framework that iteratively refines tool documentation based on execution feedback from tool-calling trajectories.

### Strengths
1. The concept of opaque tool invocation is novel and opens an interesting direction for further research.
2. The experiments are extensive and well-aligned with the proposed idea.
3. The paper presents a clear analysis, demonstrating how performance evolves across iterations and documentation levels.

### Weaknesses
1. The exploration and reflection phases in offline mode lack sufficient detail. The paper provides only a high-level description of these phases without sufficient algorithmic or implementation details.
2. ToolObserver offers limited novelty and resembles prior reflection-based methods.
3. The three benchmark scenarios may not fully represent real-world tool use.
4. Benchmark performance remains low. Even with optimization, the best reported results are still modest, which raises concerns about the practical usefulness and scalability of the proposed method.

### Questions
1. The definition of "opague" is relatively obscure. Can you explain it further?
2. If the tasks are not opqgue, how well the models performance?

### Soundness
2

### Presentation
3

### Contribution
2
