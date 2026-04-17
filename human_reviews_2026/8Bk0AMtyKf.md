# Towards Robust Agentic Systems through Generative Flow Exploration of Primitives

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
The automated design of agentic systems has emerged as a key challenge for scaling large language models (LLMs) beyond single-agent reasoning. While prior work has advanced task performance through handcrafted or automatically generated multi-agent workflows, robustness remains largely treated as an afterthought, leaving systems vulnerable to external adversaries and internal failures.  We propose AutoRAS, a framework for the Automated design of Robust Agentic Systems. The core idea is to represent system design as a sequence generation problem over symbolic primitives that jointly encode structural connections and behavioral actions. This abstraction enables (i) principled construction of executable workflows, (ii) integration of dynamic safety signals distilled from execution traces into the design loop, and (iii) flow-based optimization that propagates rewards across entire sequences to handle credit assignment and equifinality. Through this dual feedback channel, where numeric rewards guide exploration and textual signals refine behaviors, AutoRAS systematically improves both external resilience and internal reliability. Experiments on four datasets under four attack settings against 11 baselines, including handcrafted and automated designs, show that AutoRAS attains state-of-the-art results on three datasets and consistently exhibits the smallest performance drop after attacks (average 2.13). Additional transfer, ablation, and sensitivity analyses further confirm the effectiveness of our design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework called AutoRAS for automatically designing robust agentic systems. The main idea is to represent the system design as a sequence of "primitives" and then use a flow-based optimization method to find good sequences. It also uses "safety signals" from running the system to feed back into the design process. Evaluation on several benchmarks (MMLU, MATH, etc.) under different attacks shows that this method produces systems that are more robust than baselines.

### Strengths
I agree that multi-agent systems need to be more robust, and investigating how to automatically design them is an important issue. This paper focuses on this specific problem, proposes a method for doing it, and shows that it works better than other methods. The findings align with expectations, and the idea of using primitives and flow exploration is interesting for designing agent systems.

### Weaknesses
First, I find the proposed method overly complex. It combines a sequence generator, GFlowNets, Trajectory Balance, and a separate "analyzer" LLM for "textual gradients". It feels like a lot of moving parts just to generate a simple workflow. The 'primitives' idea seems like a re-branding of what many people are already doing with graph generation or even just scripting agent interactions. I find it hard to tell how this is fundamentally different from other automated agent design work like GPTSwarm or AFlow, which are cited but the distinction isn't made clear enough.

Second, the whole contribution seems to hinge on the "primitives". But the authors just hand-crafted these primitives themselves in Appendix D . The system is just learning how to stack these pre-defined blocks (like SAFE_Filter or AGT_COT). This doesn't seem to solve the design problem; it just pushes it down a level. The system isn't inventing a new robust workflow, it's just picking from a list of safety tools the authors already gave it. I would argue this is a much simpler search problem than the paper claims, and it's not clear if this approach can generalize beyond the authors' hand-picked primitive set.

Third, I have concerns about the evaluation. The paper's method, AutoRAS, is explicitly trained using a reward function that includes robustness scores (r_ext, r_int). It's no surprise that it does well on robustness when it's the only method being optimized for it. I find it hard to tell if this is a fair comparison against baselines like AFlow or GPTSwarm, which were likely designed for performance, not for these specific attacks. Furthermore, the dataset sizes in Table 4 are extremely small. For ProgramDev, it only uses 6 training samples, and for MSMARCO, only 20. How can a complex GFlowNet model learn anything from 6 or 20 samples? This makes me question if the system is really 'learning' a general design policy or just overfitting to a few examples.

Some typos:  line 155, the figure index is missing.

### Questions
See weakness.

### Soundness
2

### Presentation
2

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
This paper introduces AutoRAS, a framework for the Automated design of Robust Agentic Systems. It formulates agentic system design as a sequence generation problem over symbolic primitives that encode both structural and behavioral elements. The framework integrates flow-based optimization (GFlowNet) and textual feedback from execution traces to iteratively refine system robustness and reliability.

### Strengths
1. The method reframes agentic system construction as primitive-sequence generation with built-in structural validation and dynamic safety signals, offering a novel and generalizable paradigm.

### Weaknesses
1. While performance gains are strong, the paper does not detail the training overhead or inference efficiency compared to simpler baselines.
2. Could the abstraction limit the flexibility of agentic systems? For example, the real-world agents may have more diverse multi-agent systems and orchestration.

### Questions
1. How sensitive is AutoRAS to the design of the primitive vocabulary—does adding or removing primitives materially affect convergence and robustness?
2. Could the authors provide quantitative measures of computational efficiency and cost, particularly during the flow exploration phase?

### Soundness
4

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
4

### Summary
The paper introduces, AutoRAS, a framework for the Automated design of Robust Agentic Systems.
AutoRAS represents agentic systems as a sequence generation problem over symbolic primitives encoding both structural connections and behavioral actions. AutoRAS can be used to improve any agentic's system reliability. The authors also report experiment results on four datasets against 11 baselines showing how AutoRAS achieves the best result under attack.

### Strengths
- Significance: Designing robust agentic systems automatically is an important problem given the push in both academia and industry to move towards AGI. The majority of production agentic systems today need constant maintenance as the frontier models get updated or the tool APIs improve. Significant engineering time is spent on optimiznig the prompt and system design of agentic systems to adopt to the new model or tool api updates. Therefore, any system that can automate agent design reliably significantly contributes to all frontier agents in industry and academia. 
- Clarity: The paper does a great job clarifying why its hard to design a reliable agent system automatically. It categorizes the challenges into three categories of entanglement, unpredictability, and equifinality and explains each category clearly. 
- Novelty: The authors leverage two techniques that are useful. 1) they integrate robustness in the reward function. 2) they leverage textual gradients to refine the prompts.

### Weaknesses
- Application: Many of the real agentic systems working in production today leverage complex elements that cannot be mapped to the set of primitives the paper defines to begin with. For example, a coding agent such as cursor leverages fine-tuned models to apply a code change efficiently and reliably, which primitive can capture such modules? and how an automated system will arrive at such design? Many of the components in the current production agents come from carefully assessing the performance of different part of the system and crafting solutions to improve the performance. A system such as AutoRAS cannot replicate this process.
- Experiments: The manually crafted agentic systems that the authors have picked for evaluations are very primitive. It would be great if they could also compare with a production grade agentic system.

### Questions
- Why did the authors used smaller models the mini series from openAI and haiku and flash series from Anthropic and Google? Would the vanilla baseline results change switching to more powerful frontier models?

### Soundness
2

### Presentation
3

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
This paper introduces AutoRAS, a framework to automatically design robust, multi-agent LLM systems that are resilient to both external attacks and internal failures. The core idea is to represent the complex design of an agentic system—including its structure and behaviors—as a sequence generation problem using a defined set of symbolic "primitives." AutoRAS then uses flow-based optimization, guided by a novel dual-feedback loop of numeric rewards (e.g., accuracy) and textual safety signals (distilled from execution traces), to explore this design space. This method discovers agentic workflows that achieve state-of-the-art performance while exhibiting the smallest performance degradation under various adversarial attacks compared to existing methods.

### Strengths
- The paper reframes the agentic system design into a concrete sequence generation problem over symbolic "primitives." This abstraction makes the vast search space of possible designs (combining different structures, communication patterns, and agent behaviors) tractable and optimizable.
- AutoRAS embeds robustness directly into the optimization loop. The feedback signals about failure modes, allows the system to learn why designs fail and proactively favor more resilient architectures. 
- The framework demonstrates empirical advantages with state-of-the-art accuracy on several benchmarks and shows the smallest average performance drop under four different attack types.
- The workflows demonstrate transferability and high performance when executed with different backbone models (e.g., GPT-40-MINI, DeepSeek-V3.1, Claude-3.5-Haiku)

### Weaknesses
- Potentially Saturated Benchmarks: The paper relies on benchmarks like MMLU and MATH, which are becoming saturated in the sense that top-tier SOTA models (like the latest GPT, Claude, or Gemini series) can already achieve very high performance. This makes it difficult to assess how much of the performance is from the novel AutoRAS framework versus the underlying power of the base LLM it uses.
- Complexity of Optimization: The system uses flow-based optimization to search the sequence space, which can be computationally expensive and complex to train. The performance is also sensitive to parameters like sequence length and training samples per iteration.
- Missing Cost-Benefit Analysis: The AutoRAS algorithm is significantly more complex than a standard single-agent baseline. The paper does not provide a direct comparison of the computational cost against these simpler baselines, making it hard to evaluate if the performance and robustness gains justify the added complexity.

- Dependency on the Primitive Set: The entire framework's success is contingent on the quality and completeness of the predefined "primitive" vocabulary. If a highly effective or robust design pattern is not expressible using the existing primitives, AutoRAS will be unable to discover it.
- Dependency on LLMs: The framework's core optimization loop is critically dependent on the quality of its own internal LLMs. The "monitor" (Sec 4.2) uses an LLM to detect failures and generate safety signals; if this LLM fails to detect a novel or stealthy attack, the system cannot learn to defend against it. Similarly, the "analyzer" (Sec 4.1) and "textual gradient" (Sec 4.3) rely on an LLM to refine prompts, meaning the quality of the final agent behaviors is capped by this meta-LLM's capabilities.

### Questions
There are a number of typos such as line 047 and line 155.

### Soundness
2

### Presentation
2

### Contribution
2
