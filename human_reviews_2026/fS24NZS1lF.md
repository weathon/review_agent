# MAS-Zero: Designing Multi-Agent Systems with Zero Supervision

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Multi-agent systems (MAS) leveraging the impressive capabilities of Large Language Models (LLMs) hold significant potential for tackling complex tasks. However, most current MAS depend on manually designed agent roles and communication protocols. These manual designs often fail to align with the underlying LLMs’ strengths and struggle to adapt to novel tasks. Recent automatic MAS approaches attempt to mitigate these limitations but typically necessitate a validation set for tuning and yield static MAS designs lacking adaptability during inference, while also removing the flexibility to reduce to simpler systems. We introduce MAS-ZERO, the first self-evolved, inference-time framework for automatic MAS design. MAS-ZERO employs meta-level design to iteratively design, critique, and refine MAS configurations tailored to each problem instance, without requiring a validation set. Critically, it enables dynamic problem decomposition and agent composition through meta-feedback on solvability and completeness, and reduction to simpler systems when appropriate. Experiments across reasoning (math and graduate-level QA), coding, and agentic (search-based) benchmarks, using both closed-source and open-source LLM backbones of varying sizes, demonstrate that MAS-ZERO outperforms strong manual and automatic MAS baselines. It achieves substantial average accuracy improvements of up to 16.69% on reasoning, 16.66% on coding, and 5.45% on agentic tasks, while maintaining cost efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MAS-ZERO, a novel framework for automatically designing Multi-Agent Systems at inference time without requiring a supervised validation set. The core of the system is a meta-agent that iteratively refines an MAS configuration for each specific problem instance. The process involves three stages: MAS-Init, which generates baseline solutions from a set of predefined "building block" strategies; MAS-Evolve, where the meta-agent decomposes the problem, designs an MAS, and refines it based on self-generated feedback on sub-task solvability and completeness; and MAS-Verify, which selects the best final answer from all generated candidates. The authors demonstrate that MAS-ZERO achieves state-of-the-art performance across challenging reasoning, coding, and agentic benchmarks, establishing a new Pareto frontier for the accuracy vs. cost trade-off.

### Strengths
1. The motivations are very great. For example, this paper wants to eliminate the need for a labeled validation set.
2. The framework's explicit mechanism for breaking down complex problems into manageable sub-tasks and designing tailored sub-MAS for each is a robust approach to tackling intricate challenges.
3. The paper can be easily understood.

### Weaknesses
1. The framework centralizes a heavy cognitive load on the meta-agent, which must simultaneously evaluate agent capabilities, refine system architecture, and verify final answers. This creates a potential single point of failure and places a high floor on the required capability of the underlying LLM.

2. While the framework avoids a traditional validation set, it replaces it with an online, self-referential validation loop (Meta-Feedback and MAS-Verify). This raises concerns about evaluation blind spots or self-reinforcing biases, as the "judge" is the same model responsible for the design.

3. The system's creative scope is limited by the initial, manually-provided set of building blocks. The meta-agent primarily optimizes the composition and parameters of these existing modules rather than inventing fundamentally new agentic behaviors or interaction patterns.

4. The iterative, multi-candidate generation process incurs substantial computational cost at inference time for every single problem. While presented as a favorable trade-off, this per-instance expense could be prohibitive for many practical, large-scale applications.

### Questions
1. How does the performance gap between MAS-ZERO and simpler baselines (e.g., CoT-SC) evolve as the underlying LLM's capability increases? Does the sophisticated design process yield diminishing or increasing returns with more powerful models?

2. Did the self-evolution process uncover any surprising or counter-intuitive MAS designs that challenge conventional human intuition? Are there examples of novel agent compositions or problem decompositions that were consistently discovered for specific task archetypes?

3. As single models become more capable and integrate complex reasoning and planning internally, how do you see the role of external multi-agent orchestration frameworks evolving? Will their functions be absorbed into the base models, or will systems like MAS-ZERO remain essential?

### Soundness
2

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
This work proposes MAS-ZERO, a self-evolved, inference-time framework that requires no validation set and can iteratively optimize Multi-Agent System (MAS) configurations through meta-level design, which enabling dynamic task decomposition and agent composition, simplifies the system when appropriate, and outperforms both manual and existing automatic MAS baselines across multiple tasks and different Large Language Models (LLMs).

### Strengths
1. Originality: a new zero-supervision, inference-time self-evolving MAS framework, breaking validation set dependence and supporting dynamic task decomposition/complexity switching.
2. Quality: Comprehensive experiments across domains/models, comparisons with 11 baselines, ablation studies validating core modules, and planned open-sourcing ensuring reproducibility.
3. Clarity: Clearly describes the framework (3 key steps) and details (code templates, prompts), with complete appendices reducing understanding barriers.
4. Significance: Outperforms baselines on high-difficulty tasks (AIME24, SWE) and reveals the new insight that "simpler systems excel in some scenarios," with practical and guiding value.

### Weaknesses
1. From my understanding, this paper focuses on prompt engineering and involves no model training. Its performance upper bound is constrained by the capabilities of the underlying model, and its effectiveness bears similarities to test-time scaling. I consider its contributions limited, that is, had the authors proposed a training paradigm or a data framework to explore ways of pushing the model’s performance ceiling, the work would have been far more impactful.  
2. Drawing parallels to test-time scaling, increasing the inference budget will undoubtedly yield performance gains. I believe presenting a table that quantifies the trade-off between effectiveness and efficiency (e.g., latency, token consumption) would be highly valuable. It appears that time consumption surges with the system’s complexity, which poses a relatively significant challenge for user experience and real-world robotics deployment.  
3. The experiments lack in-depth analysis. For instance, there is no exploration of the specific patterns or scenarios where different methods fail.

### Questions
Address my weeknesses.

### Soundness
3

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
The authors propose MAS-ZERO, an inference-time-only framework for automatically designing Multi-Agent Systems (MAS) without supervision or validation sets. The core idea is to use a "meta-agent" that operates in a three-stage process: 1) MAS-Init, which runs a set of predefined "building block" strategies (like CoT, Debate) to gather initial solutions; 2) MAS-Evolve, an iterative loop where the meta-agent decomposes the problem, designs a novel MAS by composing the building blocks, executes it, and then critiques its own design based on "solvability" and "completeness" feedback; 3) MAS-Verify, which selects the best final answer from the pool of candidates generated by both MAS-Init and MAS-Evolve. The authors demonstrate that this self-evolving, per-instance approach outperforms both manual and prior automatic MAS baselines on tasks in reasoning, coding, and agentic benchmarks.

### Strengths
1. The framework's primary strength is its ability to adapt its design to each specific problem instance at inference time, without the need for a pre-tuned configuration or a validation set. This is a compelling alternative to static, validation-set-tuned systems.

2. The expanded ablation studies in Section 4.2 provide a clear and valuable breakdown of the system's performance. The ablations on MAS-Init, MAS-Evolve, meta-design, and meta-feedback effectively demonstrate that all components contribute meaningfully to the final result.

3. The method shows consistent and sometimes substantial improvements over strong baselines across multiple challenging domains (AIME24, SWE-Bench), demonstrating the practical effectiveness of the self-evolving design.

### Weaknesses
1. While the empirical results are now much stronger, the core conceptual framework can be viewed as a very sophisticated and well-executed combination of existing ideas (problem decomposition, self-refinement loops, agent routing) orchestrated via prompt engineering. The "meta-design" component, while effective, is fundamentally a well-crafted heuristic rather than a wholly new paradigm.

2. The entire system's success hinges on the reasoning capability of the meta-agent to correctly decompose problems and accurately critique its own designs. The paper acknowledges this the ablations and show the components are necessary, but the system is still vulnerable to meta-agent flaws. A failure in meta-feedback could send the iterative process in a useless direction.

3. For Table 5, the current experiment only swaps the meta-agent or the agents. A more powerful extension, which the framework seems suited for, would be for the meta-agent to dynamically assign sub-tasks to different agent models based on their known strengths (e.g., assign a reasoning-focused sub-task to o3-mini and a coding sub-task to a coder model), rather than using a homogeneous pool of agents.

### Questions
1. The cost-efficiency analysis (Fig. 1) is useful. However, for a practical application, what is the inference-time latency comparison? MAS-ZERO runs 9 total candidate generations (4 init + 5 evolve). How does this end-to-end time/cost compare to a single inference pass from a baseline like AFlow or MAS-GPT on the same test problem?

2. Figure 4B shows a massive performance leap with an oracle verifier, implying the MAS-Verify step is a significant bottleneck. Since MAS-Verify is also an LLM call, have the authors experimented with using a stronger, dedicated model (e.g., O3) only for the final verification step, even when the agents and meta-agent are weaker (e.g., GPT-4o)?

3. The system is initialized with four building blocks (CoT, CoT-SC, Debate, Self-Refine). How sensitive is the final performance to this specific set? For instance, what happens if it is only given CoT and Debate? Does the "MAS-Evolve" step successfully reinvent a Self-Refine-like process, or is its creativity constrained to what it is initially given?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work proposes an optimization framework that optimizes a meta-agent which initializes and designs the multi-agent framework. It gathers experiences during the training time and can directly generate well-performing multi-agent during the inference time. It's evaluated on two reasoning tasks, compared against broad baselines, and shown to achieve improved performance and higher cost-efficiency.

### Strengths
- Comprehensive baselines.
- Relatively clear presentation.

### Weaknesses
- The system seems to need quite some manual design, for example, the modules in the `MAS-INIT`, and prompts in Appendix J. It's not clear how the system is robust w.r.t these manual design and how much replies on human instructions.
- Following the above reason, the novelty might be lacking because the lack of automation.
- Some implementation details to ensure fair comparison is not clear, see questions.
- The title doesn't seem to be precise? Specifically, "ZERO SUPERVISION" holds in the sense that " support adaptivity at inference time, so that MAS designs can betailored per problem instance without relying on training or validation sets. ", but it still requires supervision during meta-agent training.

### Questions
- In Table 1, how do you control the input/output token budget to be the same? 
- How did you implement the baselines? 
- In Figure 1, is it inference cost or optimization cost? If it's one of these two, what's the other?

### Soundness
2

### Presentation
2

### Contribution
2
