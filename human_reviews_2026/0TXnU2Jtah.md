# Modular Distillation Makes Small Models Think Like Big Ones

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Large Language Models (LLMs) have demonstrated exceptional performance in knowledge-sensitive reasoning tasks, but their practical application is still restricted by high computing demand. To address these challenges, we propose a novel modular distillation framework that breaks down knowledge-intensive reasoning tasks into three distinct components: an Analyzer for question decomposition, a Informant for context building, and a Reasoner for step-by-step reasoning inference. Unlike previous distillation methods that focus only on matching final outputs or step-by-step reasoning, our approach introduces a structured pipeline that enables the student model to learn both the analytical and reasoning abilities of the teacher model, while also capturing the teacher’s internal knowledge to guide more accurate and informed inference. This architecture improves interpretability, efficiency, and modularity, allowing for independent optimization of subcomponents. Empirical tests on three different benchmarks—OBQA, StrategyQA, and MedQA—show that our framework outperforms monolithic baselines in accuracy and computing efficiency while achieving competitive performance with much smaller models. Our findings demonstrate that smaller language models can do reasoning more efficiently when the whole process is divided into more manageable distinct components. This modular approach offers a practical and transparent alternative to relying on extremely large, resource-intensive models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a modular distillation framework that transfers reasoning abilities from a large teacher model to a smaller student by breaking reasoning into several interpretable modules. The goal is to help small models mimic large reasoning LLMs’ internal process more efficiently and effectively.

### Strengths
The problem of making small models perform comparably to large ones is both interesting and highly practical—especially for reasoning LLMs, where inference cost becomes huge as response length grows. Tackling reasoning efficiency through modular distillation is a direction that could be valuable for real-world deployment.

**I didn't read the proof part carefully because I don't have a strong theory background. So I may miss the strengths that lie in that part.**

### Weaknesses
I’m mainly concerned about the generalizability of the approach, since it imposes a strong human prior on what the teacher’s reasoning structure should look like.
+ Conceptually, this is a strong assumption and may not generalize well, given that SOTA reasoning LLMs often exhibit very diverse reasoning patterns.
+ Empirically, according to this paper (https://arxiv.org/abs/2503.01307), SOTA reasoning models go beyond just decomposition, grounding, and synthesizing—they can also branch or perform self-reflection, which the current method doesn’t capture.
+ This connects to a weakness in experimental setup: the evaluation only tests weaker base models (e.g., not ones like Qwen2.5 or Qwen3 also at the 1.5B/3B-level ) and on relatively simple reasoning benchmarks (e.g., StrategyQA). Stronger models on harder datasets (like MATH or AIME) would show whether the approach scales to more diverse reasoning behaviors.

### Questions
Is the DeepSeek model used here R1 or V3? Please clarify this in the paper.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a modular distillation framework, which decomposes a model’s reasoning capability into three specialized modules: Analyzer, Informant, and Reasoner. During training, each module is optimized independently through three dedicated loss functions, enabling them to acquire their respective specialized competencies. The framework is evaluated on OBQA, StrategyQA, and MedQA benchmarks. Experimental results demonstrate that the proposed modular approach outperforms traditional monolithic model baselines.

### Strengths
- Low memory footprint: The system achieves performance exceeding that of a distilled 8B model using three 3B models, making it suitable for deployment on resource-constrained or edge devices.

- Modularity: Each component (Analyzer, Informant, Reasoner) can be independently optimized, fine-tuned, or replaced, allowing the system to adapt to diverse applications without retraining the entire model.

- Theoretical justification: The paper provides theoretical evidence supporting modular distillation over conventional approaches, demonstrating that the modular method yields a tighter Evidence Lower Bound (ELBO), higher statistical efficiency (via Fisher information advantages), and lower approximation error.

### Weaknesses
1. Error propagation: The system follows a sequential pipeline (Analyzer, Informant, Reasoner). If the Analyzer fails to generate high-quality sub-questions, the errors propagate downstream, degrading the performance of the Informant and Reasoner and potentially leading to reasoning failure. A more detailed analysis of error propagation across the three modules could further strengthen the paper’s presentation and provide deeper insight into the limitations of modular reasoning pipelines.

2. Potential inference latency: Although VRAM usage is reduced, inference involves three serial forward passes (one per module). Compared with a single larger model that performs one forward pass, this setup may increase the overall end-to-end latency. While the paper emphasizes computational efficiency in terms of FLOPs, it does not explicitly report real-world latency measurements that account for model loading and runtime overhead.

3. Limited task generalizability: The framework is specifically designed for knowledge-intensive reasoning tasks and evaluated on QA benchmarks. It remains uncertain whether the “Analyze–Inform–Reason” decomposition can effectively generalize to other types of large language model (LLM) tasks, such as creative writing, summarization, or dialogue systems.

### Questions
1. For relatively simple tasks, such as straightforward translation, the Analyzer module might be unnecessary, whereas for more complex problems, such as advanced mathematical reasoning, the Informant module may play a less critical role. Since the paper primarily evaluates on multiple-choice question datasets focusing on textual reasoning, how well does this modular distillation framework generalize to other task types and modalities?
2. I am also curious about the number of training tokens used for each dataset in both the modular distillation and the Direct Reasoning Distillation settings. Understanding this would help clarify whether the performance gain primarily stems from the proposed method itself or from a larger training budget.
3. Would it be feasible to generate training data through the modular framework, then mix all modules’ outputs to train a single unified model, and finally compare its performance against the modular setup? I believe this could lead to a more fair and informative comparison between modular and monolithic training strategies.

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
This paper proposes a novel modular distillation framework (TMD) that decomposes knowledge-intensive reasoning tasks into three specialized modules: Analyzer, Informant, and Reasoner. The key contribution lies in distilling both analytical and reasoning capabilities of a teacher LLM into smaller student models, improving interpretability and efficiency. The main contributions include:
① A structured distillation pipeline capturing latent reasoning structures beyond output matching;
② Enhanced interpretability through modular intermediate steps;
③ Empirical evidence that smaller models achieve comparable performance to larger monolithic baselines.

### Strengths
① The modular design explicitly models latent variables (subquestions and knowledge snippets), advancing beyond previous methods.
② Section 3.1 provides variational bounds (ELBO) and Fisher information analysis, strengthening methodological foundations.
③ The pipeline enables traceability of reasoning steps.

### Weaknesses
① Comparisons are restricted to DRD, omitting state-of-the-art distillation methods (e.g., CoT distillation, retrieval-augmented KD).
② FLOPs and memory usage are reported, but real-world metrics (e.g., latency, energy consumption) are absent. And sequential module execution may introduce overhead.
③ Ablation tests only use 3B models, ignoring capacity allocation issues between modules (e.g., imbalanced parameter sizes).

### Questions
① How does TMD handle error accumulation across modules (e.g., incorrect subquestions from Analyzer affecting Reasoner)?
② Is TMD’s performance highly correlated with teacher quality? What safeguards exist for noisy teacher-generated labels?
③ Can the framework handle ultra-complex tasks (e.g., math reasoning requiring 50+ steps) without bottlenecks from sequential modules?

### Soundness
3

### Presentation
3

### Contribution
3
