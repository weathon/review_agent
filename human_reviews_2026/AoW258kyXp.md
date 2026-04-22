# First Things First: Teaching LLM-Based Agents to Prioritize Must-Haves before Nice-to-Haves

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Recent progress in multimodal large language models (MLLMs) has fueled significant enthusiasm in their potential to act as autonomous agents for real-world tasks. However, scenarios requiring agents to fulfill users’ complex, structured requirements remain largely underexplored. In this work, we examine reasoning tasks under three distinct requirement scenarios, each defined by the feasible solution set delineated by must-have and nice-to-have requirements: (i) Must-have requirements uniquely determine a unique feasible solution; (ii) Multiple candidate solutions satisfy the must-have requirements and are prioritized via the nice-to-have requirements; and (iii) No candidate solution satisfies the must-have requirements, in which case the agent should abstain from generating a response. We evaluate state-of-the-art MLLMs on 3,649 carefully constructed problems that reflect realistic service scenarios, including e-commerce platforms, booking systems, and map-based or ride-hailing applications. Our evaluation reveals that existing MLLMs exhibit catastrophic failures in all scenarios. Specifically, these models frequently misinterpret task requirements, violate must-have requirements, and produce invalid solutions. To address this critical gap, we propose First Things First Reinforcement Learning (FTF-RL) that explicitly optimizes reasoning over multi-priority user requirements. Experimental results show that our method substantially improves the task success rate compared to strong baselines. Moreover, FTF-RL yields general effectiveness on popular logical and mathematical reasoning tasks, including LogicVista, MathVision, and MathVista. Our finding suggests that enhancing requirement comprehension provides a simple yet effective pathway toward improving the broad generalization of MLLMs. Code and evaluation data are available at anonymity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the problem that current Multimodal Large Language Models (MLLMs), when acting as agents, often fail to handle complex user requests that include both essential "must-have" requirements and desirable "nice-to-have" preferences. Existing models tend to either violate must-haves or fail when requirements conflict. To address this, the authors introduce: FTF-BENCH: A new benchmark with 3,649 problems across realistic domains (e-commerce, booking, maps) designed to test MLLMs' ability to prioritize requirements.

### Strengths
1. The paper highlights a critical flaw in current agents, their inability to handle requirement priorities, which is crucial for real-world usability. 

2. The benchmark is comprehensive, covers realistic domains, and crucially includes the three distinct answer scenarios (Single, Multiple, Unanswerable) needed to properly evaluate requirement prioritization.

3. The paper demonstrates catastrophic failures in existing models and significant, consistent improvements with FTF-RL.

### Weaknesses
1. The benchmark uses synthesized user requests. Real user requests can be far more ambiguous, implicit, or contradictory than the structured (even if colloquial) prompts generated for the benchmark.

2.  Implementing a multi-objective RL framework like FTF-RL is significantly more complex than standard supervised fine-tuning (SFT) or basic RLHF, potentially limiting its adoption.

3. While the domains (shopping, booking, maps) are relevant, the findings might not directly generalize to all types of agent tasks (e.g., complex software control, scientific discovery).

### Questions
1. How robust is FTF-RL when faced with genuinely ambiguous user requests where the distinction between a must-have and a strong preference is unclear even to a human?

2. FTF-RL uses a rule-based reward model. How well would this scale to more complex tasks with potentially hundreds of implicit or explicit requirements? Would it require excessive manual effort to define rewards?

3.  Could models trained with FTF-RL become overly rigid, always asking for explicit must-haves vs. nice-to-haves, potentially disrupting natural conversation flow in simpler scenarios?

### Soundness
3

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
This paper proposes the FTF-RL (First Things First Reinforcement Learning) approach, which optimizes model reasoning through a multi-objective reward function encompassing format compliance, answer correctness, and requirement classification accuracy. It has achieved good performance on the FTF-BENCH proposed in this paper and some public benchmarks.

### Strengths
1. This paper is clearly written, with explicit introductions to its methods and experiments, making it easy for readers to follow .
2. The research field focused on in this paper holds significant research value .
3. This paper constructs a new evaluation benchmark, which is conducive to the development of this field .

### Weaknesses
1. The comparative experiments in the paper are really weak. They only compare the performance before and after using FTF-RL, without any horizontal comparison with other strategies. This makes it impossible to demonstrate the relative advantages of the proposed method and is completely insufficient to support the paper's conclusions.
2. The ablation experiments are also incomplete, as they only demonstrate the role of the single module $R_{requirement}$.
3. The paper has limited innovation. Although the proposed FTF-BENCH covers the "must-have requirement - nice-to-have requirement" hierarchy, its core idea still does not deviate from the existing benchmark paradigm of "instruction following + multi-scenario verification", and the multi-objective reward function is more of a combination of existing technologies.

### Questions
None.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a requirement hierarchy—must-have vs. nice-to-have—for real-world service-oriented multimodal agents. It builds FTF-BENCH (3,649 examples across e-commerce, ticketing/booking, and maps/transport) with three settings: single-solution, multi-solution, and unanswerable (requiring refusal). The authors propose FTF-RL, which adds three reward components in reinforcement learning—structured formatting, final-answer correctness, and requirement classification—to force the model to first parse and separate requirements before choosing and reasoning. Across multiple MLLMs, Direct is far below the Upper bound (given gold requirements); FTF-RL closes the gap substantially and shows some out-of-distribution gains on LogicVista/MathVision/MathVista.

### Strengths
**Novel angle**: In real services, the must-have vs. nice-to-have hierarchy is crucial yet often overlooked; FTF-BENCH quantifies this gap.

**Benchmark design addresses a blind spot**: The large gap between Direct and Upper helps disentangle perception errors from requirement parsing/priority errors.

**Transfer signal**: Gains on LogicVista/MathVision/MathVista suggest the “parse requirements first, then reason” paradigm has broader applicability.

### Weaknesses
**Task narrowness**: Although the angle is strong, the scenarios are relatively limited, making the practical generality of the approach less clear.

**Lack of tests on public benchmarks**: Because the task setup is narrow, the data distribution during testing is also narrow; it’s hard to judge whether the approach works across tasks in the same broad category (e.g., web information retrieval, etc.).

**Model-dependent bias in data creation and evaluation**: Prompts/requirements are model-generated and then human-checked; final-answer correctness is judged by a strong MLLM. The paper should disclose the specific generator/judger models and report human–model agreement, plus variance when swapping to a different model family, to reduce same-source bias and evaluation skew.

### Questions
**Judger and agreement**: Which model/version is used as the judger? What is the agreement with two human annotators on a random subset? What is the variance when replacing the judger with a different model family? (Please include sample size.)

**Unanswerable threshold and calibration**: What triggers refusal? Did you try confidence calibration or temperature scaling?

**Add at least one public benchmark**: Please include evaluation on at least one public benchmark.

### Soundness
3

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
4

### Summary
This paper proposes a new benchmark and training method for multimodal large language models (MLLMs). It introduces FTF-Bench, a dataset of 3,649 realistic service tasks (e.g., booking, shopping, maps) designed to test whether models can distinguish and prioritize must-have from nice-to-have requirements. Current MLLMs often misinterpret or violate essential constraints, leading to poor reasoning performance. To address this, the authors develop FTF-RL, a multi-objective reinforcement learning framework that rewards correct requirement classification, structured reasoning, and valid outputs. Experiments show that FTF-RL significantly improves task success rates and general reasoning benchmarks (LogicVista, MathVision, etc.). The work highlights requirement-aware reasoning as a key factor in building reliable, generalizable MLLM agents

### Strengths
1. The paper introduces FTF-Bench, a large and well-designed benchmark that captures real-world service scenarios with clear distinctions between must-have and nice-to-have requirements. This enables precise evaluation of models’ ability to handle multi-priority reasoning, which previous datasets ignored.
2. The proposed FTF-RL method integrates multi-objective rewards to improve requirement understanding, structured reasoning, and output validity. Experiments demonstrate significant accuracy gains across both in-domain (FTF-Bench) and out-of-domain reasoning benchmarks.
3. The study conducts comprehensive evaluations on proprietary and open-source MLLMs, showing consistent improvements and clear diagnostic insights. Moreover, models fine-tuned with FTF-RL generalize better to unrelated logical and mathematical reasoning tasks, proving its broader effectiveness.

### Weaknesses
1. The number of baselines in the experiments is too limited; adding more baseline results would make the experimental section more comprehensive and convincing.  
2. The paper’s analysis of why requirement-aware reasoning can enhance general reasoning capabilities is overly vague and unconvincing. In tasks like Math, reasoning intuitively does not involve must-have and nice-to-have requirements, so the contribution of FTF-trained models to general reasoning tasks needs more detailed quantitative and qualitative analysis.  
3. There are some inconsistencies in the experimental results section: Section 5.3.2 claims evaluations on LogicVista, MathVision, and MathVista, but in Section 5.3.3, the “four benchmarks” include InfoQA instead of MathVista. This inconsistency raises doubts about whether the performance of FTF-RL on InfoQA (in Table 3) and the Ablation Study on MathVista (in Table 4) aligns with the authors’ analysis.

### Questions
1. It would be helpful to show how the model distinguishes between must-have and nice-to-have requirements to better understand its capability boundaries on such tasks, for example by presenting confusion matrices for different purposes.  
2. I suggest adding examples and qualitative analyses of the model’s performance on general reasoning benchmarks to explain how FTF-RL’s capabilities transfer to them, further emphasizing the scalability of this work.  
3. Providing metric changes during the FTF-RL training process (e.g., answer correctness and classification accuracy) would better highlight the necessity of adopting RL instead of direct SFT in this domain.  
4. To more fairly evaluate the effectiveness of FTF-RL, I would like to know whether the 10% evaluation subset contains any bias. Including results of other models on this subset and comparing them with the full dataset outcomes would effectively clarify this point.

### Soundness
3

### Presentation
3

### Contribution
3
