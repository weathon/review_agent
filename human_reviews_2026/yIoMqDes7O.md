# When Agents “Misremember” Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Recent advancements in large language models (LLMs) have significantly enhanced the capabilities of collaborative multi-agent systems, enabling them to address complex challenges. However, within these multi-agent systems, the susceptibility of agents to collective cognitive biases remains an underexplored issue. A compelling example is the Mandela effect, a phenomenon where groups collectively misremember past events as a result of false details reinforced through social influence and internalized misinformation. This vulnerability limits our understanding of memory bias in multi-agent systems and raises ethical concerns about the potential spread of misinformation. In this paper, we conduct a comprehensive study on the Mandela effect in LLM-based multi-agent systems, focusing on its existence, causing factors, and mitigation strategies. We propose ManBench, a novel benchmark designed to evaluate agent behaviors across four common task types that are susceptible to the Mandela effect, using five interaction protocols that vary in agent roles and memory timescales. We evaluate agents powered by several LLMs on ManBench to quantify the Mandela effect, and analyze how different factors affect it. Moreover, we propose strategies to mitigate this effect, including prompt-level defenses (e.g., cognitive anchoring and source scrutiny) and model-level alignment-based defense, achieving an average 74.40% reduction in the Mandela effect compared to the baseline. Our findings provide valuable insights for developing more resilient and ethically aligned collaborative multi-agent systems. Code and dataset are available at https://github.com/bluedream02/Mandela-Effect.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
An interesting paper studies a collective cognitive bias in multi-agent LLM systems akin to the human “Mandela effect,” where group interaction and specious evidence induce shared false memories that can persist over time. The authors also introduce MANBENCH to measure this phenomenon.

### Strengths
1. Introduces MANBENCH, the first systematic benchmark to evaluate the Mandela effect in LLM-based multi-agent systems, covering tasks, interaction protocols, and metrics.
2. Quantifies how the effect manifests and operates across model families, group compositions, group sizes, memory timescales, and knowledge domains.
3. Proposes a two-layer defenses—prompt engineering and alignment via SFT—with dataset design and training details, and empirically validates their effectiveness.

### Weaknesses
1. While the paper proposes two types of mitigation strategies (prompt-level and model-level defenses), these approaches remain relatively preliminary. For instance, the prompt-level methods rely heavily on predefined rules (e.g., cognitive anchoring and source scrutiny), which may not generalize well to unseen cases or adversarial scenarios. 

2. The paper would benefit from adding a discussion section to explore the practical risks of the Mandela Effect in real-world tasks, such as sensitive decision-making scenarios.

### Questions
See weakness

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
1

### Summary
This paper investigates the "misremember" effects that could potentially happen among multi-agent systems. It involves repurposing an existing QA benchmark and investigating several settings where agents could be wrong. 
Evaluation with several current SOTA LLMs shows that all suffer from such effects.
Furthermore, this work investigates prompting and sft methods to defend models against such effects and shows improvement.

### Strengths
I agree that multi-agent systems are increasingly involved in complex workflows, and investigating the failure case of a multi-agent system is an important issue. This paper focuses on a specific phenomenon, proposes an evaluation, and shows improvement methods. The findings align with expectations, and the ablations show more insight into the failure of mult-agent systems in LLM era.

### Weaknesses
Overall, I’ve seen many recent works draw on psychological concepts from human society and apply them to multi-LLM agent systems. However, we should keep in mind that LLMs differ fundamentally from humans; they have near-perfect memory, remain largely homogeneous, and are trained to play sycophantic roles as “user” and “assistant,” making them naturally inclined to agree with users. Specifically, I have several concerns about the work. 

First, the idea of a specific malicious agent that may render the whole multi-agent system down is not new, for example, https://arxiv.org/abs/2408.00989 (btw, this work is not cited in the paper). Although the specific topics might slightly differ from each other, I find it hard to tell how the evaluation framework is fundamentally different from previous work. 

Second, the contribution of the dataset seems to be simply repurposing an existing dataset, and I would argue that such context might not be the most practical scenario where the ground truth is aware to each party. That being said, the proposed benchmark is probably not a challenging benchmark, as each model would actually have the correct answer, and the most straightforward approach is to simply prompt every agent not to listen to each other. This hypothesis seems to be verified by the authors' later effort in climbing on the benchmark. The real challenge, however, is for scenarios where agents have to collaborate with each other to get information or knowledge. Can authors think of more scenarios in this direction?

Third, the experiments are only done with a single LLM for each simulation. Do authors have plans to run multiple LLMs in one scenario? Especially, this setting would be more aligned with the human society Mandela effect, where each party has different knowledge.

### Questions
see Weakness.

### Soundness
3

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
4

### Summary
This paper investigates the Mandela effect (collective false memory) in LLM-based multi-agent systems, a critical yet underexplored issue in collaborative AI. The authors introduce MANBENCH, a novel benchmark with 4,838 questions across 4 task domains (e.g., History, Domain-Specific Knowledge) and 5 interaction protocols (e.g., Role-based Short-term/Long-term) to evaluate the phenomenon. They test 13 LLMs (7 commercial, 6 open-source), confirming all models are susceptible to the Mandela effect—for instance, Qwen3-235B’s error rate rises from 25.48% (baseline) to 74.75% under the Role-based Short-term Protocol.

### Strengths
+ The idea proposed in this paper is interesting and insightful. 
+ This study is the first to systematically explore collective false memory (Mandela effect) in multi-agent systems, addressing a critical gap between individual LLM hallucination and group-level cognitive biases.
+ The paper provides comprehensive experiments and analysis, including diverse interaction protocols (simulating short/long-term memory and generic/role-based groups), and tailored metrics (reality shift rate, σ_max) etc.

### Weaknesses
- Although MANBENCH tasks are adapted from BIG-Bench Hard, they may not fully reflect the complexity of real-world multi-agent interactions (e.g., dynamic role changes, unstructured dialogue), potentially limiting the ecological validity of results.

- While defenses reduce the Mandela effect on MANBENCH, the paper provides little evidence of their performance across unseen tasks or domains (e.g., highly specialized fields like healthcare), leaving uncertainty about their broader applicability.

-  The interaction protocols predefine agent roles (e.g., Error Conclusion Initiator) and consensus direction, but real multi-agent systems often involve uncoordinated, conflicting inputs—this simplification may underestimate or distort how the Mandela effect emerges naturally.

### Questions
1. For the model-level defense, how does the balance between the resilience set and cooperative set (e.g., ratio adjustments) impact performance? Could a dynamic ratio (tailored to task domains) further improve both error resistance and knowledge absorption?
2. The paper notes that some models (e.g., GPT-5) self-correct false memories long-term, while others (e.g., Claude 3.5 Haiku) do not. What underlying LLM characteristics (e.g., context window size, training data) drive this difference in memory integrity?
3. MANBENCH focuses on verifiable factual tasks—would the Mandela effect manifest differently in subjective or creative tasks (e.g., collaborative content generation), and how might the proposed defenses adapt to such scenarios?
4. For role-based groups, the "suspicion-induced vigilance" effect reduces the Mandela effect when group size exceeds 9. Does this threshold vary across LLM types (e.g., open-source vs. commercial) or knowledge domains, and can this effect be proactively leveraged in defense design?

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
3

### Summary
This paper introduces MANBENCH, a novel benchmark to measure and diagnose the Mandela effect—the formation of shared false memories—in systems of collaborating LLM agents. MANBENCH comprises 4 838 multiple-choice questions drawn from BIG-Bench Hard, organized into four knowledge domains, and five interaction protocols varying in group composition (Generic vs. Role-based) and memory timescale (Short-term vs. Long-term). The authors evaluate 13 state-of-the-art LLMs, quantify a large reality-shift effect across all models, analyze key drivers (group size, domain, model scale, etc.), and propose both prompt-level (cognitive anchoring, source scrutiny) and model-level (supervised fine-tuning with resilience/cooperative data) defenses that reduce the effect by up to 74.4%.

### Strengths
1. The problem framing is novel. This paper is the first systematic study of collective false memories in LLM-based multi-agent systems, extending beyond individual hallucinations to social contagion effects.

2. This paper conducts a comprehensive evaluation across 13 models (commercial + open-source) and five protocols, with well-defined metrics (error rate, reality shift rate, maximal shift rate).

3. This paper provides not only an evaluation benchmark but also mitigation methods: two prompt-based strategies and an SFT-based model intervention.

### Weaknesses
1. The “specious evidence” dialogues are synthetic. It remains unclear how these engineered narratives map onto real-world multi-agent deployments or user-driven misinformation.

2. The benchmark is limited to multiple-choice questions. The transfer of findings to free-form, open-ended tasks (e.g., long-form debate or planning) is not evaluated.

### Questions
1. It is better to include significance tests (e.g., paired bootstrap) to support claims of “significant reduction.”

2. It will be interesting to involve humans in the loop in multi-agent collaboration and to see if the findings still hold. For example, what is the human performance on MANBENCH.

3. Is there any cross-task contamination or transfer of false memories when agents move between distinct domains within the same session?

### Soundness
2

### Presentation
3

### Contribution
3
