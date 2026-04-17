# Scaling Generalist Data-Analytic Agents

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework. Built on DataMind, we curate DataMind-12K, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10%. We also incorporate some empirical insights gained from our exploratory trials into the analysis experiments, aiming to provide actionable insights about agentic training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DATAMIND, a framework for training open-source data-analytic agents. The paper tackles three major challenges:

- **Insufficient data resources:** proposes a fine-grained task taxonomy and a recursive query synthesis pipeline to generate diverse, multi-turn analytic problems, coupled with a knowledge-augmented, self-consistency-filtered data curation process that produces the DATAMIND-12K dataset.

- **Training strategy:** designs a hybrid SFT + RL objective with dynamic weighting to balance imitation and exploration during training.

- **Unstable multi-turn rollouts:** develops a memory-efficient rollout mechanism with asynchronous interaction and chunked code management to improve reliability in multi-turn, code-based roll-outs.

Fine-tuned with DATAMIND, a ReAct-style agent with Qwen-2.5-Coder-14B backbone achieves state-of-the-art results (average pass@1 = 71.16 across three benchmarks including DABench, TableBench, and BIRD), outperforming both open-source and proprietary baseline models including DeepSeek-V3.1 and GPT-5.

### Strengths
1. **Comprehensive framework:** the paper proposes a unified framework that integrates training data synthesis and SFT + RL fine-tuning for training open-source data-analytic agents, offering a reproducible recipe for scaling open-source data-analytic models.

2. **Strong empirical performance:** with DATAMIND, Qwen-2.5-Coder-14B model outperforms all baseline proprietary and open-source models, while Qwen-2.5-Coder-7B model achieves performance comparable to GPT-5.

3. **Comprehensive analyses:** the paper provides insightful ablation studies that offer practical understanding in curating high-quality training data and stabilizing the training procedure.

### Weaknesses
1. Since part of the training data is sourced from Kaggle, is there a possibility that the training data may overlap with public benchmarks used for evaluation? Some discussions or analysis to rule out such possibility would further enhance the validity of the results.

2. All expert trajectories are sampled from DeepSeek-V3.1, and GPT-4o-mini serves as the judge during both trajectory filtering and reward modeling. This raises concerns about introducing potential biases from these two models, and some discussions on how to mitigate such biases would be very helpful.

3. LLM-as-judge with GPT-4o-mini is used to evaluate the performance, although for some benchmarks objective metrics are available (e.g. Rouge-L for TableBench as described in Appendix D). This raises concerns about the reliability of the evaluation, especially when GPT-4o-mini is also used as the judge during training trajectory curation and reward modeling. Including objective metrics results would improve the robustness of the evaluation.

### Questions
1. For Rule-based Trajectory Filtering, especially 1) Format compliance and 3) Linguistic integrity, were these checks also performed using LLM-as-judge, or implemented through e.g. regular expression?

2. How is the format reward computed?

3. In table 1, Qwen-2.5 Coder-14B fine-tuned with TableLLM and Table-R1 perform worse on TableBench than the pretrained Qwen-2.5 Coder-14B (ReAct). Is there any insight on why fine-tuning on TableInstruct led to worse performance?  

4. Regarding Fig 6, my understanding is that the conclusion “RL can narrow the performance gap between different base models” is based on models with the same backbone but different in SFT epochs (e.g., Qwen-7B with 1 v.s. 2 epochs). Could this conclusion extend to base models of different sizes (e.g., Qwen-7B v.s, Qwen-14B)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents DATAMIND, a data synthesis and training pipeline for building generalist data-analytic agents that operate through code-based multi-turn ReAct loops over heterogeneous files (.csv, .xlsx, .sqlite). The recipe includes: an 18-category task taxonomy with recursive easy-to-hard composition for query generation; knowledge-augmented trajectory sampling with self-consistency filtering and rule-based checks; a dynamically weighted SFT+RL objective (DAPO) with void-turn masking; and a memory-frugal rollout framework with sandboxed execution.

### Strengths
End-to-end, scalable pipeline. The work integrates dataset harvesting, fine-grained taxonomy, compositional query generation, trajectory vetting, and stable multi-turn rollout, offering a complete path from raw files to trained agents suitable for community reuse.

Training design that addresses common failure modes. The dynamic SFT+RL weighting, void-turn masking, and asynchronous chunked execution directly target instability in code-based multi-turn RL and are supported by ablations on reward/entropy dynamics.

Strong empirical results with broad format coverage. Consistent gains across three benchmarks and multiple file modalities, including conversion stress tests (e.g., TableBench/BIRD to unified formats), suggest robustness beyond narrow table QA.

### Weaknesses
Evaluation and reward rely on model-as-judge. Using GPT-4o-mini for both training rewards and final evaluation introduces circularity and potential stylistic bias; limited exact-match or verifier-based checks reduce measurement rigor, especially for descriptive answers.


Potential data overlap and protocol discrepancies. Training on BIRD/OmniSQL-adjacent corpora and converting evaluation benchmarks to new formats risk contamination or distribution shifts; fairness of comparisons may be affected by enforcing a single ReAct prompt across heterogeneous baselines.


Scope and generality claims are ahead of evidence. Results are confined to reasoning-oriented analytics; visualization, forecasting, and end-to-end scientific workflows are excluded. Claims of outperforming top proprietary models would benefit from additional third-party evaluators and held-out, verifier-checkable tasks.

### Questions
Please specify total RL budget: number of episodes, average steps per episode, total generated tokens, and wall-clock/compute (GPU type, days). The paper mentions ∼350 RL steps; what exactly constitutes a “step” here?

For the dynamic weighting γ, provide the full schedule, per-step values, and an ablation where γ is learned or adapted from validation signals instead of cosine.

How do SFT and RL gradients interact on overlapping tokens when a trajectory passes the RL filter? Are environment-emitted tokens always masked in both losses, and is there any credit assignment to code vs. natural-language tokens?

### Soundness
3

### Presentation
4

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
This paper addresses three key challenges in developing data analysis agents:
(i) the scarcity of training data resources,
(ii) the instability of long-horizon agent training and the unclear trade-off between SFT and RL training stages, and
(iii) the complexity of memory management in parallel agentic rollouts and multi-turn code generation.

To tackle these challenges, the authors propose:
(i) constructing a specialized training corpus,
(ii) dynamically combining SFT and RL losses with an adaptive weighting scheme, and
(iii) introducing an asynchronous agent generation and code execution framework with a chunk-wise code maintenance strategy to reduce peak memory usage.

### Strengths
1. The idea of building open-source data-analytic agents is interesting and timely.
2. The proposed system demonstrates promising results across multiple benchmarks.

### Weaknesses
1. It is unclear whether the observed performance improvements primarily stem from the proposed training methodology or from the curated dataset. A clearer ablation isolating these factors would strengthen the claim.
2. The benefit of dynamically combining SFT and RL remains ambiguous. From the rightmost subfigure in Figure 4, the training process appears to continuously shift from SFT to RL, although the paper does not follow an “SFT-first-then-RL” paradigm. It is not evident whether this dynamic approach yields tangible advantages over the “SFT-first-then-RL” paradigm. An ablation study is needed.

### Questions
Given the results in Table 1, it seems there is a clear upper bound on the performance across all models on the multiple benchmarks listed in the table. Do the authors have any insights into why this could happen and how to further improve the performance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a pipeline for constructing generalist data-science agents, addressing key challenges such as data scarcity, unstable multi-turn rollout, and balancing RL and SFT training. The authors curate a 12K-sample dataset and train models on it. The resulting DATAMIND-7B/14B models show promising performance on several data-analytic benchmarks.

### Strengths
The paper is well-motivated, and the proposed pipeline is comprehensive, combining a thoughtful data-synthesis process with a robust training strategy. The experimental results are promissing and the authors also analyze useful empirical insights into agent training.

### Weaknesses
1. Missing baseline comparisons: The paper does not include comparisons with other data-analysis agents such as DS-Agent or Data-Interpreter, which would strengthen the empirical claims.
2. Limited base model and benchmark diversity: The model is trained only on Qwen, which may introduce family bias and inflate the reported performance. Moreover, evaluation is restricted to three tabular/SQL-focused benchmarks; the generalization to broader data-analysis scenarios remains unclear.
3. Additional studies on different base models, dataset sizes, and the impact of recursive composition depth would make the analysis more comprehensive.

### Questions
1. Since gold answers are already available, why use a judge model for evaluation? Could this introduce noise, and how might that be mitigated?
2. Can DATAMIND handle multimodal inputs (e.g., CSV + plots), or is it limited to text-based tabular data?

### Soundness
3

### Presentation
2

### Contribution
3
