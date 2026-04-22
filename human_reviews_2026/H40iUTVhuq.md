# Progressive Reverse Understanding Improves Text-to-SQL Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Recent advances in Text-to-SQL have significantly improved natural language interfaces to databases. Despite this progress, existing approaches still exhibit limitations of dependence on high-quality prompts and costly task-specific data, underscoring the need for more efficient and adaptable solutions. In this paper, we propose Progressive Reverse Understanding (\textbf{PRU}), a novel training paradigm that incrementally enhances Text-to-SQL generation by encouraging models to reason in reverse: from SQL back to natural language. PRU decomposes the learning process into progressive stages where models first learn to construct user intents from structured SQL, and then iteratively refine the mapping between language and structured queries. This bidirectional learning enables deeper semantic alignment while alleviating reliance on prompt quality and reducing the need for additional large-scale task-specific data by leveraging the data pairs inherent in the dataset. Extensive experimental results on the Spider dataset with open-source language models demonstrate the effectiveness of our approach in enhancing Text-to-SQL performance. Moreover, ablation studies highlight the critical role of progressive reverse understanding in improving both syntactic correctness and semantic fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a training schedule for text-to-SQL called Progressive Reverse Understanding (PRU). The model is first trained on SQL→NL (reverse) to internalize SQL structure, then the ratio is gradually shifted toward NL→SQL, keeping a small (~5%) reverse signal at the end to stabilize mapping. On Spider, this multi-stage, ratio-based curriculum outperforms plain forward SFT and a one-shot bidirectional baseline.

### Strengths
1. Clear, easy-to-implement recipe; fits well with open LLM text-to-SQL efforts.
2. Ablations are solid: varying reverse stage length and residual ratio show the gains are real.
3. Motivation is reasonable.
4. Writing is clear; pipeline is easy to follow.

### Weaknesses
1. The contribution appears incremental: SQL-driven / bidirectional augmentation has been explored in recent open-source text-to-SQL systems such as CodeS (SQL-to-Question Augmentation), and the main novelty here seems to lie in the specific progressive scheduling strategy rather than in the idea of leveraging SQL-side signals itself. 
2. The current evaluation focuses on Spider only; adding results on more recent and challenging benchmarks (e.g., Spider 2.0, BIRD, and robustness-oriented suites like Spider-Realistic / Dr.Spider) would strengthen the claim that the proposed schedule generalizes beyond a single dataset.

### Questions
Please refer to the above questions.

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
The paper proposes Progressive Reverse Understanding (PRU), a bidirectional training paradigm for Text-to-SQL that explicitly incorporates reverse reasoning (SQL→Text) both before and during forward training (Text→SQL). The method progressively shifts from reverse-only training to forward-dominant training while retaining a small reverse signal, based on the hypothesis that early reverse exposure improves grounding and the residual reverse signal stabilizes alignment. Experiments on Spider show gains over prompt-based and fine-tuning baselines, reporting 75.8% EX and 70.6% EM. Ablations attribute improvements to the reverse pre-training stage, the progressive schedule, and maintaining a small residual reverse ratio. Sensitivity experiments examine the duration of reverse pre-training and curriculum shaping.

### Strengths
The idea is clear and straightforward with a concrete training recipe. A progressive curriculum that begins with SQL→Text and gradually shifts to Text→SQL is intuitive, easy to implement, and broadly applicable. PRU demonstrates solid empirical gains on Spider: the reported improvements over baselines suggest that PRU helps produce syntactically precise queries.

### Weaknesses
1. The evaluation scope is very limited; this will be the primary reason for recommending rejection. The paper evaluates only on the Spider dataset (released in 2018). There are widely recognized Spider variants (e.g., Spider-Syn, Spider-DK) and the more recent and important BIRD benchmark [1] for LLM-based text-to-SQL. A method claiming generality in an LLM setting should be validated in large-scale, real-world scenarios.

2. The baseline comparison is insufficient. The authors should further review recent literature to provide fair and representative competing baselines [2]. In the public Spider leaderboard [3], many frameworks— including PLM-based systems—outperform the proposed PRU.

3. The description of the reverse understanding component lacks clarity. The reverse task trains 𝑦→𝑥 using the original question 𝑥 as the target. If the reverse task simply reconstructs the paired natural language question, it risks being a trivial inverse mapping of the dataset rather than promoting robust abstraction (e.g., recovering intent instead of replicating surface form). It is also unclear whether paraphrasing, schema verbalization, or value normalization are applied to encourage generalization beyond copying. Relatedly, the reverse objective conditions on (𝑦, 𝑆). How 𝑆 is injected, and whether reverse training improves schema linking in the forward direction, are not sufficiently explained.

4. The error analysis is limited. Although overall performance improves, there is no qualitative or category-level error investigation. The claim that PRU improves syntactic and semantic fidelity would be stronger with finer-grained diagnostics.

[1] Jinyang Li, et al. "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs" In Proceedings of NeurIPS, 2023. 
[2] Zijin Hong, et al. "Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL" IEEE TKDE, 2025.
[3] Spider Leaderboard. https://yale-lily.github.io/spider

### Questions
1. What is the impact of reverse training on schema linking and compositional generalization?

2. Can the PRU curriculum be adapted or optimized dynamically during training, rather than using a fixed schedule?

3. See weaknesses above.

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
4

### Summary
This paper presents a novel training paradigm, Progressive Reverse Understanding (PRU), for Text-to-SQL generation. The core idea is to use bidirectional learning—first training a model to generate natural language questions from SQL (reverse construction) and then progressively shifting focus to the standard Text-to-SQL task—to improve semantic alignment and syntactic correctness.

### Strengths
1. Novel and Well-Motivated Methodology: The proposed PRU framework is genuinely innovative. It moves beyond standard forward-only fine-tuning by incorporating a curriculum of reverse reasoning (SQL-to-Text), inspired by human learning processes. The progressive scheduling strategy, which gradually phases out reverse data in favor of forward data, is a thoughtful design that mitigates error propagation and provides a smooth learning curve, strengthening the model's structural comprehension.

2. Extensive and Rigorous Experimental Validation: The paper provides a comprehensive evaluation on the standard Spider benchmark. The results are compelling, showing state-of-the-art performance, particularly a significant improvement in Exact Match accuracy. Furthermore, the authors go beyond a simple comparison by including a thorough ablation study and sensitive analyses on critical hyperparameters (e.g., M1 duration, curriculum shape, residual reverse ratio). This systematically validates the contribution of each component of their proposed framework.

3. Significant and Meaningful Performance Improvement: The method achieves a top-tier Execution Accuracy (75.8% EX) and a notably higher Exact Match accuracy (70.6% EM) than the strongest baseline. The substantial lead in EM is particularly important, as it indicates the model is generating more syntactically precise and semantically faithful SQL queries, not just queries that happen to execute to the correct result. This demonstrates a qualitative improvement in the model's understanding.

### Weaknesses
1. Computational Inefficiency and Training Complexity: The proposed progressive training strategy involves multiple sequential fine-tuning stages (M1 to Mn). While the use of LoRA makes this feasible, the multi-stage process is inherently more complex and computationally intensive than single-stage fine-tuning. The paper does not discuss the total training time or resource cost compared to baselines, which could be a practical limitation for adoption in resource-constrained environments.

2. Limited Generalization and Scalability Assessment: The empirical validation is conducted exclusively on the Spider dataset. The generalizability of the PRU method to other Text-to-SQL benchmarks with different characteristics (e.g., WikiSQL for simplicity, BIRD for handling large, noisy real-world databases) remains unverified. Additionally, the approach is demonstrated primarily on the Llama-3.1-8B model; its effectiveness across other model architectures or sizes is not explored.

3. Insufficient Theoretical and Mechanistic Explanation: While the concept of "reverse understanding" is intuitively appealing and motivated by human cognition, the paper lacks a deep investigation into why and how it works so effectively. There is no analysis of how the model's internal representations change during the progressive stages or which specific aspects of SQL semantics and syntax are improved by the reverse training. A deeper dive into the mechanistic underpinnings would strengthen the theoretical contribution.

### Questions
1. some papers, e.g., [1],  have shown that sql-to-text is not necessarily helpful for text-to-sql. Can you explain it so that this is consistent with your ideas?
2. can you compare other two-round sql-generation methods, eg. [2].


[1] Benchmarking the text-to-sql capability of large language models: A comprehensive evaluation.  
[2] Pet-sql: A prompt-enhanced two-round refinement of text-to-sql with cross-consistency

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
This paper presents a bidirectional training strategy for Text-to-SQL fine-tuning. It generates reverse data, specifically generating a natural language question given SQL and schema. The training is structured as curriculum learning: it starts by training the model with the reversed data, then executes a scheduled shift to the forward data (the standard Text-to-SQL task) while retaining a residual reverse signal for stability. Evaluated on the Spider benchmark, the results show high performance in Exact Match accuracy compared with the selected baselines.

### Strengths
- The experimental methodology is rigorous, featuring a valuable hyperparameter analysis that explores key model variations. 
- The overall results are convincing, demonstrating the proposed method's good performance in the Text-to-SQL task.

### Weaknesses
A primary limitation is the evaluation scope, which relies solely on the Spider dataset. To fully demonstrate the method's robustness and generalization capabilities, it would be beneficial to include results from at least one additional, distinct Text-to-SQL benchmark. Furthermore, the paper would be significantly strengthened by extending the evaluation to other code generation tasks beyond SQL, which would further illustrate the generalizability of the proposed Progressive Reverse Understanding paradigm.

### Questions
Why didn't you evaluate in BIRD?

### Soundness
3

### Presentation
3

### Contribution
2
