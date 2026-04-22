# RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 2, 8, 6

## Abstract
Today's LLM ecosystem comprises a wide spectrum of models that differ in size, capability, and cost. No single model is optimal for all scenarios; hence, LLM routers have become essential for selecting the most appropriate model under varying circumstances.
However, the rapid emergence of various routers has led to fragmented evaluation practices and inconsistent metrics, making it difficult to systematically assess progress in this space. To address this problem, we need a comprehensive router comparison and a standardized leaderboard, similar to those available for models. In this work, we introduce RouterArena, the first open platform enabling comprehensive comparison of LLM routers. RouterArena has (1) a principally constructed dataset with broad knowledge domain coverage, (2) distinguishable difficulty levels for each domain, (3) an extensive list of evaluation metrics, and (4) an automated framework for evaluation and leaderboard updates. Leveraging this framework, we have produced the initial leaderboard with detailed metrics comparison.
Figure1 provides a preview of the leaderboard. The complete framework and the latest router leaderboard are publicly available at https://routeworks.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a benchmark for LLM Routers that evaluates different routing systems along multiple axes—including *accuracy*, *cost*, *latency*, and *routing robustness*. Concretely, the authors curate *8,400* queries from 21 open-source datasets and, via an LLM-as-Judge procedure using DeepSeek-V3.1, assign each query to one of three difficulty levels: easy, medium, or hard. On the collected dataset, the paper conducts objective evaluations of both open-source routers (e.g., KNN- and MLP-based methods) and commercial routers (e.g., GPT-5, Azure Router).

### Strengths
1. It is necessary and practically valuable to objectively evaluate LLM routers, as this helps users who are unfamiliar with model details select configurations that best fit their needs.
2. The benchmark combines 5 evaluation perspectives (including cost, accuracy, robustness, latency, and routing optimality). In addition, it supports commercial routers, making the evaluation more complete than prior work.

### Weaknesses
### Weaknesses

1. Query selection bias. The query set consists primarily of objective questions and excludes creative/open-ended tasks. For objective questions, users often prioritize accuracy over cost; however, for open-ended or batch-processing tasks (e.g., large-scale data filtering), cost can be the primary concern, leading to different routing preferences. This preference shift can further affect the validity of conclusions drawn in the Experiments section. It is advisable to include open-ended queries and update the results and analyses accordingly.

2. Difficulty grading misalignment with real-world scenarios. Difficulty is determined solely via an LLM-as-Judge setup, which may introduce model-induced bias; human annotation should be incorporated. Moreover, the benchmark should expand to more realistic routing demands, such as *long-context generation*, *tool use*, *code agents*, and *deep research*. In real deployments, achieving good performance often involves *orchestrating multiple models* collaboratively rather than invoking a single model.

3. Limited comparability due to heterogeneous model pools. Different routers operate over different model pools. A router that can access a cheaper yet strong model may rank higher, but this does not necessarily demonstrate a better routing algorithm. Introducing a unified model pool and a corresponding leaderboard would improve comparability.

4. Evaluation dimensions remain limited. Additional aspects—such as *response time* or *response length*—should be considered. For example, if the pool contains a cheaper *reasoning* model and a more expensive *instruction* model, then under the cost formulation
$ \text{cost} = C_{\text{in}} \cdot N_{\text{in}} + C_{\text{out}} \cdot N_{\text{out}} $,
two routing strategies might achieve the *same cost* and *same accuracy*, yet differ meaningfully in behavior; how should their relative quality be judged?

### Notation

1. *All displayed formulas lack numbering*, which reduces ease of reference and clarity in discussion.

### Questions
See above weaknesses

### Soundness
1

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
This paper introduces RouterArena, which the authors claim to be the first comprehensive open platform for evaluating and comparing large language model (LLM) routers. The platform aims to meet the growing need for systematic router evaluation and provides: (1) a principled dataset consisting of 8,400 queries spanning 9 domains and 44 categories, organized using the Dewey Decimal Classification system and incorporating Bloom’s taxonomy to define difficulty levels; (2) a comprehensive set of evaluation metrics covering accuracy, cost, routing optimality, robustness, and latency; and (3) an automated evaluation framework that supports both open-source and commercial routers. The authors evaluate 12 representative routers and present a multidimensional leaderboard summarizing their overall performance.

### Strengths
S1. The paper addresses a critical gap in the LLM ecosystem, the research motivation is clear and timely.
S2. The evaluation metrics are comprehensive, with five well-defined dimensions that reflect real-world deployment considerations.
S3. The automated framework enables dynamic leaderboard updates and supports the evaluation of both open-source and commercial routers.

### Weaknesses
W1. The overall contribution of the paper is limited.
W2. For commercial routers, internal routing decisions are inaccessible, making many metrics uncomputable and thus limiting a full evaluation.
W3. The use of DeepSeek-V3.1 for automated difficulty annotation may introduce systematic bias; no quantitative bias analysis or annotation-consistency study is provided.
W4. The robustness test is limited, and the actual evaluation method appears inconsistent with the definition of robustness given in Section 4.
W5. The paper spends substantial space describing the construction of a dataset with distinct difficulty levels but does not subsequently analyze results based on these levels.

### Questions
Q1. Regarding the validation of Bloom’s taxonomy classification—was any human sampling verification conducted, and how consistent were the results with the LLM’s judgments?
Q2. The robustness definition in Section 4 differs from the implementation described in Section 6.2 (“adding irrelevant keywords”). If only keyword-insertion was used, does this cause inconsistency between the definition and the actual test?
Q3. Figure 6 looks more like a scatter plot than a curve plot. Would the authors consider renaming it for accuracy?
Q4. The paper builds a dataset with three difficulty levels, but all evaluations are aggregated. Why not provide router performance broken down by difficulty level?
Q5. For commercial routers that cannot expose routing decisions, is there a plan to design alternative indicators to enhance ranking reliability?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces "RouterArena," a new open-source platform for benchmarking Large Language Model (LLM) routers. The authors argue that as the LLM ecosystem has produced numerous models with varying costs and capabilities, routers have become essential for selecting the right model for a given query. This, in turn, has created a new need for a standardized way to evaluate the routers themselves.


An automated framework and leaderboard designed to be "live," allowing researchers to submit new routers (both open-source and commercial) for evaluation and comparison.

### Strengths
1. The paper addresses a practical problem. As model routing becomes a standard component in AI stacks, the need for a robust, standardized benchmark to compare routers is high. This work is well-motivated and highly relevant to the community.

2. Principled Dataset Construction: A major strength of this paper is its novel and well-justified dataset construction.

3. Novelty: An automated, "live" platform and leaderboard, distinct from a static dataset. It is designed for continuous benchmarking and community engagement, allowing researchers to submit and compare new open-source and commercial routers.

4. Comprehensive, Multi-Dimensional Evaluation: The paper correctly identifies that router performance is multi-faceted. The inclusion of Routing Optimality is a key metric, as it reframes the goal from just "being correct" to "being correct efficiently." Measuring Robustness and Latency further strengthens the benchmark's utility for real-world applications.

5. Interesting Initial Analysis and Insights: The initial evaluation of 12 routers provides valuable insights. The findings—that commercial routers do not necessarily lead, that all routers are inefficient, and that performance is a complex trade-off—are important takeaways for the field.

### Weaknesses
Discussion of Benchmarking Philosophy: The paper positions itself as superior to prior work like RouterBench (Table 1), but it misses an opportunity for a deeper discussion. RouterArena evaluates live routers (a "hot" evaluation), whereas RouterBench uses a large, static dataset of pre-computed outcomes for "offline" evaluation. This offline approach is significantly cheaper and faster for iterating on router designs, presenting a different benchmarking philosophy. A more nuanced discussion of the pros and cons of these different approaches would improve the paper's contribution.

Dataset Scale: While the dataset is principled in its design, its size (~8,400 queries) is a potential limitation. When spread across 44 categories and 3 difficulty levels, some cross-sections may be too small to draw statistically significant conclusions. The paper would be stronger if it addressed this limitation or included an analysis of the sample size per category.

### Questions
See weaknesses.

### Soundness
2

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
The paper tackles a timely problem—standardized router evaluation—with a clear systemization (dataset + metrics + framework) and substantive empirical coverage (open-source and commercial).

### Strengths
S1. Well-scoped problem & gap analysis.

S2. Good dataset, metric, and evaluation design.

S3. Comprehensive experiments and visualizations.

### Weaknesses
W1. The $\log 2$ cost normalization with fixed ( $c_{min}⁡ = 0.0044$, $c_{max} ⁡ = 200$ ) and $\beta=0.1$ may bias rankings toward certain price bands; no sensitivity analysis shown in Sec. 5.

W2. LLM-as-judge labeling (DeepSeek-V3.1) lacks human validation studies or inter-rater checks.

### Questions
I think this is a very interesting topic. Specifically, I have the following two questions:

Q1. Your evaluation reports per-query accuracy as a binary outcome and aggregates it—without explicit weighting by Bloom difficulty levels—into the composite Arena score $S_{i,\beta}$. 
-  Is “completion” strictly 0 or 1 correctness per query? If so, why not support rubric-based partial credit for partially solved answers (e.g., correct plan but incomplete final step)?
- Some longer (higher-token) first-round answers can enable success in later turns. Do you plan a multi-turn setting that measures cross-round utility versus added first-round cost (e.g., a “round-2 success gain” vs. round-1 token spend)?

Q2. Among answers that are equally correct, a longer response may offer clearer reasoning, citations, or actionable steps, potentially increasing user satisfaction even at a higher token cost. Do you collect any user-satisfaction / explanation-quality signal (human Likert ratings or a calibrated LLM-as-judge) in addition to accuracy/cost?

### Soundness
3

### Presentation
3

### Contribution
3
