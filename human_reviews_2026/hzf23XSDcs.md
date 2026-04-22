# CitySeeker: How Do VLMs Explore Embodied Urban Navigation with Implicit Human Needs?

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Vision-Language Models (VLMs) have made significant progress in explicit instruction-based navigation; however, their ability to interpret implicit human needs (e.g., ''I am thirsty'') in dynamic urban environments remains underexplored. This paper introduces CitySeeker, a novel benchmark designed to assess VLMs’ spatial reasoning and decision-making capabilities for exploring embodied urban navigation to address implicit needs. CitySeeker includes 6,440 trajectories across 8 cities, capturing diverse visual characteristics and implicit needs in 7 goal-driven scenarios. Extensive experiments reveal that even top-performing models (e.g., Qwen2.5-VL-32B-Instruct) achieve only 21.1% task completion. We find key bottlenecks in error accumulation in long-horizon reasoning, inadequate spatial cognition, and deficient experiential recall. To further analyze them, we investigate a series of exploratory strategies—Backtracking Mechanisms, Enriching Spatial Cognition, and Memory-Based Retrieval (BCR), inspired by human cognitive mapping's emphasis on iterative observation-reasoning cycles and adaptive path optimization. Our analysis provides actionable insights for developing VLMs with robust spatial intelligence required for tackling ''last-mile'' navigation challenges.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces *CitySeeker*, a novel benchmark for evaluating Vision-Language Models (VLMs) in embodied urban navigation driven by implicit human needs. Unlike prior VLN benchmarks that rely on explicit instructions, *CitySeeker* focuses on abstract, functional, and semantic goals (e.g., “I’m thirsty”), spanning 6,440 trajectories across 8 cities. The authors propose a framework for evaluating spatial reasoning and decision-making and identify key bottlenecks in current VLMs. They further introduce three human-inspired strategies—Backtracking, Spatial Cognition Enrichment, and Memory-Based Retrieval (BCR)—to enhance navigation performance.

### Strengths
- **Benchmark Contribution**: *CitySeeker* fills a critical gap in VLN research by targeting implicit human needs in dynamic, real-world urban environments. It is the first large-scale benchmark to do so across multiple cities and diverse task categories. 
- **Realism and Diversity**:  The benchmark includes diverse urban layouts and visual characteristics, making it highly relevant for embodied AI applications. 
-  Comprehensive Evaluation： 
- **Comprehensive Evaluation**: The paper presents extensive empirical results across 27 VLMs, including proprietary and open-source models. The analysis spans task categories, cities, and trajectory patterns. The authors also  proposed BCR strategies are well-motivated and show measurable improvements in task completion and path efficiency.
- **Clarity and Presentation**:  The paper is generally well-written, with a clear logical flow and a well-motivated problem statement.

### Weaknesses
1.  **Problem Setup**:  

    - The task formulation leans more toward object navigation than traditional VLN. In partially observable environments, defining the agent’s state solely as the current observation $o_t$ is insufficient.
    - A more appropriate formulation would involve belief state estimation using historical observations $\{o_0, o_1,...,o_t\}$, which is a regular practice in POMDP-based problem.

2. **Symbol and Notation Clarity**:

    - The definition of $v$ as a graph node and $v_t$ as the location is inconsistent.
    - Subscripts such as $o_i$ (different views at a single node) and $o_t$ (timesteps) are used ambiguously.
    - Lacks a clear explanation of how reasoning $\phi$ and action $a$ are computed -- what inputs are used ($O_t$ or $S_t$ or something else?).

4. **Evaluation Metrics Misalignment**:

    - Metrics like **nTCE**, which measure trajectory deviation from ground truth, are more suitable for instruction-following tasks (in previous VLN tasks, the agents are asked to follow the excact GT trajectory given a detailed user instruction).
    - In implicit-need-driven navigation, multiple valid trajectories may exist. Metrics like SPL (Success weighted by Path Length) or goal proximity might be more appropriate.


5. **Low Task Completion Rates**:

    -  Even the best-performing models and human baselines achieve low success rates under strict metrics (e.g., 5.7% TCE for humans), raising concerns about benchmark difficulty or metric suitability.

### Questions
**Questions**: See weakness.

### Soundness
3

### Presentation
4

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
As the ability of vision-language models (VLMs) to interpret *implicit human needs* in dynamic urban environments remains underexplored, this paper proposes CitySeeker, a novel benchmark designed to evaluate VLMs’ spatial reasoning and decision-making capabilities in *embodied urban navigation* tasks that involve implicit objectives. CitySeeker includes 6,440 trajectories across 8 cities, capturing diverse visual characteristics and implicit needs within 7 goal-driven scenarios. Extensive experiments demonstrate that even top-performing models (e.g., Qwen2.5-VL-32B-Instruct) achieve only 21.1% task completion, revealing fundamental weaknesses in long-horizon reasoning, spatial cognition, and experiential recall. To further diagnose these issues, the authors explore a set of human-inspired strategies—Backtracking Mechanisms, Enriched Spatial Cognition, and Memory-Based Retrieval (BCR)—reflecting iterative observation-reasoning cycles and adaptive path optimization in human navigation. The analysis provides valuable insights into developing VLMs with more robust *spatial intelligence* for tackling “last-mile” navigation challenges.

### Strengths
1. The motivation of this paper is well-grounded. Identifying that VLMs’ ability to interpret implicit human needs in dynamic urban environments remains underexplored is both timely and significant. It provides a new angle for examining VLMs’ world knowledge and decision-making capabilities.

2. The paper is clearly written and visually appealing. The figures effectively illustrate the framework and experiments, helping readers understand the design and reasoning process.

3. The authors conduct extensive experiments involving 27 different VLMs, and their findings are deep and thought-provoking, revealing critical gaps in current models’ embodied spatial reasoning.

### Weaknesses
1. Overall, this is a clear accept-level paper in terms of novelty, clarity, and experimental depth. However, there is a **serious ethics concern**. The paper states:
   *“CitySeeker dataset was sourced from publicly available APIs (Google Maps and Baidu Maps) and is used in accordance with their terms of service for non-commercial research purposes only.”*
   After reviewing Google Maps’ Terms of Service[1], it explicitly states:
   **“Downloading Street View images to use separately from Google services (such as an offline copy) is prohibited. These restrictions apply to all academic, nonprofit, and commercial projects.”**
   This implies that **the dataset collection may violate Google’s ToS**, introducing a significant **ethical and legal issue**. Consequently, the dataset **cannot be publicly released**, which severely limits the reproducibility and extensibility of this research. This is a **veto-level weakness** for a top-tier venue.

[1] [Brand Resource Center | Products and Services - Geo Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/)

### Questions
1. Please clarify the ethical compliance issue.

2. The paper claims that these navigation tasks are highly challenging even for humans (e.g., “humans achieve only 5.7% accuracy”). However, tasks such as finding a restaurant or locating a place with Wi-Fi do not seem inherently difficult for human participants. The paper does not adequately explain why humans perform so poorly, or how the tasks are defined, constrained, or evaluated. This undermines the interpretability and credibility of the benchmark’s human baseline.

### Soundness
4

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
This paper introduces CitySeeker, a framework that leverages VLMs to understand and predict human urban mobility behaviors based on natural-language queries. The system maps user needs to POI categories using multimodal embeddings and evaluates performance on multiple cities. The goal is to test VLMs’ ability to align semantic and spatial reasoning in real-world city environments.

### Strengths
Addresses an interesting interdisciplinary question.

The dataset construction across multiple major cities, combining geospatial and textual information.

The evaluation is systematic, covering both semantic matching and spatial reasoning tasks.

The idea of connecting natural language intent to spatial decision-making is novel and potentially impactful.

### Weaknesses
The mapping from need to POI type assumes a fixed, deterministic relationship, which may not hold in practice, human intent is subjective and context-dependent.

The model implicitly assumes people choose the shortest path or most direct POI option, which is unrealistic, behavioral factors like preference, familiarity, and accessibility play major roles.

Cross-cultural generalization is a concern: the same “need” may imply different POIs across societies.

The paper lacks an analysis of cultural and linguistic bias, despite using mixed data from Beijing and New York.

The need-to-POI mapping seems to reflect designer bias rather than emergent patterns from real user behavior.

No mention of whether the system can adapt to multi-intent or ambiguous queries.

It’s unclear whether the evaluation reflects real mobility choices or just semantic alignment accuracy.

### Questions
How is user intent variability modeled, are there multiple valid POIs for the same need, or just one ground truth?

How does the model handle ambiguous or multi-intent queries?

Is the need-POI mapping empirically validated with real mobility data (e.g., GPS traces, check-ins)?

Could the framework integrate behavioral priors (e.g., time-of-day, personal preferences) to better capture real-world decision patterns?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces CitySeeker, a comprehensive benchmark designed to evaluate the spatial reasoning and decision-making capabilities of vision-language models (VLMs) in the context of embodied urban navigation for addressing implicit user needs. Extensive evaluations across a wide range of VLMs reveal key limitations in long-horizon reasoning, and the authors propose effective strategies to improve model performance.

### Strengths
- Clear and Well-Structured: The paper is well-organized, with thorough explanations of the data collection process, benchmark design, and task formulation.

- Novel and Interesting Setting: The paper proposed the task of embodied urban navigation guided by implicit human needs. This task is currently not widely explored and has significant potential for real-world deployment.

- Extensive Evaluations: A wide range of VLMs are evaluated on the curated benchmarks, accompanied by comprehensive analysis and discussion.

- Actionable Insights: The authors propose concrete strategies to enhance VLM performance, offering insights for real-world deployment scenarios.

### Weaknesses
I don't find significant weaknesses in this submission. However, I do have some concerns as follows. Therefore, I give a conservative score of borderline accept. I may consider increasing the rating if the authors adequately address these concerns.

- Open-Source Model Superiority: The paper observes that open-source VLMs (such as Qwen) occasionally outperform the proprietary VLMs. The submission would benefit from a deeper analysis of the underlying reasons behind this phenomenon.

- Presentation Issue: Some figures in the submission (e.g., Figures 5 and 6) require further refinement to improve readability and visual clarity.

- The manuscript would benefit from including illustrations of typical failure cases.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
4
