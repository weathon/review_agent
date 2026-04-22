# TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 4

## Abstract
Travel planning is a valuable yet complex task that poses significant challenges even for advanced large language models (LLMs). However, existing benchmarks primarily equate planning ability with solving rigid constraint satisfaction problems. Solvers that excel at synthetic logic puzzles often fail to handle the ambiguity of real-world user intents. To address this, we present TripScore, a behavior-grounded benchmark and evaluation framework designed to align agent development with real-world utility. We release a large-scale dataset of 4,870 queries including 219 real-world, free-form requests for generalization to authentic user intent. We propose a unified evaluation reward that fuses feasibility and quality into a granular scalar reward. Our evaluator achieves moderate agreement with travel-expert annotations (60.75%) and outperforms multiple LLM-as-judge baselines. Leveraging TripScore, we conduct extensive experiments across diverse paradigms, including neuro-symbolic solvers, test-time search and fine-tuning. Our results reveal that while rigid solvers flounder on real-world queries, RL fine-tuning (e.g., GRPO) utilizing our unified reward significantly outperforms other methods with the same base model, effectively bridging the gap between open-source models and proprietary baselines in authentic travel planning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new benchmark, TripScore, designed to evaluate the travel planning capabilities of large language models (LLMs). The authors also compare different methods, including standard LLMs, supervised fine-tuned LLMs, GRPO-based models, and neuro-symbolic approaches. While the topic is relevant and the implementation appears complete, the novelty and overall contribution of the paper are quite limited.

Major Comments:
1. Limited novelty of the benchmark.
As acknowledged by the authors, there already exist several travel-planning-related benchmarks and datasets. The incremental contribution of TripScore over these existing resources is unclear. The paper should clearly justify why constructing yet another benchmark is necessary and what unique aspects it provides beyond existing ones. Table 1 is not enough to demonstrate the necessary of this benchmark.
2. Lack of clarity in requirement collection.
The authors mention that task requirements were collected from human users. However, there is insufficient information about the background, diversity, and quality control of the participants. Without this information, it is difficult to assess the reliability and representativeness of the benchmark queries.
3. Unclear benchmark difficulty and comparative validation.
The difficulty level of the benchmark tasks is not analyzed or compared with existing datasets. It would strengthen the paper if the authors could show how the same methods perform on other travel-planning datasets to contextualize TripScore’s difficulty.
4. High similarity to existing work.
The paper closely resembles prior studies in both concept and presentation. For instance, the constraints discussed are similar to those in ChinaTravel. Additionally, Figure 1 in this paper appears visually and structurally similar to Figure 1 in TravelPlanners, raising concerns about originality in presentation.
5. Lack of experimental insights.
While several models are compared, the experimental section does not yield clear insights or actionable conclusions. The paper would benefit from deeper analysis—for example, discussing why certain methods perform better, and what the benchmark reveals about LLM reasoning or planning limitations.

### Strengths
This paper proposes a new benchmark, TripScore, designed to evaluate the travel planning capabilities of large language models (LLMs). The authors also compare different methods, including standard LLMs, supervised fine-tuned LLMs, GRPO-based models, and neuro-symbolic approaches.

### Weaknesses
1. Limited novelty of the benchmark.
As acknowledged by the authors, there already exist several travel-planning-related benchmarks and datasets. The incremental contribution of TripScore over these existing resources is unclear. The paper should clearly justify why constructing yet another benchmark is necessary and what unique aspects it provides beyond existing ones. Table 1 is not enough to demonstrate the necessary of this benchmark.
2. Lack of clarity in requirement collection.
The authors mention that task requirements were collected from human users. However, there is insufficient information about the background, diversity, and quality control of the participants. Without this information, it is difficult to assess the reliability and representativeness of the benchmark queries.
3. Unclear benchmark difficulty and comparative validation.
The difficulty level of the benchmark tasks is not analyzed or compared with existing datasets. It would strengthen the paper if the authors could show how the same methods perform on other travel-planning datasets to contextualize TripScore’s difficulty.
4. High similarity to existing work.
The paper closely resembles prior studies in both concept and presentation. For instance, the constraints discussed are similar to those in ChinaTravel. Additionally, Figure 1 in this paper appears visually and structurally similar to Figure 1 in TravelPlanners, raising concerns about originality in presentation.
5. Lack of experimental insights.
While several models are compared, the experimental section does not yield clear insights or actionable conclusions. The paper would benefit from deeper analysis—for example, discussing why certain methods perform better, and what the benchmark reveals about LLM reasoning or planning limitations.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a comprehensive benchmark designed to evaluate and improve the complex travel-planning capabilities of LLMs.
It adopts real-world user queries and constructs a comprehensive evaluation framework.
The extensive experiments show that the designed reward is beneficial in RL training.

### Strengths
1. This paper adopts real-world user queries, bridging the gap between previous benchmarks and applications.
2. It proposes a unified and actionable reward score, which can be further used as  a reward model for such tasks and provides deeper insights beyond pass/fail.
3. This paper conducts extensive experiments on different methods and base models.

### Weaknesses
1. Although the authors claim the dataset to be a real-world dataset, it only contains 219 real-world queries out of 4870 total queries, which is not convincing enough for me to view this dataset as a real-world dataset.
2. The evaluation contains LLM-as-judge and the results in Table 6 show that only 61.32% is correctly evaluated, which poses doubt on the precision of such method.
3. This paper is an incremental extension of TravelPlanner by extending constraints and constructing a trivial evaluation method.

### Questions
1. How is TripScore's unified reward a fundamental advance over these existing, complex reward and evaluation models?
2. The TripScore evaluator relies on an LLM to score soft and preference constraints. The authors admit this model only achieves "moderate agreement (60.75%)" with human travel experts. This implies that for nearly 40% of comparisons, the benchmark's ground truth may be wrong. Why should this 60.75%-accurate reward signal be trusted to train a superior model, and how does this level of noise affect the stability and reliability of the GRPO training?
3. What is the fundamental difference between TripScore and existing benchmarks?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses a critical gap in travel planning benchmarks by unifying fine-grained evaluation criteria into a single reward and incorporating real-world user queries. The 4,870-query dataset (including 219 real-world requests) and four-category constraint framework (format, commonsense, soft, preference) are valuable contributions, filling the void of authentic scenario coverage in existing benchmarks.

### Strengths
1. Addresses a key limitation of existing travel planning benchmarks (e.g., TravelPlanner, ChinaTravel) by unifying four types of fine-grained constraints into a single interpretable reward score, providing a more coherent and scalable evaluation mechanism.

2. Constructs a high-quality real-world dataset, mitigating the overreliance on LLM-generated data in prior benchmarks and enhancing generalization and practical applicability.

3. Extensive experiments with diverse algorithms demonstrate the effectiveness of reinforcement learning for travel plan generation.

### Weaknesses
1. The framework relies on LLM-based soft or preference evaluation, which introduces potential inaccuracy and computational overhead due to the absence of a purely rule-based alternative.

2. Given the proliferation of travel planning benchmarks, including ChinaTravel, the novelty appears somewhat limited in terms of benchmark design.

### Questions
Q1: Why do strict hard constraints often lead to no feasible solutions in NeSy-based approaches to travel plan generation?

Q2: What are the benefits of using a unified evaluation metric in this context?

Q3: Compared with ChinaTravel, which focuses on NeSy reasoning, what type of approach do you personally find more promising for advancing travel planning tasks, and what are your key insights?

Q4: Why was the ReAct baseline not included in the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers the travel planning, proposes TripScore, and a unified reward for evaluating multi-constraint travel itineraries. This reward function integrates the rule-based evaluation and LLM-as-judge. Experiments compare direct prompting, test-time reasoning, neuro-symbolic approaches, and fine-tuning including GRPO, reporting gains in delivery rate, commonsense pass rate, and the unified reward, with analyses of error types and trip-duration sensitivity.

### Strengths
1. The problem is practically relevant and the authors implement a comprehensive evaluation workflow that attempts to separate hard feasibility from softer quality criteria. 
2. The paper provides careful engineering details, ablations over trip duration, error breakdowns, and an expert-annotation study to partially validate the reward.

### Weaknesses
1. The reliability of the unified reward is limited for its intended purpose. Agreement with human experts is only 60.75%. Because the reward is heavily shaped by the gating and by narrow penalty ranges, the final score collapses much of the variation into a few bands that correlate strongly with format and commonsense feasibility, undermining its claim to measure fine-grained quality beyond validity. From my perspective, a unified reward design is a methodological requirement rather than a benchmark requirement. However, the RL algorithm based on this reward proposed by the authors does not seem to have achieved a significant improvement compared to training-free algorithms.  
2. In summary, from a product development perspective, if the proposed reward design ultimately yields an effective RL solution, then I would consider this work to be more beneficial than TravelPlanner. However, from Benchamrk's current perspective, the incremental benefits of this work compared to TravelPlanner are limited.  
3. The use of charts is too similar to that of TravelPlanner, especially Figure 1, which is almost identical. Table 2 and Figure 4 are also very similar in design. This level of similarity necessitates special attribution to the figure as originating from TravelPlanner.

### Questions
1. Compared to TravelPlanner, what are the core contributions of this article? 
2. What impact does the author believe the presented unified reward will have on the travel planning community?
1. How do you address the training–evaluation mismatch that risks Goodhart’s law, given RL/RFT use only the rules-based reward while the test metric includes LLM-scored components?

### Soundness
2

### Presentation
2

### Contribution
2
