# ChinaTravel: An Open-Ended Travel Planning Benchmark with Compositional Constraint Validation for Language Agents

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 8, 2, 6

## Abstract
Travel planning stands out among real-world applications of \emph{Language Agents} because it couples significant practical demand with a rigorous constraint-satisfaction challenge. However, existing benchmarks primarily operate on a slot-filling paradigm, restricting agents to synthetic queries with pre-defined constraint menus, which fails to capture the open-ended nature of natural language interaction, where user requirements are compositional, diverse, and often implicitly expressed. To address this gap, we introduce \emph{ChinaTravel}, with four key contributions: 1) a practical sandbox aligned with the multi-day, multi-POI travel planning, 2) a compositionally generalizable domain-specific language (DSL) for scalable evaluation, covering feasibility, constraint satisfaction, and preference comparison 3) an open-ended dataset that integrates diverse travel requirements and implicit intent from 1154 human participants, and 4) fine-grained analysis reveal the potential of neuro-symbolic agents in travel planning, achieving a 37.0\% constraint satisfaction rate on human queries, a 10$\times$ improvement over purely neural models, yet highlighting significant challenges in compositional generalization. Overall, ChinaTravel provides a foundation for advancing language agents through compositional constraint validation in complex, real-world planning scenarios. Project Page: https://www.lamda.nju.edu.cn/shaojj/ChinaTravel/index.html

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ChinaTravel, a new benchmark designed to evaluate language agents on realistic, multi-day, multi-POI (point-of-interest) travel planning tasks within Chinese cities. The benchmark integrates a realistic sandbox environment, a domain-specific language for compositional constraint validation. The authors further explore Neuro-Symbolic planning methods, combining LLM-based understanding with symbolic reasoning for constraint satisfaction. Empirical studies show that NeSy agents outperform pure LLM-based methods in meeting logical and environmental constraints.

### Strengths
1. This paper proposes a domain-specific language that programmatically composes atomic concepts of travel attributes.
2. This paper makes a meaningful attempt to evaluate Neuro-Symbolic Agents for real-world travel planning tasks, illustrating the potential.

### Weaknesses
1. The claimed contributions concerning the Sandbox and Open-Ended Travel Dataset are not sufficiently distinct from existing approaches. The work does not clearly demonstrate a research gap relative to previous studies.
2. The authors propose a contextual grounding task to capture implicit user intent. While this is both practical and interesting, the benchmark primarily serves to identify such phenomena rather than offering effective solutions or promising directions to address them. It would be more insightful if the paper included a deeper analysis or exploration of potential approaches to tackle this issue.

### Questions
1. For GPT-4o, why does the NeSy Planning with Oracle Translation method lead to a significant performance decrease compared to the vanilla NeSy Planning approach?
2. Would the same DSL and sandbox design generalize to non-Chinese or multilingual settings?

### Soundness
2

### Presentation
2

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
This paper presents ChinaTravel, a new benchmark that addresses a critical and increasingly obvious gap in language agent research.
They argue that current benchmarks adopt synthetic queries and have limited constraints.
Thus, they provide a more diverse-formatted and a complicated sandbox.
They find that near-symbolic planning perform better than the LLM agents.

### Strengths
1. The benchmark extends the diversity and complexity of existing benchmarks.
2. The paper adopts DSL to evaluate the outputs of the LLMs.

### Weaknesses
1. The conclusion that the symbolic methods perform better than the LLM agents is obvious. I think the reason of proposing such benchmarks should be testing LLM agents to plan with constraints. We can definitely foresee a greedy search could complete such tasks. Yet, with such external help, the performance of LLM agent itself does not improve.
2. The DSL is a powerful contribution for validating compositional logical constraints. However, real-world travel planning is filled with subjective, non-functional requirements that are difficult to verify, for example, the user may ask for a chill journey.

### Questions
1. The paper identifies that the primary bottleneck in the NeSy is the initial NL to DSL translation. Is there any solution for such problem given you are using existing neuro-symbolic methods?

### Soundness
3

### Presentation
3

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
This paper constructs a dataset for the travel planning problem to address the limitations of existing benchmarks, which typically rely on synthetic queries with limited constraints and explicit intent, diverging from real-world scenarios.

### Strengths
1. The paper is well-written and clearly communicates its key contributions.
2. The constructed dataset is comprehensive and appears to be thoughtfully designed.

### Weaknesses
I do not have significant concerns about the technical aspects of the paper. The following are not weaknesses, but some thoughts shared with the authors. 

Travel planning is inherently complex, making it genuinely useful for people involves challenges beyond optimization or scheduling. If we view travel planning purely as a multi-day, multi-point-of-interest (POI) planning problem, as the paper does, the formulation seems reasonable. However, from a broader perspective, I am not convinced that the proposed dataset truly reflects how people make travel decisions, or that strong model performance on this dataset necessarily implies alignment with real user preferences.

In practice, travel planning goes far beyond cost and time optimization. Human preferences are deeply subjective and influenced by social and contextual factors. For example, people often seek advice from friends who have visited a destination, or they explore social media (e.g., Instagram, TikTok) to find inspiration from influencers or acquaintances. These behaviors are difficult to capture in a dataset that models travel purely as a structured multi-step planning task.

Therefore, I recommend that the authors consider reframing their work not strictly as a travel planning problem, but more generally as a multi-step or long-horizon planning task. This broader framing could make the work more generalizable and align it with recent efforts in benchmarking agentic reasoning and planning capabilities. The following papers might be relevant references (I am not the author of any paper that I listed):

References

[1] WideSearch: Benchmarking Agentic Broad Info-Seeking

[2] BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent

### Questions
Check the weaknesses section for more details.

### Soundness
3

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
4

### Summary
The paper presents ChinaTravel, a benchmark for realistic multi-day travel itinerary planning with open-ended user requirements and compositional constraints. It introduces a domain-specific language (DSL) for formally encoding user constraints and preferences, enabling automatic validation of itinerary feasibility. The benchmark includes real travel data for 10 Chinese cities and 1,154 authentic human queries annotated with executable DSL code. Experiments show that pure LLM-based agents perform poorly on complex constraint satisfaction, while neuro-symbolic methods that combine LLM interpretation with structured planning achieve higher success. Despite these gains, overall performance remains low, indicating the benchmark’s difficulty and the need for more robust compositional reasoning in travel planning agents.

### Strengths
- Addresses key gaps in prior travel-planning benchmarks by focusing on realistic and multi-day intra-city trips with authentic human queries. Covers diverse implicit constraints and overcomes prior limitations like synthetic prompts and fixed rules. Strongly establishes novelty and real-world relevance.
- Introduces a well-designed Python-like DSL that enables expressive and composable user constraints. Allows logical, arithmetic, and relational conditions to be defined and verified automatically.
- Defines rigorous metrics and ensures queries have feasible solutions. The C-LPR metric and per-query vs. per-constraint evaluation demonstrate careful empirical design. Methodology ensures credible and fine-grained performance assessment.
- Benchmarks a wide range of methods across multiple models. Results show NeSy methods outperform others. Thorough failure analysis adds diagnostic insight.

### Weaknesses
- A potential concern is that the benchmark is tied to China-specific travel data. Agents might overfit to domain idiosyncrasies or to the Chinese language expressions of requirements. The authors do mitigate this by providing English translations of all information for international researchers, but it’s not fully clear how an English-only LLM would perform if the underlying database is Chinese. Moreover, the DSL’s set of primitive concepts is tailored to travel. It’s not obvious how easily this framework would extend to other planning domains, say, event scheduling or travel in countries without the same data availability. The reliance on a predefined set of attributes means truly novel user constraints outside those attributes still cannot be captured without augmenting the DSL, e.g. “scenic beauty of train route” which is not a stored attribute.
- The neuro-symbolic pipeline is effective but complex and brittle, with multiple interdependent stages prone to cascading failures. Each stage can introduce failure points: e.g., an error in DSL translation can completely derail the planner. Indeed, the results show a sizable drop from an “oracle” DSL setting to the normal setting, indicating the pipeline’s success is heavily dependent on the NL-to-DSL conversion quality. It means the proposed solution might not scale gracefully to even more complex queries or to smaller and less capable LLMs. In essence, the current neuro-symbolic method feels like a very involved engineered solution.
- The use of micro vs. macro scores for five different rates resulted in dense tables that were a bit hard to parse on first read. For instance, Table 2 presents many numbers for each method, and it wasn’t immediately obvious to a reader what “macro LPR = 0” signifies without reading the fine print. The paper might benefit from a brief explanation in the text of how micro vs macro are calculated (the appendix defines it, but a sentence in main text would help guide the reader).
- Additionally, the treatment of soft preferences in the evaluation is somewhat under-emphasized. Preferences are encoded as optimization objectives in the DSL, but the main results focus on binary pass/fail of constraints. It appears that preference satisfaction was analyzed separately rather than as part of the overall success metric. This separation makes sense since preferences aren’t hard requirements, but it leaves open questions: for example, if one method produces a plan that satisfies constraints but with suboptimal preferences, e.g., it visits fewer attractions than possible, how is that reflected?
- Planning remains computationally intensive; even top methods require multi-minute searches and repeated LLM calls. MILP and large-context approaches become intractable for complex itineraries. The paper could discuss strategies to handle combinatorial scaling and efficiency improvements more explicitly.

### Questions
- The DSL-based evaluation is powerful, but it relies on a fixed library of concept functions. How easily can this DSL be extended to accommodate new types of constraints or domains? For example, if a user asks for a constraint involving a notion not currently encoded, say, “scenic rating of a route” or “avoid areas with high COVID-19 cases”, would adding such a constraint simply be a matter of defining a new attribute function in the DSL and updating the database?
 - Have the authors considered training a dedicated model for this *NL2DSL translation* task? For instance, using the large set of synthesized queries (Stage II) with their DSL annotations as training data to fine-tune a transformer that directly outputs DSL code could potentially improve accuracy and consistency. It could also avoid the iterative Reflexion loop that sometimes prunes constraints. If this was attempted or ruled out, could the authors elaborate on the challenges?
 - How exactly are soft preferences evaluated in ChinaTravel’s scoring?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ChinaTravel, an open-ended multi-day travel planning benchmark designed for evaluating language agents under real-world, compositional constraint scenarios.  It introduces a domain-specific language (DSL) for formalizing diverse logical and preference-based requirements, alongside a large-scale dataset combining human-authored and LLM-generated queries. Extensive experiments highlight the advantages of neuro-symbolic methods in constraint satisfaction and preference reasoning over pure LLM baselines. However, the authors also mention that challenges persist in semantic grounding and compositional generalization, where even state-of-the-art models like GPT-4o and DeepSeek-V3 show limited performance.

### Strengths
1. The proposed ChinaTravel benchmark is solid and realistic, encompassing diverse travel scenarios, multi-day itineraries, and compositional constraints that closely align with real-world planning needs. It effectively addresses several limitations of previous benchmarks.
2. The experiments are comprehensive and well-designed, covering a wide range of models, evaluation metrics, and analytical perspectives.
3. The paper not only introduces a new benchmark but also proposes an integrated methodology that combines large language models with traditional neuro-symbolic reasoning, pointing toward a promising direction for advancing constraint-aware planning.
4. The analysis is detailed and insightful, offering in-depth examinations of model behavior, constraint satisfaction, and preference reasoning.

Overall, this work presents a well-rounded and impactful contribution, bridging benchmark construction, methodological innovation, and empirical understanding in the field of language-agent-based planning.

### Weaknesses
1. This work is solid and well-executed, featuring a carefully designed benchmark, strong experimental methodology, and comprehensive analysis. However, since it largely builds upon prior innovations rather than introducing a paradigm shift for the field, I cannot assign it the highest rating.
2. While the authors mention introducing up to 12 constraints per query, I wonder whether real users would naturally provide such detailed inputs when first interacting with a “travel assistant.” In most real-world scenarios, users tend to refine their requirements through multi-turn interactions, and incorporating such a dialogue-based setting could make the benchmark even more realistic and insightful.
3. Another concern is that the authors did not include large reasoning models in their experiments, which are expected to outperform general-purpose LLMs on complex planning and constraint-satisfaction tasks.

### Questions
n/a

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The paper introduces ChinaTravel, a realistic Chinese-language benchmark for multi-day, multi-POI travel planning grounded in real human use cases.

- It provides a compositional DSL and neuro-symbolic framework to express and verify logical and preference-based travel constraints.

- The benchmark enables evaluation of LLMs and hybrid reasoning models across feasibility, constraint satisfaction, and preference comparison.

### Strengths
- The paper builds a realistic and complex dataset grounded in real human travel use cases, making it well aligned with real-world scenarios.

- The authors provide a Chinese-language benchmark, offering a valuable resource beyond existing English-centric datasets.

- The benchmark enables multi-constraint, compositional planning that captures the true complexity of real travel tasks.

- The study offers comprehensive evaluation with diverse models, baselines, and metrics.

### Weaknesses
**W1. Language diversity**

* The dataset is **Chinese-only**. While introducing a non-English benchmark is valuable, including English or multilingual settings would improve usability and comparability with other benchmarks.

**W2. Limited novelty**

* Although the dataset moves closer to real-world settings, the contribution feels **incremental** — mainly increasing constraint complexity within existing benchmark scopes, without introducing fundamentally new ideas or designs.

**W3. Lack of comparison with existing benchmarks**

* While the paper claims that LLMs struggle in ChinaTravel, it does not compare performance on prior datasets (e.g., TravelPlanner, TripPlanning).
Although direct comparison is difficult due to language differences, the authors could still replicate the settings of existing benchmarks to provide a contextual baseline.
  Such analysis would strengthen the benchmark’s validity.

**W4. Limited evaluation depth**

* The evaluation section lacks a detailed analysis of *why* models fail.
  Studying factors such as the **number and composition of constraints** or their combinations could provide deeper insights into agent behavior.

**W5. Preference metric design**

* The current preference evaluation (e.g., Fig. 8) treats preferences as independent objectives.
  In real-world scenarios, multiple preferences coexist and may conflict with hard constraints.
  Therefore, a more holistic metric would be considered.

### Questions
- (minor) The citation style at L 41 needs to be corrected.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 7

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ChinaTravel, a new benchmark for evaluating language agents in complex, real-world travel planning. The authors argue that existing benchmarks are overly synthetic and lack realistic, open-ended human queries. ChinaTravel contributes: (1) a sandbox environment for multi-day, multi-POI planning in Chinese cities; (2) a domain-specific language (DSL) for compositional constraint specification and automated validation ; and (3) an open-ended dataset derived from 1154 human participants, featuring implicit intent and novel constraint compositions. Experiments show that pure LLM agents completely fail , and while neuro-symbolic methods perform better, significant challenges in contextual grounding and compositional generalization remain.

### Strengths
- The benchmark is novel and well-motivated. The paper clearly articulates the limitations of prior benchmarks like TravelPlanner. 

- The benchmark includes 154 human-validated queries and 1,000 survey-collected queries reflecting real-world travel requirements with implicit expressions 

- The experiments convincingly show that ChinaTravel is a challenging benchmark. Pure LLM methods (ReAct, Act-only) completely fail, achieving near-zero Environmental Pass Rates (EPR) and Final Pass Rates (FPR). This confirms the task is beyond the reach of current text-wise planning approaches.

### Weaknesses
- It is unclear how much of the performance gain (e.g., 37.0% C-LPR in Table 3)  comes from the (1) iterative NL2DSL translation, (2) the symbolic search sketch, or (3) the LLM-driven POI ranking within the search. The ablation study in Sec 4.3 only explores preference ranking (PEQ vs. PDS) and doesn't dissect the core "NeSy Planning" search algorithm itself.

- Inter-annotator agreement is not reported for DSL annotation process. With five annotators performing initial revision and three developers conducting verification, consistency metrics (Cohen's kappa, Fleiss' kappa) would strengthen quality claims.

- Temporal coverage is unclear: Were queries collected during specific seasons? Travel requirements vary seasonally (festivals, weather, peak/off-peak periods), but dataset doesn't indicate collection timeframe or seasonal distribution.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
