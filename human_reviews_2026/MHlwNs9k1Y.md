# PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Achieving political consensus is crucial yet challenging for the effective functioning of social governance. However, although frontier AI systems represented by large language models (LLMs) have developed rapidly in recent years, their capabilities in this scope are still understudied. In this paper, we introduce PoliCon, a novel benchmark constructed from 2,225 high-quality deliberation records of the European Parliament over 13 years, ranging from 2009 to 2022, to evaluate the ability of LLMs to draft consensus resolutions based on divergent party positions under varying collective decision-making contexts and political requirements. Specifically, PoliCon incorporates four factors to build each task environment for finding different political consensus: specific political issues, political goals, participating parties, and power structures based on seat distribution. We also developed an evaluation framework based on social choice theory for PoliCon, which simulates the real voting outcomes of different political parties to assess whether LLM-generated resolutions meet the requirements of the predetermined political consensus. Our experimental results demonstrate that even state-of-the-art models remain undersatisfied with complex tasks like passing resolutions by a two-thirds majority and addressing security issues, while uncovering their inherent partisan biases and revealing some behaviors LLMs show to achieve the consensus, such as prioritizing the stance of the dominant party instead of uniting smaller parties, which highlights PoliCon's promise as an effective platform for studying LLMs' ability to promote political consensus. The code and dataset are released at [PoliCon Website](https://zowiezhang.github.io/projects/PoliCon).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a benchmark to evaluate whether LLMs can draft consensus resolutions under realistic political-committee constraints. The dataset is built from European Parliament records (2009–2022) and, per the paper, comprises 2,225 cleaned issues, each with (issue, topic, debates, resolution, votes). Each evaluation scenario specifies: (1) a topic (5 coarse / 19 fine categories), (2) a political goal (passing a resolution with different voting rules; Rawlsianism; Utilitarianism), (3) participating parties, and (4) a power structure (randomized seat shares and, optionally, a veto party).
Six LLMs are benchmarked. The benchmarked models often succeed in simple-majority settings but struggle with two-thirds, veto, security topics, and Rawls. They also exhibit partisan biases aligned with real EP voting patterns. The authors claim the judge correlates with “ground-truth” party votes with Pearson 0.83, and >72% of simulations fall within 2 SD of ground truth.

### Strengths
1. Evaluating LLMs on stakeholder consensus formation (with explicit power structures and collective-choice objectives) is a significant step beyond standard dialogue or persuasion tasks
2. Tying tasks to EP issues, debates, and resolutions is realistic (and much better than synthetic setups)
3. The topic taxonomy is broad and policy-relevant
4. Detailed scraping pipelines, prompt templates, and task descriptions are provided (high evaluation transparency)
5. Multiple models, tasks, and topic analyses expose nuanced failure modes (e.g., difficulty with Security and Rawls and specific biases)

### Weaknesses
1. LLM-as-judge with same-family system under test (GPT-4o judge vs GPT-4o candidate) risks systematic biases. A human (expert) verification, cross-judge, or calibration would be helpful.
2. The core of the aggregation, i.e., u_i = JUDGE(· | background, s_i, resolution) remains undefined (clearly under-specified). How exactly is alignment and feasibility integrated into this score?
3. It is not clear how the SD in Fig. 4 is exactly calculated. Can you please provide the exact error formula you are using?
4. Obviously, some scenarios have been omitted from the analyses. Can you please state how exactly you proceeded in the selection phase? In addition, you state that there were 2,225 "high-quality", i.e., complete records, but in Table 6 (last row), you report more than 2,700 total valid records. Can you please clarify?
5. Your vetoing rule seems to be contradictory. Does rejection by the vetoing party happen if support is under 60% or at least 60%? Please clarify.
6. DeepSeek-R1 is used to summarize, clean, and to extract party stances. How do you avoid errors or potential hallucinations? Human-based quality assessment, preferably with inter-annotator checks would be helpful. 
7. It is not clear how the number of seats per party affects the veto robustness. For each issue, you could sample 20 to 50 random seat distributions (and veto assignments where applicable) and report the mean with CI per model and task.
8. It is difficult to put the LLM performance into relation to a baseline because there is no simple (i.e., interpretable) baseline in the paper. Would it be possible to provide such a baseline? For example, if a proposal fails to meet the threshold (e.g., simple majority, 2/3), iteratively sacrifice the interests of the party that contributes least to passing (i.e., with smallest w_i). This would be a greedy strategy to pass the threshold.

### Questions
See comments above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PoliCon, a new benchmark to evaluate how large language models (LLMs) handle political consensus building. The authors collect real parliamentary data to create diverse decision-making scenarios and define several types of consensus goals. They design an LLM-based evaluation framework that measures whether a model’s generated resolutions can reach agreement among simulated parties. Experiments show that current LLMs perform well on simple majority decisions but struggle with more complex or fairness-oriented objectives. The results also reveal clear topic sensitivity and political bias, indicating that existing models still lack balanced reasoning in social and political contexts. Overall, the work introduces a valuable and well-structured framework for assessing LLMs’ ability to reason about group decisions and fairness.

### Strengths
1. The paper tackles a fresh and meaningful problem and it builds a solid and realistic benchmark using real parliamentary data, making the evaluation credible and grounded.

2. The evaluation setup is thoughtfully designed and connects well with social choice theory.

3. The experiments are thorough and provide clear insights into where current models perform well and where they fail.

### Weaknesses
1. The evaluation still depends on another LLM as a judge, which could introduce hidden bias.

2. The dataset only covers European Parliament data, so it might not generalize to other regions or political systems.

3. There’s no human validation to confirm that the evaluation results truly match real consensus reasoning.

4. Using LLMs in both dataset creation and evaluation could lead to subtle data leakage.

### Questions
1. How do you make sure the LLM-based judge is fair and not favoring certain model types?

2. Did you involve any human experts to check whether the model’s “consensus” decisions make real-world sense?

3. How could this benchmark be extended to other political or cultural settings beyond Europe?

4. What steps did you take to avoid overlap or leakage between LLM-generated data and evaluation content?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces PoliCon, a novel benchmark for evaluating LLMs' ability to achieve political consensus objectives. Built from 2,225 European Parliament deliberation records (2009-2022), PoliCon tests LLMs across diverse collective decision-making scenarios incorporating political issues, goals, participating parties, and power structures. The benchmark defines five distinct political consensus objectives (Simple Majority, Two-thirds Majority, Veto Power, Rawlsianism, and Utilitarianism) and employs an evaluation framework based on social choice theory that simulates real voting outcomes. The authors evaluate six state-of-the-art LLMs, revealing significant performance gaps in complex consensus scenarios and uncovering inherent partisan biases.

### Strengths
1. First benchmark specifically designed to evaluate LLMs' political consensus-building capabilities across diverse objectives
2. 2,225 high-quality parliamentary records with extensive cleaning and processing, integrating multiple sources 
3. Diverse task settings: 15 different configurations combining party numbers, voting mechanisms, and political goals, creating 28,620 distinct scenarios

### Weaknesses
1. Using GPT-4o-mini as evaluator could introduce circular biases when testing other LLMs

### Questions
Could the authors clarify whether the GPT-4o-mini evaluator was temperature-controlled (e.g., deterministic setting) during scoring, and whether variance across seeds was analyzed?

### Soundness
4

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
5

### Summary
This paper evaluates LLMs' capabilities to generate consensus statements on political issues. The authors created a novel benchmark constructed from 2,225 high-quality deliberation records of the European Parliament over 13 years. With an automatic evaluation pipeline, this paper demonstrates that LLMs have varied capabilities on generating consensus statements and most of the models remain undersatisfied with complex tasks.

### Strengths
1. This paper tackles an important topic 
2. The results are presented clearly

### Weaknesses
1. One of the major weaknesses of the paper is the reliability of the LLM-as-judge pipeline. LLM-as-judge has long been criticized for its generalizability, which applies to this paper as well. If I understand it correctly, the LLM-as-judge evaluation heavily relies on existing statements and voting data. However, it is really unclear whether this pipeline could create generalizable results for new statements. I believe this is the major issue of this paper. I'm happy to discuss with the authors about this.

2. A second weakness is regarding the motivation and practical value of this work. In parliament deliberations, it is the deliberation process that leads to the final voting results instead of the statement itself. Therefore, this task setting may not reflect real-world settings where the consensus statements are actually needed.

### Questions
please see weakness section

### Soundness
3

### Presentation
3

### Contribution
2
