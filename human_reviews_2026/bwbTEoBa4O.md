# PartnerMAS: An LLM Hierarchical Multi-Agent Framework for Business Partner Selection on High-Dimensional Features

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 4, 2, 4

## Abstract
High-dimensional decision-making tasks, such as business partner selection, involve evaluating large candidate pools with heterogeneous numerical, categorical, and textual features. While large language models (LLMs) offer strong in-context reasoning capabilities, single-agent or debate-style systems often struggle with scalability and consistency in such settings. We propose \name, a hierarchical multi-agent framework that decomposes evaluation into three layers: a Planner Agent that designs strategies, Specialized Agents that perform role-specific assessments, and a Supervisor Agent that integrates their outputs. To support systematic evaluation, we also introduce a curated benchmark dataset of venture capital co-investments, featuring diverse firm attributes and ground-truth syndicates. Across 140 cases, \name consistently outperforms single-agent and debate-based multi-agent baselines, achieving up to 10–15\% higher match rates. Analysis of agent reasoning shows that planners are most responsive to domain-informed prompts, specialists produce complementary feature coverage, and supervisors play an important role in aggregation. Our findings demonstrate that structured collaboration among LLM agents can generate more robust outcomes than scaling individual models, highlighting \name as a promising framework for high-dimensional decision-making in data-rich domains.
Our implementation is available at https://anonymous.4open.science/r/Partner-MAS-7DCE.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the challenge of high-dimensional, heterogeneous decision-making, using business partner selection as a representative task. The authors argue that existing single-agent or simple debate-based multi-agent (MAS) LLM systems struggle with the scalability and consistency required for such complex evaluations.

The primary contributions are twofold:
(1) A new benchmark dataset. The authors curated a dataset of 140 venture capital co-investment scenarios, featuring a large candidate pool and diverse (numerical, categorical, textual) features.
(2) The PARTNERMAS framework. A novel hierarchical, three-layer multi-agent system designed to decompose and manage the evaluation process.

The PARTNERMAS architecture consists of:
(1) A Planner Agent: Analyzes the task context to design an evaluation strategy and configure a team of specialists.
(2) Specialized Agents: Each agent is assigned a specific role and evaluates the entire candidate pool from its narrow perspective, implicitly performing feature selection and producing a ranked shortlist.
(3) A Supervisor Agent: Aggregates the multiple shortlists from the SAs. It first identifies consensus picks and then uses strategic guidance and agent importance weighting to resolve conflicts and produce the final shortlist.

### Strengths
The paper tackles a practical, high-stakes problem that is an excellent fit for LLM-based reasoning. The proposed PARTNERMAS architecture is well-motivated and logical. The decomposition of the problem into "Planner," "Specialist," and "Supervisor" roles mirrors high-level human expert workflows and is a clean, valuable contribution.

The strongest result is that a well-structured MAS (PARTNERMAS) using a smaller, more efficient model can outperform a larger, more expensive model in a simpler configuration (Figs 2 & 3). This "architecture over scale" finding is an important one for the field, highlighting the value of structured collaboration versus brute-force scaling.

### Weaknesses
A dataset of 140 cases is very small. While curating such data is difficult, this small sample size calls into question the statistical robustness of the 10-15% improvement. It is hard to be confident that the results will generalize, especially given the high variance in individual specialist agent performance (Fig 5). The dataset is also confined to a single, niche domain (US VC).

The Supervisor is identified as the most critical and failure-prone component, yet its mechanism is the most opaque. The paper describes a "Consensus" and "Conflict Resolution" step (Sec 3.2), but the prompts in Appendix F.4 show three distinct potential strategies: "by Importance," "by Weight," and "by Majority Vote." The main paper does not specify which of these was used for the main experiments.

### Questions
Your analysis in Table 2 shows the Planner's strategy is not adaptive to the specific case context. Is this the intended behavior? Have you considered alternative prompting strategies (e.g., chain-of-thought, few-shot) to encourage the Planner to dynamically select or even generate novel specialist roles based on the specific target company's profile.

Given the small dataset of 140 cases and the high variance in specialist agent performance (e.g., "Risk & Compliance" at 83% vs. "Investment Stage" at 38% for gpt-4.1-mini), how concerned are you about overfitting? Is it possible that the "best" supervisor strategy is simply the one that, by chance, happened to correctly weight the high-performing (but lucky?) specialists for this specific set of 140 cases?

Could you please clarify the "feature selection" mechanism used by the Specialized Agents?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PartnerMAS, a hierarchical multi-agent system for business partner selection in high-dimensional settings. The framework decomposes the partner selection task across three layers: (1) a Planner Agent that designs evaluation strategies, (2) Specialized Agents that perform domain-specific assessments, and (3) a Supervisor Agent that aggregates outputs. The authors propose a benchmark dataset of 140 venture capital co-investment cases and demonstrate improvements of 10-15% in match rates over single-agent and debate-based baselines.

### Strengths
1. This paper raises an interesting topic about business partner selection. It's a real-world business problem that is underexplored in the MAS. The application to VC co-investment selection is well-motivated and practically relevant.

2. The paper evaluates different LLM backbones and prompt strategies. The author did the ablation study and provided a detailed analysis of agent behavior, feature selection patterns, and component contributions.

3. The author introduces a new VC investment dataset with diverse features (numerical, categorical, textual).

### Weaknesses
1. I have several questions about the evaluation metrics.  Firstly, the Match Rate is essentially recall only. What about other metrics? such as precision. The paper doesn't discuss whether non-matched selections in the shortlist are reasonable alternatives. Secondly, Ground truth is actual co-investors, but this has survivor bias. It doesn't mean they were necessarily the best choices, just that they were selected.

2. I also have concerns about the experiment design part. Firstly, there is a lack of comparison with traditional ML methods. Secondly, Single Agent with k=4 self-reflections vs. MAS with ~4 agents uses similar computation but very different strategies. This doesn't cleanly isolate the benefit of multi-agent collaboration.

3. Business guidance is manually designed with only two conditions tested (with/without). Could the author provides more explaination on this point?

4. Encourage authors to do some failure case analyses. If the proposed framework fails, what consequences will it bring? Or what types of cases are challenging?

### Questions
See in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a Multi-Agent System (MAS) approach for selecting business partners / co-investors, which are represented by numerical, categorical or textual features. The approach consists of a planner agent, specialized agents and a supervisor agent.

The approach is compared to a single agent setup and a debtate MAS setup, using the  London Stock Exchange Group dataset. The result show that the approach is superior. The authors show that the pure debtate approach is not enough, making it important to model more complex MAS.

### Strengths
- Sensible approach, extending single agents and debate MAS with more complex MAS
- Comparison to debate MAS successful for the dataset, showing superior performance
- Helpful ablation studies

### Weaknesses
- Missing grounding in MAS research. There are many more general approaches to building/designing MAS. The proposed approach is a manually tailored MAS to a single dataset. How does the approach then behave wrt [1,2] or other seminal MAS LLM papers? These should be at least mentioned and thoroughly compared to in related works. It should therefore be made clear what added value the current paper makes and why one does not have to empirically compare against them.
- In a similar vein, it should be made clear what research in tabular LLMs is available in similar direction, especially single agent / LLM approaches.

[1] Ke, Z., Xu, A., Ming, Y., Nguyen, X.P., Xiong, C. and Joty, S., 2025. MAS-ZERO: Desi
[2] Li, J., Zhang, Q., Yu, Y., Fu, Q. and Ye, D., 2024. More agents is all you need. arXiv preprint arXiv:2402.05120.

### Questions
- How does your approach relate to mentioned recent works on MAS / automating MAS design / tabular LLMs?
- How does your approach benefit related domains or tasks?
- Would your approach generalize to the general downstream task of ranking tabular data?

### Soundness
2

### Presentation
3

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
This paper proposes PARTNERMAS, a hierarchical multi-agent system for business partner selection based on high-dimensional firm features. The framework consists of three layers—a Planner Agent that analyzes context and creates evaluators, multiple Specialized Agents that assess candidate firms from different perspectives, and a Supervisor Agent that integrates results for the final decision. The authors also introduce a tabular benchmark for co-investor selection reflecting real-world, multi-criteria decision-making. Experiments show that PARTNERMAS improves performance by about 15% over single-agent and debate-based baselines.

### Strengths
1) The paper is well-written and clearly organized, with informative visualizations that effectively illustrate the experimental outcomes.

2) The topic of business partner selection is intriguing and addresses a real-world challenge that has been relatively underexplored within the MAS domain. The application to VC co-investment selection is well-motivated and carries strong practical relevance.

3) The setup of the ablation studies is rasonable, and the detailed analyses help to better illustrate the contribution of each component.

### Weaknesses
1） One major concern is from the experimental set-up: Although the experimental evaluation includes several LLM backbones, the study would be strengthened by incorporating fair comparisons with alternative approaches, such as classical machine learning and deep learning methods designed for the same task.


2） While the paper presents a clear experimental setup, one potential limitation lies in the relatively simple choice of evaluation metrics. The authors primarily assess system performance using match rate, which, although intuitive, may not fully capture the multifaceted nature of business partner selection tasks. To more comprehensively evaluate the robustness and reliability of the proposed MAS framework, I suggest incorporating a broader range of metrics.  For example,  metrics such as F1-Score could provide a more balanced view of both accuracy and completeness. Ranking-based metrics like Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (nDCG) can be also valuable to be included into evaluation metric set.

### Questions
Same as what I mentioned in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
