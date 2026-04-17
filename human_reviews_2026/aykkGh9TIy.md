# UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Recent advancements in LLM-based information-seeking agents have achieved record-breaking performance on established benchmarks. However, these agents remain heavily reliant on search-engine-indexed knowledge, leaving a critical blind spot: Unindexed Information Seeking (UIS). This paper identifies and explores the UIS problem, where vital information is not captured by search engine crawlers, such as overlooked content, dynamic webpages, and embedded files. Despite its significance, UIS remains an underexplored challenge. To address this gap, we introduce UIS-QA, the first dedicated UIS benchmark, comprising 110 expert-annotated QA pairs. Notably, even state-of-the-art agents experience a drastic performance drop on UIS-QA (e.g., from 70.90 on GAIA and 46.70 on BrowseComp-zh to 24.55 on UIS-QA), underscoring the severity of the problem. To mitigate this, we propose UIS-Digger, a novel multi-agent framework that incorporates dual-mode browsing and enables simultaneous webpage searching and file parsing. With a relatively small $\sim$30B-parameter backbone LLM optimized using SFT and RFT training strategies, UIS-Digger sets a strong baseline at 26.36\%, outperforming systems integrating sophisticated LLMs such as O3 and GPT-4.1. This demonstrates the importance of proactive interaction with unindexed sources for effective and comprehensive information-seeking. Our work not only uncovers a fundamental limitation in current agent evaluation paradigms but also provides the first toolkit for advancing UIS research, defining a new and promising direction for robust information-seeking systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the problem of Unindexed Information Seeking, where vital information is inaccessible to search engines. 
The authors propose UIS-QA, a benchmark of 110 expert-annotated questions requiring interactions beyond indexed snippets, and present UIS-Digger, a multi-agent system with dual-mode browsing and file reading capabilities. 
Experiments show that UIS-Digger achieves 27.3% accuracy, outperforming stronger LLMs and existing multi-agent systems on UIS-QA, highlighting the importance of addressing UIS for real-world research agents.

### Strengths
- It identifies a novel and important gap that existing benchmarks overlook, and proposes the first so-called UIS benchmark with carefully validated data.
- Presents a multi-agent architecture with dual-mode browsing and file parsing.
- Strong empirical evaluation with comparisons against diverse baselines.
- Detailed error analysis and ablation studies provide useful insights.

### Weaknesses
- Absolute performance on UIS-QA remains very low (27.27%), raising concerns about practical utility.
- Training relies heavily on synthetic QA pairs, which may not fully reflect real-world UIS complexity.

### Questions
- How do you ensure UIS-QA’s representativeness given its small scale? Though 100+ samples are common, how do you ensure that diversity?
- How sensitive is UIS-Digger to backbone choice? Results suggest raw model quality still dominates. Does the framework generalize to weaker LLMs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the underexplored challenge of Unindexed Information Seeking (UIS) — retrieving essential information unavailable through standard search engines. The authors introduce UIS-QA, a benchmark specifically designed to evaluate agent performance on UIS tasks, and propose UIS-Digger, an agent system equipped with enhanced web-interactive tools and trained via sequential supervised and reinforcement fine-tuning. Experiments show that UIS-Digger achieves state-of-the-art results on UIS-QA, though overall accuracy (27.27%) highlights the inherent difficulty of the task. The work is valuable for formalizing UIS as a benchmarkable problem and demonstrating a viable training strategy, but the gains remain modest and practical effectiveness is still limited.

### Strengths
1. This work identifies an interesting unindexed information-seeking problem, which is practical in reality and still under-explored.
2. This work introduces a dataset to study this problem and also constructs an agent to solve this challenge correspondingly.

### Weaknesses
1. The scope definition of UIS is not crystally clear. How is “unindexed” information precisely defined? Does it include API-gated, or private web data?
2. The boundary with traditional search is not super clear as well. How to distinguish UIS tasks from regular information-seeking tasks where the answer is simply poorly ranked or paraphrased online?
3. This work claims to build a benchmark, but the characteristics of the benchmark are not enough. For example, what fraction of examples involve dynamic pages, paywalls, or databases?

### Questions
1,  Is there any overlap between UIS-QA examples and the pretraining or fine-tuning corpora (e.g., cached webpages or documentation dumps)? Can using DeepSeek to filter out data address this?
2. Can UIS-Digger transfer to new domains or unseen data environments, or is it overfit to UIS-QA’s structure?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the concept of "Unindexed Information Seeking" (UIS), which refers to information that cannot be directly retrieved through search engine queries and requires deeper interaction with websites (e.g., clicking, form filling, file downloading). The authors formalize the distinction between indexed information (II) - content available in search snippets or one-step crawling - and unindexed information (UI) - everything else requiring interactive navigation. They contribute: (1) UIS-QA, a benchmark of 110 expert-annotated QA pairs designed to test UIS capabilities, (2) empirical evidence showing state-of-the-art agents experience 47-54% performance drops on UIS-QA compared to existing benchmarks like GAIA, and (3) UIS-Digger, a multi-agent system with dual-mode browsing trained via SFT and RFT that achieves 27.27% accuracy, outperforming baselines including those using o3 and GPT-4.

### Strengths
1. **Valuable dataset contribution with clear curation criteria**: The manually created dataset of 110 validated QA pairs that explicitly avoid search engine shortcuts represents a concrete contribution.
2. **Valuable empirical evaluation**: Testing 13+ baseline systems across multiple categories provides valuable comparisons.
3. **Detailed failure analysis**: The breakdown of error modes and tool usage evolution across training stages offers useful insights.

### Weaknesses
1. **Insufficient differentiation from existing web agent research and benchmarks**: The paper claims that UIS represents an "underexplored challenge" and a "critical blind spot" in current agent systems, yet the capabilities required (e.g., interactive navigation, form filling, file downloading, multi-step exploration) are precisely what existing web agents can already do. The action space that UIS-Digger employs (search, crawl, click, scroll, type, download files) already exists in prior systems. Benchmarks like WebArena and Mind2Web also evaluate complex web interactions across diverse websites. The distinction appears to be one of benchmark curation methodology (filtering out search-solvable questions) rather than a discovery of a new research problem requiring novel capabilities.
2. **Critical missing baselines**: The experimental evaluation contains glaring omissions that prevent assessment of whether UIS is actually an unsolved problem for current systems. Most notably, the paper cites OpenAI Deep Research, Google Gemini Deep Research, and Grok-3 Deep Research in the related work but does not evaluate any of these systems. Recent web agent systems are absent. Without evaluating actual state-of-the-art deep research systems and web agent systems, the paper cannot substantiate its central claim that UIS represents a fundamental limitation or that the proposed 27.27% score represents competitive performance.
3. **Limited technical novelty**: UIS-Digger employs standard components from existing work: multi-agent architecture with specialized sub-agents; dual-mode browsing switching between textual and visual observation; the tools (search, crawl, browser automation, file readers); and the SFT + RFT training pipeline is a standard approach. More problematically, the system achieves only 27.27% accuracy, barely outperforming the Memento baseline at 25.5% (which was tested in a handicapped configuration without its case bank). This marginal improvement despite task-specific training on synthesized data similar to the test distribution might suggest that the paper has not discovered an effective solution methodology.

### Questions
1. WebArena and Mind2Web already test interactive navigation. What would be the difference between UIS-QA and them?
2. What would be the performance of UIS-Digger with the SoTA LLMs (like GPT, Claude) as the backbone?

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
3

### Summary
The paper tackles the interesting task for Unindexed Information Seeking (UIS), which involves answering questions based on vital information that is not directly surfaced by the search engines. The paper addresses this with a new UIS-QA benchmark along with the UIS-Digger approach that has a dual-browsing mode for interactively accessing information. The authors further finetune multiple backbone models using synthetic trajectories and demonstrate that the trained models outperform existing approaches on this task.

### Strengths
1) The experiments results are impressive, with exploration of the SFT and RFT finetuning approaches
2) The paper has a comprehensive analysis of the different actions and tool calls used.

### Weaknesses
1) The proposed UIS-QA benchmark is a bit limited in size with only 110 examples (with a split of 84 questions in Chinese and 26 questions in English). The authors should consider expanding the dataset to ~300 instances. Moreover, it would be worthwhile to also report expert human performance on this dataset to give a sense of upperbound. 

2) While the proposed UIS-Digger models excel on the UIS-QA dataset, the performance is low on GAIA and BrowseComp-zh datasets, bringing into question the generality of the approach. The authors should also show performance of zero-shot, SFT and RFT variants on the GAIA and BrowseComp-zh datasets to give a sense of whether the finetuning degrades performance on other benchmarks. 

3) The paper is missing a lot of important experimental details, particularly in how the SFT and RFT finetuning was done. Moreover, I have concerns about the reproducibility of Section 3.2.1 and Section 3.3 as minimal details have been provided.

### Questions
1) For the SFT and RFT processes, are the trajectories for the Planner, WebSearcher, WebSurfer and FileReader agents separately aggregated and then used for training?

### Soundness
3

### Presentation
2

### Contribution
3
