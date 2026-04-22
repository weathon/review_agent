# Improving Code Localization with Repository Memory

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Code localization is a fundamental challenge in repository-level software engineering tasks such as bug fixing. While existing methods equip language agents with comprehensive tools/interfaces to fetch information from the repository, they overlook the critical aspect of *memory*, where each instance is typically handled from scratch assuming no prior repository knowledge. In contrast, human developers naturally build long-term repository memory, such as the functionality of key modules and associations between various bug types and their likely fix locations. In this work, we augment language agents with such memory by leveraging a repository's *commit history* - a rich yet underutilized resource that chronicles the codebase's evolution. We introduce tools that allow the agent to retrieve from a non-parametric memory encompassing recent historical commits and linked issues, as well as functionality summaries of actively evolving parts of the codebase identified via commit patterns. We demonstrate that augmenting such a memory can significantly improve LocAgent, a state-of-the-art localization framework, on both SWE-bench-verified and the more recent SWE-bench-live benchmarks. Our research contributes towards developing agents that can accumulate and leverage past experience for long-horizon tasks, more closely emulating the expertise of human developers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes RepoMem, a lightweight repository memory for code localization agents, built from each repo’s commit history. It instantiates two complementary, non-parametric memories: (1) Episodic—a searchable store of prior commits and linked issues, queried via SearchCommit / inspected via ExamineCommit; and (2) Semantic—LLM-generated functionality summaries for frequently edited (“hotspot”) files, accessed via ViewSummary / SearchSummary. Integrated into LocAgent, RepoMem improves localization on SWE-bench-verified and a filtered SWE-bench-live subset, using GPT-4o (2024-05-13) as the backbone and reporting Accuracy@k.

### Strengths
The paper introduces a practical repository-memory mechanism for SWE-bench localization, combining (i) an episodic store built from pre-issue commit history and linked artifacts that can be searched/inspected at inference time, and (ii) a semantic memory of LLM-generated summaries for frequently edited (“hotspot”) files; the design is simple to integrate, and aligns with how human developers leverage history to narrow the search scope. In extensive evaluations, the memory delivers consistent performance gains over a strong LocAgent baseline, improving Accuracy@k on SWE-bench-verified and a live subset, with ablations indicating complementary benefits of episodic + semantic memories.

### Weaknesses
1. Limited Novelty in Core Mechanism: The agent's memory system, while presented as a compositional structure, relies heavily on established prompting techniques and existing memory concepts rather than introducing fundamentally new mechanisms for agent memory or reasoning. The contribution seems more focused on engineering specific prompts for memory management than on novel agent capabilities.
2. Ablation gaps (end-to-end impact). Empirics focus on localization Acc@k, but the paper does not quantify downstream SWE-bench end-to-end effects (e.g., Resolved@1, test pass rate, patch compile/apply rate, tool-call path length, wall-clock). Without E2E deltas versus a strong baseline (and versus LocAgent without memory), it is hard to assess whether memory truly improves fix outcomes.

### Questions
1. End-to-end impact: Can you report SWE-bench end-to-end results (resolution rate, test pass) with/without RepoMem, and per-component ablations, to quantify how localization gains translate to actual fixes?
2. Cost/latency & token accounting: What are (a) average prompt tokens per component, (b) offline token cost to build semantic memory (top-200 file summaries), and (c) online latency/$ overhead relative to LocAgent?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper aims to enhance the fault localization agent with two types of memory designs.

The intuition is clear and well justified: A human developer would not consider a code repo as a fresh problem instance. Instead, she will consider the commit histories and other background knowledge about the code repo.

Specificially, this work introduces two types of memories:
1. A memory that summarizes key information in commit histories
2. A memory that summarizes the code semantics of some key files.

To identify the key files, the proposed system identifies the files that are most frequently modified in previous commits.
then they use LLM to generate summaries for those key files.

To enhance the localization agent system, the proposed memories are provided as tools to the agent.

The paper evaluates the localization agent on two issue fixing benchmarks: swe-bench and swe-bench-live. With the augmented tools, the agent demonstrates better performance in fault localization on those two benchmarks.

### Strengths
The paper targets an important problem and proposes a neat and intuitive solution.

### Weaknesses
Q1: Justify the active code identification. The proposed approach identifies frequently modified code files and only summarizes the semantics of those files. I think it is a strong prior to assume most issues only happens in those files. Are there any statistical support for this observation/heuristics? Are they specific to those two benchmarks?

Q2: Clarify the details about code summary generation. How the code summaries are generated? Many code snippets/functions have complex dependences across files. Do the generated summary consider the contexts across different files?

Q3: The proposed technique summarizes the top_k active edited file. Is the system performance sensitive to the selection of k?

Q4: Adaptation to newer commits. When the code repo receives new commits, how the pre-generated memories would be updated? For example, some of the code summaries may be out-dated.

### Questions
Please see above.

### Soundness
2

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
The paper augments repository-level localization agents with repository memory built from pre-issue commit histories—an episodic memory of past commits/issues and a semantic memory of summaries for frequently edited files—yielding consistent gains over LocAgent on SWE-bench-Verified.

### Strengths
- The framing—that human-like “long-term repository memory” is missing in current agents—is clear and well motivated, and the two concrete tools are simple yet effective: SearchCommit / ExamineCommit (BM25 over commit messages + diffs) and SearchSummary / ViewSummary (LLM summaries of “hot” files). Integration into LocAgent’s ReAct loop is clean and modular, with sensible leakage controls (time-slicing to pre-issue commits and filtering text overlaps). 

- On SWE-bench-Verified, RepoMem improves Acc@5 from 71.6 → 76.5 (+4.9 abs) over LocAgent; on SWE-bench-Live, 63.1 → 66.2. The setup details (GPT-4o, 7k prior commits, top-200 files for summaries) are reported, and both episodic- and semantic-only ablations are positive, suggesting complementary value. 

- Insightful analyses. (i) Per-repo analysis shows larger gains where commit histories are rich; (ii) cost analysis reveals instance-level heterogeneity—memory sometimes reduces cost dramatically but can add overhead on hard cases; (iii) retrieval study finds BM25 (with an LLM-aware tokenizer) outperforms a strong dense retriever (GritLM-7B) on Django. These help practitioners reason about when/why the method helps.

### Weaknesses
- Sensitivity to history density. Benefits correlate with repository commit volume; performance degrades on the “others” bucket with sparse histories (−13.1 Acc@5 vs. LocAgent), underscoring brittleness when history is limited or off-target. Some mitigation is discussed qualitatively but not implemented. 


- Dataset selection and external validity. For SWE-bench-Live, the paper evaluates an intersectioned subset (Lite and Verified) and filters to ≤5 modified files (130 examples/62 repos). This practical choice is understandable but may bias toward easier issues; a sensitivity analysis to the filter would strengthen claims. 


- Reporting gaps on efficiency & scalability. While costs are compared, the paper lacks index/build-time, memory footprint, and latency for constructing/searching the 7k-commit memory and top-200 summaries—numbers crucial for monorepos and CI integration.

### Questions
Retrieval extensions: given BM25’s strong showing on Django, report whether the LLM-aware tokenizer generalizes across repos and whether hybrid sparse+dense retrieval helps when commit phrasing diverges from issue wording

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose integrating commit history and recent activity via a "memory" interface as context for LLM agents when tackling fault localisation. To demonstrate the value of their approach, they integrate with LocAgent and expose the memory through tools/API calls. They evaluate on SWE-bench-verified and SWE-bench-live, showing consistent improvement in performance over the vanilla LocAgent. The authors also analyse the tools employed by the agent during localisation and the cost trade-off depending on success (either of the memory-based tool or the vanilla tool), demonstrating that the cost landscape is non-trivial. This work serves as a first step in considering external project information sources that can serve as project/institutional memory (commit history, issue trackers, code reviews, etc.).

### Strengths
- combining project history (commit history) with repository structure for better localisation in the context of LLMs.
- demonstrated consistent improvement in performance over the extended baseline/SOTA
- analysis of the cost trade-off between using/not using memory tools
- careful removal of information that can cause data leakage (through issues/transitive relationships)

### Weaknesses
- The cost overhead is significantly higher, specifically for cases where vanilla LocAgent fails. (I am not convinced about tool allocation, especially since hotspots and recent development can be a very strong signal, at least in just-in-time defect prediction)
- The memory construction hyperparameters seem odd (I expected runtime compression + last 14 days of commits or similar thresholds from SE, not 7000 commits as pre-processing)
- Some of the metadata exposed/used by the proposed approach depends on links that are known to be noisy (Issue-Commit links are known to be missing: seminal paper for the problem definition -- Bachmann, Adrian, et al. "The missing links: bugs and bug-fix commits." Proceedings of the eighteenth ACM SIGSOFT international symposium on Foundations of software engineering. 2010. But research has progressed since)
- Memory construction may be inefficient, but the efficiency of memory construction may be out of scope. (reference paper -- S. Kim, T. Zimmermann, K. Pan and E. J. Jr. Whitehead, "Automatic Identification of Bug-Introducing Changes," 21st IEEE/ACM International Conference on Automated Software Engineering (ASE'06), Tokyo, Japan, 2006, pp. 81-90, doi: 10.1109/ASE.2006.23. And the area of just-in-time defect prediction). Specifically, borrowing ideas of how to narrow which commits to summarise/choose.

### Questions
- Comment1: While not directly related, a "sister" area of research is just-in-time defect prediction, and the literature reference for this is missing. Consider this survey as a starting point: Yunhua Zhao, Kostadin Damevski, and Hui Chen. 2023. A Systematic Survey of Just-in-Time Software Defect Prediction. ACM Comput. Surv. 55, 10, Article 201 (October 2023), 35 pages. https://doi.org/10.1145/3567550. The core of the comment is that, by the nature of the just-in-time version of the fault localisation problem, the data that is to be used is the commit history. Still, to my knowledge, this work is the first to consider this source of information in an agentic framework, enabling access through tools.

- Q1: In JIT defect prediction, features such as recent commits, recent commits by the same author, historically co-chaged files, etc., are used as manually created features. In this work, such metadata enrichment is not considered. Was such, or different, preprocessing considered beyond the scope of the work? For example, I would expect even a time-stamp filter to aid when retrieving development history for an agent to use; 7000 commits before the current base commit is considerably more than what is standard in SE work.

### Soundness
3

### Presentation
4

### Contribution
3
