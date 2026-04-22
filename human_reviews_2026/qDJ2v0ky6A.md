# AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Fine-tuning large language models for code editing has typically relied on mining  commits and pull requests. The working hypothesis has been that commit messages describe human intent in natural language, and patches to code describe the changes that implement that intent. However, much of the previously collected data is noisy:  commit messages are terse, human-written commits commingle several unrelated edits, and many historical commits come from simple, rule-based bots.

The recent adoption of software engineering agents changes this landscape. Code changes co-authored by humans and agents are often accompanied by substantially more explicit natural-language descriptions of intent and rationale. Moreover, when these changes land in public repositories, they are implicitly filtered by humans: maintainers discard low-quality commits to their projects.

We present AgentPack, a corpus of 1.3M code edits co-authored by Claude Code, OpenAI Codex, and Cursor Agent across public GitHub projects up to mid-August 2025. We describe the identification and curation pipeline, quantify adoption trends of these agents, and analyze the structural properties of the edits.  Finally, we show that models fine-tuned on AgentPack outperform models trained on prior human-only commit corpora, highlighting the potential of using public data from software engineering agents to train future code-editing models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AGENTPACK, a large-scale dataset of 1.3M code edits co-authored by AI agents and humans. The core hypothesis is that this data, curated from public repositories and implicitly filtered by human maintainers, is superior to traditional commit-mined data. The paper includes a comprehensive analysis of the dataset's properties and an evaluation of DeepSeekCoder (1.3B, 6.7B) models, finding that fine-tuning on AGENTPACK yields generally positive results compared to the prior EditCoder dataset.

### Strengths
- Originality: The central idea of mining agent-human collaborative edits is novel and highly relevant. As agent use becomes ubiquitous, this data source will only grow.

- Quality & Scale: The resulting 59GB dataset is an order of magnitude larger and more complex than prior work (e.g., EditCoder). It also captures a wide variety of languages and task types, as shown in the analysis.

- Clarity: The paper is well-written. The data collection, filtering, and processing pipeline are all clearly articulated.

### Weaknesses
- Marginal Performance Gains: The performance gains over EditCoder are not uniformly strong. While significant on the 1.3B model, the improvement is minimal on the more capable 6.7B model (+6.7% on HumanEvalFix) and even shows a minor regression (-2.4% on CanItEdit). This suggests the dataset's primary value might be in bootstrapping weak models, while stronger models may not benefit as much.

- Outdated Model Evaluation: For a paper targeting a 2026 conference, the choice of base models is a significant weakness. The experiments use DeepSeekCoder V1 (released Nov 2023), while DeepSeekCoder V2 (June 2024) and DeepSeek V3 (Dec 2024) have been available for over a year. This makes it difficult to assess the dataset's relevance. The key open question is whether AGENTPACK provides a meaningful signal to SOTA models or if its benefits are limited to older, less capable models.

- Mismatch in Evaluation Complexity: The paper's evaluation is limited to relatively simple, single-function benchmarks (HumanEvalFix, CanItEdit). This is a missed opportunity IMO, as the AGENTPACK dataset itself contains complex, multi-file edits. Evaluating on more challenging benchmarks like SWE-bench would have been a far more convincing validation of the dataset's utility.

### Questions
1. Could the authors provide results on more complex, multi-file editing benchmarks like SWE-bench? This would better align the evaluation with the claimed complexity of the AGENTPACK data.

2. The paper frames the verbose, LLM-generated instructions as a positive. However, does this create a distribution mismatch and potentially hurt model learning? Could training on these hyper-detailed descriptions harm performance when the model is given a concise, or even vague, instruction from a real user? If the model is trained on these "solution-aware" descriptions, is it actually learning the difficult tasks of problem-solving, intent inference, and bug localization? Or is it just learning a simpler translation from a detailed commit description to code?

3. The training format ("ellipsis-hunk") seems mismatched from the evaluation format (full file output). Why was this format chosen over others more suitable for model generation, such as OpenAI's [diff format](https://cookbook.openai.com/examples/gpt4-1_prompting_guide#reference-implementation-apply_patchpy) or [Aider's search-replace format](https://aider.chat/docs/more/edit-formats.html)? Have other training representations been explored?

### Soundness
3

### Presentation
4

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
This paper introduces AGENTPACK, a large-scale dataset of 1.3M code edits (~60GB) co-authored by LLM-based coding agents (Claude Code, OpenAI Codex, Cursor Agent) and humans, mined from GitHub between April–Aug 2025. The authors argue that agent–human commits exhibit clearer natural language descriptions, more structured changes, and higher quality due to human filtering (merged PRs). They describe a pipeline to detect agent commits, extract diffs, filter dependency files, and align metadata. The paper analyzes code/description properties and task types, and demonstrates that fine-tuning models on AGENTPACK improves HumanEvalFix and CanItEdit performance over CommitPackFT and EditCoder datasets

### Strengths
Relevant and timely dataset capturing the rise of coding agents.
Solid motivation: agent-written commits include richer natural language intent than human-only commits.
Careful filtering: merged commits only; node_modules removal.
Extensive dataset analysis across languages, file types, patch sizes, and edit tasks.
Demonstrated performance gains over strong editing baselines (CommitPackFT, EditCoder).
Addresses an emerging paradigm shift in software engineering: human–LLM collaboration.
Useful resource for future research in code editing and agent behavior.

### Weaknesses
1. **Evaluation scope too limited**
   - Only Python subset used, despite claiming multi-language strengths.
   - Benchmarks are relatively small; improvements modest.
   - No test on real repositories or multi-file editing despite dataset emphasis.

2. **Assumptions not empirically validated**
   - “Merged commits = high quality”
   - “LLM commit messages = true intent”
   - No human study confirming commit message correctness or utility.

3. **Agent attribution uncertain**
   - Signature-matching may miss cases or misattribute commits.
   - Cursor backend ambiguity creates label noise.

4. **Missing critical ablations**
   - Quantity vs quality effect not disentangled.
   - No filtering strategy ablation.
   - No comparison to synthetic R1/agent-rationale datasets.

5. **Licensing and usage implications unclear**
   - Commercial training concerns left ambiguous.
   - No discussion of potential security or insecure-code contamination.

6. **Underexplored failure cases**
   - Where does AGENTPACK underperform?
   - Does training produce verbose or agent-like coding artifacts?

### Questions
1. How reliable is the agent detection pipeline? Did you manually sample for attribution accuracy?
2. Do LLM commit messages ever hallucinate rationale? Any validation?
3. Why not evaluate on JS/TS (largest portion of dataset)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces AgentPack, a valuable dataset that captures code changes co-authored by LLM agents and human developers and successfully merged into production branches. The dataset’s methodology of using human acceptance as a critical quality filter, combined with the LLM’s ability to generate rich context, results in a corpus superior to historical human-only commit data for training advanced code-editing models. The contribution is significant and timely for advancing software engineering agent research.

### Strengths
1. Existing datasets for code changes are known to be noisy, with several tangled commits. In this work, the authors explicitly focus on commits created by coding agents with humans as gatekeepers, including only those that successfully merged into the main/master branch. As a result, the dataset benefits from an inherent human-in-the-loop filtering, ensuring that the data captures production-ready changes for training reliable code editing models. 

2. The LLM-generated commit messages are not merely longer (median ~10x), but provide a qualitatively superior, explicit rationale for the change. This rich context is invaluable for teaching models the complex mapping between a detailed, single-purpose instruction (the "intent") and the corresponding code implementation.

3. The dataset captures structurally multi-hunk, multi-file code changes, often spanning large patch sizes (median 70 lines). The authors correctly point out that these are often logically unitary changes (e.g., implementing one feature across several architectural layers) that older datasets wrongly filtered out, providing a more realistic and challenging benchmark for agents.

### Weaknesses
1. The main limitation, IMO, is that the dataset is defined by the outcome (i.e., the merged commit) rather than the process (i.e., the full agent-human interaction history). As a result, it is not possible to track the agent trajectory, as well as the actual contribution of the human developers in refining it. This means the dataset trains models on a "successful collaborative output"rather than pure agent autonomy, limiting how much can be inferred about the agent’s standalone capability.

2. The claim that AgentPack commits are inherently "less tangled" compared to human-only data is hard to fully accept given their median size (70 lines across 2-3 files). A commit of this magnitude has a high structural possibility for introducing logical tangling, even if the agent's intent was singular. The paper’s reliance on the length and detail of the commit message to define quality and untangledness is a strong, yet potentially insufficient, assumption. A qualitative analysis categorizing the logical unity of a sample of these large commits would significantly strengthen this claim.

### Questions
1. What criteria or methodology were used to conclusively determine that a long, multi-file commit is logically focused rather than merely a combination of several documented changes?

2. Given the risk of LLM hallucination, how did the authors verify or ensure that these detailed rationales are factually correct and accurately describe the changes made to the code, rather than fabricating justifications? Is there a metric provided for the correctness of the commit messages?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces AgentPack, a corpus of 1.3M code edits co-authored by Claude Code, OpenAI Codex, and Cursor Agent across public GitHub projects. This corpus can be used to fine-tune large language models for code editing. Empirical results in this paper show that models fine-tuned on AgentPack can outperform models trained on prior human-only commit corpora.

### Strengths
1. This paper proposes a new dataset co-authored by agents and humans for fine-tuning LLMs on code editing tasks. 
2. Empirical results show that models like DeepSeekCoder fine-tuned on the proposed corpus outperform models trained on prior human-only commit corpora.
3. This paper is easy to follow.

### Weaknesses
1. There are limited empirical results. For example, in Table 4, this paper only fine-tuned one LLM, i.e., DeepSeekCoder, and only compared with one prior training dataset EditCoder. More LLMs like CodeLlama and Mixtral can be included in the experiments, to make the results more solid.
2. Neither codebase nor dataset has been released to ensure reproducibility.
3. Some settings are unclear. Which exact model does Claude Code and Codex use?

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
