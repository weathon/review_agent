# Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
As model context lengths continue to grow, concerns about whether models effectively use the full context length have persisted. While several carefully designed long-context evaluations have recently been released, these evaluations tend to rely on retrieval from one or more sections of the context, which allows nearly all of the context tokens to be disregarded as noise. This represents only one type of task that might be performed with long context. We introduce Oolong, a benchmark of long-context reasoning tasks that require analyzing individual chunks of text on an atomic level, and then aggregating these analyses to answer distributional questions. Oolong is separated into two task sets: Oolong-synth, a set of naturalistic synthetic tasks, where we can easily ablate components of the reasoning problem; and Oolong-real, a downstream setting which requires reasoning over real-world conversational data. Oolong requires models to reason over large quantities of examples, to perform both classification and counting in-context, and to reason over temporal and user relations. Even frontier models struggle on Oolong, with the best model, GPT-5, achieving less than 50% accuracy on both splits at 128K. We release the benchmark examples, code to construct additional evaluation examples for Oolong-synth, and full outputs and task-specific evaluation results for all models tested to enable further development of models that can reason over large quantities of text.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces OOLONG, a long‑context benchmark aimed at measuring information aggregation rather than pure retrieval. It has two parts: OOLONG‑synth, which composes many short, easy ICL classification snippets into long contexts and asks distributional/user/timeline questions; and OOLONG‑real, which builds questions over transcripts from Critical Role with per‑episode statistics as gold labels. They find that strong models degrade markedly as context grows; even the best model tested struggles on both splits. The paper also provides ablations and a brief analysis of “reasoning effort” controls.

### Strengths
- Two splits. OOLONG‑synth allows controlled ablations; OOLONG‑real grounds the task in messy conversational data with externally curated labels. This pairing makes the claims more convincing than using only synthetic data.
- Performance vs. context length shows consistent degradation.
- “Reasoning level” comparisons are informative: more “reasoning effort” helps only at short lengths.

### Weaknesses
1. **Missing strong iterative/pipeline baselines.** - The paper does not report any strong baseline for tackling long-context problems (e.g. map‑reduce style baseline, an oracle‑labels pipeline upper bound, or a retrieval‑plus‑program). Including these would show how much of the gap stems from a single pass of LLMs vs. the fundamental difficulty of the task.

2. **OOLONG‑real data leakage.** - The gold labels come from CritRoleStats, a public source which have the possibility of being scraped for pre-training LLMs. Any efforts to check how much data leakage is present (e.g., just trying to ask the question without the context) can shed some more light on this.

3. **Limited Novelty.** - Several papers ([1,2]) have benchmarked LLM capabilities over long contexts that go beyond needle-in-the-haystack and measure the reasoning capabilities over the long context. Comparison against such benchmarks might make the paper's position clearer.

[1] Wang, Cunxiang, et al. "Novelqa: A benchmark for long-range novel question answering." CoRR (2024).
[2] Shaham, Uri, et al. "Scrolls: Standardized comparison over long language sequences." arXiv preprint arXiv:2201.03533 (2022).
[3] Bai, Yushi, et al. "Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks." arXiv preprint arXiv:2412.15204 (2024).

### Questions
Several typos:
1. “WWe believe…”, line 100
2. “cummulative”, multiple times
3. Table 2 (instead of Table 1) in line 151

### Soundness
3

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
The paper introduces OOLONG, a benchmark for evaluating long-context reasoning in LLMs. It is split into two parts:

OOLONG-synth, which builds very long prompts by concatenating many short, labeled classification examples (spam detection, sentiment, etc.) and then asks distributional / aggregation questions like “how many positive reviews?” or “which label is most frequent?”, including temporal and per-user variants.

OOLONG-real, which uses long, noisy, real conversational transcripts from Critical Role (Dungeons & Dragons actual play) and asks questions about events and statistics across one or more episodes (“how many spells of type X were cast?”, “what’s the cumulative roll count by episode N?”).

Across a range of large models (GPT-5, Claude-Sonnet-4, DeepSeek R1, etc.), accuracy drops sharply as context grows, and no model exceeds 50% accuracy at 128K tokens on either split. The authors argue this shows that current LLMs still struggle not just with retrieval (“find the one relevant span”) but with aggregating many small local decisions across extremely long inputs.

### Strengths
1. OOLONG frames long-context reasoning as multi-step aggregation: identify relevant spans, classify locally, and pool globally (counts, timelines, user-specific patterns). This is positioned as closer to realistic analytics tasks than classic “needle in a haystack.”

2. OOLONG-synth is controllable: it uses standard classification datasets (spam, sentiment, NLI, etc.) and scales to millions of tokens, letting the authors ablate factors like context length, label access, and time/user structure.

3. OOLONG-real uses long, messy human dialogue from Critical Role, combined with fan-annotated gold stats (dice rolls, spell usage), to approximate “real-world” multi-episode log analysis.

4. The paper confirms a now-familiar story: models nominally supporting 100K+ tokens still degrade badly when asked to actually integrate information across that full window, especially beyond ~64K. Even top models sit <50% at 128K.

5. They dig into DeepSeek R1: on OOLONG-real it can behave competitively, but on OOLONG-synth it underperforms even the random baseline because it burns tokens “thinking out loud,” tries to label every item, and then times out before answering. That illustrates the bottleneck they care about: planning and aggregation under length pressure.

### Weaknesses
My main issue is novelty. I did not find a clear, convincing argument that OOLONG measures a fundamentally new capability beyond what recent long-context benchmarks and analyses already target.

1. The stated motivation is: most long-context tests are retrieval-style (needle-in-a-haystack, MRCR, etc.), whereas OOLONG requires aggregation/counting over many items. But this does not feel substantially different from what earlier work like BABILong / long multi-step reasoning tasks including object tracking and counting (and similar multi-supporting-fact QA) that explicitly require pooling facts from many snippets across long input. RULER / HELMET-style setups already do with multi-hop tracing, counting, and distributional questions across long synthetic inputs, where relevant bits are dispersed and must be combined rather than simply located. The paper itself cites RULER, HELMET, MRCR, ZeroSCROLLS, and related aggregation-style tasks such as estimating label distributions in reviews.

The paper positions OOLONG as fundamentally new (“to the best of our knowledge, there are no benchmarks that evaluate LM’s ability to perform information aggregation at scale”). I don’t think that claim is fully justified. The differences feel incremental (different source data, different instructions), not qualitatively new.

2. Findings largely reproduce existing conclusions. The high-level conclusions are not surprising relative to prior long-context studies, and in fact mostly restate them:
- Performance falls off quickly with longer contexts.
- Even frontier models under 200K+ context struggle to actually use that context.
- Asking for “more reasoning” doesn’t magically fix degradation at very long lengths.
- Some models try brute-force per-chunk labeling, run out of space, and never answer.

These are all consistent with what benchmarks like BABILong/RULER/HELMET already report: long-context models often fail not because they can’t read long input but because they can’t maintain and aggregate distributed evidence across it. The paper presents similar failure patterns, just in a new wrapper.


Overall, I agree that long-context aggregation is important. I am less convinced that OOLONG, as currently framed, is novel enough beyond BABILong-/RULER-style counting & aggregation tasks and MRCR-/narrative-memory-style multi-session tracking. The empirical story (“even SOTA models still can’t do it at 128K+ tokens”) mostly confirms what those benchmarks already argued.

### Questions
1. Can you articulate what OOLONG measures that BABILong/RULER/HELMET/MRCR-style multi-hop aggregation does not already measure? Point to a specific capability that would let a model ace BABILong/RULER but fail OOLONG.

2. For OOLONG-real, can you show that Critical Role transcripts introduce qualitatively new reasoning structure (e.g. cross-episode state changes, retcons, overlapping threads) that existing narrative / long-dialogue benchmarks don’t already capture? Right now it reads mostly like very long multi-session dialogue logs plus counting.

3. Is there any system-level insight that changes how we evaluate or train long-context models because of OOLONG, beyond “they still struggle past 64K”?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Oolong is a long-context reasoning benchmark with two sets of tasks. One set uses existing long-context datasets as subtasks and requires answering 3 types of composite questions based on them. Another set uses long conversational data and formulates 3 types of questions based on counting, enumeration and indexing of events in the data. The experiments show that the tasks are challenging for frontier long-context models, with performance significantly decreasing with context size.

### Strengths
- Oolong provides two challenging sets of tasks that underscore the limitations of LLMs in long-context scenarios.

- The sample size is scalable, while context usage remains more dense compared to classic needle-in-a-haystack - based benchmarks. Relevant facts are hard to distinguish from irrelevant ones, which makes the tasks challenging for LLMs.
- Following relevant works, Oolong-synth inherits the validation-test structure, with no overlap between them, which allows for more fair evaluation.

### Weaknesses
- Conceptually the contribution of the current work is limited, as LLM degradation in long-context scenarios has been demonstrated by multiple other synthetic and real-world benchmarks. Some of these works were listed by authors in Section 5, but several relevant works that require long-context reasoning with aggregation are missing, including BABILong (Kuratov et al., 2024) and InfinityBench (Zhang et al., 2024). Discussing their differences and limitations compared to Oolong is advised. 
- If the primary contribution of the work is the study of aggregation capabilities, a more in-depth analysis is needed, not only reported degradation with length.
- The task complexity of the Oolong-synth set increases with context size, making it hard to estimate the effect of these two matters independently. Generally speaking, it is not entirely clear whether the performance degrades solely because of the increasing context size or because the underlying aggregation subtasks become more complex. Evaluating retrieval-augmented models or multi-step prompting would provide some insight in this regard, however no such baselines are present.
- The evaluation of certain models such as Deepseek R1 might be not completely fair. As authors state in section 4.3, the results are inconsistent, and traces often end mid-sentence, without outputting any answer. In line 201 it is confirmed that some models ran out of output tokens before providing the answer. Instead of evaluating the reasoning quality, this way of inference measures the performance under the CoT length limitation. It is not necessarily wrong, but this can cause confusion and misinterpretation of the results.

### Questions
- Oolong-synth samples are constructed by mixing contexts of various long-context tasks. One concern is related to information overlapping between these contexts: can one task affect the solution of another one? If so, what can be done to prevent this effect?
- The types of questions in Oolong-real can be more suitable for code executing models, they can have a significant upper hand. Have you tried using such models in the evaluation?
- Oolong-real has really long samples of length 1.3M, and Oolong-synth up to 4M tokens,  but were they actually used? Max length reported in Table 4 is 175K and 512K on Figures 2, 3, 4.


comments / typos:
- Appendix B3: results for all models are mentioned but not provided
- Line 96: missing space
- Lines 100-101, typo: We believe

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
3

### Summary
The paper introduces Oolong, a benchmark for long-context reasoning and aggregation that goes beyond simple retrieval. Oolong comprises two splits: Oolong-synth and Oolong-real (Dungeons & Dragons campaign transcripts). Tasks require analyzing atomic chunks and aggregating the results (classification, counting, temporal/user relations). Experiments show that even frontier models perform poorly, with accuracy dropping below 50% already at 128K context length. The authors release data and code to facilitate further work on models that can reliably aggregate information over long contexts.

### Strengths
* Practical Significance: Proposed benchmark tries to move beyond common needle-in-a-haystack or generic summarization setups, targeting fine-grained analysis and aggregation. These areas underrepresented in existing long-context benchmarks.
* Originality: Requires atomic reasoning, counting/classification, and temporal/user-relational aggregation rather than simple retrieval, reducing shortcut solutions.
* Realistic long-horizon data: Uses Dungeons & Dragons campaign transcripts, which feature coherent narratives, long-range dependencies, and non-encyclopedic facts less likely to be memorized from pretraining.

### Weaknesses
* Limited experimental validation of claims: The paper argues Oolong differs from retrieval-centric long-context benchmarks by requiring multi-chunk analysis and logical aggregation, but provides no targeted experiments to substantiate this. To demonstrate the distinction empirically, the evaluation should include not only base LLMs but also strong single- and multi-step RAG baselines. Divergent performance of such RAG systems on Oolong vs. needle-in-a-haystack / multi-hop retrieval tasks would directly support the benchmark’s claimed novelty.

* Insufficient result granularity: Results are reported mainly by model and average sample length, with no breakdown by task or question type. A per-category analysis (for example, classification, counting, and temporal or user-relational queries) is needed to reveal inherent difficulty and interactions with context length. Some categories may degrade disproportionately at longer contexts. Inclusion of plots or tables that slice performance by task type and by length would strengthen the paper.

### Questions
* How was the evaluation metric chosen in lines 203–208, and will you report additional metrics?

* Real data quality. The transcripts are probably auto-generated from hundreds of 4 hour long streams and hard to verify manually, and the website-sourced data may include errors without accuracy guarantees. Can you more thoroughly describe your methods for assessing data quality and quantify how transcription or source errors affect model scores on Oolong-real?

* Can parts of data cleaning and annotation be automated, for example with LLM-based pipelines?

### Soundness
2

### Presentation
3

### Contribution
3
