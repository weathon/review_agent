# FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6

## Abstract
As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions  and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. Together, these findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench, establishing a strong foundation for future research on RP evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FURINA-Builder, a novel and scalable multi-agent collaboration pipeline designed to automatically construct fully customizable Role-Playing benchmarks. The authors address the limitations of existing static and narrowly-scoped RP benchmarks by enabling the creation of benchmarks tailored to arbitrary roles, scenarios, and evaluation dimensions.

The core mechanism involves simulating group chat dialogues between the test character and other NPCs extracted from a Character-Scene Pool. A key feature is the use of a Judge Model to select fine-grained evaluation dimensions and choose the superior response between a "source model" and a "base model," ensuring high dialogue quality. Using this builder, the authors construct FURINA-Bench, a comprehensive benchmark that unifies established and synthesized characters and provides systematic analysis on reasoning ability and RP hallucination. Key findings include performance disparity between established and synthesized roles, and a critical trade-off where reasoning capability improves RP performance but exacerbates RP hallucination.

### Strengths
- The FURINA-Builder pipeline is highly innovative. It is the first RP benchmark construction tool that allows users to fully customize crucial elements such as role definition, scenario pool, dialogue structure, and evaluation dimensions (Section 3.1, 3.2). This flexibility directly addresses the critical issue of existing benchmarks becoming quickly obsolete and enhances the applicability of RP evaluation across diverse real-world applications (e.g., custom NPC design).

- FURINA-Bench is a robust benchmark that pioneers the unified evaluation of Established and Synthesized characters within a group chat setting. The systematic inclusion of reasoning and the dedicated analysis of RP hallucination lead to novel and valuable insights, particularly the discovery of the performance-vs-hallucination trade-off in reasoning models.

### Weaknesses
- Limited Comparison to Relevant Prior Work (Missing Benchmarks): The paper’s related work section and comparison table (Table 1) are incomplete, lacking key references in the RP field. Specifically, the authors fail to compare FURINA-Bench against the prominent RoleBench dataset (from RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models. ACL 2024). Furthermore, in characterizing the "Dynamic" features (persona changing across time), the work omits a comparison to methods focused on character transfer, such as Neeko (Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent. EMNLP 2024). A comprehensive survey and comparison are necessary to properly position FURINA-Bench's unique contributions.

- Insufficient Evaluation of Dedicated RP LLMs: The experimental evaluation is primarily focused on general-purpose LLMs (e.g., o3, DeepSeek-R1, Llama-2/3) and lacks assessment against specialized RP LLMs. To fully demonstrate the benchmark's differentiating capability, the authors should evaluate dedicated RP models like RoleLLaMA, Neeko, or other Character-LLMs built upon Llama-based architectures. Evaluating only general models might underestimate the current SOTA performance in RP.

- Lack of Deeper Analysis on Reasoning Utility: The paper observes that "reasoning enhances RP performance" but lacks a deeper analytical explanation for this finding (Line 455). The authors do not elaborate on why the inclusion of a large reasoning model in the pipeline (or the reflective reasoning dimension) leads to an improved role-playing fidelity. Is the improvement due to better internal consistency, better adherence to complex plot points, or simply generating longer, more elaborate responses? A detailed qualitative and quantitative breakdown is needed to justify the strong claim regarding the role of reasoning.

### Questions
The experimental evaluation primarily focuses on general-purpose and large reasoning LLMs (such as DeepSeek-R1 and the Qwen series) to establish the initial performance on FURINA-Bench. However, a more critical validation of the benchmark's differentiating power is its ability to assess specialized Role-Playing Agents or LLMs.

How do other dedicated RP LLMs, such as RoleLLaMA, RoleGLM, Neeko, or other Character-LLMs, perform on FURINA-Bench?

Specifically, providing a comparative analysis of these specialized models against the currently observed top performer, the large Reasoning Model (DeepSeek-R1), is essential. Such a comparison would not only validate the benchmark's ability to distinguish between dedicated and general models but also provide the community with a clearer understanding of the true SOTA performance boundary in the customized RP evaluation setting introduced by FURINA.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes the FURINA-Builder, a pipeline to create role-playing benchmarks, and the FURINA-Bench, a specific benchmark created with this system.

The actors:
- Evaluated model: $M_{source}$.
- Baseline model: $M_{base}$, GPT-4.1 in FURINA-Bench.
- Judge model: $M_{judge}$, GPT-4.1 in FURINA-Bench.
- Scene character model: $M_{scene}$, Qwen3-235B-A22B in FURINA-Bench.
- Director model: $M_{director}$, Qwen3-235B-A22B in FURINA-Bench.
- A model that checks hallucination-related keywords in  $M_{judge}$ CoT: $M_{checker}$

The pipeline is the following:
1. The set of characters with public/private attributes is defined. Characters might be either manually or automatically generated.
2. Scenarios are extracted from a set of books.  
3. A test character is inserted in a scenario, where it communicates with the original characters from this scene, played by $M_{scene}$.
4. $M_{director}$ decides which character speaks in the current conversation turn. It might be $M_{scene}$ or $M_{source}$ + $M_{base}$.
5. If it is a test character, the utterance is generated both by the evaluated model $M_{source}$ and by the baseline model $M_{base}$.
6. Both utterances are scored by $M_{judge}$. First, it selects the most relevant evaluation dimension for the current turn. Second, it outputs a number from the 5-point Likert scale, where 1 indicates a strong preference for $M_{source}$ and 5 indicates a strong preference for $M_{base}$, judging by the selected evaluation dimension. If the score is 4 or 5, the $M_{base}$ response is stored in the conversation history.
7. Finally, all the scores are aggregated and normalized, and this is how the final leaderboard appears.

The FURINA-Bench specifics:
- 2 languages: English and Chinese
- At least 10 scores for each character and each dimension
- Specific models (see above)
- 5 specific evaluation dimensions
- 20 test characters, ~1500 conversations, ~7000 test utterances

Human validation:
- Accuracy of $M_{judge}$ dimension selection (with 5 specific evaluation dimensions), 1000 samples.
- Pearson correlations of $M_{judge}$ scores with human scores in 5 specific evaluation dimensions, 400 samples.

Results:
- Bigger models are better.
- Reasoning models are also better.
- The model receives better scores when provided with established characters instead of synthesized.
- "Reasoning improves RP performance but amplifies hallucinations."

### Strengths
Overall, the paper is well organized. Figures are clear, and the appendices are extremely comprehensive (prompts, examples, annotation UI). It's nice that it has evaluations not only for English.

The pipeline makes sense, and the actors are clearly defined. Overall, the design of the self-play framework is reasonable.

### Weaknesses
Major points:

1. The main weakness: it's not clear what problem this pipeline/benchmark actually solves. Usually, in all role-playing applications, models interact with **users**. The primary complexity in building a role-playing benchmark lies in user simulation. This paper just removes users from the picture. The remaining self-playing framework evaluates whether models are good at interacting with each other. But why should anyone care about that?

2. Another significant problem is in the positioning as a "first RP benchmark builder". It claims that users can customize a set of characters, scenarios, and evaluation criteria. However, isn't it the case for almost any other open source benchmark?

3. Authors claim that they "conduct human evaluations, which confirm the builder’s effectiveness and benchmark’s soundness". But all human evaluations use the specific evaluation dimensions from a specific benchmark. So they just can't validate "builder’s effectiveness". Overall, the "builder" was used to build only a single benchmark.

4. The evaluation schema is a standard pairwise schema. For instance, it was used in the [RPBench-Auto](https://www.boson.ai/blog/rpbench-blog). Overall, I think that the originality of the approach is very limited.

5. The paper is too verbose on non-important things and not verbose enough on important things. For instance, what is the inter-annotator agreement? Without it, the reported accuracies/correlations are hard to interpret.

Minor points:

1. The section about the "emerging Pareto frontier" is very weak. It "emerges" only because of the single outlier, GPT-4o.

2. The whole "hallucination" story is not validated in any way. The accuracy of the hallucination checker is not reported.

3. The evaluation model is always GPT-4.1, and there are no ablations or robustness checks about that.

4. There are potential leaks in the evals: Qwen3-235B-A22B is both source/director/scene, and it is reported in the leaderboard table. This could advantage models in contexts in their own style and violates the strict independence of eval data.

5. The procedure of inserting the base model utterance instead of the target model utterance if the judge's score is 4 or 5 is questionable. Measuring cascading errors is important, and it is not clear why we would want the history to always be perfect.

### Questions
See the questions in the "Weaknesses" section.

Additional questions:

1. What is "CustomRPBench"? It appears in many figures. Sounds like a deprecated name.

2. How were the models selected for evaluation? For instance, why are there no Gemma models?

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
This paper addresses the limitations of existing role-playing (RP) benchmarks, which are often static, narrow in scope, and quickly become obsolete. The authors introduce FURINA-Builder, a novel multi-agent collaboration pipeline designed to automatically construct fully customizable and scalable RP benchmarks. This pipeline leverages a well-constructed character-scene pool for simulation, using a Director Model and a Judge Model for dynamic selection during dialogue generation. This process yields diverse, high-quality dialogue data annotated with fine-grained evaluation dimensions.
Using this builder, the authors develop FURINA-Bench, a new comprehensive RP benchmark that, for the first time, integrates both "established" and "synthesized" characters within group chat scenarios. Extensive evaluations of cutting-edge LLMs on this benchmark reveal several key insights: 1) o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. 2) The reasoning capabilities of models, while improving RP performance, also significantly increase RP hallucinations. 3) The paper uncovers a Pareto frontier between RP performance and reliability (low hallucination rate), which has significant implications for future RP model development.

### Strengths
1. Novelty of Methodology: The primary contribution is FURINA-Builder itself—a shift from creating a static benchmark to providing a dynamic benchmark builder. This is a significant conceptual advance in the RP evaluation space, effectively addressing the fundamental problem that existing benchmarks (like CoSER, CharacterBench) struggle to adapt to new characters, scenarios, and rapidly evolving LLM capabilities. This "benchmark for benchmarks" design is highly forward-looking.
2. Sophisticated Pipeline Design: The multi-agent collaboration pipeline of FURINA-Builder is well-designed. The "Selection" phase (Figure 1) is particularly clever: the Judge Model first determines the most appropriate evaluation dimension (Context Reliance, Factual Recall, etc.) for the current turn before comparing the source (Msource) and base (Mbase) models. This mechanism not only ensures high-quality dialogue trajectories (by filtering out inferior responses) but also automatically labels each data point in the benchmark with its most relevant, fine-grained evaluation criterion, achieving two goals at once.
3. Comprehensive Benchmark and Insights: The resulting FURINA-Bench is itself a high-quality resource. It is the first to unify the evaluation of "established" and "synthesized" characters in a multi-character group chat setting, which is crucial for simulating realistic virtual worlds (e.g., game NPCs). More importantly, the experimental analysis based on this benchmark yields valuable and novel conclusions. The discovery of the Pareto frontier between RP performance and reliability (Figure 4) and the non-intuitive trade-off that "reasoning capabilities exacerbate hallucinations" (Figure 2) are both critical findings.

### Weaknesses
1. Heavy Reliance on LLM-as-a-Judge: The entire FURINA-Builder pipeline (for building the benchmark) and the FURINA-Bench evaluation protocol (for testing models) are heavily dependent on an LLM (specifically, GPT-4.1) as the judge. Although the authors provide human validation in Section 4.4 (Tables 2 & 3), the results (e.g., Pearson correlations around 0.6 in Table 3) indicate that the LLM judge is only moderately, not perfectly, aligned with human judgment. This implies that the benchmark's data quality and the subsequent evaluation results may systematically inherit the biases of GPT-4.1. For instance, it remains questionable whether GPT-4.1 can truly and accurately assess more subjective dimensions like "Reflective Reasoning" (RR) or "Preference Alignment" (PA).
2. Concerns on Cost and Scalability: The authors claim FURINA-Builder can construct benchmarks "at any scale," but the multi-agent process appears computationally expensive. Generating a single test utterance requires multiple calls to powerful LLMs (Director, Mscene, Msource, Mbase, Mjudge). The paper provides no estimation of the computational resources (e.g., API costs or token consumption) required to build FURINA-Bench. This potentially high cost could significantly hinder the community from adopting FURINA-Builder to create their own "customizable" benchmarks, thereby weakening the practical utility of the core "builder" contribution.
3. Oversimplified Definition of RP Hallucination: In Section 5.4, the paper divides RP hallucinations into "EC hallucination" (a factuality hallucination, measured by the FR dimension) and "SC hallucination" (a faithfulness hallucination, measured by the CR dimension). This demarcation seems overly simplistic and arbitrary. An "established character" (EC) can certainly produce hallucinations unfaithful to the current context (CR), just as a "synthesized character" (SC) can fabricate real-world facts (FR). Forcibly binding character types to specific hallucination types and measuring them via only two evaluation dimensions likely fails to capture the full spectrum of RP hallucinations (e.g., the more common OOC - Out of Character).

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
