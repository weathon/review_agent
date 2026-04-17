# REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?

- Decision: Accept (Poster)
- Scores: 6, 2, 4

## Abstract
Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who are the groups that robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark that systematically models vague REs grounded in pragmatic theory (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 36.9\%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompts, chains of thought, and in-context learning. By tackling the overlooked issue of vagueness, this work contributes to the research community by advancing real-world task planning and making robots more accessible to non-expert users, e.g., the elderly and children.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is the first to identify and study the challenge of LLM-based robot task planning when vague human instructions arise from referring expressions (REs). The authors propose a novel benchmark, REI-Bench, to evaluate the impact of such vagueness and demonstrate that implicit REs can significantly degrade the performance of planners. To address this, they introduce TOCC, a simple yet effective approach to mitigate the effects of vagueness, achieving superior performance compared to existing baselines.

### Strengths
1. The paper introduces a noval benchmark to simulate vagueness in human instructions which allows for a standardized evaluation of how vagueness impacts robot task planning.
2. The problem of interpreting vague instructions is crucial for making robots more accessible. This work has practical implications in real-world environments.
3. This paper conduct extensive experiments on several baseslines and demonstrates the proposed TOCC outperforms them.

### Weaknesses
1. The paper focuses primarily on a small subset of REs, specifically coreferential vagueness. Can TOCC be extended to address more descriptive phrases such as "the heavy thing"?
2. The evaluation is limited to language models with relatively small model sizes. Testing on more powerful models as a point of comparison would help contextualize the results better.

### Questions
1. Can the framework be extended to address other types of vagueness beyond coreferential vagueness? How easily can it be extended, and what would be the performance implications?
2. Will modern, larger models be able to resolve the vagueness challenge more effectively?

### Soundness
3

### Presentation
3

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
The paper shows that  referring expressions (REs),a source of coreferential vagueness would significantly degrade LLM-based embodied task planning. It introduces **REI-Bench**, a benchmark that systematically varies RE vagueness and dialogue context, and reports up to a 36.9% absolute drop in success for off-the-shelf planners (SayCan, LLM+P). The authors propose Task-Oriented Context Cognition (TOCC), a lightweight preprocessing step that resolves REs before planning, reducing object-omission errors across nine difficulty levels. The work surfaces an overlooked failure mode in HRI and offers both a rigorous evaluation protocol and a practical, immediately deployable mitigation.

### Strengths
1. Clear problem framing motivation. The paper isolates  coreferential vagueness from referring expressions (REs) as a specific, under-explored failure mode in embodied task planning, grounding the setup in linguistic theory (RE vs. DE, bridge inference).

2. Salient empirical findings. The work documents  consistent, sizable drops in task success under implicit REs and multi-turn dialogue, and includes a human reference to highlight the performance gap in natural conversational settings.

### Weaknesses
1. I have substantial concerns about how the benchmark is constructed, this is the main issue, as the dataset’s design directly affects the trustworthiness of the empirical results.
  * Sampling/selection bias: Filtering out Pick Two & Place and seeding only from tasks successfully executed by LLaMA3.1-8B+SayCan simplifies the benchmark and risks biasing task distribution; no unfiltered vs. filtered comparison is reported to rule out selection effects.
  * The Ambiguous-Name perturbation (brand/person name collisions) is too narrow and misses common home scenarios (homophones,, family/pet/room names, and visual look-alikes like pot/pan/bowl), limiting generalization beyond the injected pattern. More ecologically valid generators of referential ambiguity would strengthen the dataset’s realism and external validity.
  * Although the paper argues vagueness is more common for non-experts (elderly/children), the dataset does not model age-specific linguistic phenomena nor provide any evidence to test this claim.

2. “First coreferential-vagueness planning benchmark” claim lacks quantitative positioning. The paper does not provide a systematic, quantitative comparison against prior ambiguity/RE/coreference datasets across robotics, NLP, and HRI, nor a detailed dataset-level contrast table.

3. TOCC is under-specified at the algorithmic level. There is no precise interface or flow diagram: inputs/outputs, whether TOCC emits a disambiguated “clear instruction”, and whether it returns executable targets

4. Confounds in empirical comparisons. Although §3.3 claims CoT/ICL substantially lengthen prompts, AP/CoT/ICL are compared to TOCC without length comparisions, normalizing token cost/latency. I also concers about the fairness for baselines, which are under-specified.
Baseline improvements could be stronger with gated AP, short/segmented CoT, or ICL distillation; these are not reported, so TOCC’s margin may be overstated.

### Questions
1. Please address the concerns summarized under Weakness 1.
2. Can you provide a quantitative comparison table vs. prior ambiguity/RE/coreference datasets across Robotics/NLP/HRI ?A clear table and analysis would show REI-Bench fills a unique, previously uncovered evaluation axis (RE-driven vagueness in planning).
3. Provide an explicit and detailed description of TOCC, such as pseudocode or  data-flow diagram. The pipeline should be precise enough to reimplement.
4. Compared with stronger  variants of baselines with length comparisions, normalizing token cost/latency.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces REI-Bench, a benchmark designed to isolate and analyze how vagueness in referring expressions within human instructions affects LLM-based robot task planners. The authors automatically construct the dataset based on ALFRED, defining three levels of referential-expression difficulty and three contextual conditions. They evaluate six 7B–8B LLMs using two task planners, SayCan and LLM+P, and further investigate the causes of performance degradation under vague instructions. To mitigate the issue, they propose a simple prompting strategy, task-oriented context cognition, which decouples instruction understanding from plan generation. Overall, REI-Bench frames embodied task planning as a language understanding challenge and provides a controlled, reproducible framework for studying it.

### Strengths
* Highly practical research: Because people naturally use referring expressions, robots and embodied agents that coexist with humans must be able to interpret and resolve such expressions to ensure safe, natural, and efficient collaboration. In this context, REI-Bench provides a crucial foundation for evaluating whether embodied agents can truly comprehend human-like referring language which is a core capability required for real-world human–robot interaction.
* Appropriate benchmark design: The 3×3 structure of REI-Bench combining three levels of referring-expression difficulty with three context types enables a clear and fine-grained analysis of language understanding capabilities across varying degrees of linguistic vagueness, contextual noise, and dialogue-memory constraints.
* Transparent descriptions: The authors provide comprehensive documentation of the dataset construction process, detailing all prompts, constraints, and post-processing rules in Appendix B. Their three-stage pipeline comprising context generation, context processing, and referring-expression replacement ensures both reproducibility and systematic control.
* Multi-faceted experiments and analyses: The experiments and analyses are well structured, clearly showcasing the strengths of REI-Bench. By combining two planning frameworks and six LLMs, the study provides a comprehensive factorial analysis with detailed results for each condition and accompanying ablation studies. The findings are visualized across referring-expression levels and context variations, supplemented by human baselines and an interpretable taxonomy of errors, including object omission and execution failure.

### Weaknesses
* Potentially biased seed instructions: The dataset’s seed instructions were created only from tasks that LLaMA 3.1-8B + SayCan successfully executed in AI2-THOR. Tasks such as Pick Two & Place were simply excluded as "not reliably completed." This means the benchmark originates exclusively from simpler, one-object, short-horizon tasks, producing an inherent bias toward language situations that small models already handle. Statistics of instructions per ALFRED task types need to be presented. 
* Outdated planners: Both SayCan and LLM+P represent early-generation LLM-based planning frameworks. The evaluation excludes more recent paradigms that incorporate elaborative reasoning and context management, which could have yielded more informative and comparative experimental results. Consequently, the paper’s conclusions about the “limits of LLM-based planning” may not fully generalize to modern planning agents.
* Tested with small LLMs only: All models evaluated in the experiments are small-scale LLMs with 7B - 8B parameters. Such models are known to exhibit unstable discourse reasoning and weak coreference resolution. Consequently, the sharp performance drop from explicit to implicit referring expressions may reflect model capacity limitations rather than the intrinsic difficulty of the benchmark itself. The absence of recent larger models prevents any scaling analysis that could reveal capacity thresholds for robust language understanding.

### Questions
Please refer to the weaknesses and address the issues.

### Soundness
3

### Presentation
3

### Contribution
2
