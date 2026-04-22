# ContextBench: Modifying Contexts for Targeted Latent Activation and Behaviour Elicitation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 6

## Abstract
Identifying inputs that trigger specific behaviours or latent features in language models could have a wide range of safety use cases. 
We investigate a class of methods capable of generating targeted, linguistically fluent inputs that activate specific latent features or elicit model behaviours. We formalise this approach as *context modification* and present ContextBench - a benchmark with tasks designed to assess the capabilities of context modification methods across core capabilities and potential safety applications. Our evaluation framework measures both elicitation strength (the degree to which latent features or behaviours are successfully elicited) and linguistic fluency, highlighting how current state-of-the-art methods struggle to balance these objectives. We develop two novel enhancements to Evolutionary Prompt Optimisation, a gradient-based token-editing method: LLM-assistance and diffusion model inpainting, achieving strong performance in balancing elicitation and fluency. We release our benchmark here: https://github.com/lasr-eliciting-contexts/ContextBench.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates context modification, the generation of fluent inputs aimed at eliciting targeted model behaviors, and proposes a dataset for context modification ability evaluation. Furthermore, the authors propose two enhancements to a white-box context modification method, EPO, enabling its variants to achieve SOTA performance in both fluency and elicitation.

### Strengths
1. This paper incorporates several frontier techniques, including SAE and LLaDA.
2. In the proposed benchmark dataset, the authors include diverse scenarios and tasks to comprehensively evaluate the performance of context modification methods.

### Weaknesses
1. The text in Fig. 1 is hard to read. Since the figure is not complicated, I recommend using a larger font size.
2. I have reservations about the soundness of both contributions of this paper.

    i) For the proposed dataset, the number of samples seems too small to adequately represent the three tasks, especially given the additional subtasks under each. Moreover, since the dataset is divided by tasks, it is unclear whether these tasks comprehensively capture the concept of context modification, or whether they are representative enough for context modification. Additionally, using cross-entropy to measure text fluency may be somewhat outdated, despite its high correlation with human judgments. The authors may consider complementing it with learned metrics, which can really measure the deeper semantic appropriateness and contextual coherence.

    ii) For the two enhancement to EPO, first the authors should briefly introduce the core idea of EPO in the introduction. Second, without an explanation of why you chose EPO for improvement, these enhancements would seem random or incremental. Third, the implementation is relatively simple and heavily relies on GPT-4o and LLaDA. Will these variants still work well without such powerful auxiliary models? If not, why not directly build on these models instead?
3. The experimental comparison is not convincing. A GPT-4o-assisted variant is naturally more fluent than its baseline and more task-effective than GPT-4o itself. Since there are various prompt engineering methods, is it possible to adapt some SOTA approaches to the tasks introduced and compare the proposed methods with them?

### Questions
1. Could you briefly explain the mechanisms of honey-potting techniques mentioned in L38? I was wondering why you use this example right after stating your research focus.
2. Can I interpret context modification as a red-teaming process leveraging LLMs' prompt sensitivity, or as a subtask of prompt engineering? If so, what makes context modification stand out as a significant task?
3. Some terminology should be made consistent throughout the paper. For example, LLaDA vs. LLaDa.

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
This paper introduces ContextBench, a new benchmark designed to evaluate methods for context modification in language models, which involves generating linguistically fluent inputs that activate specific latent features or elicit targeted behaviors. This research is motivated by AI safety, seeking to identify contexts that trigger problematic model behaviors before deployment, such as sandbagging or bypassing refusal mechanisms. The authors formalize this approach and present ContextBench with three task categories: SAE Activation (targeting internal features), Story Inpainting (testing fluency in natural contexts), and Backdoors (recovering hidden triggers). To address the trade-off between elicitation strength and linguistic fluency, the authors develop two novel enhancements to Evolutionary Prompt Optimisation (EPO), demonstrating superior performance compared to vanilla EPO and black-box methods like GPT-4o. The overall goal is to advance white-box techniques for interpreting and controlling language model behavior through careful input design.

### Strengths
- This paper tackles a relevant problem in AI safety literature and present a benchmark that can be potentially useful for the community. 
- The benchmark includes tasks representative of practical safety use cases, such as finding triggers for backdoored models.
- The benchmark follows a multi-objective evaluation approach for context modification. 
- The authors provide an (anonymized) github repo with instructions and the results seem to be fully reproducible.
- The paper develops two novel variations of Evolutionary Prompt Optimisation that empirically Pareto dominate previous methods in balancing efficacy and fluency.

### Weaknesses
- There is a sort of "inconsistency" on the mentioned contributions: in the abstract, the authors start by focusing on investigating a class of methods (which one?), while in the introduction they clarify by stating they propose a new benchmark and "two state-of-the-art methods that empirically Pareto dominate previous methods on this task". Which task?
- The paper is not particularly easy to read in some parts; the flow of the paper could be significantly improved. A few more concrete examples, although the same thing applies for the actual flow ideas:
	1. Why is there a "Background" and "Related Work" section? It looks like they belong within the same section.
	2. There's a subsection (4.1) with no content on it, only on its subsubsections. Can't
 they just be subsections on their own?
 	3. SAE is never defined in the main body of the paper, only on the appendices, Figure 13 where the role of the LLM is defined. EPO is defined both in the abstract and main body, but the acronym is used in the main body of the paper before being defined on page 3 (line 109). 
- One of the core contributions of this paper is the ContextBench benchmark, yet there is not discussion on relevant/related benchmarks that a future reader might want to look into as well. For example, how do the Story inpainting tasks relate to, or could be used on existing literature? On the same note, recent work on inpainting such as "Inpainting-Guided Policy Optimization for Diffusion Large Language Models" use mathematics datasets, could a paper like this benefit from using this benchmark as well?
- The paper proposes two proposed methods seem to rely quite heavily on EPO, however there is no description or details provided, or related work regarding this method. Simultaneously, there is very little detail provided on the two methods the authors propose. From reading the short descriptions provided, they seem like a trivial modification of EPO, but this perception might be incorrect and possibly arising from the lack of information provided. I believe three short paragraphs on two core contributions of a paper is not sufficient to fully understand such a core component of the research presented here.
- In my opinion, adding one or two additional methods to evaluate the two proposed methods against would be quite important. Relying solely on GCG, Human evaluations and GPT-4o as a baseline seems insufficient to me. A broad description of the participants/human evaluators would be desirable (age group, education level, etc.)
- The human evaluation was conducted using solely two annotators, which seems insufficient.
- [Suggestion] Increasing the font size on Figure 1 would improve readability
- [Suggestion] A table summarizing the results on the Backdoor task would be useful (i.e., not in the appendix, since it's a core result).

### Questions
- Why use cross-entropy as a fluency metric if, as the authors recognize, it is an imperfect approach? What alternatives should readers consider? Why not use perplexity as a model fluency metric?
- I am curious on the choice of GCG as a solution for the tasks in your benchmarks; I was aware of this method as a jailbreaking approach, designed to introduce "nonsensical" prefixes to prompts that would enable the generation of an unsafe output, or a non-refusal. Is there any research supporting the use of GCG for anything else other than jailbreaking?
- Is this an IRB-approved study (or equivalent, if the authors are not based in the U.S.)? If so, this information should be present in the paper

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a novel benchmark to assess the potential of linguistically fluent context modifications strategies to trigger targeted features in the latent space of LLMs. Subsequently, the authors introduce two enhancements to EPO to balance model elicitation and linguistic fluency.

### Strengths
- The authors created a novel, comprehensive, and practical safety benchmark for LLMs, a timely and valuable contribution. Backdoored models are particularly innovative.
- Clear paper providing a deeply quantitative and technical approach to safety evaluation
- Rich Appendix on data, model training and evaluation methods
- The LLM-improved EPO to increase fluency and in-painting strategy with diffusion language model
- Insightful read

### Weaknesses
- Valuable benchmark and innovative methodologies, but somewhat missing clear and actionable conclusions 
- Already dense paper, and more details on the SAE itself would have been welcome in Appendix

### Questions
/

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ContextBench, a benchmark for evaluating methods that generate linguistically fluent inputs to activate specific latent features or elicit targeted behaviors in language models, with potential applications to safety-relevant scenarios such as preventing jailbreaking and sandbagging. The benchmark comprises samples across three categories: (1) producing SAE Activation for selected semantic features; (2) Story Infilling to maximize a specific next word prediction for specific token pairs, and (3) Trigger detection for backdoored models. To improve current performances on these task, the authors propose two enhancements to the gradient-based Evolutionary Prompt Optimization (EPO) method: EPO-Assist, which periodically queries GPT-4o to generate fluent variations, and EPO-Inpainting, which uses the bidirectional diffusion model LLaDa to infill non-critical tokens while preserving high-activation tokens. Experiments demonstrate that these variants achieve better trade-offs between elicitation strength and fluency compared to vanilla EPO, GCG, and GPT-4o baselines. However, results on the backdoor detection tasks show only partial success, particularly struggling with multi-token triggers.

### Strengths
* Originality: ContextBench provides useful settings to evaluate the effectiveness of input synthesis methods across various settings that are novel and well-motivated for safety applications. The use of diffusion models for iterative editing represents a creative combination of existing methods to improve the EPO technique.
* Quality: The experimental methodology is generally sound with appropriate controls: multiple random initializations, statistical significance testing, human validation of the fluency metric, and comprehensive ablations across feature dimensions (density, vocabulary diversity, locality). The benchmark design shows careful consideration of task diversity and difficulty gradients.
* Clarity: The paper is provides clear motivation, explicit task descriptions, and good use of illustrative examples (e.g., Figure 4 showing how EPO discovers that a "celebrity" feature activates more on historical figures than the max-activating examples suggested). Implementation details are thoroughly documented in the appendices.
* Significance: Eliciting specific latent representations or predictions from language models is an underexplored area with promising real-world applications for detecting problematic model behaviors. The benchmark could enable systematic progress on this problem, with potential applications to security and interpretability research.

### Weaknesses
* While cross-entropy is validated against human judgments (ρ=0.92), the paper acknowledges it "promotes generic sentences, word repetitions, and creates dependencies on the specific LLM used.", as shown in the examples from Figure 4. The  threshold of 3-9 for cross-entropy also lacks principled justification. For the human evaluation part, only 80 examples were covered with just two annotators. A more robust fluency evaluation would strengthen claims, especially given that fluency is positioned as a core contribution. Authors should consider metrics like diversity (self-BLEU), semantic coherence (sentence embeddings), or larger-scale human studies to strengthen their claims.
* The paper documents multiple instances where methods exploit shortcuts rather than providing genuine insights: (a) inserting target words directly, (b) exploiting polysemy (medical vs. behavioral "rash"), (c) using conjunctions to reverse implications, (d) question/task switching in backdoors. While acknowledged, these issues fundamentally undermine the interpretability goal. The authors mention manual inspection and cross-entropy filtering as mitigation but provide no systematic solution or quantification of how often gaming occurs. This is a critical limitation for safety applications.
* Results show only 5.1% success rate for single-token password recovery (EPO) vs. 2.5% (GCG), with complete failure on multi-token sequences. For the auditing task, GPT-4o identified generated prefixes as outliers 100% of the time. These results suggest the proposed methods are not yet practical for real-world backdoor detection. The disconnect between token-by-token optimization and multi-token trigger sequences appears fundamental but is not adequately addressed. Moreover, the use of backdoored models from other papers is questionable, if these can be gamed by single-token triggers (e.g., "Ukraine" in line 451) that defeat in part the purpose of multi-token evaluation.
* In terms of clarity, useful results such as those presented in Figure 7 are often relegated to the appendix but directly referred from the main text, making the discussion harder to follow. The "Background" section also feels out of place in its current position, as it relates more to the methods presented in Section 5, rather than the presentation of tasks from Section 4. The examples of Figure 2 can be made generic to avoid forward references to the method, and the method can be introduced at the beginning of Section 5. In general, the writing can be refined and figure/table sizing optimized to warrant the inclusion of additional relevant contents from the appendix.

### Questions
Can you quantify how frequently specification gaming occurs across all tasks? What percentage of "successful" outputs actually provide interpretable insights vs. exploit shortcuts? This is crucial for assessing practical utility.

### Soundness
3

### Presentation
3

### Contribution
3
