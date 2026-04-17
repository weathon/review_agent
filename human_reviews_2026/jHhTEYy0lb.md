# StoryCoder: Bridging Narratives and Formality for Code Generation in LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
The code generation capabilities of large language models stem largely from pretraining on structured code patterns and their strong context-based reasoning abilities. However, previous research on code generation has primarily focused on using short, fragmented, instruction-like prompts, which often fail to encourage contextual understanding. Inspired by the way humans organize fragmented information into coherent explanations, we propose a new method that reformulates coding problems as natural language narratives to promote integrative thinking. To this end, we introduce StoryCoder, a framework that reformulates code generation prompts into narrative text. Our results show that rich contextual expressions in natural language can enhance code generation performance and lead models to adopt consistent and structured problem-solving strategies. We quantitatively demonstrate that our method provides integrative information not captured by simple rephrasings and guides models to adopt correct algorithms and implementation strategies, thereby improving code generation performance. Experiments on three benchmarks, HumanEval, CodeForces, and LiveCodeBench, show an average improvement of 28.6% in the precision of zero-shot pass@10.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces STORYCODER, a narrative-based prompting framework for code generation with LLMs. The key idea is to reformulate short, instruction-like programming problems into natural-language narratives, aiming to enhance contextual understanding and algorithmic reasoning. The method converts each problem into a structured story composed of three sections: Task Overview, Constraints, and Example Input/Output. Each reformulation is paired with a narrative genre (e.g., fantasy adventure) and an algorithmic category (e.g., dynamic programming). The authors claim that these narratives help LLMs “integrate” fragmented information and reason more effectively, leading to better code solutions.

### Strengths
1. Clarity: The paper is generally well-written, structured clearly, and easy to follow. Figures (especially Fig. 1 and Fig. 2) provide an intuitive understanding of the proposed pipeline.
2. Empirical thoroughness: The experiments cover multiple benchmarks and a broad range of models. The authors also conduct detailed analyses of algorithm coverage, coherence ablations, and genre misalignment, which show effort toward understanding why the method works.

### Weaknesses
1. Naive methodological design and Limited causal attribution: The central contribution—reformulating code problems into narratives—is conceptually shallow. While improvements are reported, it remains unclear why narrative reformulation helps.
Gains might simply result from longer prompts or additional redundancy, rather than from true “integrative reasoning.”
2. Overstated cognitive connection: The discussion invokes cognitive science theories (e.g., mental models, analogical reasoning) without providing empirical evidence that such cognitive mechanisms are truly mirrored in LLM behavior. The analogy to human reasoning feels more rhetorical than scientific.
3. Evaluation scope and baselines: Baselines such as paraphrasing, chain-of-thought (CoT), or planning-style prompting are mentioned but not fully compared under equivalent prompt lengths or instruction structures. Moreover, the authors use repeated sampling as the main baseline, which is weak compared to recent structured reasoning or code planning methods.

### Questions
1. Prompt length control: Have the authors controlled for the increased token length between narrative and baseline prompts? Could the improvement stem simply from more tokens and richer context rather than from narrative structure itself?

2. Baseline diversity: Why are more competitive prompting methods (e.g., program sketch prompting, self-planning, step-by-step reasoning) excluded? Would STORYCODER still outperform these under equivalent length and structure?

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
The paper introduces StoryCoder, a framework to reformulate code generation prompts into richer, contextual expression. The experiments show that this translation leads to performance improvements across various benchmarks.

### Strengths
- StoryCoder is a simple prompting method to translate coding instructions into narrative-style instructions. The paper is clear written and presents a wide range of experiments to support its effectiveness.
- The results show significant performance improvements over using classical coding instructions, especially in settings where a proprietary LLM is used for translating into narrative-style problems and a smaller language model is employed as the solver model.
- Translating code instructions to narrative-style prompts effectively helps the language model to choose the appropriate algorithm to solve the underlying problem. Further, it additionally reduces the error rates when it comes to implementing the corresponding problem.

### Weaknesses
- The paper may be considered incremental.
  - This problem of translating coding challenges into narrative-style text has been already studied in Haller et al. (2024) (https://aclanthology.org/2024.lrec-main.1111.pdf). The findings from this paper are not discussed and should be at least mentioned in the related work section for self-containment.
  - I find it confusing that the authors explain in Section 4.2. which model actually writes the narratives because I was assuming until here it is also the model that solves the coding task. I would rather read this in the methodology section that a proprietary model required and that discuss the downsides what happens if the narrative-translation model is smaller. Given that, the method heavily relies on a strong LLM in the first place. When looking at the results, the only benefit is then for cases where we want to use smaller models, however, we still need a strong LLM to translate into narrative-style problems.

- The evaluation / experiments may be considered to have flaws:
  - The authors filter random subsets of the original benchmarks and I don't see a reason why to do so. Further, there is no explanation what kind of question, e.g. in HumanEval are considered invalid.
  - The results in Table 1 depict scores for a combined settings, namely, using both, narrative-style and original instructions. While the authors account for this in the baseline approach by sampling twice as much results, I would like to see how the proposed approach - transforming original instructions into narrative-style instructions - performs on its own.
  - The results are good for settings in which we use a strong LLM anyway and then want to use smaller models as solvers. The results of StoryCoder are not improving compared to when using a strong LLM directly (Gemini 2.5 Flash, GPT-4.1).

- Selected conclusions derived by authors may be considered incorrect or incomplete:
  - The coverage ratio in Figure 4 is defined as "the fraction of problems with at least one correct solution", so basically the pass@k metric. I would rename this for better understanding.
  - I think the key statement in line 377 is overestimating some takeaways, e.g., I don't think the authors show until here that the generated code is actually more structured and follows a stepwise design. Further, an increase in correct choice of algorithms logically means a reduction in implementation error rate. However, I find it here confusing if we are actually looking at the ratio of chosing the correct algorithm (according to the text) or at the ratio of solving the actual problem (according to the Figure). If it is the ladder, the solve ratios are way higher than in Table 1, which is confusing.
  - Section 5.4 does not mention a complete list of categories to choose from and would help the reader to understand why certain categories have been selected or chosen by the authors.

### Questions
- Clarification on lines 159-165: How does it differ? I assume you refer to repeatedly sampling from Section 3.1. If so, I suggest to rephrase this part because using repeatedly sampling for both, instructions and answers, may be confusing.
- Fix Mistral AI citation in line 206/207
- The authors often refer to additional information in the Appendix, however, the Appendix is not present.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces StoryCoder, a narrative-based prompting framework that reformulates coding problems into structured natural-language stories to improve LLM code generation. StoryCoder constructs each narrative with three components: (1) Task Overview, (2) Constraints, and (3) Example Input/Output. The narrative is then used by a solver model to generate code. Experiments conducted on HumanEval, LiveCodeBench, and CodeForces demonstrate that narrative prompting consistently enhances model performance, leading to higher pass@10 accuracy across all evaluated models. The authors further analyze 1) algorithm selection fidelity, 2) implementation error reduction, 3) narrative coherence effects, and 4) genre alignment influence on model performance.

### Strengths
1. Reframing code generation as storytelling is an interesting idea that bridges narrative cognition and programming reasoning, grounded in cognitive theories like mental models and analogical mapping. 
2. The evaluation covers 11 LLMs, including both closed-weight and open-weight models, tested across three benchmarks of varying difficulty to ensure strong validity.
3. The paper conducts in-depth analyses beyond task performance, decomposing results into factors like algorithm selection, implementation errors, and correctness. It further examines coherence and genre effects through controlled narrative variants and uses back-translation analysis to show genuine shifts in model reasoning, not just output variation.

### Weaknesses
**Major Concerns:**
1. In the introduction, the authors introduce the terms “algorithm” and “genre” in the context of narrative generation. However, these concepts are not formally defined in the methodology section. Later, the paper analyzes the effectiveness of each component, which becomes difficult to follow without clear definitions of these terms.
2. The descriptions of Repeated Sampling (RS) in Sections 3.1 (line 140) and 3.2 (line 162) are inconsistent. In Section 3.1, the authors define RS as sampling multiple responses for the same input. In Section 3.2 (line 162), they define RS again as generating multiple narratives that represent different input variants.
3. Following the previous confusion, Section 4.1 (line 247) states that the authors sample twice as many outputs, which adds more ambiguity. The reference to Equation 4 in this context is also missing.
4. The table 1 shows the results of open-source LLMs of combined setting for both narrative-only and narrative-question pair inputs, this introduce additional confusion of why doing this? Making the results more harder to interpret.
5. In table 1, the author only shows results of repeated sampling and their proposed narrative prompting, why there is no results of paraphrasing and Chain-of-Thought (CoT) which are all introduced in Section 3.1?
6. For the baselines, the author only includes non–prompt-optimization methods. Why were other prompt reformulation baselines, such as Structured Chain-of-Thought (SCoT, https://arxiv.org/abs/2305.06599), not included? SCoT was already mentioned by the author in the related work section. And I believe there are many other valid prompt-optimization for code generation baseline could be included such as Self-Planning Code Generation with LLM (https://dl.acm.org/doi/pdf/10.1145/3672456)
7. In the evaluation, the authors used a setting where narratives were generated by one model (Gemini-2.5-Flash), and each open-source model generated code based on the same narratives, as shown in Table 1. Later, they again argued that the self-solving setting is not suitable for open-source models because these models cannot produce valid narratives, and thus they used a different model (Qwen-2.5-32B) to generate the narratives. However, it is unclear why this additional setting is needed, since the same condition—using shared narratives—is already included in the table 1. This makes the logic difficult to follow.
8. In Table 2, the author reports the proportion of valid narratives generated by each open-source model but does not clearly define what constitutes a valid narrative or explain how narrative validity was assessed. In addition, it is unclear what the validity rate of narratives is for the self-solving LLM shown in Table 1.
9. In Section 5, the lack of formal definition of algorithm and genre in narrative generation makes the analysis hard to follow. For example, “Misaligned genres” are manually chosen by the authors. There’s no objective way to determine what counts as aligned vs. misaligned, which hurts reproducibility.
10. In Section 5, most of the analysis were based on LLM-as-a-Judge, there is a lack of human validation to show the correctness of LLM-as-a-Judge for each analysis.

**Minor Concerns:**
1. The date filtering criteria are missing from the main content, which I think is important for readers to understand the validity of the results. (The appendix is also missing from the submission.)
2. I believe there are many other related works on prompt optimization beyond code generation, which could be useful for better understanding this research direction and strengthening the motivation.

### Questions
1. What is the search space of algorithm and genre in narrative generation?
2. For baselines, why do you not include some reasoning model for comparison? I believe reasoning model may already has the capability to reflect on the input question and reformulate it in their internal reasoning steps.
3. Could you explain why including a story improves the quality of output code? From the example in Figure 2, the generated story does not seem to contain information that directly helps solve the coding problem (just my opinion). It might be clearer if the authors highlight which parts of the story provide relevant information.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work propose STORYCODER, a framework that transforms code generation questions into narratives (stories) before prompting LLMs. The experiments showed that prompting with natural language narratives yield substantial improvements across many LLMs on several benchmarks.

### Strengths
- The idea is converting questions into narratives for code generation is simple, yet quite effective. 
- The improvements are substantial and consistent across many models.

### Weaknesses
- Although the idea of transforming questions into narratives is new for software domains, it has been explored previously in Story of thought (SoT) (Sadiri Javadi et al., 2025), which is also cited by the authors. Due to the high similarly between the two, I would expect a more thorough discussion and empirical comparison to highlight the strengths and weaknesses of each approach.
- The experimental setting is questionable. For LiveCodeBench, why did the authors only filter out 175 out of the original 1,055? Appendix B2 states that this is to avoid data contamination, but it is not convincing since StoryCoder does not perform any additional training. Similarly, CodeForces was also filtered to have 265 out of 10k samples. Given that the text length must be less than 1k, is it true that StoryCoder cannot generate reliable narratives for long problems?
- Given that the empirical evaluation is only conducted on rather small subsets of existing benchmarks, it is unclear if StoryCoder can generalize well.
- What is the running time overhead of StoryCoder compared to baselines like repeated sampling and SoT?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
