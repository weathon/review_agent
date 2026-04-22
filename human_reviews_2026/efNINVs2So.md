# Discovering Novel LLM Experts via Task-Capability Coevolution

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6, 6

## Abstract
Frontier model developers aim to train models continually to possess emergent, diverse capabilities. 
To extend capabilities, the current pre-training and post-training paradigm requires manually starting training runs with static datasets or reward functions every time.
Addressing this limitation, our work pursues the insight that open-endedness (via the coevolution of models and tasks) can discover models with increasingly novel skills in a single run.
We introduce a new model development framework that extends coevolution to large language model (LLM) discovery, open-ended \textit{Assessment Coevolving with Diverse Capabilities} (AC/DC).
AC/DC evolves both LLMs via model merging and natural language tasks via synthetic data generation.
AC/DC discovers growing archives of LLMs that surpass the capabilities of larger LLMs while taking up less GPU memory.
In particular, our LLM populations achieve a broader Coverage of expertise than other curated models or baselines on downstream benchmarks, without \textit{any} explicit benchmark optimization.
Furthermore, AC/DC improves Coverage over time, continually innovates on tasks and models, and improves performance in multi-agent best-of-N selection.
Our findings highlight the potential of coevolution as a means of discovering broader sets of capabilities from base LLMs.
Overall, AC/DC brings us one step closer to a profoundly new paradigm of LLM development, where continual improvements to the diversity of model capabilities can be accelerated by leveraging existing models as stepping stones to increasingly powerful models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a novel LLM development framework called Assessment Coevolving with Diverse Capabilities. The framework aims to enable open-ended co-evolution between models and tasks, thereby automatically discovering a population of LLM experts with increasingly novel and diverse capabilities. This approach seeks to overcome the limitations of existing training paradigms that rely on static datasets and produce a single monolithic model.

### Strengths
1. Parameter/Resource Efficiency
The collective of models discovered using the proposed method outperforms larger-scale LLMs while using fewer total parameters, thereby saving GPU memory and computational resources.

2. Clarity of Presentation
The paper employs flowcharts and pseudocode to aid comprehension, allowing readers to quickly grasp the proposed methodology.

3. Comprehensive Analysis
The paper provides detailed qualitative and quantitative analyses and yields several interesting insights. For instance, the discovered models exhibit specialized and diverse capabilities, offering valuable guidance for future research.

### Weaknesses
1. The success of model merging depends heavily on empirical testing of initial seed model combinations, which may limit the overall effectiveness of the pipeline. Additionally, since answers in generation tasks are produced by LLMs, their correctness cannot be fully verified.

2. The current evolutionary scope is relatively limited. As shown in Figure 3, the evaluated capabilities belong to domains where the base models already perform well, making it unclear whether the proposed method can truly generate out-of-distribution (OOD) capabilities.

3. The benchmarks used in the experiments are relatively easy, with many models already achieving near-saturated performance. More challenging benchmarks are needed to better demonstrate performance improvements, for example using math tasks at the AIME or IMO level. The same applies to other domains.

4. The study would be more convincing if experiments were conducted on larger models, allowing observation of whether the proposed phenomenon scales with model size.

5. The proposed approach bears resemblance to using a Multi-Agent System (MAS) to optimize model capabilities. The authors should clarify how their method compares to MAS in terms of performance and innovation.

### Questions
Please refer to Weaknesses part.

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
2

### Summary
This paper presents AC/DC (Assessment Coevolving with Diverse Capabilities), a framework that jointly evolves large language models and synthetic tasks to automatically discover diverse expert models. The system combines evolutionary model merging, synthetic data generation, and quality diversity selection to develop populations of smaller models that collectively achieve broad skill coverage. Experiments show that these collectives can outperform much larger models, including GPT-4o, in benchmark coverage while using fewer total parameters.

### Strengths
- The combination of open-endedness and evolutionary model merging is novel and provides an interesting new direction for developing diverse and adaptive model populations.
- The empirical results are encouraging as they show performance and coverage improvements across several benchmarks and model families.
- The method achieves notable parameter efficiency, with collections of smaller models reaching or surpassing the performance of much larger single models.
- The qualitative analyses demonstrate emergent specialization and stylistic diversity among the evolved models, supporting the merits of the proposed method.
- The method achieves notable parameter efficiency, with collections of smaller models reaching or surpassing the performance of much larger single models.

### Weaknesses
- The paper does not report variance or confidence intervals, making it difficult to assess whether the performance gains in terms of coverage and Best-of-N improvement are statistically significant.
- The computational cost of the coevolution strategy is not addressed, making the efficiency claim incomplete.
- While AC/DC does not optimize for any benchmark, the synthetic task generator depends on a fixed "scientist" LLM that might produce tasks similar to benchmark data.

### Questions
- Have you verified whether the reported improvements are consistent across runs or fall within expected benchmark variance?
- How does the total computational cost of the coevolution process, including model merging and task generation, compare to standard fine-tuning or ensembling?
- How sensitive is the framework to the choice of the "scientist model" or to different prompting strategies during task generation?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors work present Assessment Coevolving with Diverse Capabilities (AC/DC) which is a framework that automatic discover multiple diverse LLM through open-ended coevolution of models and synthetic tasks. Their approach show that EvoMerge can create task forces that perform better than larger language models while using less parameters and manually curated expert ensembles. AC/DC not optimize for any downstream benchmark and achieve consistent improvements across multiple model families, with evolved populations showing wider coverage of capabilities and emergent specializations that validate discovery of complementary skills.

### Strengths
- Good open-ended approach of model merger algorithm
- Good use of minimal criterion coevolution

### Weaknesses
- Presentation can be better
- Related works can be improved

### Questions
Can we achieve the same result with a setting that involves only one agent?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AC/DC (Assessment Coevolving with Diverse Capabilities), a framework for discovering diverse LLM populations through coevolution of models via evolutionary model merging and synthetic tasks via LLM-generated data. The method evolves archives of both models and tasks, applying quality-diversity principles (DNS) and minimal criteria filters. Evaluating across multiple model families (Qwen2/2.5/3, DeepSeek, Llama3), the authors report that their evolved collectives of smaller models achieve broader Coverage on downstream benchmarks than larger models or manually curated experts, without explicit benchmark optimization. Key findings include parameter-efficient performance (N=8 evolved 7B models surpassing 72B models), emergent specialization patterns, and improved Best-of-N selection compared to baselines.

### Strengths
1. **Novel system-level integration**: First framework (to the best of my knowledge) combining evolutionary model merging with synthetic task coevolution, extending quality-diversity principles to LLM discovery in a principled way that addresses the limitations of training single static models.

2. **Multi-family empirical validation**: Evaluation across 5 model families (Qwen2/2.5/3, DeepSeek, Llama3) and 8 benchmarks provides breadth, with consistent improvements on 4 families demonstrating some generalizability despite Llama3 failures.

3. **Parameter efficiency evidence**: Concrete demonstrations that N=8 evolved 7B models surpass 72B models while using 22% fewer parameters (Qwen2: +14.59% Coverage improvement) support distributed specialization hypothesis.

4. **Emergent specialization analysis**: Figure 3 shows evolved models develop distinct category-specific performance profiles without explicit domain assignment, validated by qualitative examples (Figures 4-5) demonstrating complementary capabilities.

5. **Open-ended improvement**: Figure 1 and Appendix D.3 demonstrate that coevolution leads to continually improving performance and task diversity over generations, supporting the open-endedness claim.

6. **Comprehensive ablations and analysis**: Appendix D.1's ablation study, D.2's task force selection comparison, D.3's Vendi score analysis, and detailed lineage trees (Figures 8-9) provide thorough characterization of system components and emergent behaviors.

### Weaknesses
1. **Limited technical novelty**: AC/DC is primarily an *integration* of existing techniques (EvoMerge, synthetic data generation, DNS selection) rather than fundamental algorithmic innovation. The mutation operator (SVD perturbation) and gibberish filter are incremental additions. The contribution is valuable but more engineering than methodological breakthrough.

2. **Evaluation circularity concerns**: Task force selection maximizes Coverage on synthetic tasks generated by the same system, then evaluates on benchmarks. While claiming "no explicit benchmark optimization," the 72B scientist LLM presumably has knowledge of benchmark-style problems. Is the synthetic-to-benchmark transfer genuine generalization or implicit optimization via the scientist LLM's training data?

3. **Model family dependence unexplained**: Success on Qwen (4 families) but catastrophic failure on Llama3 (Appendix C: 46.04% vs. 51.13% at N=3) raises critical questions. The authors attribute this to "seed model compatibility" but provide no diagnostic criteria for predicting success. What percentage of model families will fail? How can practitioners know beforehand?

4. **Missing key baselines**: No comparison to vanilla EvoMerge *without* coevolution. Table 3 compares only to DNS/CQD (both using benchmark-specific training), but not to evolutionary merging with static synthetic tasks. This baseline would isolate the value of coevolution specifically.

5. **Inconsistent baseline definitions**: "Experts N=8" uses a 3-3-2 distribution that the authors admit is "manually tuned" and "arbitrary" (Appendix D.4). Different distributions yield different results, making this baseline unreliable. The proliferation of baseline variants (Control, Experts, Big Model, GPT-4o, Experts N=8) obscures the primary comparison.

7. **Statistical rigor gaps**: Tables 1-2 report improvements without confidence intervals, significance tests, or multiple comparison corrections despite testing across 8 benchmarks × 4-5 model families. Are differences statistically reliable? The ablation (Table 9) shows 0.46%-2.12% drops—are these significant given measurement noise?

8.  **Coverage-to-BoN translation failure**: Table 2 shows AC/DC often *underperforms* on Best-of-N single-answer selection despite Coverage gains (e.g., -0.19% vs. Big Model at N=3, -9.77% vs. GPT-4o). The authors note this requires "improved BoN methods" but don't explain why complementary capabilities evident in Coverage don't translate more reliably to practical deployment.

9. **Computational cost omitted**: No reporting of: (a) total GPU-hours for 50 generations, (b) cost of scientist LLM API calls, (c) comparison to fine-tuning alternatives. Is producing an evolved collective cheaper than training a single larger model? Table 4 shows 50 generations × 16 models × 250 tasks—what's the actual cost?

10. **Synthetic task quality unvalidated**: The synthetic tasks are generated by a 72B Qwen model and validated through "reflection" (self-solving), but no human evaluation confirms task quality, diversity, or realism. Do these tasks actually capture novel skills or just recombine existing problem templates?

11. **Generalization claims seem overstated**: Calling this "open-ended" evolution is premature after only 50 generations on a single synthetic distribution. True open-endedness (Stanley et al. 2017) requires demonstrating unbounded innovation, but Figure 6 shows Vendi score gains plateauing and task difficulty saturation at generation 45.

### Questions
1. Can the authors compare AC/DC to vanilla EvoMerge with a fixed synthetic task set? This would isolate whether task coevolution specifically (vs. just evolutionary merging on diverse tasks) drives improvements.

2. What properties predict whether a model family will succeed (Qwen) or fail catastrophically (Llama3)? Can the authors provide heuristics to assess compatibility before running expensive coevolution?

3. Why does optimizing for synthetic task Coverage transfer to unseen benchmarks? Is this due to: (a) the scientist LLM's implicit benchmark knowledge, (b) genuine skill generalization, or (c) synthetic tasks being representative of general problem-solving? Can the authors ablate using a scientist LLM with no benchmark exposure?

4. Can the authors provide confidence intervals for Tables 1-2 and significance tests for ablation improvements in Table 9? Are the reported gains reliable, given measurement noise and multiple comparisons?

5. Why do Coverage improvements often fail to translate to Best-of-N performance (Table 2: -0.19% at N=3 vs. Big Model)? Is this a fundamental limitation of the selection methods tried, or do complementary capabilities not compose well for single-answer tasks?

6. What are the total GPU-hours and API costs for AC/DC vs. fine-tuning a single 72B model on synthetic data? Is the evolved collective actually cheaper to produce than alternatives?

7. Why use a 3-3-2 distribution for "Experts N=8" (Appendix D.4)? Was this optimized, or arbitrary? How sensitive are results to alternative distributions (4-2-2, 2-4-2)?

8. Have humans validated the quality, diversity, and realism of generated synthetic tasks? Table 1 examples look reasonable, but systematic evaluation is absent.

9. Figure 6 shows Vendi score gains slowing and task difficulty saturation at gen 45. Does AC/DC exhibit unbounded behavior, or does it plateau after discovering capabilities within the seed models' latent space?

10. Table 3 shows AC/DC beats DNS/CQD (both trained on benchmarks), but DNS/CQD use different model families and training data. Can you compare all three methods on the *same* model family with identical conditions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces AC/DC, which performs both Model Archive Evolution and Task Archive Evolution to achieve effective collective intelligence. It enables models with smaller parameter scales to reach performance comparable to larger models. Experiments on benchmarks such as MMLU, GPQA, Minerva, and HumanEval demonstrate the effectiveness of the proposed method.

### Strengths
- The experiments in this paper are fairly extensive and include plenty of implementation details.
- I think the topic of evolving multiple base LLMs to create a more intelligent model is interesting and important.

### Weaknesses
- The Task Archive Evolution part still relies on a larger scientist LLM to generate tasks, which means it depends on external assistance and doesn't necessarily ensure the generated tasks are high quality.
- Although the method looks promising, the whole process, including Model Archive Evolution and Task Archive Evolution, involves too many implementation details. I think the paper lacks enough ablation studies to show the effectiveness of each design choice.
- Overall, this paper is really hard to follow. I strongly recommend the authors add clearer explanations of the full framework, including how LLM parameters are updated and how new tasks are generated.

I am generally inclined to accept this paper. However, since the proposed method involves many implementation details and lacks sufficient ablation studies, I'm not confident that all the technical components are truly useful, so I would lower my confidence level.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
3
