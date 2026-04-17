# Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Generative modeling is widely regarded as one of the most essential problems in today's AI community, with text-to-image generation having gained unprecedented real-world impacts. Among various approaches, diffusion models have achieved remarkable success and have become the de facto solution for text-to-image generation. However, despite their impressive performance, these models exhibit fundamental limitations in adhering to numerical constraints in user instructions, frequently generating images with an incorrect number of objects. While several prior works have mentioned this issue, a comprehensive and rigorous evaluation of this limitation remains lacking. To address this gap, we introduce T2ICountBench, a novel benchmark designed to rigorously evaluate the counting ability of state-of-the-art text-to-image diffusion models. Our benchmark encompasses a diverse set of generative models, including both open-source and private systems. It explicitly isolates counting performance from other capabilities, provides structured difficulty levels, and incorporates human evaluations to ensure high reliability.
Extensive evaluations with T2ICountBench reveal that all state-of-the-art diffusion models fail to generate the correct number of objects, with accuracy dropping significantly as the number of objects increases. Additionally, an exploratory study on prompt refinement demonstrates that such simple interventions generally do not improve counting accuracy. Our findings highlight the inherent challenges in numerical understanding within diffusion models and point to promising directions for future improvements.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analyzes the counting problem in text-to-image diffusion models. It measures accuracy on prompts that require generating 1–15 objects, under 3 different scenes and styles. It also probes whether several prompt-refinement strategies help. The takeaway is that models struggle as the count increases, and simple prompting strategies do not fix it.

### Strengths
There is generally a lack of detailed analysis of the counting problem, which makes the motivation clear. In my opinion, several aspects of the paper stand out positively:

1. Analyzing results separately by object category, scene, and style is useful, as it reveals how much each aspect affects counting accuracy.

2. The paper includes plenty of qualitative examples that make the claims easy to assess.

3. It evaluates many recent diffusion models, so the conclusions are relevant to the current state of the field.

4. The paper is organized well. It presents clear observations for each component, which invite further investigation into why they occur.

5. The position guidance and the grid prior are intuitive refinements, and the finding that they do not help is informative for practical purposes.

### Weaknesses
**Major:**

1. The benchmark relies on human evaluation by five graduate students. This makes replication and future model evaluation non-trivial. Since no automatic evaluation metric is provided, how could a new model be assessed?

2. The analysis does not necessarily depend on diffusion models. Similar counting failures have been observed in some autoregressive text-to-image models (e.g., Janus). Limiting the study to diffusion models restricts the scope of the paper.

3. It is unclear what specific objects are used within each category (e.g., "triangle" seems to represent all shapes, while "watermelon" and "apple" represent the fruit category). The "plant" category appears to shift from "flower" to "tree" in certain experiments. Clarification regarding these details is needed. Moreover, the small number (only one or two) of objects per category prevents making strong generalizations.

4. More detail is needed on how the prompts were constructed. How were scenes and styles chosen? The qualitative examples show varied options within each category, but detailed explanation of the selection process is missing.

5. There is a correlation between chosen objects and scenes (for instance, some objects are more common in Home than City). This may partially explain why City underperforms Home. Stratifying results by "typicality" would help confirm whether the observed performance gap between scenes is due to scene complexity (as claimed in Observation C.1) or dataset bias.
	
6. What guidelines were used to ensure scoring consistency among raters? How were partial objects or ambiguous cases treated? Reporting the variance across the five raters would be valuable.

**Minor:** 

1. In the multiplicative decomposition formula, the text states that b is the smallest factor greater than N / 2, which cannot generally hold (it would imply factors N and 1). I believe the authors meant square root of N (which matches the examples). Also, what happens when N is prime?

2. The claim in L.364 about a "degradation in fidelity" is qualitative. Could this be verified quantitatively, for example, by reporting an aesthetic or any image quality score?

3. In the prompt-refinement section (L.425), if the goal is to remove the "impact of extraneous factors," why was the scene fixed to Home? This still introduces correlations. It would be clearer to either evaluate all combinations, as before, or to remove the scene and style conditions entirely.
	
4. The intuition behind prompt templates 2 and 3 is unclear. Unless such phrasings commonly appear in the training data, they may only make the task harder by requiring the model to reason about arithmetic.
	
5. Some writing issues:
The main text (L.475) says the related work is deferred to the appendix, but it actually appears in the main paper.
Figure 6 duplicates Figure 11.
In Figure 6, the rightmost column’s prompt appears mismatched to the corresponding images.

### Questions
1. Using 15 numbers, 6 categories, 3 scenes, and 3 styles should yield 810 prompts, not 525. Could you clarify how you arrived at 525?

2. In L.224, why do you report success if any image in a multi-image response is correct, instead of averaging accuracy across all generated images?

3. For the grid prompts, is correctness judged solely by the count and not by adherence to the specified layout?

It would be appreciated if the authors could consider the points mentioned in the weaknesses section (mainly those in the major part) and share any feedback they might have.

### Soundness
2

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
3

### Summary
This paper proposes a new benchmark that evaluates the ability of text-to-image models to generate a specific number of objects. The proposed benchmark includes multiple compositions of number, object, scene, and style, and evaluates state-of-the-art models. The benchmark adopts a fully human evaluation process and uses accuracy as the evaluation metric. Experimental results show that recent models perform worse on the proposed benchmark, and simple prompt refinements fail to improve the performance.

### Strengths
This paper investigates the counting ability of generative models, which is an important setting. Experimental results show that state-of-the-art models achieve poor results on the proposed benchmark.

### Weaknesses
1. This paper adopts full human evaluation to get the results, which is difficult to generalize.
2. This benchmark evaluates generating images with a composition of number, object, scene, and style. The poor accuracy may come from the inability of diffusion models in generating the compositions, instead of the counting ability itself.
3. This paper takes the accuracy as the evaluation metric.  The quality of generated images is also important for evaluation, but is missing in the proposed benchmark.
4. According to lines 223-224 quoted as follows, 
> When a text-to-image model returns multiple images in a single response, the task is considered successful if at least one image is correct

This is not fair, as some models may successfully generate each image, while others may only succeed occasionally, yet they are treated the same.

5. There is a widely used benchmark that is for evaluating the counting ability of diffusion models[1], which is not discussed. Although this benchmark is published two years ago, it is easy to generalize to recent models and is still valuable.

[1] GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. NeurIPS 2023.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows a critical yet under-explored limitation of state-of-the-art text-to-image diffusion models: their inability to adhere to numerical constraints (i.e., accurate object counting). To rigorously evaluate this issue, the authors propose T2ICountBench, a specialized benchmark that isolates counting performance from other capabilities (e.g., style alignment, shape adherence) and includes 15 recent SOTA models (both open-source and private, mostly post-2024), 6 object categories, 3 scenes, 3 styles, and object counts ranging from 1 to 15. Extensive human evaluations reveal that all models fail to generate the correct number of objects consistently—accuracy drops sharply with increasing object counts (to ~10% for 11–15 objects) and degrades further in complex scenes (e.g., cityscapes). Additionally, an exploratory study of four prompt refinement strategies (multiplicative/additive decomposition, grid prior, position guidance) shows that simple interventions do not improve counting performance, highlighting inherent limitations in numerical understanding.

### Strengths
1. The paper fills a gap in existing benchmarks, which either ignore counting or conflate it with other capabilities. T2ICountBench is the first comprehensive, specialized benchmark for evaluating counting in text-to-image diffusion models, with structured difficulty levels and broad coverage of recent models.
2. The paper uncovers a fundamental, practically relevant limitation of diffusion models, critical for use cases requiring precise control (e.g., design, education, data visualization). And It demonstrates that prompt refinement, a common mitigation for model flaws, is ineffective.

### Weaknesses
1. Limited Analysis of Failure Mechanisms: The paper attributes counting failures to CLIP’s inherent limitations and insufficient human preference alignment but provides minimal empirical evidence to validate these hypotheses. For example, it does not compare models with different text encoders (e.g., CLIP vs. T5) to quantify how encoder choice impacts counting performance. A targeted ablation on text encoders would strengthen the causal analysis.
2. Narrow Scope of Prompt Refinement: The four prompt strategies tested are simple task-decomposition methods, but more advanced prompt engineering (e.g., chain-of-thought prompting, few-shot examples, explicit numerical emphasis) or hybrid approaches (e.g., combining prompt refinement with model fine-tuning) are not explored. The conclusion that “prompt refinement cannot help” may be overgeneralized without testing these alternatives.

### Questions
1. Advanced Prompt Refinement: Have you tested more advanced prompt strategies (e.g., chain-of-thought, few-shot examples with counting demonstrations, or explicit numerical constraints like “exactly N objects, no more no less”) or hybrid approaches (e.g., prompt refinement + lightweight fine-tuning)? If these methods also fail, could you explain why they are ineffective? If you have not tested them, could you elaborate on why they are outside the scope of your exploratory study?
2. Comparison to Specialized Methods: How do your findings align with prior work on enhancing counting in generative models (e.g., Paiss et al., 2023; Jiang et al., 2023)? Can T2ICountBench be used to evaluate these specialized methods, and if so, have you tested whether they outperform vanilla diffusion models on your benchmark?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces T2ICountBench, a new, comprehensive benchmark designed to rigorously evaluate the counting ability of text-to-image diffusion models.
 The authors conduct an extensive empirical study on 15 state-of-the-art generative models, including both open-source and proprietary systems from 2024-2025. 
The core findings demonstrate that all evaluated models struggle to generate the correct number of objects specified in a prompt, with accuracy dropping precipitously as the target number increases. 
Furthermore, an exploratory study reveals that simple, human-inspired prompt refinement strategies, such as additive or multiplicative decomposition, not only fail to improve but often degrade counting performance. 
The work systematically confirms a widely-observed limitation and provides a valuable resource for future research.

### Strengths
- Comprehensive and Timely Benchmark: 
The paper introduces T2ICountBench, a focused and much-needed benchmark for a critical capability. 
The evaluation is highly relevant, covering 15 very recent models, which provides an up-to-date snapshot of the state of the art.
- Rigorous Human Evaluation: 
The study's reliance on a full human evaluation protocol, involving five annotators per image, is a significant strength. 
This approach serves as the gold standard for this task, ensuring high-quality data and avoiding the potential biases and inaccuracies of automated evaluation metrics.
- Significant and Clear Findings: 
The paper provides unambiguous, large-scale evidence that modern text-to-image models cannot count reliably. 
The core findings—that accuracy plummets for counts above 5-10, and that simple prompt engineering fails—are important, well-supported, and impactful for the community.
- Excellent Presentation and Reproducibility: 
The manuscript is exceptionally well-written, structured, and easy to follow. 
The inclusion of extensive appendices with detailed per-model results, implementation specifics, and a vast number of qualitative examples greatly enhances the transparency and value of the work.

### Weaknesses
- Lack of Statistical Rigor: 
The paper's claims are not supported by any statistical significance testing.
Key metrics like inter-rater agreement (e.g., Fleiss' Kappa) are missing, making it impossible to assess the reliability of the human annotations. 
Furthermore, no confidence intervals or variance estimates are provided for the accuracy results, which is a significant omission for an empirical paper making comparative claims.
- Potentially Biased Evaluation Protocol: 
The success criterion is defined as at least one correct image in a set of generated images. 
This metric may unfairly favor models that produce more images per prompt and can inflate the reported accuracy, obscuring the true per-image performance. 
A more robust evaluation would report per-image accuracy and analyze the results for top-1 vs. any-of-k success.
- Narrow Scope of Prompt Refinement: 
The paper makes the strong claim that 'prompt refinement cannot help' based on four simple decomposition strategies. 
This claim is overstated, as the study does not compare against more sophisticated, published counting-specific techniques (e.g., CountGen, Counting-Guidance) that have shown promise, making the conclusion less general than implied.
- Insufficient Analysis of Model Differences: 
While the paper effectively documents performance disparities between models (e.g., Imagen-3's 43% accuracy vs. Recraft V3's 25%), it offers minimal analysis into the potential architectural, training data, or text encoder differences that could explain this variance. 
This is a missed opportunity to provide deeper insights for future model development.

---

- General Limitations:
The authors adequately addressed the primary limitations and potential risks of their work in Appendix E. 
However, the discussion on practical implications for real-world applications is somewhat limited, and the paper could benefit from offering more concrete recommendations or potential workarounds for practitioners facing this counting limitation.

### Questions
- To address the lack of statistical rigor, could you please report the inter-rater agreement for your human evaluation? 
Furthermore, could you provide 95% confidence intervals for the main accuracy results in Table 2 and Figure 1 to properly substantiate the claims of performance differences?
- Regarding the evaluation protocol, can you provide a breakdown of the results using per-image accuracy? 
How does this change the relative ranking of models, especially when comparing systems that may return a different number of candidate images per prompt?",
- Given that your claim 'prompt refinement cannot help' is very strong, how do you position your findings relative to existing literature on counting-specific methods like CountGen or iterative refinement techniques? 
A discussion contextualizing your results against these more advanced approaches would strengthen the paper.
- Do you have any hypotheses regarding the significant performance gap between the top-performing models (Imagen-3, Gemini 2.0 Flash) and the lower-performing ones? 
Could this be attributed to the text encoder (e.g., T5 vs. CLIP), the scale of the training data, or specific architectural choices?

### Soundness
2

### Presentation
4

### Contribution
4
