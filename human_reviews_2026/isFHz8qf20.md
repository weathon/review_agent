# An Information Theoretic Perspective on Agentic System Design

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Agentic language model (LM) systems power modern applications like "Deep Research" and "Claude Code," and leverage multi-LM architectures to overcome context limitations. 
Beneath their apparent diversity lies a recurring pattern: smaller "compressor" LMs (that can even run locally) distill raw context into compact text that is then consumed by larger "predictor" LMs. 
Despite their popularity, the design of compressor-predictor systems remains largely ad hoc, with little guidance on how compressor and predictor choices shape downstream performance.
In practice, attributing gains to compression versus prediction requires costly, task-specific pairwise sweeps.
We argue that these agentic system design questions are, at root, information-theoretic. Viewing the compressor LM as a noisy channel, we introduce a simple estimator of mutual information between the context and its compression to quantify compression quality in a task-independent way. 
We show that mutual information strongly predicts downstream performance, independent of any specific task. Through an information-theoretic framework, we perform a comprehensive empirical analysis across five datasets and three model families. Results reveal that larger compressors not only are more accurate, but also more token-efficient, conveying more bits of information per token. 
A 7B Qwen-2.5 compressor, for instance, is $1.6\times$ more accurate, $4.6\times$ more concise, and conveys $5.4\times$ more bits of mutual information per token than its 1.5B sibling. Across datasets, scaling compressors is substantially more effective than scaling predictors, enabling larger on-device compressors to pair with smaller cloud predictors.
Applied to a Deep Research system, these principles enable local compressors as small as 3B parameters to recover 99% of frontier-LM accuracy at 26% of API costs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel, information-theoretic perspective on designing agentic language model systems, particularly those following a "compressor-predictor" architecture. The authors argue that the current design process for such systems is ad-hoc. To address this, they propose using Mutual Information (MI) between the source context and the compressed summary as a task-agnostic metric to evaluate the quality of the "compressor" LM. Through extensive experiments across four datasets, the paper derives several key design principles. The most significant finding is that scaling the compressor model is substantially more effective for improving downstream performance than scaling the predictor model. The authors also show that larger compressors are more token-efficient and that the proposed MI-based metric strongly correlates with final task performance, offering a valuable proxy for system evaluation.

### Strengths
* **Novel and Principled Framework**: The primary strength of this paper is its novel application of information theory to a very practical and important problem. Moving the design of agentic systems from ad-hoc trial-and-error to a more principled framework grounded in metrics like Mutual Information is a significant conceptual contribution.
* **Actionable and Impactful Findings**: The paper produces clear, actionable, and somewhat counter-intuitive design principles. The core finding—that it is more effective to "front-load" compute by investing in a more powerful compressor rather than a larger predictor—has significant practical implications for system design, especially regarding the trade-off between local (on-device) compute and expensive cloud API calls.
* **Comprehensive Empirical Analysis**: The authors conduct an extensive set of experiments across multiple model families, model sizes, and datasets to validate their claims. The scaling law analyses are thorough and provide strong evidence for their conclusions regarding the relative importance of the compressor versus the predictor.

### Weaknesses
* **Methodological Concerns with the MI Estimator**: The validity of the paper's core metric, Mutual Information, is potentially compromised by the methodology used for its estimation. The authors state in Section 3.2 that due to the miscalibration of smaller models, they used a larger 7B "proxy model" to evaluate the log probabilities for all compressors. This introduces a significant confounding variable: it is unclear whether the MI metric is measuring the true information retained by the compressor or simply the compressor's ability to generate text that is probable under a different, larger model. This methodological choice could affect the integrity of all subsequent conclusions that rely on the MI metric.
* **Oversimplification of Agentic Workflows**: The study primarily focuses on a simple, one-shot "compress-then-predict" workflow. While this is a common pattern, it doesn't capture the complexity of more advanced, iterative agentic systems where the predictor might re-query the compressor or where multiple agents collaborate over several turns. The conclusions drawn, particularly about the primacy of the compressor, may not generalize to these more dynamic and complex architectures.
* **Limited Scope of "Compression"**: The experiments define "compression" primarily as summarization. However, in agentic systems, compression can take many forms, such as structured data extraction (e.g., to JSON), function call generation, or argument synthesis. The paper's key finding that larger compressors become more "concise" (fewer tokens) might not hold for these other compression modalities, where a better compression could be more detailed and thus longer.
* **Clarity of Figures and Captions**: The presentation of several key figures could be improved. Specifically, some figure captions are not fully self-contained or are difficult to read, forcing the reader to hunt for details in the main text to understand the axes and curves. For example, the captions for Figure 2 and Figure 3 are dense and require careful cross-referencing with the text, which hinders clarity and makes the results harder to interpret at a glance.

### Questions
See the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Agentic language model systems have become prevalent in AI workflows, often featuring smaller compressor models that distill raw contexts into summaries for larger predictor models to generate answers. The motivation stems from the ad-hoc nature of designing these systems, lacking principled ways to evaluate compressor quality independently or attribute performance gains. Challenges include the difficulty in quantifying compression efficacy without task-specific metrics and the computational intractability of exact mutual information calculations. The paper proposes an information-theoretic framework, introducing a Monte Carlo estimator for mutual information between context and compression, along with rate-distortion analysis to predict downstream performance, validated through empirical studies on four datasets showing that scaling compressors yields greater efficiency and accuracy than scaling predictors.

### Strengths
1. Empirical results demonstrate clear scaling laws favoring larger compressors. Larger models produce more concise yet informative summaries, leading to sublinear compute cost increases. These findings offer practical guidance for optimizing agentic systems in resource-constrained environments.

2. The rate-distortion analysis reveals strong correlations between information rate and accuracy. Bit efficiency metrics predict performance with high fidelity, as shown by R-squared values up to 0.71. This tool allows for quick assessment of compressor-predictor pairings.

3. Experiments across multiple model families highlight compressor choice as the dominant factor. Qwen-2.5 models show superior efficiency compared to Llama and Gemma-3. Such comparisons inform model selection for specific deployment scenarios.

### Weaknesses
1. The mutual information estimator relies on proxy models for smaller LMs. This introduces potential biases in log-probability evaluations. Such approximations may not fully capture the true information content.

2. Analysis is limited to non-reasoning GPT-style models. This restricts generalizability to reasoning-augmented or multi-turn agentic systems. Future work is needed to extend to more complex architectures.

3. The framework assumes single-round communication. This overlooks iterative multi-agent workflows common in practice. Extending to multi-round scenarios could reveal different scaling behaviors.

4. The claimed communication efficiency by text compressor actually is little, because the texts occupy much smaller memory than models or features. Considering the extra computational costs, the communication time saved by the compressor is not comparable to the extra computation time. 

5. Computational estimates use simplified FLOPs calculations. These may not account for real-world hardware variations or quantization effects. More precise benchmarking on actual devices is necessary.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper explores agentic language model systems where smaller compressor models distill raw contexts into compact summaries for larger predictor models to generate user responses. Challenges include the difficulty in measuring compression efficacy without full system evaluation and understanding scaling impacts on compute efficiency. Solutions involve introducing a mutual information estimator for task-agnostic compression assessment, conducting rate-distortion analysis to link information retention to downstream performance, and empirical studies across datasets showing benefits of scaling compressors over predictors.

### Strengths
1. The mutual information estimator offers a practical tool for evaluating compression without requiring downstream tasks. It computes effectively using modern inference servers. This method provides insights comparable to perplexity for predictors.
 
2. Rate-distortion analysis establishes strong correlations between information rate and task performance. It serves as a reliable proxy for system efficacy. The framework guides optimization of communication in agentic designs.
 
3. Empirical findings reveal a clear hierarchy where compressor family outweighs predictor size in importance. Scaling compressors yields greater gains than predictors. This principle supports front-loading compute into local devices.
 
4. Application to Deep Research pipelines achieves near-frontier accuracy with small local compressors. It reduces API costs significantly while maintaining quality. The approach demonstrates real-world utility in multi-agent workflows.

### Weaknesses
1. Reliance on proxy models for mutual information estimation in smaller LMs introduces potential biases. The approximation may not fully capture information dynamics. This could compromise the accuracy of task-agnostic evaluations.
 
2. Restriction to non-reasoning GPT-style models limits the study's scope. Reasoning tokens require separate analysis not covered here. The postponement leaves gaps in applying findings to advanced agentic architectures.
 
3. Use of subsampled or synthetic datasets may not reflect real-world variability. For instance, WildChat employs grouped synthetic users. This setup could skew scaling observations toward artificial patterns.
 
4. Assumption of Gaussian form in rate-distortion fitting oversimplifies LM data distributions. Distortion measured as 1-accuracy ignores nuances in evaluation. More flexible models might yield deeper insights into trade-offs.
 
5. Cost estimates depend on specific API rates and hardware assumptions that evolve rapidly. They overlook quantization impacts on deployment. This diminishes the practical longevity of economic recommendations.

### Questions
No other questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper examines “compressor–predictor” architectures for agentic LLM systems (e.g., Deep Research pipelines), where a smaller LM compresses the raw context and a larger LM performs reasoning/generation.
The authors bring an information-theoretic lens, framing the compressor as a noisy channel and using mutual information (MI) as a task-agnostic measure of compression quality.

### Strengths
Applying MI and rate–distortion theory to compressor–predictor pipelines is interesting and provides a principled view on design trade-offs.

The MI estimator is implementable on real inference stacks; could directly help practitioners evaluate compressors without full end-to-end sweeps.

The four distilled principles are easy to interpret and potentially impactful in industry.

### Weaknesses
- The information-theoretic analysis stops at empirical correlation and heuristic exponential fitting. No formal connection is established between MI and downstream accuracy beyond observed correlation
- No ablation quantifying the error or variance of the MI estimator.
- Experiments focus on one-shot, GPT-style instruction models; no evidence for robustness to multi-turn reasoning agents, tool-using agents, or MoE architectures.
- Some datasets or QA pairs are synthetic (GPT-generated), which may bias outcomes toward LMs similar to the generator.
- Downstream metrics limited to accuracy/perplexity; no human evaluation or diverse NLG metrics (e.g., ROUGE, BLEU, factuality) for generative tasks.
- Failure mode analysis is qualitative only; lacks quantitative measurement of how much each error type contributes to performance loss.

### Questions
How sensitive is the MI–accuracy correlation to the choice of proxy model for log-prob estimation?
Can the MI estimator be extended to handle reasoning chains or intermediate tool calls where information is not localized in a contiguous context?
Did you attempt any adjustments to the distortion metric beyond accuracy (e.g., weighted accuracy for partial credit or semantic similarity for generative tasks)?
For Deep Research, what happens if the predictor is also small and local — does the compressor's advantage still hold?

### Soundness
3

### Presentation
3

### Contribution
3
