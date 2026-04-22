# Probing Memes in LLMs: A Paradigm for the Entangled Evaluation World

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Current evaluations of large language models (LLMs) often treat datasets and models in isolation, obscuring phenomena that only emerge from their collective interaction. Items (Questions) in datasets are reduced to labeled entries, disregarding the multidimensional properties they reveal when examined across model populations. Models, in turn, are summarized by overall scores such as accuracy, neglecting performance patterns that can only be captured through diverse data item interactions. To address this gap, this paper conceptualizes LLMs as composed of invisible memes, understood as cultural genes in the sense of Dawkins that function as replicating units of knowledge and behavior. Building on this perspective, the Probing Memes paradigm reconceptualizes evaluation as an entangled world of models and data. At its core lies the perception matrix, which captures interaction patterns and enables two complementary abstractions: probe properties, extending dataset characterization beyond labels, and phemotypes, revealing fine-grained capability structures of models. Applied to 9 datasets and 4,507 LLMs, Probing Memes reveals hidden capability structures and reveals phenomena invisible under traditional paradigms (e.g., elite models failing on problems that most models answer easily). This paradigm not only supports more informative, extensible, and fair benchmarks but also lays the foundation for population-based evaluation of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Probing Memes, an evaluation paradigm that jointly considers individual samples in the dataset and models evaluated on them, in contrast to the previous convention that only looks at a single model's performance aggregated over samples. The authors design evaluation metrics to model each sample as "probes" to study the capability properties of the dataset vs. the model. Experimental results reveal distinct dataset-model properties across several benchmarks, such as a high probing value of "surprise" on MMLU-pro.

### Strengths
1. Thorough joint analysis along the dimensions of sample and model is an interesting yet underexplored direction.  
2. The proposed Probing Memes paradigm is well-motivated. The overall idea is clear.
3. Experiments are extensive and well-deliver the proposed idea.

### Weaknesses
1. The novelty of this paper lies in the proposed two-dimensional (sample-model) evaluation. However, the novelty may be undermined by the fact that previous work has been using such analysis for specific purposes. For example, [a] defines and calculates the difficulty of samples in benchmark datasets, along with 52 LLMs of different sizes, to investigate emergent abilities. This raises concern about the insufficient discussion of and distinction from related work.

2. This proposed paradigm relies on the assumption that there have been sufficiently many LLMs evaluated on the target dataset. If the number of LLMs is "not enough," the values of probes will become unreliable. On the other hand, with more LLMs evaluated on the target dataset, the values of probes and phenotypes will change since they depend on the tested LLMs.

3. I am concerned about this paper's framing. Terms like "meme", "probe," and "phenotype" might be a bit distracting and imprecise. 

4. Discussions on experiments and findings seem not deep. For example, it is no surprise that there are samples with high surprise value in MMLU-pro. Can your paradigm help answer why this happens? Arguments for the practical value of this paradigm are insufficient and unclear. Another example: Given you find that "probes in IFEval, GPQA-Diamond, and BBH exhibit relatively high uniqueness" (line 431), what can we do next to make it not only a "good to know" thing? What is this finding's practical value? 

a. [U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models](https://openreview.net/forum?id=jjfve2gIXe)

### Questions
1. Can you make a comparison of the definitions of memes in this paper and in The Selfish Gene? It's unclear to me how this term can be borrowed from the book.

2. My current understanding regarding 1. is that a meme is a facet of LLM's ability (by line 79, " From this perspective, the abilities of LLMs are conceptualized as composed of latent memes"). In this case, are the proposed 7 kinds of probes only a small subset of an unknowingly big set of all memes?

3. In Section 2.2, you define **uniqueness** through the averaged conditional entropy of two probes and define **ϕ-coefficient** to calculate the similarity of two probes. It seems to me that entropy (e.g., JS entropy) is also a natural metric to determine whether two probes should have an "edge". Is my understanding correct?

4. In Figure 6, can you explain why Astuteness has nearly identical weight distribution over the three datasets?

5. In line 429, you mention MMLU-pro has many samples with high surprise. Is this possibly due to the nature of MCQs? Assume a challenging question where a frontier LLM gets wrong. If a low-ability LLM simply randomly guesses all samples and thus picks the correct choice by chance, the surprise value of this question will be high.

6. The statement in lines 461-462 is a bit abrupt and without any supporting argument. You can move this statement to the earlier section and support it with concrete examples—text only is ok; quantitative experiments are a huge improvement.

7. In line 471, why is full reproducibility not guaranteed even with temperature=0? Isn't the probe calculation process deterministic?

My current score for this paper is 2. Given that I have many questions, I may raise my score to 4 or 6 if this paper's significance and novelty becomes clear and good to me during the discussion stage.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper solved the prob of evaluating LLMs only with coarse, accuracy-centric scores that ignore how models and datasets interact at scale. This paper proposed the Probing Memes paradigm: build a perception matrix over many models × many items, derive Meme Probe Properties, and aggregate them into model-level phemotypes to reveal fine-grained capability structure across populations.

### Strengths
1/ In this work, the authors reconceptualize evaluation as an entangled world of models and data, formalizing a perception matrix that supports probe-level properties and interpretable phemotypes; this exposes phenomena hidden by traditional benchmarks (e.g., elite models failing items most models solve) and scales to thousands of models.

2/ The authors validate the framework on a huge amount of LLMs, showing clear probe/property distributions, family-level structure in phemotype space, and practical insights (accuracy-equal models with different behavioral profiles), demonstrating both scalability and interpretability.

### Weaknesses
1/ I suggest maybe the authors can consider broadening tasks (coding, RAG, agents) and adding head-to-head baselines (e.g., IRT-based compact sets, adversarial stress tests) to verify that phemotypes add incremental value beyond existing item- and ability-modeling approaches.

2/ An ablation on property definitions, thresholds, and clustering (e.g., Leiden parameters) would clarify robustness and generality.

3/ For the evaluation results, the authors can add multi-judge adjudication, uncertainty estimates, or multi-runs to enhance the robustness of the results.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new evaluation framwork to characterise data samples from test sets and large language models, focussing strongly on model-data interactions under the umbrella of memetics. The authors test 4507 LLMS with many dataset characterisations and model capability probes on these datasets.

### Strengths
1. Capability probing is an important aspect of foundation model evaluation. It helps us make evaluations more granular and extract more information from datapoints. This work makes a positive contribution in that direction.

2. The different probes and phemotypes are well-defined, in theory.

3. Testing 4507 models from OpenLLM Leaderboard is a substantial empirical contribution.

### Weaknesses
1. The meme framework seems unnecessary and, without a more grounded theoretical framework and justification in the context of LLM evaluations, could be removed. It leads to unnecessary terminology such as perception matrix, meme probes and phemotypes. The core contributions would not be affected if the metaphor were removed. It also leads to some sections becoming quite confusing to read("latent units of model capability that can be revealed through probing"). 

2. Several relevant papers have not been cited or referenced in this work. Flexibly defining new properties based on evaluation needs and the observation that datasets “contain a large number of seemingly simple questions that are nevertheless answered incorrectly by some elite models” are both insights established in [1]. Then, “models with similar accuracy may succeed on very different types of items” is adopted from [2]. Several of the meme probe properties are established frameworks already: difficulty has been defined and used in [3] as as the IRT model in [4](cited elsewhere in the paper), the  . Other forms of capabilities are tested by works such as [5].

3. This work only considers binary (0,1) evaluations. Other works [1,4] take into consideration heterogeneous metrics such as accuracy (0/1), BLEU score ([0,1]) which is a more realistic setup, ensuring that the framework of evaluation takes into consideration all datasets.

4. There is no insight into whether the phemotypes are correlated to each other and to what extent. This paper would benefit from quantitative studies to determine this.

Minor
1. Figure 5 and 6 are not easily interpretable, more details and insights would be beneficial.
2. T-sne is known to result in spurious clusters. UMAP, on the other hand, preserves both local and global structure in the data and is a better algorithm for data visualisation.


[1] Ghosh et al. ONEBENCH to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities, ACL 2025

[2] Goel et al. Great Models Think Alike and this Undermines AI Oversight, ICML 2025 

[3] Prabhu et al. Efficient Lifelong Model Evaluation in an Era of Rapid Progress, NeurIPS 2024

[4] Polo et al. tinyBenchmarks: evaluating LLMs with fewer examples, ICML 2024

[5] Alyahya et al. ZEROSUMEVAL: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition, ACL 2025

### Questions
1. “Certain elite models that excel in overall metrics nevertheless display anomalous errors on questions that most other models solve with ease”. Could this be the case due to train-test contamination, the presence of test samples during pretraining? What would be a way to test this?
2. Is there a unique insight into evaluation provided in the memetic definition (not talking about the probe properties here)? The interaction of data and model is basically how evaluation is done. Other works such as [1] use more applicable definitions from other fields, such as social choice theory and also conduct experiments to compare their framework with other methods in that field.
3. To measure I_j, which is it logarithmic? It seems invariant to “reduces the influence of weak models while emphasizing the contribution of stronger models”.

[1] Ghosh et al. ONEBENCH to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities, ACL 2025

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
This paper introduces the "Probing Memes" paradigm for LLM evaluation, drawing conceptually on Dawkins' theory of memes as replicating cultural units. The authors propose treating LLM capabilities as composed of latent "memes" that can be revealed through carefully designed probes. They construct a “perception matrix” from model-data interactions.

### Strengths
- The problem of better analyzing model evaluations is important 
- Some of the proposed metrics are interesting

### Weaknesses
1. **Unclear motivation for the meme framing**: The conceptual link to Dawkins and memetics feels forced and adds unnecessary complexity without clear benefit. The core contributions (probe properties and phemotypes) could stand without this metaphor. The paper states that memes are "latent units of model capability that can be revealed through probing", but this is more of a renaming than a substantive theoretical contribution. Why is this memetics lens necessary or illuminating?

2. **Limited differentiation in results**: The key weakness is visible in Figure 7, which shows phemotype scores tracking remarkably closely with accuracy across models. While the paper claims to reveal "fine-grained phenomena invisible under conventional evaluations," the phemotypes appear highly correlated with overall performance. This contradicts the motivation that current approaches "obscure fine-grained differences" - if the proposed phemotypes largely parallel accuracy, what additional insight do they provide?

3. **Inconsistent with cited literature**: The paper cites Schilling-Wilhelmi et al. on IRT where different analysis methods reveal significant ranking changes. However, Figure 7 shows surprisingly consistent orderings across metrics, undermining this motivation.

4. **Unclear practical utility**: What should practitioners do differently with phemotypes versus accuracy? The paper doesn't provide clear guidance on how these metrics inform model selection, dataset design, or capability assessment in practice.

### Questions
1. Can you provide examples where phemotypes lead to different model rankings or selection decisions compared to accuracy?

2. In Figure 7, why do phemotypes correlate so strongly with accuracy if they're capturing distinct capability dimensions? What percentage of variance in phemotypes is explained by accuracy?

3. What is the empirical evidence that the meme metaphor provides insight beyond standard psychometric approaches (IRT, factor analysis, etc.)?

4. How should researchers choose which phemotype to optimize for? Are some phemotypes more important for certain applications?

### Soundness
2

### Presentation
1

### Contribution
2
