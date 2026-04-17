# Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Semantic representations can be framed as a structured, dynamic knowledge space through which humans navigate to retrieve and manipulate meaning. To investigate how humans traverse this geometry, we introduce a framework that represents concept production as navigation through embedding space. Using different transformer text embedding models, we construct participant-specific semantic trajectories based on cumulative embeddings and extract geometric and dynamical metrics, including distance to next, distance to centroid, entropy, velocity, and acceleration. These measures capture both scalar and directional aspects of semantic navigation, providing a computationally grounded view of semantic representation search as movement in a geometric space. We evaluate the framework on four datasets across different languages, spanning different property generation tasks: Neurodegenerative, Swear verbal fluency,  Property listing task in Italian, and in German. Across these contexts, our approach distinguishes between clinical groups and concept types, offering a mathematical framework that requires minimal human intervention compared to typical labor-intensive linguistic pre-processing methods. Comparison with a non-cumulative approach reveals that cumulative embeddings work best for longer trajectories, whereas shorter ones may provide too little context, favoring the non-cumulative alternative. Critically, different embedding models yielded similar results, highlighting similarities between different learned representations despite different training pipelines. By framing semantic navigation as a structured trajectory through embedding space, bridging cognitive modeling with learned representation, thereby establishing a pipeline for quantifying semantic representation dynamics with applications in clinical research, cross-linguistic analysis, and the assessment of artificial cognition. https://github.com/jesuinovieira/semtraj-iclr2026

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper models human concept production as trajectories in embedding space, extracting many geometric/dynamical markers from participant-specific paths built with cumulative text embeddings. The framework differentiates groups and concept types and appears robust across encoders for local trajectory metrics, while centroid-based dispersion shows model-dependent geometry. The authors discuss clinical and cross-lingual applications and note limitations.

### Strengths
Originality. Recasts verbal fluency/property listing as geometry + dynamics in learned representation spaces, bridging cognitive foraging accounts with modern NLP embeddings. The cumulative-embedding design captures history dependence rather than treating items independently.

Quality. Clear metric definitions (including a binarized entropy proxy) and mixed-effects modeling (GLMM via glmmTMB) are appropriate for repeated-measures data; cross-encoder replication is a strong robustness check.

Clarity. The paper’s pipeline is easy to follow; per-dataset result summaries and heatmaps for cross-model correlations aid interpretation.

Significance. Demonstrates clinically and linguistically informative signals with minimal manual annotation, and highlights encoder-agnostic local dynamics vs model-dependent global geometry, aligning with known cross-lingual structure and anisotropy issues in embeddings.

### Weaknesses
1. The analyses only adopt Euclidean differencing despite acknowledged anisotropy in contextual embeddings; consider non-Euclidean or locally whitened metrics (e.g., hyperbolic/Riemannian, subspace-projected velocities) to test robustness will be helpful.
2. Velocity/acceleration use implicit $\Delta t=1$ because timestamps are missing. Include more results to show where real inter-response times modulate the dynamics will be helpful.
3. Provide confidence intervals, seed/split variability, and multiple-comparison controls for pairwise tests in the figures; this will improve clinical interpretability.
4. Writing/format nits. A few typos (“Neurodegerative”), minor grammatical errors like “This approach hold…”; ensure all acronyms expand on first use.

### Questions
1. Do your main findings persist under whitened cosine, Riemannian distances, or other nonlinear metrics? A small ablation would help separate method from metric.
2. If inter-response times become available, do velocity/acceleration still distinguish groups once scaled by real time?
3. How do your metrics correlate with clustering/switching scores on the same sessions, and do they add incremental predictive value?
4. Given stable local dynamics across encoders, can you leverage multilingual alignment (e.g., shared subspaces) to standardize centroid-based dispersion across languages/models?

### Soundness
2

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
4

### Summary
*Concept production* refers to the task of a person listing as many words as they can within a given category in a short time period. This paper proposes a method to quantify the semantic dynamics of concept production by measuring dynamic properties of Language Model embeddings of the produced words.

They authors propose 5 summary statistics of the embedding trajectories: distance-to-next, entropy, velocity, acceleration and distance-to-centroid. They employ four datasets of to evaluate:
 - a neurodegenerative dataset containing participants with Parkinson's disease, frontotemporal dementia and healthy controls,
- a swear fluency dataset, with three categories control categories and I swear word category,
 - Italian and German datasets, where particants were asked to produce words from a variety to categories in either Italian or German

Qualitative results in the form of plots, and quantitative statistical test are performed to assess the variability between categories of the embedding dynamics within datasets.

The authors perform their experiment using three different embedding models and find similar results.

### Strengths
The paper is clearly written and the methodology is novel, providing a means to quantify properties of the semantic dynamics of words produced during these experiment using Language model embeddings as a proxy.

The paper considers a variety of tasks, datasets and models and the results are clearly explained.

### Weaknesses
While this work clearly shows how the metrics they propose vary across groups in their datasets,  it's not clear to me what this actually tells us about human semantic navigation.

I think there is some value in the proposed metrics, but the experiments only provide a weak indication that they are useful for the classification tasks described.

### Questions
- Does the findings here align with other research on human semantic cognition?
- What other methods exist to measure the semantic dynamics of words produced in these experiments? For example. what metrics are used in the papers originally proposing the datasets? You mention that your finding corroborate theirs, but how?
- What are the advantages of this method over other methods?
- In the definition of entropy, the time series $\{x_t\}_{t=1}^n$ is a vector time series, so what is the median of a set of vectors?
- Similarly, the velocity and acceleration are vectors. Do you report their magnitude as the metric, or something else?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework to quantify human semantic navigation during concept production using trajectory-based metrics in transformer embedding spaces. It represents sequential concept generation (e.g., verbal fluency or property listing tasks) as a path through semantic space and computes geometric and dynamical metrics—distance to next, entropy, velocity, acceleration, and distance to centroid—to characterize this navigation. The method is applied to four datasets (neurodegenerative, swear-word fluency, and Italian/German property listing), showing that these metrics distinguish clinical groups, semantic categories, and languages. Results are robust across embedding models (OpenAI, Google, Qwen). The authors argue this approach bridges cognitive modeling and learned representations, offering potential applications in clinical and cross-linguistic research.

Overall, the approach is interesting. However, its mathematical grounding is limited, and comparisons to prior methods are absent. It is unclear whether this paper is primarily a methodological contribution, or an empirical contribution. While it seems to be a mixture of both, neither is particularly compelling. This work may be more suitable as an expanded manuscript for a cognitive science journal venue, where the results can be more effectively situated within the broader relevant literature, and targeted towards a more appropriate community.

### Strengths
1. Potentially novel conceptual framing: the paper presents a method for modeling semantic search as trajectories in embedding space, bridging computational linguistics and cognitive science. 
2. Methodological simplicity and reproducibility: the framework requires minimal manual annotation and is implemented with publicly available datasets and embeddings, making it scalable and easy to replicate.
3. Comprehensive empirical validation: the authors test across multiple datasets (languages, clinical populations, and semantic domains), providing strong evidence of the method’s generality.
4. Cross-model robustness analysis: the inclusion of multiple embedding models (OpenAI, Google, Qwen) and the correlation analysis (Figure 6) convincingly demonstrate stability of local trajectory measures across model architectures.
5. Interdisciplinary contribution: the work effectively connects semantic cognition, NLP, and neuropsychology, potentially valuable for both cognitive modeling and clinical diagnostics.

### Weaknesses
1. Limited theoretical grounding for metrics: while the chosen metrics (velocity, acceleration, entropy) are intuitive, their psychological interpretation is underspecified. The link between these geometric quantities and cognitive mechanisms of search (e.g., clustering/switching, semantic control) could be better formalized.
2. Simplistic dynamics assumption: the framework assumes Euclidean dynamics, even though embeddings are anisotropic and often non-Euclidean. The authors acknowledge this multiple times, but do not explore or justify why Euclidean treatment suffices.
3. Use of non-causal embeddings. It is a reasonable first step to use cumulative text embeddings, rather than independent word embeddings. However, by using non-causal encoder models for embeddings, for a sequence "A B C", the representation of token B has access to token "C", whereas in the earlier part of the sequence it does not. Rather than acquiring embeddings, the authors should use a causally-masked model, such that the token representation for "B" is the same across both "A B C" and "A B". This would likely make the trajectories smoother and more amenable to the kinematic metrics employed. 
4. Figures are dense: Figures 2–5 (and corresponding appendices) show many small boxplots and correlation matrices. It may be helpful to use comparison lines above the scatter plots rather than the correlation matrices, if the goal is merely to show which effects are significant. This is a more standard approach and would take less space, improving readability. 
5. No comparison to traditional linguistic baselines: the framework is presented as superior to “labor-intensive linguistic pre-processing,” yet there’s no quantitative comparison to classical measures (e.g., clustering, switching, word frequency, semantic similarity). A comparison with these approaches would strengthen the claim of added value.
6. Missing temporal information: since no timestamp data are used, “velocity” and “acceleration” are only metaphorical. The interpretation of these measures as cognitive dynamics rather than geometric derivatives is thus limited.
7. The work only characterizes semantic trajectories, it does not model them through some sampling from the latent space of a transformer. This makes the theoretical contribution weaker.

### Questions
1. How does this work improve upon prior baselines? 
2. Why are all of the metrics needed? What are the main distinctions between them? 
3. Do you agree with weakness 3? Can you perform another analysis with a causally masked transformer?

### Soundness
3

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
4

### Summary
This paper proposes a framework for characterizing human semantic navigation by representing concept production tasks (semantic fluency and property listing) as trajectories through transformer-based embedding spaces. The authors extract geometric and dynamical metrics including distance-to-next, velocity, acceleration, entropy, and distance-to-centroid from cumulative word sequences. They evaluate their approach on four datasets spanning clinical populations (Parkinson's, frontotemporal dementia), different languages (Italian, German), and semantic categories, showing that these trajectory-based metrics can distinguish between clinical groups and concept types across different transformer models.

### Strengths
- Novel computational framework: The trajectory-based approach to semantic navigation is creative, moving beyond static embedding analyses to capture dynamic aspects of semantic search. The use of cumulative embeddings (where x_t encodes items 1:t) is particularly interesting as it captures sequential dependencies.
- Robust empirical validation: The evaluation across four diverse datasets (clinical, multilingual, different task types) demonstrates broad applicability. The consistency of findings across three different embedding models (OpenAI, Google, Qwen) strengthens the results.
- Clinical relevance: Successfully differentiating neurodegenerative groups from healthy controls using distance-to-next and other metrics provides potential clinical utility. The finding that patient groups show greater variability and entropy aligns with executive dysfunction literature.
- Minimal preprocessing: The approach requires less manual intervention compared to traditional linguistic preprocessing methods, making it more scalable and reproducible.

### Weaknesses
- Missing baselines and comparisons: The paper lacks comparison to previous computational methods for analyzing semantic fluency data. No baselines using simpler embeddings (e.g., Word2Vec, GloVe) or traditional NLP metrics are provided. Prior work like Linz et al. (2017) used word embeddings for similar tasks but isn't compared against.
- Limited theoretical grounding: While the authors claim semantic retrieval can be "understood as navigation through a multidimensional space" (Hills et al., 2015), this theoretical framework needs stronger support. The connection between observed metrics and established cognitive theories (e.g., clustering-switching models by Troyer et al., 1997) is underdeveloped.
- Interpretation of results lacks depth: The clinical findings (e.g., "greater spread, higher variability, increased entropy" in patient groups) are presented without sufficient discussion of whether these align with expected patterns from cognitive neuroscience literature. Are these results validating known theories or revealing new phenomena?
- Euclidean assumption: The authors acknowledge but don't address their "assumption of Euclidean dynamics" which "overlooks the anisotropic nature of embedding spaces" (citing Nickel & Kiela, 2017; Ethayarajh, 2019). This could significantly impact the validity of velocity and acceleration metrics.

### Questions
- Centroid computation: The distance-to-centroid shows lowest inter-model correlation. Could you elaborate on why this metric is particularly sensitive to model-specific geometry? How does collapsing repeated properties affect this measure?
- Category effects: The category-specific patterns differ between Italian and German datasets. Do these differences reflect linguistic/cultural variations or task administration differences? This needs deeper analysis.

### Soundness
3

### Presentation
3

### Contribution
3
