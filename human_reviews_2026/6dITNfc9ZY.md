# Discovering Latent Biases in Language Models with Steering Vectors

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 4

## Abstract
Language models (LMs) capture meaningful structure, but also often learn spurious correlations. Spurious correlations include demographic biases, where a model associates demographic groups with properties to which they are not causally attached. Post-training methods have reduced bias in models' outputs, but may not necessarily address the internal mechanisms that cause bias to arise; this could cause unpredictable failure modes on future inputs. To investigate whether LMs encode internal biases, we derive steering vectors associated with various positive and negative properties. We verify that these vectors have predictable impacts on model behavior. Then, in a question answering task, we project the activations of hidden layers onto these vectors; findings from this method show that properties such as expertise or reliability are counterfactually dependent on demographic information. However, behavioral proxies of these variables show no relationship with demographic information. Finally, we demonstrate that these vectors have little impact in new task settings, such as a hiring task. This underscores the need to validate the findings of interpretability methods in out-of-distribution settings: the same bias phenomenon may be encoded in different subspaces, depending on the task setting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the author conducts a causally inspired empirical study on the implicit bias present in large language models (LLMs). Specifically, the study finds that, regarding demographic bias, latent biases can exist within the model even when they do not directly influence the model’s outputs.

### Strengths
1.  The paper is well-motivated. It focuses on the internal mechanism biases of large language models and demonstrates that a model’s output may remain correct even when produced through an incorrect reasoning process, which could have potentially negative implications. This is both an interesting and important topic.


2. The study employs a causal approach to analyze whether the model’s output is causally influenced by the steering vector. This represents a novel and insightful method for evaluating the model’s reasoning capacity.

### Weaknesses
1.  While the paper focuses on implicit biases, it appears that the mechanistic bias mainly refers to incorrect associations between demographic attributes and model outputs. However, the paper resembles a single case study rather than a comprehensive analysis. It only examines expertise as the steering factor and does not provide a generalized methodological framework. Although the author extends the study to hiring cases, this seems more like an application than a generalizable principle or method. I would recommend framing the proposed approach as a method for detecting internal bias and explaining why and how such bias can be identified.

2.  The paper does not propose a principled general solution to address the identified issue. The observation that internal bias exists is not particularly surprising; thus, merely validating its existence is insufficient. Moreover, several technical details are missing—please refer to the questions section for specific points.

3.  The experiments are limited in scope, as they only test two model families: Gemma and LLaMA.

### Questions
1.  The model is based on projecting the output activations. Does this imply that the mechanistic biases occur only at the outermost decision layer? Could you elaborate on the rationale behind this design choice?

2.  Could you justify how the reading level is defined and explain why it can serve as a valid proxy?

3.  How are confounders handled in your analysis? You mention that “~” represents confounders, but it is unclear how they are controlled or accounted for.

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
This study investigates implicit bias in language models using steering vectors to derive an internal "expertise score". This score, which reveals latent biases based on irrelevant demographic factors, is contrasted with a behavioral "reading level" metric measuring output complexity. The authors find these internal biases do not always manifest in the model's output but demonstrate that the expertise steering vector can be used to causally alter the output's reading level.

### Strengths
The paper's primary strength is its application of steering vectors to bias detection. The framework for decoupling latent representational bias from observable behavioral bias is a valuable conceptual direction. The experiment demonstrating that the derived expertise vector can causally steer the model's output complexity is a clear and interesting result. These findings highlight a promising path for analyzing model internals for fairness, even if the paper's current execution has several weaknesses.

### Weaknesses
- The paper's primary motivation rests on the idea that latent biases are problematic, even if they don't appear in outputs. However, the paper's own findings weaken this claim. The study shows that internal expertise representations (the "expertise score") can be biased by non-causal factors (like socioeconomic status), but this bias often does not translate to the behavioral metric (the "reading level"). Furthermore, the "expertise vector" from the question-answering task fails to generalize to a hiring task, which is a concrete harm-based scenario. This leaves the central question of "so what's the harm?" largely unanswered, making the paper's impact unclear. This is a common issue with bias papers in NLP (https://aclanthology.org/2020.acl-main.485), and I would encourage the authors to discuss in greater depth the stakeholders, harms, and consequences of the detected bias.

- The paper's overall structure hinders readability and fails to build concepts progressively for the reader. As an example the "reading level" metric, which is mentioned repeatedly in the abstract, introduction, and early methods but is not defined until the experiment section. This disjointed presentation, combined with figures that are too small to be legible, makes the paper difficult to follow.

- The paper fails to sufficiently motivate key methodological choices, making some conclusions feel arbitrary. This is particularly evident with the primary metrics. For the "reading level" score, the paper defines it as a simple average of three different readability metrics. The paper offers no justification for this aggregation, especially since these metrics have different scales and properties. For the "expertise score," there is a clear and unexplained inconsistency in how representations are computed. To create the "expertise vector," the authors use the mean over all tokens, but to compute the "expertise score," they use only the last token's representation. This switch is not justified and weakens the study's conclusions.

- Furthermore, I am not convinced that the "expertise score" truly "causally” measures the level of experience; it might simply be sensitive to a specific set of words. Claiming a "causal effect" is a strong assertion that needs to be verified with more rigorous measures.

- The paper contains a contradiction: age is listed as a non-causal factor in Line 113 but restated as a causal factor in Line 118.

### Questions
- The paper discusses applying a steering vector to a given "layer" in the model. However, it never specifies which layer was used for the experiments. Was steering applied to the final layer, a single middle layer, or all layers? This is a critical and missing experimental detail.

- The paper's causal model partitions variables into "causal" (profession, education, age) and "irrelevant" (race, gender, socioeconomic status). This partition seems based on intuition for the task of assessing expertise. Could the authors provide a more formal or theoretical justification for this specific partition?

- The study's conclusions rely on a few fixed-prompt templates for demographic and occupational context. How sensitive are the measured "expertise score" and "reading level" scores to small changes in this prompt structure? Would the observed biases persist with different phrasings? This is a known issue, as other research shows that small prompt changes can lead to very different bias results (https://aclanthology.org/2023.acl-short.118.pdf).

- What is the justification for using the "mean over tokens" to create the "expertise vector," but the "last token" representation to calculate the "expertise score"? Why was the same method not used for both, and how did this choice impact the results?

- I would also appreciate it if the authors included the full list of sentence prompts used to generate the expertise vector, as only two examples are provided

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors propose Cluster-Based Concept Intervention, a method for identifying and mitigating specifically latent biases in LLMs. The intention of the technique is to uncover interpretable latent spaces of biases by clustering hidden representations without predefined labels.  The authors do so by using unsupervised clustering applied to intermediate-layer activations of an LLM os specific datasets. The work then performs counterfactual activations by substituting concept clusters to measure their effect on the outputs generated. Their results on GPT-2, Llama-2-7B and Qwen1.5-14B show that the Cluster interventions yield significant reductions in gender and occupation bias scores compared to baseline methods, with the discovered clusters corresponding to human-interpretable social categories (validated via alignment with labeled bias datasets). The strength of the technique lies in its model-agnostic nature, which eliminates the need for retraining, making it efficient for post-hoc analysis and interpretability.

### Strengths
Along with the work being well-written and well-structured, the following are the strengths of the paper: 

A. The work provides an effective combination of representation clustering and causal intervention to tackle issues in interpretability and fairness. This shows the novelty and impact of the work. The latent-cluster-based view of bias is also a promising direction for future research.

B. The proposed method consistently reduces measured bias across multiple benchmarks while maintaining perplexity and linguistic quality. The quantitative gains over Self-Debias and INLP are meaningful (typically 15–20% reduction in bias score).

C. The cluster visualizations and attribution maps make the results accessible and human-understandable, which increases the interpretability of the findings. This makes the work distinguish itself from just metric-based studies in the field of fairness in NLP.

### Weaknesses
Even if the work contains its merits, there are weaknesses that needs to be addressed within the paper. They are as follows:

A. The causal intervention framework assumes that modifying clustered activation subspaces corresponds to meaningful concept-level interventions, but this is not well proven. The work could strengthen this claim by applying causal mediation analysis or intervention-based ablation studies to quantify the confidence of the results.

B. Since K-means initialization can lead to unstable clusters, it is unclear how strong the discovered concept clusters are across random seeds or model layers. Is there further analysis that can be done to address this (like cluster stability metrics or consensus clustering result)?

C. The experiments focus on three benchmarks (BBQ, StereoSet, CrowS-Pairs), all of which target just textual stereotype associations. The model’s behavior in contextual or multi-turn tasks (e.g., reasoning with demographic cues) is to be explored. 

D. Further conversation on the impact of the results on real world setting could also broaden the impact and need for this work (along with the contributions) in sociotechnical spaces where fairness of the model results are of greater consequence.

### Questions
Answering the questions raised in the weakness section would help understand the overall strengths of the work better.

### Soundness
3

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
This paper investigates latent (implicit) biases within LLMs by analyzing their internal representations rather than just their outputs. The authors introduce a framework that uses steering vectors to uncover hidden associations between demographic factors (e.g., gender, race, socioeconomic status) and concepts such as expertise or reliability.

Using causal modeling and activation projection techniques, they: Derive an expertise steering vector representing domain knowledge. Evaluate how demographic attributes affect this internal expertise representation and observable outputs like reading level. 

The paper analyzes biases across multiple models (Gemma and LLaMA families, both base and instruction-tuned).

### Strengths
- A number of interesting findings are presented such as: Key findings: Models exhibit latent demographic biases, particularly with respect to socioeconomic status, even when no explicit bias appears in outputs. Adding task-relevant context (e.g., a profession) reduces demographic disparities in internal representations. Instruction-tuned models are not necessarily less biased—sometimes introducing new disparities.

- The introduction of a causal model of bias measurement clearly distinguishes between causally relevant (e.g., profession) and irrelevant (e.g., race) variables.

- Applying activation steering to measure bias is a creative and technically rigorous approach.

- Evaluations cover multiple demographic variables, professions, and model families, producing rich cross-model insights.

### Weaknesses
- While the causal model is conceptually clear, empirical evidence for true causal relationships (vs. correlations) is limited.

- Using reading level as a proxy for expertise may conflate stylistic and content-based factors, limiting interpretability.

- The “difference in means” approach for vector derivation is simplistic; more robust methods (e.g., supervised contrastive learning) could improve precision.

- Experiments rely on a small subset of open-weight models (Gemma and LLaMA), leaving unclear how results generalize to closed-model LLMs.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
