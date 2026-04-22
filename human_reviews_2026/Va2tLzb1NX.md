# From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Transformers have achieved state-of-the-art performance across diverse language and vision tasks. This success drives the imperative to interpret their internal mechanisms with the dual goals of enhancing performance and improving behavioral control. Attribution methods help advance interpretability by assigning model outputs associated with a target concept to specific model components. Current attribution research primarily studies multi-layer perceptron (MLP) neurons and addresses relatively simple concepts such as factual associations (e.g., Paris is located in France). This focus tends to overlook the impact of the attention mechanism and lacks a unified approach for analyzing more complex concepts. To fill these gaps, we introduce Scalable Attention Module Discovery (SAMD), a concept-agnostic method for mapping arbitrary, complex concepts to specific attention heads of general transformer models. We accomplish this by representing each concept as a vector, calculating its cosine similarity with each attention head, and selecting the TopK-scoring heads to construct the concept-associated attention module. We then propose Scalar Attention Module Intervention (SAMI), a simple strategy to diminish or amplify the effects of a concept by adjusting the attention module using only a single scalar parameter. Empirically, we demonstrate SAMD on concepts of varying complexity, and visualize the locations of their corresponding modules. Our results demonstrate that module locations remain stable before and after LLM post-training, and confirm prior work on the mechanics of LLM multi-lingualism. Through SAMI, we facilitate jailbreaking on HarmBench (+72.7%) by diminishing “safety” and improve performance on the GSM8K benchmark (+1.6%) by amplifying “reasoning”. Lastly, we highlight the domain-agnostic nature of our approach by suppressing the image classification accuracy of vision transformers on ImageNet.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Scalable Attention Module Discovery (SAMD), a method for identifying attention heads associated with abstract “concepts” in transformers via cosine similarity, and Scalar Attention Module Intervention (SAMI), a one-parameter mechanism to amplify or suppress their influence. The approach is demonstrated across LLMs and ViTs, with qualitative interpretability results and modest quantitative effects on reasoning, safety, and vision benchmarks.

### Strengths
- Introducing attention-head–level concept attribution is an original direction that is computationally light and easily applicable to diverse transformer architectures.

 - The same pipeline is used across text and vision models, suggesting potential generality and extensibility.

 - The paper contributes to ongoing efforts to connect internal transformer components to semantic behaviors, particularly through sparse, interpretable “modules.”

### Weaknesses
- The evaluation is dominated by qualitative visualizations and anecdotal examples. There are no robust statistical analyses, reproducible metrics, or causal validation to confirm that the discovered modules truly mediate the claimed concepts.

 - The use of cosine similarity as a proxy for conceptual alignment is not theoretically or empirically justified; results may reflect correlation, not causation.

 - Choices of K (number of heads) and s (scaling factor) appear arbitrary, tuned via small grid searches without sensitivity analysis.

 - The paper never clearly defines in what sense the approach is “concept-agnostic.” This weakens the interpretive and theoretical clarity of the contribution.

### Questions
How is a “concept” defined in this framework, and how can the method be considered “concept-agnostic” if it depends on concept-specific vectors?

Can the identified attention modules be causally validated (e.g., via path patching or feature ablation)?

How robust are the discovered modules to randomization, different seeds, or model variants?

Could quantitative measures (e.g., mutual information or probing accuracy) strengthen the claims?

The heatmap figures are difficult to distinguish between colors, which are somehow important (e.g., with bold borders) and which are not.
Please, change the color.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces an attribution method (of model components, not the input) for explaining transformer-based models. Specifically, it identifies the most important attention heads across the model which are relevant to a concept of interest i.e. the French language. To do so, they identify the maximally close (in terms of cosine similarity) attention heads to concept vectors captured from a positive dataset. They then propose that the identified components of the model can be up- or downscaled to change model behavior, thus verifying their component extraction, and providing a real use case for the method.

### Strengths
Very interesting method with good novelty. Unlike many previous MLP neuron attribution approaches, this is the first I have seen which identifies attention concepts. This is an interesting and critical result with the stronger push for mechanistic interpretability and adjacent approaches in the modern XAI literature. 

Well written with extensive experimental results.  

I think the simplicity of the approach is a benefit to its usability. I feel that I could replicate this with a few hours of work if I had access to the datasets.

### Weaknesses
Minor – plainly calling this an attribution method feels misaligned with the literature. Attribution methods often refer to input (feature) attribution. This is more aligned with neuron attribution. Perhaps it should be attention attribution but not to be confused with input attribution using attention weights/gradients. 

There are not any true comparisons against other methods. It is hard to tell if this should be negative because it may be challenging to create a fair comparison against a similar MLP based method. I think it could have been possible to use knowledge editing benchmark.

### Questions
Did the authors consider a knowledge editing style benchmark?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose SAMD and SAMI - Scalable Attention Module Discovery and Intervention. Given a feature vector $v_c$, SAMD discovers a set of $K$ attention heads whose output $a_{l,h}$ has on average high cosine similarity to $v_c$. They call this set of attention heads (which is a circuit in a way) a module.  Given a the circuit of $K$ attention heads, SAMI amplifies or suppresses the module by scaling the attention heads output $a_{l,h}$ by a scaling factor $s$ which they choose on a per problem basis using grid search. They demonstrate the effectiveness of their methods in 4 different experiments:
a) They find and steer modules that correspond to SAE features. This motivates the name - as they choose concepts, and search for relevant modules related to the concept. This is done by generating a dataset $D_p$ for which an SAE feature $v_c$ is highly activated and then running SAMD with $v_c$ and $D_p$.
b) They find and steer a module that corresponds to reasoning capabilities in the model. They report improved scores on the GSM8K reasoning benchmark.
c) They find and steer a module that corresponds to a refusal direction. The report improved attack success rate over orthogonalization of the refusal direction in two out of three cases. 
d) They find and steer a module that corresponds the classification of a given ImageNet target on ViT-B/32 21k, showcasing effectiveness of targeted unlearning of a single class.

### Strengths
* Novel and elegant method for circuit discovery
* Clear presentation of the findings
* Demonstration of effectiveness of method on a broad variety of applications over two different modalities. Especially in the vision literature this is addressing a research gap, as vision-circuit discovery remains under-explored.

### Weaknesses
* 4.2 the construction of $D_p$ is unclear from just reading the main body of the paper. 
* Concept figure should be improved
	* Font way to small
	* No order of panels provided
	* SAMI is not explained in the rightmost panel
* Comparison to baseline such as e.g. difference in means is missing for 4.1, 4.2 and 4.4. If this concern is addressed appropriately I will improve my score.
* In 4.2 the authors only report evals on the dataset that they used for construction. An OOD reasoning benchmark eval would be useful to evaluate the generality of the reasoning-module.
* Only ViT-B 32 evaluated in 4.4). Experiments with at least ViT-L would be recommended as ViT-B often shows different behaviors from its bigger counterparts. If this concern is addressed I will improve my score.

### Questions
* Did the authors explore including highly negative cosine sim attention heads (e.g. Fig 24.) and flipping the $s$ value for these heads? If so, that did they find? Especially in Fig 24 gemma 7b one head seems to have the highest absolute alignment with -0.4 similarity while the highest positive alignment is only 0.3.
* What is the rational behind the $D_p$ construction via the test samples of GSM8K? Is it that the prompts are explicitly encouraging to reason?

### Soundness
3

### Presentation
3

### Contribution
3
