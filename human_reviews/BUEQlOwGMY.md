# Object-Based Sub-Environment Recognition

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Deep learning agents are advancing beyond laboratory settings into the open and realistic environments driven by developments in AI technologies. Since these environments consist of unique sub-environments, empirical recognition of such sub-environments that form the entire environment is essential. Through sub-environment recognition, the agent can 1) retrieve relevant sub-environments for a query, 2) track changes in its circumstances over time and space, and 3) identify similarities between different sub-environments while solving its tasks. To this end, we propose the Object-Based Sub-Environment Recognition (OBSER) framework, a novel Bayesian framework for measuring object-environment and environment-environment relationships using a feature extractor trained with metric learning. We first design the ($\epsilon,\delta$) Statistically Separable (EDS) function to evaluate to show the robustness of trained representations both theoretically and empirically that the optimized feature extractor can guarantee the precision of the proposed measures. We validate the efficacy of the OBSER framework in open-world and photorealistic environments. The result highlights the strong generalization capability and efficient inference of the proposed framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes the Object-Based Sub-Environment Recognition (OBSER) framework. OBSER identifies sub-environments with three relationships: object-object, object-environment, and environment-environment relationship. The effectiveness of OBSER is measured with the proposed statistically separable (EDS) function in the Minecraft environment.

### Strengths
- OBSER identifies sub-environments with three relationships between objects and environments, and exhibits better distinguishability in terms of EDS compared to other off-the-shelf vision models.

### Weaknesses
- The application of OBSER is not clear. I'm not sure how OBSER will facilitate downstream tasks, e.g. decision agents in Minecraft like DreamerV3[1], Voyager [2], or GITM [3].

[1] Mastering Diverse Domains through World Models

[2] An Open-Ended Embodied Agent with Large Language Models

[3] Generally Capable Agents for Open-World environments via Large Language Models with Text-based Knowledge and Memory

### Questions
- What if the category of possible objects is unknown, e.g. in the open-set setting?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper proposes the Object-Based Sub-Environment Recognition (OBSER) framework, a novel Bayesian framework for measuring object-environment and environment-environment relationships using a feature extractor trained with metric learning. The key idea is the introduction of a statistically separable (EDS) function and using it to perform (i) object-object similarity, which involves obtaining the closest class of objects from a list given a query object, (ii) object-environment recognition, which involves retrieving the closest environment to a given object and (iii) environment-environment recognition, which defines the difference between two sub-environments. Experiments to recognize environments are done on two datasets, the ImageNet based dataset, and a dataset of curated environments from Minecraft.

### Strengths
-	The primary motivation behind the paper is sound. Indeed, environment recognition using relationships between objects and environments is an interesting problem in embodied agents.
-	The results do illustrate the claim that higher difference between epsilon and delta values lead to a better accuracy score. This is reflected in models for both Tables 1 and 2.

### Weaknesses
-	Not clear what objects and environment are: By reading the paper starting from the introduction, it is not clear what objects and environments mean. The authors show some examples of objects and biomes as environment in the Minecraft example, but none for the ImageNet dataset. This makes it difficult to understand the contribution of the work.
-	Results are difficult to interpret: There are mainly just two results described in the paper in Tables 1 and 2, which are the respective classification accuracies for ImageNet and Minecraft datasets. It is difficult to interpret these tables. For instance, are the differences between metric and self-supervised learning methods the main observation, or the relationship between EDS values and different models?
-	Real-world applications are unclear: While the motivation of the work is sound, it is not straightforward to interpret how this paper contributes to real-world object and scene recognition. The paper does not contain any examples of real-world object scenes or object recognition. The only examples provided are for Minecraft, which is still a simulation environment, and not reflective of human-centric objects and environments.
-	[Minor] Unnecessary math: There is a lot of mathematical terminology introduced in Sections 3-5 (besides the main contribution in Section 4.1), which are not necessary to be present in the main paper.
-	[Minor] Many details are missing from the main paper and are present in the supplementary. The authors should consider transferring some details about the implementation from supplementary to the main paper. For any details, the reader has to constantly switch between paper and supplementary, which is not a good user-experience.

### Questions
-	Is there any justification for the type of classifiers provided in Tables 1 and 2? In Table 1, linear, mean and KNN classifiers are used, while only mean and KNN classifiers are used in Table 2. Also, why the difference between the shots of mean vs KNN (1,3,5 vs 3,5,7).
-	For the ImageNet dataset, what are some examples of objects and environments? It is unclear from reading the paper.
-	In conclusion, the authors mention that by integrating the proposed method with embodied object recognition or navigation modules, inference accuracy can be improved. Can the authors provide some justification with a real-world use-case about what is the intuition behind this?
-	What classification accuracies are mentioned in Tables 1 and 2? Is it the object-object similar classes?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents a Bayesian framework to recognize sub-environments within complex, dynamic environments. OBSER enables agents, like robots, to identify sub-environments based on objects present, facilitating task-driven navigation and inference in open-world scenarios. The framework introduces EDS function to improve the robustness of feature representations and utilizes metric learning for object-object, object-environment, and environment-environment relationships.

### Strengths
OBSER provides a holistic approach to sub-environment recognition, measuring three relationships—object-object, object-environment, and environment-environment—which enables better contextual awareness.

The introduction of the EDS function to assess separability and concentration offers a robust way to manage feature representations, addressing a gap in object-based environmental recognition.

### Weaknesses
I find that this method may be challenging to implement for embodied robots. First, constructing episodic memory seems crucial for task completion success, yet several questions arise: (1) How was this memory constructed? (2) How could it be constructed effectively with limited experience? (3) How can retrieval be managed efficiently as memory size increases?

Most importantly, I am uncertain about how the object-object, object-environment, and environment-environment relationships contribute to embodied tasks. Without ablation studies or proof, it’s hard to determine the critical importance of these relationships.

Are there other baselines with which EDS could be compared? The paper would benefit from broader comparisons with other state-of-the-art environment recognition frameworks to better highlight OBSER's distinct advantages and limitations in context.

The diversities of Minecraft environment and objects seems limited.

OBSER's reliance on object distribution might limit its effectiveness in sub-environments where objects are scarce or ambiguous, which could impact performance in less structured real-world spaces.

### Questions
Please refer to the weakness.

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
The authors propose a framework for an agent to infer its sub-environment through the measurements of object-object, object-environment, and environment-environment relationships. The authors validate their framework in Minecraft, while also providing some preliminary results on the ImageNet dataset. The paper is well structured, and the authors show several relevant results, including the relevance of statistically separable EDS functions to achieve accurate measures for their downstream environment inference.

### Strengths
The paper is generally well structured. The authors explain each part of their proposed method in detail, including for example the relevance of statistically separable EDS functions to achieve accurate measures for their downstream environment inference, and the empirical implication of hyperparameter choice for downstream inference (e.g. the choice of Tau for KL divergence in Figure 6.).

### Weaknesses
It is hard to quickly have a notion of which parts of the proposed methods, exactly, are novel. The authors use several existing methodologies in their proposed framework, but fail to appropriately specify which of these, in particular, are novel propositions or implications. It is also hard to connect the motivation of this work to the tasks and results shown. In particular, the emphasis on the motivation for "real-world" applications, and complex natural environments is lost by the simplicity of the test settings (e.g. virtual world or fixed datasets).

### Questions
- It would be worthwhile to adjust the tone of the claims in the paper to better align with the results shown. The results may show interesting results in a "simulated environment, towards more complex environmental settings" perhaps even eventually leading to real-world, but as far as this work goes there is a wide gap between simulated and real-world settings, since no robotic experiments were provided. Below are some of the most relevant parts, strongly suggesting (non-existing) results in real-world settings
     - The abstract
     - The introduction should reflect this (3rd claim)
     - Figure 1: The caption should be updated (it is not, in fact, a real-world agent)  
     - Title in 6.2 should change

- Section 4: Which of these are new propositions and which of these are derived from existing work? This should be made very explicit.
- Missing y-label in Fig. 5 and Fig. 6
- English should be improved throughout:
  e.g. "which computes the kernel density accumulated with class-wise distribution.", or "We utilized pretrained weights for every models." etc.

### Soundness
2

### Presentation
3

### Contribution
2
