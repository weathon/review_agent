# MindCraft: How Concept Trees Take Shape In Deep Models

- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Large-scale foundation models demonstrate strong performance across language, vision, and reasoning tasks. However, how they internally structure and stabilize concepts remains elusive. Inspired by causal inference, we introduce the **MindCraft** framework built upon **Concept Trees**. By applying spectral decomposition at each layer and linking principal directions into branching Concept Paths, Concept Trees reconstruct the hierarchical emergence of concepts, revealing exactly when they diverge from shared representations into linearly separable subspaces. Empirical evaluations across diverse scenarios across disciplines, including medical diagnosis, physics reasoning, and political decision-making, show that Concept Trees recover semantic hierarchies, disentangle latent concepts, and can be widely applied across multiple domains. The Concept Tree establishes a widely applicable and powerful framework that enables in-depth analysis of conceptual representations in deep models, marking a significant step forward in the foundation of interpretable AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors investigate how various concepts "appear" within machine learning models through the lens of causal inference. In particular, they introduce a framework, called MindCraft, which investigates how a pair of counterfactual statements (e.g., "You are a powerful leader" and "You are a powerless leader") differ in last-token representation as we change the layer number. This essentially serves as a signature for a given concept. This can then be used to identify when concepts diverge by placing a threshold for when two concepts should differ. This leads to the creation of trees which visualize branching in this system. The paper demonstrates this for various applications, along with an exploratory experiment that explores whether initial embedding distance is predictive of the separation between high-level concepts.

### Strengths
1. **Visually Appealing Representation** - The authors present a tree-based diagram visualizing how different concepts emerge in deep learning models. The representation is visually appealing, and makes it clear the order in which these concepts occur. Such a representation could be valuable, as it allows us to better understand the dynamics underlying these deep models. 
2. **Extensive Examples** - The authors present the material in a fairly clear way through the extensive use of examples. For example, page 7 presents numerous examples of concepts emerging in this context, which makes it clear how such trees operate. Moreover, the paper is generally well-presented, with various examples of how concepts emerge, and the meaning of such emergence. 
3. **Interesting Exploratory Experiments** - In Section 5, the authors explore the disentanglement of input embeddings from concepts, essentially looking at whether token-level embeddings can predict which concepts emerge. Such an experiment is interesting because it investigates the relationship between model representation with the data itself, to identify whether such concept emergence is inevitable.

### Weaknesses
1. **Unclear Generalization** - While the authors present numerous examples throughout the paper, my biggest worry is that the patterns seen here might not generalize. Specifically, my worry is two-fold a) concepts might not always be suddenly amplified, yielding some of their analysis moot, and b) it is unclear how to interpret or use the presented concept trees in practice. What is the insight useful for/how can we interpret this insight? 
2. **Lack of Justification for Why** - The authors provide little theoretical justification for why certain layers might propagate signals for concepts moreso than other layers. Many of the plots in the experiments section are for individual examples (which helps to get the point across), but the lack of aggregate analysis makes it hard to understand whether such trends hold across settings, and moreover, why such trends should even hold in the first place.

### Questions
1. How should MindCraft be used in practice?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
To address opacity of neural networks, the paper introduces MindCraft, a method to inspect how the internal representation of an abstract concept evolves through the layers of a neural network. To do so, it creates counterfactual pairs and tracks internal representation differences between at various depths of a foundation model, identifying where concepts diverge. MindCraft can also provide interpretable visualizations for model development and debugging. The reported experiments and visualizations suggest generalization across domains.

### Strengths
- Novelty: I like the idea of studying how the internal representation of an abstract concept evolves across different layers. A tool capable of performing such analysis could be valuable, as it enables a more comprehensive understanding of the model itself.

### Weaknesses
Major:

- **W1** - The authors claim that MindCraft explains how large foundation models internally structure abstract concepts (l. 470–472). However, according to the experimental section, the proposed methodology is tested on only one LLM (Qwen2.5-7B-Instruct). Therefore, the statement in the conclusion lacks sufficient empirical support. Without this claim, the overall impact of the paper is considerably reduced.
    
- **W2** - The results presented in the experimental section are mostly qualitative (Figures 3-4-5). Although the authors provide several qualitative examples, a formal validation (established metrics and/or statistical tests) supporting the consistency and reliability of the approach appears to be missing. This limitation likely stems from the absence of a theoretical ground behind both the motivation and the methodology.
    

Minor:

- **W3** - Motivations seem weak. In l.194, the authors claim “Concept formation, therefore, follows a branch-and-stabilize process:” and, later in l.198 “concept-level organization is not static, but unfolds progressively through the network.” At this point of the paper, this is supported only using a single example (Fig.2). Also, I would expect the validity and extent of such a general claim, as well as the particular observed dynamics of the similarity score, to depend strongly on the specific examples, i.e., sentences and context.

- **W4** - The necessity of using the principal directions extracted from the SVD with respect to the raw ( $W_V$ ) is not well justified. Despite showing several examples in the appendix, the rationale for this choice remains primarily qualitative.
    
- **W5** - The concept tree is constructed considering only self-attention. However, a generic LLM is far more complex than that, as LLMs typically employ multi-head attention. Consequently, the internal representation of a specific sequence at a given depth is only partially analyzed by the proposed methodology. Shouldn't it also account for the other attention heads at the same depth?

### Questions
- **Q1** (related to W3) - Have you observed any scenario where the dynamics of the counterfactual pair is not “branch-and-stabilize“?
    
- **Q2** (related to W5): How do you think your methodology extends to or interacts with multi-head attention or more complex architectures?

While I remain open to a constructive discussion, I believe the paper requires substantial improvement, especially towards establishing a clearer/stronger theoretical foundation for both the proposed method and its evaluation. At this stage, the gap between the current submission and a version that would meet the bar for acceptance still feels wide to me.

### Soundness
1

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
The paper introduces MindCraft, a framework for analyzing how llm internally structure abstract concepts. The proposed method uses counterfactual input pairs and tracks the difference in the last token's representation across layers. The core of the method which is Concept Path, is computed by projecting the last token's attention value vector onto the principal components of that layer's value transformation matrix. By finding the layer where the Concept Paths of a counterfactual pair first diverge, a hierarchical visualization of when and where different concept split into separable subspaces.

### Strengths
1. The paper is well-written and easy to follow.
2. The branch-and-stabilize hypothesis provides a strong, intuitive foundation for the work.
3. The paper demonstrates the flexibility of the proposed framework by applying to multiple model across three domains.

### Weaknesses
1. The same citation appears in two different formats.
2. The main text of the figure needs to provide guidance on how to interpret the results in the figure (e.g., what the takeaway is).
3. The paper need to provide a claim that the Concept Tree faithfully represents the model's internal reasoning process.
4. The paper need to clarify the choice of parameters k and tau, or at least analyze the change.
5. The experiment should be compared with baselines (e.g., RepE, LRH). The only baseline comparison is "raw Value" vectors which isn't benchmarked on any metric.

### Questions
Look at the weaknesses

### Soundness
2

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
3

### Summary
The paper introduces MindCraft, a framework for explaining AI models by constructing concept trees that illustrate the hierarchy of which neural network layers concepts diverge in the internal representations. The tree is constructed by comparing the spectral decompositions of the representations of a counterfactual pair of inputs and splits the tree at the first layer in which the representations begin to differ. Experiments demonstrate clear examples of trees generated for LLM tasks and also highlight some interesting properties of the approach.

### Strengths
1. Mindcraft presents a novel framework for interpretable AI that pinpoints precisely where concepts form within the layers of the model.

2. Experiments are comprehensive, easy to understand, and show a variety of properties of the pipeline.

3. The writing is very clear.

### Weaknesses
1. There is little theoretical justification of any of the proposed methodology. The final tree is dependent on many parameters left up to the user, and the result is open to vague interpretation. The paper claims that MindCraft “systematically traces how abstract concepts emerge”, but it is unclear why the resulting concept tree answers the “how” question.

2. Counterfactual quantities, from the perspective of causal inference, are not easy to infer, especially in cases where one is attempting to simultaneously infer something about the same input in two different interventions. It is not clear what kinds of assumptions are made to allow for this.

### Questions
1. Is there a reason that concept trees are defined in tree format? It seems like all concepts are leaf nodes that branch off of a single main line of nodes that represents the undisambiguated concepts. Could concept trees be instead simply represented as a list of concepts sorted by the order of the layers in which they were disambiguated?

2. Do the patterns that arise in the presented experimental results look similar when applied to non-language tasks?

3. Are the concepts in MindCraft simply defined as sections of the input, or could they represent more abstract concepts produced within the internal workings of the neural network?

### Soundness
2

### Presentation
4

### Contribution
3
