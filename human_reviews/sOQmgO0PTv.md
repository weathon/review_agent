# Discretization of continuous input spaces in the hippocampal autoencoder

- Avg Score: 3.67
- Decision: Reject
- Scores: 5, 3, 3

## Abstract
Understanding the encoding mechanisms of hippocampal place cells remains a significant challenge in neuroscience. Although sparse autoencoders have been shown to exhibit place cell-like activity, the underlying processes are not fully understood. In this study, we compare spatial representations learned by dense and sparse autoencoders trained on images of 3D environments and find that only sparse autoencoders with orthonormal activity regularization in latent space produce place cells. We then show that this regularization promotes similar images to map onto the same neurons, acting as a locality-sensitive hash function. Notably, we demonstrate that these neurons are visually interpretable through activity clamping and decoding, suggesting the formation of detailed episodic memories at the single-neuron level. We then introduce a novel metric to quantify how neurons discretize the image space into disjoint receptive fields, revealing that sparse autoencoders tile input spaces with minimal overlap. Furthermore, we observe that whereas dense autoencoders generate population codes resembling visual cortex activity near criticality, sparse autoencoders produce higher-dimensional codes, thus suggesting a similar coding strategy in the hippocampus. Extending our approach to the auditory domain, we also replicate the emergence of "frequency place cells" by training sparse autoencoders on audio snippets sampled from a frequency-varying signal, and show that population representations retain the statistical structure of the sample distribution. Lastly, we demonstrate that reinforcement learning agents can leverage these high-dimensional image representations to solve complex spatial-cognitive tasks, despite their inherent brittleness. Overall, our findings elucidate how sparse input compression in autoencoders can give rise to discrete, interpretable memories, establishing an explicit link between episodic memory formation and spatial representations in the hippocampus.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper demonstrates a sparse autoencoder-based modelling approach to understand the encoding mechanism of place cells. Through a variety of experiments on spatial navigation tasks with visual stimulus, sound frequency and reinforcement learning agents, the authors have demonstrated that sparse autoencoder can be used to explain the computational mechanism behind the discretization of continuous, high-dimensional inputs i.e., place cells in the hippocampus, and that memory encoding and place cells are closely linked.

### Strengths
* The authors have conducted a comprehensive set of experiments, including:
   * spatial navigation with visual inputs: I really appreciate the usage of Animal-AI environments for the experimentation, which distinguishes this paper with earlier ones like Benna and Fusi (2021) that only performed the experimentation on simulated tasks with low-dimensional inputs.
   * In-depth analysis of the latent structure of the autoencoder.
   * Investigation of frequency-sensitive “place cells”, which shows the generalizability of the model.
   * Reinforcement learning agents.
* The writing is clear and easy to follow.
* There is no obvious error in the paper.

### Weaknesses
* Main contribution is unclear: 
  * Despite the comprehensive experiments presented in this work, I found the main contribution poorly defined, especially given that earlier works (e.g., Benna and Fusi 2021, Santos-Pata et al. 2021) have followed similar autoencoder-based approaches. 
  * For example, the authors stated in the final sentence of abstract that the findings demonstrate how sparsity gives rise to interpretable memories and thus establishes a link between memory and place cells. However, it has been shown in multiple models that sparsity is a necessary condition for obtaining place cells, regardless of the model (e.g. Chen et al., 2024, Benna and Fusi, 2021), and the second part of the claim is exactly what has been studied in Benna and Fusi 2021. Given these, what is the novel contribution of this paper?
  * One possible contribution that the authors can highlight is that the paper provides a full-scale investigation of 1) the necessity of sparsity and 2) the relationship between memory and space representation with experiments on higher-dimensional inputs of different modalities. However, this is only my opinion, and the authors may have other contributions they wish to highlight. In that case, please make sure they are sufficiently highlighted in the paper.
  * Related to this, I feel the paper could be improved by having a ‘Related Work’ section, which highlights what has been done and what hasn’t been done in earlier literature, and how this paper addressed these gaps.
* Lack of key experimental details in the main text:
  * I agree many experimental details (e.g., training information like learning rates) don’t have to be included in the main text. However, this paper has missing experimental details that affects understanding. Here’s an incomplete list of them:
    * Section 2.1: I hope the authors can use a few sentences to describe what exactly does the agent do in these 4 animal-ai environments, or what exactly the task is, as some readers might be unfamiliar with the dataset.
    * Section 2.2: In the first sentence of the second paragraph the authors mentioned “embedding of all samples” – as a reader I’m not sure what all samples refers to. I assume they are the images sampled from the tasks? However, which task did you use to get the samples? One of them or all? Also, although CLIP, UMAP, DBSCAN are well-known embedding/dim-reduction methods, you should cite relevant literature.
    * Section 2.4: The authors mentioned that the autoencoder is trained on one environment and tested on all others. Which environment is the training one and which are the testing ones? Did you do a cross-validation like testing? If not, I think it is preferable to do so as it will demonstrate the robustness of the results better.
    * Section 2.5: I struggle to understand the purpose of Fig 5d and the sentence associated with it: why do we need to bias the sampling towards certain frequencies?
    * Section 2.6: The authors moved the whole task description of these RL experiments to the Appendix, so it is quite unclear to me what exactly is the performance in Fig 6b measured on. It would be much nicer to include in the main text what the tasks and metrics are.

* Redundant experiments: I suggested in the previous point that the paper could benefit from adding more experimental details. This might be challenging given the limited number of pages. However, I believe there are some redundant experiments in this paper that can be removed to give space to more details for other, more informative experiments. For example, I didn't find the results in Fig 3 very unexpected/intriguing. Since orthogonality constraint is imposed on the autoencoder, it is unsurprising that the eigenspectrum of the sparse autoencoder is a flat line, because it is exactly what orthogonality means i.e., neurons will be more independent and explain equal amounts of variance in the data. I understand that the authors wanted to reproduce the results with mice in Stringer et al. 2019 but keep in mind the causality of the results: in mice, the eigenspectra reveal a coding mechanism, whereas in autoencoders, the coding mechanism (orthogonality/sparsity) is first imposed so the observed eigenspectra are expected. I’m wondering if the authors are willing to remove this experiment, or simply move it to the Appendix.

Overall, given the final two points above, I suggest the authors to think about the allocation of paper content on different experiments more carefully. Specifically, they may want to consider removing contents related to Fig3 and add more details to other experiments.

### Questions
* Abstract: I hope the authors could shorten the abstract and extract important message they want to deliver in this paper. Currently, the abstract is a pretty detailed summary of all the experiments. I don’t feel this level of details is necessary for the abstract.
* I wonder if the results (e.g., place fields) in the experiments are a function of \lambda? For example with a smaller \lambda i.e., weaker sparsity constraint, will the rate maps become more band-like? You may want to include these as appendix, just to show the sensitivity of the results to a quite important hyperparameter. 
* I found the results in Fig 2d on memory quite interesting. I wonder if the authors have come across Radhakrishnan et al. 2020 that shows that autoencoders are associative memory models (like Hopfield Nets). This may offer you an interpretation of your results that in sparse autoencoders, individual neurons are basically “attractor neurons”.
* Is it possible to map the sparse autoencoders to the hippocampal circuit? For example, dentate gyrus is known to perform some kind of orthogonalization. How can it be related to the circuit mechanism implied by the sparse autoencoder?

Overall, I will suggest a score 5 to this paper. However, if they authos can address the concerns raised above, particularly on highlighting the contribution better, I'm happy to raise it to 6.

References: 
Radhakrishnan et al. 2020: Overparameterized neural networks implement associative memory

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper investigates how sparse autoencoders, equipped with orthonormal activity regularization, can simulate place cell-like activity in artificial neural networks, potentially providing insights into hippocampal encoding mechanisms. Through comparisons between dense and sparse autoencoders trained on images of 3D environments, the authors demonstrate that sparse autoencoders generate discrete, interpretable representations by mapping similar inputs to localized receptive fields with minimal overlap. This mechanism is similar to locality-sensitive hash, creating high-dimensional representations that resemble spatial maps, similar to place cells in the hippocampus. Additionally, the study generalizes these findings to the auditory domain, showing that sparse autoencoders can produce frequency-specific receptive fields, suggesting modality-independent encoding.

The authors introduce a novel metric to quantify the discretization of image space and present results that reinforcement learning agents benefit from these high-dimensional sparse representations in some environments. The paper concludes with a discussion on the biological plausibility of orthonormal regularization, suggesting it could be implemented through neural processes like lateral inhibition and homeostatic plasticity. Overall, the work bridges computational neuroscience and artificial intelligence by proposing a model where sparse input compression supports episodic memory formation and spatial representation in a hippocampal-inspired framework.

### Strengths
#### **Originality**
This paper introduces an innovative approach by using sparse autoencoders with orthonormal regularization to simulate place cell-like behavior, bridging neuroscience and machine learning. This combination of techniques is a creative application in hippocampal modeling, especially as it applies across sensory modalities (visual and auditory) and introduces a new metric for discretization of input spaces.

#### **Quality**
The experimental design is thorough and well-documented, including comparisons across dense and sparse models, detailed model architecture, and training procedures. The experiments are contextualized within neuroscience theories like pattern separation/completion, lending scientific depth and grounding to the study. However, while rigorous, the experiments are limited to specific, controlled environments, which may impact generalizability.

#### **Clarity**
The paper is generally clear in explaining complex ideas, with visuals that effectively illustrate neuron activity maps and spatial encoding properties. The appendix offers additional transparency regarding methods, adding to reproducibility. Nonetheless, some technical details—such as orthonormal regularization’s biological plausibility—could be expanded to improve accessibility.

#### **Significance**
This work has notable implications for both neuroscience and AI, providing a potential model for hippocampal-like representation and spatial coding that might inform further studies in spatial cognition. Its cross-modal approach (visual and auditory) hints at a generalizable model for sensory representation, though further validation across a broader range of tasks and sensory modalities would strengthen its practical impact.

### Weaknesses
### **Inaccurate use of terms**
The paper’s use of terms like "memory" and "locality-sensitive hashing (LSH)" misrepresent its functionality, suggesting capabilities that the model doesn’t fully achieve.

The term "memory" implies persistence, flexible retrieval, and time-based storage. Episodic memories, in particular, include context, time, and location (what, when, where). The model, however, lacks mechanisms for retaining and recalling information across time, relying instead on transient representations accessed by direct activation (neuron clamping). Referring to these as "representations" or "spatial encodings" would more accurately reflect their function.

LSH typically involves hashing for efficient, similarity-based retrieval, using probabilistic mappings to preserve approximate similarity and enable efficient search. The model’s rigid clustering lacks LSH’s probabilistic flexibility and retrieval efficiency, as it does not include true hashing or indexing. Describing the model as “localized spatial encoding” or “input-space tiling” would avoid implying LSH-like efficiency or robustness.

__Suggestion__: Replace "memory" with "representation" and "LSH" with “localized encoding” to better match the model’s capabilities. If these terms are retained, clearly specify that the model only partially mimics these concepts. Compare to methods like receptive field mapping or population coding for a more accurate portrayal.

### **zero-shot learning**
The authors claim that sparse autoencoders exhibit zero-shot learning of place cells across novel environments, but there is a limited exploration of why or how this occurs. The model’s capacity for generalization appears promising but has not been properly examined. Analyzing the network's internal structure or activity patterns when exposed to novel environments could reveal mechanisms that enable this generalization. Running ablation studies on the architecture (e.g., removing the orthonormal regularization) or visualizing the learned feature space across environments would provide insights into which aspects drive the process. Also, how does the decoding look like during zero-shot learning?

Moreover, when the model encounters unseen data, interpreting its representations as "memory" becomes challenging, as true memory implies both stored information and the ability to generalize or recall similar contexts flexibly. When the model encounters unseen data, it generates new, sparse representations in the latent space without direct prior storage or memory of these specific inputs. This process resembles on-the-fly encoding rather than recall from stored experiences, suggesting that the model is generating new representations rather than accessing a pre-existing memory.

In zero-shot learning, the model's response to new data could reflect a form of pattern generalization based on similarities to previously encoded features. While this may resemble aspects of memory, it lacks associative recall or the ability to adapt previously stored information to novel contexts, a hallmark of true memory.

The representations formed for new inputs suggest flexible encoding rather than dynamic recall, as they are generated without modification of or access to previously stored states. This is akin to a flexible encoding mechanism rather than a memory system, as it adapts to new data but does not actively retrieve, compare, or contextualize it based on stored knowledge.

### **bio plausability and regularization**
The paper claims that sparse autoencoders with orthonormal regularization simulate place cell-like properties in the hippocampus, but the biological plausibility of this regularization remains speculative. Orthonormal regularization is uncommon in neural modeling and lacks a clear parallel with hippocampal processes.

Moreover, it is not yet clear if the place cell-like properties are specific to this regularization or could be achieved with alternative, potentially more biologically plausible methods.
Implementing and testing additional regularization strategies, such as L1 or L2 regularization combined with decorrelation constraints, could reveal whether place cell emergence depends specifically on orthonormal regularization. This would not only validate the robustness of the current approach but also explore alternative mechanisms that might be more feasible in biological systems. There are several propositions in the literature see e.g. [Schaeffer et al.](https://arxiv.org/abs/2311.02316)

### **Interpretabilty**
The interpretability claims based on clamping neurons to their maximum observed activity are primarily qualitative, which may limit reproducibility and objectivity.
Including quantitative metrics—such as cosine similarity or structural similarity index (SSIM) scores—between the decoded outputs and training images would provide a more rigorous assessment of interpretability. If visualizations remain subjective, a standardized metric would enhance reproducibility and add credibility to the interpretability claims.

### **Sensory modalities**
While the authors extend the model to the auditory domain, the tested signals are relatively simple (frequency alone), which limits understanding of the model’s adaptability to more complex auditory stimuli.
Testing with more complex auditory signals, such as temporal sequences or multi-dimensional auditory stimuli (e.g., varying intensity, background noise), would demonstrate if the model’s generalization holds for richer sensory environments.  Moreover, the study from Aronov focused on a goal-directed task.

### **RL**
The paper’s reinforcement learning (RL) results show that sparse autoencoders perform inconsistently, which the authors attribute to the brittleness of high-dimensional representations. However, the exact limitations and causes of brittleness are not thoroughly analyzed.

A more detailed analysis of the conditions under which sparse representations lead to brittle behavior could provide actionable insights for improving stability. Testing across a wider range of RL architectures (beyond DQN) or incorporating additional stability measures, such as other regularization techniques, dropout, or noise injection, could yield strategies to mitigate brittleness and enhance adaptability in RL tasks.

### Questions
In general, I liked the paper and applaud the innovative use of autoencoders and the clarity of the authors’ representations (pun intended). However, the biological relevance and the application of terms like ‘memory’ and ‘LSH’ feel overstated, making the results overly speculative. If the claims were substantially backed up with well-stated limitations or if terminology were revised to better match the model's demonstrated capabilities (e.g., ‘representation’ rather than ‘memory’) and the other stated weaknesses were addressed, I would be inclined to increase my score.

### Soundness
1

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper models the formation of hippocampal-like spatial representations by training sparse autoencoders on visual images from 3D environments. By applying an orthonormal activity regularization to encourage sparsity, the authors observe place cell-like behavior, where certain neurons respond selectively to specific regions in the input space. They argue that this sparsity enables the model to discretize continuous spaces into distinct memory-like representations. The study also extends to audio data, showing similar localized tuning, and evaluates how these representations impact reinforcement learning agents. The paper argues that such sparse coding mechanisms could illuminate how episodic memory and spatial representations emerge in the hippocampus.

### Strengths
1. Interesting use of the Animal-AI environment to study how phenomena commonly found in biological brains (place cells) might emerge.
2. Paper is clearly written (minus a few missing experimental details)

### Weaknesses
1. Confounding Visual and Spatial Representations: The study compares sparse autoencoder representations derived from static visual images to hippocampal place cells, effectively confounding visual encoding with spatial encoding. The hippocampus integrates both sensory cues (such as vision) along with self-motion cues (vestibular, proprioceptive, and motor information). The analogy to hippocampal place cells is therefore weak, as the model solely captures visual features, which may not directly translate to spatial encoding in biological brains.
2. "Sparsity" Enforced by Design, Not Necessarily Task-Relevant: The network's sparse representations are enforced through an orthonormal regularization term, which directly promotes decorrelation (not sparsity!) in the latent space. This enforced structure is interesting but does not convincingly link the sparsity to improved performance on the core task of reconstructing visual inputs. The paper lacks an analysis on whether the sparse representations improve or impair reconstruction accuracy, raising questions about whether these representations are useful for encoding spatial features or are merely artifacts of the regularization. I think the paper should compare both dense (no penalty), decorrelated (the orthonormal penalty used in the paper), as well as sparse (l1 penalty) networks; and show the performance on the task (reconstruction error) for each, ideally for a range of regularization strengths (values for ). How do the results in the paper depend on either (a) the choice of penalty used or (b) the regularization penalty strength ()?
3. Unclear Image Sampling and Environmental Modeling: The training dataset appears to consist of randomly sampled images from a virtual environment, but the paper provides little detail on the spatial structure of these samples. If the images are not derived from a structured, agent-driven navigation through space, it is unclear how the autoencoder captures spatial rather than purely visual information. This choice weakens claims that the model learns spatial representations akin to the hippocampal encoding of navigated environments. More rigorous sampling or agent-driven exploration would be necessary to substantiate claims of spatial learning. This would also ideally include some understanding of how the results depended on the choice of agent-driven exploration used.
4. Inadequate Evaluation of Representation Utility: The paper’s core claims focus on the emergence of hippocampal-like representations, but the practical utility of these representations remains untested in terms of spatial or episodic memory tasks. While place cell-like activity is demonstrated in the autoencoder’s neurons, it is unclear if these representations are suitable for real-world tasks that would benefit from spatial memory, such as pathfinding or goal-directed behavior. The RL experiments suggest the opposite (that the penalty is harmful), if I understand them correctly.

### Questions
* How were the images sampled when training the autoencoder?
* Could you be more specific about what exactly didn't work with using an explicit sparsity penalty (l1 norm)?
* How should I think about multi-modal signals related to navigation (e.g. vestibular, proprioceptive, and motor information)? Is the paper suggesting they are irrelevant for generating place cells?

### Soundness
1

### Presentation
3

### Contribution
2
