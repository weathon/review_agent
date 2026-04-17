# Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
This paper investigates how multilayer perceptrons (MLPs) can be represented in non-Euclidean spaces, with emphasis on the Poincaré model of hyperbolic geometry. We aim to capture the geometric evolution of their weighted topology and self-organization over time. Instead of restricting analysis to single checkpoints, we construct temporal parameter-graphs across $T$ snapshots of the optimization process. This reflects the view that neural networks encode information not only in their weights but also in the trajectory traced during training. Drawing on the idea that many complex networks admit embeddings in hidden metric spaces where distances correspond to connection likelihood, we present a geometric and temporal graph-based meta learning framework for obtaining dynamic hyperbolic representations of the underlying neural parameter graphs. Our model embeds temporal parameter-graphs in the Poincaré ball and learns from them while maintaining equivariance to within-snapshot neuron permutations and invariance to permutations of past snapshots. In doing so, it preserves functional equivalence across time and recovers the network’s latent geometry. Experiments on regression and classification tasks with trained MLPs show that hyperbolic temporal representations expose how structure emerges during training, offering intrinsic explanations of self-organisation in a given model training environment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes to incorporate temporal and geometric considerations into study neural networks as data points (e.g. in INRs). The authors propose a methodology the process INR trajectories using the Poincare ball model of hyperbolic geometry to use for metanetwork purposes (process NNs as datapoints e.g. for classification of INRs or predicting generalistion from the trained weights themselves)

I have given a 4 but would give a 3 if there was the option.

### Strengths
1. The paper seems correct and seems to have succeeded in adding temporal and geometric considerations into a framework for meta-networks.
2. There is a lot of background material involved in the paper. this is both a strength, as there is a lot of technical detail to get right, but also  a weakness (see below).

### Weaknesses
I'd say overall my criticisms of the paper stem from the fact that I found it hard to read. This is by no means my area, but even still I think the authors could have made several improvements to the presentation:

1. The motivation isn't very strong. The paper focuses on the technical problem (of adding temporal and geometric aspects to meta-networks), but not enough of the paper addresses why this is a problem to focus on.
2. Likewise, the empirical results for the methodology (the proposed GTH-GMN) of this paper are quite bare. There are 6 terms in the loss function, each of which are motivated in section 3: how is the empirical performance of GTH-GMN affected without each of the 6 terms? This is essential for motivating the method. Why do we need two optimisation steps? The paper is missing quite a lot of ablations imo. Relatedly, is it essential to have temporal *and* geometric aspects to the framework: what happens if you remove one (how does performance vary as the number of checkpoints changes)?
3. The contributions section (last paragraph of section 1) isn't very clear imo. The paragraph reads as if the contributions are methodological, are there any theoretical challenges that the authors needed to overcome in order to create the methodology (or was it mainly a case of applying and combining existing theory into a method). Likewise, I think a background section is missing between the related work and the research method sections in order to provide context for the reader of the most relevant background info (e.g. what is the most relevant methodological baseline and how does it work? Is this GMN, based on the name GTH-GMN?)
4. How much more expensive is it to have to track the full trajectory that say the final checkpoint?

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper models the training process of neural networks as a sequence of evolving graphs. Each of these graphs is embedded in hyperbolic space. The authors introduce a Hyperbolic Graph Meta-Network (GTH-GMN) that learns to encode the temporal evolution of these graphs using hyperbolic attention and recurrent updates. This geometric representation is intended to provide intrinsic explainability by revealing structure in training dynamics. Experiments on small-scale tasks show consistent improvements over prior Euclidean and static graph baselines, along with interpretable visualizations of how networks evolve during training.

### Strengths
The paper's main strength is in its main idea: to think about the training of a neural network not just as a sequence of parameter updates, but as a geometric process that unfolds in hyperbolic space. This view allows the authors to tie together many ideas from disparate fields. The resulting metrics for tracking training seem to capture meaningful structure in how networks evolve, and the visualizations provide some insights into the learning process.

### Weaknesses
I should preface my comments by noting that I am not an expert in the specific technical areas this paper draws on, which is reflected in my relatively low confidence score. I found the paper quite challenging to follow at times, largely due to its dense mathematical notation and the level of background knowledge it assumes from the reader. Despite the impressive mathematical machinery, I am not entirely convinced that the insights gained from this approach justify the conceptual and computational complexity it introduces.

### Questions
1. How sensitive are the results to the choice of curvature or manifold model? For example, if the same temporal graph encoder were trained in Euclidean or spherical space, would the geometric patterns and performance differences still hold?

2. Can the authors clarify what specific types of interpretability their method enables? Beyond visualizing trajectories, are there measurable insights (such as neuron redundancy, layer specialization, or early-stopping indicators) that can be extracted from the hyperbolic embeddings?

3. The experiments focus on relatively small MLPs and simple datasets. How might this framework scale to larger architectures, such as Transformers or CNNs, where parameter spaces and symmetries are more complex? Are there computational or conceptual challenges expected in extending the approach?

### Soundness
3

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
2

### Summary
This paper proposes a novel framework for studying MLP training dynamics through hyperbolic geometry. The authors construct temporal parameter graphs from checkpoints during training and embed them in the Poincaré ball using hyperbolic graph attention networks (HGAT) with kernel meta-evolution. The approach maintains permutation equivariance and ties edge weights to hyperbolic distances via power-law relationships, providing interpretable geometric representations of network self-organization during training.

### Strengths
- Novel geometric perspective: The application of hyperbolic geometry to temporal parameter graphs is original and well-motivated. The connection to network science findings (scale-free, hierarchical organization) provides strong theoretical justification.
- Temporal modeling: Unlike most prior meta-learning work that analyzes single checkpoints, this work explicitly models training trajectories.
- Comprehensive methodology: The signed weight regressor with magnitude-distance power law (Eq. 13) and conformal sign prediction (Eq. 15) shows careful design. The two-phase optimization (Euclidean + Riemannian) is well-executed.

### Weaknesses
- Limited performance gains: On CIFAR-10 generalization prediction, the method achieves τ=0.846±0.004, notably below NFN baselines (0.922-0.934). While the authors acknowledge this is "expected," it raises questions about practical utility. The gap suggests the geometric compression may discard accuracy-correlated information that end-to-end methods capture.
- Architectural limitations: The approach is restricted to MLPs. The authors mention this limitation but don't provide a clear path to extending to CNNs, Transformers, or other architectures that dominate modern deep learning. This severely limits practical applicability and impact.
- Computational cost not addressed: The paper doesn't discuss training time, memory requirements, or scalability. Hyperbolic operations, Riemannian optimization, and temporal processing likely add significant overhead.
Tables 5-6 mention "practical caps" but no runtime comparisons are provided.
- High variance in some experiments: Sinusoid task: 1.06±0.24 MSE vs GMN's 1.13±0.08 - the 3x higher variance is concerning and poorly explained. The authors attribute this to "known sensitivities in Riemannian optimization" but don't investigate mitigation strategies.

### Questions
- Can you provide runtime/memory comparisons with baselines? How does the approach scale with model size?
- What specific architecture modifications would be needed to extend beyond MLPs? Is this fundamentally intractable?
- Could you provide ablations showing the contribution of each loss term (Eq. 32)?

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
2

### Summary
This submission builds on prior work that seeks to develop methods for identifying the behavior of a random deep neural network (DNN), as well possibly train the DNN. The authors use temporal information (i.e., the optimization history) to learn models with greater accuracy. In addition, the authors embed the snapshots of the DNNs (across training) in a hyperbolic geometry. This can aid in performance and interpretability.

### Strengths
1. The work was well motivated. The utility of metanetworks was clearly presented and the idea that optimization trajectories could be leveraged for further improvement was good. 

2. The authors test on several different experimental set-ups, finding strong performance on in all cases. While they are not always best, their method is not so far off. 

3. The visualization of the hyperbolic embedding (Figure 2) was very interesting and points to possibly new ways to understand DNN training.

### Weaknesses
1. I think the biggest weakness of this work is that it is so dense. I found the last paragraph of the Related Works Section and Sec. 3.2-3.4 just full of acronyms, method names, and details. I'm sure that this is partially due to the fact that I am not so familiar with this area, but it felt very difficult to follow.

2. The result of having so much detail crammed into the Methods section was that then there was little room for discussion on the experiments. It's not necessary to have all the model and experimental details, but it wasn't always clear to me what was even really being tested in the 3 experiments in Sec. 4. What exactly was tested in "Classification of Images via INR traces"? Which image is being shown to a MLP? if so, this seems to not be so aligned with the motivation in the Introduction. Similarly, I was confused as to what a "sinusoid–MLP" is and how the developed method was being used. 

3. A minor point, but one way to study the optimization trajectory is to use tools from dynamical systems. These tools can be invariant to node ordering and can extract interpretable and comparable structure (e.g., https://proceedings.neurips.cc/paper_files/paper/2024/hash/2a07348a6a7b2c208ab5cb1ee0e78ab5-Abstract-Conference.html). How might the authors' work be extended/improved by including such dynamical characterization?

### Questions
1. What exactly was tested in "Classification of Images via INR traces"?

2. What is a "sinusoid–MLP" and what exactly was being tested in "Predicting Sine Wave Frequency"? 

3. How might the authors' method be improved with - instead of passing in many optimization graphs - the dynamics of the optimization were first filtered with dynamical systems approaches?

### Soundness
3

### Presentation
3

### Contribution
3
