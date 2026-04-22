# Pick Your Channel: Ultra-Sparse Readouts for Recovering Functional Cell Types

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Clustering neurons into distinct functional cell types is a prominent approach to understand how the brain integrates information about the external world. In recent years, digitial-twins of the visual system based on deep neural networks (DNNs) have become the de facto standard for predicting neuronal responses to arbitrary stimuli. Such DNNs are designed with a common core that learns a representation of the visual input that is shared across neurons, and a neuron-specific readout that linearly combines the core outputs to predict single neuron responses. Here, we propose a novel way to learn an ultra-sparse readout that, instead of linearly combining the shared core features, learns to pick a single channel for each neuron. For retinal ganglion cells, we find that, unlike the previous unconstrained models, this ultra-sparse readout triggers the neural predictive model to innately learn functional cell types with minimal loss in predictive performance. Furthermore, we show that state-of-the-art adaptive regularization models are unable to find such single channels, and that applying strong regularization to encourage sparse channels not only deteriorates performance but also results in response shrinkage. When applied to primary visual cortex neurons, our model exhibits a larger drop in performance compared to the unconstrained model, perhaps indicating a more continuous organization of neuronal function.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces ultra-sparse readouts for neural encoding models to uncover functional cell types in neural populations by enforcing sparsity in the mapping from shared visual representations to individual neuron responses. The ultra-sparse readouts combines 3 strategies: Gumbel-Softmax readout, 3D Grid readout, and REINFORCE readout. Using mouse retinal ganglion cell (RGC) and primary visual cortex (V1) datasets, the authors show that the Gumbel-Softmax readout has almost the same predictive performance as unconstrained models while naturally grouping neurons into consistent functional types. In the retina, the model identifies canonical cell classes with high internal consistency. However, performance drops in V1, which suggests that cortical neurons have a more continuous functional organization rather than discrete clustering.

### Strengths
1. This paper presents a method to improve the interpretability of vision-based neural encoding models. The approach is mathematically grounded and shows good empirical performance based on the reported results.

2. The analyses are detailed and comprehensive, and support the authors’ claims.

3. Improving interpretability in neural encoding models is a meaningful contribution to the field.

### Weaknesses
1. Although the proposed method improves interpretability, there is a notable performance drop in V1. This raises concerns about its general applicability. Ultimately, we want models that not only provide interpretability but also accurately predict neural responses to visual stimuli, as predictive performance is a prerequisite for studying how visual information is represented in the retina and the brain.

2. Although the method successfully identifies functionally consistent neuron groups, it does not directly uncover the underlying receptive field computations associated with each channel. Developing methods to this end would improve the scientific value of these interpretability methods.

### Questions
1. The paper uses a fixed CNN core as a proof of concept, but does the proposed method remain model-agnostic and compatible with other modern architectures, such as Vision Transformers, which are likely to be used in future large-scale studies?

2. The single-trial correlation metrics in Fig. 2 are quite low. Could the authors clarify the reason for this? For example, is it due to the use of naturalistic video stimuli rather than repeated trials? Also, why was correlation chosen as the primary evaluation metric, given that it only captures linear relationships and does not account for nonlinear dependencies or variance structure in the data?

### Soundness
3

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
2

### Summary
The paper presents a novel approach to learning ultra-sparse readouts in deep neural network models for predicting neuronal responses to arbitrary visual stimuli. This sparsity constraint minimally degrades predictive performance relative to unconstrained models, while inherently revealing functional cell types consistent with known retinal ganglion cell categories in mice. When applied to V1 neurons, the model performs worse, supporting the hypothesis of a more continuous and less discrete functional organization in V1.

### Strengths
The paper is scientifically motivated and proposes a conceptually novel method that bridges interpretability and performance in neural response modeling.

### Weaknesses
A key missing element is a clear justification of why identifying functional cell types from readout channels is important. The paper would benefit from clarifying whether the discovered functional cell types differ meaningfully from those obtained by clustering readout vectors in unconstrained models. It remains ambiguous whether the primary contribution lies in biological interpretability, computational efficiency, or predictive insight. A more explicit articulation of this contribution—and comparisons to simpler clustering-based baselines—would strengthen the paper.

### Questions
- In Figure 5, additional details are needed: What do the solid lines and shaded regions represent? Are these population averages and confidence intervals?

- Are the responses of known functional cell types illustrated in these plots?

- Did the method identify any novel or previously unreported cell types?

- As noted above, a more direct comparison between the sparse readout–based classification and clustering results from unconstrained models would clarify the added value of sparsity.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how adding sparse constraints to neural networks trained on neural data can improve the interpretability and selectivity of modeled neural responses. Specifically, it explores whether such constraints allow the network to maintain comparable performance to unconstrained models while aligning more closely with biological upstream channels. The authors analyze model performance across different visual processing stages (retina and V1), attempting to explain the differences in prediction accuracy through response continuity and channel specialization. The overall goal is to assess whether the model can learn neuron-like selectivity properties found in biological visual systems.

### Strengths
* The idea of incorporating sparse regularization to simulate selective neural representations provides a biologically inspired direction for neural modeling.

* The model retains comparable predictive performance despite the addition of sparse constraints, suggesting robustness and flexibility.

* Effort to connect neural network representations with retinal and cortical processing – The attempt to interpret model behavior in relation to retinal and V1 responses adds neuroscientific relevance.

### Weaknesses
* While the sparse constraint helps maintain similar performance, the work doesn’t convincingly demonstrate why this matters or what new insights it provides beyond showing robustness.

* The link between learned representations and specific neuronal functions (e.g., orientation selectivity, spatial frequency tuning) is vague, reducing the biological interpretability of results.

* The drop in predictive accuracy for V1 neurons indicates that the model may fail to capture hierarchical processing or contextual integration occurring in cortex.

* The authors attribute V1–retina performance differences to “response continuity,” but this reasoning feels weak and insufficiently validated.

* Both retina and V1 were modeled using the same CNN architecture, but the paper doesn’t discuss whether this choice limits representational specialization.

* The experiments do not comprehensively test how different constraints or model components affect performance and feature representation.

### Questions
I have serveral questions listed as belows:

Does the sparse constraint truly promote the emergence of biologically meaningful features (e.g., orientation, direction selectivity)? Can these be visualized or quantified?

Why was the same CNN architecture used for both retina and V1? Could architectural differences (e.g., receptive field size, nonlinearity) better reflect biological distinctions?

How sensitive are the results to the degree or form of sparsity applied?

Could the authors provide more evidence supporting their “response continuity” explanation for the retina–V1 performance gap?

Would more targeted ablations or neuron-type-specific analyses reveal whether the model captures functionally distinct neural subpopulations?

### Soundness
3

### Presentation
3

### Contribution
2
