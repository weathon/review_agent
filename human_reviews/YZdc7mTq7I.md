# GeoMind: A Geometric Neural Network of State Space Model for Understanding Brain Dynamics on Riemannian Manifold

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 6, 6

## Abstract
State space model (SSM) is a powerful tool in neuroscience field to characterize the dynamic nature of brain functions by elucidating the mechanism of how brain system transits between brain states and how underlying states give rise to the observed neural activities. Although tremendous efforts have been made to lend the power of deep learning and mathematical insight of SSM in various functional neuroimaging studies, current state-of-the-art methods lack a holistic view of brain state evolution as a self-organized dynamical system where each part of the brain is functionally inter-connected. Since the topological co-activation of functional fluctuations exhibits an intrinsic geometric pattern (symmetric and positive definite, or SPD) on the Riemannian manifold, the call for understanding how a selective set of functional connectivities in the brain supports diverse behavior and cognition emerges a new machine learning scenario of manifold-based SSM for large-scale functional neuroimages. To that end, we propose a geometric neural networks, coined *GeoMind*, designed to uncover evolving brain states by tracking the trajectory of functional dynamics on a high-dimensional Riemannian manifold of SPD matrices. Our *GeoMind* demonstrates promising results in identifying specific brain states based on task-based functional Magnetic Resonance Imaging (fMRI) data, as well as in diseases early diagnosis for Alzheimer's disease, Parkinson's disease and Autism. These results highlight the applicability of the proposed *GeoMind* in neuroscience research. Furthermore, to assess the generalization capabilities of our model, we applied it to the domain of human action recognition (HAR), achieving promising performance on three benchmark datasets (UTKinect, Florence and HDM05). This demonstrates the scalability and robustness of the proposed geometry deep model of SSM in capturing complex spatio-temporal dynamics across diverse fields.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces GeoMind, a geometric neural network designed to analyze brain state dynamics using state space models within a Riemannian manifold of symmetric positive definite (SPD) matrices. By leveraging the geometric properties of functional connectivities, GeoMind effectively uncovers evolving brain states from task-based fMRI data, showing promise for early diagnosis of neurological conditions such as Alzheimer’s, Parkinson’s, and Autism. Additionally, the model demonstrates versatility by achieving strong performance in human action recognition across multiple benchmark datasets.

### Strengths
The paper presents several strengths while also having notable limitations. One significant strength is the introduction of GeoMind, a geometric neural network that employs Riemannian manifolds and symmetric positive definite (SPD) matrices to analyze brain state dynamics. This innovative approach provides a unique framework for modeling complex neural activities, with promising results in identifying specific brain states from task-based fMRI data. Additionally, the model shows potential for early diagnosis of neurodegenerative diseases such as Alzheimer’s, Parkinson’s, and Autism, which underscores its relevance in clinical applications.

### Weaknesses
The paper has several limitations that affect its overall contribution. While it compares GeoMind with Mamba, it does not achieve state-of-the-art performance on key datasets such as ADNI, OASIS, ABIDE, and PPMI. Furthermore, the lack of comparison with other leading methods on these datasets makes it difficult to evaluate the model's effectiveness and its advancements over existing approaches. Additionally, the paper provides insufficient geometric explanation regarding the integration of Riemannian manifolds and symmetric positive definite (SPD) matrices, which may hinder understanding of the model's innovations. The rationale for choosing a geometric approach to analyze brain dynamics is also not clearly justified. Finally, the methodology for implementing the geometric components is inadequately discussed, which could impact reproducibility and practical application of the model.

### Questions
1. Insufficient Comparisons with State-of-the-Art Methods: The paper does not provide adequate comparisons with existing leading models on critical datasets such as ADNI, OASIS, ABIDE, and PPMI. This omission limits the ability to assess GeoMind's performance relative to established approaches and diminishes the impact of its contributions.

2. Lack of Detailed Geometric Explanation: The integration of Riemannian manifolds and symmetric positive definite (SPD) matrices within the GeoMind framework is not sufficiently explained. A clearer discussion of these geometric components and their relevance to the model would enhance understanding and strengthen the overall argument for its innovative approach.

3. Methodological Clarity and Reproducibility: The methodology for implementing the geometric aspects of the model is not adequately detailed, which could hinder reproducibility. Providing a more thorough description of the implementation and its effects on model performance would be beneficial for researchers looking to apply or build upon this work in practical settings.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents GeoMind, a novel geometry deep model to learn spatial-temporal dynamics of brain functional networks, by the incorporation of state space model (SSM) and Riemannian manifold learning, considering the intrinsic geometric pattern of brain functional networks. Convolutional operation is adopted to reformulate the SSM to improve the inference efficiency. The proposed model has been evaluated using both brain connectome data and human action data.

### Strengths
The Riemannian manifold learning is introduced to the SSM to capture the spatial-temporal dynamics of brain functional networks.

### Weaknesses
1. The convolution operation on SPD. It seems that certain inductive biases of convolution on images do not hold for convolution on SPD, e.g., the locality of convolution assumes local image patterns are significant (local pixel patterns are similar). However, as the order of brain ROIs in brain atlas/connectome is kind of arbitrary, so that features in SPD at nearby locations may represent significantly different brain connectivity measures. This means the feature representation learned is difficult to interpret from the viewpoint of neuroscience. On the other hand, this means the feature representation learned may vary a lot if the order of brain ROIs is shuffled.

2. Lack of comparison with dynamic-FC based models. As the proposed method basically works on dynamic FCs, it should be compared with dynamic-FC based models, such as STAGIN (Kim et al., 2021), DynDepNet (Campbell et al., 2022), NeuroGraph (Said et al., 2023).

### Questions
1. The convolution operation on SPD. It seems that certain inductive biases of convolution on images do not hold for convolution on SPD, e.g., the locality of convolution assumes local image patterns are significant (local pixel patterns are similar). However, as the order of brain ROIs in brain atlas/connectome is kind of arbitrary, so that features in SPD at nearby locations may represent significantly different brain connectivity measures. This means the feature representation learned is difficult to interpret from the viewpoint of neuroscience. On the other hand, this means the feature representation learned may vary a lot if the order of brain ROIs is shuffled.
2. More implementation details are needed, such as kernel size in convolution and size of sliding window. It is also helpful to see how they will affect the performance.
3. In addition to feature/model interpretation, does the GaA module also improve the prediction?
4. As the proposed method basically works on dynamic FCs, it should be compared with dynamic-FC based models, such as STAGIN (Kim et al., 2021), DynDepNet (Campbell et al., 2022), NeuroGraph (Said et al., 2023).
5. In addition to manifold learning, it is also common to apply the tangent space projection to brain connectome data before feeding it into ML/DL models. Comparison with it would also be helpful to demonstrate the strength of learning on SPD.
6. In model comparison, BOLD signals were used as graph embeddings for the graph-based models. As BOLD signals are not directly comparable across different individuals (especially for resting-state fMRI), using static FC as embeddings may improve these models' performance. 
7. In Fig.3, it looks like the region (orange circle) for HCP-WM is not in the dorsal attention region, please double check it.

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
This paper presents a novel geometric neural network, GeoMind, designed to capture brain dynamics by modeling functional connectivity on Riemannian manifolds. By leveraging state space models (SSM) with a focus on spatial and temporal brain dynamics, GeoMind aims to track evolving brain states effectively, addressing fMRI prediction problems. Additionally, the model's adaptability is validated through applications in human action recognition, showcasing its generalization capabilities.

### Strengths
1. The paper introduces an innovative method that combines manifold learning with SSM, effectively extending state space modeling beyond Euclidean spaces.
2. The figures and explanations are clear and well-organized, making complex ideas easier to understand. The writing is also clear and easy to follow.
3. The experiments are comprehensive, covering multiple datasets and a variety of comparison methods.

### Weaknesses
1. It seems that the paper does not clearly explain why it chose to use a Riemannian manifold for modeling brain dynamics. While this approach allows the model to capture certain geometric properties of brain data, it’s unclear why a simpler, more straightforward method wasn’t used instead. A stronger motivation, with an explanation of the specific benefits that the Riemannian manifold brings to neuroimaging tasks, would make the choice more convincing.
2. In my view, the main contribution of the paper lies in replacing the Euclidean algebra of conventional SSMs with Riemannian geometric algebra and applying this to fMRI images. However, the integration feels somewhat contrived, lacking a more natural or cohesive blend.

### Questions
1. Why was the Riemannian manifold chosen over simpler geometric spaces? Are there specific neuroimaging characteristics that necessitate this complex representation?
2. How does the choice of manifold-specific metrics like the Stein metric impact model performance compared to more straightforward metrics?

### Soundness
3

### Presentation
3

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
The paper introduces GeoMind, a geometric neural network model that uses state space models to track brain dynamics on a Riemannian manifold. By modeling brain connectivity with symmetric positive definite matrices, it preserves the geometric structure of brain activity. A geometric-adaptive attention mechanism helps identify critical brain connections relevant to diseases like Alzheimer's and Parkinson’s.

### Strengths
GeoMind outperforms most competing models in both brain activity and action recognition tasks. This high performance shows its effectiveness and robustness across different applications.

### Weaknesses
1. Although the model includes an attention mechanism, the manifold-based learning approach and complex neural architecture could still make it challenging to interpret results fully, especially for clinical use.

2. The use of Riemannian geometry and manifold learning may introduce additional computational overhead, which could be a drawback in large-scale clinical settings.

### Questions
1. How does the manifold-based approach improve insights into brain behavior? Does it reveal connections between brain structure and specific cognitive outcomes that simpler models miss?

2. fMRI data can be noisy due to patient movement or equipment issues. Has GeoMind been tested with noisy or incomplete data? Are there any built-in ways to handle this?

### Soundness
3

### Presentation
2

### Contribution
3
