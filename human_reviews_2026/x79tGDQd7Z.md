# Scaling Quantum Machine Learning without Tricks: High-Resolution and Diverse Image Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Quantum generative modeling is a rapidly evolving discipline at the intersection of quantum computing and machine learning. 
Contemporary quantum machine learning is generally limited to toy examples or heavily restricted datasets with few elements. This is not only due to the current limitations of available quantum hardware but also due to the absence of inductive biases arising from application-agnostic designs.
Current quantum solutions must resort to _tricks_ to scale down high-resolution images, such as relying heavily on dimensionality reduction or utilizing multiple quantum models for low-resolution image patches.
Building on recent developments in classical image loading to quantum computers, we circumvent these limitations and train quantum Wasserstein GANs on the established classical MNIST and Fashion-MNIST datasets. Using the complete datasets, our system generates full-resolution images across all ten classes and establishes a new state-of-the-art performance with a single end-to-end quantum generator without tricks.
As a proof-of-principle, we also demonstrate that our approach can be extended to color images, exemplified on the Street View House Numbers dataset.
We analyze how the choice of variational circuit architecture introduces inductive biases, which crucially unlock this performance. Furthermore, enhanced noise input techniques enable highly diverse image generation while maintaining quality. Finally, we show promising results even under quantum shot noise conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents QGAN, a single, end-to-end quantum circuit based generative framework capable of generating high-quality, diverse images. It moves beyond previous approaches that relied on classical dimensionality reduction or patch-based generation, which limited the quantum model's role and overall performance.

### Strengths
1) This work introduces that a single quantum generator can generate diverse images without relying on patches or dimensionality reduction. 

2) It proposes a structured ansatz for image generation, moving beyond generic, task-agnostic circuits.

3) It introduces a learnable, multi-modal latent space for the generator leveraging reparametrization tricks, enabling generation of diverse, high-quality images.

3) Extensive experiments are performed mimicing realistic conditions, e.s.p. the rebustness study.

### Weaknesses
1. Clarity of the work needs to be improved, e.s.p. regarding the model architecture. The current explanation of the model make it hard to intepret and understand. A more direct visualization of the model is helpful.

2. Lack of theoretical analysis (or advantage)  and comparisons with the existing methods. Without theoretical analysis, it is hard to understand what quantum properties can help the generative process.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Previous quantum machine learning works (especially quantum generative models, as considered in this work) are often with limited size due to difficulty of data uploading and training. As an example, for the MNIST datasets, many previous works first reduce the dimension of images. By introducing inductive bias created by an application-specific quantum circuit design, this work is able to train a quantum GAN without reducing any dimension.

### Strengths
Scaling is one of the most important issue regarding to the quantum machine learning models. Though heuristic, it is good to see that the work makes some positive progress towards this point. The overmoding, i.e., more input noise modes, inspired by the classical over-parameterization, seems to be interesting. Shown as Fig.6 in the paper, the numerical result agrees with the authors' argument that increasing mode can increase the model performance. The size for QGAN models with 64 layers and 40 noise modes for about 50 000 generator updates is also relatively big compared with other QNN models.

### Weaknesses
The weakness of this paper is that it does have any theoretical analysis to support their claim, and also the method is model-specific, and thus it is not that clear how good this method will be.

### Questions
1. I would like the authors to clarify more about the model-specific trick. What properties or structure are essentially required for the input data set? 
2. For the overmoding, is it possible to provide more theoretical aspect and seek any potential for a quantum advantage?
3. Hope to see a bit more discussion about the scalability based on the numerical result.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel approach to quantum generative modeling by training a single end-to-end quantum Wasserstein GAN (QGAN) on full-resolution datasets such as MNIST, Fashion-MNIST, and SVHN. Unlike prior methods that rely on dimensionality reduction or patch-based generation, the authors leverage application-specific quantum circuit designs and enhanced multimodal noise input to achieve high-quality and diverse image generation. The work demonstrates scalability, improved fidelity, and robustness under quantum shot noise, setting a new benchmark for quantum image synthesis.

### Strengths
1. One key strength of this paper is the extensive experimental validation across multiple full-resolution datasets, including MNIST, Fashion-MNIST, and SVHN.

2. The authors demonstrate strong empirical performance, showcasing the scalability and robustness of their quantum generative model.

3. The work also stands out for eliminating classical post-processing and relying solely on quantum-native design, which enhances its novelty and practical relevance.

### Weaknesses
The paper lacks a clear and detailed structural presentation of the proposed framework. As a result, the overall methodology remains vague, making it difficult for readers to grasp the model’s design and workflow.

The core approach builds on Quantum Wasserstein GANs, which were introduced by Chakrabarti et al. in 2019 (NeurIPS 32). However, the current work does not offer substantial architectural innovation beyond that baseline, raising concerns about novelty.

Although the authors present extensive experiments, many critical details are missing or ambiguously described. For instance, the computational environment, model configuration, and algorithmic components are not clearly specified. This lack of transparency undermines reproducibility and casts doubt on the robustness of the reported results.

Overall, the paper suffers from weak technical exposition. The narrative is often unclear, and key concepts are buried in verbose descriptions, which makes the model harder to understand rather than illuminating its strengths.

### Questions
Could you provide a more detailed schematic or modular breakdown of your proposed QGAN framework? The current description lacks structural clarity, making it difficult to understand how components interact and how the model is trained end-to-end.

Given that Quantum Wasserstein GANs were introduced by Chakrabarti et al. (NeurIPS 2019), what specific architectural or algorithmic innovations distinguish your model from theirs? A clearer comparison would help assess the novelty of your contribution.

Can you elaborate on the experimental setup, including quantum simulator or hardware specifications, number of shots, and noise models used? These details are essential for reproducibility and for evaluating the robustness of your results.

The paper mentions application-specific circuit designs, but does not provide concrete circuit layouts or gate compositions. Could you include representative circuit diagrams or pseudocode to clarify the model’s internal structure?

You mention multimodal noise input as a key design choice. Have you conducted any ablation studies to isolate its impact on generation quality? This would help validate its contribution.

What quantitative metrics were used to assess image quality and diversity? Including standard benchmarks like FID or IS would strengthen your claims and allow comparison with classical GANs.

How feasible is your approach on current noisy intermediate-scale quantum (NISQ) hardware? Please provide resource estimates (e.g., qubit count, circuit depth) and discuss any hardware-specific constraints.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a quantum Wasserstein GAN (QWGAN) capable of generating full-resolution MNIST, Fashion-MNIST, and SVHN images using a task-specific variational quantum circuit aligned with FRQI/MCRQI encodings. The key contributions include (1) an architecture with inductive bias derived from hierarchical pixel indexing and FRQI structure, (2) a multimodal, learnable noise injection mechanism intended to improve sample diversity, and (3) experiments showing high-quality image generation in numerical simulation. The paper includes ablations on ansatz design, noise modeling, overmoding, and finite-shot effects.

### Strengths
- The paper identifies and addresses a primary challenge in QML: the inability of most quantum models to handle high-dimensional classical data directly.
- The ablation between task-agnostic vs. FRQI-aligned circuits is convincing and highlights the importance of structured ansatz in QML. The ablation study in Figure 4 successfully demonstrates that a circuit ansatz specifically designed for the FRQI encoding produces higher-quality images than a generic ansatz, though this result is somewhat expected.

### Weaknesses
- All results are purely classical simulations, making claims of scalability speculative. The proposed models use 11-13 qubits and 64-layer deep circuits with >10k parameters, which are far beyond the capabilities of NISQ hardware. No evidence is provided that these circuits can be trained or executed on real devices or even on moderate-scale noisy simulators. No evidence is provided that these circuits are not suffering from the barren plateau phenomenon.

- The entire premise of the paper's "scaling" advantage is built on the FRQI encoding, which compresses an $N$-pixel image into $O(\log N)$ qubits. While the authors celebrate this exponential compression in storage, they fail to address the well-known and fatal flaw of this method: recovering the $N$-pixel image requires $O(N)$ measurements. This exponential cost in runtime (measurement) completely negates any benefit from the qubit compression. A method that scales exponentially with the problem size cannot, by definition, be considered "scalable." The authors relegate this fundamental issue to a brief discussion of "future work" (e.g., compressed sensing, shadow tomography), which is unacceptable. This is not a minor detail to be "left for future work"; it is a core flaw that invalidates the paper's central claim of having achieved a scalable solution.

- The paper compares only against previous quantum baselines. There is no evaluation or any metric that would place quantum performance relative to classical GANs of similar size. The absence of such baselines makes it difficult to judge the relevance of the results for the broader machine learning community. No evidence is provided that the quantum model would outperform the classical models.

### Questions
- Given the hybrid setup, isn't the classical CNN discriminator doing almost all of the intelligent work? How can you be certain that the quantum generator is anything more than a complex, over-parameterized noise source that the powerful classical discriminator simply learns to interpret? How does performance compare quantitatively to classical GANs with a similar number of parameters?

- The "future work" proposals for the measurement problem are purely speculative. Can you provide any concrete evidence or preliminary data that methods like compressed sensing would actually work? For many natural images, the sparsity constant $k$ is still proportional to $N$, meaning a sub-sampling approach would still be $O(N)$ and offer no real savings.

- The literature review is somehow inadequate, e.g., the very first paper that proposed quantum WGANs (https://arxiv.org/abs/1911.00111) is not even mentioned here.

### Soundness
2

### Presentation
3

### Contribution
2
