# Riesz Neural Operator for Solving Partial Differential Equations

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Local non-stationarity is pivotal to solving partial differential equations (PDEs). However, in operator learning, the spatially local information inherent in the data is often overlooked. Even when explicitly modeled, it is usually collapsed into local superpositions within the model architecture, preventing full exploitation of local features in physical phenomena. To address this limitation, our paper proposes a novel Riesz Neural Operator (RNO) based on the spectral derivative representation. Since PDEs are fundamentally governed by local derivatives, RNO leverages the Riesz transform, a natural spectral representation of derivatives, to mix global spectral information with local directional variations. This approach allows the RNO to outperform existing operators in complex scenarios that require sensitivity to local detail. Our design bridges the gap between physical interpretability and local dynamics. Experimental results demonstrate that the RNO consistently achieves superior prediction accuracy and generalization performance compared to existing approaches across various benchmark PDE  problems and complex real-world datasets, presenting superior non-linear reconstruction capability in model analysis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper considers the problem of learning PDE solutions with neural operators, and proposes Riesz neural operator (RNO), which works on the spectral domain to improve spatial locality representation. The authors provide approximation error bounds with the use of first order Taylor expansion, and empirically show performance improvements over a couple baselines on several PDEs.

### Strengths
1. The idea of using Reisz transformation to capture local derivatives is interesting and seems beneficial in learning PDEs with highly changing solutions.
2. The paper is in general clearly written with definitions of the core Reisz transformation explained in the Appendix.

### Weaknesses
1. There seems to be a lack of comparison (both conceptually and experimentally) with an important baseline WNO [1], which uses wavelet transformation and addresses localization in both time/space and frequency.

2. A detailed training procedure is missing. A full algorithm in the appendix should help.

3. The ERA5 problem seems to serve as an important realistic application of the method, but misses key details such as training/testing data size, solution visualization etc. Providing a public link does not make the paper self consistent.

[1] Tripura, Tapas, and Souvik Chakraborty. "Wavelet neural operator: a neural operator for parametric partial differential equations." arXiv preprint arXiv:2205.02191 (2022).

### Questions
1. PSD is used without definition, is that power spectral density?
2. I believe the Reisz space in the paper means the Reisz field or Reisz representation in signal processing, instead of the vector lattice in functional analysis. The authors should clarify and be more accurate in wording.
3. How are the directional derivatives effectively learned during training?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The key idea of this paper is to embed the Riesz transform, a spectral tool representing directional derivatives, into the operator learning framework. By doing so, RNO blends global spectral features with local directional variations, addressing a noted gap in existing operator-learning methods. The paper demonstrates that RNO achieves state-of-the-art accuracy on a broad suite of PDE benchmarks and a real-world dataset.

### Strengths
1. The use of the Riesz transform to incorporate local spatial derivatives into a neural operator is a fresh and well-motivated idea.
2. The RNO delivers consistent and significant improvements across a wide range of PDE problems.
3. The paper is supported with rigorous theoretical justification. (Though the reviewer does not have enough time to justify its correctness)

### Weaknesses
1. I think some of the baselines are higher than those reported in the original paper. For instance, LNO on beam & Duffing. On some datasets, this would affect the ranking of different methods. The reviewer would like some explanation on this. 
2. The experiment on Navier-Stokes is not rigorous enough, given the limited baselines.
3. The writing of this paper could be hard to follow. For instance, some of the hypotheses proposed in the introduction are not fully justified later.
4. Baseline, specializing in capturing high-frequency features, is also missing.
5. The implementation code of the paper is also missing.

### Questions
See cons

### Soundness
2

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
3

### Summary
This paper proposes the Riesz Neural Operator (RNO), which leverages the Riesz transform to enhance neural operator learning for PDEs. By mapping inputs into a Riesz domain that encodes directional and derivative information, RNO integrates both local and global spectral components to model anisotropic and high-frequency behaviors more effectively, introducing direction-aware mixing for richer feature representations. The method is supported by comprehensive experimental validation across diverse PDE settings.

### Strengths
This paper presents a novel integration of the Riesz transform to encode directional and derivative information within a neural operator framework. The design is theoretically grounded and suggests the potential for constructing neural operators that effectively capture both local and global information. The proposed architecture is sound and successfully extends the representational capacity of neural operators, and the paper is overall well written. Furthermore, it provides thorough experimental validation across diverse datasets and baselines. In particular, the studies on NS equations with varying Reynolds numbers and the ablation analyses of the proposed architecture were especially effective in verifying the model’s components and advantages.

### Weaknesses
* The paper emphasizes that using the Riesz transform enables efficient representation of higher-order derivatives. However, employing the Riesz transform is primarily a design strategy that maps inputs into a space where higher-order derivatives are representable—it does not inherently guarantee that the model will learn or reproduce higher-order derivatives more accurately. To substantiate this claim, it would be important to experimentally evaluate how well the model reconstructs derivatives of various orders (e.g., $L^2$ and $L^\infty$ errors of $\nabla u$, $\nabla^2 u$, $\nabla^3 u$).

* The modular structure and the use of complex-valued or monogenic representations may introduce nontrivial computational and memory overhead. A fair comparison of training time, inference time, peak memory allocation during training, and parameter efficiency  against baselines is needed.

* The proposed model may introduce potential instability in practical scenarios, as Riesz transforms can amplify high-frequency noise and encounter numerical issues near singularities. Therefore, experimental validation of stability under such conditions (i.e., non-smooth solutions and noisy data) is necessary. This is especially important because real-world data often contain noise, and many practical PDEs exhibit singularities; thus, these experiments are crucial to assess the practical robustness of the proposed method.

* Tables 3 and 4 could be better positioned; their current placement does not align well with the corresponding discussion in the text.

### Questions
Please address the noted weaknesses, particularly by providing experimental evaluations of computational complexity and memory usage compared to baselines. Additionally, testing the model’s stability and performance on noisy data and singular solutions would greatly strengthen the validation of the proposed method’s practical robustness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the Riesz Neural Operator, a novel architecture designed to enhance the ability of neural operators to model complex dynamic variations. The key insight is to overcome the limitation of current methods that often overlook or collapse local non-stationarity and spatial locality. The authors demonstrate that RNO achieves better prediction accuracy and generalization performance on various PDE benchmarks, including Navier-Stokes and the real-world ERA5 climate dataset.

### Strengths
- The central idea of explicitly embedding the Riesz transform to extract direction-aware spectral derivatives.

- Results on 2D NS and the ERA5 climate dataset shows improvement on high-frequency modeling.

### Weaknesses
- Lack of complexity analysis: This new neural operator architecture needs be accompanied by an analysis of its computational overhead, especially given that Riesz transform is not a popular thing in terms of both a analysis tool and practical implementations. The paper didn't provide comparison of training time, memory consumption, or parameter count against popular baselines, which are known to have very efficient implementations. If RNO is implemented in terms of FFT, it becomes crucial to ablate the model design (see below).

- Novelty of local features. The local-global split has already been applied to standard FNOs using e.g. local convolutions. Given the relation between Riesz transform and FFT, it is not clear whether the improvement comes from the representation itself, or a carefully tuned model architecture/hyper-parameter.

- Motivation of derivatives. The authors motivate their architecture by emphasizing the importance of derivatives in solving PDEs. While this is a generally safe statement, the characteristics of different PDE classes makes it less convincing. There are many (spatially) localized PDEs, but there are also PDEs that are global, or even scale-local. Motivating an architecture based on a vague view of all PDEs does not convince me. Nevertheless, I acknowledge that spectral bias is a very significant deficiency of neural operators. I suggest the authors to emphasize on how RNO properly models the full spectrum, why different rank-reduction method are useful for global vs local dynamics.

### Questions
- What is the complexity of your Riesz transform implementation.

- Can you differentiate your method from FNO + CNN?

### Soundness
2

### Presentation
3

### Contribution
3
