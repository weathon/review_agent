# DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Functional data, representing curves or trajectories, are ubiquitous in fields like biomedicine and motion analysis. A fundamental challenge is phase variability—temporal misalignments that obscure underlying patterns and degrade model performance. Current methods often address registration (alignment) and classification as separate, sequential tasks. This paper introduces DeepFRC, an end-to-end deep learning framework that jointly learns diffeomorphic warping functions and a classifier within a unified architecture. DeepFRC combines a neural deformation operator for elastic alignment, a spectral representation using Fourier basis for smooth functional embedding, and a class-aware contrastive loss that promotes both intra-class coherence and inter-class separation. We provide the first theoretical guarantees for such a joint model, proving its ability to approximate optimal warpings and establishing a data-dependent generalization bound that formally links registration fidelity to classification performance. Extensive experiments on synthetic and real-world datasets demonstrate that DeepFRC consistently outperforms state-of-the-art methods in both alignment quality and classification accuracy, while ablation studies validate the synergy of its components. DeepFRC also shows notable robustness to noise, missing data, and varying dataset scales. Code is available at https://github.com/Drivergo-93589/DeepFRC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel end-to-end deep learning model DeepFRC that integrates deformable registration with classification, tasks that are so far adressed usually separately. This is achieved utilizing a diffeomorphic neural registration operation, Fourier spectal representation for smooth functional encoding and a class-aware contrastive objective. Experiments on both synthetic and real datasets demonstrate consistently improved alignment together with competitive or superior classification performance.

### Strengths
1. Offers and end-to-end method to jointly learn functional registration and classification in a unified architecture rather than sequentially.
2. Theoretical guarantees are provided that limk registration fidelity to classification generalization.
3. Improved performance across real world and synthetic dataset and comprehensive ablation for each architectural component.
4. Robustness is shown against noise and missing data.
5. The language of the paper is clear. 
6. Git link to code is available.

### Weaknesses
**Weaknesses and Questions**

1. Since TTN is also addressing jointly the registration and classification tasks, can the authors elaborate and highlight a bit more the architectural and conceptual differences between these methods?
2. How sensitive is the model to mis-specified diffeomorphic constraints if real warpings violate the diffeomorphic guarantees?
3. In line 082: the paper claims that the paper of Tang et al. is heavily reliant on assumptions,…
Can the authors clarify which assumptions those are and whether they avoid making these assumptions in this work?
4. In the experiments the paper claims that simulated data provide insights regarding registration, classification and reconstruction. I would like to understand why is it important to draw conclusions on the reconstruction along with the other 2 tasks discussed thoroughly in the paper.
5. In table 1 we sometimes see that several methods demonstrate high acc and F1 score while their registration performance is not optimal. Is there any intuition behind this? Does it mean that the SrvfRegNet was not tuned properly? Does this not make those methods robust against misregistration which can be an interesting and useful property? What is the intuition behind SrvfRegNet always not being able to recover the registration as accurately as the proposed method?
6. Another question regarding the results is whether there is any intuition why in the case of Symbol 3 the SrvfRegNet + TSLANet is delivering higher classification accuracy compared to Symbol2. Is there any correlation to the number of classes?
7. I would like to request some addition of limitations of the current method and of why one should not consider to addressing classification and registration jointly.

### Questions
Please see above.

If the authors address some of the discussion points above I am willing to increase my score in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DeepFRC, an end-to-end deep learning framework that jointly addresses functional data registration (alignment) and classification. Its main contribution is a unified architecture that integrates a neural deformation operator for diffeomorphic warping, a spectral representation for smooth functional encoding, and a class-aware contrastive loss. This approach eliminates the need for separate preprocessing steps and allows the alignment and classification tasks to mutually enhance each other. The work is further supported by theoretical guarantees on its approximation capability and generalization error, and is empirically validated to outperform state-of-the-art methods on both synthetic and real-world datasets.

### Strengths
The paper demonstrates good originality by proposing the first end-to-end unified framework for joint functional data learning. 
The research quality is solid, well-supported by comprehensive theoretical analysis and systematic experimental validation.

### Weaknesses
1. It is not clear about the network architecture choices.
2. Current computational complexity analysis is not sufficient.
3. Lack of a detailed description of the datasets, which helps to understand the possible applications of the proposed method.

### Questions
1. The paper uses a 1D CNN to parameterize the neural deformation operator (the registration module) but does not explain why Transformer-based models were not chosen. Can you explain the core reasons for this selection?
2. It is recommended to supplement the experimental results with statistical significance analysis to prove that the performance improvements are statistically significant.
3. The current computational efficiency comparison in the paper only mentions runtime. Would supplementing it with the total training time, model parameter count, and FLOPs provide a more comprehensive demonstration of DeepFRC's efficiency advantages?
4. Can you further clarify the sampling methods of the four real-world experimental datasets? Additionally, could you explain how the proposed method handles highly irregularly sampled functional data?
5. I would suggest adding more application analysis to demonstrate the effects of the proposed method.

### Soundness
2

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
The paper proposes DeepFRC, an end-to-end framework for function/sequence alignment and representation-driven registration/classification. The method builds a deformation operator for time warping, a spectral/spectral-coefficient representation of aligned functions, and a classifier that uses contrastive geometric alignment objectives.

### Strengths
The authors provide theoretical guarantees, proving that the model can approximate optimal warping functions and establishing a data-dependent generalization bound that links registration fidelity to classification performance.

### Weaknesses
1.	The authors claim “but rarely addressing both simultaneously”, however, there are several works addressing registration and classification simultaneously, for example,
[1] Zhang, Y. and Telesca, D., 2014. Joint clustering and registration of functional data. arXiv preprint arXiv:1403.7134.
2.	Why transform the latent features into a monotone cumulative sum can guarantee diffeomorphism?
3.	Novelty is limited, for registration, only introduce neural deformation operator for alignment
4.	The claim that DeepFRC is "efficient" is poorly supported. The authors only report inference time, which is already longer than other learning-based methods. Crucially, they omit training time, computational complexity analysis, and a comparison of model parameters (number of weights). 
5.	The theoretical results rely on assumptions (e.g., Lipschitz continuity, compactness) that should be discussed more critically.
6.	The comparison methods are from 2021, the authors should compare most recent methods, like the methods published in 2025 and 2024.

### Questions
1. add more comparisons to current methods.
2. explain why transform the latent features into a monotone cumulative sum can guarantee diffeomorphism?

### Soundness
3

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
This paper presents DeepFRC, a deep learning framework that jointly performs functional registration and classification for functional or trajectory data. It integrates a neural deformation operator with 1D CNN to learn diffeomorphic time-warping functions for temporal alignment, a spectral representation module based on Fourier bases for smooth function embedding, and a classifier trained with a contrastive–geometric loss to align within classes and separate between classes.

Theoretical analysis shows that DeepFRC can approximate optimal warpings and achieves bounded generalization error. Experiments on both synthetic and real-world datasets (Wave, Yoga, Symbol, MotionSense) demonstrate improved alignment and classification accuracy over baselines such as TTN and SrvfRegNet.

### Strengths
1. The paper presents a well-motivated problem by targeting the joint challenge of phase variability and classification in functional data analysis (FDA).
2. Empirical results are consistent: DeepFRC outperforms alternatives across several datasets, enhancing both alignment and classification and confirming the effectiveness of joint optimization.
3. Theoretical discussions and included proofs, effectively situate the model within the mathematical landscape of FDA.

### Weaknesses
1. The neural components (1D CNN, MLP, Fourier basis) are standard. The main contribution is integrating known elements instead of developing a new architecture or loss function.
2. The baseline models are too few and outdated; it would be better to include more recent baseline models for comparison.

### Questions
1. Theorems 3.1 and 3.3 rely on many assumptions such as smoothness, bound and compactness. How realistic are these for real-world datasets with irregular sampling or noise?
2. How would the baseline models perform when scaling to the large datasets (time complexity, ATV, ACC, etc.)

### Soundness
3

### Presentation
3

### Contribution
3
