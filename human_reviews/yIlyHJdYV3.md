# A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Translations

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 8

## Abstract
In subsurface imaging, learning the mapping from velocity maps to seismic waveforms (forward problem) and waveforms to velocity (inverse problem) is important for several applications. While traditional techniques for solving forward and inverse problems are computationally prohibitive, there is a growing interest to leverage recent advances in deep learning to learn the mapping between velocity maps and seismic waveform images directly from data. Despite the variety of architectures explored in previous works, several open questions still remain unanswered such as the effect of latent space sizes, the importance of manifold learning, the complexity of translation models, and the value of jointly solving forward and inverse problems. We propose a unified framework to systematically characterize prior research in this area termed the Generalized Forward-Inverse (GFI) framework, building on the assumption of manifolds and latent space translations. We show that GFI encompasses previous works in deep learning for subsurface imaging, which can be viewed as specific instantiations of GFI. We also propose two new model architectures within the framework of GFI: Latent U-Net and Invertible X-Net, leveraging the power of U-Nets for domain translation and the ability of IU-Nets to simultaneously learn forward and inverse translations, respectively. We show that our proposed models achieve state-of-the-art (SOTA) performance for forward and inverse problems on a wide range of synthetic datasets, and also investigate their zero-shot effectiveness on two real-world-like datasets. The code is available at https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work summarizes recent advances in deep learning-based mapping between velocity maps and seismic waveforms, introducing a unified framework (GFI) to systematically characterize prior research. Using GFI, key factors affecting mapping performance are investigated, alongside the proposal of two novel model architectures. Comprehensive experiments reveal the impact of these factors, with the proposed models achieving state-of-the-art (SOTA) performance in both forward and inverse problems.

### Strengths
1. This work is the first to investigate key factors influencing mapping performance.
2. Extensive experiments were conducted to conclude the influence of these factors.
3. Two novel networks are introduced for mapping between velocity maps and seismic waveforms, achieving SOTA performance.

### Weaknesses
1. This paper resembles a survey and might be more suitable for journal submission.
2. A significant portion of the content is placed in the appendices, making it less reader-friendly.
3. Aside from extensive experiments on factor influences, the work offers limited novelty and lacks theoretical analysis.
4. Additional factors should be investigated, such as the robustness of data noise across different network architectures.

### Questions
Invertible Neural Networks (INNs) often have limited flexibility in adjusting the dimensions of input and output, the number of layers, filters, and convolutional kernel sizes. These constraints can impede performance for complex tasks.

When mapping between velocity maps and seismic waveforms—data from two distinct domains (one spatial and the other spatio-temporal), why Invertible X-Net could demonstrate superior performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new training scheme combined with two new model architectures for the problem of learning the forward and inverse map in deep learning for subsurface imaging. 
The training scheme consists of jointly learning the encoders and decoders for the two domains (velocity maps and seismic waveforms) with the translation network transforming the latent space representations from one domain to another.
The two architectures differ in the characteristics of the translation network with one consisting of two separate unidirectional U-nets (latent U-net) and the other of a single bidirectional invertible network (invertible X-net).
Both architectures achieve a what seems to be significant improvement in performance over existing works.
Finally the paper formulates a unifying framework in which existing approaches can be categorized and distinguished.

### Strengths
1.	(Quality) High quality of experimental evaluation: Experiments are extensive over a range of datasets including an extensive evaluation of out-of-distribution performance and qualitative results are provided in addition to quantitative results. Hence, the conclusion of generally improved performance seems sound. 
2.	(Significance) The improvements in performance seem to be significant, however interpretability of how significant could be improved (see weaknesses point 5)
3.	(Significance) The finding that jointly solving forward and inverse problems is helpful for the forward problem but not for the inverse problem is interesting. 
4.	(Originality) The proposed unifying framework helps to quickly understand the different existing approaches and will be helpful for future works to expand the framework or propose improvements within it.
5.	(Clarity) The paper is overall well written and easy to follow also for researches not actively involved in the field of subsurface imaging.

### Weaknesses
Major concerns

1. (Clarity) The experiment 5.1.3 addresses the question ‘should we train latent spaces solely for reconstruction?’, which I assume corresponds to the question 3) formulated in line 083 ‘what is the role of manifold learning?’.  
1.1 For increased clarity I’d suggest not changing the phrasing of the questions.  
1.2 The setup is unclear to me. What are training fractions? A fraction of the training set or a fraction of the training time? And why is it instructive to look at 5% and 10% training fraction?  
1.3 So, is the conclusion form the experiment that learning manifolds is not important? It is difficult to grasp the conclusion if the wording is not consistent. Is “influence the training of the latent spaces based on the translation objectives” identical to “not learning manifolds”?  
1.4 To better answer the question I think the following baseline is missing that combines the two alternatives (reconstruct then translate and directly translate) into directly learning the translation networks while having at the same time reconstruction losses (basically what is described in lines 243-245). Seeing how this combination performs would help answering the question if manifold learning helps.  
2. (Originality) The idea of the Latent U-net seems to be very similar to the InversionNet following the comparison in Table 1.  
2.1 The Modelling Mode is the same because the forward and inverse part of the Latent U-net are completely disjoint and hence correspond to an InversionNet and a ForwardNet (as long as no manifold learning is performed).  
2.2 While the latent U-net allows manifold learning, the experimental results pertain to the case without manifold learning same as the InversionNet.  
2.3 Latent space  translation is said to be identity for the InversionNet and U-net for the Latent U-net, but couldn’t we also just claim some middle layers of the InversionNet to perform latent space translation?  
2.4 The size of the latent space is low in both cases.  
--> So to my understanding the difference between the two boils solely down to the architectural design and size of the network but not to any conceptual differences. If that is the case, it should be stated more directly in the paper and the exact architectural differences should be explained in more detail. 
3. (Clarity) From Section 5.1.4 it is not entirely clear what the answer to the question raised in the paragraph (Is it useful to jointly solve forward and inverse problem) is. Is the answer that no it is not useful for learning the inverse map but yes it is useful for learning the forward map (both answers based on the findings in Figure 3)?

Minor concerns

4. (Clarity) Please provide links to the exact Section in the Appendix rather than just referencing the entire appendix (e.g. lines 308, 311, 314, 377, 388). 
5. (Significance) Include quantitative scores in the qualitative examples (e.g. in Figures 4 and 8) to give a better understanding of how differences in scores translate to significantly perceptible differences.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores deep learning methods to address the forward and inverse problems in subsurface imaging by mapping between velocity maps and seismic waveforms. The authors introduce the Generalized Forward-Inverse (GFI) framework, which unifies past approaches and addresses open questions around latent spaces, manifold learning, and model complexity. They propose two novel architectures, Latent U-Net and Invertible X-Net, which achieve state-of-the-art results on synthetic datasets and show promising zero-shot performance on real-world-like datasets.

### Strengths
The paper presents a unified architecture for addressing both forward and inverse subsurface imaging problems, while previous studies typically focus only on inversions. Modeling these problems together within a single framework could offer valuable insights, though it may also introduce potential challenges.

### Weaknesses
1. The paper’s narrow focus on subsurface imaging may limit its relevance for a broader audience at ICLR. If there are analogous problems in other fields where their framework could be applied? 
2. Comparisons to prior work are outdated, with main competing methods from 2019, like InversionNet and VelocityGAN. The authors are encouraged to consider more recent advances, such as “Physics-Informed Robust and Implicit Full Waveform Inversion Without Prior and Low-Frequency Information (2024)” and other state-of-the-art methods from 2023, for a fairer and more relevant assessment.
3. While the authors aim to model the forward problem using neural networks, it is uncertain if the learned mapping can capture complex physical dynamics, as seen in the governing equations of seismic data. Extensive out-of-domain experiments are recommended to assess the model’s robustness and generalizability to real-world data.

4. The paper's structure could be streamlined to highlight the proposed methodology. Currently, a substantial background section precedes a brief description of the methods. Specific sections of the introduction and related work and preliminary knowledge could be condensed.

5. The model specifications lack detail; adaptations made to standard UNet for subsurface imaging and the mechanisms enabling X-Net’s invertibility are unclear. Clarifying these aspects and specifying when each model might be preferred would improve understanding.
6. Experimental comparisons with recent baselines (published within the last 2-3 years) would add rigor to the evaluation, as several recent methods could serve as relevant benchmarks. To name a few, “Physics-Informed Robust and Implicit Full Waveform Inversion Without Prior and Low-Frequency Information 2024”; “Full-waveform inversion using a learned regularization 2023”; “Wasserstein distance-based full-waveform inversion with a regularizer powered by learned gradient 2023”; “Full-waveform inversion using a learned regularization 2023”, and more.
7. Key numerical results in Section 5 are deferred to the appendix, which reduces clarity. Summarizing key findings directly in the main text would enhance readability.
8. The Invertible X-Net significantly underperforms other models, raising questions about its ideal use cases. Further context on scenarios where the X-Net's design offers advantages would be valuable.

### Questions
please see my concerns in the Weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Subsurface imaging is a technique for identifying the geophysical properties of layers underneath the Earth's surface. There are two directions: learning the mapping from velocity maps to seismic waveforms, called the forward problem, and the inverse problem which is the mapping from seismic data to velocity maps. Most of the previous deep learning-based research in the field focuses only on the inverse problem also referred to as full waveform inversion (FWI). This paper proposes a generalized forward-inverse (GFI) framework that unifies both directions, i.e., FWI and the forward problem. The framework builds on the assumption of manifolds and latent space translations, proposing two model architectures for the latter, namely Latent U-Net and Invertible X-Net. Latent U-Net architecture employs two U-Nets for translation between the velocity and waveform latent representations and vice versa, while Invertible X-Net uses a single IU-Net that simultaneously learns forward and inverse translations. The GFI framework encompasses previous works in deep learning for subsurface imaging, at the same time trying to answer questions such as the effect of latent space size, the importance of manifold learning and the value of jointly solving forward and inverse problems. The models were evaluated on the synthetic OpenFWI dataset and their generalization ability was tested on two real-world-like datasets.

### Strengths
- The paper is clearly presented and well-organized.
- Even though the GFI framework is not the first idea that unifies seismic imaging forward and inverse problems (e.g., Auto-Linear [1]), it tries to systematically characterize and unify prior research in this area. The paper answers a few questions that were open in the research field. The paper concludes the following: the size of the latent space should be decided based on the complexity of the problem, the training of the latent spaces should be influenced based on the translation objectives and jointly solving forward and inverse problems is helping the model to achieve better forward solutions. In my opinion, these are answers to important questions that were backed up by the experiments in the paper.
- The paper introduces U-Net and IU-Net architectures for learning translations between velocity and seismic waveforms in the latent space. This leads to two new architectures for unified seismic imaging, namely Latent U-Net and Invertible X-Net. In comparison, the Auto-Linear framework's architecture comprises two separate linear layers trained for the forward and inverse translations in the latent space.
- Extensive and systematic experiments. The proposed approach was compared to the existing state-of-the-art methods both for the forward and inverse seismic imaging problems. The proposed method employing Latent U-Net architecture outperforms the existing methods in most of the datasets in both forward and inverse directions.

---
[1] Y. Feng, Y. Chen, P. Jin, S. Feng, Y. Lin, "Auto-Linear Phenomenon in Subsurface Imaging
." ICML, PMLR:235, 2024, pp. 13153-13174.

### Weaknesses
- Section 5.1.1. "What is the effect of latent space sizes on translation performance?" is kind of vague. It states that the ideal size of the latent space should be decided based on the complexity of the problem being solved. It is not clear how to determine the complexity in real-world applications. I think the authors should be more specific in this section.
- In Section 5.1.2. "Do we need complex architectures for translations?", there is a comparison between large and small Latent U-Net. The only difference was the complexity (size) of the U-Nets for the latent space translation and the influence of having skip connections. What is the relation between the number of parameters in encoders and decoders and the U-Net and how does it influence performance? The Invertible X-Net was not included in the experiments and only one size was used throughout the paper. It would be useful to see how its size influences the performance.
- The Invertible X-Net model achieves worse results than the Latent U-Net model for the inversion problem and often fails to outperform other state-of-the-art methods. In contrast, it outperforms most of the methods for the forward problem. It is not clear from the paper what are the benefits of having such a unified framework. I think the limitations should be stated more clearly in the paper.
- In the results section, in qualitative comparison, Figure 4. and Figure 8. are missing a measure of error (at least MAE). The picked images should have approximately the same MAE as the reported results in the tables presenting a realistic case and fair comparison. It is not clear whether the presented results were in a sense cherry-picked, e.g., the proposed method achieved better results than the average and other methods below the average for the illustrated example.
- Instead of having a qualitative comparison of the methods on the two real-world-like datasets, I think it would be much more representative to have a quantitative comparison. Here, the presented examples might also be cherry-picked.

### Questions
1. Could you comment on the complexity of the problems and how to "calculate" it when choosing the latent space size? How to choose this size for real-world applications and are there any limitations in real-world scenarios?
2. Why have you decided to remove Invertible X-Net from the experiments in Section 5.1.2.? Have you tried different model sizes for this architecture?  What is the relation between the number of parameters in encoders and decoders and the U-Net and how does it influence performance? 
3. It is not clear from the paper what are the benefits of having such a unified framework. Could you comment on the limitations? What would be your recommendation on which architecture to use and when?
4. Could you comment on the examples in Figure 4. and Figure 8. and the performance of the models achieved in terms of MAE?

### Soundness
3

### Presentation
4

### Contribution
3
