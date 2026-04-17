# Bidirectional Predictive Coding

- Decision: Accept (Poster)
- Scores: 8, 10, 2, 6

## Abstract
Predictive coding (PC) is an influential computational model of visual learning and inference in the brain. Classical PC was proposed as a top-down generative model, where the brain actively predicts upcoming visual inputs, and inference minimises the prediction errors. Recent studies have also shown that PC can be formulated as a discriminative model, where sensory inputs predict neural activities in a feedforward manner. However, experimental evidence suggests that the brain employs both generative and discriminative inference, while unidirectional PC models show degraded performance in tasks requiring bidirectional processing. In this work, we propose bidirectional PC (bPC), a PC model that incorporates both generative and discriminative inference while maintaining a biologically plausible circuit implementation. We show that bPC matches or outperforms unidirectional models in their specialised generative or discriminative tasks, by developing an energy landscape that simultaneously suits both tasks. We also demonstrate bPC's superior performance in two biologically relevant tasks including multimodal learning and inference with missing information, suggesting that bPC resembles biological visual inference more closely.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a hierarchical neural network architecture in which the same error-minimization process is used to reconcile both bottom-up and top-down signals. Neural activity is determined using an iterative process of gradient decent to minimize these errors. Weights are also updated to minimize theses errors.

### Strengths
The proposed architecture employs only local information for inference and learning.

The proposed architecture is very flexible and can be used for a range of tasks, such as image classification, image generation, supervised and unsupervised learning.

### Weaknesses
Fig. 1 is not very helpful in showing the differences between architectures. One issue is that arrows of the same colour seem to mean different things in different panels.

The method is only tested using small, simple, data-sets.

Many inference steps are performed to process each sample. The number of steps seems too large for this to be a viable model of biological visual inference or a method that could have practical, machine learning, applications.

The relatively small differences in performance are difficult to see given the large scales used in the graphs showing the results (Figs. 3, 4, and 5).

### Questions
The proposed architecture seems to be very similar to that proposed in Qiu et al. (2023). Apart from the lack of tied weights (mentioned on l.155) are there any other differences? Why not compare the performance of your network to this one?

Could the proposed method scale-up to deal with large data-sets, such as ImageNet? 

How was the number of iterations chosen? How sensitive are the results to this choice? Would the number of inference steps need to increase with the size of the architecture? 

How do the computational costs of the different methods compare?

Could the proposed method be used for semi-supervised learning?

It is claimed (l.292) that the proposed architecture only requires half as many neurons as an alternative architecture using separate bottom-up and top-down networks. Is this claim ignoring the error-neurons? If so, why?

What do the error bars on the results figures represent? Are the differences in performance of the compared methods statistically significant?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The brain is able to do both generative and discriminative tasks, however the current most influential model for how the brain does that, predictive coding, has great trouble doing both in the same network. The authors present a surprisingly simple bidirectional version of the classical Rao&Ballard predictive coding model and show how that then a single model is able to solve both generative and discriminative tasks.

### Strengths
This is a great paper. Conceptually simple, though with many smaller innovations that are only mentioned in passing, the presented work marks a great step towards highly functional predictive coding networks, fixing previous issues with capabilities and, importantly, scaling. 
The work is exceptionally thorough, well designed and -mostly- very clear (see below). 
Noteworthy is also the work in the appendix where the models are scaled up to large networks and many-class classification benchmarks. This solves another big problem with predictive coding networks, that is, the previous work simply does not scale (we, and others, tried). For example, the approach by Song et al mentioned fails for cifar-100 and also for deeper networks. The approach presented here does scale, and this should be mentioned more clearly in the final version of the paper.

### Weaknesses
Minor issues:
- the issue of symmetric/shared bottom-up/top-down connections (l154) is a bit exaggerated, as it is easy to show that as long as identical local learning signals can be applied at either end, weight decay can result in symmetrical weights. (Tim Lillicrap wrote on this). 
- section 4.5 lacks clarity, in particular on what exactly is shown in Figure 7. What am I seeing here, what is the paradigm?
- the issue of scaling, which this approach solves to a large extend, is only mentioned in the appendix.

### Questions
- beyond fig 4e, what is the influence of the number of inference steps for the various problems, and is inference done sequentially per layer or in parallel? 
- can you say something on whether/how the approach would extend to dynamic rather than static stimuli?
- can you comment more on how the networks are initialized for unsupervised learning (l106/l302)?
- the VAE may not be the best self-supervised model for both generative and discriminative tasks within the set framework, that seems to rather be Masked-Autoencoders. Can you comment on this? Ideally compare to MAEs?
- In 4.3, it is noted that learning rarely takes place only in an unsupervised setting. I would argue however that supervised learning is much rarer than RL. Can you comment on whether the approach extends to RL?
- the fact that Max-Pooling has such detrimental results is noteworthy (l 364), and would seem to argue for for example all-convolutional networks. Comment?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Bidirectional Predictive Coding (bPC), a biologically inspired architecture that aims to integrate discriminative and generative learning within a single predictive coding framework. The authors claim that bPC can simultaneously achieve high classification accuracy and realistic image generation, and present results across supervised, unsupervised, and multimodal settings using datasets such as MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100. 
Although the paper is clearly written and the experiments are extensive, the overall novelty and empirical significance of the work are limited. The methodological design and experimental choices do not convincingly demonstrate contributions beyond existing predictive coding or joint generative–discriminative approaches.

### Strengths
1. The idea of integrating bidirectional inference under a unified predictive coding framework is conceptually clear and aligns with prior theoretical literature.
2. The paper reports a broad range of experiments, including classification, image generation, multimodal learning, and occlusion robustness, showing some effort toward comprehensive evaluation.

### Weaknesses
1. The central idea of combining discriminative and generative pathways within a shared latent representation is not novel in the context of modern machine learning. Many recent architectures, such as VAVAE[1] or VAR[2], already unify discriminative and generative learning with better theoretical grounding and empirical performance. The proposed bPC model appears to be a minor variation of existing predictive coding formulations (discPC, genPC, hybridPC) with shared weights, rather than a fundamentally new mechanism.
2. The network design is extremely simple. Most experiments are conducted with two hidden layers of 256 neurons or shallow convolutional networks. This small scale makes it difficult to evaluate whether the claimed properties would generalize to larger or more realistic settings.
3. The choice of benchmarks (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100) is outdated and insufficient to support strong claims of scalability or generality. While such datasets can be acceptable for conceptual neuroscience modeling, they are not appropriate for evaluating new machine learning architectures, especially in a venue like ICLR which emphasizes methodological innovation and empirical rigor in AI.
4. Although biologically inspired models sometimes employ simplified architectures to study neural mechanisms, those works typically analyze real neural data or test neuroscientific hypotheses. This paper does neither. Without such grounding, the justification for using small-scale networks and toy datasets becomes unclear and weakens the relevance of the work to both neuroscience and AI audiences.
5. In Figure 1, subfigures (A) and (B) show no visible difference between discPC and discBP, or between genPC and genBP, making it difficult for readers to understand the structural differences among these variants.
6. In Figure 7(A), the authors mention using one-hot labels as text inputs. However, multimodal setups typically require a meaningful text encoder (e.g., CLIP). Using one-hot vectors does not constitute true multimodal learning, and the symbol shown in the figure (an ear icon) is also misleading in this context.
7. Many mathematical expressions lack punctuation marks at the end of equations. 

[1] Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models. CVPR 2025

[2] Visual autoregressive modeling: Scalable image generation via next-scale prediction. NeurIPS 2024

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a bidirectional predictive coding (PC) model. Unlike Rao and Ballard’s formulation, where feedforward signals primarily carry prediction errors and feedback connections carry predictions, this model allows both feedforward and feedback pathways to carry predictions or expectations. The latent activity at each level is jointly constrained by top-down expectations and bottom-up proposals, with the energy function minimizing both discrepancies. This formulation brings predictive coding closer to classical interactive activation models (McClelland and Rumelhart), adaptive resonance theory (Grossberg and Carpenter), and hierarchical Bayesian inference via bottom-up and top-down belief propagation (Lee and Mumford). While the conceptual shift is incremental, the formulation is clean and biologically motivated, and the empirical comparisons establish the model’s competitiveness relative to other predictive coding models, particularly the hybrid predictive coding schemes.

### Strengths
1. The bidirectional predictive coding formulation is conceptually simple, elegant, and aligns more closely with classical interactive cortical models. It is worth emphasizing, however, that in Rao and Ballard’s model, the initial feedforward sweep does convey bottom-up evidence (the bottom-up prediction), and only subsequent iterations carry the prediction residues or prediction error signals. Thus, the primary difference is in the steady-state treatment and symmetry of information flow.

2. The inference and learning rules are derived from an energy function, and learning is Hebbian, giving this approach greater biological plausibility than backprop-based deep networks.

3. In contrast to autoencoder architectures, encoding and decoding (recognition and generation) in this model are unified. The same circuitry supports inference, classification, generation, and cross-modal completion without explicit mode-switching.

4. Demonstrating cross-modal pattern completion within a predictive-coding framework is novel and compelling.

5. From a neuroscience perspective, the model avoids assuming separate error and representation neurons; instead, each unit receives both bottom-up and top-down drives and evolves dynamically to reduce error. This yields an elegant single-population interpretation consistent with cortical microcircuit ideas.

### Weaknesses
1. The conceptual innovation is modest and closely related to earlier models of bidirectional inference (interactive activation, adaptive resonance, hierarchical Bayesian models). The paper could better situate itself in this lineage.

2. Although the results show benefits over prior predictive-coding variants, scalability to real-world architectures and large-scale deep learning benchmarks remains unclear; current experiments appear focused on moderate-scale or simplified tasks.

3. Historically, biologically motivated predictive coding architectures such as PredNet (Lotter, Kreigman, Cox) achieved strong performance in video prediction. It would be useful to discuss the connection and differences, particularly regarding information flow vs. error flow debates.

### Questions
1. How do the authors position this model relative to classical bidirectional inference frameworks such as interactive activation, adaptive resonance, hierarchical Bayesian inference, and encoder-decoder networks with skip pathways (e.g., U-Net), all of which integrate bottom-up and top-down signals during inference?

2. Prior work (e.g., PredNet) argued that whether feedforward activity represents proposals or prediction errors may not significantly change performance in video-prediction tasks. Have you evaluated a variant using pure error propagation feedforward, and how do the results compare?

3. What are the main obstacles to scaling this architecture to large-scale tasks comparable to modern deep nets? Are the challenges computational, architectural, representational, or learning algorithms?

### Soundness
4

### Presentation
3

### Contribution
3
