# PAPM: A Physics-aware Proxy Model for Process Systems

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 6

## Abstract
Process systems, which play a fundamental role in various scientific and engineering fields, often rely on computational models to capture their complex temporal-spatial dynamics. However, due to limited insights into the intricate physical principles, these models can be imprecise or inapplicable, coupled with a significant computational demand exacerbating inefficiencies. To address these challenges, we propose a physics-aware proxy model (PAPM) to explicitly incorporate partial prior mechanistic knowledge, including conservation and constitutive relations. Additionally, to enhance the inductive biases about strict physical laws and broaden the applicability scope, we introduce a holistic temporal and spatial stepping method (TSSM) aligned with the distinct equation characteristics of different process systems, resulting in better out-of-sample generalization. We systematically compare state-of-the-art pure data-driven models and physics-aware models, spanning five two-dimensional non-trivial benchmarks in nine generalization tasks. Notably, PAPM achieves an average absolute performance improvement of 6.4%, while requiring fewer FLOPs, and only 1% of the parameters compared to the prior leading method, PPNN. Through such analysis, the structural design and specialized spatio-temporal modeling schemes (i.e., TSSM) of PAPM exhibit not only the most balanced trade-off between accuracy and computational efficiency among all methods evaluated, but also an impressive out-of-sample generalization.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces PAPM, a spatio-temporal model to capture complex dynamics which arguably follow similar patters, i.e., a mixture of diffusion and/or convection flows, internal and external source terms. The PAPM architecture encodes the state of the system, and depending on the problem at hand applies either localized, spectral, or hybrid operators to parameterize the different operators. Subsequently, time-stepping schemes are applied to mimic temporal updates. PAPM is tested on 4 known 2D fluid mechanics benchmarks systems.

### Strengths
- Introducing parameterized operators is a very interesting contribution.

### Weaknesses
- The presentation is slightly hard to follow, it is not clear to me how exactly all these operators are parameterized and how such models can be scaled up. Is there only one operator block used or can these modules be stacked? Pseudocode / real code would definitely help.
- The models are evaluated on a fixed grid with fixed resolution. For such systems standard models such as modern U-Nets and / or convolutional based neural operators should be used for comparison (Raonic et al, Gupta et al), or even Vision Transformers. An alternative is to showcase resolution independency to justify the comparisons. 
- I am pretty puzzled by the low number of parameters. It seems that hardly any model uses more than 1 million parameters. This is in my opinion a heavy under-parameterization for 2D problems. Compare for example Fig 1 in Gupta et al?
- The paper makes a strong claim for better physics modeling, i.e., strong physics bias, yet there is no evidence that with low number of samples the performance is better compared to baseline models.
- Figure 6 is not comparing to the best baseline model but FNO which has 10 times worse performance than Dilated ResNets on the RD2d task.
- It is impossible to judge how the individual components contribute to the results - ablation would help.


Raonić, B., Molinaro, R., Rohner, T., Mishra, S., & de Bezenac, E. (2023). Convolutional Neural Operators. arXiv preprint arXiv:2302.01178.

Gupta, Jayesh K., and Johannes Brandstetter. "Towards multi-spatiotemporal-scale generalized pde modeling." arXiv preprint arXiv:2209.15616 (2022).

### Questions
- How can PAPM be extended to variable grid sizes, or to non regular grids?
- How can PAPM be scaled up to larger number of parameters?
- Would it be possible to resort to the standard terminology of "operator learning" which is now standard in the community?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel method, the Physics-Aware Proxy Model (PAPM), aimed at improving the efficiency and accuracy of process systems modeling. PAPM incorporates a portion of prior physical knowledge (including conservation and constitutive relations) into the model and introduces a new Temporal and Spatial Stepping Method (TSSM), which is claimed to enhance the model's applicability and predictive ability. The authors conduct several tests, indicating that PAPM seemingly outperforms existing data-driven and physics-aware models.

### Strengths
1. The paper addresses a critical issue in the field of process systems modeling, proposing an innovative solution that combines partial prior mechanistic knowledge with a holistic temporal and spatial stepping method.
2. The PAPM model shows impressive results in terms of both improved performance and reduced computational costs compared to existing methods.
3. The paper is well-structured and the methodology is clearly explained, with extensive validation.

### Weaknesses
1. The paper could dive further into limitations of the method.
2. The paper could benefit from a more detailed comparison with existing methods. While the authors compare their method to state-of-the-art models, it would be helpful to see a more detailed analysis of why their method outperforms these existing approaches.

### Questions
- How well would the PAPM model perform on process systems with less well-understood or more complex physical principles?
- Could the proposed model be applied to other types of systems beyond process systems?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a way to leverage process systems, which is a key model that can be used to emulate a number of physics models. The authors claim that process models are in general complex and difficult to understand and can also lead to incorrect results. In this paper they propose PAPM (physics-aware proxy model) which has the claimed benefit of including physics priors to accomplish better performance on prediction tasks.

### Strengths
1. Paper is mostly well written
2. Experiments are clear

### Weaknesses
1. While I appreciate the intuitive explanations, process systems are not defined adequately, and this really impedes assessment of the paper. The terms describing this main concept are vague (abstract, introduction and in section 3), and qualitative. Nevertheless, I hope authors can clarify this in the discussion phase (see questions).
2. It is unclear what is required in training vs. at inference
3. The experiments seem to be run for one setting (no monte-carlo simulations)
4. The experiments only consider classical, highly-structured pdes, it is unclear how the proposed model can be used for real-world settings where the dynamics are unknown and may not follow the underlying assumption of (eq.1)

### Questions
### Understanding Process Models: 

While the contributions seem important it is difficult to understand what process models are. Following are questions which can help authors identify what the reviewer is struggling with, hopefully to help update the paper for a wider audience.
1. Why are the dynamics/equations of the process model unknown? Isn't it defined by the practitioners?
2. In relation to 1, it seems that authors consider dynamics which take the form of eq.1, while the exact values that these quantities take are unknown? Is this true? 
3. How are process models different from the proposed model in relation to eq1 and Fig. 3?

### Understanding PAPM:
4. \lambda is defined as "coefficients" in sec 4.1, but it is unclear how they related to eq 1.
5. During training the quantities, t, \lambda, \Phi_0 etc. are available, but during inference, what all inputs are assumed to be available?
6. What is the impact of missing quantities on training, can the model still learn?
7. The structures in Fig 3 (b and c) are still blackboxes, how do these assist in understanding the system as opposed to a process model?


### Minor/semantics/other comments:
1. Why use TSSM for temporal-spatial modeling method (TSSM), TSMM or TSM is more appropriate?
2. The acronyms DF, CF, IST, and EST can be defined just below eq(1) for clarity.
3. Decomposing pde as spatial and temporal modules has been studied in PIML. It is important to discuss these similarities in the present work; see Seo 2021. 


Seo et al. 2021, Physics-aware Spatiotemporal Modules with Auxiliary Tasks for Meta-Learning, IJCAI 2021.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a specific structure to encode physics prior to the training and use Euler/RK for time stepping to achieve good generalization capability under a data-scarce situation.

### Strengths
The paper explicitly takes into account the physics of the system when designing the system, yielding better generalization capability compare to baselines like FNO

### Weaknesses
I am a bit confused with the experimental setting. I really like the argument of baking more physics prior to the model. However, it seems that during the training, the model is still trained with a large-scale dataset - where one needs up to 10^6 times to generate this dataset.

### Questions
1. I am curious any thoughts on why FNO performs so badly even with the full dataset for training? This is different from what I generally get from various literature. 
2.  I am curious why different padding strategy corresponds to boundary condition. How does it help enforce the boundary condition?
3. how could it generalize to mesh base simulation with adaptive resolution?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
