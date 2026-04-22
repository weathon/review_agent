# Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 2

## Abstract
Diffusion policies for visuomotor robot manipulation tasks achieve remarkable dexterity and robustness while only training on a small number of task demonstrations.  However, the reason for this performance remains a mystery. In this paper, we offer a surprising hypothesis: diffusion policies essentially memorize an action lookup table---\emph{and this is beneficial}. We posit that, at runtime, diffusion policies find the closest training image to the test image in a latent space, and recall the associated training action  (i.e. action chunk), offering reactivity without the need for action generalization. This is effective in the sparse data regime, where there is not enough data density for the model to learn action generalization. We support this claim with systematic empirical evidence, showing that even when conditioned on highly out of distribution (OOD) images, Diffusion Policy still outputs an action chunk from the training data. We evaluate and compare three representative policy families on the same data set: Diffusion Policy, Action Chunking with Transformers (ACT), and GR00T, a pre-trained generalist Vision-Language-Action (VLA) model.  We show that Diffusion Policy gives strong action memorization giving surprising robustness in OOD regimes, ACT shows action interpolation with poor robustness in OOD regimes, and GR00T (benefiting from substantial pre-training) shows both action interpolation and OOD robustness. As a simple alternative to Diffusion Policy, we introduce the Action Lookup Table (ALT) policy, showing that an explicit lookup table policy can perform comparably in this low data regime. Despite its simplicity, ALT attains Diffusion Policy–level performance while also providing faster inference and explicit OOD detection via latent-distance thresholds. These results reframe diffusion policies for robot manipulation as reactive memory retrieval under data sparsity, and provide practical tools for interpreting, evaluating, and monitoring such policies. More information can be found at: \url{https://stanfordmsl.github.io/alt/}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The work investigates the effectiveness of Diffusion Policy (DP) for visuomotor robot manipulation, proving that their performance in low-data regimes stems from action memorization rather than any sort of generalization. Empirical evidence demonstrates that the DP tends to recall action sequences from its training data, even when faced with highly OOD inputs, providing robustness in those scenarios. The authors compare this behavior with other architectures like ACT, which shows action interpolation but lacks robustness, and GR00T N1.5, which shows generalization due to pre-training. To validate their hypothesis on DP, the researchers introduce a simpler, faster alternative called the Action Lookup Table (ALT) policy, which explicitly relies on nearest neighbour lookup in an embedding space, to match DP's performance while also providing OOD detection. The (image & joint pose) encoder that creates the embedding is trained via contrastive learning.

Basically, this work does a great job reframing DP, in data-sparse regimes, as a memory retrieval system.

### Strengths
1. Well explained idea. Strong paper to put forward the point of memorization.
2. The example of grasping the cup, along with the memory-audit metric S, is simple and effective.

### Weaknesses
1. Sec 3.3 & Fig 3. More OOD variations could be added. Specifically, what if you change the cup object itself?
2. Fig 4 is quite confusing at first sight. I had to read it again to understand that the lines indicate distance scores. Considering removing information to make it clearer. Do you need these absolute distance scores, when you already have a similarity metric? in that metric, you can see the one-hot like nature of DP anyway.
3. ALT does not attempt to interpolate between training data actions. However, this may be a design choice by the authors.

### Questions
1. "The diffusion model seems to revert to one or two fallback action sequences when presented with OOD images." Based on FIg-4, it seems like that trajecotry is traj4 ? Why do you think there is a single fallback demo? And why is that the chosen traj? Is it the one earliest in the training? What characteristic does traj4 have that sets it apart from the others. A deeper discussion into might provide insight into the importance of each demo in the training pipeline.

2.Fig 5: Please improve visibility of the lines in this figure. The green and blue arrows are too similar, especially for such a small figure. Consider higher-contrast colours, or use dots and dashes.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors hyphothsize that visuomotor policies, e.g., diffusion policies (DP) memorize the image-action pairs instead of learn to generlize on them. To verify this, they conduct a systematic emprical study including analysis and experiments ranging from toy data to real-robot data. Moreover, they evaluate on three representative policy architectures, namely DP, Action Chunking with Transformers (ACT), and GR00T and found that they exhibit different behaviors. Based on this, a simple alternative called action lookup table implemented as observation encoder for action retrievial is introduced and requires less memory while providing greater run-time efficiency and OOD monitoring capability.

### Strengths
- An interesting and comprehensive emprical study for a highly relevant but under-explored problem of current visuomotor imitation learning framwork. I believe that this investigation is inspiring for both researchers and practioners.

- The presentation of this submission is clear, well-organized and easy-to-follow along with illusrative figures and visualization;

- The toy experiments and visuliazation provide a great example for the reader to under the problem at the beginning;

- The analysis of results across three architectures is presented clear and concise, while also making connection to the manifold attraction effects. 

- The experiment is extensive from the perspective of hyphothesis validation. Morever, the experimental design (ID and OOD setups), policy architectures and evaluation metrics selection further consolidate the strictness of this empirical investigation.

- This study is practical-relevant and highly informative for practioners, because in the real-world scenarios a large amount of robot data may be costly to collect.

### Weaknesses
- The current study and robot-related experiments are mainly done under the low-data regime. It remains unclear what implications this study can bring for the visuomotor imitation learning at larger scale. It would be valuable to provide insights (discussion or experimental study) about the validity for a large scale of data, e.g., 200, 2000 or 20k demostrations are provided.

- Though the presention is clear and great, I would still suggest to shorten the part of analysis for the three architectures or to present the difference or similarites of them in a more visual way. Then more space can be left for a slightly proper experiment part, including setup introduction and more results. 

- In the experiment, the performance of ACT and GR00T for reference is missing; In table 2., success rates on OOD scenarios would be nice to have to investigate the expected behaviors described in section 3.3.

- Regarding the proposed ALT, besides the improved computation efficiency, its OOD detection capability is highly desirable for robustness enhancement. It would be highly valuable to perform more experiments for this part to understand how effective it can be. For example, a comparison with other density-based approaches such as normalizing flows, which can be trained on the latent features. 

[1] Feng, J., Lee, J., Geisler, S., Günnemann, S., & Triebel, R. (2023, December). Topology-Matching Normalizing Flows for Out-of-Distribution Detection in Robot Learning. In Conference on Robot Learning (pp. 3214-3241). PMLR.

### Questions
- The influence of the feature dimensionality of ALT on the task performance seems relevant. Is this a core desgin question?

### Soundness
3

### Presentation
4

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
The main claim of the paper is that diffusion policy in low-data regime memorize demonstration actions during training and looks it up during inference, and proposes Action Lookup Table policy. Experiments with small datasets show that diffusion outputs memorized action chunks even when given out of distribution inputs. ALT also introduces out-of-distribution detection and stochastic kNN sampling, and observes that overfitting can be beneficial in manipulation setups.

### Strengths
- Clear claim with sound evidence from sim and real experiments.
- ALT performs well in the low data regime and is orders of magnitude faster than diffusion / ACT.
- Out-of-distribution experiments show that diffusion policy replays action chunks from the training set.

### Weaknesses
- My main concern is novelty. The idea that kNN lookup as a policy class on contrastively learned features works well in the low-data regime has been well-established in literature [1], and it is also well-known in the field that overfitting can help with downstream rollouts in behavioral cloning.
- Limited experiments. Main claims are mostly validated with cup grasping experiments, with low effective state and action space dims. It would be good to see whether the diffusion memorization holds in higher dim action / state spaces (e.g. dexterous manipulation for higher dim actions, or benchmarks with more objects like LIBERO), as well as a sweep over dataset scale.

[1] The Surprising Effectiveness of Representation Learning for Visual Imitation. Pari et al., 2021.

### Questions
- Does this claim hold on a higher dim action space, and environments with multiple objects? Does it hold on a larger scale dataset?
- Ablation: how much does the representation learning matter?

### Soundness
3

### Presentation
3

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
This paper investigates why diffusion policies perform remarkably well in robot manipulation despite being trained on very few demonstrations. The authors hypothesize that diffusion policies do not truly generalize actions, but instead memorize and retrieve action sequences-similar to a nearest-neighbor lookup table-based on the closest-matching observation in latent space. Through real and simulated pick-place experiments, they show that Diffusion Policy consistently reproduces training trajectories even under out-of-distribution inputs, while ACT interpolates actions but fails in OOD settings, and GR00T combines some generalization with robustness due to large-scale pretraining. Motivated by these findings, they introduce a simple Action Lookup Table (ALT) approach using contrastive visual embeddings, which matches Diffusion Policy performance in low-data regimes while being faster and explicitly flagging OOD cases. The key takeaway is that the strength of Diffusion Policy in low-demo robot learning stems largely from memorization, not generalization, and that simple retrieval-style methods can provide similar performance with far lower cost and greater interpretability.

### Strengths
1. The paper is well-organized, progressing logically from hypothesis to experimental validation and practical takeaway.
2. The paper performs a sound analysis of Diffusion Policy and other single-task imitation learning methods that are known to perform well in the low-data regime. Diffusion Policy is indeed recommended to train to overfitting, and the paper tries to make the effects of that trend very clear.
3. The authors design structured experiments (In-distribution, OOD-interpolate, OOD-extrapolate, and OOD-distractor) to dissect memorization versus generalization behavior, making the conclusions grounded and interpretable.
4. The introduction of the Action Lookup Table (ALT) policy provides a transparent, computationally efficient alternative that matches diffusion policy performance while being ~300x faster and offering explicit OOD detection.

### Weaknesses
1. While I appreciate the analysis, the paper, in my opinion, does not provide a novel viable solution. It just presents a simple alternative to Diffusion Policy, which is not scalable to something that current large behavior models aim for. While I agree that Diffusion Policy is prone to overfitting, then so is any high-capacity supervised learning model trained on less data. Diffusion Policy did not just present one specific architecture or loss function, but provided a starting point for future approaches, which might be able to have less overfitting and more generalization while maintaining the multi-modality and dexterity that Diffusion Policy has. As such, I did not see anything in this paper which I did not expect from low-data supervised learning models, but it does not mean that researchers should replace them with non-scalable methods such as presented in this paper.
2. The experiments in this paper are primarily conducted on simple manipulation tasks (e.g., cup grasping, pick-and-place), making it unclear whether the findings generalize to more complex, multi-skill or long-horizon robotic tasks.

### Questions
1. How much does ALT generalize, or is capable of generalizing to novel object placement arrangements, lighting, etc.? How does lookup table size scale with generalization capabilities?
2. As the number of demonstrations grows, how does ALT’s retrieval latency, memory footprint, and OOD detection accuracy scale? Would a nearest-neighbor search remain feasible for larger datasets or multi-task scenarios?
3. How would the ALT policy perform on tasks that require longer-horizon reasoning, contact-rich dynamics, or multi-step planning? Would its lookup-based retrieval break down when task stages are less visually correlated?
4. The paper uses a contrastive encoder to map images to latent representations. How sensitive is ALT’s performance to the choice of encoder architecture or the quality of contrastive learning?

### Soundness
3

### Presentation
2

### Contribution
2
