# MePo: Meta Post-Refinement for Rehearsal-Free General Continual Learning

- Decision: Reject
- Scores: 6, 6, 2

## Abstract
To cope with uncertain changes of the external world, intelligent systems must continually learn from complex, evolving environments and respond in real time. This ability, collectively known as general continual learning (GCL), encapsulates practical challenges such as online datastreams and blurry task boundaries. Although leveraging pretrained models (PTMs) has greatly advanced conventional continual learning (CL), these methods remain limited in reconciling the diverse and temporally mixed information along a single pass, resulting in sub-optimal GCL performance. Inspired by meta-plasticity and reconstructive memory in neuroscience, we introduce here an innovative approach named **Me**ta **Po**st-Refinement (MePo) for PTMs-based GCL. This approach constructs pseudo task sequences from pretraining data and develops a bi-level meta-learning paradigm to refine the pretrained backbone, which serves as a prolonged pretraining phase but greatly facilitates rapid adaptation of representation learning to downstream GCL tasks. MePo further initializes a meta covariance matrix as the reference geometry of pretrained representation space, enabling GCL to exploit second-order statistics for robust output alignment. MePo serves as a plug-in strategy that achieves significant performance gains across a variety of GCL benchmarks and pretrained checkpoints in a rehearsal-free manner (e.g., 15.10\%, 13.36\%, and 12.56\% on CIFAR-100, ImageNet-R, and CUB-200 under Sup-21/1K).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel method involving meta learning + covariance alignment for real world Continual Learning tasks. This requires them to refine pretrained models first using some "pseudo" tasks, and then simply align the features during online learning. This seems to show 10-15% improvements as a plug-and-play method.

### Strengths
1. The paper shows strong empirical evidence towards REALISTIC CL scenarios, and is not just constrained to ideal conditions. They also cover multiple datasets / types.
2. The method is generalist enough to be broadly applicable.
3. The ablations are sensible and comprehensive.

### Weaknesses
1. The paper is a bit too empirical - there is no theory regarding why these "pseudo" tasks transfer downstream.
2. There is no numerical stability analysis for the strong distributional assumptions the paper makes.
3. There seems to be substantial overhead to this sort of meta-training - this is not a simple "one-time-cost".
4. The choice of using weighted combination seems a little ad-hoc, and the paper limits itself to the Si-Blurry setting only.

### Questions
1. I would appreciate a stronger theoretical justification for why the downstream task transfer works. Could you provide some?
2. What is the performance like without pretraining data availability?
3. What is the numerical stability of the Cholesky decomposition in this setting?
4. Is there evidence of generalization of this method beyond the Si-Blurry setting?

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
This paper studies the setup of general continual learning (GCL) with pretrained models, where the model performs a single pass through a sequence of tasks, with potentially blurred task boundaries. The paper proposes (a) an extended pretraining phase, where the model meta-learns to refine its pretrained representations for improved adaptation to sequential learning, and (b) a method to align the output feature covariance with that from pretraining. The proposed method can be applied on top of existing GCL approaches, and the authors empirically demonstrate its effectiveness on CIFAR-100, ImageNet-R, and CUB-200 task sequences using different pretrained backbones. Ablation studies further show the benefit of each component of the proposed method.

This paper is generally sound, and the proposed method shows convincing performance improvements over baselines. The authors also conduct extensive analyses of their approach. My main concerns are the motivation for the proposed output alignment method and the clarity of certain claims and statements.

### Strengths
- The proposed method is clean, and the meta-representation learning component is intuitive.
- Extensive experimental analysis demonstrates the effectiveness of the proposed method.

### Weaknesses
1. The arguments made in Sec. 2.2 are not clearly stated; the offline, online, and GCL settings are not defined and the claims in the paragraph starting at L159 lack sufficient evidence. For instance, from Fig. 2b, MVP-Rep appears equally ineffective in both online and offline settings, but the authors state that it does not address online data streams as well. Several claims related to Fig. 2c also seem somewhat overstated, as all methods show similar performance between the online and GCL settings.
2. I find the motivation for output alignment unclear. Specifically, if the data in the incoming batch are drawn from a different distribution than that of pretraining (e.g., imbalanced classes), why should we expect the covariance matrix of their features to match the pretrained one (e.g., balanced classes)? Also, Fig. 4 suggests that pre-aligned features are more separable; but in classification, wouldn’t greater separability generally be desirable?
3. It seems that the pseudo task sequence used in this work always mirrors the structure of the downstream continual learning sequence. In practice, how should such a sequence be constructed during pretraining without knowing the structure of the downstream sequence? It might be useful to analyze the impact of a mismatch between the two.

### Questions
1. L426: "...due to imbalanced classes." It’s unclear how the performance drop can be attributed to class imbalance without comparing against a balanced setup under otherwise identical conditions.
2. L428: Isn’t this finding contradictory to prior work suggesting that SSL representations are more robust to continual learning than supervised ones [Gal+21, Dav+22]? Could the authors hypothesize why?
3. Fig. 7 shows that MePo produces sparser activations, but *why* are sparser activations desirable?


### Questions/comments that did not impact the score
4. L465: "We empirically validate..." Which specific result does this statement refer to?
5. I wonder if the joint training phase should be considered as the final task in the inner loop, since its only difference from other pseudo-tasks appears to be data composition.
6. The initialization of $\psi$ is not described in the main text but is provided in Algorithm 1 in the Appendix.
7. $\Sigma\text{pre}$ >> $\Sigma_\text{pre}$ in Eq. 8.
8. In Table 1, I recommend either bolding the better method within each of the two rows (w/ vs. w/o MePo) or explicitly reporting the improvement for clarity.

[Gal+21] Self-Supervised Training Enhances Online Continual Learning. Gallardo et al. BMVC 2021.\
[Dav+22] Probing Representation Forgetting in Supervised and Unsupervised Continual Learning. Davari et al. CVPR 2022.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes meta post-refinement (MePo for rehearsal-free general continual learning. The main ideas of the proposed method are MePo for representation learning and Mepo for feature alignment. The paper is well presented and shows a promising performance as reported in the numerical results. However, the paper has several issues that need to be clarified; please see the weaknesses.

### Strengths
(1). The paper is presented in a clear yet comprehensive way, and it is easy to follow. 

(2). The numerical results show a promising margin over the existing methods.

### Weaknesses
(1). From the perspective of methodology, it is questionable whether the paper has enough novelty. Training a base model with meta-learning guidance and feature alignment (combination) by weighted sum is nothing new. I do not see any new/novel mechanisms in either of the ideas.

(2). The proposed method is not supported by theoretical analysis, e.g., how the meta learner helps the base learner to achieve a better stability-plasticity, or how it elevates the base model to achieve a better adaptability to a new task.

(3). It is questionable why "self-supervised PTM is more realistic", while we have free/open supervised PTM models such as pretrained ViT on imagenet-21K or DINO.

(4). Continual learning is the art of defying catastrophic forgetting (CF). But I do not see forgetting measurement and discussion about it. 

(5). GCL that has online learning should be concerned about model throughput (training and inference) as it works on streaming data. Similarly, I do not see measurements and discussion about it.

(6). It is arguably unfair to compare the GCL/Online CL method with offline CL methods such as L2P, DualPrompt,  and CODA-P, especially when you did not search for their best hyperparameter settings in your experiment. Their best parameter setting should be suitable for CL but not for GCL/OCL.  Also, please kindly compare it to the other GCL/OCL SOTAs, i.e., RanPAC[1], RanDumb[2], F-OAL[3], PROL[4], and the newest PEFT (prompt, LoRA, and Adapter) structure.

(7). The pseudo code in Algorithm 1 is not clear on how the model processes each streaming chunk and how the learned knowledge is consolidated from many processed chunks.


reference:
[1]. Ranpac: Random projections and pre-trained models for continual learning (NeurIPS 2024)

[2]. Randumb: A simple approach that questions the efficacy of continual representation learning (NeurIPS 2024)

[3]. F-OAL: Forward-only online analytic learning with fast training and low memory footprint in class incremental learning (NeurIPS 2024)

[4]. PROL: PROL: Rehearsal Free Continual Learning in Streaming Data via Prompt Online Learning. (ICCV 2025)

### Questions
(1). Please address the weakness.

(2). Is 256 batch size considered as a realistic batch/chunk-size on GCL, as the previous online CL uses a small batch/chunk-size, e.g., 10.

(3). The post-pretraining process (Figure 1) shows that the whole pre-trained model is finetuned (learnable). But Table 2 shows far lower parameters. Could you please clarify this issue? From my perspective, computing additional learnable parameters only by the number of parameters for GCL phase is not fair and objective.

(4). Table 2 shows that w/ MePO requires larger parameters but lower time than w/o MePO. How is this possible? 

(5). Could you explain in more detail how each class samples (both from disjoint tasks and blurry tasks) are partitioned into 2 models (base learner and meta learner)?

### Soundness
2

### Presentation
3

### Contribution
1
