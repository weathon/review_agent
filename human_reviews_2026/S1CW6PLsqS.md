# SupCLAP: Controlling Optimization Trajectory Drift in Audio-Text Contrastive Learning with Support Vector Regularization

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Contrastive language-audio pretraining, which aims to unify multimodal representations in a shared embedding space, serves as a cornerstone for building a wide range of applications, from cross-modal retrieval to cutting-edge multimodal large language models. However, we find that the perpendicular component of the pushing force from negative samples in contrastive learning is a double-edged sword: it contains rich supplementary information from negative samples, yet its unconstrained nature causes optimization trajectory drift and training instability. To address this, we propose Support Vector Regularization (SVR), a method that introduces an auxiliary support vector to control this perpendicular component, aiming to harness its rich information while mitigating the associated trajectory drift. The efficacy of SVR is critically governed by its semantic radius, for which we explore two unsupervised modeling strategies: direct parameterization and an adaptive radius predictor module enhanced with constraints to improve its predicting accuracy. Extensive experimental results demonstrate that our method surpasses widely used baselines like InfoNCE and SigLIP loss across classification, monolingual retrieval, and multilingual retrieval on standard audio-text datasets. Both the theoretical analysis and the experimental results on optimizing trajectory drift validate the correctness and effectiveness of our SVR method. Notably, our method is highly efficient, it operates without the need for extra training data or inference computation, and adds only a negligible overhead to the training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies contrastive language–audio pretraining (CLAP) and identifies a phenomenon called optimization trajectory drift, caused by the perpendicular component of the negative-sample pushing force in contrastive learning. The authors propose Support Vector Regularization (SVR), which introduces auxiliary support vectors and a semantic radius to selectively suppress this perpendicular component while preserving useful negative-sample information. They present both static and dynamic strategies for modeling the semantic radius. Experiments across monolingual and multilingual audio–text retrieval, as well as zero-shot classification, show consistent improvements over InfoNCE and SigLIP with negligible overhead.

### Strengths
1. The work offers a fresh angle by decomposing contrastive gradients into pulling and perpendicular pushing components, which provides new insights into training instability in CLAP.
2. The SVR formulation is presented with solid mathematical justification, and the derivations clearly show how trajectory drift is effectively suppressed.
3. Experiments across retrieval, classification, and multilingual tasks, along with ablations, demonstrate consistent improvements while keeping the computational cost minimal.

### Weaknesses
1. While the paper offers a new analytical angle, the proposed SVR is similar to a regularization tweak of InfoNCE and conceptually close to earlier margin-based or support-vector–style methods.

2. The dynamic semantic radius predictor is unsupervised and fragile under noise. Its meaning is not clearly explained, and why it shrinks during training or how it reflects semantic difficulty is described more heuristically than rigorously.

3. The baseline setup makes comparisons harder to follow. In Tables 1 and 2, many models are cited only by reference instead of name. Some strong recent baselines are also missing, including DS-CLAP (Liu et al., 2024), Cacophony (Zhu et al., 2024), and ATRI (Yin et al., 2025).

4. The experiments are limited to AudioCaps and Clotho, with multilingual results relying on machine translation. Compared with recent work that uses larger datasets such as AudioSet, WavCaps, and LAION-Audio and covers broader tasks like captioning or grounding, the generalization claims are less convincing.


[a] Liu et al., 2024, DSCLAP: Domain-specific contrastive language-audio pre-training

[b] Zhu et al., 2024, Cacophony: An improved contrastive audio-text model

[c] Yin et al., 2025, ATRI: Mitigating multilingual audio-text retrieval inconsistencies by reducing data distribution errors

### Questions
1. The dynamic semantic radius relies on an unsupervised predictor. How robust is it under noisy embeddings (e.g., low-resource or weak encoders)? Can the authors show sensitivity or variance analysis to demonstrate stability?

2. The radius shrinks during training, but the explanation is heuristic. Can the authors provide clearer evidence that its values correlate with semantic difficulty (e.g., hard-negative density or retrieval errors)?

### Soundness
2

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
2

### Summary
The paper proposes SupCLAP with a Support Vector Regularization (SVR) that shifts the text embedding towards its positive audio.They add an extra contrastive term, claiming to suppress the perpendicular component of the negative “pushing force” and reduce optimization trajectory drift.  And this will help improve performance.

### Strengths
This paper gives a novel regularization term on the InfoNCE loss,  their experiments show that this regularization indeed bring benefits. They also provide their intuition in section 2.  And The writing of this paper is good.

### Weaknesses
1. In the section 2, the authors discuss the optimization trajectory drift. However, the causal link “lower perpendicular component → better performance” is not theoretically or empirically established. So this section is not very convincing to me. The author may provide theoretical or empirical evidence.
2. For the InfoNCE loss, we originally want the text embedding being away from the negative sample. So why we need to reduce the perpendicular component of the negative “pushing force”? 
3. Moreover, the proposed Support-Vector Regularization (SVR) seems closely related to simply re-scaling or re-weighting the contrastive gradients. It would strengthen the paper if the authors compared their method to simpler baselines—e.g., emphasizing the parallel component only or giving positive sample gradient more weights—to demonstrate that the observed improvement is not merely a side-effect of gradient magnitude control.

### Questions
See weakness.

### Soundness
2

### Presentation
3

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
This paper presents SupCLAP, a method for regularizing the contrastive learning optimization process of standard objectives (InfoNCE, SigLIP) to improve performance on downstream retrieval and 0-shot classification tasks.

### Strengths
- Motivation is reasonably strong, and mathematical rigor to the proposed method is appreciated.
- Despite the overall mathematical maturity of SupCLAP, the proposed method is still intuitive and straightforward, aiding the clarity of the paper.
- The experimental results, while generally modest, show clear gains to the using the proposed method relative to the baselines.

### Weaknesses
In general, there is not much massively wrong with the present manuscript, my two biggest concerns are:
- It would be useful to report confidence intervals for the retrieval / classification experiments, as the results are generally modest, and it is hard to tell whether this gain in performance is truly statistically significant.
- Overall, there is not much clear evidence as to *why* SupCLAP seems to work better empirically. While there is ample connection to *in theory* what might be going on during optimization, this is never attempted to be empirically confirmed. It would be very useful to analyze parts of the optimization process for SupCLAP (vs InfoNCE or SigLIP) such as gradient norm, gradient variance, or simply convergence speed to see whether the surmised theoretical reasons for its performance actually translate in practice.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses the issue of optimization trajectory drift during contrastive learning due to the perpendicular component of pushing force from negative samples. It highlights how this aspect can lead to instability in training and proposes a novel solution called Support Vector Regularization (SVR).

SVR introduces an auxiliary support vector that allows for better control of negative sample influence, leveraging their informative content while mitigating trajectory drift. The effectiveness of SVR is critically tied to the selection of the semantic radius, and the paper explores two unsupervised modeling strategies: static direct parameterization and an adaptive radius predictor module with constraints.

Extensive experimental results demonstrate that SupCLAP significantly outperforms various baseline methods across tasks such as classification and both monolingual and multilingual retrieval, validating its potential in stabilizing optimization paths and improving model performance.

### Strengths
Addresses Optimization Trajectory Drift: The proposed Support Vector Regularization (SVR) method effectively controls optimization trajectory drift caused by the influence of negative samples, enhancing training stability.

Performance Improvement: SupCLAP significantly outperforms various existing baselines across tasks such as classification and monolingual/multilingual retrieval, demonstrating its superior performance in multimodal learning.

Exploration of Unsupervised Modeling Strategies: The study explores both static parameterization and adaptive radius prediction as unsupervised modeling strategies, providing new methodological insights for future research.

### Weaknesses
Limited Generalizability Beyond Benchmarks: While the performance in the tested tasks is impressive, the study may have limitations in generalizability. The results may not translate well to all real-world applications, particularly those outside the tested domains or with different data characteristics.

### Questions
Given that the SVR method introduces additional parameters that require careful tuning, how does this impact the overall efficiency of the method during training and implementation? Specifically, what are the trade-offs between the improved performance and the potential increases in optimization time?

### Soundness
3

### Presentation
3

### Contribution
3
