# Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Video anomaly detection (VAD) aims to identify unusual events in continuous video streams, yet most existing systems either rely on domain-specific retraining or fail to meet strict real-time demands. We present **Flashback**, a zero-shot and real-time paradigm that reframes VAD as retrieval over an offline pseudo-scene memory. Inspired by how humans recall past experiences to judge the present, Flashback constructs a large set of normal and anomalous captions entirely offline with a language model, embeds them once with a frozen video-text encoder, and reuses this memory online. At inference, each segment is matched against the memory to produce both an anomaly score and a textual rationale, eliminating all online LLM calls and sustaining per-segment deadlines. Three lightweight controls improve robustness: _repulsive prompting_ separates normal and anomalous caption spaces, _scaled anomaly penalization_ corrects residual anomaly bias, and _certainty-driven runtime encoder selection_ maintains weakly-hard real-time guarantees by allocating extra compute only to difficult segments. On UCF-Crime and XD-Violence, Flashback achieves 87.7 AUC and 75.0 AP, outperforming prior zero-shot methods while providing human-readable explanations at up to 43.8 fps on a single consumer GPU. The result is the first VAD system that is simultaneously zero-shot, real-time, and explainable.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Flashback, a novel paradigm for video anomaly detection (VAD) designed to be simultaneously zero-shot, real-time, and explainable. It reframes VAD as a retrieval task by constructing a large "pseudo-scene memory" entirely offline using an LLM to generate text captions . At inference, it avoids all online LLM calls, simply matching video segments to this memory to produce an anomaly score and a textual rationale . The method's robustness is improved by controls like repulsive prompting (RP), scaled anomaly penalization (SAP), and a certainty-driven runtime encoder selection . Flashback achieves state-of-the-art zero-shot performance on UCF-Crime (87.7 AUC) and XD-Violence (75.0 AP) while operating at high throughput.

### Strengths
1.Novel Unification of Critical VAD Properties. The paper's primary strength is its innovative system design that successfully unifies three properties critical for real-world deployment: zero-shot generalization, real-time inference, and explainability . This is achieved by cleverly reframing VAD as a retrieval task, moving all computationally expensive LLM-based reasoning to an "offline recall" stage, thus eliminating online LLM calls and solving the speed-reasoning trade-off.

2.State-of-the-Art Empirical Performance. Flashback achieves outstanding zero-shot results, significantly outperforming prior work like LAVAD on both UCF-Crime and XD-Violence, and even surpassing many supervised methods. This strong accuracy is paired with exceptional real-time throughput (up to 43.8 fps) and backed by quantitative evidence that its retrieved captions are semantically meaningful explanations.

3.Thorough Component Validation and Ablation. The paper provides a comprehensive set of ablation studies in Section 4.5 that strongly justify the novel design choices. The necessity of Repulsive Prompting (RP) is clearly demonstrated both quantitatively and qualitatively. Furthermore, the study validates the system's scalability with memory size and its robustness to different random subsets of the pseudo-caption memory, inspiring confidence in the method's stability.

### Weaknesses
1.The paper's impressive results are tightly coupled with large-scale, proprietary models, raising significant reproducibility concerns. The 1M-entry pseudo-scene memory was generated using the proprietary gpt-4o-2024-08-06 , which incurs substantial cost ($181.43) and compute time (76 hours). Furthermore, all embeddings rely on the PerceptionEncoder. The paper provides no ablation to show how the Flashback paradigm itself would perform with a standard, open-source LLM or a common frozen encoder like CLIP. This makes it difficult to disentangle the contribution of the novel retrieval method from the power of its backbone models.

2.The certainty-driven runtime encoder selection (Sec 3.4) presents several issues. First, its mathematical formulation in the main paper is overly dense and lacks intuition, with all critical derivations and justifications deferred to Appendix C, hindering clarity. Second, this mechanism introduces new, sensitive hyperparameters ($Q, R, \tau$), yet the paper provides no sensitivity analysis to show how they were chosen or how robust the system is to their variation. Finally, the paper asserts the Kalman-based likelihood is effective but fails to compare it against simpler uncertainty heuristics (e.g., the entropy of the retrieved label distribution) to prove that this complex state-space model is justified.

3.The paper's definition of "explanation" is ambiguous. The anomaly score $A_s$ is calculated by a weighted average of the anomaly flags from the Top-K (K=10) retrieved captions. However, the qualitative examples (e.g., Fig 7, 8) present this entire list of K captions as the human-readable rationale. This is problematic, as the list may contains a confusing mix of both normal and anomalous descriptions for the same segment. It is unclear if the intended rationale for an operator is just the Top-1 caption or this entire, potentially conflicting, list.

### Questions
1.Regarding the Runtime Encoder Selection: (a) Can the authors provide a more intuitive explanation for choosing a Kalman filter over simpler uncertainty heuristics (e.g., entropy of the Top-K label weights)? (b) How sensitive is the performance of FlashbackX to the choice of the $Q, R, \tau$ hyperparameters? (c) Could you provide a direct comparison (in both accuracy and overhead) to using a simpler metric, like the entropy of $\{w_{s,k}\}$, as the certainty score?

2.Regarding Model Dependency: To what extent is the SOTA performance bound to the specific PerceptionEncoder and gpt-4o? Have the authors experimented with using a standard CLIP (e.g., ViT-L/14) encoder and an open-source LLM (e.g., Llama 3) for memory generation? This ablation is critical for understanding the generality of the Flashback paradigm.

3.Regarding Explainability: Could the authors please clarify what the intended "explanation" for a human operator is? Is it (a) the Top-1 retrieved caption, or (b) the full Top-K list? If it is (b), how does the system recommend handling the common case where this list contains contradictory (both normal and anomalous) descriptions for a segment?

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
This paper introduces Flashback, a novel paradigm for Video Anomaly Detection (VAD) that reframes the task as retrieval from a large, offline, text-only pseudo-scene memory generated by an LLM. By eliminating online LLM calls, Flashback achieves simultaneous zero-shot, real-time, and explainable anomaly detection. It outperforms prior zero-shot methods on UCF-Crime and XD-Violence and incorporates three lightweight controls—Repulsive Prompting, Scaled Anomaly Penalization, and runtime encoder selection—to enhance robustness and meet real-time constraints.

### Strengths
1. **Novel and Practical Paradigm**: The major strength is its core idea of redefining VAD as a retrieval task over an offline text memory generated by an LLM. This is not only conceptually elegant but also highly practical as it directly addresses the bottleneck of online inference with VLM/LLM.
2. **Excellent Real-Time Performance**: The paper makes a strong commitment to "real-time" and shows high throughput (e.g., 43.8 fps).

### Weaknesses
1. Ambiguity in Zero-Shot Definition:
The method is claimed to be "strictly domain-agnostic," yet the use of domain-specific context prompts (e.g., "university campus" for ShanghaiTech) during memory construction implies reliance on target-domain knowledge. This conflicts with the standard zero-shot assumption and may limit true plug-and-play applicability in completely unseen environments.

2. Dependence on LLM Knowledge and Coverage:
Flashback can only detect anomalies that are pre-generated in the pseudo-scene memory. Anomalies outside the LLM's knowledge or imagination—especially novel, rare, or domain-specific events—will be missed, leading to false negatives and limited generalization.

3. Scaled Anomaly Penalization (SAP) May Not Generalize:
The scale factor α = 0.95 was tuned on UCF and XD-Violence. It is unclear whether this value generalizes to other domains (e.g., daily-life anomalies in ShanghaiTech). This raises concerns about the need for per-domain tuning, undermining the zero-shot claim.

4. Explainability May Be Noisy or Misleading:
As shown in Figure 7, some retrieved captions are irrelevant or inconsistent with the video content. Aggregating top-K captions without summarization or ranking can lead to confusing or redundant explanations, reducing the practical utility of the interpretability feature.

5. Limited Validation of Repulsive Prompting (RP):
The RP ablation is supported by only one qualitative example. More cases are needed to convincingly demonstrate its necessity and effectiveness across diverse anomaly types and domains.

6. Incomplete Handling of Label Ambiguity:
In Figure 4(c), a detection is dismissed as a "label mismatch," but no comparison with other methods (e.g., LAVAD or weakly-supervised baselines) is provided to contextualize this failure. This weakens the claim of superior robustness.

### Questions
See weakness above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces Flashback, a model that constructs a set of normal and abnormal captions offline with the support of a LLM.

The proposed model is able to produce both an anomaly score and a textual rationale for video anomaly detection.

They conduct evaluations on two benchmark datasets such as UCF-Crime and XD-Violence, and show improved performance.

### Strengths
+ The paper is technical sound.

+ The proposed model shows improved performance on both UCF-Crime and XD-Violence.

+ Some interesting visualisations such as Fig 4.

### Weaknesses
- The review of existing works tend to be limited. What are the current challenges in this area, why existing methods are unable to address these issues, and how the proposed model handles these challenges are unclear. Although some of the insights are provided in the last few sentences per paragraph of the related work, it could be more clearly presented.

- The method section is overall clearly written. It would be better to have a notation section detailing the maths symbols and operations used in the paper, such as what are scalars, vectors and matrices etc.

- A good paper should use enough figures to show the network design, modules and blocks, currently in this paper, only one figure is provided (fig. 2), and the module details, their setups and arrangements, particularly the only and output dimensions are unclear to reviewer. What are the core innovations compared to existing works that use eg prompts and templates with LLM/VLM? The discussions and comparisons regarding these are limited. How to position this work clearly in the current literature?

- Most experimental results are presented in the form of tables, no any visualisations on attentions, plots to show eg the hyperparameters evaluations etc, and the comparisons to closely related SOTA models. These limit the impact of this work, and the current evaluations tend to be potentially biased, as only two datasets are being used. Fig. 3 only shows with and without the use of RP, and it is not being compared with eg existing SOTA methods. 

- The datasets used in evaluations tend to be small-scale and a bit old fashioned. The authors should try to explore new and more challenging datasets in evaluations such as [A]. How the proposed model handles, eg, scenario-level and anomaly-type-level detection tasks?

[A] L Zhu, L Wang, A Raj, T Gedeon, and C Chen. Advancing Video Anomaly Detection: A Concise Review and a New Dataset. Advances in Neural Information Processing Systems (NeurIPS). 2024.

- Ablation studies are lengthy in texts, but the core discussions and analysis such as section 4.2, 4.3 and 4.6 tend to be very limited. The paper needs to be revised to reflect on how the proposed model is robust, efficient and effective on diverse scenarios and in handling different anomaly types.

### Questions
Refer to weaknesses.

- It is suggested not to heavily use “—” as this looks like machine generated contents/patterns.

- Limitations and future research directions could be included in the paper.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Flashback, a memory-driven approach for zero-shot, real-time video anomaly detection (VAD). By reformulating VAD as a retrieval task over a pre-generated text-only memory, the method eliminates online LLM calls and achieves strong performance on UCF-Crime and XD-Violence while providing textual explanations.

### Strengths
1. This paper proposes a novel and practical framework that effectively unifies zero-shot capability, real-time inference, and explainability.

2. The proposed model achieves SOTA zero-shot accuracy, outperforming prior works significantly, with high throughput (up to 43.8 fps).

3. The ablation studies convincingly validate key components such as repulsive prompting and memory scaling.

### Weaknesses
1. The whole method heavily relies on proprietary models (GPT-4o, PerceptionEncoder) without ablation using open-source alternatives (e.g., CLIP, LLaMA), raising reproducibility concerns.

2. The runtime encoder selection mechanism is complex and poorly motivated; no comparison with simpler uncertainty metrics (e.g., entropy) is provided.

3. Ambiguity in the definition of “explanation”—whether it is the top-1 caption or the full top-K list, and how conflicting captions are handled.

### Questions
1. How was the Kalman filter chosen for uncertainty estimation? Have you compared it with simpler metrics like entropy?

2. To what extent does performance depend on the backbone models? Are results reproducible with open-source alternatives?

3. What is the final explanation presented to users? If it is the top-K list, how should operators interpret contradictory captions?

### Soundness
3

### Presentation
3

### Contribution
2
