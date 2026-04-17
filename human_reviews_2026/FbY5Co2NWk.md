# Unmute the Patch Tokens: Rethinking Probing in Multi-Label Audio Classification

- Decision: Accept (Poster)
- Scores: 8, 2, 4

## Abstract
Although probing frozen models has become a standard evaluation paradigm, self-supervised learning in audio defaults to fine-tuning {when pursuing state-of-the-art on AudioSet}. A key reason is that global pooling creates an information bottleneck causing linear probes to misrepresent the embedding quality: The $\texttt{cls}$-token discards crucial token information about dispersed, localized events in audio. This weakness is rooted in the mismatch between the pretraining objective (globally) and the downstream task (localized). Across a comprehensive benchmark of 13 datasets and 6 spectrogram-based encoders, we investigate the global pooling bottleneck. We introduce binarized prototypical probes: a lightweight and simple pooling method that learns prototypes to perform class-wise information aggregation. Despite its simplicity, our method notably outperforms linear and attentive probing. Our work establishes probing as a competitive and efficient paradigm for evaluating audio SSL models, challenging the reliance on costly fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the underperformance and underutilization of probing as an evaluation paradigm for self-supervised audio models, particularly in multi-label classification tasks. The authors hypothesize that the standard approach of using a linear probe on a globally pooled [cls] token creates an information bottleneck, discarding crucial information about localized sound events contained within the patch tokens. To address this, they introduce "binarized prototypical probes" (protobin), a lightweight and efficient pooling method that learns a set of per-class prototypes to aggregate information directly from the full token map. Through a comprehensive benchmark, the authors demonstrate that their proposed method significantly outperforms standard linear and attentive probing. The work makes a strong case for establishing prototypical probing as a more faithful and efficient evaluation standard in the audio SSL community.

### Strengths
- Systematic and Insightful Problem Analysis: The paper's greatest strength is its methodical approach. Rather than just presenting a new method, it first provides a deep and convincing analysis of the existing problem—the "pooling bottleneck"—and validates this hypothesis with extensive empirical evidence.
- Methodological Elegance and Efficiency: The proposed protobin method is simple to understand and implement, yet highly effective. It is parameter-efficient and, through binarization, memory-efficient, making it a practical tool for a wide range of applications, including on-device scenarios.
- Comprehensive and Rigorous Benchmarking: The scale of the experimental study is a major strength. The evaluation across 13 datasets, 6 encoders (and their supervised variants), and 10 pooling methods provides a robust and generalizable foundation for the paper's claims. The structured presentation of results in response to explicit research questions is exemplary.
- Clarity of Presentation: The writing is clear, and the high-quality visualizations (especially Figures 1, 2, and 5) are instrumental in conveying the paper's core ideas and findings effectively.

### Weaknesses
- Fixed Number of Prototypes: The number of prototypes is set to 20 per class across all datasets, following a heuristic from a prior work. While this seems to work well, the paper would be slightly stronger with a brief sensitivity analysis or discussion on how this hyperparameter might be optimally chosen for datasets with different characteristics (e.g., number of classes, intra-class variance).
- Trade-off of Binarization: The protobin method is compared to a float-based proto version, and while it performs competitively and often wins, it is not uniformly superior. A brief discussion on the specific trade-offs of binarization (e.g., in which scenarios might the precision of float-based prototypes be advantageous?) could add more nuance.

### Questions
Please refer to the weaknesses above for the questions.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
## Rethinking probing for Audio SSL models

1. The paper proposes prototypical probing strategies for probing frozen Audio SSL models.
2. Several pooling approaches have been evaluated across several audio SSL models with different pretext tasks for a holistic evaluation of the proposed pooling approaches.
3. The empirical evaluation conducted in the paper is solid, and showcases that the proposed pooling approach works better than alternatives for pooling.

### Strengths
1. For the most part, the paper is well written and well presented. Diagrams and illustrations aid in understanding the paper.
2. The proposed binarised prototypical probing methods are well presented. 
3. The evaluation protocol is thorough, with a good selection of audio SSL models and several probing strategies.
4. Results show that protobin probing works well, especially on 3 of the 5 evaluated tasks.

### Weaknesses
## Weakness 1

a. The core argument/motivating principle, repeated several times in the paper, is that "audio SSL works default to fine-tuning instead of probing", but only a handful of papers that support the argument are cited throughout the paper. E.g.

Lines 52-53
> Given probing’s apparent practicality and widespread adoption as an evaluation paradigm in computer vision (Oquab et al., 2024; Przewi˛e´zlikowski et al., 2025; Darcet et al., 2025), why does the audio SSL community still default to resource-intensive fine-tuning (Alex et al., 2025; Chen et al., 2024)?  

Lines 72-73
> Therefore, the limited adoption of probing in audio SSL ....

Lines 214-215
> Evaluation in audio SSL. While probes are widely used in computer vision (Oquab et al., 2024), audio SSL defaults to fine-tuning (Rauch et al., 2025b).

Statements such as the above make it seem like the audio SSL domain is not doing probing-based evaluation on frozen encoders at all.  However, the paper only cites a handful of papers that support this notion, while ignoring the plethora of audio SSL works that evaluate frozen encoders using probing [1-15] (to mention a few). This includes Dasheng [8], the model you do cite in the paper and conduct experiments on.

---

## Weakness 2

1. Further, the core motivation/hypothesis/guiding principle of the paper is that there is a "pooling bottleneck", and that standard single vector probes and [cls]-token only pooling under utilize token embeddings. However, several missing references listed in W1 are already aware of the drawbacks of conducting probing experiments only on [cls]-token or single vector representations obtained by pooling of information across all the patches.

2. In MIM style SSL from spectrograms, every patch is a fragment in time and frequency. Thus, aggregation strategies that merge information across patches will inherently decimate frequency information. Niizumi et al. [02] were aware of this and proposed a simple strategy to aggregate temporal features while preserving frequency information.

3. Thus, the <[cls]-linear probe> strategy is not representative of the entire audio SSL community's approach to evaluate audio SSL models, unlike what the paper in its current form makes it seem like.

---

## Weakness 3

From lines 135-136
> If the model exposes a last-layer....... If not, we can mean pool $\tilde{z}_i$ by averaging all token positions.

As mentioned in Weakness 2, Point 2. This pooling strategy across token positions is suboptimal and puts plain linear probe at a disadvantage for the models that do not expose cls tokens. 

---

## Weakness 4 

In several instances, the paper makes unjustified statements.

> All current spectrogram-based audio SSL encoders apply MIM-style objectives, often coupled with student-teacher distillation (Chen et al., 2024; Alex et al., 2025).

These two papers are not representative of the entire audio SSL field. For e.g. [9, 13, 15] do not use a straightforward MIM style objective for learning audio representations through SSL.

> While probes are widely used in computer vision (Oquab et al., 2024), audio SSL defaults to fine-tuning (Rauch et al., 2025b).

How does the paper on the BIRDNET dataset (Rauch et al., 2025b) justify that audio SSL defaults to using fine-tuning?

---

## REFERENCES

01. Koutini et al., "Learning General Audio Representations With Large-Scale Training of Patchout Audio Transformers", 2022.  
02. Niizumi et al., "Masked spectrogram modeling using masked autoencoders for learning general-purpose audio representation",   2022.  
03. Anton et al., "AUDIO BARLOW TWINS: SELF-SUPERVISED AUDIO REPRESENTATION LEARNING", 2023.  
04. Niizumi et al., "BYOL for Audio: Exploring Pre-Trained General-Purpose Audio Representations", 2023.  
05. Niizumi et al., "Masked Modeling Duo: Towards a Universal Audio Pre-Training Framework", 2024.  
06. Yadav et al., "Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners", 2024.  
07. Yadav et al., "Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations", 2024.  
08. Dinkel et al., "Scaling up masked audio encoder learning for general audio classification", 2024.  
09. Li et al., "Self-Supervised Audio Teacher-Student Transformer for Both Clip-Level and Frame-Level Tasks", 2024.  
10. Yadav et al., "AxLSTMs: learning self-supervised audio representations with xLSTMs", 2025.  
11. Yuksel et al., "GRAM: Spatial general-purpose audio representation models for real-world applications", 2025.  
12. Schmid et al., "Effective Pre-Training of Audio Transformers for Sound Event Detection", 2025.  
13. Pepino et al., "EnCodecMAE: leveraging neural codecs for universal audio representation learning", 2025.  
14. Niizumi et al., "M2D-CLAP: Exploring General-Purpose Audio-Language Representations Beyond CLAP", 2025.  
15. Chang et al., "USAD: Universal Speech and Audio Representation via Distillation", 2025

### Questions
No specific questions. Please address the weaknesses stated above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper primarily investigates the limitations of the existing linear evaluation paradigm for audio encoders.  

The authors show that the use of [cls] tokens  for linear probing does not take into account the time-space localised representations contained in patch tokens, and this hurts downstream linear probing performance in audio encoders, especially given that most audio encoders employ reconstruction objectives where information is rich in the patch tokens.  

The paper runs an evaluation benchmark on a set of linear probing methods on multiple downstream tasks, evaluated over multiple pre-trained audio encoders, and show that [cls] based methods perform poorly as opposed to learned pooling methods. 

Motivated by this, the authors propose a pooling strategy (protobin) which adds to the existing method of prototype pooling by introducing binary constraints.

### Strengths
Clarity: The paper is well written and can be easily understood.  

Motivation: The paper is well motivated, and the motivation is justified by the results in Appendix A, where [cls] tokens alone perform much worse than learned pooling methods. 

Contribution 1: The paper presents a benchmark and evaluation of recent SSL methods under multiple pooling/aggregation strategies. Using this benchmark, SSL methods on audio data can be fairly evaluated without a bias towards the [cls] token, thereby nullifying the apparent advantage seen by discriminative-only based methods on linear probing performance. 

Contribution 2: The results by using the binarized prototype tensor (protobin and proto combined) yield SoTA  linear probing performance across almost all tasks and pre-trained models. Therefore, prototype-based aggregation can act as a standard method for performing linear probing for multi-label audio classification.

### Weaknesses
Significance: The fact that [cls] tokens alone perform worse than learned pooling mechanisms is already mentioned in literature [1]. Therefore, the same message being repeated for audio needs a stronger justification, with possible reasons as to why the message in [1] might or might not hold for audio data, thereby justifying the motivation for evaluation.  

Originality: The protobin aggregator is simply a binarization of the prototype tensor derived from Bird-MAE. Looking at Table 4 and 5, proto and protbin share similar overall performance, with protobin performing marginally better in more tasks than protobin (approximately 2/3), but not significantly enough to justify the addition of binarization as a contribution. The actual performance gain over other methods seems to be gained by the use of prototypes, as in Bird-MAE [2]. 

The paper does not clearly explain why the cosine similarities between prototypes are mostly zero after training, despite the absence of any explicit mechanism to control the prototype distribution. It is unclear how the \(\pm{1}\) constraint alone enforces this near-orthogonality or prevents different prototypes from converging to similar representations. 

Multi-label classification as a standalone downstream task: Although the extension towards other tasks is mentioned in future works, having multi-label classification alone as the evaluation framework does not provide a full picture on leveraging information from frozen features.  

Minor: Some of the formatting seems to be too cluttered, for example the contributions and Q sections in results. It takes away from the clarity of the paper rather than adding to it. However, I acknowledge that this is subjective. 

 

[1] Przewięźlikowski, Marcin, et al. "Beyond [cls]: Exploring the true potential of Masked Image Modeling representations." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025. 

[2] Rauch, Lukas, et al. "Can Masked Autoencoders Also Listen to Birds?." arXiv preprint arXiv:2504.12880 (2025).

### Questions
Control of prototype distributions: 
How are the cosine similarities of prototypes mostly zero after training (Figure 4) if there is no explicit method to control the distribution?
What is preventing two prototypes from learning the same thing? More specifically, why does the \(\pm{1}\) constraint force near-0 similarity? It is not clear in the paper, and how this prototype-to-class contributions emerge is not explained in the results.  

Why exactly does the binarization work better in some tasks and not in others?

### Soundness
3

### Presentation
2

### Contribution
2
