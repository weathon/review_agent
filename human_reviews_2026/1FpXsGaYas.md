# Improving Dynamic Object Interactions in Text-to-Video Generation with AI Feedback

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Large text-to-video models hold immense potential for a wide range of downstream applications. However, they struggle to accurately depict dynamic object interactions, often resulting in unrealistic movements and frequent violations of real-world physics. One solution inspired by large language models is to align generated outputs with desired outcomes using external feedback. In this work, we investigate the use of feedback to enhance the quality of object dynamics in text-to-video models. We aim to answer a critical question: what types of feedback, paired with which specific self-improvement algorithms, can most effectively overcome movement misalignment and realistic object interactions? We first point out that offline RL-finetuning algorithms for text-to-video models can be equivalent as derived from a unified probabilistic objective. This perspective highlights that there is no algorithmically dominant method in principle; rather, we should care about the property of reward and data. While human feedback is less scalable, vision-language models could notice the video scenes as humans do. We then propose leveraging vision-language models to provide perceptual feedback specifically tailored to object dynamics in videos. Compared to popular video quality metrics measuring alignment or dynamics, the experiments demonstrate that our approach with binary AI feedback drives the most significant improvements in the quality of interaction scenes in video, as confirmed by AI, human, and quality metric evaluations. Notably, we observe substantial gains when using signals from vision language models, particularly in scenarios involving complex interactions between multiple objects and realistic depictions of objects falling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper utilizes AI feedback to enhance object interaction dynamics in text-to-video models.  The first part of the paper establishes how RWR and DPO can be derived from a unified objective.  The second part of the paper compares different metrics and reward algorithms for finetuning a video model for object interactions in an offline manner, settling on AI feedback from internet-scale pretrained vision language models.

### Strengths
The papers strengths include its derivation of a unified objective, which was interesting.

### Weaknesses
This reviewer pushes back on the assertion that just because existing algorithms can be derived from a unified objective, that there is no algorithmic superiority.  There are multiple works in our field that seek to unify methods, but certain practical implementations make all the difference.  Unfortunately this comes with being in an empirical field, working with black-box models.  Some noted examples are the connections between flow models, diffusion models, and hierarchical VAEs - whereas we can derive diffusion models from hierarchical VAEs with certain assumptions (e.g. that encodings are Gaussian transitions with no learnable parameters) [1] , it is precisely these assumptions that end up making the method work empirically.  Similarly, flow models can be converted to diffusion models and have been shown to be "equivalent"[2, 3]; even within diffusion models although there may be mathematically equivalent training objectives (predicting clean sample, predicting source noise, predicting score, predicting velocity) the actual practical implementation choices result in performance differences.  Whereas it is interesting and useful that a unified objective can flexibly be turned into RWR and DPO, the design choices that differentiate them are not negligible, as the authors claim.

In fact, the structure of the paper is a bit strange.  The first half of the paper seeks to justify why the difference between DPO and RWR is negligible, leading the reader to believe that evaluating on only one choice is necessary.  The rest of the paper not only still evaluates both DPO and RWR separately, but actually does indeed show differences between the two ("DPO often performs slightly better than RWR" [Line 358], "Comparing RWR and DPO, ..." [Lines 373-376], Section 4.5 comparing when to use RWR versus DPO).  Intuitively, if the first part of the paper is to believe, there would be no such need for a practical comparison between RWR and DPO.

Firstly, the Supervised Finetuning baseline is not clearly explained - leaving the reader to guess how it is designed and implemented.  From this reviewer's assumption of the SFT implementation, the comparison does not seem particularly fair, in that the SFT will finetune on incorrect as well as correct generations without discrimination.  It is well expected that such a direct finetuning approach on self-generated data will eventually cause generative models to "go Mad" [4].

The authors should also compare against filtered finetuning rather than SFT; where rather than finetuning on all self-generated samples indiscriminately, the model is only finetuned on samples deemed successful by the same VLM used in other runs for fair comparison.

[1] Luo et al., Understanding Diffusion Models: A Unified Perspective, 2022.

[2] Gao et al., Diffusion Models and Gaussian Flow Matching: Two Sides of the Same Coin, 2025.

[3] Albergo and Vanden-Eijnden, Building Normalizing Flows with Stochastic Interpolants, ICLR 2023.

[4] Alemohammad et al., Self-Consuming Generative Models Go MAD, ICLR 2024.

### Questions
It is unclear to this reviewer why the authors only focus on offline RL finetuning techniques.  Intuitively, this loop should support learning from online experience.

Separately, apart from the reward algorithm and the metrics used, there are a variety of techniques to actually perform the update.  The authors should compare with ones that treat the finetuning process as updating an internal MDP [1], or with a residual network [2], or other attempts listed in the related work.

[1] Black et al., Training Diffusion Models with Reinforcement Learning, ICLR 2024.

[2] Ankile et al., Residual Off-Policy RL for Finetuning Behavior Cloning Policies, 2025.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to improve the physical realism in text-to-video generation. It involves fine-tuning a pre-trained video diffusion model based on Reinforcement Learning (RL) with a binary AI feedback signal. The authors compare this reward against other metric-based rewards and claim that AI feedback improves in various evaluations.

### Strengths
1. The paper tackles the lack of realistic dynamic object interactions, which is a critical failure of current video models.

2. The authors show that VLM-based AI feedback signal can be effective than frame-wise reward metrics like CLIP score or HPSv2, and demonstrate continuous improvement through RWR-based fine-tuning iteratively.

3. The experiments use a more diverse set of prompts (120 for training) compared to some prior works in RL fine-tuning for diffusion models.

### Weaknesses
1. The experimental setup, based on a 3D-Unet trained solely on Something-Something-V2, may not be fully representative raising concerns about generalizability. While the authors provide an evaluation of modern models (Fig 5), they conclude these models lack the prior ability for this task and do not apply RL fine-tuning to them. This makes it difficult to assess the broader generalizability of the proposed AIF method and limits the claims of its effectiveness.

2. The theoretical unification in Sec 3.1 appears to hinge on a significant simplification in Eq. 5, where an exponential transform exp$(\beta^{-1}r)$ is approximated as an identity mapping $r$. The paper would be strengthened by providing a more rigorous justification for this assumption.

3. It is unclear how the preference pairs required for DPO training were constructed from the point-wise Accept / Reject VLM labels.

4. The support for the claim that VLMs simulate human judgment (L284-285) could be stronger. The provided evidence (a VLM's ability to prefer true videos over generated ones) does not necessarily demonstrate that the VLM can distinguish between plausible and implausible generated dynamics, which is the core of the task.

5. The reward shaping for baseline metrics in Sec 3.2 uses scaling ($\eta$) and shifting ($\gamma$) factors that are selected heuristically.

### Questions
1. Why did you choose a binary signal over a more granular reward from the VLM (e.g., a 1-5 score or a critique)?

### Soundness
1

### Presentation
1

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
This paper addresses the challenge of unrealistic object dynamics in large text-to-video (T2V) models. The authors investigate how to improve T2V generation quality through external feedback. The paper first presents a theoretical perspective, positing that various offline RL fine-tuning algorithms can be unified under a single probabilistic objective. Based on this insight, the authors propose leveraging advanced Vision-Language Models (VLMs) to provide perceptual, binary feedback on the quality of object dynamics. The experimental results reported in the paper indicate that this VLM-based feedback approach yields significant improvements in video quality, particularly for complex interaction scenes involving multiple objects. These improvements are validated through a combination of AI-based, human, and metric-based evaluations, demonstrating the method's superiority over feedback derived from standard video quality metrics.

### Strengths
1. The manuscript is well organized and easy to follow.
2. The motivation is clear: to study what feedback signals are viable and effective for text-to-video models.
3. The experimental evaluation is comprehensive.
4. The topic is meaningful. As the paper notes, current SOTA models (e.g., VideoCrafter, CogVideo-X, Wan2.1 1.3B) struggle with generating videos with complex interactions, and similar limitations likely persist even for Sora and Veo. This supports the necessity of RL-based fine-tuning.

### Weaknesses
1. I am concerned that the contributions may be limited and not enough. The paper appears to introduce neither a new algorithmic design nor particularly promising empirical phenomena, analyses, or conclusions.
    - Using VLMs to provide feedback is common in prior work and is no longer novel. The paper does not propose a new algorithm; instead, it explores combinations of existing RL methods and feedback sources, which reads more as engineering exploration than a research advance.
    - The primary conclusion that rewards from advanced VLMs outperform metric-based rewards is sound, but it is also intuitive and not particularly surprising. Metric-based rewards have inherent limitations, and advanced VLMs trained on large-scale data, being closer to human preferences, are naturally better reward models, leading to improved performance. Consequently, the analysis does not appear to offer substantially new insights. However, I remain open to further discussion on this point.
2. The proposed approach is built on DDPM, while Flow Matching is now a prevalent training paradigm. It is unclear whether the proposed objective and the conclusions drawn can be generalized to Flow Matching models.
3. Many recent works adopt GRPO. Does GRPO fall under the proposed unified RL objective? Empirically, would GRPO outperform RWR and DPO in the studied setting?
4. For complex interactions involving compositions of multiple actions, can advanced VLMs reliably and accurately serve as reward models?

### Questions
See Weaknesses

### Soundness
3

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
3

### Summary
This paper proposes to improve the physical realism of dynamic object interactions in text-to-video models using reinforcement learning from AI feedback (RLAIF). The core idea is to leverage a large vision-language model (VLM) as a preference oracle to finetune a video diffusion model.
The method is evaluated on five challenging dynamic scenarios (Object Removal, Multiple Objects, Deformable Objects, Directional Movement, Falling Down). Experiments show clear improvements: AI feedback–guided models generate more physically plausible and coherent videos, increasing human preference scores by 11–15%.

### Strengths
1. Concentrates on dynamic object interactions, a difficult and underexplored subproblem, and introduces a standardized 5-category evaluation protocol.
2. Demonstrates that VLM feedback significantly outperforms traditional metrics (CLIP, HPSv2, optical flow) in both human and AI preference alignment.

### Weaknesses
1. Limited Conceptual Novelty: The core idea, using AI-generated feedback for RL-based alignment, is a well-established method. Porting this RLAIF framework to video by replacing unimodal reward models with a VLM is a predictable and incremental step. The paper does not identify or solve any non-trivial challenges inherent to this adaptation that would constitute a novel contribution. 
2. Lack of Algorithmic Contribution: The methodology relies entirely on existing RL algorithms (RWR and DPO). The theoretical section provides a useful re-interpretation of these methods under a unified lens but does not introduce any new algorithms, optimization techniques, or theoretical insights into preference learning itself. The contribution is purely empirical, applying off-the-shelf components to a new problem.
3. Results are Heavily Contingent on a Specific Baseline: The experiments are conducted on a self-trained 3B parameter model. Is this method genuinely advancing the state-of-the-art, or is it merely patching the deficiencies of a relatively weak baseline? The significance of the findings is questionable without evidence that the method provides meaningful improvements on stronger, more capable foundation models.

### Questions
- Beyond simply applying an existing framework, what are the non-trivial, video-specific challenges you identified and solved that make this work more than an incremental application?
- Can you provide any evidence that the method can generalize across different models? (or different model size)
- It would be better to provide analysis of the VLM's failure modes as a physical commonsense judge. A deeper investigation into where the VLM's preferences diverge from true physical plausibility would significantly increase the paper's contribution.

### Soundness
2

### Presentation
3

### Contribution
2
