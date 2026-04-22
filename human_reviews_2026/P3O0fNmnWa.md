# Towards One-step Causal Video Generation via Adversarial Self-Distillation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Recent hybrid video generation models combine autoregressive temporal dynamics with diffusion-based spatial denoising, but their sequential, iterative nature leads to error accumulation and long inference times. In this work, we propose a distillation-based framework for efficient causal video generation that enables high-quality synthesis with extreme limited denoising steps. Our approach builds upon Distribution Matching Distillation (DMD) framework and proposes a novel form of Adversarial Self-Distillation (ASD) strategy, which aligns the outputs of the student model's $n$-step denoising process with its $(n+1)$-step version in the distribution level. This design provides smoother supervision by bridging small intra-student gaps and more informative guidance by combining teacher knowledge with locally consistent student behavior, substantially improving training stability and generation quality in extremely few-step scenarios. In addition, we present a First-Frame Enhancement (FFE) strategy, which allocates more denoising steps to the initial frames to mitigate error propagation while applying larger skipping steps to later frames. Extensive experiments on VBench demonstrate that our method surpasses state-of-the-art approaches in both one-step and two-step video generation. Notably, our framework produces a single distilled model that flexibly supports multiple inference-step settings, eliminating the need for repeated re-distillation and enabling efficient, high-quality video synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a method for the distillation of bidirectional video contraption models into autoregressive causal few-steps generators capable of real time inference, basing their work on the popular Self Forcing framework. The paper introduces two advancements with respect to Self Forcing: 1) it enables performing inference with a variable number of sampling steps (1, 2, 4); 2); It improves performance for the case of 2 and 1 step inference with respect to Self Forcing, producing 2 step generation results on par with the original 4 step inference according to VBench scores. The technical contribution enabling these advancements consist of: 1) an adversarial loss term matching N+1 with N step generator distributions; 2) forcing the first frame to be inferred using 4 steps. Both contributions appear sound and easy to implement (authors provide code in the supplement). The quality of the results in convincing as seen qualitatively and in VBench and user studies. Overall, the work is likely to be adopted by the community due to its simplicity, convincing quality of the results and relevance of the problem.

### Strengths
- The work tackles the task of making slow bidirectional video models, fast real-time autoregressive generators which if of high practical importance
- The quality of results is convincing as seen in the provided qualitative samples
- Quantitative evaluation confirms qualitative assessment that the method matches or surpasses Self Forcing under the same amount of sampling steps
- The method is simple to implement, and authors provide the source code for review
- Ablation studies show convincingly that ADS and FFE are both having a positive impact on the method and ablations on the optimal value for adversarial loss weighting are shown

### Weaknesses
- Tables are missing an analysis of first frame latency and throughput (see Self Forcing). I suggest the authors to report these numbers. FFE will cause first frame latency to match the original 4 steps Self Forcing
- The paper considers only the setting where chunks of 3 latent frames are predicted simultaneously and does not show results for frame-by-frame autoregressive generation. Frame-by-frame prediction is a setting of high practical importance as it minimizes latency. Evaluation would be strengthened by showing qualitatives and quantitative results for this setting too.
- Evaluation is performed on the 5 second setting and no results are shown beyond it. Self Forcing can generalize beyond 5s generation, an important capability for an autoregressive causal generator. Such capability should be demonstrated and evaluated. Without this capability, practical significance of the method would be reduced.
- The paper is unclear in some key parts as Algorithm 1 and discriminator design. See questions.
- Some typos and missing spaces before citations

### Questions
- An adversarial term is introduced to match distribution between N+1 and N step predictions, showing performance improvement. Did the authors consider extending use of the same adversarial term to its canonical usage for matching the real data distribution with the 4 step generator distribution similarly to DMDv2?
- Adversarial losses are proposed as the tool for matching the N+1 and N step generator distributions. Why usage of an adversarial term is the ideal choice in this context? Could we have used a DMD formulation instead by introducing additional fake score prediction networks, either one for each value of n, or sharing the same fake score prediction network with conditioning on n? This could result in a more elegant framework only relying on DMD.
- FFE relies on the assumption that generation of the first frame is the hardest, because successive frames can be generated by copying content from the first frame. Thus allocating more sampling steps to the first and less to the subsequent ones makes sense and is shown to improve performance. The assumption however holds less strongly if videos with high camera motion or complex object movements are considered. In this setting each frame will need to generate a more significant amount of content without possibility for copying it from previous frames. Can the authors show that in this setting, their method with 2 steps inference is still matching performance of the original 4 steps Self Forcing?


- Algorithm 1: LL217 shows that a schedule with a fixed number of steps N is instantiated. LL222 suggests that the actual number of sampling steps for the current iteration n is sampled. I believe LL217 should instead instantiate a different schedule for each possible value of n
- Algorithm 1: ll224 suggest a rectified flow setting. I suggest making this explicit in the paper
- Algorithm 1: LL224 LL226 suggest two different time steps are sampled for x^1 and x^2 for use in the adversarial loss term. I'd like to confirm this understanding is correct. Could the authors discuss why this is preferable to having a shared timestep t to ease the role of the discriminator?
- LL352-355 are unclear. How is the discriminator implemented? Does the discriminator receive as input the current timestep t in addition to backbone features? D_n and D^n seem to be used interchangeably
- Eq 2 has incorrect parenthesis
- Could authors report all VBench evaluation metrics in the supplement?
- How is Fig 1 produced? Did the authors perform experimentation on a gaussian mixture?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenges of slow inference speed and error accumulation in causal video generation models. The primary goal is to enable high-quality video synthesis in extremely few denoising steps, the authors propose two main contributions:
1. Adversarial Self-Distillation (ASD): A novel strategy that moves away from traditional distillation. Instead of matching a few-step student model to a multi-step teacher model, ASD adversarially aligns the output distribution of the model's own n-step generation with its (n+1)-step generation.
2. First-Frame Enhancement (FFE): An inference strategy designed to mitigate error propagation. Based on the observation that the initial frame is most critical in causal generation, FFE allocates more denoising steps to the first frame and significantly fewer steps to subsequent ones, enhancing overall video quality with minimal computational overhead.

### Strengths
1. Well-Motivated and Significant Problem: The paper addresses a highly relevant and challenging problem in generative AI: achieving high-fidelity video synthesis under extreme computational constraints (one or two inference steps). The motivation is clear, and a successful solution to this problem would have a significant practical impact, making the research direction valuable.
2. Empirically Effective and Intuitive Core Ideas:
a) ASD's Practical Efficacy: The core idea of Adversarial Self-Distillation (ASD)—aligning the model's n-step output with its (n+1)-step counterpart—is an intuitive approach to breaking down a large distillation gap. While its theoretical underpinnings could be further explored, its practical effectiveness is undeniably demonstrated by the experiments. The concept of progressively refining the model based on its own slightly improved outputs proves to be a powerful empirical strategy in the few-step regime.
b) Pragmatic and Data-Driven FFE: The First-Frame Enhancement (FFE) strategy is a pragmatic and effective solution grounded in a clear empirical observation (Figure 4). Although a simple heuristic, it demonstrates a thoughtful consideration of the error propagation dynamics in causal models. This data-driven approach to allocating computational resources where they are most needed is a clever and impactful inference-time optimization.
3. Rigorous and Comprehensive Experimental Validation: 
a) Strong and Fair Baseline Construction: The authors' decision to train their own few-step versions of a powerful SOTA model (Self-Forcing) for comparison is a sign of rigorous scientific practice. This "apples-to-apples" comparison effectively isolates the contribution of their proposed methods (ASD and FFE) from confounding variables like model architecture, which makes the reported gains highly credible.
b) State-of-the-Art Empirical Performance: The paper presents compelling quantitative and qualitative results that convincingly demonstrate state-of-the-art performance in the challenging one- and two-step video generation tasks. The significant lead over a fairly-trained baseline, supported by extensive ablation studies (Table 2) and a strong user preference study (Figure 6), provides undeniable proof of the method's empirical superiority.
4. Clarity and High-Quality Presentation: The paper is well-written, clearly structured, and easy to follow. The figures and tables are informative and effectively communicate the core concepts and results, contributing to a high-quality presentation of the work.

### Weaknesses
1. Methodological Foundation of ASD Lacks Rigor: The central contribution, Adversarial Self-Distillation (ASD), is built on a foundation that is more intuitive than it is rigorous. The paper's core claims—that the "intra-student gap" is smaller and that adversarial alignment provides "smoother supervision"—are presented as assertions rather than demonstrated principles.
a) There is no formal analysis or empirical measurement to quantify this "gap" (e.g., in terms of a specific distribution divergence metric).
b) The claim of "smoother supervision" from a GAN objective is counter-intuitive, given the well-documented instability of adversarial training. The paper fails to provide evidence to substantiate why this would be the case, especially compared to simpler, more stable alignment objectives.
2. Unaddressed Risk of Error Reinforcement in Self-Distillation: The ASD mechanism, where the model learns from a slightly better version of itself, introduces a significant and unexamined risk of "model drift" or "error reinforcement." If the (n+1)-step generation is flawed (e.g., contains artifacts or mode collapse), ASD could perversely train the n-step model to replicate these very flaws. The paper relies on the DMD loss to anchor the model to the true data distribution but provides no analysis of the training dynamics or the delicate balance required to prevent the self-distillation objective from amplifying its own mistakes.
3. The FFE Strategy is Heuristic and Its Generalizability is Questionable: The First-Frame Enhancement (FFE) strategy is presented as a key contribution, but it is fundamentally an empirically-driven heuristic rather than a principled method.
a) Its justification rests entirely on a single observation on a specific dataset (Figure 4), and its generalizability to different video content (e.g., videos with major mid-sequence scene changes) is not explored.
b) The paper completely ignores the potential negative side-effects of this strategy. Creating a sharp drop in denoising steps between the first and second frames could introduce significant temporal discontinuity and artifacts, undermining the very quality it aims to enhance. This critical aspect is neither analyzed nor discussed.
4. Insufficient Discussion on Training Complexity and Stability: The proposed training framework is remarkably complex, involving a generator, a discriminator, and a "teaching assistant" (TA) score model trained in an alternating fashion. The paper largely overlooks the significant practical challenges this entails. There is no discussion of the training stability, the sensitivity to the delicate balance of multiple loss terms and optimizers, or the total computational overhead of this complex setup compared to simpler distillation baselines. For a paper focused on efficiency, the lack of transparency regarding its own training costs is a notable omission.

### Questions
1. The central premise of ASD is that the "intra-student gap" is smaller and easier to bridge than the "teacher-student gap." Could you provide a more formal or empirical justification for this claim? For instance, have you measured this "gap" using any distribution divergence metrics (e.g., KL, Wasserstein) to validate this core assumption?
2. Given the known training instabilities of GANs, the claim that ASD provides "smoother supervision" is counter-intuitive. Could you elaborate on this and provide evidence (e.g., loss curves, gradient norm analysis) to support that the adversarial self-distillation process is indeed more stable than direct distillation from a fixed teacher?
3. The self-distillation mechanism seems to carry an inherent risk of error reinforcement, where the model could amplify its own artifacts over time. How does the framework explicitly guard against this "model drift"? What is the role of the DMD loss in anchoring the training?
4. The training procedure appears significantly more complex than the baseline. Could you provide a more transparent comparison of the training time, computational resources, and overall stability of your method compared to the standard distillation approach used to train the Self-Forcing† baseline?

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
This paper works towards limiting the diffusion denoising steps of hybrid video generation frameworks, that leverage autoregressive models to model temporal dynamics and diffusion-based spatial denoising, to as few as one step. The proposed method falls under the category of model distillation and strongly builds on an existing method called Distribution Matching Distillation with a substantial addition: A novel form of Adversarial Self Distillation is proposed, aligning the student model’s n-step denoising process with its (n+1)-step version on a distribution level. Results on VBench and a custom user study show state of the art results on 1 and 2 step video generation. The model further removes the limitation of fixed inference steps after training, allowing flexibility for multi-step settings.

### Strengths
- The paper is generally well written and presented.
- Provided 1 and 2 step video results, both in the paper and supplementary material, show better results than current competitors.
- The method supports both single and few / multi step inference which is a major advantage over fixed step trained methods
- The observation First Frame Strategy seems to be an important observation, by itself already boosting state of the art results
- The influence of ASD and FFE cleanly ablated showing the superior results of the combination

### Weaknesses
## Incremental contribution:
- The used components are not fundamentally new in nature. DMD is very well established for model distillation and remains a core component also in this work. 
- Similar adversarial diffusion distillation has been proposed before and is well established in the community
- As a such there are no fundamentally new concepts presented, but their combination provides a nice contribution to the current state of research.

## Limited information on experiments:
- There are several information missing on some of the shown experiments
- The user study is missing the number of participants and further statistical values
- For the main comparison for 1 or 2 step distillation, the authors had to retrain Self Forcing. Retraining parameters are missing i.e. indicating that the model has been trained long enough to convergence

## Figure 1:
- Figure 1 seems not really on point, i.e. the adversarial self-destillation on the right seems oversimplified and not aligned with Algorithm 1. 

## Minor:
- Algorithm 1, 1: Typo: origianl

### Questions
- Eq. 2: isn’t there a bracket missing?
- How is Fig. 4 created? Is this just an example or averaged results? Are the results corresponding to results from Self Forcing or from the provided method?
- Regarding user preference study: Please provide more details on e.g. how many users were selected, were measures taken to ensure independence?
- In the ablation Table 2: Why are the total scores for the first row (ASD, FFE) so much worse than the corresponding pure Self Force values from Table 1?
- The distilled diffusion process is optimized with a 4-step denoising process. Was this number of steps determined to optimal?
- Influence of $\alpha$: Figure 8 already shows influence of alpha on the Total Score. However it is interesting to see that the order of the alpha values with increasing scores (for 2 or 1 step) goes with: alpha=0 < alpha=20 < alpha=10 < alpha=30, which shows a non-monotonuous increase of the total score with increasing alpha. How do you explain this behavior?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a framework for accelerating causal video diffusion models via Adversarial Self-Distillation (ASD) and First-Frame Enhancement (FFE). The method extends Distribution Matching Distillation (DMD) by introducing a discriminator that aligns the student model’s n-step and (n+1)-step denoising distributions, instead of aligning directly with a multi-step teacher. This step-wise self-alignment aims to stabilize training under extreme few-step (1–2 step) scenarios. Additionally, FFE allocates more denoising steps to the first frame to mitigate error propagation in causal video generation. Experiments on VBench show that the proposed model surpasses Self-Forcing and CausVid under 1-step and 2-step configurations, while achieving comparable performance to multi-step baselines such as Wan2.1 and SkyReels with much fewer steps.

### Strengths
1. The paper is clearly written and easy to follow. The proposed methods (ASD and FFE) are well-motivated and clearly elaborated. 

2. The results look very impressive. Compared to the Self-Forcing baseline, the 1-step video generation exhibits a great boost in quality. The speed of 1-step causal generation will enable a wider deployment of streaming video generation.

### Weaknesses
One minor concern might be conceptual novelty. The main method of this work, ASD, is not fundamentally new [1,2]. 
Considering the value and impact of 1-step causal video generation, the engineering effort to tune an end-to-end pipeline is a significant contribution, especially that the authors provide the code for replication in the supplementary material. 


[1] Zhang et al., SF-V: Single Forward Video Generation Model. 

[2] Lin et al., Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation.

### Questions
1. The proposed FFE seems to benefit early generation more, right? Can you provide more results and comparisons for longer generations, such as 10s-20s level? 

2. In the user study, the proposed method is on par with Self-Forcing at the 4-step setting (exactly 50%). Can you elaborate more on why ASD is beneficial at 1-2 steps (as in the ablation) but not helpful in the 4-step setting?

### Soundness
4

### Presentation
4

### Contribution
4
