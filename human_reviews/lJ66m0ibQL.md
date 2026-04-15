# Diversity-Rewarded CFG Distillation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Generative models are transforming creative domains such as music generation, with inference-time strategies like Classifier-Free Guidance (CFG) playing a crucial role. However, CFG doubles inference cost while limiting originality and diversity across generated contents. In this paper, we introduce diversity-rewarded CFG distillation, a novel finetuning procedure that distills the strengths of CFG while addressing its limitations. Our approach optimises two training objectives:  (1) a distillation objective, encouraging the model alone (without CFG) to imitate the CFG-augmented predictions, and (2) an RL objective with a diversity reward, promoting the generation of diverse outputs for a given prompt. By finetuning, we learn model weights with the ability to generate high-quality and diverse outputs, without any inference overhead. This also unlocks the potential of weight-based model merging strategies: by interpolating between the weights of two models (the first focusing on quality, the second on diversity), we can control the quality-diversity trade-off at deployment time, and even further boost performance. We conduct extensive experiments on the MusicLM text-to-music generative model, where our approach surpasses CFG in terms of quality-diversity Pareto optimality.  According to human evaluators, our finetuned-then-merged model generates samples with higher quality-diversity than the base model augmented with CFG. Explore our generations at https://musicdiversity.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a distillation based method to improve text to music models (specifically MusicLM and MusicRL). The student network learns the CFG-modified logits of the teacher model, and therefore running the teacher model with CFG is approximated with running the student network directly, which reduces inference compute to half. To address less diversity caused by CFG and by the CFG distillation, the paper proposes to maximize the diversity computed from inter-similarities during CFG distillation. The paper also finds directly interpolating two model weights computed from different $\beta$ leads to better quality-diversity tradeoff.

### Strengths
The paper is extremely well written and the motivation is very clear. It aims to solve two very important challenges in text-to-music generation: inference overhead and bad quality-diversity tradeoff. The proposed method makes a concrete understanding towards these challenges.

The methods including CFG distillation, diversity maximization, and the use of model weight interpolation are straighforward and simple enough that these ideas could be potentially easily and widely adapted to the deep generative model community. 

The experimental results show concrete improvements on the quality-diversity tradeoff compared to the base model with or without CFG.

### Weaknesses
The major weakness of the CFG distillation is that it destroys the student model's ability to take different CFG scale as input, which is key in many downstream applications. I understand that this could be useful in applications where the inference strategy (i.e. CFG) is fixed by nature, but in many cases we need the flexibility to take different CFG scales at inference time. Therefore, I would expect the student model to take the CFG $\gamma$ as inputs as well, which could be done in a similar way how diffusion models are conditioned on timesteps.

The second weakness is that in Figure 5, the proposed model does not outperform CFG significantly. The quality is about the same, and the diversity is 57-vs-43. In addition, the diversity is much worse than the base model after model weight merging, indicating that the merged model still suffers from lack of diversity. 

The third weakness is that the paper does not evaluate the proposed distillation method on other baseline models and tasks (e.g. audio generation and speech generation) -- especially open-sourced models -- which limits the contribution of the paper. As a result, it is unclear how general the proposed method is when applied in other data and tasks. 

Minor weaknesses:

One minor weakness is that the paper does not analyze in depth the diversity drop of the vanilla CFG distillation. From Fig 2 one could infer it is most likely because the student model learns the CFG-modified logits well, but it is also possible that the KL loss in eq (2) amplifies it. It would be useful to study different divergence losses and different model capacities.

Another minor weakness is that Fig 3 is almost impossible to read. Please improve the display of this figure.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors introduce a novel finetuning recipe for increasing the runtime efficiency of the model and improving diversity at the same time, through Diversity-rewarded CFG distillation and model merging. The authors show that by distilling the CFG behavior into the model with an RL-based diversity regularizer and then merging high diversity with high-quality model finetunes, they can achieve strong diversity and quality tradeoffs and improved results over the base model.

### Strengths
- Proposed CFG distillation method seems strong and is reasonably novel in the context of autoregressive LM models
- As a whole, proposed method is exceedingly clear, and while simple, shows the strengths of each facet of their 3-pronged method articulately.
- The significant time dedicated to discussing results and ablations is particular welcome, and the authors do a great job of exploring the design space of their method.
- The insight from Fig. 4 is in particular quite fascinating, do you have any idea why this merging behavior actually improves quality beyond the interpolating models?

### Weaknesses
Overall, while I think this is a reasonably strong work, there are a number of concerns I have addressed below that shift my overall recommendation to reject. In fixing some or all of these issues (in particular improving the discussion of CFG distillation novelty, using publicly available eval metrics such as FAD/CLAP/Recall/Coverage, including human eval error bars / statistical tests, and focusing more on the diversity model and its calibration in the main text) I would raise my score accordingly.

# Overclaiming of CFG Distillation Novelty:
In general, while the core contribution of the CFG distillation approach is interesting, I think it’s stated novelty is an over claim (such as the line 492-493 “In contrast, we are the first to distill CFG…for music,” which is false). While CFG Distillation may be novel in the case of autoregressive audio generation (MusicLM, MusicGen), CFG distillation is a common practice in diffusion models for the same reasons motivated in this paper (namely, reducing inference cost), normally coupled with other distillation techniques. While the authors rightly mention Meng et al., this work is rather old and the stated novelty of their “two-stage” approach vs. the authors’ “single-stage” approach (461-462) does not really hold, as most works after Meng et al. (and before the present work) use a single stage approach (and in general, the paper is presented as if the whole of CFG distillation is a fully novel contribution). Single-stage CFG distillation has been a standard practice in image domain distillation [1, 2], but more saliently has appeared in a number of audio (including music) generation [3,4] and music-specific generation [5] distillation works. As an actionable change, it would be useful to explicitly mention all of these connections in the paper (in both the introduction, the related works, and the section on CFG distillation), and make clear that the novelty of the proposed method comes from how they achieve CFG distillation (and the first to do so on autoregressive LM-based music models), rather than the *idea* of CFG distillation for music generation.


# Reduced Discussion on Diversity Model:
- It is odd that perhaps the most novel portion of the paper of the RL Diversity Objective (as both CFG distillation and model merging are standard practices) is relegated to most details of the core embedding model being in the appendix. To me, this is a core part of the paper (as defining diversity through some model underpins the entire method, and is very interesting), and it would be useful to include this in the main body of work (and I think there is a good deal of text in the results section that is redundant and could be condensed). 

# Issues with Evaluation Suite:
- While the authors make no mention of open sourcing model weights or code (which in my opinion is fine and shouldn’t be counted as a negative against them), the overall evaluation suite **also** uses fully closed-source models, severely limiting the reproducibility of the work and how further research could use it as a baseline. While I am sure that the MuLan embeddings and User Preference score are fine metrics in their own right, they are wholly inaccessible to the wider community. To accurately assess how the proposed method performs, not only to their own baselines but also external models, it is imperative to include metrics that are publicly usable in the literature, such as CLAP score to assess text-relevance, FAD/MMD (with CLAP backbone) to assess audio quality w.r.t. a reference set (like SongDescriber), and so on. I very much recommend the authors include some number of these (such as FAD and CLAP) in order to contextualize their performance in the larger space of TTM generation. In parallel, running existing open source models (such as AudioLDM2 or Stable Audio Open) on the present closed source eval suite would also work.
- The definition of “quality score” (259-260) is somewhat divorced from the literature on TTM and generative evaluation. This seems to combine MuLan score (theoretically a measure of “text relevance”) and user preference, but this explicitly entangles these two different axes of behavior that are generally kept separate (as how previous works do not combine CLAP and FAD for instance [3,4]). It should either be explicitly stated as to *why* this combination can be made, or more explicitly simply uncombine them and present all results for user preference and MuLan score separately, as currently it is impossible to tell *which* of these metrics the proposed method actually improves.
- I have strong worries on the overall statistical power of the human eval. As the human eval is a core part of the presented results, both in order to prove the efficacy of the proposed method and to back-verify that the diversity model correlates with human perception, it is unclear what sort of statistical significance actually occurs here, as there seem to be no hypothesis tests run nor error bars reported for any of the human eval metrics, exacerbated by the fact that there seems to be only 3 users in the listening study (which is far lower than even the small sample sizes common in generative media research). Specifically, it would be useful to see paired t-tests run between each method on the scores (averaged across human raters) to determine significance.
- There is some circularity to the evaluation of diversity in this work, despite it being a core focus. Though the authors rightly identify that the quality score is a good measure as it is not used in training, using the diversity model for eval and RL fine-tuning by nature biases results towards the proposed method (as it directly optimizes for this result). This ignores the fact that there are a number of established different metrics for assessing diversity in generative media, including Inception Score (IS), Recall, Coverage, and Vendi Score to name a few. Including some or any one of these scores would be useful both to bring it more inline with existing TTM literature as well as assess the output diversity of the model in an unbiased fashion and help verify whether the trained diversity model correlates with these values.
- Along with the above 2 points, the overall justification of the diversity model’s success hinges on Figure 7. (Middle)’s analysis that the fine-tuned model shows similar Elo ranking to the base model. I’m not sure if Elo score is the best measure for this, as it is a purely relative measure and details on its exact calculation are not provided.  It would be better to show the *absolute* scores for each model (with error bars) and see how well this correlates with the diversity model, and in general justify more in the paper that the diversity model is reasonably calibrated given its underpinning for the whole work.

[1] Luo, Simian et al. “Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference.” ArXiv abs/2310.04378 (2023): n. pag.

[2] Yin, Tianwei et al. “Improved Distribution Matching Distillation for Fast Image Synthesis.” ArXiv abs/2405.14867 (2024): n. pag.

[3] Bai, Yatong et al. “ConsistencyTTA: Accelerating Diffusion-Based Text-to-Audio Generation with Consistency Distillation.” Interspeech 2024 (2023): n. pag.

[4] Saito, Koichi et al. “SoundCTM: Uniting Score-based and Consistency Models for Text-to-Sound Generation.” ArXiv abs/2405.18503 (2024): n. pag.

[5] Novack, Zachary et al. “DITTO-2: Distilled Diffusion Inference-Time T-Optimization for Music Generation.” ArXiv abs/2405.20289 (2024): n. pag.

### Questions
- Line (254-255): In discussing batch size, is this for the number of samples or the number of pairs samples generated? In general, it is not clear when discussing the RL objective how these pairs are chosen in the context of training, and whether the RL objective is only calculated on single pairs of samples or multiple pairwise comparisons.
- In Figure 3, is there a reason there is only a single LERP data point for the 4-way merge? It would be useful to see if further interpolation between the 4 models is able to achieve a strictly dominant Pareto front above the LERP(0,15) model, as the claim that “This validates that we can merge an arbitrary number of models.” (Line 359) seems somewhat under-validated in this context.
- Line 962, what is 1-similarity?
- While certainly not important, I wonder if the authors have insight as to how techniques from LLMs in the contrastive decoding literature (especially works like DoLa which are self-contrastive) are related to CFG for autoregressive LM TTM models. In fact, the update equations from CFG and contrastive decoding/DoLa are practically identical, as they both involve the log difference in probabilities between the base model and some “worse” reference model.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method called Diversity-Rewarded Classifier-Free Guidance (CFG) Distillation for improving generative models, particularly in creative tasks like music generation. The authors introduces two training objectives:
1. Distillation: The method distills CFG’s quality improvements directly into the model weights, which eliminates the extra computational overhead during inference.
2.Reinforcement Learning (RL) for Diversity: A reinforcement learning objective rewards the model for producing diverse outputs, balancing the quality-diversity trade-off.
The approach focuses on a text-to-music generation task, where it successfully improves quality and diversity, validated by human evaluations.

### Strengths
1. This work provides a new strategy for managing the quality-diversity balance in generative models and has potential applications in other areas of creative content generation.
2. To achieve both diversity and quality, instead of doing direct multi-objective training which can downgrade the performance, the authors first distill model in terms of audio quality with classifier free guidance, which increase the quality while maintaining the capability of text understanding to music.
3. The analysis of quality-diversity tradeoff provides further insights to music generation models, where quality and diversity are actually two factors that need to be balanced.
4. Great demos and great results! Did not listen to all of them, in particular, I like LERP 50/50 of “Melodic Danceable Brazilian Music with Percussions” the most.

### Weaknesses
1. For music, the terms “quality” and “diversity” are not obvious in the paper. “Quality” can not only refer to audio quality, but can also refer to other aspects of music, like style consistency, the key consistency, rhythmic patterns, etc. I assume “quality” in this paper means the audio quality only. "Diversity" is specified in Appendix C.1, which I think it "Quality" and "Diversity" should be mentioned clearly in the introduction or methodology section.
2. In terms of evaluation of music, using MuLan score is great in terms of text coherence of generated music, while assuming “quality” means audio quality, it would be nice to include Fréchet Audio Distance (FAD) (Kilgour et al., 2019).
3. RL with a diversity reward objective maybe is newly applied to the realm of music, but this is not new in the RL world. I believe at least the work “Effective Diversity in Population Based Reinforcement Learning” (Parker-Holder et al., 2020) is worth mentioning and I did not see this work is cited in the current paper.  I would like to see some editing in the paper to change the wordings. If I miss this citation or relevant citations, please point out and correct me.

### Questions
1. Rewarding to Weakness 2, would you mind including FAD or other metrics to state the quality? 
2. "Diversity" is conceptualized as "Are these two audio excerpts derived from the same audio recording or not?". While this diversity is essential to study and learn, since we are dealing with music, is that possible that we can have better conceptualization of "Diversity"? For example, "Given one reference audio recording, how diverge are these two audio excerpts in terms of genre comparing to the reference audio?" In short, I think the word "Diversity" can be more music related.
3. Same logic applies to "Quality". Audio quality can be applied to speech as well. Since this work only focuses on music, can "Quality" be more music focused? For example, the structure and key coherence in the short excerpt. 
4. Again, RL with a diversity reward objective seems like a general objective used or inspired from other fields, like population-based RL. While the application to music is novel, I don't think this is novel enough to be highlighted as novelty. Would you mind defending your argument of novelty? Why this objective special to music (this objective can be applied to speech as well)?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a fine-tuning method that enhances both quality and diversity in generative models, targeting CFG’s limitations in computational cost and diversity reduction. The authors propose a two-part training objective: distilling CFG’s quality-enhancing effects into model weights and using reinforcement learning to encourage diversity. They also introduce model merging, which allows flexibility in balancing quality and diversity at deployment. Experiments on MusicLM show improvements over CFG, with human evaluations confirming superior quality-diversity trade-offs.

### Strengths
Originality: The paper presents an innovative approach to overcoming CFG’s limitations, particularly by combining distillation with a diversity-promoting reinforcement learning reward. It proposes a metric of diversity based on pair-level dissimilarity. 
Quality: The methodology is robust, with well-designed experiments that rigorously evaluate quality-diversity trade-offs. The use of human evaluations adds significant credibility, and the proposed model-merging strategy for adjustable quality-diversity control demonstrates strong, reliable results.
Clarity: The paper clearly articulates its objectives, methods, and findings, making complex topics accessible. Detailed diagrams and well-structured explanations approach and results easy to understand, enhancing the overall readability and impact.
Significance : By improving both quality and diversity without increasing inference cost, this work addresses an important limitation in generative models. While primarily applied to music generation, the approach has potential implications across other generative domains, offering a valuable contribution to fields where creativity and efficiency are essential.

### Weaknesses
About the diversity training
According to equation 3, D(\theta) dependents on the expected value of dissimilarity of y given x. How many ‘y’s  are sampled given an x ? Besides, this paper does not further investigate the impact of the sample number of an x on the diversity training . 


The definition of diversity is based on pair-level dissimilarity. It may be better to use some metrics like gini or Kurtosis to measure the global diversity of the generated songs.

### Questions
About music embedding
Why not use the open-source music embedding models to facilitate the reproducibility of this work? 

Are there any results of embedding evaluation like acc of linear probing that can be public for the music embedding model?

### Soundness
2

### Presentation
3

### Contribution
3
