# VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Despite tremendous progress in the field of text-to-video (T2V) synthesis, open-sourced T2V diffusion models struggle to generate longer videos with dynamically varying and evolving content. They tend to synthesize quasi-static videos, ignoring the necessary visual change-over-time implied in the text prompt. Meanwhile, scaling these models to enable longer, more dynamic video synthesis often remains computationally intractable. To tackle this challenge, we introduce the concept of Generative Temporal Nursing (GTN), where we adjust the generative process on the fly during inference to improve control over the temporal dynamics and enable generation of longer videos. We propose a method for GTN, dubbed VSTAR, which consists of two key ingredients: Video Synopsis Prompting (VSP) and Temporal Attention Regularization (TAR), the latter being our core contribution. Based on a systematic analysis, we discover that the temporal units in pretrained T2V models are crucial to control the video dynamics. Upon this finding, we propose a novel regularization technique to refine the temporal attention, enabling training-free longer video synthesis in a single inference pass. For prompts involving visual progression, we leverage LLMs to generate video synopsis - description of key visual states - based on the original prompt to provide better guidance along the temporal axis. We experimentally showcase the superiority of our method in synthesizing longer, visually appealing videos over open-sourced T2V models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors first analyze the behavior of temporal attention when generating longer videos. Then based on the observation, they propose VSTAR, a tuning-free method for longer video generation. The framework supports the transition of mult-prompt through the planning of pre-trained LLMs.

### Strengths
1. The analysis of temporal attention has some inspiration.
2. The design of TAR enables longer video generation in a one-pass behavior.
3. Benefiting from one-pass generation, the framework naturally supervises the coherence of results described by VSP, making it easier to achieve the transition of mult-prompt than the previous methods.
4. Although the quantitative results in this paper are still to be completed, the qualitative results are relatively sufficient, fully explaining the role of each module of VSTAR and other methods compared.

### Weaknesses
1. This paper does not evaluate some classic metrics, such as FVD. Alternatively, evaluating on a comprehensive benchmark like VBench will make the comparison more convincing.
2. Table 2 lacks a setting of Baseline+TAR. Purely Baseline+VSP is supposed to bring obvious content mutation while gaining better scores compared to Baseline, further illustrating that the two indicators in the paper are not enough to fully evaluate the quality of video generation.
3. When comparing VSTAR with FreeNoise in a mult-prompt setting, noticeable flickering in FreeNoise exhibits the implementation is not aligned with the original paper. (Although this does not affect my recognition of the effectiveness of VSTAR in the mult-prompt setting.)

### Questions
Most concerns are listed in weakness. Out of curiosity, FreeLong also mentions a similar phenomenon in temporal attention. Can you briefly explain the pros and cons of the two designs? Of course, it is not necessary to discuss the concurrent work according to the Reviewer Guide.

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
5

### Summary
The paper proposes an attention regularization technique for long video generation. In addition to that, the paper also proposes to use existing LLMs to generate better video text descriptions. The main contribution of this work lies in the temporal attention regularization technique, which has been presented with good motivations and was able to regularize the temporal attention to mimic the ones in real videos. The other merit of this approach is that it can be directly applied to certain T2V models without retraining.

### Strengths
The main strength of this paper is the proposed Temporal Attention Regularization (TAR), which has been inspired by comparing temporal attention maps between the real and the generated videos. The motivation for this idea is clear and strong, and we can see the differences in attention maps. This helps motivate the use of the proposed TAR approach. In particular, the demonstration of attention to visual patterns offers convincing reasons why the regularization should be performed. 

Another good thing about this TAR is that it could be applied to existing T2V models without re-training as long as the existing models satisfy certain conditions. The paper also performs extensive experiments to justify such a design, pertaining to the impact of resolutions and number of frames.

### Weaknesses
The per-layer temporal attention analysis part is not very clear in Sec. 3.2. Are the resolutions corresponding to the layer dimensions in the UNET? Does that occur in both the encoder and decoder parts of UNET? 

The use of additional matrix "max" in Eqn. (3) needs to be further validated. The Toeplitz matrix should ensure that the values for distance frames will decrease. The multiplication of delta_A with the max function will only scale the values but does not change the ordering. I am just curious about the impact of this "max" function and how will affect the regularization performance. It appears that the paper does not offer a more detailed analysis of this part, including experimental validations. 

The use of DreamSim in Sec. 4.1 could transfer the dynamics from real videos to the target one based on this delta_A. By looking at the results presented in Figure 8, it appears that the dynamics transfer also includes the style transfer. Intuitively, the attention map was used to regularize the interactions among frames to ensure that the evolving content was consistent with the real videos. With that being said, should we assume that these "dynamics" also model the video motions? If that is the case, the style transfer appears to be confusing here.  Is this more due to the use of DreamSim to calculate the similarity matrix? Please also clarify what are the captured dynamics by using the default delta_A. 

Based on the discussions from Sec.5, there are limitations to applying the TAR to existing T2V models, depending on how the temporal attention was designed. Though this was not specifically mentioned, was this mainly due to the use of a Transformer backbone instead of the UNET? The Transformer does use frequently with the position embeddings. Most existing video diffusion models, however, are based on the Transformer architecture, such as W.A.L.T, Latte, and VDT. Please offer more comments in this regard by looking at these Transformer-based architecture and whether the proposed approach can be applied to them. Some of them are also open-sourced. The paper have already provided some analysis. But it will be appreciated with further analysis.  

The use of VSP in Sec 4.3 is straightforward. After applying the LLM, does the new generated prompt considered as the whole prompt to be used in cross-condition? Or does the paper treat the generated prompts as sub-segments and sequentially input that into the model? Please also clarify that. Compared to StoryDiffusion, which also models long-range video generation, what will be the advantage of the VSP? In general, what is the VSP+TAR compared to StoryDiffusion in particular in terms of long-range video generation.

### Questions
There are a bunch of questions raised in the weakness section. I would appreciate it if the authors can address these questions, especially these related to the TAR.

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
5

### Summary
The authors systematically analyze the role of temporal attention in Video Generation and contribute a training-free technique called “Generative Temporal Nursing” to create more temporal dynamics for long video generation.  Besides, LLM-based video synopsis prompting is explored to facilitate gradual visual changes. Experimental results show the effectiveness of the proposed methods in terms of enhancing video dynamics.

### Strengths
1. The idea is simple and easy to follow.
2. The motivation of the method is reasonable and strong.
3. The analysis of temporal attention in T2V model may benefit other research.
4. The paper is well-written and well-organized.

### Weaknesses
1. The introduction to the VSP is short and some of the details are not clear: do all frames share one interpolated text embedding or does each group share a different embedding?
2. Introducing a Toeplitz matrix to temporal attention could help to improve temporal dynamics. What I am concerned about is that this kind of hard modification may break the original motion, since I find that the motions of Superman and Spiderman are wired.
3. I wonder about the proportion of the "less structured" temporal attention maps in video diffusion models because only some cases and layers do not satisfy the band-matrix pattern according to my personal experience.
4. It seems that MTScore from ChronoMagic-Bench is adopted to evaluate the video dynamics. I want to see other metrics in the same benchmark like Coherence Score (CHScore). I am happy to raise my rate if more quantitative results are provided.

### Questions
Please refer to the weakness part.

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
4

### Summary
The paper proposes a new concept of "Generative Temporal Nursing" and aims at enhancing the temporal dynamics of long video synthesis without requiring additional training or introducing significant computational overhead during inference. Specifically, this paper first find out that the temporal correlation varies a lot betwen frames, and introduce TAR module to adjust the temporal correlation with regularization. Then this paper propose VSP module to provide key visual states to decompose the transition along the temporal axis with LLM.

### Strengths
1. The analysis of temporal correlation is meaningful and interesting.

2. TAR module is well-motivated, effective and readily applicable to pre-trained T2V models.

3. The paper is well-structured and clearly presents the methodology, experiments, and results.

### Weaknesses
1. The side effect of TAR: In the Fig.5, the introduction of TAR in generation models may reduce the amplitude of motion.

2. Further ablation on TAR. Does VSP module is used in Fig.9? How do the TAR module and attention map change with the introduction of VSP module?

3. Temporal consistency. Does the introduction of VSP lead to a decrease in temporal consistency? Can authors provide some video demos?

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
