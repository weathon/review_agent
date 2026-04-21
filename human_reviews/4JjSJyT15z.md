# NaturalSigner: Diffusion Models are Natural Sign Language Generator

- Avg Score: 4.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 8, 3, 3

## Abstract
Generating natural and expressive sign language pose sequences from text has important practical significance.
However, current sign language generation (SLG) methods suffer from low quality and limited expressiveness.
In this work, we propose NaturalSigner, a classifier-free diffusion-based generative model designed specifically for SLG.
 Specifically, it consists of a mixed semantic encoder that enhances the semantic consistency and expressiveness of the generated sign language, which takes both text and gloss as input; and a novel sign language denoiser that generates natural sign language pose sequences according to the output of the semantic encoder.
In addition, to achieve more natural and high-quality SLG, we design a sign language prompting mechanism to facilitate in-context learning in the diffusion model and duration predictor.
  Experiments on two datasets show that NaturalSigner significantly outperforms the state-of-the-art methods in terms of semantic consistency, naturalness, and expressiveness.
  On the Phoenix-2014T dataset, compared with the previous best end-to-end SLG method, NaturalSigner improves the BLEU-4 score of the back translation metric by more than **40\%** and reduces the Frechet Inception Distance (FID) by more than **12 times**.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The focus of this paper is on development for improved models for signal language generation. They propose a classifier-free diffusion model that goes from gloss or text to animation and produces parameters derived from SMPL-X. Results on Phoenix-2014 and Phoenix-2014T are significantly better than competing methods according to translations metrics (Rough, BLEU-4). Qualitative results show that animations from this model are preferred over one competing model.

### Strengths
* The motivation behind using diffusion models is sound (e.g., overcoming the one-to-many problem)
* The use of SMPL-X seems like a nice improvement over more common key point-based approaches. 
* The experiment ablations are nice and I appreciate the authors for breaking apart the prompting mechanism, semantic encoder, and deef forward denier.

### Weaknesses
* Related work is missing many references in the NLP, HCI, and Accessibility communities. There has been a lot of interest in SL modeling over the past 1-2 years but many of the references refer primarily to a line of work by Saunders et al. ending in 2021.
* The descriptions of the model/system formulation could benefit from more depth. Specifically, the introduction of the duration model isn't entirely clear. My hypothesis is that this is being used in the same way as duration models in TTS speech systems, but I'm not entirely sure. As an aside, given connections to TTS systems, it may be worth digging into that more in the related work section. 
* I didn't quite understand the motivation behind the prompt encoder. Maybe I missed it, but it might be worth clarifying why this is used. Are the results in Table5 and 6 statistically significant? The results with prompt encoder and without are very similar and in one case (Table 5 test) better without this encoder. 
* Is FID a good metric for this problem? Is there a demonstrable correlation between improved FID and improved signing quality?
* Subjective evaluation shows that this is better than Saunders, but doesn't give an overall sense of quality. Without videos it's unclear how well the approaches work (subjectively).

### Questions
I am on the fence about this paper. Can the authors provide video references for the animations? And answer some of the questions in the 'weakness' section?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors propose a diffusion model based SLG approach. The presented model uses SMPL-X body parameters as sign representations, and is able to generate realistic signer poses when being prompted with sign gloss and spoken language text. The authors conduct extensive experiments on the Phoenix datasets and report significantly better back-translation and FID scores compared to the state-of-the-art.

### Strengths
- Although diffusion models have been used for SLG before (https://arxiv.org/pdf/2308.16082v1.pdf) for photo realistic avatar generation, this paper is the first application of diffusion models to generate sequence of signer poses given sign glosses and spoken language text, to the best of my knowledge.
- The authors are sharing their source code. 
- The proposed approach achieves significantly better back-translation and FID scores on the Phoenix2014 datasets compared to the state-of-the-art.
- The authors conduct a user study with 10 participants to qualitatively assess their approach. I should stress the value of this, as the CV community does not commonly conduct user studies.
- Ablation study was informative.

### Weaknesses
- Not including facial expressions was disappointing. As the authors would appreciate, to fully convey and understand meaning of sign language utterances one must consider facial expressions, mouthings, gaze and mouth gestures. 
- Given the authors use a parametric body model as their sign representation, it was quite surprising to see that they've chosen skeletons as their visualization instead of a canonical body being driven by the generated pose configurations. 
- Although popular, Phoenix datasets are quite limited in terms of domain and signing variance. I'd have greatly strengthened the paper if the authors conducted studies in other datasets, like OpenASL or CSL, to set baselines for future research.

### Questions
Q: User Study: Were the participants deaf and proficient in DGS? I am asking since the authors mention that the participants evaluate the approaches based on the generated sequence being "easier to understand and closer to the ground truth”. Please clarify this in the manuscript. 

Suggestions and Minor Fixes: 
- Please use the term "Deaf and Hard of Hearing" instead of "Hearing-impaired"
- Use “ “ instead `` ‘’ in the latex to fix the quotation mark issues. 
- Page 2 naturalSigner -> NaturalSigner

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes to use a diffusion model to generate sign language keypoints. More specifically, the authors first design a sign language prompting mechanism considering the signer identity inforation. Then a mixed semantics encoder considering both text, gloss, and prompt, and a duration predictor are proposed as a prior model. Finally, a diffusion process is achieved as usual. The overall method achieves new SOTA performance on a series of metrics.

### Strengths
1. Use SMPLX parameters to represent motion is reasonable.
2. Sign languge prompting considers signer identity information.
3. SOTA performance on Phoenix-2014T on multiple metrics.
4. A user study is conducted.

### Weaknesses
1. The completeness of the paper is low.

a) I don't think generated keypoints are understandable to the deaf. Facial expression and mouth movement are important to sign language understanding but they are not included as shown in the demo. The generated keypoints may be better than baselines, but they are still far from being understood by the deaf people.

b) A classic prior work, FS-Net [1], has already achieved **video** generation using the keypoints to drive a signer image. But the proposed method doesn't support video generation.

c) Similarly, a recent open-sourced work, SignDiff [2], also applies the diffusion model on sign language generation, while it also provides video results.

2. The novelty is limited. The paper is more like an application of diffusion model on sign language generation, while the applicability has already been verified in SignDiff [2]. Although some adaptations are proposed, some of them, e.g., duration predictor, already appear in existing sign language papers [3,4].

3. It is good to involve signer identity information in the sign langauge prompting, but there are not corresponding experiments to verify its effectiveness. I don't think current generated results can reflect signer identity information.

4. Need more details for the user study. Since the task is for the deaf people, how are they fluent with sign language? Are they Germany sign language users? How many videos are given to the users? Are the text and gloss annotations given to them?
 
5. An important benchmark, CSL-Daily, is missing. Phoenix-2014 and Phoenix-2014T are quite similar, and thus the conclusions on these two benchmarks are always consistent. Other benchmarks from a different language, e.g., CSL-Daily, is necessary.

[1] Signing at Scale: Learning to Co-Articulate Signs for Large-Scale Photo-Realistic Sign Language Production, CVPR 2022

[2] SignDiff: Learning Diffusion Models for American Sign Language Production, arXiv 2023

[3] SimulSLT: End-to-End Simultaneous Sign Language Translation, MM 2021

[4] Towards Fast and High-Quality Sign Language Production, MM 2022

### Questions
See weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies the topic of sign language generation. It proposes a NaturalSigner framework to leverage the strong modeling capability of diffusion model. The experiments on two datasets demonstrate the effectiveness of the proposed method.

### Strengths
This paper studies the topic of sign language generation. It proposes a NaturalSigner framework to leverage the strong modeling capability of the diffusion model. 

The experiments on two datasets demonstrate the effectiveness of the proposed method.

### Weaknesses
One of the main concerns is that the presented qualitative results are not satisfying. For the keyframe in Figure 4/video on the demo webpage, I do not see a clear improvement over the previous method.

The title is confusing. What does the word natural mean? 

The novelty is somewhat limited. The authors should clearly state why diffusion better works for SLG.

I do not think section 3.2 should be called in-context learning. It is just an embedding technique.

What is the specific setting of zero-shot SLG?

### Questions
The questions are listed above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
