# Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
While recent zero-shot text-to-speech (TTS) models have significantly improved speech quality and expressiveness, 
mainstream systems still suffer from issues related to speech-text alignment modeling: 1) autoregressive large language models are inefficient and not robust in long-sentence inference; 2) non-autoregressive diffusion models without explicit speech-text alignment require substantial model capacity for alignment learning; 3) \textcolor{red}{predefined alignment-based diffusion models suffer from naturalness constraints of forced alignments} and a complicated inference pipeline. This paper introduces \textit{S-DiT}, a TTS system featuring an innovative sparse alignment algorithm that guides the latent diffusion transformer (DiT). Specifically, 1) we provide sparse alignment boundaries to S-DiT to reduce the difficulty of alignment learning without limiting \textcolor{red}{the search space}; 2) to simplify the overall pipeline, we propose a unified frontend language model (F-LM) training framework to cover various speech processing tasks required by TTS models. Additionally, we adopt the piecewise rectified flow technique to accelerate the generation process and employ a multi-condition classifier-free guidance strategy for accent intensity adjustment. Experiments demonstrate that S-DiT matches state-of-the-art zero-shot TTS speech quality while maintaining a more efficient pipeline. Moreover, our system can generate high-quality one-minute speech with only 8 sampling steps. Audio samples are available at https://sditdemo.github.io/sditdemo/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents S-DiT, a zero-shot text-to-speech system that addresses speech-text alignment issues by using a sparse alignment algorithm to guide a latent diffusion transformer. By providing sparse alignment boundaries and simplifying the pipeline with a unified frontend language model, S-DiT reduces alignment learning difficulty without limiting expressiveness.

### Strengths
The paper addresses a clear problem in zero-shot text-to-speech systems by introducing S-DiT, a method that employs a sparse alignment algorithm to guide a latent diffusion transformer, reducing the difficulty of alignment learning without limiting expressiveness. Multi-condition CFG looks promising for cross-lingual settings. The authors have conducted most of the necessary experiments to demonstrate their claims.

### Weaknesses
Quality:
- The paper presents sparse alignment as a major contribution, but its impact is not clearly demonstrated. While experiments show improvements over the "Diffusion w/o PA" model, this is expected since models like ns3 already outperform it. This raises questions about the specific benefits of sparse alignment, especially since it requires similar preprocessing efforts but appears to omit some information to achieve sparsity. Providing detailed comparisons with ns3 and illustrating how sparse alignment enhances aspects like expressiveness would clarify its advantages. Currently, multi-condition CFG and multi-task learning F-LM seem to offer more substantial advancements independently of sparse alignment, which may not align with the paper's intended focus.
- Since F-LM is also an autoregressive model, it would strengthen the paper to directly compare its robustness to long sequences against other AR models. For instance, does the combination of F-LM and S-DiT exhibit less performance degradation as sequence length increases compared to traditional AR approaches?
- In line 155, the paper claims improvements in efficiency. However, the required data preprocessing steps (phonemization, MFA, duration prediction, etc.) remain the same and have been unified, yet inference still relies on an inefficient AR model. To convincingly demonstrate efficiency gains, it would be helpful to measure and compare the average frontend processing time of the "Duration w/ PA" model, similar to how S-DiT's frontend time is measured in line 412. This comparison would offer concrete evidence of any efficiency improvements and validate the claim of enhanced efficiency.
- Regarding the Speech Compression module in Section 3.1, the reconstruction performance has not been clearly verified. Given the existence of various established neural codecs (e.g., DAC [1], EnCodec [2]) and recent advancements in neural vocoding (e.g., BigVGAN [3]), it would be pertinent to explain why the paper adopts a structure based on image processing techniques. Since the performance of the speech compressor sets an upper bound on the final TTS quality, evaluating its reconstruction performance is crucial for properly assessing S-DiT. Comparing the performance of different speech compression methods, including DAC and EnCodec, using objective metrics like PESQ or ViSQOL, and evaluating the TTS performance resulting from each method would provide a comprehensive understanding of the speech compression's impact on the overall system.

Clarity:

- The boldface in each table does not have a consistent meaning. Please standardize it according to a consistent criterion (e.g., highlighting the highest-performing values).
- In line 91, please specify exactly what is meant by "certain commonalities."
- In line 101, please clarify what the "interesting conclusions" are.
- In line 249, for "Piecewise Rectified Flow Acceleration," please indicate at the beginning of the paragraph that this represents distillation, meaning that a pretrained model is required.
- In line 335, regarding "1M steps for S-DiT," does this 1 million include both pre-training and distillation? What is the proportion between the two?

[1] Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., & Kumar, K. (2024). High-fidelity audio compression with improved rvqgan. *Advances in Neural Information Processing Systems*, *36*.

[2] Défossez, A., Copet, J., Synnaeve, G., & Adi, Y. (2022). High fidelity neural audio compression. *arXiv preprint arXiv:2210.13438*.

[3] Lee, S. G., Ping, W., Ginsburg, B., Catanzaro, B., & Yoon, S. (2022). Bigvgan: A universal neural vocoder with large-scale training. *arXiv preprint arXiv:2206.04658*.

### Questions
- In the demo, are the results for "Zero-Shot TTS" and "Accent Intensity Control" from the 7B model? Why are there no 7B samples in "Incredible Improvement Brought by Scaling"? In "Robustness" and "Code-Switched Generation," is the comparison made with the 0.5B model to match the baseline in terms of model parameters?
- What issues arise if Duration Control is made more extreme? For example, if it's too long, does the voice change or words repeat? If it's too short, does the speech become flat or omit words? I'm curious about the extent of robustness problems.
- It would be helpful to dedicate a separate section in the demo to add samples that clearly showcase the advantages of S-DiT in terms of expressiveness compared to duration w/ PA models like ns3. This would help intuitively understand the main claims of the paper.
- Multi-condition CFG seems particularly useful in code-switching. In the code-switching section of the demo, it would be nice to have a comparison of the results from the three models used in Table 7—Ours, w/ Standard CFG, and w/o CFG.
- Section 3.2
    - In the Sparse Alignment Strategy, the experiments in the paper appear to assume a setting with only one anchor for rough alignment. Is there a trade-off between expressiveness and robustness depending on the number of anchors? Is having one anchor the optimal choice in this context?
    - In the Multi-condition CFG, regarding the experiments mentioned in line 273 that examined changes in points 1), 2), and 3), what exactly does this experimental setup entail? Does it refer to fixing \( a_{\text{spk}} \) at a specific value (e.g., 1) in Equation (5) and varying \( a_{\text{txt}} \)?
- Section 3.3
    - When examining the input \( h \), it seems that \( t \) addresses the ASR task, while \( p \) simultaneously tackles alignment, DP, and G2P. In Figure 2, grouping these latter three components together instead of distinguishing them separately might aid in understanding.
    - Is there a particular reason why the speech encoder and language model were not initialized with pretrained weights? Utilizing existing speech compression modules and small LMs for initialization could potentially enhance performance.
    - In line 305, regarding \( t \) (BPE) in the input \( h \), is it necessary to apply a loss to match parts that are not from the speech prompt? Was including this in the multitask learning beneficial for improving performance in alignment, DP, and G2P, prompting its inclusion in training?
    - It is unclear how, during inference, the model is constrained to generate only up to the text corresponding to the speech prompt when the speech prompt is provided. For instance, when randomly discarding the speech encoder output, one might expect the model to align and truncate the text accordingly, insert a speech prompt EOS between the prompt text and target text during training, and then generate only up to that point during inference to concatenate with the given target text. However, the paper does not detail this process.
- Appendix
    - Could the paper provide a more detailed explanation of "the simple scaling of weakly supervised pre-training" mentioned in line 1052? Why is this referred to as weakly supervised pre-training? Additionally, how were the MFAs in Section 4.4 trained?

Typos

- Line 100: In the phrase "expressiveness of 'diffusion w/ PA' with the robustness of 'diffusion w/o PA'," it appears that "w/ PA" and "w/o PA" might be mistakenly swapped.
- Line 112: The phrase "but can also" might be more appropriately written as "but also can."

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper proposes a new method for learning the alignment between text and speech for diffusion-based TTS models. Instead of providing the full duration information or not providing any duration information, the authors propose a method to provide partial duration information and let the models figure out the masked part of the duration to make the duration prediction model more robust. The results show that the model has achieved a performance similar to the previous state-of-the-art model while being significantly faster.

### Strengths
* **Originality**: The paper proposes to condition the diffusion-based TTS models on partial duration information instead of no duration or full duration during training and inference, and the duration is predicted autoregressively through a multi-tasking frontend LLM, which has been shown to increase the robustness and performance.
* **Quality**: The paper has conducted extensive experiments to compare with various models and has provided comprehensive evaluations on the effectiveness of several proposed components and their performance in a few tasks. 
* **Clarity**: The paper is quite clear in its writing and has a simple, not overly complicated presentation. It is easy to follow and intuitive to understand.
* **Significance**: The model has surpassed various previous models and achieved close-to-SOTA performance with significantly reduced inference time.

----------------

**UPDATE (11/22/24):** After the authors addressed my questions, here is a revised significance (as the inference time is not significantly reduced). 

* **Significance**: The model has surpassed various previous models and achieved close-to-SOTA performance.

### Weaknesses
**Major**:
1. One major issue is many claims are unsupported. Here are some examples of unsupported claims by the authors:

>predefined alignments constrain the model’s ability to produce expressive and natural-sounding speech (Yang et al., 2024b; Chen et al., 2024)

I read Yang et al., 2024b and Chen et al., 2024 but did not find any supporting evidence that predefined alignments hinder model’s ability to produce expressive and natural-sounding speech. In fact, Yang et al. 2024b uses "Diffusion w/ PA" and does not compare its method or any "Diffusion w/ PA" method against others and conclude that "Diffusion w/ PA" falls short of expressiveness. Similarly, Chen et al., 2024 is an "AR LM" model that does not conclude that "Diffusion w/ PA" is worse than "AR LM" in expressiveness. It can be helpful to add the exact experiment conducted in the paper. If no experiment is available, you may revise the statement to more accurately reflect the current state of knowledge in the field or provide relevant experiments in the paper. 

> ARDiT (Liu et al., 2024b) proves that when compared under an identical number of parameters, methods without explicit duration modeling exhibit some decline in speech intelligibility and speaker similarity.

I read and searched through the entire paper of ARDiT (Liu et al., 2024b) and found no experiment that compares the number of parameters. It can be helpful to add the exact experiment conducted in the paper or follow the same suggestion as above. 

>Experimental results demonstrate that S-DiT ... exhibiting the expressiveness like codec language model based approaches.

There is no experiment that examines the "*expressiveness*" in the paper. If there is, please kindly point out the experiment that specifically examines "expressiveness." Throughout the entire paper, the only conclusion in the results section is "*S-DiT significantly
surpasses all baselines in terms of CMOS, demonstrating the expressiveness and naturalness of the proposed sparse alignment strategy*," but CMOS is not the same as the expressiveness. In fact, codec language model based models do not have the best CMOS according to Table 2 (the highest CMOS is NaturalSpeech 3, which is classified as "Diffusion w/ PA" by the authors), but the authors claim that S-DiT has achieved “the expressiveness of codec language model”, which is not well supported by the results in the paper. 

> “Diffusion w/o PA” requires more parameters due to the difficulty in end-to-end modeling of speech-text alignment non-autoregressively.

Since Liu et al., 2024b did not make such a claim and the authors did not examine the effect of number of parameters for "Diffusion w/o PA", this claim is unsupported. Please kindly add the experiments that support this claim. In fact, Eskimez et al. 2024 is a 333M model that has achieved SIM-O of 0.675 and WER of 0.2, better than NaturalSpeech 3, and a recent work, F5-TTS [1], has reproduced Eskimez et al. 2024 results of 333M parameters with SIM-o of 0.69 on LibriSpeech test-clean dataset. Both models have fewer parameters than S-DiT (0.3B vs. 0.5B).

> On the other hand, the use of predefined hard alignment paths limits the model’s expressiveness

Since neither Yang et al., 2024b nor Chen et al., 2024 supports this claim, please kindly add the experiment that shows predefined hard alignment limits the model's expressiveness. In fact, on the NaturalSpeech 3 demo page, there is a section particularly dedicated to the emotional speech capability and the model shows no constraint in expressing various emotions. 

2. Another major issue is the contribution of this work is unclear. The authors have summarized the pros and cons of existing methods compared to S-DiT in Table 1. However, the proposed method does not solve either of the problems (i.e., complicated inference pipeline and expressiveness constraint by "Diffusion w/ PA", and increased amount of parameters by "Diffusion w/o PA") identified by the authors in the introduction section. The proposed pipeline is still complicated compared to "Diffusion w/ PA", if not more so, as the training still involves ground truth duration labels (possibly from FMA or some other aligners), and the inference still involves duration prediction. On the other hand, E2-TTS (Eskimez et al. 2024) has no such requirements during training, no P2G and duration label required duration training with fewer parameters than S-DiT but demonstrated similar or better performance in terms of SIM-O and WER. The experiments did not support the expressiveness constraint by "Diffusion w/ PA" and increased parameter demand by "Diffusion w/o PA". Hence, the contribution of this paper is unclear. The authors should consider revising the paper to emphasize more on its contributions, such as the robustness of masking the duration prediction in diffusion-based models, instead of expressivness and parameter demand, which is unsupported and likely untrue given existing literature. 


3. Additionally, the result presented by the paper is not convincing enough. For example, the claim that "*S-DiT matches state-of-the-art zeroshot TTS speech quality*" is not fully supported by the supplementary materials. The authors claim that the performance of S-DiT is close to the state-of-the-art model (NaturalSpeech 3), but based on samples on the demo page, it is likely worse than NaturalSpeech 3. For example, using the WavLM-TDCNN large fine-tuned speaker embedding model, I calculated the similarity score for all 4 samples in the "Zero-Shot TTS" section of the supplementary material, and I obtained the following results (NS3 means NaturalSpeech 3):

| Text | S-DiT | NS 3 | Difference |
| -------- | ------- |------- | ------- | 
| *His death in this conjuncture was a public misfortune.* | 0.7463   |  **0.7577**    | -0.0114   |
| *For if he's anywhere on the farm, we can send for him in a minute.* | **0.7499**   | 0.6905     | 0.0594 |
| *John Wesley Combash, Jacob Taylor, and Thomas Edward Skinner.* | 0.7145   | **0.7662**     | -0.0517|
| *The strong position held by the Edison system under the strenuous competition that was already springing up was enormously improved by the introduction of the three wire system and it gave an immediate impetus to incandescent lighting.* | 0.7050 | **0.8103** |  -0.1053 |

It is difficult to believe that S-DiT has achieved the same SIM-O as NaturalSpeech 3, as presented by the author in Table 2 since 3 out of 4 samples fall short of NS3, and the total difference is large (-0.11). Moreover, the fourth sample, "The strong position ...", has clear distortion in the S-DiT's sample (but not in NS3), and "John Wesley Combash, Jacob Taylor ..." has unnatural pauses and some distortions (but not in NS3). This makes it hard to believe that S-DiT outperforms NS3 in CMOS too. 

I understand it is unfair to judge the quality of this work based on four samples, but evaluating the demo page of S-DiT and NaturalSpeech 3, I can see that S-DiT is likely worse than NaturalSpeech 3 (more distortions, unnatural prosody and lower similarity to the prompt in all sections). Since only four samples are presented on the demo page that directly compare S-DiT to NaturalSpeech 3, it is helpful to provide more samples to convince the readers that the claim is indeed true. 

Moreover, as the details of the evaluations are missing (i.e., how many samples are used for SIM-O evaluations, how the prompt/sample pairs are generated for objective evaluations, and what the demographics are like for the raters in subjective evaluations, whether there are attention checkers, how the results are quality-checked etc.), I suggest the authors provide more information about how the evaluations are conducted so the results can be more convincing. Part of these information is available in NaturalSpeech 3 paper but none of it is in S-DiT.


**Minor**:
- Some grammatical issues, for example: 

> neural codec language models (Wang et al., 2023; Chen et al., 2024) first can autoregressively synthesize speech that rivals human recordings in naturalness

should be corrected to "neural codec language models are the first that can ..." Moreover, Wang et al., 2023 does not claim to have synthesized speech that rvials human recordings; it is Chen et al., 2024 that makes this claim. 

- Please kindly cite Yang et al. 2024 [2] for multi-condition CFG, as this paper was published 2 months before the ICLR deadline and should be mentioned. 

**References**:

[1] Chen, Y., Niu, Z., Ma, Z., Deng, K., Wang, C., Zhao, J., ... & Chen, X. (2024). F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching. arXiv preprint arXiv:2410.06885. 
>Note: This work is published after the ICLR deadline and the author is not required to compare to this work. However, this work has reproduced the results from a previous work (Eskimez et al. 2024) that was published 2 months before the ICLR deadline, which directly contradicts the author's claim; hence, I believe this reference is relevant for this review. 

[2] Yang, J., Lee, J., Choi, H. S., Ji, S., Kim, H., & Lee, J. (2024). Dualspeech: Enhancing speaker-fidelity and text-intelligibility through dual classifier-free guidance. arXiv preprint arXiv:2408.14423.

---------------------

**UPDATE (11/22/24):** The authors have addressed my concerns over weaknesses 1 and 3. However, weakness 2 still persists and is now even worse. Since the model produces worse sound quality (due to autoencoder melspectrogram) and requires a more complicated training pipeline (due to F-LM and duration input) compared to E2-TTS, it has limited practical use and lower scalability. The only improvement over E2-TTS appears to be higher intelligibility, which could be caused by different training data and implementation (E2-TTS reported a WER of 1.9%, A$^2$-Flow reproduced the model with the same WER, but the authors reported a WER of 2.95% for E2-TTS), so it hardly justifies the lower sound quality and significantly more complicated training pipeline of S-DiT.

Contribution: 2 -> 1 (as weakness 2 gets worse after clarification)

Soundness: 1 -> 3 (as weaknesses 1 and 3 are addressed)

### Questions
1. Can you please tell me where you draw the conclusion that “Diffusion w/o PA” needs more parameters and "Diffusion w/o PA" has limited expressiveness? 

2. How is the pipeline less complicated compared to ""Diffusion w/o PA", since the training clearly still needs the alignment to deduce the ground truth duration?

3. How is the ground truth duration obtained during the training of frontend LLM? What is the ground truth of duration in Table 6? If the ground truth duration is from FMA, does that mean that the duration produced by F-LM is better than ground truth? 

4. The frontend LLM is an AR LLM, and the DiT is NAR, what are the parameters for each module? I believe the reported #parameter (0.5B) is a combined AR-LLM and DiT, and DiT should have way fewer parameters than 0.3B, or the RTF does not make any sense (25 steps of NFE is two times faster than 16 NFE of E2-TTS as reported in [1]). This DiT has to be incredibly small to have this RTF since the inference pipeline also has an AR component that is way slower, so please kindly report the #parameters for DiT and AR LLM separately. 

**Reference**:

[1] Chen, Y., Niu, Z., Ma, Z., Deng, K., Wang, C., Zhao, J., ... & Chen, X. (2024). F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching. arXiv preprint arXiv:2410.06885.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the S-DiT model, which, although it utilizes predefined alignment, enables expressive TTS comparable to models that learn alignment implicitly. Unlike previous models that use a phoneme duration predictor and rely on dense alignment, S-DiT employs sparse alignment by providing only partial alignment information randomly. This approach allows the model to learn expressively while maintaining alignment robustness. To provide phoneme duration prediction and a transcript for prompts, the paper introduces a frontend language model (F-LM) framework and adopts piecewise flow matching to accelerate the sampling process. The model demonstrates high speaker similarity and maintains good pronunciation accuracy in zero-shot TTS tasks, with significantly fast sampling speeds.

### Strengths
* By compressing the spectrogram into a latent representation that is 8 times shorter through speech compression, the model can be trained more efficiently and generates shorter sequences during inference, offering advantages in terms of inference speed.

* The model retains the robustness of previous models that use duration predictors while overcoming their limitations in expressiveness by introducing a sparse alignment mechanism. This allows it to achieve performance comparable to models that learn alignment implicitly.

* The introduction of F-LM simplifies the inference process by predicting phoneme durations and representations using only the prompt audio and the target text sentence, eliminating the need for a transcript of the prompt.

* The paper shows that sampling speed can be further accelerated using piecewise rectified flow.

* The paper shows that the model's performance progressively improves with the scale of the model and the amount of training data.

### Weaknesses
* To provide alignment information during training, the Montreal Forced Aligner (MFA) is additionally required, which may limit the model’s extension to other languages depending on the aligner’s performance.

* In the Method section, the descriptions of each module are quite simplified, making the process of inputting each condition into the model only roughly understandable and not precisely detailed. For instance, the explanation of whether the model performs masked speech modeling is very brief. Additionally, it is unclear how the rough alignment information is provided to S-DiT. In Figure 1 (b), it appears that a representation of the text token from F-LM is also provided to S-DiT, but these aspects are not clearly explained.

### Questions
* To provide rough alignment during inference, is alignment provided randomly? If so, I wonder how the samples change depending on the method of providing rough alignment.

* Was phoneme duration stochastically sampled from F-LM? If it was sampled deterministically, how does the performance change when using stochastic sampling?

* When training on 600k hours of speech data, how was the duration label obtained and provided?

Typos
* In equation (5), x_t -> z_t.

* In Appendix E, in the Special Tokens section, <End of BPE]> ->  <End of BPE>

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents S-DiT, a zero-shot text-to-speech (TTS) synthesis model that addresses critical challenges in alignment modeling. The authors introduce a sparse alignment mechanism to guide a latent diffusion transformer, balancing expressiveness and robustness without over-complicating the pipeline. The work also proposes a frontend language model (F-LM) that unifies various speech processing tasks including grapheme-to-phoneme and duration predcition, and a piecewise rectified flow technique, which reduces inference steps without significant performance degradation. Experiments show that the resulting TTS system outperforms state-of-the-art zero-shot TTS baselines in quality and efficiency.

### Strengths
* The sparse alignment mechanism effectively combines the advantages of predefined and non-predefined alignment strategies, improving both the expressiveness and robustness of TTS synthesis.
* The paper demonstrates effective use of piecewise rectified flow to reduce inference time without significant quality loss.
* The paper is clearly written and easy to follow, with the proposed model architecture and training algorithms well-explained. The detailed analysis of experimental results further enhances comprehension.

### Weaknesses
* Although precise phoneme-speech alignment is not required for training the diffusion model, it is necessary for training the frontend language model. This requirement limits the scalability of the approach to larger datasets and multilingual scenarios. In contrast, latent diffusion models without such alignment dependencies offer better scalability, while models with exact alignment perform more efficiently when data is scarce. Consequently, the proposed model may struggle with smaller-scale data and face preprocessing inefficiencies that hinder scalability as the dataset size increases.
* The paper does not adequately discuss potential failure cases in speech generation. It would be beneficial for the authors to address scenarios where the model might underperform, such as specific sparse alignment patterns that lead to failures, alignment difficulties with long text inputs, or challenges with extremely long or short phoneme durations. Understanding these limitations would provide a clearer understanding of the proposed model.

### Questions
* It would be helpful to include an evaluation of the reconstruction quality of the speech compression model.

### Soundness
3

### Presentation
3

### Contribution
2
