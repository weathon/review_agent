# MuVi: Video-to-Music Generation with Semantic Alignment and Rhythmic Synchronization

- Decision: Reject
- Scores: 3, 8, 6

## Abstract
Generating music that aligns with the visual content of a video has been a challenging task, as it requires a deep understanding of visual semantics and involves generating music whose melody, rhythm, and dynamics harmonize with the visual narratives. This paper presents MuVi, a novel framework that effectively addresses these challenges to enhance the cohesion and immersive experience of audio-visual content. MuVi analyzes video content through a specially designed visual adaptor to extract contextually and temporally relevant features. These features are used to generate music that not only matches the video’s mood and theme but also its rhythm and pacing. We also introduce a contrastive music-visual pre-training scheme to ensure synchronization, based on the periodicity nature of music phrases. In addition, we demonstrate that our flow-matching-based music generator has in-context learning ability, allowing us to control the style and genre of the generated music. Experimental results show that MuVi demonstrates superior performance in both audio quality and temporal synchronization. The generated music video samples are available at muvi-v2m.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes MuVi, a new method for generating music that aligns with video content, focusing on both semantic alignment and rhythmic synchronization. MuVi's design includes a "visual adaptor" that extracts relevant visual features from videos, which helps guide music generation to match the mood and rhythm of the video. To improve synchronization between visual events and musical beats, the authors use a pre-training technique that contrasts synchronized and unsynchronized video-music pairs, helping the model learn rhythmic alignment.

### Strengths
1. This paper introduces a new method for generating music that aligns with video content, focusing on both semantic alignment and rhythmic synchronization within a generative video-music Diffusion Transformer framework.
2. The model employs a joint encoder-decoder architecture that integrates a contrastive pre-training scheme for improved synchronization. The inclusion of a "visual adaptor" enhances the model’s ability to compress and process high-frame-rate visual inputs, capturing video cues for music generation.
3. The paper is well-organized, presenting MUVI's methodology alongside a series of experiments. The framework demonstrates superior performance over the baseline on the test dataset across various evaluation metrics, showcasing its effectiveness in video-to-music generation.

### Weaknesses
1.Novelty and Contribution: The paper presents its main contributions as a visual adaptor and a contrastive training scheme, but visual adaptor techniques and contrastive learning have already been used in video-to-music generation tasks [1, 2] and are commonly employed in multi-modal learning [3, 4]. The design of the visual adaptor lacks unique innovation, primarily involving a selection of common aggregation and pooling methods, which appears more as an ablation study to find the best setting. Overall, the proposed method lacks novelty, and the results in Table 2 indicate that the proposed method does not outperform the baseline across all metrics.

2.Lack of Justification and Explanation: Another weakness is the lack of clear justification and explanation across different sections, from design choices to metric selection.

2.1 The adaptor design section lacks a clear justification. For instance, why were these three adaptor methods chosen, instead of exploring alternative multi-modal adaptors [1, 3]? Why is CLS set as the query instead of the key-value pair?

2.2 The paper introduces various metrics for evaluating model performance, but lacks explanations for each metric. For instance, in lines 446-449: “resulting in lower sound quality, diversity, and poorer synchronization,” it is unclear which metrics specifically measure sound quality, diversity, or synchronization. Additionally, the statement “the channel-wise fusion of the visual conditioning still aids in synchronization” lacks experimental evidence to substantiate this claim.

2.3 Ambiguous phrases like “we believe” (lines 222, 483) and “might lead to” (lines 77, 483) appear multiple times in the paper. Clear support or reasoning should be provided for these assertions.

3. Presentation and Writing: There are some presentation and writing issues within the paper.

3.1 The introduction (line 63) highlights tackling “Integration of foley and sound 	effects,” yet no further details or experiments addressing this topic are provided in the rest of the paper.

3.2 The temporal shift method introduced in Section 3.2 is motivated as a significant contribution, but it lacks a clear explanation. Additionally, some symbols, like "m," are redefined multiple times, "C'" is not defined, which may cause confusion for readers.


4. Experimental Comparison: A main weakness of the paper is lack of the experimental comparisons, which include only one baseline method.

4.1 A simple baseline could have been constructed by combining an existing video understanding model with a music generation model, similar to the approach in [2, 6].

4.2 The experiments omit comparisons with several relevant state-of-the-art methods, such as Diff-BGM [5], VidMuse [6], and Dance2Music-Diffusion [7].

4.3 The M^2Ugen method shows comparable or superior results in terms of audio quality (Table 2). Fine-tuning this method on the dataset used in this paper could provide additional insight into its performance.

References: 
[1] Liu S, Hussain A S, Sun C, et al. M$^{2}$UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models[J]. arXiv preprint arXiv:2311.11255, 2023.

[2] Lin Y B, Tian Y, Yang L, et al. VMAS: Video-to-Music Generation via Semantic Alignment in Web Music Videos[J]. arXiv preprint arXiv:2409.07450, 2024.

[3] Zhang R, Han J, Liu C, et al. Llama-adapter: Efficient fine-tuning of language models with zero-init attention[J]. arXiv preprint arXiv:2303.16199, 2023.

[4] Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[C]//International conference on machine learning. PMLR, 2021: 8748-8763.

[5] Li S, Qin Y, Zheng M, et al. Diff-BGM: A Diffusion Model for Video Background Music Generation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 27348-27357.

[6] Tian Z, Liu Z, Yuan R, et al. VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling[J]. arXiv preprint arXiv:2406.04321, 2024.

[7] Zhang C, Hua Y. Dance2Music-Diffusion: leveraging latent diffusion models for music generation from dance videos[J]. EURASIP Journal on Audio, Speech, and Music Processing, 2024, 2024(1): 48.

### Questions
In addition to the weaknesses, here are some points that raise further confusion or seem inconsistent in the paper:

1.Claim on Previous V2M Methods (lines 39-41): The authors claim that "Previous V2M methods focus on global features," presenting this as a limitation of past approaches. However, this appears inconsistent with prior work, as several existing methods focus on local clip features for training. For instance, V2Meow [1] and VMAS [2] emphasize local clip-based features, while VidMuse [3] captures both local and global features through long-short-term modeling. The authors should clarify and provide evidence to support their assertion about the emphasis on global features in previous V2M approaches.

2.Choice of Beat Synchronization Metrics and Exclusion of Dance Video Music Generation for Comparison: The authors select Beats Coverage Score (BCS) and Beats Hit Score (BHS) as metrics to evaluate beat synchronization, following the approach in [4] (line 346), which specifically targets music generation for dance videos. However, the authors then claim in line 364 that "D2M-GAN are not considered for comparison because their scope of application differs from ours." If dance-related videos are outside MuVi’s intended scope, it is unclear why dance-specific metrics are being applied for evaluation. This raises a need for clarification.

3.Choice of MuVi(beta) Setting for Comparison: The paper claims "use CLIP-ViT(base-patch16) and the attention pooling adaptor as the visual encoder" for MuVi(beta) (lines 366-367). However, Table 1 shows that the VideoMAE V2 with a Softmax adaptor yields better results for this setting. It is unclear why a suboptimal setting was selected for MuVi(beta), as this choice could impact the fairness and interpretability of the comparisons. An explanation from the authors on the rationale for this choice would provide more clarity.

[1] Su K, Li J Y, Huang Q, et al. V2Meow: Meowing to the Visual Beat via Video-to-Music Generation[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(5): 4952-4960.

[2] Lin Y B, Tian Y, Yang L, et al. VMAS: Video-to-Music Generation via Semantic Alignment in Web Music Videos[J]. arXiv preprint arXiv:2409.07450, 2024.

[3] Tian Z, Liu Z, Yuan R, et al. VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling[J]. arXiv preprint arXiv:2406.04321, 2024.

[4] Zhu Y, Olszewski K, Wu Y, et al. Quantized gan for complex music generation from dance videos[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022: 182-199.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel video-to-music generation method that improves semantic similarity and rhythmic consistency by enhancing the visual encoder and adopting a new contrastive learning pretraining approach. The authors also conducted a series of ablation studies to examine the specific impact of different modules on the model’s performance.

### Strengths
The writing in this paper is clear, and the motivation aligns well with intuition. Overall, the experiments are reasonably thorough and adequately conducted.

The assumptions about video-to-music generation are sensible. The authors address key challenges of the task, such as temporal semantic consistency in videos, rhythm alignment, and distinguishing between sound effects and music. Therefore, the proposed improvements seem well-suited for tackling these issues.

The evaluation metrics are also well-chosen, covering metrics that sufficiently measure the model’s performance across various aspects. The experimental results are convincing.

### Weaknesses
The paper includes an additional section on the in-context learning (ICL) capability of music generation models, which I feel deviates from the main theme of the proposed task. From the limited experiments presented, this section does not add to the paper’s persuasiveness, nor does it convincingly demonstrate actual ICL capabilities. I suggest removing this contribution from the paper.

The discussion on semantic synchronisation is lacking. The paper initially introduces an example: “the style, melody, and emotions of the music will evolve in harmony with the video content,” and Figure 2 also touches on this aspect. The authors attempt to measure semantic synchronisation using the SIM metric and imply that Section 4.2 will discuss this metric in detail. However, I found no mention of this metric in Section 4.2. Thus, I believe the discussion on semantic synchronisation is relatively insufficient. I recommend further analysis of the model’s performance in semantic synchronisation, perhaps by including additional case studies.

Training data concerns for the music generation model. I am curious about one particular issue: since the Jamendo dataset does not include semantic variations, requiring it to generate music that aligns with video semantic changes is effectively an out-of-distribution (OOD) problem. How do the authors plan to address this point?

Choice of baseline. I fully understand the difficulty in comparing to baseline models due to the lack of open-source availability. In this situation, using M2UGen as a baseline is acceptable. However, I would like to note that M2UGen is a weak baseline and performs poorly on the video-to-music (V2M) task. While not a strict requirement, I noticed that VidMuse has released its code. Considering that this baseline is also prominently discussed in the paper, I encourage the authors to include a comparison with it.

### Questions
N/A

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
3

### Summary
The paper proposed a method for video to music generation. There exist not many previous works that utilizes the concept of (video)sequence-to-(music)sequence generation. Most of the previous works tackled the video-to-music generation task as mediaContent-to-sequence task, so the global video features has been used for generating music. Previous works that used the concept of the sequence-to-sequence was mainly studied in the dance-to-music generation task. Therefore, in this paper, the authors tried to generate music using frame-level video information so that each the generated music frame is synchronized with the video frames. To do this, they mainly suggested two techniques which are an visual adaptor module and contrastive video-music pre-training. Visual adaptor aggregates frame-level video features to match with music frames. And, contrastive video-music pre-training is aimed to make synchronization of music beat and video event while preserving overall mood/style coherency. Both the strategies were verified to be effective through the ablation studies.

### Strengths
The proposed visual adaptor and pre-training technique with the two negative samplings seems to be working for better modeling synchronization between video and music.

### Weaknesses
In Introduction, "Integration of foley and sound effects" paragraph seems not necessary. The paper does not tackles foley sound generation neither sound effects. If we listen to the demo samples, It's more like rhythmic synchronization + bpm modulation, not foley or sound effects.

### Questions
In Section 4.1, the authors noted that they used MTG-Jamendo Dataset + 33.7K music tracks from the internet. I think the reason why they have used more 33.7K music tracks from the internet should be explained more in detail. 

Also, if we see Appendix, the collected video mostly includes Disney, Tom and Jerry, and Silent Films. I think this fact should also be described in the paper well since the proposed method is valid only on these kind of video contents for now. (The authors mentioned that for the video that contains vocal singing, they excluded the vocal part through source separation technique, however, if there exist some narration or voice of the actors, the proposed technique will not be valid unless they delete speech parts.)

### Soundness
3

### Presentation
3

### Contribution
2
