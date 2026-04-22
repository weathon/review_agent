# HiViBiX: Hierarchical Visually-informed Binaural Audio Generation using Ambisonics

- Avg Score: 4.80
- Decision: Reject
- Scores: 4, 6, 4, 8, 2

## Abstract
Binaural audio, a specialised form of stereo sound, provides depth and spatial localisation for highly immersive listening experiences, making it fundamental in modern entertainment. Prior research has largely relied on visual cues to directly adapt mono signals into binaural or to estimate transfer functions that induce spatiality. In contrast, we introduce HiViBiX, a novel framework that redefines the music representation by predicting first-order Ambisonics channels, which explicitly control the spatial positioning of the audio components in the generated binaural signal. Unlike existing multimodal approaches that extract spatial cues exclusively from full-frame RGB images, HiViBiX incorporates a hierarchical visual encoder that jointly models local music sound sources and their spatial depth with global environmental context. This design enables richer multimodal grounding and more precise spatialization. Extensive experiments on three widely used musical benchmarks: FAIR-Play, Music-Stereo, and YT-Music demonstrate that HiViBiX establishes new state-of-the-art performance for mono-to-binaural generation. We also show that our method achieves good results in out-of-domain context, using a simple adapter. Samples are available in the following repository: \href{https://hivibix.vercel.app}{https://hivibix.vercel.app}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents a new framework for image-conditioned mono-to-binaural conversion, HiViBiX, that redefines the audio representation by predicting first-order Ambisonics channels. HiViBiX jointly models local sound sources, their spatial depth, and global environmental context. Comprehensive experiments on three widely used benchmarks (FAIR-Play, Music-Stereo, and YT-Music) show that HiViBiX outperforms previous models in the mono-to-binaural generation task. Generated samples are also provided, which support the paper.

### Strengths
1. Although there is room for improvement in presentation, the paper itself is written well enough to make readers understand the authors' motivation, the proposed method, and the experimental results.
2. The proposed framework is well-designed. The ablation studies are comprehensive and support the design.
3. The experiments on three benchmarks are comprehensive, and both the quantitative and qualitative results are good.

### Weaknesses
1. Although the proposed framework is well-designed, it assumes that a sounding object is bound with a person. This assumption would work well on the three specific benchmarks (FAIR-Play, Music-Stereo, and YT-Music), which are related to music performance. On the other hand, it is doubtful that the framework is generalizable to open-domain videos. Even though the HiViBiX model outperforms previous models on the three benchmarks, previous models might perform better in open-domain scenarios.
2. L.218 says, "Because the input-output representations are not semantically similar, we omit the skip connections that are common in such settings." However, I disagree that the input-output representations are not semantically similar for now. The representations should be semantically similar, though they would express different acoustic aspects. The current explanation is confusing to me.
3. This is a minor weakness, but the paper fails to cite recent papers about spatial audio generation (not mono-to-binaural generation) as related works. Citing the following papers will organize related lines of studies and clarify the position of this paper.
    - Sun et al., "Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation", ICLR 2025. https://arxiv.org/abs/2410.10676
    - Kim et al., "ViSAGe: Video-to-Spatial Audio Generation", ICLR 2025. https://arxiv.org/abs/2506.12199
    - Liu et al., "OmniAudio: Generating Spatial Audio from 360-Degree Video", ICML 2025. https://arxiv.org/abs/2504.14906

### Questions
I have questions/concerns about the paper, which I provided in "Weaknesses". I would appreciate it if the authors could share their thoughts on them.

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
5

### Summary
The authors introduce HiViBiX, a novel framework that redefines the audio representation by predicting first-order Ambisonics channels, which explicitly control the spatial positioning of audio components in the generated binaural signal.

### Strengths
1. The vsualization looks good and is helpful for understanding.
2. The writing is clear and concise.
3. A demo page is provided for readers.

### Weaknesses
1. Citation style: The manuscript mixes \citet{} and \citep{}, which hurts readability and does not conform to ICLR’s guidelines. Please standardize to a single in-text citation style consistent with ICLR (e.g., \citep{} for parenthetical citations and \citet{} for narrative citations) and apply it uniformly across the paper, including figures and tables.

2. Baselines and recency: The closest baseline listed is [CCStereo](https://github.com/SheldonTsui/PseudoBinaural_CVPR2021) dated Jan 6, 2025. And only two models from the last two years are compared. These are insufficient for a fair state-of-the-art comparison. If certain methods are excluded, explicitly justify and, where possible, include their reported numbers on the same benchmarks.

3. Evaluation standard and split: PseudoBinaural reorganizes the evaluation set and provides new non-overlapping splits in the new_splits directory. Methods after PseudoBinaural should be evaluated on those splits separately from earlier protocols. In line 351, specify exactly which evaluation standard you use and cite the paper. Otherwise, it is hard to recognize the evaluation method.

4. Demo page evidence: The demo page currently lacks side-by-side comparisons with popular baselines, making it hard to assess improvements and raising the concern of cherry-picked examples.

5. Notation clarity: Although bold/underline semantics may be known to some readers, including me, please add a brief one-line note in the main text explaining all typographic marks (e.g., bold = best, underline = second-best). Ensure consistency across all tables.

6. Title specificity: “Binaural Audio Generation” is now ambiguous due to the rise of semantic generative models [1,2,3]. To avoid a minor misclaim, revising the title to explicitly say “mono-to-binaural” can help readers get the outline of the paper. Also, you should better cite these papers in the related works section and explain the “mono-to-binaural generation”.

7. Choice of expert models: DINOv2 depth is not the common choice for depth estimation; recent depth experts (e.g., UniDepth series [4,5] or Depth-Anything series [6,7]) are stronger default baselines. Similarly, YOLOv8 is no longer the most up-to-date detector like [8,9]. Please justify these choices or replace them with current state-of-the-art models. If you retain older experts for efficiency or compatibility, state the trade-offs (speed, memory, training stability) and provide controlled comparisons.

[1] Sun, P., Cheng, S., Li, X., Ye, Z., Liu, H., Zhang, H., ... & Guo, Y. (2024). Both ears wide open: Towards language-driven spatial audio generation. arXiv preprint arXiv:2410.10676.

[2] Marinoni, C., Gramaccioni, R. F., Shimada, K., Shibuya, T., Mitsufuji, Y., & Comminiello, D. (2025). StereoSync: Spatially-Aware Stereo Audio Generation from Video. arXiv preprint arXiv:2510.05828.

[3] Liu, H., Luo, T., Luo, K., Jiang, Q., Sun, P., Wang, J., ... & Xue, W. (2025). Omniaudio: Generating spatial audio from 360-degree video. arXiv preprint arXiv:2504.14906.

[4] Piccinelli, L., Yang, Y. H., Sakaridis, C., Segu, M., Li, S., Van Gool, L., & Yu, F. (2024). UniDepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10106-10116).

[5] Piccinelli, L., Sakaridis, C., Yang, Y. H., Segu, M., Li, S., Abbeloos, W., & Van Gool, L. (2025). Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110.

[6] Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10371-10381).

[7] Chen, S., Guo, H., Zhu, S., Zhang, F., Huang, Z., Feng, J., & Kang, B. (2025). Video depth anything: Consistent depth estimation for super-long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 22831-22840).

[8] Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., & Han, J. (2024). Yolov10: Real-time end-to-end object detection. Advances in Neural Information Processing Systems, 37, 107984-108011.

[9] Khanam, R., & Hussain, M. (2024). Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725.

### Questions
The questions are detailed in the Weaknesses.

My main concern is the lack of sufficient fair comparison. But I think it can be easily improved.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HiViBiX, a framework for visually-informed mono-to-binaural audio generation. The method's core is a hierarchical visual encoder (HiVi) that leverages a rich set of multi-modal visual cues—including local object features, global scene context, depth, and positional information—to guide the generation process. A key aspect of the proposed method is its use of an Ambisonics-inspired intermediate representation. Instead of directly predicting the stereo channels, the network is trained to predict first-order Ambisonics-like channels (X and Y), which are then combined with the input mono signal (treated as the W channel) via a fixed decoding formula to synthesize the final binaural audio. The authors demonstrate state-of-the-art results on several benchmarks through extensive objective and subjective evaluations.

### Strengths
The paper is well-structured and presents its proposed system, HiViBiX, in a clear manner. The idea of incorporating an Ambisonics-based architectural prior is conceptually sound, and the design of the hierarchical visual encoder demonstrates a thoughtful approach to feature integration. The ablation study is organized and helps to understand the contribution of different components within their proposed pipeline.

### Weaknesses
Despite its clear presentation, the paper suffers from significant weaknesses regarding the scope and significance of the task, its novelty, and the overall experimental validation, which collectively place it below the acceptance threshold.

1.  **Limited Significance Due to Small-Scale Datasets:** A fundamental issue lies in the scope of the problem as defined by the available datasets. The benchmarks used (FAIR-Play, Music-Stereo, YT-Music) are relatively small and contain mostly constrained, non-diverse scenarios (e.g., static musical performances). This raises questions about the actual difficulty and real-world significance of the task. Achieving state-of-the-art results on small, homogenous datasets does not necessarily translate to robust performance on the vast and complex variety of "in-the-wild" videos. The paper would be much stronger if it acknowledged this limitation and either tested its method on a more challenging, larger-scale custom dataset or tempered its claims about general applicability.

2.  **Incomplete Literature Review and Questionable Task Formulation:** The paper fails to position itself within the most current research landscape, critically omitting highly relevant works. This includes:
    *   **`OmniAudio` (Liu et al., 2025)**, which directly generates spatial audio (FOA) from 360-degree video.
    *   **`Both Ears Wide Open` (Sun et al., 2024)**, which explores controllable spatial audio generation from language.
    The existence of methods like `OmniAudio` that perform direct Video-to-FOA generation challenges the core premise of this paper. The authors must justify why the Mono+Video-to-Binaural pipeline remains a significant problem to solve when direct spatial audio synthesis from video is an emerging and potentially more powerful paradigm. Without this crucial discussion and comparative analysis, the motivation for the work is weakened.

3.  **Overstated Novelty and Writing Clarity:** The paper's framing of its contributions could be more precise and less grandiose.
    *   **Novelty of HiVi Encoder:** The HiVi encoder is presented as a primary innovation, yet it is fundamentally an intricate assembly of existing, powerful models (YOLO, CLIP, DINOv2). While the integration is effective, the paper does not sufficiently articulate a new, generalizable principle beyond the task-specific observation that "combining more features helps." The contribution feels more aligned with clever system engineering than with fundamental research on representation learning.
    *   **Ambiguous "Learning" of Ambisonics:** The claim of "internally learning" Ambisonics channels is imprecise. The model is not supervised with Ambisonics ground truth but rather learns to produce intermediate representations that fit a proxy objective. The writing should more accurately describe this as leveraging an Ambisonics-based architectural prior, rather than implying the model learns the physical properties of the sound field.

4.  **Flawed Experimental Design and Limited Generality:**
    *   **Fragile Design Assumptions:** The method's architecture is brittle. Its heavy reliance on YOLO-based "person" detection makes it unsuitable for any scenario without a human performer, a major limitation for a general V2A system. Similarly, its dependence on a single static frame ignores temporal dynamics, a critical aspect of video. These design choices severely restrict the model's practical utility. The experiments should have included an analysis of these failure modes to provide a more honest assessment of the method's capabilities.
    *   **Inconclusive User Study:** The user study, while a welcome addition, is weakened by its comparison set. By omitting the current SOTA baseline (`CCStereo`), it fails to provide a conclusive statement on the method's perceptual quality relative to its strongest competitor.

### Questions
1.  Given that recent works like `OmniAudio` pursue direct Video-to-Spatial-Audio generation, could you elaborate on the continued significance of the Mono+Video-to-Binaural task? What are the scenarios where your approach would be uniquely advantageous?
2.  The datasets used are relatively small and constrained. How can you be confident that your method's strong performance is not simply a result of overfitting to the specific biases of these benchmarks, and how would you expect it to perform on a much larger, more diverse "in-the-wild" video dataset?
3.  The HiVi encoder is a complex integration of multiple off-the-shelf models. Can you distill the core, generalizable insight from this design that is not just specific to this task or these datasets?
4.  Why was the current SOTA competitor, `CCStereo`, excluded from the perceptual user study? Without this comparison, the claims of subjective superiority are not fully supported.

### Soundness
2

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
4

### Summary
This paper presents an approach to generate binaural audio from mono audio and RGB images. The novel contribution lies in predicting ambisonics to model positions of audio sources explicitly. The authors also include Yolo, for finding humans playing instruments in music videos, and Dino to get depth information. All these information are mapped into features which are then combined via a hierarchical attention module with positional features. The approach beats the state of the art with clear distance.

### Strengths
The idea to predict explicity ambisonics is interesting as it allows to build systems based on a large foundation of prior work on ambisonics but can benefit from including information from the visual signal.

The performance when comparing to real Ambisonics on the YT-Music dataset seem to indicate the approach performs well at this complex prediction task.

The way how the individual modules are included may not be groundbreaking but are still a solid engineering achievement.

### Weaknesses
The largest weakness seems to be not the fault of the author. The approach is only tested on music datasets yet the paper describes the contribution as predicting binaural audio in a way that it could make it seem as if it works for arbitrary scenes and arbitrary sounding objects. However, comparing the state of the art in that field, approaches like CCStereo, CLUP and CNC seem to do the same thing. Therefore, it would be good to make clear in the introduction that this is tailored for binauralization of music and therefore limitations, such as using Yolo, a negligable. Still, for the purpose of acceptance this is not a major issue as it fits well in the landscape of that sub-field.

The related work seems to ignore room impulse response prediction completely. There are many approaches who predict RIRs based on audio-visual data and which can be used to create binaural data. Some approaches predict binaural RIRs directly like [1] or [2]. Sometimes an intermediate Nerf representation is created to improve the quality. While the input may be too different, needing meshes of rooms or panoramic images, to be directly comparable, these approaches should be mentioned in the related work.

The examples contain exclusively fixed-camera setups and rarely moving sound sources. That makes it very hard to judge sound localization qualitatively and impossible to judge potential issues with moving sound sources. Binauralization beyond music, e.g. from ambient sounds, object sounds or simple human conversation, can also not be judged based on the selection of examples. 

It seems non-intuitive why the stereo waveform would be downmixed to Mono. While this makes the computation easier, there must be spatial information which would be useful in the stereo signal. It is nice that the framework can solve this more complex task but comparing against stereo seems like a straightforward thing to do.

Is the reason why the research was done mostly on music datasets because Yolo finds humans but more effort is needed to define sounding objects and to go through Yolo detections to find them? This may mean that this approach is biased to work well on humans operating a musical instrument but would not generalize well to other situations. This seems like a heavy limitation.


[1] Ratnarajah, A., & Manocha, D. (2024, March). Listen2scene: Interactive material-aware binaural sound propagation for reconstructed 3d scenes. In 2024 IEEE Conference Virtual Reality and 3D User Interfaces (VR) (pp. 254-264). IEEE.

[2] Su, K., Chen, M., & Shlizerman, E. (2022). Inras: Implicit neural representation for audio scenes. Advances in Neural Information Processing Systems, 35, 8144-8158.

### Questions
Why are the skip connetions of the UNet not used? Many approaches have the same issue, of representations which are not similar and yet skip-connections have been shown to improve the quality for no obvious reason. One example where depth is predicted from audio and skip connections seem to show improved performance is [3].


[3] Brunetto, A., Hornauer, S., Stella, X. Y., & Moutarde, F. (2023, October). The audio-visual batvision dataset for research on sight and sound. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 1-8). IEEE.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes HiViBiX, a novel framework for image-conditioned mono-to-binaural audio generation that predicts first-order Ambisonics (FOA)-like channels (W, X, Y) as intermediate representations. It introduces a Hierarchical Visual Encoder that extracts both global and local visual cues, combining RGB, depth, and positional features, to guide audio spatialization. HiViBiX is evaluated on FAIR-Play, Music-Stereo, and YT-Music, outperforming several baselines (e.g., CMC, CLUP, CCStereo) across STFT, ENV, and SNR metrics. Ablation studies support the benefits of the hierarchical visual and Ambisonics components.

### Strengths
1. The core idea of predicting Ambisonics-like channels and gains as a learnable intermediate step is a strong contribution. The "Ambisonics FiLM" layer (Algorithm 1) provides a physics-inspired way to combine these components with the mono input, which is more structured than direct end-to-end prediction.
2. The combination of local and global cues (scene, depth, and position) is well-motivated. The use of CLIP and DINOv2 provides powerful multimodal grounding.
3. The paper achieves strong quantitative improvements on three standard benchmarks (FAIR-Play, Music-Stereo, YT-Music) when compared against a good range of recent methods like CLUP and CCStereo.

### Weaknesses
1. I find this paper pretty similar to the CCStereo. The key difference is just the visual encoder and the modality formulation (using the ambisonics formulation vs. left-right difference). 
2. The paper is missing comparisons to several highly relevant and recent works in visually-conditioned spatial audio. Specifically, OmniAudio [Liu et al. 2025] and ViSAGe [Kim et al. 2025] are critical baselines. These are concurrent or very recent, but their inclusion is necessary to truly contextualize the performance of HiViBiX. The related work section would also be strengthened by discussing "Both Ears Wide Open" [Sun et al. 2024].
3. The experimental evaluation is limited to FAIR-Play, Music-Stereo, and YT-Music. While these are standard, they are heavily focused on musical performances. This introduces a strong domain bias. To truly validate a model aimed at "Auditory Reality"  and general spatial audio, the evaluation must include more diverse scenarios. I would strongly suggest the authors include evaluations on datasets like YT-Ambigen [Kim et al. 2025] or Sphere360 [Liu et al. 2025], which offer more complex acoustic scenes and non-musical content.
4. The HiVi encoder's local features are entirely dependent on YOLOv8 detecting the "person" label. The authors justify this by stating "most instruments require a human operator". In my opinion, this is a significant design limitation and a likely failure mode. It's unclear how the model would handle non-human sound sources (e.g., a loudspeaker, a vehicle, an animal) or even off-screen sounds where the 'person' is not visible. This dependency seems to contradict the goal of a general-purpose spatial audio generator. A thorough analysis of this limitation is missing.
5. This paper uses a deterministic mapping between input and output, which is clearly not reasonable and will lead to over-smoothing issues. 
6. The model relies on a single anchor image $v^{(i)}$ to condition the generation of an entire audio clip. This static-image-to-clip approach is acknowledged as a limitation, but its impact is significant. It means the model cannot capture any visual dynamics (e.g., a musician walking across the stage, a car moving) that would be critical for realistic spatial audio rendering. This is a key weakness, especially when missing baselines (like ViSAGe) do explicitly model video dynamics.
7. The paper's claim of learning intermediate Ambisonics-like channels is a key part of its novelty. However, the validation for this is weak. The authors state they have access to ground-truth Ambisonics data for the YT-Music dataset. However, the validation is restricted to a simple visual comparison in Appendix E (Fig. 7). A quantitative evaluation (e.g., STFT distance) between the predicted $\hat{X}$/$\hat{Y}$ channels and the ground-truth $X$/$Y$ channels would be a far more convincing demonstration that the model is learning the correct intermediate representation.

### Questions
1. Could you please comment on the omission of recent baselines like OmniAudio and ViSAGe? Given that some of these model temporal dynamics, how do you expect HiViBiX's static-image approach to compare?
2. Given that you have access to ground-truth Ambisonics data for the YT-Music dataset13, could you provide a quantitative evaluation comparing your predicted $\hat{X}$/$\hat{Y}$ channels to the ground-truth $X$/$Y$ channels? This would be much stronger than the visual comparison in Figure 7.
3. The HiVi encoder's reliance on "person" detection seems to be a major limitation. How does the model perform on scenes with non-human sound sources (e.g., a radio, a barking dog) or on-screen instruments without a visible person?
4. Could you elaborate on why more diverse, non-musical datasets (like YT-Ambigen or Sphere360) were not used for evaluation? The current selection seems heavily biased towards music, which may not reflect general "auditory reality".

### Soundness
2

### Presentation
3

### Contribution
2
