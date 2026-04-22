# AlignSep: Temporally-Aligned Video-Queried Sound Separation with Flow Matching

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 2, 8

## Abstract
Video Query Sound Separation (VQSS) aims to isolate target sounds conditioned on visual queries while suppressing off-screen interference—a task central to audiovisual understanding. However, existing methods often fail under conditions of homogeneous interference and overlapping soundtracks, due to limited temporal modeling and weak audiovisual alignment.
We propose \textbf{AlignSep}, the first generative VQSS model based on flow matching, designed to address common issues such as spectral holes and incomplete separation. To better capture cross-modal correspondence, we introduce a series of temporal consistency mechanisms that guide the vector field estimator toward learning robust audiovisual alignment, enabling accurate and resilient separation in complex scenes.
As a \textit{multi-conditioned generation} task, VQSS presents unique challenges that differ fundamentally from traditional flow matching setups. We provide an in-depth analysis of these differences and their implications for generative modeling. To systematically evaluate performance under realistic and difficult conditions, we further construct \textbf{VGGSound-Hard}, a challenging benchmark composed entirely of separation cases with homogeneous interference and strong reliance on temporal visual cues.
Extensive experiments across multiple benchmarks demonstrate that AlignSep achieves state-of-the-art performance both quantitatively and perceptually, validating its practical value for real-world applications. More results and audio examples are available at: \url{https://AlignSep.github.io}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors address the persistent challenge of temporal misalignment in video-queried sound separation, where separated audio often contains sounds from the correct source but with slight temporal shifts relative to the visual cues. To this end, they propose ALIGNSEP, a novel framework that recasts the separation task from a deterministic regression problem into a conditional generative modeling problem. The core of the contribution of this paper is the use of continuous-time flow matching, a technique where the model learns a vector field (a flow) that transforms a simple noise distribution into the complex conditional distribution of the target audio, given the input mixture and the silent video query. By explicitly conditioning the entire generative flow on the video features, the model is forced to learn a tight and continuous temporal coupling between the visual and auditory signals, thereby directly mitigating the alignment issues that can affect traditional masking-based or regression-based approaches. Their experiments show that this flow-based generative method achieves state-of-the-art separation performance while producing audio with significantly improved temporal alignment to the corresponding video.

### Strengths
1. The paper's primary contribution is its fundamental reframing of the separation problem, shifting away from traditional deterministic methods. Instead of formulating video-queried separation as a regression task, such as estimating a time-frequency mask, the authors reconceptualize it as a conditional generative modeling problem. This is a more powerful and principled approach, as it aims to model the entire conditional probability distribution of the target audio. By using continuous-time flow matching, the model learns to transform a simple noise distribution into this complex target distribution, which it seems from the authors’ experiments that is better suited to handle the ambiguity and variability in audio signals compared to predicting a single point estimate.
2. A significant strength is that the proposed framework is explicitly designed to solve the specific, well-known issue of temporal misalignment between video and separated audio. The core mechanism which is conditioning the entire generative flow on the video query, seems that forces a continuous and tight temporal coupling between the modalities from the very beginning of the generation process. This represents a more elegant and integrated solution than methods that might require post-processing or auxiliary alignment losses, as it embeds temporal consistency directly into the generative dynamics of the model itself.
3. Finally, the thorough empirical evaluation successfully demonstrates that the proposed generative approach does not come at the cost of raw separation quality. The paper shows that ALIGNSEP achieves state-of-the-art performance on standard separation metrics while also providing a clear, measurable improvement in temporal alignment. This dual achievement is critical, as it proves that the method is not a narrow solution to a single problem but a more robust overall framework that advances the field on multiple fronts, validating the effectiveness of the flow-matching approach for this complex audio-visual task.

### Weaknesses
Although I do not think that the paper has major flaws, there are some weaknesses and further explorations that the authors can consider to further improve the impact of their paper. I will try to write all these things down with decreasing order of significance.

1. I would like to see the authors analyze more what is happening with the negative examples, meaning what this model is doing when it is presented with only off-screen sounds as analyzed in [A]. Without knowing how robust the model is, I can imagine cases where the ODE solver could break by needing to estimate a zero waveform. One can question the stability of the ODE solver in such a degenerate case; a model trained to map a noise distribution to a complex signal manifold might struggle or even fail when the learned vector field is conditioned to collapse to a single point (the origin). This analysis is in my opinion really important for real-world use-cases cause many parts of the videos do not contain any on-screen sound for long periods of time.
2. The application of flow matching is central to the paper's contribution, but in my opinion there is a lack of principled justification or empirical comparison against other relevant conditional generative frameworks, such as score-based diffusion models. It is not demonstrated whether the specific choice of continuous-time flow matching provides a unique advantage for temporal alignment over these alternative generative techniques. The paper would be significantly strengthened by an ablation study that compares different generative modeling families, which would help to isolate whether the observed benefits stem from the specific dynamics of flow matching itself or merely from the general shift from a deterministic to a generative formulation.

[A] Tzinis, E., Wisdom, S. and Hershey, J.R., 2022. Don’t Listen to What You Can’t See: The Importance of Negative Examples for Audio-Visual Sound Separation. arXiv preprint arXiv:2209.12934.

I would gladly increase my score if the authors work properly to address the above issues to a proper degree.

### Questions
The proposed approach is really interesting and I was considering what could happen if you apply the same ODE solver in the video side by using the Gaussian interpolation at the video embedding frames. Would you expect a similar performance on the video side?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces AlignSep, a generative model for Video-Queried Sound Separation based on conditional flow matching. The primary goal is to address common failures in existing methods, particularly when dealing with acoustically similar (homogeneous) on-screen and off-screen sound sources. The authors argue that prior work relies too heavily on semantic cues and fails to model fine-grained temporal alignment between audio and video. AlignSep is designed to overcome this by using a temporally-aware visual encoder and a generative framework that mitigates issues like spectral holes. To facilitate evaluation, the paper also proposes a new challenging benchmark, VGGSound-Hard, specifically curated with examples of homogeneous interference. Experiments show that AlignSep outperforms existing methods on several benchmarks, especially in metrics related to temporal alignment and perceptual quality.

### Strengths
1. The paper correctly identifies and tackles a key weakness in existing VQSS methods: the failure to separate homogeneous sound sources due to a lack of temporal modeling. This is an important problem.
2. The proposed VGGSound-Hard is a strong contribution. By creating a testbed where semantic cues are insufficient, it will enable more rigorous evaluation of temporal alignment capabilities in future VQSS models.
3. Includes human MOS scores in addition to semantic and temporal metrics.

### Weaknesses
1. My primary concern is the complete absence of SDR and SI-SNR. The authors justify this by stating these metrics "correlate poorly with human perception" and are sensitive to minor waveform deviations in generative models. While these are known limitations, it is standard practice in the audio separation community to report them, even with caveats. Omitting them entirely makes it difficult to compare AlignSep to the vast body of literature in source separation and understand its performance in terms of signal reconstruction fidelity. Relying solely on embedding-based scores (CLAP, ImageBind) and MOS provides an incomplete picture. These metrics measure semantic/temporal consistency but not the faithfulness of the separated audio to the original source signal.
2. Section 3.3 introduces the concatenation-based temporal fusion and "feedforward Transformer" without self-attention. However, it remains unclear how much each design choice contributes: 1) What happens if temporal alignment is omitted or replaced with standard cross-attention? 2) Is the observed improvement mainly from temporal encoding or from generative modeling itself? An ablation comparing different temporal fusion strategies (concatenation vs. attention) would be valuable.
3. The choice of the CAVP visual encoder is highlighted for its ability to capture temporal correlations. A crucial ablation would be to replace CAVP with a more semantic, less temporally-aware encoder (e.g., a standard CLIP image encoder applied per-frame) within the AlignSep framework. This would directly validate the claim that the specific choice of a temporally-supervised visual encoder is responsible for the large gains in the $T_{A-V}$ metric.
4. The authors state they "adapt" OmniSep and CLIPSep by "segmenting the mixed audio according to each frame's semantic information". This process is vague and sounds like a heuristic post-processing step. It is unclear if this is a fair or optimal way to allow these models to use frame-rate information. This adaptation could be unfairly disadvantage the baselines, potentially leading to AlignSep's superior performance in Figure 3.
5. While the new benchmark is a great contribution, the paper provides almost no details on its construction. How were samples selected from VGGSound? What is the size of the test set? What are the specific criteria for "strong reliance on fine-grained temporal visual cues"? Without these details, the benchmark is not reproducible, and the community cannot fully adopt it. These details should be included in the appendix.

### Questions
See Weaknesses

### Soundness
2

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
The paper proposes a flow-matching generative model for video-queried sound separation that explicitly preserves temporal alignment between visual motion and target audio. It uses a temporally synchronized visual encoder, a VAE audio latent space, and a temporally aligned vector-field estimator. The authors also introduce VGGSound-Hard, a benchmark with homogeneous on/off-screen interference requiring fine-grained temporal cues. The paper achieves SOTA semantic and temporal metrics and higher MOS on MUSIC-Clean, VGGSound-Clean, and VGGSound-Hard.

### Strengths
- The paper identifies temporal misalignment and mask-method limits for current video query sound separation systems, then introduces the first flow-matching generative model to address them.

- The paper proposes a new hard benchmark. The proposed VGGSound-Hard targets homogeneous on/off-screen interference, stressing temporal grounding beyond prior clean sets.

- The system achieves SOTA semantic and temporal scores and better MOS across MUSIC and VGGSound.

### Weaknesses
- For the architecture design, the “self-attention-free feed-forward Transformer” with simple temporal concatenation may struggle with long-range dependencies or complex multi-object scenes.

- The authors claim that reconstruction-based metrics such as SDR could correlate poorly with human perception. But without standard separation metrics being reported, this will limit comparability to audio-separation literature.

- Experiments are limited to MUSIC/VGGSound variants, broader coverage like more in-the-wild AV datasets would strengthen generalization claims.

### Questions
- Could the author add standard metrics? For example, report SDR, SIR and SAR results.

- Could the author add evaluations beyond MUSIC/VGGSound (e.g., in the wild audio-visual videos), additional qualitative results are also fine to evaluate the generalizability of the proposed method.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents AlignSep, a generative framework for video-queried sound separation based on flow matching. The model explicitly addresses the challenges of disentangling temporally-aligned audio sources from mixed inputs, using visual signals as queries. AlignSep incorporates a temporal consistency mechanism within the vector field estimator, enabling robust cross-modal alignment. The work introduces VGGSound-Hard, a challenging benchmark featuring homogeneous interference that requires fine-grained temporal audiovisual correspondence, and demonstrates AlignSep’s superior performance over existing baselines on multiple quantitative, perceptual, and qualitative metrics.

### Strengths
1. The paper tackles VQSS in settings requiring not just semantic but precise temporal alignment between video and audio, a scenario where many existing methods falter, particularly under homogeneous interference and overlapping sources.

2. The introduction of VGGSound-Hard could be potentially valuable to the audio-visual separation community by operationalizing more realistic, temporally complex scenarios compared to prior datasets.

### Weaknesses
1. Overclaiming and Missing Related Work: The paper substantially overclaims novelty and omits several key prior works in video-queried sound separation (VQSS).
At L95–98, the authors state:

“We revisit the task of video-queried sound separation (VQSS) and provide a detailed analysis of its unique challenges, including homogeneous interference, overlapping soundtracks, and the need for precise audio-visual temporal alignment.”

However, these issues were already identified and discussed in "High-Quality Visually-Guided Sound Separation from Diverse Categories" [1] (Fig. 1), which explicitly analyzes overlapping-sound challenges and shows that generation-based separation outperforms masking-based methods. Furthermore, at L75–77, the paper claims:

“We propose AlignSep—the first generative video-queried sound separation model based on flow matching (Lipman et al., 2022) designed for robust audiovisual separation.”

This statement is inaccurate. The generative approach to VQSS was introduced in [1] (2023), and later extended by [2], which directly applies flow-matching for video-based sound separation. These works are closely related in both goal and methodology, but are entirely unacknowledged in the submission.
Without discussing or comparing against these prior methods, the claimed novelty of AlignSep remains unclear and potentially misleading.

[1] High-Quality Visually-Guided Sound Separation from Diverse Categories
[2] High-Quality Sound Separation Across Diverse Categories via Visually-Guided Generative Modeling

2. Incomplete Comparisons in the Main Table

The main quantitative table should include comparisons with other video-based or multimodal separation models, such as:

[3] Language-Guided Audio-Visual Source Separation via Trimodal Consistency
[4] iQuery: Instruments as Queries for Audio-Visual Sound Separation

Both [3] and [4] are directly comparable in setting and should appear in the benchmark to provide a fair evaluation of AlignSep’s relative performance.

3. Missing Architectural Ablations: Although Table 3 examines the number of denoising steps and Table 5 lists hyperparameters, the paper does not analyze architectural design choices. For example, the rationale behind adopting a self-attention-free Transformer or comparing with different audio-visual encoding architectures. Such ablations are important to understand which design decisions truly contribute to performance improvements.

At present, the paper’s novelty is overstated, and its evaluation is incomplete. The work could become more compelling if the authors (i) properly position AlignSep within the context of prior generative VQSS research, (ii) include stronger baselines for comparison, and (iii) provide architectural ablations to support their design choices.

I am currently leaning toward rejection, but if the rebuttal offers convincing clarification and expanded comparisons, I would reconsider and potentially raise my score.

### Questions
1. How do alternative audio-visual fusion mechanisms (e.g., cross-attention, feature-wise modulation) compare to the concatenation approach used here in the temporal vector field estimator? Are there empirical results or reasons for not adopting these?

2. Have you tested the performance of AlignSep in real-world settings where video-audio correspondence is noisy or ambiguous?

3. Given that the classifier-free guidance scale is set to 4.5, could you explain the choice and its impact on sample quality or diversity?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a method of video-guided sound separation.  Instead of a traditional regression or detection-based approach that masks out the unwanted sound, this paper developed a generative approach that generates the clean-up sound instead.

The paper pointed out that current video-guided sound separation methods have two significant limitations.  First, semantic cues alone can't separate acoustically similar sources.  It requires capturing the temporal alignment of video motions and audio soundwaves. Second, simple time–frequency masking can not handle multiple sources overlapping in both time and frequency.  

The core technical component involves using a flow-matching-based algorithm to regenerate a clean soundtrack conditioned on video features.  

The algorithm outperforms OmniSep and CLIPSep, two of the leading algorithms.  The paper presents a detailed experimental analysis of the proposed method, including a thorough examination of the performance and efficiency tradeoff of the flow-based generative method.  It also developed a new dataset, VGGSound-Hard, which focuses on overlapping space-time interference where temporal grounding is critical for successful separation.

### Strengths
The overall idea is novel.  Treating sound separation as a clean-sound generation process enables this method to surpass the limitations of the current mask-based removal approach.  The integration of video motion cues, including both semantic and lower-level motion, provides the flow-based generation with the essential temporal grounding condition to resolve ambiguity. 

The experimental results are extensive and detailed.  The performance vs efficiency analysis of the flow-based method addressed the practical issue of potential high computation cost.  The new dataset is valuable for future work in this direction.

### Weaknesses
None.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
