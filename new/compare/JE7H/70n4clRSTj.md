---
job_id: 4ba26f0b-0c09-4c33-b606-30b9e220bfd8
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 70n4clRSTj.pdf
paper: 70n4clRSTj.pdf
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper introduces a video-language benchmark focused on temporal reasoning and evaluates modern Video-VLMs, which fits squarely within ICLR’s remit on representation learning, multimodal models, benchmarks, and evaluation.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method / Dataset, Experiments, Results & Discussion, Conclusion, Ethics, Reproducibility) are present, the paper is in English, and the core methodology and experiments are coherent and technically non-trivial. While I identify significant weaknesses, they do not rise to the level of a desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any embedded prompts, hidden instructions to reviewers, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces **SpookyBench**, a synthetic video benchmark designed such that all meaningful information is conveyed **purely through temporal patterns of structured noise**, with individual frames appearing as noise-like. Three categories are considered: temporally encoded text, isolated object images, and dynamic scenes derived from depth maps. The authors show that human participants achieve around **98% accuracy**, while a wide range of state-of-the-art open- and closed-source Video-VLMs (including specialized temporal models and GPT‑4o / Gemini) achieve **0% accuracy** under both direct and chain-of-thought prompting. The benchmark is accompanied by a temporal SNR analysis and a small finetuning study, and is proposed as evidence that current Video-VLM architectures are “time-blind” in the absence of spatial cues.

## Strengths

1. **Clear, focused benchmark idea with strong qualitative appeal.**  
   The central construct of SpookyBench, where content is **invisible frame-wise but emerges only through motion**, is well-motivated and sharply isolates temporal perception. **Figure 2 (left and right panels)** nicely visualizes the key mechanism of opposing foreground/background motion and convincingly shows why static frames look like noise while the moving sequence is recognizable to humans. This is a compelling stress test for existing video models.

2. **Striking empirical gap between humans and current Video‑VLMs.**  
   The main quantitative result in **Table 1** is very clear: across a diverse set of open-source (VideoLLaMA, TimeChat, Qwen‑VL, InternVL, InternVideo, etc.) and commercial models (GPT‑4o, Gemini), accuracy is exactly 0% on all categories and both prompting variants. Combined with the human study in **Table 3** (≈98–99% accuracy, perceptibility ≈4.7/5), the evidence that current Video‑VLMs completely fail on this benchmark is strong and easy to interpret.

3. **Careful temporal signal analysis.**  
   Section 3.3.1 and Equations (1)–(4) define multiple SNR metrics (Basic, Perceptual, Temporal Coherence, Motion Contrast). **Table 2** and later **Table 6** provide concrete numbers showing that frame-based SNR is extremely low (≈ −40 to −60 dB), while temporal coherence and motion contrast can be relatively high. The accompanying visualizations in **Figures 9–11** (motion direction coherence, extracted masks, and motion boundaries of the ant silhouette) effectively illustrate how temporal integration makes shapes salient to humans despite noisy frames.

4. **Systematic examination of some confounders (frame rate and finetuning).**  
   The FPS experiments in **Table 4** and **Table 5** examine human and model performance from 1–30 FPS. Humans degrade gracefully and hit ≈95% at 20–30 FPS, while models stay at 0% throughout, supporting the claim that the failure is not a simple undersampling artifact. Section 4.4 additionally reports finetuning InternVL2.5‑8B and Qwen2‑VL‑7B for 10 epochs on 400 benchmark videos with still 0% test accuracy, arguing that mere exposure does not solve the problem.

5. **Neuroscience‑inspired framing and connection to human temporal perception.**  
   The paper contextualizes the benchmark using neuroscience work on distributed timing mechanisms (Mauk & Buonomano, Paton & Buonomano, Bueti & Walsh). While high-level, this framing is coherent and helps situate the claimed “time blindness” as a gap between artificial and biological vision systems.

6. **Clarity and accessibility of the core idea and dataset design.**  
   Despite some rough edges in the writing, the main construction via **Algorithm 1** and **Algorithm 2** is easy to follow: binary noise patterns, opposing motion for foreground vs background, tileable noise, and controlled speckle size / density. The dataset statistics in **Figure 5** (category distribution) and Section 3.3 are straightforward.

## Weaknesses

I list several substantial concerns, many of which would materially impact the scientific value or the strength of the conclusions.

1. **Extremely coarse evaluation metric and ambiguous labeling / matching.**  
   - The evaluation uses a single “exact match” criterion with a manually defined set of acceptable labels for objects and dynamic scenes (Section 4.1). For text, the requirement is literally the exact word; for objects/scenes, the paper states that “a video showing ‘a man playing basketball’ accepts responses such as ‘playing basketball,’ ‘man’, ‘human’, or ‘woman playing basketball’.”  
   - This is problematic given that VLM outputs are free-form and can contain synonyms (“person shooting hoops”, “guy dribbling”), compound phrases, or minor spelling variation. The authors state that *none* of the models produced any acceptable label, but there is no description of how matching was implemented (case normalization, stemming, token-level inclusion, synonym mapping, etc.).  
   - If matching is literal string equality against a small handcrafted set, 0% accuracy is unsurprising even if models partially perceive the temporal pattern but express it differently. This is especially crucial for Table 1’s headline claim of “0%” across 15+ models. At a minimum, the paper should:  
     - Describe in detail the normalization and matching pipeline,  
     - Provide examples of typical model responses (especially for GPT‑4o / Gemini) to show they truly do not identify the content, rather than simply failing lexical match,  
     - Report looser metrics (e.g., token overlap, embedding similarity, or human-judged correctness).  
   - Without this, the 0% claim reads stronger than the evidence justifies.

2. **Potential mismatch between synthetic stimulus and model input preprocessing not fully controlled.**  
   - The videos are high-resolution binary-noise patterns (960×540, hundreds of frames). Many evaluated models rely on spatial encoders pre-trained on natural images, often coupled with aggressive frame sampling, cropping, and downscaling. The paper does not systematically analyze how these preprocessing pipelines transform SpookyBench stimuli.  
   - For instance, downsampling or interpolation could blur the binary noise, potentially destroying the subtle motion contrast that humans use. The paper claims in **Section 4.3** that changing FPS does not help, but that is only varying the **number** of frames, not the spatial sampling or the model’s choice of frames internally.  
   - A convincing argument that models fail due to “architectural time-blindness” should at least check whether (a) lower resolution / patchification erases the motion signal, and (b) alternative encoders (e.g., optical-flow based or 3D ConvNets, if supported) perform differently. As is, part of the failure could simply be that the synthetic stimuli are not well aligned with the models’ visual front-ends rather than a fundamental limitation in temporal computation.

3. **Finetuning experiment is under-specified and not yet convincing as evidence of architectural impossibility.**  
   - Section 4.4 reports fine-tuning InternVL2.5‑8B and Qwen2‑VL‑7B on 400 SpookyBench videos for 10 epochs using LlamaFactory, yielding 0% test accuracy. However, critical details are missing:  
     - Train/validation/test split sizes and whether videos are randomly partitioned or split by word/object/video identity,  
     - Whether only the language head was finetuned or parts of the vision encoder were updated,  
     - Loss function used (e.g., instruction-following vs captioning vs classification) and how outputs were mapped to supervision labels,  
     - Any training curves (loss vs epoch) or sanity checks that the model overfits the training set.  
   - Given how synthetic and “unnatural” the stimuli are, 400 videos is a very small training corpus. Without evidence that models can at least memorize the training set or that the vision encoder is allowed to adapt, the claim that failure is “fundamental” and “architectural” is premature.

4. **Human study is tiny and somewhat inconsistently reported.**  
   - The human evaluation uses only **six participants** (Section 4.2), each viewing the entire 451-video set. This is enough to show that the stimuli are not impossible for humans, but is weak as a quantitative characterization of “human-level” performance.  
   - Moreover, **Table 3** reports ≈98–99% accuracy for text and images and ≈94% for dynamic scenes, whereas the abstract and main text repeatedly highlight “over 98% accuracy” as if it were uniform across all categories. This over-emphasis on the headline number glosses over the lower performance for dynamic scenes (and the accompanying interpretation).  
   - There is no analysis of inter-annotator agreement, response variability across SNR levels, or learning effects over time (participants see 451 similar stimuli). This is not fatal, but weakens some of the stronger claims about “humans effortlessly” achieving near-perfect performance.

5. **Causal attribution to “time blindness” is somewhat overstated given alternative explanations.**  
   - The narrative claims that models are “fundamentally time-blind” and lack mechanisms for “motion-based figure-ground segregation”. While the performance gap is real, the paper does not make much effort to disentangle:  
     - lack of explicit temporal inductive bias vs.  
     - training distribution mismatch (models never seeing anything like binary noise motion illusions), vs.  
     - limited low-level motion sensitivity in ViT-based encoders not trained for optical flow.  
   - For instance, some works show that LLMs + external motion features (optical flow, trajectories) can support temporal reasoning. Here, we do not see an experiment where a simple handcrafted motion-feature baseline (e.g., classical optical flow + template matching or k-means on flow directions) is applied to SpookyBench. That baseline would help clarify if the task is solvable with straightforward temporal signal processing or truly requires new architectures.  
   - Without such analysis, the strong “architectural limitation” and “time blindness” language feels ahead of the evidence.

6. **Temporal SNR math is promising but under-integrated with the empirical results.**  
   - Equations (1)–(4) define SNR metrics over optical flow and motion boundaries. However, the way these metrics are computed on SpookyBench is described only qualitatively, and several details are ambiguous:  
     - Equation (1) uses \(P_S = \mathbb{E}[\|\nabla \mathbf{F}\|^2]\), but it is unclear over what domain the expectation is taken (per pixel, per frame, whole video) and how optical flow \(\mathbf{F}\) is estimated under such noisy conditions.  
     - Equation (3) defines \(C = e^{-\mathrm{Var}_\theta(\mathbf{F})}\cdot \mathbb{1}(\|\mathbf{F}\| > \tau)\); the subsequent SNR\(_T\) uses \(\operatorname{Var}(C)\) and \(\mathbb{E}[\operatorname{Var}_{\text{local}}(C)]\), but no window size, temporal horizon, or implementation details are given.  
     - Equation (4) uses means and variances of flow within/without mask \(M\), yet later the text says “mask \(M\) is estimated from motion boundaries”, effectively making this a circular definition. It is unclear how robust this masking is and whether errors in \(M\) bias SNR\(_M\).  
   - These issues are not catastrophic, but they make it hard to reproduce or to directly connect the SNR values in **Table 2** / **Table 6** to the qualitative human and model results. The paper would benefit from a more worked-through example (perhaps for the ant video in Section E) that steps through SNR computation with explicit parameter choices.

7. **Limited exploration of model variations and ablations.**  
   - All Video‑VLMs in **Table 1** are used “as is” with prompt engineering. Beyond frame rate and short finetuning, there are no ablations on, for example:  
     - feeding **shorter clips** or sliding windows vs the full hundreds of frames,  
     - converting the input to a simpler modality such as explicit motion vectors or frame differences and feeding these as images,  
     - seeing whether models trained or prompted specifically for OCR (for the text category) perform differently.  
   - Even a simple ablation where the temporal sequence is turned into a static average frame (which should be indistinguishable from noise) and fed to models would help isolate whether they ever rely on subtle temporal cues vs purely static ones.

8. **Some exposition issues, repetition, and minor inconsistencies.**  
   - There are scattered typos and formatting issues, e.g., “five key 2. SNR metrics” in Section 3.3.1; repeated author-name placeholders in the references; duplicated Carlson & Copeland citation.  
   - The description of the SNR-threshold phenomenon in Section 3.3.2 refers to **Figure 4**. While the plot is useful (showing an abrupt accuracy jump around −2.5 dB SNR), the text says “threshold (~2.5 dB)” but Table 2 has **negative** SNRs; the relation between absolute SNR values used for the detection curve and the category-level statistics is not entirely clear.  
   - The benchmark categories are variously called “Words”, “Text”, “Temporal symbol recognition”, etc., which is mildly confusing.

Collectively, these weaknesses do not negate the interest of the benchmark, but they do undermine the strength of the central “architectural time blindness” conclusion and leave substantial open questions about evaluation, finetuning, and the degree to which the task reflects real-world temporal reasoning.

## Potentially Missing Related Work

I did not identify obvious, directly-related recent benchmarks or temporal illusion datasets that are clearly missing, beyond what is already cited (TemporalBench, TVBench, VITATECS, VidHalluc, VideoVista, etc.). The related work section is reasonably comprehensive for Video‑VLM temporal reasoning and neuroscience of time perception.

If anything, the paper could briefly connect to older psychophysics or computational vision work on motion-based figure-ground segmentation and structure-from-motion illusions, but that is more of a depth improvement than a critical omission.

So I would mark this as:

N/A.

## Questions

These are questions where a detailed response or additional results could change my opinion:

1. **Matching and 0% accuracy.**  
   - How exactly is the exact-match accuracy computed? Please specify:  
     - Preprocessing of model outputs (lowercasing, punctuation removal, stemming, handling of plurals),  
     - The size and construction process of the acceptable label set \(Y_i\) for objects and dynamic scenes,  
     - Whether synonyms and paraphrases were considered (e.g., “soccer ball” vs “football”, “guy playing basketball” vs “man playing basketball”).  
   - Could you provide a small table of sample GPT‑4o / Gemini outputs on a few videos, along with whether they were marked correct or incorrect, so reviewers can see that 0% is not an artifact of string matching?

2. **Finetuning details and learning curves.**  
   - For the InternVL2.5‑8B and Qwen2‑VL‑7B finetuning experiments:  
     - What was the train/validation/test split strategy?  
     - Were vision encoders updated or frozen?  
     - What learning rate, optimizer, and batch size were used?  
     - Did the models overfit the training set (e.g., near 100% train accuracy)?  
   - If possible, please show training/validation curves and a confusion-style analysis on the test set, even if accuracy remains 0%. This would significantly strengthen the claim that SpookyBench is hard for these architectures, not just under-trained.

3. **Effect of visual preprocessing and resolution.**  
   - Many models internally resize and crop inputs. Have you inspected the intermediate frames or embeddings after their built-in preprocessing to confirm that the motion contrast survives?  
   - As a sanity check, could you try feeding lower-resolution versions of SpookyBench (e.g., 224×224) and see whether human accuracy is still high? If humans fail at downsampled resolution while models only see such downsampled frames, then part of the gap is about pre-processing, not architecture.

4. **Classical vision baselines.**  
   - Have you considered running a simple hand-engineered baseline such as:  
     - compute optical flow on SpookyBench videos,  
     - aggregate motion direction histograms or apply template matching against static masks, and  
     - classify words/objects based on these signals?  
   - If such a baseline works reasonably well, it would substantiate that the temporal information is algorithmically accessible and the failure is specifically about current VLM design. If it fails, that also informs the discussion.

5. **Robustness of the reported SNR threshold effect.**  
   - Figure 4 shows a very sharp transition around ~−2.5 dB SNR for text detection. How many data points / videos went into this curve? Is this threshold consistent across different words and speckle sizes, or averaged?  
   - Is the same kind of abrupt threshold observed for images and dynamic scenes, or is it specific to text?

Clarifications or additional experiments along these lines would significantly increase my confidence in both the benchmark’s interpretation and the central claims.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

**2: fair.**  
The core benchmark construction and measurements are sound, and the empirical gap is real. However, the evaluation pipeline (exact matching), under-specified finetuning, and somewhat loose causal attribution to architectural “time blindness” reduce the technical robustness.

## Presentation Rating

**3: good.**  
The main ideas, figures, and tables are clear, and the paper is generally readable. Some sections (especially the SNR math and finetuning details) are under-explained, and there are minor typos and repetitions, but overall exposition is above average.

## Contribution Rating

**2: fair.**  
The idea of a “purely temporal” video benchmark is interesting and likely useful for the community, but the empirical and analytical support for the strongest claims is not yet at the level I would expect for a solid accept at ICLR. With stronger methodology and deeper analysis, the contribution could be elevated.

## Overall Rating

**4: Marginally below the acceptance threshold. But would not mind if paper is accepted.**  
The benchmark is conceptually compelling and the experimental results highlight a worrying blind spot in current Video‑VLMs. However, several important methodological gaps (string matching and 0% accuracy, limited finetuning analysis, missing baselines, and only modestly developed SNR/theoretical section) weaken the scientific conclusions. I lean toward rejection in the current form, but with revisions addressing these issues, this could become a valuable benchmark paper.

## Reviewer Confidence

**4: confident.**  
I am reasonably familiar with multimodal / Video‑VLM literature and temporal reasoning benchmarks, and I carefully checked the math and experimental setup as far as described. Some uncertainty remains due to missing experimental details, but it is unlikely that I have fundamentally misunderstood the work.