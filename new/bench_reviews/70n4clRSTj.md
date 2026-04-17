Let me check the relevant section about the "40%" claim in the paper content I was given. Looking at Section 3.3.2 of the paper:

> "The words exhibited negligible detection (~0%) below 2.5dB SNR, but jumped to 85.7% accuracy above this threshold... Prompts performed best (40% accuracy), with Chain-of-Thought reasoning improving general identification tasks compared to direct prompting."

This is confusing - it seems to mix human and model results, and the "40%" is unclear. This confirms the harsh critic's observation about a contradiction with Table 1. Let me also verify the evaluation protocol claims.

Looking at the paper again for frame sampling details: In Section 4.1: "We input sequences of multiple video frames simultaneously for models that do not directly support video input." No further specification of how many frames, ordering, etc.

For closed-source models: No documentation of exact call patterns.

For the exact match metric: The paper says "All prompts instruct models to respond with only 1-5 words identifying the content" and "none of the models tested produced responses that matched any of the acceptable options."

Now let me synthesize my final review.</think>Now I have all the data I need to write the final review. Let me carefully synthesize based on verification of claims against the paper.

## Summary

SpookyBench is a novel benchmark that encodes information (text, images, dynamic scenes) exclusively in temporal sequences of noise-like frames, eliminating all spatial cues. While humans achieve ~98% accuracy, 27 state-of-the-art Video-VLMs—including GPT-4o, Gemini, and Qwen—all achieve 0% accuracy, even with chain-of-thought prompting and fine-tuning. The benchmark is designed to expose a "time blindness" in current architectures that rely on spatial-first processing paradigms.

## Strengths

- **Creative and well-specified benchmark design**: SpookyBench's temporal encoding framework (opposing-motion noise masks in Algorithm 1, threshold-based depth map animation in Algorithm 2) is novel, rigorously described, and reproducible. The approach of completely eliminating spatial cues to isolate temporal processing is a clean experimental manipulation that genuinely isolates a neglected capability gap.

- **Comprehensive model evaluation across diverse architectures**: Testing 27 models spanning open/closed-source, parameter scales from 2B to 78B+, and specialized temporal architectures (TimeChat, InternVideo2.5) provides strong evidence this is a systematic architectural limitation rather than a model-specific issue. The consistency of the 0% result is striking.

- **Systematic confound analysis**: The frame-rate experiment (Section 4.3, Tables 4–5) showing humans degrade gracefully at low FPS while models stay at 0%, and the finetuning experiment (Section 4.4) showing targeted training fails, both strengthen the interpretation that the limitation is not simply an interface or distribution mismatch.

- **Rigorous temporal metrics**: The four SNR metrics (Basic, Perceptual, Temporal Coherence, Motion Contrast in Section 3.3.1) provide quantitative characterization of stimulus properties beyond anecdotal observation, demonstrating careful attention to perceptual science.

## Weaknesses

### Major:

- **Conflation of motion-based perceptual grouping with temporal reasoning**: SpookyBench tests a specific low-level perceptual capability—detecting coherent motion patterns in noise to achieve figure-ground segregation. This is distinct from "temporal reasoning" as commonly understood in the video understanding literature (reasoning about event ordering, causality, duration, etc.). The paper repeatedly generalizes from this narrow stimulus class to sweeping claims about "time blindness" and "fundamental inability to process temporal information," while also invoking safety-critical applications like autonomous driving. However, recognizing shapes in noise-flicker is fundamentally different from recognizing temporal patterns in natural video. The paper does not adequately acknowledge this distinction, which inflates the scope of what the benchmark actually demonstrates. (The paper's own neuroscience section underscores that human perception uses dedicated low-level motion pathways for these illusions—current VLMs were not designed to emulate these.)

- **Evaluation protocol under-specification undermines causal attribution**: The paper claims the universal 0% demonstrates "fundamental architectural inability," but key experimental details are missing. For models that "do not directly support video input," only "We input sequences of multiple video frames simultaneously" is stated (Section 4.1)—without specifying how many frames are sampled, whether they are ordered, or whether temporal markers are provided. For closed-source APIs (GPT-4o, Gemini), no documentation of the exact call pattern is given (video upload vs. frame-by-frame vs. GIF). Different input pipelines could bypass the video processing entirely. The claim that this is a "fundamental architectural" limitation vs. an interface-level limitation is not adequately controlled.

- **Finetuning experiment is insufficient to support "fundamental architectural inability"**: The paper's strongest claim—that failure is not just distributional but architectural—rests on Section 4.4, where two models finetuned on 400 videos for 10 epochs still achieve 0%. However, this is very weak evidence for architectural impossibility: (1) only 2 of 27 models were tested, (2) no training details are provided (loss function, how frames are provided during training, learning rate), (3) 400 videos × 10 epochs is a small supervised signal for models of this scale on such an unusual perceptual pattern, and (4) no training curves or analysis of what models learned are provided. The result shows standard instruction-style finetuning is insufficient, but does not establish architectural impossibility.

- **Small and weakly characterized human evaluation**: The 98% human accuracy claim is central to the paper's framing, but rests on only 6 participants with no reported demographics (age, visual acuity, technical background), no description of recruitment procedure, and no inter-rater reliability beyond mean and SD. For a perceptual task that may vary across age and visual processing differences, this limits the reliability of the human baseline. The claim that humans achieve this "without training" and "effortlessly" is partially rhetorical—the study design doesn't systematically test the role of exposure or practice.

### Minor:

- **0% accuracy across all models limits diagnostic value**: While striking, uniform 0% accuracy provides no gradient for comparison between architectures. The paper provides no qualitative analysis of model outputs—to what extent do models report "noise only" vs. hallucinate semantic content? This distinction matters for understanding whether models lack the signal entirely vs. receive it but cannot decode it to lexical outputs.

- **Confusing text in Section 3.3.2**: The "binary SNR threshold" analysis states "Prompts performed best (40% accuracy)" which appears to contradict Table 1's universal 0%. It is unclear whether this 40% refers to humans, a different condition, or model prompting. This needs clarification.

- **Dataset size and imbalance**: 451 videos total with only 57 in Dynamic Scenes (12.6%) is relatively small and imbalanced. While the generator allows unlimited extension, the evaluated benchmark is limited.

### Trivial:

- The SNR threshold analysis (Figure 4) is based on limited data points near the critical threshold; confidence intervals for the step-function claim would strengthen it.

## Nice-to-Haves

- Testing whether optical flow (e.g., RAFT) computed on SpookyBench videos reveals hidden content, and providing flow as auxiliary input to VLMs—this would clarify whether the bottleneck is low-level motion extraction vs. higher-level temporal reasoning.

- Evaluating models with early 3D spatiotemporal convolutions (e.g., S3D, I3D-based) that operate at the pixel level rather than via ViT tokenization, to test whether the limitation is specifically about spatial-first tokenization.

- Gradual spatial signal reinstatement experiments (adding 1%, 5%, 10% spatial contrast) to characterize the transition from 0% to functional performance.

- Probe intermediate model representations (e.g., attention maps, ViT features) to determine where temporal information is destroyed in the pipeline.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic Issue #2 (exact-match metric too strict)**: The paper does describe flexible label sets for images/videos (Section 4.1: "a set of acceptable labels Yi = {yi1, yi2, . . . , yin} to account for semantic ambiguity") and explicitly states models failed to match ANY acceptable option. While more granular analysis (e.g., partial credit, response distributions) would be valuable, the claim that exact match is "brittle" is weakened by the existence of flexible label sets. Kept as a minor concern about response analysis but removed as a major criticism.

- **Harsh Critic Issue #1 claim that "interface mismatch" explains all failure**: While the evaluation protocol is under-specified (a legitimate major concern), the finetuning experiment (where models are trained directly on the task) and frame-rate experiment partially address this. The claim that interface differences trivially explain results is too strong—the models truly do seem unable to process these patterns. Kept the protocol under-specification concern but removed the framing that this is the sole or primary explanation.

- **Harsh Critic claim about "asymmetric" human vs. model evaluation**: The paper uses exact match for both humans and models. For humans, Section 4.2 shows very high inter-annotator agreement (Table 3). The claim that human evaluation is "much looser" is not well supported—the typed responses appear genuinely high-accuracy. Removed.

- **Spark suggestion about "Gradual spatial signal reinstatement"**: This is a nice experiment but asking for it in the context of a benchmark paper is scope creep beyond reasonable expectations. Moved to Nice-to-Have.

- **Several Spark/human-finder suggestions about testing optical flow models, 3D convolutions**: These would strengthen the paper but are not standard requirements for a benchmark paper. Moved to Nice-to-Have.

## Novel Insights

The core insight—that current Video-VLMs process video through a spatial-first pipeline and are genuinely incapable of extracting meaning from purely temporal cues—is valid and important. However, the paper would be significantly stronger if it more precisely identified what is being tested: not "temporal reasoning" broadly, but a specific low-level motion-based perceptual grouping mechanism that is a prerequisite for certain types of temporal understanding. The key distinction is that SpookyBench reveals VLMs lack a perceptual *front-end* for motion-based figure-ground segregation, which is different from (though related to) lacking higher-level temporal reasoning capabilities. This distinction has practical implications: the fix likely requires architectural changes at the visual encoding level (e.g., pixel-level temporal differencing before spatial tokenization), not necessarily changes to how models reason about temporal relationships at the semantic level.

## Suggestions

- **Narrow the scope of claims**: Explicitly acknowledge that SpookyBench tests motion-based perceptual grouping rather than temporal reasoning broadly, and clarify the relationship between this low-level capability and higher-level temporal understanding tasks. The paper's impact would increase by being precise about what capability is missing rather than overclaiming.

- **Document evaluation protocol fully**: Specify frame sampling strategy, temporal ordering, and input format for every model tested. For closed-source models, document the exact API call pattern and video upload format. This is critical for the paper's central claim to be evaluable.

- **Provide qualitative model output analysis**: Report what models actually predict—do they say "noise," "I cannot identify," or hallucinate objects? This diagnostic information would significantly strengthen the paper.

- **Strengthen the finetuning experiment**: At minimum, report training details (loss, learning rate, frame format during training, training curves) and test whether models learn any signal (e.g., binary discrimination) rather than only testing open-ended identification.

## Score and Decision

**Calibration comparison:**

- **TOMATO** (temporal reasoning benchmark, similar topic): Scores 6,8,8,5, accepted as poster. Strong principles-based design, 1,484 QA pairs, finetuning analysis requested but not done. More rigorous construction than SpookyBench but similar overclaim issues noted.
- **TVBench** (temporal evaluation redesign): Scores 6,5,3,6, rejected. Similar "models can't do temporal reasoning" finding but with partial model scores providing diagnostic value. Rejected partly due to under-specified comparisons.
- **Vinoground** (temporal benchmark with counterfactuals): Scores 6,6,5,6, rejected. Similar scope to SpookyBench—demonstrates VLM weakness on temporal tasks but with limited novelty beyond demonstrating the weakness.
- **TemporalBench** (fine-grained temporal benchmark): Scores 3,5,5,3,5, withdrawn/rejected. Noted as lacking innovation versus existing temporal benchmarks—similar concern applies here.
- **VideoNIAH** (synthetic video benchmark): Scores 5,6,6,6, accepted as poster. Synthetic but scalable framework with clearer diagnostic value. Noted ecological validity concerns.

SpookyBench's benchmark design is more creative than TVBench or TemporalBench (the noise-encoding mechanism is genuinely novel), and the 0% vs 98% contrast is more dramatic than Vinoground's partial failure findings. However, the overclaiming about "fundamental time blindness" and "architectural inability" based on under-specified experiments is a significant weakness. Compared to TOMATO (accepted), SpookyBench lacks the principled evaluation framework and depth of analysis. Compared to VideoNIAH (accepted), it has similar synthetic/ecological concerns but less diagnostic depth (all-zero provides no gradient). The paper is between Vinoground/TVBench (rejected at ~5.5 average) and VideoNIAH/NovelQA (accepted at ~6.3 average). Given the creative benchmark design but significant overclaiming and under-specified experiments, I place this below TOMATO and VideoNIAH but above the rejected temporal benchmarks.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>