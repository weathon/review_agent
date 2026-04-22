# Discovering and Steering Interpretable Concepts in Large Generative Music Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
The fidelity with which neural networks can now generate content such as music presents a scientific opportunity: these systems appear to have learned implicit theories of such content's structure through statistical learning alone. This offers a potentially new lens on theories of human-generated media. When internal representations align with traditional constructs (e.g. chord progressions in music), they show how such categories can emerge from statistical regularities; when they diverge, they expose limits of existing frameworks and patterns we may have overlooked but that nonetheless carry explanatory power. In this paper, focusing on autoregressive music generators, we introduce a method for discovering interpretable concepts using sparse autoencoders (SAEs), extracting interpretable features from the residual stream of a transformer model. We make this approach scalable and evaluable using automated labeling and validation pipelines. Our results reveal both familiar musical concepts and coherent but uncodified patterns lacking clear counterparts in theory or language. As an extension, we show such concepts can be used to steer model generations. Beyond improving model transparency, our work provides an empirical tool for uncovering organizing principles that have eluded traditional methods of analysis and synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a comprehensive framework for uncovering and manipulating interpretable latent features in pretrained music generation models.
The authors apply sparse autoencoders (SAEs) to the residual streams of MusicGen, discovering sparse “concept vectors” that correspond to meaningful musical dimensions (e.g., timbre, rhythm, genre).
The framework further proposes automatic labeling of each discovered feature via LLM-based captioning and MIR taggers, and demonstrates preliminary steering by injecting these features back into the model.

Overall, the work is technically sound, conceptually elegant, and well-motivated, providing a significant step toward mechanistic interpretability in multimodal (music–audio–text) systems.

### Strengths
Novel application of mechanistic interpretability to music.
The use of SAEs to analyze music model activations is both original and conceptually clean. It connects recent interpretability work (e.g., Anthropic’s neuron-level SAE studies) to a new and highly complex modality.

Bridging analysis and control.
The proposal to “steer” MusicGen by manipulating discovered features introduces an exciting bridge between interpretability and controllability—potentially enabling a new paradigm of interpretable generation.

Strong conceptual framing.
The paper explicitly situates itself within the context of “unsupervised concept discovery,” distinguishing itself from traditional probing methods that rely on predefined human labels. This distinction is theoretically well-grounded and clearly communicated.

### Weaknesses
Steering demonstration missing.
The paper’s most exciting claim—that discovered features can steer MusicGen—remains theoretical.
A simple demo (e.g., modulating one feature and showing changes in audio or spectrogram) would transform this from a conceptual claim to a verifiable capability.

Emergent ability needs stronger evidence.
To argue that new control dimensions have emerged, the authors should show that these SAE-derived features are not trivially mappable to existing models such as CLAP or other audio–text embeddings.
In other words, the learned features should correspond to new, controllable factors that cannot already be directly used to condition MusicGen or described by current models.
This would validate that the SAE uncovers genuinely new interpretable concepts rather than re-labeling known ones. 

(my understanding is that because the automatic labeling step relies on existing music/audio to text models, the performance is somehow bounded by their capability. E.g., if CLAP cannot do a good job on recognizing the emotion of music, the proposed model won't have good emotion interpretability. That said, novel combinations of labels are very possible, and such combinations cannot be directly used as the text prompt of musicgen. )

happy to adjust the rating if these issues are resolved.

### Questions
1 Could you include an explicit steering demo—for instance, modifying one SAE latent and showing corresponding audio differences or spectrogram shifts?
2 could you show "strong emergence" in the sense that the learned concept cannot be directly used as text prompt to steer MusicGen (show that doing this will lead to poor results); also CLAP cannot output such a concept.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
I am a little confused by the premise of this paper, and would appreciate it being explained in the rebuttal (in which case I will be most willing to revise my score).  It seems that the paper is arguing that there are concepts humans don’t have words for and that “Electronic Beeps and Boops” is such an example.  Yet the authors are validating using both a multimodal audio -> text model (gemini) and CLAP, which seems to presuppose (and, as I have seen it used, correctly) that CLAP itself can map “Electronic Beeps and Boops” and audio to a valid score.  If one defines theory as “the things that show up in 20th century textbooks”, “beeps and boops” wouldn’t be there (for more sociological/historical than cognitive reasons), but I wouldn’t label them as “concepts we don’t have names for”, particularly given the success with CLAP.  

But to summarize the paper's technical contributions - they have used sparse autoencoders (SAE's) to infer common patterns of activation.  They use this in the context of interpretability by using Gemini to provide labels for these common patterns, do some basic analysis of these data patterns, and try to steer the MusicGen model using these patterns (although it's unclear if that was successful or not).

### Strengths
This paper engages with the interpretability literature to provide a lens into the relationship between a highly complicated model and the sort of language we use as humans to describe music.  This provides much needed research at the intersection of model interpretability and music generation.  They successfully determine that they can use SAE's to uncover important information about how music data is structured, including not only the fact that there are such labels, but that they are in some instances highly correlated with each other.

### Weaknesses
1. I am unconvinced by the contributions of this paper.  Fundamentally, the claim that they are producing concepts we "don't have a name for" seems blatantly false given that they use a multimodal model to label them (unless I'm misunderstanding something).  

2. The numbers for the steering example don't seem impressive without more information - having "any improvement at all" on 20-25% of features compared to None could be noise.

3. For a paper about music, it is very limiting to take the "mean activation" of a feature over 10 seconds.  For instance, unless I'm misunderstanding, the example they gave about chord progressions being something we do have interpretable concepts for, \textit{unlike} "beeps and boops", actually reveals a significant flaw in their approach - I don't think it would be mathematically possible to infer a chord progression feature from mean activations over 10 seconds.

### Questions
1. Do you have any way of measuring “degree to which this is a previously untheorized label”?

2. On line 288 – could you elaborate on what the “baseline” is?  Uncontrolled musicgen?

3. On line 310 – could you possibly provide examples of co-occurring activating features, and mention if there were any there that were unexpected?  I’d imagine that determining cooccurrences might be more enlightening than the concept names themselves.

4. Can you clarify the diagram on line 385?

5. Can you speak as to how your features are different than the features CLAP can recognize, or why your contribution is an improvement over just having access to CLAP?

6.   Could you describe the essentia tags used?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper extends sparse autoencoder (SAE) methodology from language models to generative music models, specifically MusicGen-Small and Large. The authors train k-sparse autoencoders on residual-stream activations, automatically label discovered features using multimodal LLMs and audio analysis tools (filtered by CLAP alignment), and demonstrate concept steering by intervening on SAE decoder directions. The work reports thousands of interpretable features per configuration, shows that deeper layers in larger models yield more interpretable concepts, and achieves measurable steering success on 15–35% of tested features.

### Strengths
To the best of my knowledge, this is the first attempt to apply SAE-based interpretability to audio generation LM; this is a meaningful extension beyond NLP/vision that addresses an underexplored modality. The modular pipeline is well-documented and reproducible. The authors conducted systematic exploration across model scales, layers, sparsity levels, and expansion factors provides useful data on where interpretable structure emerges in music LMs. Table 1 and Fig. 3 offer valuable design guidance. The paper is generally well-written and I find the figures to be informative.

### Weaknesses
(1) I notice CLAP serves as both the filter for accepting labels and the metric for evaluating interpretability and steering success. This creates a validation loop where the authors are essentially measuring "does this feature change CLAP scores" rather than "does this feature control meaningful musical concepts"; The human study is too limited to break this circular argument issue.

(2) 15–35% success rate with a single prompt and CLAP-only evaluation is insufficient to claim "robust" controllability. I suggest to add the following: (a) listening studies, (b) diverse prompts, (c) baseline comparisons (e.g., random directions, PCA, supervised probes), (d) negative controls (inactive features, orthogonalized vectors). Per my first point, current results may reflect CLAP bias rather than perceptual control.

(3) Key design choices (θ_min/θ_max thresholds, k values, rarity cutoffs) lack sensitivity analysis. How much do interpretability rates depend on these arbitrary choices? Some statistical report is needed.

(4) Restricting to MusicGen limits this paper's impact. Currently, diffusion/flow-matching based models dominate audio generation; Appendix C acknowledges difficulties but doesn't attempt even a small-scale demonstration (as an example, Stable Audio Open small is small-scale; only 341M parameters). Claims of generality are thus in my opinion unsupported.

(5) Mean-pooling activations across time may obscure temporal dynamics, and there's no analysis of within-track feature stability or polysemantic behavior at different time scales.

### Questions
(1) Can you provide human listening tests comparing steered vs. unsteered outputs on perceptual dimensions (genre, timbre, rhythm)? Please compare this with random-direction baselines.

(2) How do interpretability quality metrics (human agreement, not just counts) vary with θ_min, θ_max, k, and EF? 

(3) Add negative controls (steering on low-activation features, cross-layer swaps, random directions with matched norm) and report effect sizes with statistical tests.

(4) Can you demonstrate the pipeline on at least one small-scale diffusion-based model to support cross-architecture claims?

(5) How does SAE steering compare to supervised linear probes, activation patching, or MusicGen's own conditioning mechanisms?

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
This paper uses SAE to discover interpretable concepts in a pretrained music generation model MusicGen. The process is unsupervised, using pretrained audio taggers/LLM for automatic concept labeling.

### Strengths
1. This is the first paper to perform concept discovery with SAE in pretrained music generators, and the results are much better compared to the previous probing works.

2. The design choices in Sec. 3.3 provide very useful insights for future researchers in concept discovery for pretrained audio/music models.

3. The automatic labeling pipeline is promising (with some limitation as stated in weakness) and could be applied to other models (audio generation/understanding models).

In general, this work has the potential to bring significant impact to explanibility in large audio/music models.

### Weaknesses
1. In sec. 3.5 automated interpretability, the automatic pipeline might harm the interpretation of some concepts (i.e., chord & keys) since neither gemini nor essentia has such ability.

2. Currently the number of examples are very limited. More examples/case study would be useful in the appendix, including possible failure cases where the automatic labeling pipeline fail to conclude. I.e., more examples where:

(1) A concept could be successfully extracted and named;
(2) A concept could be extracted, but could not be correctly named;
(3) A concept that could not be extracted.

Different types of concepts could also be tested, including genre, instruments, playing techniques, rhythm pattern, groove, sound field, audio quality, mixing techniques, key modes, tempo, velocity, moods etc. Even testing on a portion of them would be very useful.

### Questions
1. Line 210: what is a "track" in this context? Does it refer to a single piece of generated music, or a separated stem of a music?
2. Line 397: If later layers encode more interpretable features, why is the numbers in early layers (e.g., L2) larger compared to later layers in Table 1?

### Soundness
3

### Presentation
4

### Contribution
4
