# Chord-Transformer:Chord-Progression Guided Transformer for Long-Sequence Symbolic Music Generation

- Decision: Reject
- Scores: 4, 0, 4, 6

## Abstract
Transformer-based symbolic music generation models are increasingly becoming a vital approach for music composition and editing. Current music generation models face a main challenge in lacking effective structural control mechanisms, making it difficult to maintain harmonic coherence and structural integrity in generated music. This paper presents the Chord-Transformer architecture, which uses chord progression sequences as high-level semantic features to guide the music generation process. Our approach employs an energy-based dynamic programming algorithm to extract chord progressions from the input data. These progressions are used as structural constraints, integrated with a Transformer architecture, to enable autoregressive chord-to-music generation. To enhance the model's ability to capture musical structure, we design a chord-aligned positional encoding scheme and introduce a fusion module that combines cross-attention for chord progression sequences with self-attention for music sequences. This mechanism strengthens collaborative modeling of local and global chord contexts, effectively improving harmonic consistency and structural integrity of generated music. Experimental results show that, compared to state-of-the-art baselines, our proposed method shows significant improvements in key metrics including scale consistency, polyphonic quality, and user preference scores.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on solving the challenge of maintaining harmonic coherence and structural integrity in long-sequence symbolic music generation, proposing a novel architecture called Chord-Transformer. Its core technical design includes three key parts: first, an energy function-based dynamic programming algorithm to extract core chord progressions from raw chord data, filtering redundancy and preserving structure-determining harmonic features by balancing chord frequency and repetitiveness; second, a chord-aligned positional encoding (CAPE) scheme that uses regional masks (1 for chord-sensitive regions, 0 for transitional regions) to align chord progression and music sequence positions, addressing the length mismatch between short chords and long music; third, a parallel fusion module to balance chord cross-attention and music self-attention.
However, the practical musical quality of the generated content appears insufficient. The sheet music examples only demonstrate marginal improvements in simple chord consistency. Generated samples, particularly those on the demo webpage, show significant issues such as abrupt harmonies and dissonant notes, suggesting a lack of basic musical aesthetics and flow.Furthermore, the paper's claimed advancements lack strong support. The musical quality metrics are not clearly defined or fail to adequately capture subjective "musicality." Consequently, the presented results do not convincingly show a practical improvement over existing baselines, raising concerns about the immediate real-world impact of the current model.

### Strengths
- Originality in Technical Route Design
The paper targets the core limitation of existing models—insufficient structural control via chord progressions—and proposes a targeted technical chain. It integrates chord progressions as high-level constraints through three key components: energy-based dynamic programming for redundant chord filtering, chord-aligned positional encoding (CAPE) for position alignment, and a parallel fusion module for balancing chord and music sequence autonomy. This creative combination of existing technologies and domain-specific innovations shows originality in technical integration.

- Rigor in Mathematical Modeling
Key components are supported by detailed mathematical formulations, enhancing reproducibility. These include the objective function for chord progression extraction (Formula 1), the fusion module’s attention integration formula (Formula 5), and the cross-entropy loss function for training. These explicit formulations reflect basic mathematical rigor, facilitating technical validation.

- Clarity in Structure and Expression
The paper follows a standard academic structure with clear logical connections. Related work is categorized to highlight limitations, complex technical details are visualized via figures and algorithms, and the experimental section clarifies dataset, baseline, and evaluation protocols—ensuring transparency and readability.

- Significance in Domain Exploration
By integrating structured musical knowledge (chord progressions) into deep learning, the work aligns with human music creation logic, providing a new direction for long-sequence music generation’s structural control challenge. The explicit constraint design also holds potential for extension to other structured sequence generation tasks.

### Weaknesses
- Evaluation Results Are Unreliable and Samples Lack Transparency 
The paper claims Chord-Transformer got high subjective scores (e.g., 4.4 for pleasantness on Pop909), but the demo webpage’s samples have obvious flaws—frequent harmonic breaks, dissonant notes, and no logical phrasing. It’s unclear if evaluators got these low-quality samples, and the 15 "professionals"’ musical background isn’t detailed. Also, the paper’s sheet music examples (Figs. 5–8) with basic melodic-harmonic coordination aren’t on the demo page, and only MIDI files (no timbre/dynamics) are provided. This makes the "good musical quality" claim untrustworthy.
- Objective Metrics Are Problematic: Unclear and Oversimplified
Most metrics lack clear definitions and formulas. For example, empty beat rate doesn’t say if it’s counted per beat or per measure; groove consistency has no way to calculate "rhythm regularity"; scale consistency doesn’t explain how to determine the "current scale" or handle modal shifts. Also, the metric system only looks at basic stats, ignoring key musical traits like melodic contour, inter-track rhythm coordination, and emotional expression. The "comprehensive evaluation framework" claim is wrong.
- Rigid Chord Constraints Hurt Transformer’s Generalization
The paper uses dynamic programming to get fixed chord progressions and strict CAPE masking to force music to align with chords. This "hard constraint" stops creative changes (like passing tones that make melodies richer) and makes the Transformer less general—chord extraction may overfit to common progressions (e.g., pop’s I-IV-V), leading to less diverse music. Prioritizing chord consistency over flexibility goes against the Transformer’s strength in modeling diverse sequences.\end{itemize}

### Questions
- For subjective evaluation: Please list all samples used (with MIDI links) and confirm if low-quality demo samples were included; tell the 15 "professionals"’ background (e.g., years of training) and full evaluation rules. If low-quality samples were excluded, explain why.
- For objective metrics: Give formal math definitions, calculation steps, and appendix code snippets for empty beat rate, groove consistency, and scale consistency. Also add metrics for melodic contour, inter-track coordination, and phrase completeness, and prove these metrics match human aesthetic judgments.
- For chord constraints: Can the model generate non-chord tones in chord-sensitive regions? If yes, show examples and explain how the fusion module balances constraints and flexibility; if no, say how to fix it (e.g., probabilistic alignment). Also test generation in classical/jazz/pop to check if chord extraction avoids overfitting.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper presents a long-term symbolic music generation model that reinforces global structure. The authors posit that chord progression is a primary feature of music form. First, a coarse chord progression is extracted using a dynamic programming algorithm, and then a Transformer-based model, with a tailored positional encoding of chord progression, is used to generate music conditioned on the extracted chords.

### Strengths
The authors emphasize that musical structure is crucial for effective generation.

### Weaknesses
1. Chord-conditioned generation is common, and it is unclear why chords alone suffice to capture musical structure. What role do other factors, like melody, motivic similarity, phrasing, and sectional form, play? The paper should also clarify the objective of extracting a coarse chord progression and justify why this representation is preferable to raw chord labels. Concrete examples would help demonstrate that the proposed abstraction is more structurally informative.
2. The writing needs improvement. For example, Section 3.1 is difficult to follow due to missing notation/definitions, and Figure 2 is hard to understand, etc.
3. The musical quality is a major concern. On the demo page, the proposed method does not outperform the baseline to my ear. In Figure 3, all chord labels annotated on the score appear incorrect.

### Questions
Q 1-3: see weaknesses.
Q4: How does eq. (4) solve the problem raised at the second para. of Section 3.3.1?

### Soundness
1

### Presentation
1

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
This paper proposes Chord-Transformer, a chord-progression–guided Transformer model for long-sequence symbolic music generation. The main goal is to address the lack of structural control and harmonic coherence in existing Transformer-based music generation systems. To achieve this, the authors first introduce an energy function–based dynamic programming algorithm to automatically extract core chord progression sequences from raw chord data. These extracted chord progressions serve as high-level semantic constraints that guide the music generation process. Building upon this, the proposed Chord-Transformer architecture integrates: a chord-aligned positional encoding (CAPE) and a parallel fusion module. Experimental results show that Chord-Transformer outperforms state-of-the-art baselines (NotaGen, Multi-Genre Music Transformer, MusicLM) on both objective metrics (e.g., scale consistency, polyphony degree) and subjective listening tests (pleasantness, coherence, richness). The paper further includes an ablation study demonstrating the contributions of CAPE and the fusion module.

### Strengths
1. The motivation is clear and meaningful — introducing chord progressions as high-level semantic features helps maintain musical structure during long-sequence generation.

2. The paper proposes an energy function–based dynamic programming algorithm that can automatically extract core chord progressions, providing a practical and interpretable way to obtain chord progressions.

3. The proposed Chord-Transformer effectively incorporates chord information into an autoregressive Transformer through chord-aligned positional encoding and a fusion module.

4. Both objective and subjective experiments demonstrate improvements in musical quality, structural coherence, and controllability, especially in long-form music generation.

### Weaknesses
1. The approach still depends on externally provided chord progressions and manual alignment. This partially undermines the claim of solving long-sequence structural coherence in an autonomous way—structure is preserved because it is externally imposed rather than learned. The work would be stronger if the model could infer or generate chord progressions jointly.

2. The model architecture is relatively simple and conceptually close to existing systems such as Chord-Conditioned Song Generation (Gao et al., 2024) and MusicGen (Copet et al., 2023). The paper does not clearly demonstrate advantages over these baselines.

3. The energy-based dynamic programming step is a core component, but its correctness and robustness are not evaluated. The paper lacks quantitative or qualitative validation of extracted progressions against human annotations or standard chord recognition benchmarks.

4. The baselines (NotaGen, Multi-Genre Music Transformer, MusicLM) are not chord-conditioned, making the comparisons partially biased in favor of the proposed approach. This experiment only demonstrates one significant conclusion: the introduction of chord progressions helps improve structural consistency, but it does not prove the superiority of the introduction method proposed in this paper.

5. The key recent works like MusicGen, or MeLoDy should be included.

6. The evaluation lacks confidence intervals or significance testing for subjective scores. Additionally, the proposed metrics (e.g., groove consistency, scale consistency) only partially reflect structural coherence and controllability; chord adherence and global form structure could be evaluated more rigorously.

### Questions
1. The paper should provide a quantitative or qualitative comparison between the extracted chord progressions and human-annotated chord labels.

2. The computation method for the chord hit rate is unclear.

3. The paper lists several objective evaluation metrics (e.g., pitch class entropy, groove consistency, scale consistency, polyphony degree, empty beat rate) but does not explain how they are computed. The authors should provide clear definitions, formulas, and sources for these metrics.

4. The model currently depends on externally provided chord progressions. It remains unclear whether the system can generate coherent and structurally consistent music without chord input.

### Soundness
2

### Presentation
3

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
This paper presents the Chord-Transformer, a novel architecture for symbolic music generation that uses chord progressions as high-level semantic features to guide the autoregressive generation process. The core contributions include an energy-based dynamic programming algorithm for extracting core chord progressions from raw sequences, a chord-aligned positional encoding (CAPE) scheme to address the length discrepancy between chord and music sequences, and a parallel fusion module to balance cross-attention (on chords) and self-attention (on music).

### Strengths
The proposed Chord-Aligned Positional Encoding (CAPE) is a solution to a fundamental problem in conditioning on short control sequences for long generation targets and and the regional masking strategy effectively bridges the granularity gap. 

It employs a robust evaluation strategy, combining multiple objective metrics (pitch entropy, groove, scale consistency, etc.) with a double-blind subjective study involving both experts and general listeners. This multi-faceted approach strengthens the validity of the claimed performance improvements.

### Weaknesses
The energy-based chord progression extraction algorithm is presented as a solved component. However, its performance and potential failure modes (e.g., on harmonically complex or ambiguous passages) are not critically discussed. The quality of the entire pipeline is highly dependent on this first step, and its robustness deserves more scrutiny. The demo page needs more pair-wise comparisons.

While strong baselines are selected, a comparison with another chord-conditioned model (e.g., a version of MusicFrameworks or a re-implementation of a relevant approach) is missing. This makes it difficult to isolate the improvement stemming from chord-conditioning itself versus the specific architectural innovations of Chord-Transformer.

Also, the model and evaluation are firmly rooted in the harmonic principles of Western tonality (major/minor keys, triadic chords). The paper does not discuss the generalizability of the approach to music from other traditions (e.g., non-Western music) where the very definition of a "chord" and its structural role may differ significantly.

### Questions
Could the authors provide more details on the quantitative evaluation of the chord progression extraction algorithm? For instance, what is its accuracy or agreement rate against a established chord annotation tool or human expert annotations on a subset of the datasets?

The concept of "Chord Progression Hit Rate" is useful but relatively coarse. Could the authors provide more nuanced analysis on how the chord conditioning works? For example, does the model struggle more with certain types of chord transitions compared to others?

The paper focuses on harmonic coherence. Beyond chord-by-chord adherence, does the model, aided by the structural bias of the chord progressions, also generate higher-level formal structures (e.g., AABA form, verse-chorus patterns) more effectively than the baselines? An analysis of section repetition and contrast could be insightful.

Lastly, I feel like hearing some unaligned onsets between different tracks (e.g. LMD sample) on the demo pages. Does it always appear in the experiments?

### Soundness
2

### Presentation
3

### Contribution
2
