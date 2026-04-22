# LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
Music learners can greatly
benefit from tools that accurately detect errors in their practice.
Existing approaches typically compare audio recordings to music scores using heuristics or learnable models.
This paper introduces \textit{LadderSym}, a novel Transformer-based method for music error detection.
\textit{LadderSym} is guided by two key observations about the state-of-the-art approaches: (1) late fusion limits inter-stream alignment and cross-modality comparison capability;
and (2) reliance on score audio introduces ambiguity in the frequency spectrum, degrading performance in music with concurrent notes.
To address these limitations, \textit{LadderSym} introduces (1) a two-stream encoder with inter-stream alignment modules to improve audio comparison capabilities and error detection F1 scores, and (2) a multimodal strategy that leverages both audio and symbolic scores by incorporating symbolic representations as decoder prompts, reducing ambiguity and improving F1 scores.
We evaluate our method on the \textit{MAESTRO-E} and \textit{CocoChorales-E} datasets by measuring the F1 score for each note category.
Compared to the previous state of the art, \textit{LadderSym} more than doubles F1 for missed notes on \textit{MAESTRO-E} (26.8\%~$\rightarrow$~56.3\%) and improves extra note detection by 14.4 points (72.0\%~$\rightarrow$~86.4\%).
Similar gains are observed on \textit{CocoChorales-E}. Furthermore, we also evaluate our models on real data we curated. This work introduces insights about comparison models that could inform sequence evaluation tasks for reinforcement learning, human skill assessment, and model evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
LadderSym introduces an interleaved two-stream transformer for music practice error detection.
By inserting cross-attention alignment modules and symbolic score prompts, it achieves substantial improvements over Polytune and classic DTW-based baselines. The model detects missed, extra, and wrong notes with high precision.

### Strengths
1. Clear empirical improvement with solid ablation and baseline comparison. Music error detection is, in general, a challenging problem without elegant solutions so far. Very glad to see such a contribution.

2. A fusion design that enhances alignment while preserving modality specialization.

3. A connection with a larger picture: the model presented is, in spirit, well resonant with "function alignment"(https://arxiv.org/abs/2503.21106), a new theory of mind that is also closely related to music perception. I recommend citing and relating to it.

### Weaknesses
Possible improvements:

The paper could emphasize what is newly learned: attention visualization would clarify whether the model develops interpretable alignment behaviors.

The method could potentially apply to other music performance problems  (say, expressive performance).

### Questions
Is the demo in real time? It looks like so from the demo page, but the method in the paper does not seem to support a real-time function yet.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces *LadderSym* for music error detection, featuring (1) a two-stream encoder with inter-stream alignment modules, and (2) the integration of both audio and symbolic scores as reference inputs. The model achieves state-of-the-art performance on both the *MAESTRO-E* and *CocoChorales-E* datasets.

### Strengths
1. The improvement over the previous model (*Polytune*) is substantial.
2. The proposed *Ladder* structure could potentially influence other tasks that require comparison between two sequences.
3. The analysis of fusion stages provides useful insights for designing other multi-input models.
4. Experiments conducted on a curated real-world dataset add value; if released publicly, this dataset could have a meaningful impact on piano error detection research.

### Weaknesses
1. The effect of incorporating symbolic scores is not sufficiently strong, as the relatively small gap shown in Table 4 weakens one of the paper’s two main contributions.
2. The contribution of the *Ladder* design itself is questionable. Based on Tables 3 and 4, the position of the fusion stage appears to have a greater influence than the Ladder architecture, which fuses at every layer. On the *CocoChorales-E* dataset, simply moving from late fusion (*Polytune*) to the last three fusion layers improves the Extra-note F1 score from 46.8% to 59.6%. Introducing Ladder further raises it only slightly, to 62.3%. This suggests that the fusion location, rather than the Ladder mechanism, drives the improvement, making the claim that “cross-modal alignment should happen frequently” less convincing.
3. The writing could be improved. Figures and tables are scattered throughout the text, and Section 3 (*Method*) is intertwined with experimental results. It is recommended to separate methodological and experimental sections more clearly for better readability.

### Questions
Could the authors please address the following:

1. What does *globality* refer to, and why is coarse clip-level energy information used to measure it?
2. Why is asymmetric feature extraction important for error detection? Intuitively, both practice and score streams should capture local information to localize and classify erroneous notes.
3. Given the additional fusion steps, why does *LadderSym* have fewer parameters and faster inference speed compared to *Polytune*?
4. In Figure 6, for the score stream, why are the attention maps concentrated at the beginning, especially in the last few layers (8–11), when the corresponding horizontal frames appear almost silent?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces LadderSym, a transformer-based model that improves error detection in music practice through two innovations:
1)	The paper proposes a two-stream “Ladder” encoder that processes the practice audio and reference audio separately but aligns them at each layer using cross-attention (“inter-stream alignment modules”), which improves synchronization and comparison without forcing both streams to share parameters.
2)	Symbolic prompting, a prompting method that adds a symbolic (MIDI) score as a decoder prompt alongside the audio input, which reduces ambiguity caused by overlapping frequencies in the score’s audio.
These components together allow more accurate note-level classification of mistakes.

### Strengths
The system utilizes inter-stream alignment in every encoder layer, enabling the model to learn temporal dependencies similar to classical DTW but more robustly. The symbolic prompt adds contextual knowledge that helps with subtle or overlapping notes. Probing experiments show that each stream learns to specialize.

### Weaknesses
The model itself still struggles towards dense musical passages which might mask missed notes. Some errors near sequence boundaries may be mislabeled. The model is proposed under the assumption of roughly stable tempo, and extreme tempo deviations still require pre-alignment.

### Questions
1.	The authors mention “20 beginner performances” but do not specify: a) The number of unique players (only “three graduate students”) vs. performances (20 total — overlap unclear). b) The recording conditions (microphone type, room noise, instrument consistency). 
2.	How long are the symbolic prompts relative to the output sequence (do they truncate or pad)? Does the model ever ignore the prompt when symbolic and audio information disagree?
3.	Was any ablation done on partial or noisy symbolic prompts (realistic for imperfect MIDI scores)?

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
3

### Summary
The paper introduces LadderSym, a multimodal Transformer architecture for music practice error detection, aiming to identify missed, extra, and wrong notes by comparing student practice audio to a reference score. LadderSym addresses these with two key innovations:

1. A two-stream “Ladder” encoder that incorporates inter-stream alignment modules at every layer, enhancing fine-grained alignment between score and performance audio.

2. A symbolic-score prompting strategy, where symbolic (MIDI) tokens are fed to the decoder, improving interpretability and reducing ambiguity.

### Strengths
• Proposes a novel interleaved encoder with layer-wise cross-attention modules, representing a meaningful architectural advancement over late-fusion approaches like Polytune.

• Integrates symbolic prompts—a creative use of decoder conditioning that blends multimodal reasoning with interpretability.

• The model is well-motivated and rigorously validated: strong baselines (Polytune, DTW-based), comprehensive ablations (fusion depth, input modality), and statistically meaningful improvements across datasets.

• The paper is well-organized and accessible, explaining design intuitions (e.g., balancing local vs global feature specialization) with illustrative figures and tables.

### Weaknesses
• It would strengthen claims of generality to include instruments with continuous pitch (e.g., strings, voice) or evaluate robustness to stylistic variation.

• The system’s inference latency (8.1 tokens/s) may hinder real-time feedback applications in tutoring contexts. A short section on runtime optimization or lightweight variants would make the work more actionable.

### Questions
see weakness

### Soundness
4

### Presentation
3

### Contribution
4
