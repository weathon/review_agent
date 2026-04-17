# MAPSS: Manifold-based Assessment of Perceptual Source Separation

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Objective assessment of audio source‑separation systems still mismatches subjective human perception, especially when interference from competing talkers and distortion of the target signal interact. We introduce Perceptual Separation (PS) and Perceptual Match (PM), a complementary pair of measures that, by design, isolate these leakage and distortion factors.
Our intrusive approach generates a set of fundamental distortions, e.g., clipping, notch filter, and pitch shift from each reference waveform signal in the mixture. Distortions, references, and system outputs from all sources are independently encoded by a pre-trained self-supervised model, then aggregated and embedded with a manifold learning technique called diffusion maps, which aligns Euclidean distances on the manifold with dissimilarities of the encoded waveform representations.
On this manifold, PM captures the self‑distortion of a source by measuring distances from its output to its reference and associated distortions, while PS captures leakage by also accounting for distances from the output to non‑attributed references and distortions.
Both measures are differentiable and operate at a resolution as high as 75 frames per second, allowing granular optimization and analysis.
We further derive, for both measures, frame-level deterministic error radius and non-asymptotic, high-probability confidence intervals.
Experiments on English, Spanish, and music mixtures show that, against 18 widely used measures, the PS and PM are almost always placed first or second in linear and rank correlations with subjective human mean-opinion scores.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes two new perceptual evaluation metrics for blind audio source separation, called Perceptual Match (PM) and Perceptual Separation (PS).

PM is meant to measure self-distortion: how much the separated output of a given source has been degraded (i.e. does it still "sound like me," or did the separator damage me?).

PS is meant to measure leakage: how much other sources are still audible in the separated output for the target source (i.e. how much bleed from others remains?).

How it works:

For each ground-truth source, the authors generate a bank of controlled distortions (filtering, gating, etc.) that are supposed to mimic plausible separator artifacts.

All short frames (tens of ms) from the clean source, its distorted variants, and the separator output are embedded into a perceptual manifold. This manifold is built with self-supervised audio embeddings (wav2vec2-like) followed by diffusion maps.

Each source becomes a cluster in that manifold, with a mean and covariance.

PM: does the separated output fall inside its OWN source's cluster? (High PM = low self-distortion.)

PS: is the separated output closer to ITS OWN cluster than to ANY OTHER source's cluster? (High PS = low leakage.)

PM/PS are computed framewise and then aggregated to clip level.

The authors evaluate on SEBASS (>11k human ratings over ~900 clips, 32 systems, English speech, Spanish speech, and music). They report that PM and PS correlate well with MOS (Pearson and Spearman), often outperforming SDR/SIR/SAR, PESQ/STOI-style metrics, DNSMOS, etc.

The paper also gives per-frame uncertainty for PM/PS: (i) a deterministic bias radius due to manifold truncation and (ii) a high-probability CI due to limited distortion samples. They propagate this to the final MOS correlation numbers.

### Strengths
S1. Clear problem framing.
The paper separates two real failure modes in source separation: "you damaged my target" vs "you left the other source in."
This is actually how practitioners think about failure, but most metrics don't expose it.

S3. Human-grounded evaluation.
They compare directly to listening scores (SEBASS), not just SDR.
They include English speech, Spanish speech, and music (with and without drums), and 32 different separators.

S4. Uncertainty / confidence intervals.
Section 5  gives an error band on scores and pushes that through to MOS correlation.


S5. Actionability.
PM and PS are frame-level before pooling. This helps to point where leakage happened" or "this is where I over-gated the vocal."
That's a lot more usable than one global SDR number.

### Weaknesses
W1. Novelty vs SDR/SIR/SAR is under-explained.
The paper says PM/PS are the first metrics to "functionally disentangle" leakage and self-distortion.
But classical BSS Eval already has SIR (interference from other sources) and SAR (artifacts).
The paper mentions SDR/SIR/SAR but does not clearly argue why PS is NOT just SIR in a fancier embedding,

why PM is NOT just SAR in a fancier embedding.
Without that, some readers will see this as repackaged SIR/SAR.

W2. Assumptions in the distortion bank.
The method assumes that the authors' hand-designed distortion bank actually spans "plausible" separator artifacts.
If the bank misses common artifacts (phasiness, musical noise, transient smearing), PM could wrongly accept bad audio as "still me," or wrongly reject good audio that falls outside their curated distortion styles. There is no ablation/sensitivity analysis on which distortions are used, so robustness is not proven.

W3. Monolingual-only evaluation in speech.
For speech, they say they split SEBASS mixtures into English-only and Spanish-only pairs "because realistic conversations are monolingual." That is not generally true (code-switching and bilingual overlap are routine in real deployments).
This matters because PS is partly "do you sound like the other source?"
If the other source is always a different language, that discrimination problem is easier.
So PS may look stronger than it would in truly multilingual overlap.
This should be acknowledged as a limitation.

W4. PS pooling over time is not fully specified.
PM is averaged over frames.
PS is said to use a PESQ-like perceptual weighting so that short, audible bleed is penalized more.
But the exact weighting rule is not written in the main text.
Without that, PS is not actually reproducible from the description.



W5. External validity.
All clips are ~10 seconds, 16 kHz, fairly clean, from curated mixtures.
There is no far-field mic, noisy phone audio, long meetings, reverberation, etc.
That is exactly the setting where applied systems struggle.
So the claim is really "PM/PS correlate with MOS on SEBASS-like curated data," not "PM/PS solve perceptual eval in the wild."
The paper should say that clearly.

### Questions
Q1. You claim PM and PS are the first metrics to disentangle leakage and self-distortion, but classical BSS Eval already uses SIR (interference from other sources) and SAR (artifacts). Can you give a concrete example (qualitative or quantitative) where SIR says “good” but PS clearly identifies audible leakage, or where SAR says “good” but PM clearly identifies audible self-distortion? Without that, it’s hard to judge whether PM/PS are fundamentally new or essentially SIR/SAR in a learned perceptual space.

Q2. For SEBASS speech mixtures, you evaluate only English-only and Spanish-only overlaps and justify this by saying “realistic conversations are monolingual.” In many deployment settings (code-switching, bilingual overlap), that assumption does not hold. How would PS behave on overlapping multilingual speech, where two simultaneously active speakers may differ by identity but not cleanly by language? Is PS partly exploiting language identity to discriminate sources?

Q3. PM is averaged over frames, but PS is pooled using a “PESQ-like perceptual weighting,” which is not explicitly defined in the paper. Can you provide the exact temporal pooling rule you use for PS? Right now PS is not reproducible from the description, and your correlation claims with MOS depend on that pooling.

Q4. PM and PS depend on a hand-crafted distortion bank that defines the “perceptual neighborhood” of each source. How sensitive are your metrics to the choice of those distortions? For example, if we remove one distortion type or add an unrealistic one, do PM/PS scores or MOS correlations change significantly? This matters for whether others can safely reuse your metric or will overfit to your particular bank.

Q5. The perceptual manifold uses SSL embeddings + diffusion maps with α = 1, diffusion time t = 1, and dimensionality chosen by a 99% spectral-mass rule (≈20–40 dims). Were any of these hyperparameters, or the PS pooling weights, tuned using SEBASS MOS labels (even indirectly)? If yes, how did you prevent test-set leakage when reporting final Pearson/Spearman correlations with MOS?

### Soundness
3

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
The paper proposes two new measures of audio source separation quality, called Perceptual Separation (PS) and Perceptual Match (PM), that respectively aim to quantify the amount of
- "leakage": to what extent an estimated source is closer to (distorted versions of) the corresponding clean source than to (distorted versions of) the other clean sources
- "self-distortion": how close is the estimated source from the clean one up to "standard" distortions
The approach is primarily frame-based and relies on the knowledge of reference clean sources, which are data-augmented using various "standard" distortions. 

For a given frame index, the time frame of each clean, augmented, and estimated source is first mapped from the waveform domain to an embedding domain using a pre-trained embedding (wav2vec 2.0 based, or MERT based, using hand-picked intermediate layers). An additional spectral-like embedding (diffusion map -based) is then computed for each source, and used to define two Mahalanobis metrics and two representatives (a centroid, and the spectral embedding of the reference clean time frame) for each source. The distance between the spectrally embedded *estimated* time-frame of a target source and these representatives, measured with these Mahalanobis metrics, then yields quantities that are aggregated to provide the final frame-based measures for each source. 
Eventually, frame-based measures are aggregated over a whole signal either by averaging (for PS) or using a "perceptually weighted scheme inspired by PESQ" not described in the main text (for PM).

After conducting an error analysis, the authors evaluate the proposed measures using the SEBASS database, which provides MUSHRA human ratings over a collection of signals separated using various algorithms, gathered from pre-existing source separation evaluation campaigns. The proposed measures are compared to existing quality measures using two indices : the Pearson product-moment correlation coefficient (PCC), and Spearman's rank-order correlation coefficient (SRCC). In most of the considered settings, the proposed measures achieve the best PCC/SRCC scores (or scores comparable to the best ones).

### Strengths
Performance measures reflecting human ratings are very helpful to help design better algorithms. In the specific case of audio source separation, 
Table 1 illustrates well that existing measures can have limited correlation with human scores from the SEBASS database, a situation that is significantly improved by the proposed measures. 
The approach seems sensible, as it combines a first level of "generic" audio features from a pre-trained embedding, a second level of embedding that somehow learns the impact of "standard" distortions in a fine-grained way (source-specific and time-frame specific). Its soundness is confirmed by the experimental evaluation, which can in fact seem surprisingly good for an approach where no human rating is explicitly used to train the designed measures (although given the number of parameters of the method, their selection with "cherry picking" can be interpreted as supervised training). 
For these reasons I rate the contribution to 3.

### Weaknesses
The paper seems to have been written very hastily, with e.g. L075 "*Two lines about appendices and disclaimers on LLM usage*"

As a result, the abstract is hard to grasp due to the choice of wording. For example,  the notion of "intrusive approach" is certainly not familiar to the average ICLR reader, and it would be better to write more explicitly that the original sources used to generate the mixture need to be available for the evaluation. The notions of "system output" and "non-attributed references" were also not immediately clear, I only guessed its meaning after reading the rest of the paper. The notion of "granular optimization" is also quite cryptic (as well as further usage of the word "granular" in the main text).
 
Even more importantly, the paper misses a clear overview of the overall computing pipeline corresponding to the proposed measure, and I had to make my own drawings to provide the (hopefully correct) summary above. Combined with the fact that the treatment of the different sections is significantly unbalanced,  this makes the paper quite hard to read (hence my score 1 on the presentation).

Section 2 is very short, while one would expect to have there an overview of the overall context and pipeline from a high-level perspective. This would seem to be the right place to highlight that the approach is frame-based, and that the frame index will be dropped later on. I missed it at first read.
Section 3 is essentially a reminder on diffusion maps, in my opinion this would have been better fit to an appendix : isn't it enough to know that it replaces a collection of vectors by a spectral-like embedding of the said vectors ? By the way I believe that it is an abuse of notation to consider $\Psi_t$ (or $\Psi_t^{(d)}$) as a function defined on $\mathbb{R}^D$: is only defined at the considered $x_i$'s, there is no natural out-of-sample extension for an arbitrary $x \in \mathbb{R}^D$. 
Suggestion: since the time-frames are denoted y_*, their (e.g. word2vec) embedding $x_*$, it would lighten the notations and ease the reading to use another letter, e.g. $v_*$ to denote what is currently written $\Psi^{(d)}_t(x_*)$. It would be sufficient to say that the definition of $v_*$ depends on some parameters ($t$, $d$ or $\tau$, but also $\alpha$)
Given the choice of $t=1$ in all experiments, the approach seems to fall back on a simpler method without diffusion, which seems to be very close to a plain Laplacian eigenmaps.

Section 4 is clear, but in my opinion would gain much by reworking the notations as suggested above. 

The claim about the limits of existing measures L042 ("defined with an intrinsic ambiguity) does not seem supported by evidence, and seems in contradiction for example with the very definition of SIR/SAR/SDR.
Table 1 does provide some evidence that the new measures "better correlate with human global quality assessment", yet this does not per se support a better 

The notion of intrusiveness seems to mean in the introduction that clean reference sources are required. I could not relate it to the wording "mildly-intrusive distortions". 

The notations are somewhat unclear, balancing between being too allusive or too explicit. For example, the problem formulation is frame-based, involving an index $f$ (which by the way in a signal-processing context often indicates frequency, I found this confusing at first read), but the indices are then "repurposed" and $f$ is dropped, this is also confusing.

The definition of some acronyms (MOS, NMI) should be recalled when they are first used several pages after being introduced.


There is no explanation on the definition of the "waveform-only" variant of PS/PM.

The aim and significance of the whole Section 5 on "error analysis" is totally unclear (hence my score 2 on soundness), and this reads as a series of tedious but relatively straightforward computations.

Although the results shown in Table 1 seem to provide evidence of the interest of the proposed measures, underlining the scores on the first two lines seems superfluous.
The database description of Section 6.1 is ambiguous : at first I understood the SEBASS precisely provides human ratings. Then L414 seems to suggest that 15 new certified raters were involved. Were they the 15 certified raters from SEBASS ? Can this be clarified, traced and referenced ?
If this paper involved new certified raters, this should be reflected in the Ethics part.

My understanding is that MUSHRA and MOS are different approaches to obtain statistically significant human ratings. The authors however seem to use both acronyms as synonyms.

The values of many parameters ($L$, $\alpha$, $t$, $d/\tau$, $M$) and some choices (e.g., which layers of the wav2vec / MERT models are used to generate the first embedding)
are provided without saying much on how they were selected, and whether this involved trial and error. Can you explain ?
If trial and error was conducted, some mild level of learning to match human ratings has been conducted, and this should be made explicit.

### Questions
- How does the diffusion map step with $t=1$ and $\alpha=1$ relate to Laplacian eigenmaps or a related simple spectral embedding approach ?
- Do you have evidence that the proposed measures "disentangle" leakage from "self-distortion" ?
- What is the "waveform-only" variant of PS/PM ?
- What is the aim and significance of the "error analysis" ?
- Did you conduct new listening test with 15 raters ?
- How were the parameters tuned ? Was there trial and error ?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces two frame-level, intrusive metrics: Perceptual Separation and Perceptual Match, for the assessment of audio source separation systems. The approach decomposes perceptual errors into leakage and self-distortion by encoding distorted signal variants through a pre-trained self-supervised model and embedding the resulting features onto a manifold using diffusion maps. Experimental results on the SEBASS benchmark demonstrate that both metrics offer robust correlation with human MOS compared to 14 existing measures, and the paper derives error and confidence bounds for both metrics.

### Strengths
1. The PS and PM measures distinctly address leakage and self-distortion, providing interpretable, complementary diagnostics as established in both the main text and Figure 1. This is a clear improvement over single-score evaluation methods, which frequently conflate distinct error sources.

2. The derivations for both measures are grounded in Mahalanobis distance computation on a perceptual manifold, with clear steps and full error analysis.

3. The paper goes beyond point estimates, deriving deterministic and probabilistic error bounds.

### Weaknesses
1. Several directly related and highly contemporary works on perceptual audio metrics, unsupervised perceptual embeddings, and manifold learning for audio are not discussed, despite their clear relevance.

[1] DeePAQ: A Perceptual Audio Quality Metric Based On Foundational Models and Weakly Supervised Learning
[2] NOMAD: Unsupervised Learning of Perceptual Embeddings for Speech Enhancement and Non-matching Reference Audio Quality Assessment

2. The paper states the hypothesis that diffusion distances on the perceptual manifold align with perceived similarity, but aside from empirical validation, this foundational assumption is not theoretically justified.

3. Despite benchmarking against many baselines, the evaluation gives limited direct insight into when/why PS or PM fails relative to credible alternatives (e.g., PESQ, STOI). The qualitative and error-case analysis is not sufficiently granular.

### Questions
1. Can the authors empirically or theoretically justify the assumption that diffusion map Euclidean distances on the perceptual manifold directly correspond to perceptual similarity, especially for out-of-domain distortions?

2. How robust are the PS and PM measures to errors in the choice of self-supervised model?

### Soundness
3

### Presentation
3

### Contribution
3
