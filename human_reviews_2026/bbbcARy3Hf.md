# PETRA: Pretrained Evolutionary Transformer for SARS-CoV-2 Mutation Prediction

- Decision: Reject
- Scores: 2, 0, 2, 6

## Abstract
Since its emergence, SARS-CoV-2 has demonstrated a rapid and unpredictable evolutionary trajectory, characterized by the continual emergence of immune-evasive variants. This poses persistent challenges to public health and vaccine development. 

 While large-scale generative pre-trained transformers (GPTs) have revolutionized the modeling of sequential data, their direct applications to noisy viral genomic sequences are limited. In this paper, we introduce PETRA(Pretrained Evolutionary TRAnsformer), a novel transformer approach based on evolutionary trajectories derived from phylogenetic trees rather than raw RNA sequences. This method effectively mitigates sequencing noise and captures the hierarchical structure of viral evolution. 

With a weighted training framework to address substantial geographical and temporal imbalances in global sequence data, PETRA excels in predicting future SARS-CoV-2 mutations, achieving a weighted recall@1 of 9.45% for nucleotide mutations and 17.10% for spike amino-acid mutations, compared to 0.49% and 6.64% respectively for the best baseline. PETRA also demonstrates its ability to aid in the real-time mutation prediction of major clades like 24F(XEC) and 25A(LP.8.1).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes PETRA, a transformer trained on evolutionary trajectories extracted from phylogenetic trees (UShER) rather than raw RNA sequence. They found PETRA can predict future SARS-CoV-2 mutations with a weighted recall@1 of 9.45% for nucleotide mutations and 17.10% for spike amino-acid mutations, which is a large improvement compared to 0.49% and 6.64% of the respective baselines.

### Strengths
The paper is overall well-structured and clearly written. It tackles an important problem of predicting SARS-CoV-2 evolution.

### Weaknesses
- The quality of the underlying phylogenetic trees data is questionable. The authors themselves acknowledge that the variant definitions from UShER, Nextstrain, and Cov-Spectrum “disagree in a lot of corner cases.”
- The sampling probability and temporal reweighting appears somewhat arbitrary, and it risk overfitting to recent and over-represented regions.
- Treating each country as homogeneous can distort the representativeness weighting and exaggerate biases toward well-sequenced urban centers.

### Questions
- The three-step variant-definition pipeline is reasonable, but appears heuristic. Have you evaluated the robustness of your method when trained on different tree construction methods? 
- Any ablations to show sensitivity of sampling probability and temporal factor parameters?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors trained a GPT model on top of existing sars-cov-2 sequences and try to predict the possible "next-strain" using the learned "evolutionary plausibility". While previous community effort on protein engineering have proven this strategy is effective, this paper is one of the several attempts in applying the same schema to dangerous infectious virus.

### Strengths
The paper is well written.

### Weaknesses
1. The model learns to extrapolate the pre-existing UShER tree, not viral evolution itself, making it useless for novel variants like Omicron where no such tree exists. It is pattern-matching on a graph, not learning biology.
2. The evaluation is critically flawed by the omission of direct comparisons to the actual state-of-the-art viral forecasting models discussed in recent scientific literature (e.g., scientific works mentioned in Nature News: https://www.nature.com/articles/d41586-024-04195-3).
3. The necessity of a massive GPT architecture is unproven, as the paper fails to benchmark against a much simpler, non-GPT autoregressive model applied to the same trajectory data.
4. Despite its predictive goal, the paper offers zero actionable scientific insights or generalizable rules of evolution, failing to justify the ethical risks of training a generative model on a dangerous pathogen.
5. The authors' ethical defense—that the model only predicts "natural" mutations—is doubly flawed: if so, why do we need your model? And the authors naively ignores the well-known risk of Transformer hallucination. Training generative models on viral sequences is fundamentally irresponsible.

### Questions
See comments above

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a transformer-based model, called PETRA, for learning the sequence of SARS-CoV-2 mutations which accumulate over time in different variants. A time- and geography- based weighting scheme is used to mitigate the effects of sampling biases and sampling time. The model can then be probed to predict the next mutations. On a new benchmark introduced in this paper, PETRA outperforms Bloom scores on predicting novel mutations of SARS-CoV-2.

### Strengths
- The time- and geography- based weighting is interesting and sounds more broadly applicable.
- The way in which the sequence is encoded is interesting -- one-hot encoding each site x mutation pair and concaternating them.
- Careful temporal train/test splits.

### Weaknesses
- The paper is poorly written in terms of grammar and phrasing. The abstract is split into three paragraphs. Grammatical errors are broadly present. Citation is non-standard, with citations after the period. An oddly harsh and dismissive phrase is used when referring to existing work:

"There also exist researches attempting to build up transformer-based models directly for SARS-CoV-2. (Shou et al., 2023; Feng et al., 2024) Nevertheless, these attempts focus on specially framed datasets of sequences from certain countries and time periods, and are hard to generalize and update according to developments of the virus, making them practically useless."

It is very likely that (1) this is the first time the authors are submitting to a major ML conference, (2) the authors are not fluent in English. These are not grounds for rejection, but it undermines the quality of the work.

-  The Bloom baseline is described as a "deep-mutational-scanning(DMS) based project" [side note: the space IS missing from the main text. There are several instances of these kinds of formatting oversights]. I took a look at the Bloom paper and it seems that Bloom is not a DMS-based method. The Bloom method used phylogenetic trees (much like PETRA) to map mutations and count their frequencies, leading to the fitness estimates for different mutations. The Bloom method is *validated* against DMS data, but is not a DMS-based method.
- PETRA only evaluates on its own mutation prediction task. How do the PETRA predictions correlate to the DMS datasets used in the Bloom paper? This would at least provide a clearer comparison against Bloom.
- It is not clear to me that Bloom scores are being used as intended by the Bloom paper. The PETRA paper proposes a composite Bloom score via s = ce^{\alpha f} which they show does better on their mutation prediction task than the Bloom fitness score or expected counts. This is quite odd. Why wouldn't the original Bloom paper propose such as score? Would this composite score s = ce^{\alpha f} also improve the Bloom correlations against DMS? Overall, it is not clear to me that the Bloom scores are being used as expected. Some discussion is necessary. The paper also arbitrarily sets \alpha=1 with no explanation.

### Questions
- Why did you split the abstract into three paragraphs?
- Why do you place citations after the period?
- Do you agree that the phrase "making them practically useless" is oddly harsh and dismissive of current published scientific work?
- Why do you call Bloom a "deep-mutational-scanning(DMS) based project"? From what I gathered from the Bloom paper, Bloom is a tree-based method (much like PETRA), which is *validated* against DMS data.
- Have you considered validating against the same DMS data as in the Bloom paper? While DMS data is not ground truth (several counterintuitive DMS scores are discussed in the Bloom paper), it would provide additional support for the performance of PETRA.
- How did you come up with the s = ce^{\alpha f} score for Bloom? How did you choose \alpha=1?
- Do you think the s = ce^{\alpha f} score for Bloom would improve correlation to DMS data?
- How exactly do you use Bloom scores to rank mutations? Provide further background on Bloom and how you use it in your benchmark.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript introduces PETRA, a pretrained evolutionary modeling framework designed to estimate virus mutations. PETRA leverages large-scale evolutionary sequence data to learn statistical patterns of antigen diversification, eliminating the need for experimental assays such as deep mutational scanning. The model outputs mutation-level fitness scores and functional annotations, and is intended to support high-throughput antigen evaluation and rational vaccine design. By learning both global evolutionary constraints and local biophysical signals, PETRA aims to generalize to previously unseen antigens and assist in predicting which mutations are likely to emerge or be tolerated under immune selection pressure.

### Strengths
A key strength of PETRA is that it addresses a major bottleneck in immunogen design: experimental characterization of mutational effects is slow, expensive, and inherently incomplete. The model’s zero-shot capability suggests it captures generalizable evolutionary principles rather than memorizing training examples. The authors emphasize that mutational fitness depends jointly on structural context and immune-driven selection, an important insight consistent with known antigenic drift dynamics. PETRA’s ability to annotate mutation functionality at scale is highly relevant for surveillance pipelines, early variant risk assessment, and computational vaccine candidate prioritization. The approach is also timely, given the growing interest in foundation-style models for biological sequence evolution.

### Weaknesses
One limitation is that the manuscript appears to focus primarily on sequence-based learning; explicit integration of three-dimensional structural context, epistatic coupling, or antibody–antigen interface geometry is not clearly articulated. Viral evolution is strongly epistatic, yet the evaluation setup seems to emphasize single-mutation effects, leaving open how well PETRA handles combinatorial variants observed in real variants of concern. The experimental validation section would benefit from broader benchmarking against state-of-the-art protein language models, sequence-to-fitness predictors, or phylogenetic fitness estimators. It is also unclear how robust PETRA is across diverse viral families with different evolutionary pressures. Finally, while zero-shot results are highlighted, the study does not deeply quantify failure cases, calibration, or false-positive risk.

### Questions
1. How does PETRA model non-additive interactions among multiple mutations, especially those common in real antigenic evolution (e.g., RBD co-mutational clusters)?
2. Does the model incorporate protein structure (e.g., contact maps, surface exposure), and if not, how might this limitation impact predictions at antibody epitopes?
3. How well does PETRA transfer to viruses with distinct evolutionary constraints, such as influenza HA or HIV Env?
4. Are fitness scores calibrated to biological magnitude (e.g., effect sizes comparable to deep mutational scanning measurements), or only relative rankings?
5. Can PETRA highlight specific residues or functional regions driving predicted fitness changes, and are these consistent with known neutralizing epitopes?
6. What controls are in place to avoid over-prediction of beneficial mutations, given that most random mutations are deleterious in nature?

### Soundness
2

### Presentation
3

### Contribution
2
