# Controlling Repetition in Protein Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Protein language models (PLMs) have enabled advances in structure prediction and de novo protein design, yet they frequently collapse into pathological repetition during generation. Unlike in text, where repetition merely reduces readability, in proteins it undermines structural confidence and functional viability. To unify this problem, we present the first systematic study of repetition in PLMs. We first propose quantitative metrics to characterize motif-level and homopolymer repetition and then demonstrate their negative impact on folding reliability. To address this challenge, we propose UCCS (Utility-Controlled Contrastive Steering), which steers protein generation with a constrained dataset.
Instead of naively contrasting high- vs. low-repetition sequences, we construct contrastive sets that maximize differences in repetition while tightly controlling for structural utility. This disentanglement yields steering vectors that specifically target repetition without degrading foldability. Injected at inference, these vectors consistently reduce repetition without retraining or heuristic decoding. Experiments with ESM-3 and ProtGPT2 in CATH, UniRef50, and SCOP show that our method outperforms decoding penalties and other baselines, substantially lowering repetition while preserving AlphaFold confidence scores. Our results establish repetition control as a central challenge for PLMs and highlight dataset-guided steering as a principled approach for reliable protein generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the problem of pathological repeats in sequences sampled from protein language models. The authors devised metrics (token entropy, n-gram diversity, homopolymer diversity) to quantify this phenomenon and leverage contrastive steering at inference time via a dataset of utility matched pairs to alleviate this issue. I appreciated the motivation of this work and the thorough investigation of failure modes in the design of the metrics. I also appreciate the simplicity of the contrastive steering solution and its broad applicability among protein language models. However, there are a few concerns with the current paper that make me lean towards rejection. I am open to increasing my score if these concerns are well addressed.

1) While the metrics to capture pathological repetitiveness seem well justified, the claim that they *sharply separate natural proteins (POS) from PLM generations (NEG)* seems overstated as the distribution plot of $R_{hpoly}$ for ProtGPT2 samples is actually quite similar to natural datasets.
2) I believe that the poor performance of ESM3 is related to the extremely low number of decoding steps (20) used in the experiment. Could the authors provide results with higher number of decoding steps (ideally the same number as the protein length)?
3) Given the generality of the framework, could the authors expand the scope of PLM models they evaluate? ESM3 also models structure and function which could confound some of the results here. There are many popular and performant PLMs such as the ProGen family and ESM2. 
4) While the authors provide thorough results on the utility and repetitiveness of generated samples, they did not report the impact of each method on sequence diversity and novelty. I am concerned that the steering could restrict sample diversity and bias generations towards positive samples in the contrastive set.

Comments to improve the paper, not related to score
- The plots in the paper are all quite small and difficult to read without zooming in
- I would move the biological discussion supporting pathological repetition in proteins from the Appendix A to the main text

### Strengths
- well motivated and tackles an under-researched problem 
- thorough and sound experiments
- positive results for contrastive steering

### Weaknesses
- did not evaluate enough PLMs
- sampling choices for ESM3 are too simple
- did not report effect on sequence diversity and novelty
- some results are overstated (see summary point 1)

### Questions
See summary points 1-4

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
4

### Summary
This paper studies the phenomenon existing in the modern protein language models (PLMs) that the generated protein sequences usually exhibit  ill repetitions as a negative effect regarding the utility like foldability. The authors define the repetition score, utility score, and propose the UCCS, as a remedy strategy to help PLMs have improved repetition score while keeping utility. Experiments shows the UCCS can help ESM3 and ProtGPT2 reduce repetition and achieve comparable or higher plddt scores.

### Strengths
- The problem studied is relevant and interesting. The repetition problem in PLMs deserves more attention for the related research efforts
- The proposed method, UCCS is model-agnostic and plug-and-play, making it easy to implement and adapt for existing PLMs
- The experimental results support the claim that the proposed UCCS “mitigates degeneracy while preserving foldability”, yet I believe it can be further improved (refer to cons/questions).

### Weaknesses
- This study still does not explicitly answer (or explore further) why common PLMs (or some of the PLMs) intrinsically exhibit repetition in their generated samples, which I think is more important to guide the PLMs design for the community; furthermore, AlphaFold-like confidence score (aka foldability) only serve as a probe and can hardly counted as strong indicators for good design, broader metrics should be considered in the main results to let this paper become a comprehensive study. I do not think the current shown results are convincing enough to make UCCS really useful in the real design cases.
- Given the authors claimed "model-agnostic" and “plug-and-play” steering method, benchmarking broader families of PLM should be considered, such as Progen[1] and DPLM[2] series. Improve the limitation in base model selections shall make the study more complete and resutls convincing and have larger impacts.

### Questions
- Though the paper has already included three small structure and sequence-only dataset, I think the “sensitivity” of steering datasets should be more systematically studied. Recent protein design models may involve structure data other than PDB (eg. AFDB or SwissProt) into training, Could the authors try quantitatively studying the dataset sensitivity of involved steering methods (such as temperature sampling and the proposed UCCS)? This may help the readers understand which strategies work consistently in reducing the repetitions.
- (referring to one of the concerns above) Could the authors additionally consider benchmarking broader families of PLMs into the result table?
- There are existing in-practice strategies to reduce sequence repetition (such as the remasking in MultiFlow[3] and gumbel+resampling in DPLM[2], to name a few). Could the authors also incorporate  these strategy as practical and competitive “heuristics and interventions” by showing similar benchmark and visualize the pareto front of R and U?

Minor typos:

- Ubiquitous UUCS -> UCCS?
- Column 1 in Table 1 seems not balanced in the cell
- The text in Figure 1 is too small to read

References:

[1] Nijkamp, Erik, et al. "Progen2: exploring the boundaries of protein language models." Cell systems 14.11 (2023): 968-978.

[2] Wang, Xinyou, et al. "Diffusion language models are versatile protein learners." arXiv preprint arXiv:2402.18567 (2024).

[3] Campbell, Andrew, et al. "Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design." arXiv preprint arXiv:2402.04997 (2024).

### Soundness
3

### Presentation
2

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
The paper systematically studies pathological repetition in protein langauge model. 
First, the authors formally define repetition in PLM and introduce quantitative metric that capture both motif-level and homopolym er repetition. These metrics are strongly correlate with structural utility measured by AlphaFold pLDDT.
Second, the authors introduced Utility-Controlled Contrastive Steering. They propose a representation-level intervention that disentangles repetition from structural utility. And inject such representation during inference stage to generate less repeated sequence.
The experiments show that UCCS reduces repetition in both MLM and autoregressive PLM.

### Strengths
1. This paper first systematically identify and formalize pathological repetition in PLM, with quantitative metric. The metrics are well-motivated and interpretable.
2. The UCCS approach is simple, training-free and model agnostic. It greatly reduces repetition while maintaining foldability.
3. The experiments cover both MLM and autoregressive models, both unconditional and conditional generation settings with multiple datasets.

### Weaknesses
1. Limited Novelty. I'm not familiar with steering methods during inference time with LLM, but according to the cited papers, it looks like deriving the difference vector and adding it during inference has been greatly explored in LLM. The adaptation to PLM is incremental rather than fundamentally new.
2. Comparison with learning based methods. Similarly, I have no idea about if people reduce repetition with learing-based method in LLM. If so, is it possible to compare UCCS with these learning based methods?
3. Lack of interpretation. While UCCS greatly improve the results, the paper would be stronger if it included analysis and visualization about the steering direction about biological meaning. For example, the model was about to generate AA...AA, and how steering method "pull" it back to a normal protein and why.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents the first systematic study of pathological repetition in protein language models (PLMs), proposing a novel method called Utility-Controlled Contrastive Steering (UCCS). The method effectively reduces repetitive amino acid sequences during generation, critical for biological viability, without sacrificing structural plausibility. The approach is elegant, practical, and clearly advances both the theoretical framing and engineering solution space for PLMs.

### Strengths
a.	This paper is the first to clearly define and systematically address the problem of repetition in protein language models. This is done while understanding biological consequences. Rather than borrowing loosely from NLP, the authors develop a domain-aware framework that captures why repetition is especially damaging in proteins. This makes framework, truly domain specific.
b.	The proposed method, UCCS, stands out for its simplicity. It doesn’t require retraining the model or altering its architecture, just a novel use of internal representations to steer generation away from pathological repetition. That makes it both practical and applicable to different PLMs.
c.	The technical work is robust and well-grounded. The authors design meaningful metrics, like Rhpoly, to detect homopolymer collapse, and run thorough experiments across two leading PLMs (ESM-3 and ProtGPT2), three canonical datasets (CATH, SCOP, UniRef50), and multiple generation settings. Their method reduced repetition while preserving structural plausibility, as measured by AlphaFold confidence scores.

### Weaknesses
a.	While UCCS performs well on ESM-3 and ProtGPT2, it’s still an open question how well the approach would transfer to larger or more diverse protein models like ProGen2 or ProteinMPNN? It would be valuable to see whether the same steering technique holds up as models scale or shift in architecture.

### Questions
a.	Did you experiment with combining UCCS with decoding heuristics (e.g., top-p + UCCS)? Is there synergistic gain?

### Soundness
4

### Presentation
3

### Contribution
4
