# The EEG activation maps in recent work are uninterpretable by experts

- Avg Score: 1.50
- Decision: Reject
- Scores: 0, 2, 2, 2

## Abstract
Recent papers claim to decode object class from EEG recordings of
subjects viewing image stimuli from ImageNet and to use that
classifier to construct activation maps for the depicted object class
that are consistent with neuroscience knowledge.  Empirical evaluation
of the activation maps by EEG experts calls this claim into question.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper presents an empirical evaluation of activation maps (generated in another paper) and argues based off the result that the evaluators were unable to map the object class and activation map with above chance accuracy that those activation maps therefore call the validity into question. Furthermore, based off this result the paper argues that this implies that the dataset used in the other paper (Spampinato et al 2017) and other datasets with a similar temporal confound should be avoided and any results drawn from these datasets should be discounted.

### Strengths
It is meaningful to highlight limitations in prior work to both help correct misunderstandings that may arise from that prior work and to help push the field forward.

### Weaknesses
1. Relevance to call for papers: This work is not aligned with the focus of this conference as defined in the call for papers (https://iclr.cc/Conferences/2026/CallForPapers). While "applications to neuroscience & cognitive science" is a subject area in the call for papers, this is not an application of "feature learning, metric learning, compositional modeling, structured prediction, reinforcement learning, uncertainty quantification and issues regarding large-scale learning and non-convex optimization" or related topics to neuroscience & cognitive science.
2. Relevance to ICLR audience: While highlighting limitations in prior work is valuable, of all the prior work that is being highlighted none of these works were previously published in ICLR. This further supports that this workshop is not the correct venue for this paper, as this limitation will likely be of little interest to the ICLR audience. If there is interest, as the paper itself mentions there are other papers that have previously pointed out limitations to the same body of work this paper focuses on that could be read. 
3. Soundness, Claim is not scientifically rigorous: The paper presents extremely limited details about how the "experts" who conducted the empirical evaluation were identified, which limits the strength of the only claim in the paper. Additionally, only summary statistics are presented which prevents the reader from being able to assess the data and draw conclusions on their own about the implication of the empirical evaluation. Furthermore, one of two examples of the empirical evaluation that is shown (Fig 2) raises the question of how the experts would be expected based off their neuroscience knowledge identify which activation map was correct as both overlap with the occipital cortex and the specific localization of jack-o-lanterns is not something that is taught as part of EEG analysis even to experts. 
4. Presentation, Motivation is unsupported: The paper boldly claims that "Palazzo et al" generally claims that the "activation maps are consistent with neuroscience knowledge" and then presents 7 quotes from "Palazzo et al" papers. From these 7 Palazzo et al" quotes, only 1 quote presents any support for this claim. The fact that the evidence presented in the paper to support the motivation for highlighting a limitation in the "Palazzo et al" papers is unclear in how it supports the claim is a key weakness.

### Questions
It would be valuable if the paper explained: 
1. How the experts expertise in EEG analysis was tested before they participated in the empirical evaluation? 
2. Whether the experts are in fact experts at object classification activation maps, or if there EEG expertise in in another area? 
3. What were the raw results of the empirical analysis? Were there any categories that the experts agreed with the activation maps? 
4. The reason why quotes were included in the intro (taking up almost 1 full page of text) when 6 of the 7 quotes do not support the claim they are included to support? Perhaps there is nuance that is not clear from merely reading the quotes. 

However, clarifying these points will not chance that the paper is not relevant to the call for papers nor for the ICLR audience. Therefore, I'd recommend the paper be updated to address the above points and submitted at a neuroscience focused venue instead.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses a critical issue in EEG-based visual decoding: the lack of expert interpretability of EEG activation maps proposed in a series of high-impact works. It shows that the activation maps—purported to align with neuroscience knowledge—are not interpretable by EEG experts.

### Strengths
1. The proposed question is interesting.
2. The references are comprehensive.

### Weaknesses
1. Asking humans to interpret EEG spectrograms and map them with true stimuli is an excessively difficult task. Generally, EEG can only capture superimposed general visual features. Hence, the proposed experimental paradigm is not feasible. Meanwhile, this design cannot verify that deep models are unable to extract distinguishable features, as human visual perception is inherently limited.
2. Lack of analysis: only averaged results are reported, no detailed analysis and case-by-case discussions are provided. For example, is there any case where humans can reach high accuracy?
3. The study only demonstrates that the EEG response to a single image cannot be identified by humans, but it cannot prove that the EEG response related to visual perception is uninterpretable.
4. Lack of insight and contribution on how this research can advance the area.

### Questions
Could you clarify the definition of "activation maps are consistent with neuroscience knowledge"? Is it a necessary conclusion that the EEG response to a single image cannot be identified by humans?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper mentioned a crucial problem faced by the field of EEG-based visual decoding, which was incurred by the flawed dataset for a previous paper. However, this paper is more like an unfinished comment that needs substantial data analysis.

### Strengths
This work pointed out an important issue that is easily faced in the experimental design for brain decoding.

### Weaknesses
Hardly any analysis was presented in the article.

### Questions
The paper from Li et al. [1] thoroughly analyzed the shortcomings of the work from Spampinato et al. On this basis, what improvements does this article propose?

[1] R. Li et al., "The Perils and Pitfalls of Block Design for EEG Classification Experiments," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 1, pp. 316-333, 1 Jan. 2021
[2] C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, and M. Shah, “Deep learning human mind for automated visual classification,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017, pp. 6809–6817.

### Soundness
1

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper audits published EEG "activation maps" from a prior pipeline by running a blinded expert-judgment study (two forced-choice tasks) to test whether the maps are interpretable to humans. The authors report that experts do not reliably match maps to stimuli and conclude that such maps should not be used to support neuroscience claims about the underlying dataset or model line. The stated contributions are: a focused human-study evaluation of published EEG maps from a prominent prior work, and a critique arguing that these maps, as presented in the literature, do not carry the neuroscientific signal they are often claimed to reflect.

### Strengths
1. Clear, concrete target: evaluates what readers actually see in prior publications (published maps), not a moving reimplementation.

2. The overarching question whether these maps are meaningfully interpretable by domain experts is timely and under-tested.

3. Simple, transparent task design that is easy to reproduce.

### Weaknesses
1. This is a narrow human-subjects audit with minimal algorithmic or methodological innovation. For an ICLR main track, the CS contribution feels thin; the work reads as a focused methods/ethics evaluation better suited to a special track or journal.

2. The study evaluates maps from a single prior pipeline yet the paper’s narrative gestures toward invalidating a broad literature. The conclusion should be bounded to the artifacts actually tested. This is not a request to "test all papers"; it’s a request to either restrict the claim to the evaluated pipeline, or add a small, diverse sample of other pipelines if the broader claim is essential.

3. There is no positive control using a known good EEG paradigm and/or dataset to verify that the interface/task can detect interpretability when it’s expected to be present. Without this, the null finding could be driven by task design, map rendering, or the natural limits of EEG for fine-grained image classes. Likewise, no negative controls (e.g., randomized/permuted or class-swapped maps) are included to calibrate chance-level behavior.

4. Per-class topographies for natural images are an extremely demanding target for EEG (low SNR, coarse spatial resolution, limited coverage). A failure on this task does not uniquely imply "non-interpretability" of the underlying neural activity; it may reflect a mismatch between the claim tested and what EEG can plausibly support.

5. "EEG expert" is a broad term. The sample mixes backgrounds and seniority, and it’s unclear how many participants have direct experience with cognitive/vision EEG map interpretation versus clinical EEG. This weakens the negative inference.

6. The maps appear heavily smoothed/rasterized rather than recomputed from source. If originals cannot be regenerated, the paper should analyze and justify how extraction/smoothing affects discriminability and discuss robustness to these presentation choices.

7. The tone and structure feel informal for an A* venue (short abstract that under-reports the methods/results, heavy quoting/bullets). The paper would benefit from tighter framing, clearer hypotheses, and more rigorous statistical exposition without going into excessive polemics.

### Questions
Will you constrain the central claim to the specific pipeline you evaluated, or add a small sample from other pipelines to justify broader statements?

1. Can you include a positive control (even a compact add-on) using a well-established EEG paradigm to demonstrate that your task/interface can detect interpretability when present?

2. How were "EEG experts" screened for relevance to cognitive/vision EEG topographies (as opposed to clinical EEG)? Would results change if limited to that subgroup?

3. If recomputing maps from source is infeasible for the rebuttal, can you document the exact provenance and quantify how rasterization/smoothing might degrade discriminability (e.g., show side-by-side renderings with/without smoothing or different interpolation)?

4. Can you refine the hypothesis to acknowledge EEG modality limits and rephrase conclusions accordingly (e.g., "these published maps, as rendered and for this task, do not support the claimed interpretability")?

### Soundness
2

### Presentation
1

### Contribution
2
