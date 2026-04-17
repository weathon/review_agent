# SCRIBE: STROKE- AND CONTEXT-REGULARIZED TEST-TIME ADAPTATION FOR HANDWRITTEN TEXT RECOGNITION

- Decision: Reject
- Scores: 2, 4, 6, 8

## Abstract
Handwritten text recognition (HTR) converts images of handwritten text—from
lines to full pages—into accurate, machine-readable transcriptions. However, it
often operates under distribution shift—new writers, historical substrates, scan-
ning artifacts, layouts, and even cross-language use—precisely when target la-
bels and source data are unavailable. Although recent foundation models per-
form well on their training distributions, their generalization across domains is
fragile. Limitations in capacity, inadequate pretraining scale, or corpus–domain
mismatch frequently lead to pronounced errors, underscoring the need for effi-
cient adaptation even with state-of-the-art pretrained models. We fill this gap by
adapting a foundation model at test-time without labels or source data. To the
best of our knowledge, this is the first HTR test-time adaptation approach that
jointly optimizes a lightweight stroke-structure loss with a document-conditioned
language prior, rather than treating linguistic (LM decoding/reranking) and vi-
sual (self-training/normalization) cues separately. Evaluated on four benchmarks
(George Washington, IAM, RIMES, Bentham), our approach achieves an aver-
age absolute reduction of 0.0341 in CER and 0.0427 in WER, corresponding to
mean relative improvements of 20.8% and 12.8%, respectively. These findings
demonstrate that integrating lightweight visual and linguistic priors provides an
effective strategy for test-time adaptation in HTR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SCRIBE, a framework for test-time adaptation of handwritten text recognition OCR models, using unpaired data from the target domain to improve the performance. The experiments are done on the TrOCR family of models, using up to 5 updates on a batch of 8 samples of unlabeled test data. The adaptation has 3 main components:

1. A "skeleton decoder" head attached to the visual encoder, enforcing consistency of the features coming out of the encoder with the skeleton extracted from the original image (the metric used is Chamfer pairwise set distance + IoU-type Dice distance on the set of pixels in the boolean masks). The skeletonization is done using erosion + dilation..

2. LM-Guided KL regularization with GPT-2 LM, where the probability distribution coming from the model is fused with probabilities from the LM - KL divergence term between this distribution and the one from the model decoder pushes the decoder to produce distribution closer to the fused one.

3. LM Rescoring (post-adaption) - During decoding, beam search elements are re-ranked based on fusing the score with the score coming from the language model (GPT-2).

The evaluation is done on 4 datasets (IAM, RIMES (French), GW (18th century English letters), Bentham (19th century English manuscripts)). The comparison is done to a number of other HTR models (much worse than the base TrOCR model). The ablation study compares to the base TrOCR model itself as well as leave-one-out on 3 suggested components. Comparison to other, simpler TTA approaches is performed. Comparison of fine-tuning and fine-tuning with TTA is performed in appendix C.

### Strengths
Originality: The individual ideas (LM rescoring, skeleton losses, TTA) are not new. The originality lies only in the specific combination of a geometric loss and an LM-based loss within a TTA gradient loop for HTR.

Quality & Clarity: The paper is clearly written and easy to understand, the proposed approaches show some improvements compared to the baseline model, the ablation study studies the components well. However, the actual results presented by the authors (Table 3) undermine the paper's main narrative about the need for joint visual and textual test-time adaptation, furthermore there is significant risk of contamination affecting the results (hence the soundness score. See 'Weaknesses' below).

Significance: Between the results of the paper suggesting that the central idea of joint visual+text TTA is not necessarily the main driver of the results, the risks of possible contamination, and absence of details on the computational complexity of the proposed approach, the significance is limited.

### Weaknesses
The main weakness of the paper is the disconnect between the central claim about the usefulness of visual-textual TTA and the presented results:

1. The results in Table 3 suggest that effect of visual adaptation is minimal (removing it minimally increases the WER) and that generally the effect of using LLM for re-scoring the beam-search decoding (which is not TTA) is the strongest among all 3 components. The ablation study could be significantly strengthened by showing the results of the baseline that does not use any TTA but only uses LLM-based rescoring for decoding.

2. The datasets used for evaluation, in particular IAM (which is 25+ years old)  and RIMES (which is publicly available on HuggingFace / Kaggle since a long time) are quite likely present in the pre-training corpus of the GPT-2 model, which makes it impossible to decouple the effect of using the LM in general, from the result of training data contamination. The evaluation could be significantly strengthened by evaluation on the datasets released recently and with larger dataset drift than the English-French (ex. math handwriting recognition based on handwritten portion of UNIMER-1M or MathWriting).

3. The computational burden of the proposed approach is not discussed, but given that TrOCR-base is ~224M and TROCR-large is~493M parameters, while GPT-2 Large is ~774M parameters, this likely increases the computational footprint by a factors of 2.5x-5x. Currently results from Table 3 shows that improvement from going TrOCR-base to TrOCR-large (5.1->3.6 CER) helps more than adding the proposed approach (5.1->3.7 CER), while also being more parameter-efficient.

To sum up, in my opinion, for this paper to be a significant contribution, it needs to show:
- That TTA is actually brining measurable improvement on top of the LLM-based reranking with beam-decoding
- Performance on datasets that have clearly not contaminated the LLM pre-training corpus
- That increase in computational footprint is justified.

### Questions
No additional questions, but I would love for the authors to address the 3 points I've outlined in the "Weaknesses" section.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focus on test-time adaptation for HTR. The authors present a label-free and source data-free test-time adaptation technique that enhances the generalization of HTR models on unseen domains by jointly optimizing a lightweight stroke-structure loss and a document-conditioned language prior. Experiments demonstrate the effectiveness of the proposed method.

### Strengths
1.	This approach involves updating models online on unlabeled target data. It does so without source replay or target-only fine-tuning, which is an interesting idea.

2.	The proposed method achieves competitive results.

### Weaknesses
1.	The author fails to clearly elaborate on whether the Test-time adaptation of HTR is still worthy of research in the era of Multimodal Large Language Models (MLLMs), and whether MLLMs have achieved sufficient generalization across different scenarios such that further research on the test-time adaptation of HTR is unnecessary.

2.	The foundation model (eg GPT-2) being used is relatively old. The foundation model used, TrOCR, is two years old, and the language model, GPT-2, is also relatively outdated.

3.	HTR models typically have strict requirements regarding model parameter and inference speed. However, the LM reranking method proposed in this paper substantially increases the model parameter and reduces inference speed, which may limit its practical applicability.


4.	The authors describe the method as plug-and-play, yet it has only been validated on TrOCR and lacks validation on many other relative models.

### Questions
Please see the Weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors proposed an interesting approach of test-time adaptation for handwritten text recognition. This application normally suffers from distribution shift, for example, new writers, new historical substrates, a lot of noise, different layouts, and languages. The idea is to couple a stroke-preserving geometric with a document-conditioned language. The paper shows that the approach improves TrOCR performance across four benchmarks. 

The proposed label-free skeleton loss uses an uncertainty-gated KL for a document-aware LM teacher, they also added a sequence-level LM reranking, which offers an interesting combination for source-free that adapts optics recognition without language or applies LM guidance without geometric anchoring. I see this as the main contribution.

The idea may help with operational constraints of handwritten text recognition and privacy-sensitive uses.

### Strengths
I liked the idea of a stroke-preserving self-supervision on an auxiliary skeleton head, especially the Dice and Chamfer loss. The document-conditioned language proposed is using both loss as an uncertainty-gated KL distillation term, which helps a kind of sequence-level shallow-fusion reranking of the outputs and improving performance. The authors should explore this part in the paper.

The paper also demonstrates consistent gains on IAM, RIMES, GW, and Bentham. The interesting part is that they have the same budget and without a source reply.

The main idea is to couple visual topology preservation with linguistic regularization and this occurs at both token and sequence levels.

Also liked the losses used, the Dice+Chamfer and uncertainty-gated LM-KL. The strategy for sensible gating and the reset strategy help to limit drift. 

The authors perform ablations to show the component’s effect.

The experimental results show gains on TrOCR backbones over multiple datasets.

### Weaknesses
Not sure I understand correctly, but in Table 2, the proposed method adapts all encoder+decoder parameters, while TENT/EATA baselines adapt LN/bias only. If it is true, that is an unfair comparison that can inflate margins.

Table 3 removes components from the full system but never shows frozen TrOCR + LM reranking or frozen TrOCR + LM-KL, which seems to have no weight updates. Without this it is not easy to understand which component is responsible for the results, part of the improvement could be from better decoding rather than adaptation. Please add frozen + LM-rerank and frozen + LM-KL rows and also quantify the net “adaptation-only” delta.

I don't understand the description of Fig. 3 that alternately describes erosion, centerline/contour overlays, and cite thinning in that context, these are not equivalent and affect the claimed topology preservation.

I believe there is some backbone inconsistency across datasets, as IAM is evaluated with stage-1 while others use base (and sometimes small/large), which may confound average gains.

Please add confidence intervals/repeated runs to evaluate the statistical significance of the results.

Improve the methodology description, as some reproducibility hyperparameters (λ’s, α, β, τ, K, δ, skeleton resolution/threshold) are missing from the text and are needed for reproducibility.

The claim on novelty also needs a better discussion, the “first source-free HTR TTA” could be regarded as overreaching since Tula’23 and Gu’25 also present contributions in this direction. This could be better addressed by the authors in the discussion.

### Questions
The experiments only addressed Latin-script datasets. Do you think performance will be maintained with a non-Latin corpus?

What is the delta from frozen TrOCR + LM reranking and frozen + LM-KL (no gradient) relative to full TTA?

Why does GPT-2 large outperform CamemBERT on RIMES? Please compare with a French causal LM of similar capacity and detail tokenization/normalization.

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
5

### Summary
This paper deals with test-time adaptation (TTA) of handwritten text recognition models. The authors use a pre-trained TrOCR model and joinltly optimize the encoder and the decoder at test-time, without any labels.

    The encoder is optimized with a skeleton-aware optimization, with skeleton & topology losses.
    The decoder is optimized with teacher guidance of an external LM that reranks the top K predictions of a greedy decoding of TrOCR.
    Finally, once the model is adapted, the top K candidates of the adapted model are re-scores by the LM and the top 1 prediction is kept.

Evaluation is performed on 4 datasets: Bentham, IAM, GW (all English) and RIMES (French). Results show that test time adaptation consistently reduces error rates on OOD domains, but also in the fine-tuning framework

### Strengths
- The paper is very clear and well illustrated. 
- The methodology is sound and many details are provided, ensuring reproductibility. Moreover, the code will be released. 
- The paper focus on an under-explored paradigm (TTA) which addresses a practical need in OCR/HTR deployment. This is particularly valuable as: 
    - Most papers focus on improving fine-tuning strategies
    - But annotations are not always available and can be expensive & time-consuming to obtain
    - Foundation models (like TrOCR, but also most recent VLMs) are difficult to fine-tune due to data requirement, computational cost, large computing ressources...
- The proposed method is strong and its robustness has been demonstrated in the paper. 
    - Multiple checkpoints and LM are evaluated on 4 datasets
    - The proposed approach is compared to other TTA methods
    - The authors have conducted an ablation study and demonstrated that all 3 components are important
    - Consistent improvements are reported across different settings
- This is the first TTA approach to optimize both the encoder & decoder at test-time
- I believe this paper will encourage research towards this direction, and will be of interest for many ICLR attendees.

### Weaknesses
- The comparison with SOTA models is not fair. 
    - The models presented as SOTA were heavily specialized on a single collection of documents, while TrOCR is a generic model trained on millions of synthetic lines and other datasets 
    - Comparing specialized single-domain models to a generic model on OOD data is unfair (of course the specialized models perform worse...)
    - Legends of Fig 1 and Table 1 should be rephrased. At this point of the reading, the scope of the paper is not defined, so context is needed (e.g. replace "SOTA" by "specialized models evaluated on OOD data"). Otherwise the "SOTA" CER/WER are misleading.
    - Missing experiments: it would have been interesting to apply your method to these specialized models and/or to other generic models (VLM)
- The inference cost/adaptation cost is not presented
    - The method requires: greedy decoding → adaptation (U steps) → beam decoding → LM reranking
    - How much time does this pipeline require, compared to basic inference? 
    - How does this compare in terms of time & money to the traditional pipeline: annotate → fine-tune → deploy?
- Missing experiments
    - Evaluation of specialized models on ID domains (to have an idea of what a good CER/WER is on each dataset)
    - Fine-tuning of TROcr on all datasets, then applying TTA and comparing results 
        - I know some experiments are presented in the Annex - they should be in the main paper in my opinion
        - Experiments on Bentham are missing + error rates are very high for Rimes -> why?

### Questions
- Context buffer (M)
    - How is M chosen? Did you experiment with different values?
    - How is context handled for the first line(s) in a batch/document?
    - How does the method perform with batch size = 1?
    - What happens when a page contains multiple writing styles or writers?
    - This adds a decoding constraints (parallelization is difficult) that should be discussed
- Language model: 
    - GPT-2 is an English model / CamemBERT a French model - why not use a multilingual LM?
- Skeleton: 
    - How would this work on degraged / hard to binarize documents? 
    - What are some failure cases?
- Inference time 
    - What inference time with / without TTA?
    - How does this scale with batch size, number of updates U, and beam size B?
- Results
    - Why do TENT and EATA work well on IAM but still show high errors on other datasets?
    - Why are fine-tuned results missing for IAM and Bentham?
    - Table 2: how do you explain that IAM results are good, while other datasets still have very high CER/WER?
- Evaluation
    - Are metrics computed at line-level or at document-level?

### Soundness
4

### Presentation
3

### Contribution
3
