# UNISE: Unified Noise-Invariant Learning for Speech Enhancement toward Improved Content Preservation

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
The importance of semantic information in speech enhancement (SE) has recently been emphasized to improve intelligibility, whereas earlier work primarily focused solely on acoustic perceptual quality. To address this, recent approaches leverage pre-trained self-supervised representations, which have shown strong performance on {discriminative} tasks. However, such representations are less effective for {generative} tasks and, since they are typically trained only on clean data, struggle to fully preserve content under noisy or distorted conditions. 
In this work, we aim to bridge this gap by introducing a unified generative SE model, called \textbf{UNISE}, that incorporates noise-invariant representation learning. By jointly learning an encoder using noise-invariant clustering and a generative decoder, our model produces robust speech representations well suited for the SE task.
As a result, UNISE achieves improved linguistic content preservation while maintaining competitive perceptual quality\footnote{Audio samples are available at: \url{https://tinyurl.com/UNISE-ICLR2026}}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a "unified generative SE" model called UNISE. The model aims to incorporate a noise-invariant representation to improve robustness and content preservation. The noise-invariant representation is achieved by defining a learning target that maps multiple views of the same clean speech under different distortions to the same target cluster. The clustering objective is employed after each layer of the proposed encoder. A syllabic label prediction task is used as an auxiliary loss to improve content preservation. The noise invariant encoder is combined with a generative decoder employing flow-matching. The authors compare their model to multiple other competing models and demonstrate that their approach yields benefits in content preservation against most of the tested models. They legitimize their findings by multiple ablations or further tests on ASR performance, different loss combinations and a consideration of mutual information between their learned embedding and a phone target across multiple SNRs.

### Strengths
The problem of content preservation in generative models is indeed a relevant and timely research topic, which makes the proposal interesting for the community.

The proposed method follows past research and expands upon it in a plausible way. The proposed method achieves a reduction in word error rate over compared generative speech enhancement models while achieving comparable DNSMOS and SpkSim values, indicating preserved quality. 

We appreciated the audio samples, which support the claim of increased content preservation in contrast to the also featured SGMSE.

Appendix 2 is a welcome additional analysis comparing the mutual information between the encoder output and corresponding phone labels, which shows a high degree of mutual information  that remains high under low SNRs in case of the proposed model.

### Weaknesses
While the paper is well-written w.r.t. phrasing, grammar and overall clarity, aspects of the presentation inhibit effortless understanding. 

Mathematical notations and figures have alignment issues.
- Section 2.1 introduces output representations z_a and z_b but Figure 2 uses different labels z_1 and z_2 for presumably the same concept? 
- Equation (3) includes a summation over k, which is not explicitly defined. 
- The pseudo-label prediction task is defined as y but y is never referred anywhere. The same goes for the Sylber features s, which do not occur in any figure or equation thereafter.
- It is furthermore unclear what the index c in Equation (1) means. Defining (capital) C as a learnable codebook does not obviously define the summation.

We find that a very muc h intensified connection between the figures and their textual description is necessary. In the aforementioned cases, the missing definition of symbols made a verification of mathematical correctness impossible.

Furthermore, model sizes are not included in Table 1 and Table 2. It is therefore hard for the reader to separate architecture and model design from model size. The textual discussion references model size on multiple occasions, but no number for the parameter count is mentioned for any model.

The dataset that Table 1 reports on is only implied to be the DNS Challenge test set in Section 3.2. The table's caption should state the dataset clearly.

In Appendix A.1, you state "As shown in Table 6, our model achieves better results than common representation." but Table 6 clearly shows most models beating your model in all metrics. There must be a mistake in either the table or your argument.

The appendix is not referenced anywhere in the text. If space permits an introductary sentence in the main text helps to contextualize the appendix.

### Questions
The discussion of Table 2 is not sufficiently discussing the two test sets independently. It is only stated that UNISE achieves best WER for linguistic reconstruction. However it slightly falls behind HiFi-GAN-2 on the EARS test set. On EARS---with DNSMOS only slightly higher than HiFi-GAN-2 and SpkSim slightly below HiFi-GAN-2---the benefit of UNISE over HiFi-GAN-2 is not self-evident. Could you clarify your phrasing in this respect?

Also, did your analysis show a reason for the less pronounced benefit on EARS?

### Soundness
2

### Presentation
2

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
This paper proposes a unified framework that combines noise-invariant self-supervised representation learning with generative speech enhancement to address limitations of existing methods. Experimental results demonstrate empirical performance outperforming baselines on WER for content preservation while maintaining competitive perceptual quality and speaker similarity.

### Strengths
- The paper identifies two unresolved challenges in SE using SSL representations, noise robustness and generative capability, and proposes a unified solution. This addresses a critical gap between discriminative SSL and generative SE.

- The noise-invariant contrastive learning explicitly disentangles speech content from noise, a improvement over prior noise-augmented SSL that only mitigates noise rather than isolating content.

- The joint training of encoder and flow-matching decoder creates a feedback loop: the encoder learns content-preserving features guided by the decoder’s reconstruction loss, while the decoder uses these features to avoid hallucinations.

### Weaknesses
- The paper states Sylber provides "sparse syllabic embeddings" but does not clarify: (i) how Sylber features are extracted from clean speech, (ii) why syllabic pseudo-labels are more effective than phonetic or word-level labels for content preservation, (iii) how non-speech frames are "filled with null tokens", what defines a non-speech frame?

- The paper emphasizes UNISE’s strength in WER but does not address a critical question: Does the focus on WER come at the cost of perceptual quality in extreme noise? For example:In Table 1, UNISE’s DNSMOS score (3.334) is lower than FlowSE’s (3.601) under non-reverberant conditions. Is this a necessary tradeoff for better WER, or can the model be adjusted to improve both?

- UNISE’s encoder has lower mutual information with speaker labels than WavLM/HuBERT, but the model still achieves competitive speaker similarity (Table 1). The paper attributes this to "acoustic features in the decoder," but this is vague.

### Questions
- Noise-Invariant Contrastive Learning DetailsCodebook Design: The paper mentions a learnable codebook with V=2048 codewords but provides no details on: (i) how codewords are initialized (e.g., random vs. pre-clustered on clean speech), (ii) how optimal transport "smooths target distributions", what OT cost function is used? (iii) why V=2048 was chosen (ablation over codebook size would strengthen this choice).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a unified generative speech enhancement method that aims to improve both acoustic quality and linguistic content preservation. The authors argue that prior approaches mainly emphasize perceptual quality while overlooking semantic consistency. To address this, they introduce an encoder trained with a contrastive loss to learn noise-invariant representations, coupled with a generative decoder that reconstructs enhanced speech from this bottleneck. Experimental results on several benchmarks show moderate improvements in content-preservation metrics. However, the evaluation is limited to speech enhancement and speech recognition tasks, which do not sufficiently demonstrate the generality of the proposed method. Moreover, the ASR performance on the CHiME-4 dataset remains notably below the mainstream average level.

### Strengths
The paper addresses an important and somewhat under-explored aspect of speech enhancement, content/semantic preservation rather than just clean sounding output. The framework is presented as unified representation learning + generation rather than simply plugging in a new decoder. The provided experimental results demonstrated the effectiveness of the proposed method.

### Weaknesses
1. The novelty appears limited. While combining contrastive representation learning with generative speech enhancement is interesting, the method largely builds upon existing methods without introducing a clear architectural or theoretical innovation.
2. The description of the noise-invariant encoder and the contrastive training setup is insufficiently detailed. It is unclear how noise invariance is explicitly enforced, how positive and negative pairs are constructed, and how this representation interacts with the generative decoder during training and inference.
3. The experimental validation needs improvement. It is unclear whether the reported improvements are statistically significant or robust across various corruption types and severities. The evaluation covers a narrow set of datasets and noise conditions, which raises concerns about the model’s generalization to unseen corruptions such as reverberation, clipping, or codec artifacts. The claim of enhanced content preservation is not well supported, as the ASR performance on the EARS test sets is even lower than that of the unprocessed noisy speech.

### Questions
1. Could the authors clarify the exact formulation of the contrastive loss? Specifically, how are the positive and negative pairs defined, e.g., clean vs. noisy, same vs. different speakers, or different noise types?
2. What types and severities of corruptions are considered in the evaluation? 
3. How can “content preservation” be quantitatively measured? Are intelligibility metrics such as word error rate or human listening tests used to verify that linguistic content is better preserved?
4. How can the authors demonstrate that the observed gains originate from the proposed noise-invariant representation, rather than from increased model capacity or a stronger decoder architecture?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes UNISE (Unified Noise-Invariant learning for Speech Enhancement), which jointly trains (1) a contrastive, noise-invariant representation encoder (inspired by SwAV / swapped prediction with Sylber pseudo-labels) and (2) a Flow-Matching based generative decoder conditioned on the encoder’s latent to improve speech enhancement while better preserving linguistic/semantic content. Experiments are reported on multiple benchmarks (DNS-Challenge, VoiceBank-DEMAND, EARS, CHiME-4) comparing regression, diffusion, language-model-based, and other generative approaches. The authors show notable WER improvements and competitive perceptual metrics.

### Strengths
1. Clear motivation addressing semantic loss/hallucination in generative SE.
2. A joint training framework (encoder + conditional flow decoder) with supporting ablation studies. 
3. Empirical improvements in WER while maintaining competitive perceptual metrics, suggesting a good trade-off between intelligibility and quality.

### Weaknesses
1. Reproducibility details missing: many hyperparameters and exact procedures (codebook training, Sinkhorn params, pseudo-label pipeline, training compute/time, seeds) are not fully specified.
2. Subjective evaluation lacks clarity: If human listening tests were conducted, their protocol is not sufficiently described; if not, the absence of blind subjective tests is a limitation. 
3. Failure cases: Some benchmarks (e.g., CHiME-4) show that UNISE is not always best; the manuscript’s discussion of limitations and failure modes is brief. 
4. Ambiguous effect of codebook / pseudo-label design choices. Key design parameters (codebook size, k-means clusters, pseudo-label source data) may strongly influence representation quality; the paper lacks comprehensive ablation experiments to validate its robustness.
5. No clear error analysis for cases where WER degrades or hallucinations occur.

### Questions
1. For the clustering/codebook and Sinkhorn steps: what were the exact hyperparameters (ε, number of iterations, temperature τ)? How was the collapse prevented? How sensitive is the model to the temperature or codebook size in contrastive learning?
2. Does joint optimization ever cause overfitting to certain noise types or degrade generalization to unseen environments?
3. Please detail the Sylber pseudo-label generation: which datasets were used to train the k-means, initialization method, how frame-level alignment was handled, and any preprocessing used. Would alternative pseudo-labels change results? 
4. Why do you freeze the encoder during finetuning? What happens if the encoder is unfrozen for end-to-end finetuning—does performance improve or degrade? Please report experiments. 
5. Please include more failure-case examples and a deeper analysis of when and why semantic hallucinations or distortions occur, and possible mitigation strategies. 
5. Have the authors explored scaling (e.g., larger encoder or DiT-like decoder) and its effect on robustness?

### Soundness
2

### Presentation
3

### Contribution
2
