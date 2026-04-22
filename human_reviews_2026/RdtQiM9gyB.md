# LEGATO: Large-scale End-to-end Generalizable Approach to Typeset OMR

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
We propose Legato, a new end-to-end  model for optical music recognition (OMR), a task of converting music score images to machine-readable documents. Legato is the first large-scale pretrained OMR model capable of recognizing full-page or multi-page typeset music scores and the first to generate documents in ABC notation, a concise, human-readable format for symbolic music. Bringing together a pretrained vision encoder with an ABC decoder trained on a dataset of more than 214K images, our model exhibits the strong ability to generalize across various typeset scores. We conduct comprehensive experiments on a range of datasets and metrics and demonstrate that Legato outperforms the previous state of the art. On our most realistic dataset, we see a 68\% and 47.6\% absolute error reduction on the standard metrics TEDn and OMR-NED, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Legato, the first large-scale end-to-end optical music recognition (OMR) model capable of transcribing multi-page typeset scores into ABC notation. Legato combines a frozen pretrained vision encoder from Llama 3.2 with a trained transformer decoder, processing segmented images rather than requiring pre-split pages or systems. The authors construct PDMX-Synth, a dataset of 238,386 image-ABC pairs derived from the PDMX dataset. Evaluation on multiple held-out datasets shows that Legato substantially outperforms Sheet Music Transformer++ on the TEDn metric.

### Strengths
* Legato is the first model to handle multi-page typeset scores end-to-end without the need of pre-splitting scores into pages or systems.
* The evaluation results show substantial improvements across multiple datasets and metrics, particularly on realistic camera images.
* Publicly releasing implementation code and PDMX-Synth dataset with 238k (multi-page typeset score image, ABC) pairs is beneficial to the community.
* Multiple held-out datasets for the evaluation.

### Weaknesses
* Legato is essentially multimodal Llama with a smaller transforemer decoder for OMR task. It has very limited architectural novelty.
* Applying BPE tokenization needs more justification than just providing 4 examples in Figure 3. and writing the tokenizer 'captures some composite musical concepts.'  Since ABC-notation have limited number of symbols and combinations, there could be a more efficient domain-knowledge-based method of tokenization. To support BPE tokenization, authors need to provide more systematic vocabulary analysis and comparison with expert-defined tokenization.
* The comparion between Legato and SMT++ heavily relies on the format conversion, which cannot ensure intergrity of the output.
* Legato and SMT++ have drastically different number of parameters. Additionally Legato includes pre-trained vision encoder compared to SMT++'s vision encoder, which is trained from scratch.
* Since the main difference of Legato and SMT++ is the vision encoder part and SMT++'s implementation is publicly available, the authors could have applied an enlarged SMT++ model with a matching number of parameters in the decoder trained on the same PDMX dataset.
* The GPT-5 evaluation's prompting strategy could be explored more thoroughly.

### Questions
No further question

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- This paper proposes a large-scale end-to-end generalizable approach to typeset OMR. The authors created paired image–ABC data from the PDMX dataset. The training system adopts a vision encoder and language decoder architecture. The proposed system is capable of recognizing full-page and multi-page typeset music scores into ABC notation. Although there is a lack of research on comparing different vision encoders and extending the LLM decoder to larger, the overall framework shows its effectiveness. The results are promising. I like this work.

### Strengths
- OMR is an interesting and important research topic in music information retrieval with strong ongoing research interest.

- The proposed vision encoder and Transformer decoder form a well-designed architecture for OMR.

- The system is trained on a large-scale dataset of 214k images, demonstrating strong generalization ability.

- The procedure for creating the dataset is well-designed and systematic.

- Comprehensive experiments are conducted across a range of datasets, showing state-of-the-art performance.

- The training and testing datasets are separate, further validating the system’s generalization capability.

- The recognition results shown in the Appendix are impressive and of high quality.

### Weaknesses
- The proposed PDMX-Synth dataset is derived from the PDMX dataset; therefore, it is effectively a subset and smaller in scale. Will this affect the performance?

- The model architecture shares similarities with the design of SMT++, indicating limited novelty in architectural design.

- There is limited comparison of different vision encoders and more decoder sizes, leading to the technical part a bit weak. 

- There is a lack of equations, such as autoregressive prediction, and loss functions to show how the LLM is trained. 

- The paper lacks discussion on human manuscript transcription — how would the system’s performance degrade in that case?

### Questions
- How would different vision encoders affect the recognition performance?

- As Table 1 shows, Legato outperforms Legato_small by a large margin. Would increasing the decoder size further improve system performance?

- Will hyperparameters, such as temperature affect the prediction performance? Do authors choose the best or are there randomness in next token prediction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces Legato, an end-to-end OMR system that converts full-page/multi-page typeset score images into ABC notation using a frozen pretrained vision encoder (Llama-3.2-Vision) and a lightweight transformer decoder with a multimodal projector. Training uses a new 214K-image synthetic corpus (PDMX-Synth). Evaluation is standardized via MusicXML conversion and measured with TEDn and OMR-NED, where Legato reports large absolute error reductions (e.g., −68% TEDn, −47.6% OMR-NED) over prior work (e.g., SMT++). The paper also presents a data-driven tokenization that appears to learn composite musical concepts (e.g., triads, short phrases).

### Strengths
Clear problem focus & scope. Tackles practical, full-page OMR with multi-system/stave layouts—beyond monophonic or single-system settings.

Strong empirical gains. Large improvements on TEDn/OMR-NED across several datasets, including an IMSLP sample.

Standardized evaluation. Using MusicXML as a unifying target for comparison reduces format bias; reporting ABC/kern helps triangulate.

Scalable training recipe. Pretrained vision encoder + compact decoder yields a seemingly efficient path to high accuracy.

Tokenizer insight. Data-driven tokenization that captures chords/phrases is promising and could simplify decoding of frequent musical patterns.

### Weaknesses
Attribution of gains is unclear. It’s hard to disentangle where improvements come from: ABC target format, frozen VLM encoder, data scale (214K), or architecture. A dedicated ablation is missing.

Evaluation confounds. Conversions among ABC/ kern/MusicXML may introduce asymmetric errors; conversion failure rates and their impact on metrics are not reported.

Metric interpretability. TEDn/OMR-NED reductions are compelling but lack qualitative error analysis or audible case studies linking metric changes to musically meaningful differences.

Generalization gaps. Focus is on typeset scores; robustness to handwritten, degraded scans, complex lyrics/ornaments, and non-Western notation is not demonstrated.

Robustness & reliability. No tests on scan noise, resolution changes, staff skew, lighting artifacts, transposition/key/time-signature shifts, or symbol vocabulary tails.

### Questions
Why better than SMT++?
Please quantify the contribution of (a) ABC output vs. MusicXML/kern, (b) data scale (learning curves), (c) using a pretrained vision encoder (frozen vs. finetuned), and (d) decoder capacity. A controlled ablation matrix would clarify attribution.

Case studies & perceptual relevance.
Provide score-level examples where Legato reduces TEDn/OMR-NED, with before/after renderings and (ideally) audio renderings to illustrate musically meaningful fixes (e.g., rhythm beaming, voice assignment, chord spelling).

Handwritten & degraded inputs.
How does Legato perform on handwritten scores (e.g., MUSCIMA++, historical manuscripts) or noisy scans? Any domain-adaptation strategy planned?
Please include perturbation tests (resolution, blur, skew, lighting, JPEG artifacts) and transposition/key/time-signature shifts to probe symbol and structure invariance.

Data hygiene & licensing.
Clarify licensing of PDMX-Synth sources and safeguards against train-test leakage across editions/engraving variants of the same piece.

Extending beyond rhythm/pitch.
Do you plan to evaluate semantic elements (lyrics alignment, articulations, ornaments, dynamics) and non-Western notation to support broader claims of generalizability?

### Soundness
3

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
This paper presents a comprehensive solution for OMR on long and complex real-world music sheets, including the collection and preprocessing of a large-scale dataset, as well as an end-to-end recognition model, LEGATO. Experimental results demonstrate that LEGATO exhibits superior performance and generalization capabilities, outperforming both prior specialized OMR model SMT++ and general-purpose vision-language model GPT-5.

### Strengths
- The PDMX-Synth dataset curated in this paper addresses the scarcity of real-world, complex music score data for OMR. Its public release would significantly benefit research in this field.
- Experimental results demonstrate that LEGATO significantly outperforms both SMT++ and GPT-5, with strong performance on real-world scenarios (camera versions of OpenScore String Quartets and OpenScore Lieder)—highlighting the effectiveness and practicality of the proposed approach.

### Weaknesses
1. Limited in performance analysis of general VLMs. 
    1. The paper relies solely on GPT-5 as the representative general-purpose vision-language model for comparison, which somewhat limits the persuasiveness of the evaluation. Including additional state-of-the-art multimodal models—such as those from the Gemini, Claude, and Qwen families—would provide a more comprehensive and robust assessment.
    2. The evaluation of GPT-5 lacks sufficient detail. The paper does not specify the prompts provided to the model, nor does it indicate whether exemplars were used in the input to optimize the model's output. Furthermore, the absence of a comparative case study between the outputs of GPT-5 and LAGETO prevents a thorough analysis of GPT-5's deficiencies. This raises concerns about whether the authors have adequately explored the capabilities of the general VLMs.
2. Lack of experimental analysis regarding the training data. The large-scale dataset introduced in this paper constitutes one of its core contributions, and considerable space is devoted to detailing the dataset’s collection and preprocessing pipeline. However, the paper does not include experimental comparisons evaluating the impact of the proposed dataset versus existing datasets on model performance (although the new dataset is strongly likely to yield better results). Moreover, the authors do not provide empirical ablation studies to assess the effectiveness of their data cleaning and data augmentation strategies.

### Questions
1. When calculating the TEDn metric, did the authors attempt to have GPT-5 directly output MusicXML? If so, were there any changes in the results?
2. Have the authors attempted more data augmentation strategies such as adjusting brightness, applying affine transformations, or adding noise—on the rendered score images in the training data to better align them with real-world scenarios?

### Soundness
3

### Presentation
3

### Contribution
3
