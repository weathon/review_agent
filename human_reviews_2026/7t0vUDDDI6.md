# StreamUni: Achieving Streaming Speech Translation with a Unified Large Speech-Language Model

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
Streaming speech translation (StreamST) requires determining appropriate timing, known as policy, to generate translations while continuously receiving source speech inputs, balancing low latency with high translation quality. However, existing StreamST methods typically operate on sentence-level speech segments, referred to as simultaneous speech translation (SimulST). In practice, they require collaboration with upstream segmentation models to accomplish StreamST, where the truncated speech segments constrain SimulST models to make policy decisions and generate translations based on \textcolor{blue}{pre-defined contextual information preset by the upstream models}. Moreover, SimulST models struggle to learn effective policies due to the complexity of speech inputs and cross-lingual generation. To address these challenges, we propose StreamUni, which achieves StreamST through a unified Large Speech-Language Model (LSLM). Specifically, StreamUni incorporates speech Chain-of-Thought (CoT) in guiding the LSLM to generate multi-stage outputs. Leveraging these multi-stage outputs, StreamUni simultaneously accomplishes speech segmentation, policy decision, and translation generation, completing StreamST without requiring massive policy-specific training. Additionally, we propose a streaming CoT training method that enhances low-latency policy decisions and generation capabilities using limited CoT data. Experiments demonstrate that our approach achieves state-of-the-art performance on both SimulST and StreamST tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this work, the authors propose a streaming speech translation system based on large speech language model. The system includes a truncation policy to keep the input audio length in a reasonable number and suitable for streaming translation task last for hours. The ASR transcription 
is also included to the language model output, i.e. COT, to boost the performance. The final results are very competitive. However, my main concern is the novelty of this work. Combining both speech recognition and translation results for streaming application has been proposed before [1]; The generation policy is based on wait-k, which has already shown been suboptimal in many streaming translation papers, such as [2]; Also, the comparison in the experiment is unfair and it is hard to figure out the main factor contributing to the good performance. 

[1] Weller, Orion et al. “Streaming Models for Joint Speech Recognition and Translation.” EACL (2021).
[2] Papi, Sara et al. “Attention as a Guide for Simultaneous Speech Translation.” ACL (2023).

### Strengths
- A good streaming system to leverage SOTA LLM for streaming speech translation.
- A truncation policy to limit the input audio length.

### Weaknesses
- COT proposed has been studied before as discussed in the summary section
- wait-k based generation policy is suboptimal.
- the comparison in the experiment is unfair and it is hard to figure out the main factor contributing to the good performance.  See Question section for more details. 
- Forcealignment is required to train the model
- Truncated speech input/text output, especially text output,  leads to context information loss. It actually degrade the documentation level translation to utterance level based translation.

### Questions
- Not sure the main difference between SimulST and StreamST. Methods like Transducer, CAAT and TAED can be used for streaming transcription or translation for audio last for hours too, though the results reported in the literature are utterance based. 
- How to add incremental speech chunk inputs during inference? Do we re-evaluated everything, from truncation policy, ASR outputs, to translation results?
- How do you determine the end of transcription text token generation give $x_n$? Will the LLM generate end of sentence token for the partial sentence. 
- The comparison in the experiment is unfair and we don't know which factor contributing most to the good performance. LLM, COT or something else?
- What's the offline translation results from Phi-4-Multimodal? Does it leverage ASR results?
- What's the WER for the transcription generated?
- Why Phi-4-Multimodal - VAD is much worse than the proposed system? The proposed system will detect utterance boundary (Truncation time) by comparing decoding results from 3 consecutive chunks. Assume each chunk is 320ms, the silence segment should be around 640ms to 960ms. It should not be hard to for the VAD system to detect that.
- Which dataset is used in Table 2?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces StreamUni, a method for streaming speech translation that leverages a large speech language model to achieve various steps (segmentation, policy decision, translation).
The method shows better latency-quality tradeoffs than the baseline and various ablations are presented.

### Strengths
* the proposed method is interesting
* the empirical results are strong
* the method is applicable to both streaming and simultaneous ST

### Weaknesses
* situate the work better, for example https://arxiv.org/abs/2502.03382 is not cited/compared to
* given that simulst is quite a practical application, add considerations on real-time deployment and clarify whether the evaluation takes computational latency into account

Due to these weaknesses, the current overall rating is a more conservative 6 but the reviewer is looking forward to authors' answers.

### Questions
* 066-067: “but also present difficulties in efficiently transferring to newly advanced LSLMs.” Can you elaborate on this particular point?

* 309-311 are you comparing to the competition winners of the annual IWSLT competition? This would ensure that the proposed method is SoTA as claimed in the paper.

* 344: “using Stream SacreBLEU”: what does Stream SacreBLEU refer to?

* Could you elaborate on the anticorrelation between comet and bleu in Table 1?

Typos/Presentation

* Suggest defining more formally the tasks of simulst and streamst. In the current version, the definitions are somewhat implicit (simulst operates on a sentence while streamst operates on longer input (but unclear how long, infinite? etc.))

259: “for a instance”: for an instance

The conclusion is very short (probably to fit the limit), consider expanding.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents StreamUni, a streaming translation model designed to handle unbounded speech input. The model operates within a single speech language model that jointly performs transcription, truncation, and streaming translation. Specifically, the system first transcribes incoming speech, then performs truncation based on detected silences and sentence boundaries, and finally generates translations in a streaming fashion using a wait-k like policy conditioned on partial speech, prior transcripts, and partial translations. Experiments on the MuST-C dataset demonstrate that StreamUni achieves a superior quality–latency trade-off compared to baseline systems when processing unbounded speech streams. Additional ablation studies highlight the effectiveness of the training data mixture and the impact of truncation strategy on performance.

### Strengths
1. **Unified Speech Language Model for Multiple Tasks:**
    
    StreamUni effectively fine-tunes a single speech language model to perform three key actions—transcription, truncation, and streaming generation—within a unified framework. This design demonstrates strong empirical performance despite being trained on relatively limited data.
    
2. **Reinforcement of Data Mixing Importance:**
    
    Although the benefit of mixing offline and streaming data during training for streaming translation has been discussed in prior work [1], this paper provides additional empirical evidence supporting its effectiveness, further validating this important training strategy.
    

[1] Fu, Biao, et al. "Efficient and Adaptive Simultaneous Speech Translation with Fully Unidirectional Architecture." *arXiv preprint arXiv:2504.11809* (2025).

### Weaknesses
1. **Flaws in Experimental Design:**
    - **Unfair baseline comparison:** The baselines (e.g., EDAtt for SimulST and StreamAtt for StreamST) rely on offline speech translation models trained with significantly weaker base architectures than Phi-4-Multimodal, which serves as the backbone of StreamUni. Since even their *offline translation quality* differs substantially, the comparison does not fairly isolate the advantage of the proposed method itself.
    - **Lack of cascade baseline:** The entire StreamUni pipeline—comprising transcription, truncation, and wait-k–style streaming generation—could be replicated with a cascaded ASR + MT architecture. Including such a comparison would provide a stronger and more interpretable baseline.
2. **Inaccuracies and Ambiguities in Writing:**
    - In the abstract, the authors claim that prior works translate using only limited contextual information, yet StreamUni also conditions translation on a restricted context window, potentially leading to term inconsistency across long-form speech.
    - **Lines 68–70:** The claim that prior methods rely on an upstream segmentation model is inaccurate, as [2] demonstrates direct streaming translation on unbounded speech using attention-sink techniques, without requiring segmentation.
    - **Equations (1) and (2):** These equations should use the product of probabilities, not their summation.
    - **Lines 175–179:** The explanation of the truncation process is ambiguous and only becomes clear after reading the following “Truncation Policy” subsection; this section should be reorganized for clarity.
    - **Figure 1:** The diagram provides little information into what the generation policy entails or how it operates.
    - **Figure 4:** The term “Partial Streaming” is undefined—its precise meaning should be clarified.
    - **Table 1:** The label “Human segmentation” is misleading, since MuST-C segmentations are automatically derived, not human-annotated.

[2] Siqi Ouyang, Xi Xu, and Lei Li. 2025. InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model. In *Findings of the Association for Computational Linguistics: ACL 2025*, pages 3032–3046, Vienna, Austria. Association for Computational Linguistics.

### Questions
Do the authors have any intuition or analysis on why the model’s truncation strategy outperforms MuST-C’s segmentation in terms of COMET scores?

### Soundness
2

### Presentation
2

### Contribution
2
