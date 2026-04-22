# SenSE: Semantic-Aware High-Fidelity Universal Speech Enhancement

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Generative universal speech enhancement (USE) methods aim to leverage generative models to improve speech quality under various types of distortions. Diffusion- or flow-based generative models are capable of producing enhanced speech with high quality and fidelity. However, they typically achieve speech enhancement by learning an acoustic feature mapping from degraded speech to clean speech, while lacking awareness of high-level semantic information. This deficiency tends to cause semantic ambiguity and acoustic discontinuities in the enhanced speech. In contrast, humans can often comprehend heavily corrupted speech by relying on semantic priors, suggesting that semantics play a crucial role in speech enhancement. Therefore, in this paper, we propose SenSE, which leverages a language model to capture the semantic information of distorted speech and effectively integrates it into a flow-matching-based speech enhancement framework. Specifically, we introduce a semantic-aware speech language model to capture the semantics of degraded speech and generate semantic tokens. We then design a semantic guidance mechanism that incorporates semantic information into the flow-matching-based speech enhancement process, effectively mitigating semantic ambiguity. In addition, we propose a prompt guidance mechanism, which leverages a short reference utterance to alleviate the loss of speaker similarity under severe distortion conditions. The results of several benchmark data sets demonstrate that SenSE not only ensures high perceptual quality but also substantially improves speech fidelity while maintaining strong robustness under severe distortions. Codes and demos are available$\footnote{\url{https://anonymous.4open.science/r/SenSE_ICLR-B262/}}$$\footnote{\url{https://anonymous.4open.science/w/SenSE-demo-25F9/}}$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes SenSE, a semantics-aware universal speech enhancement framework. The framework comprises two major components: (1) a speech language model that extracts high-level semantic representations from noisy speech; (2) a semantic-guided speech enhancement module (based on flow matching), which uses semantic tokens and noisy speech to recover (generate) clean speech.

### Strengths
- **Writing.** The paper is well-written and clearly structured, making it easy to follow and pleasant to read.
- **Task Significance.** The study focuses on universal speech enhancement, which is an important and meaningful research direction. Developing a generalizable enhancement framework that can handle diverse acoustic conditions is of high practical value and scientific interest to the speech processing community.
- **Performance.** The experimental results are comprehensive and well-presented across multiple datasets. The proposed SenSE framework consistently outperforms all baselines, especially achieving notable improvements on the dWER metric, demonstrating its strong semantic preservation.

### Weaknesses
- **Inconsistency.** There is a writing inconsistency between the Abstract (and Title) and the Introduction. The Abstract suggests that this work appears to be the first to incorporate semantic information into speech enhancement. However, in the Introduction, the authors acknowledge that previous work, such as GenSE, has already leveraged semantic information. The authors are encouraged to revise the Abstract to ensure a more accurate and rigorous presentation of the paper’s contribution.
- **Novelty.** The authors did not explicitly state their main contributions, so I summarized them as follows (please correct me if I am mistaken): this work proposes a new framework, SenSE, for universal speech enhancement, incorporating three strategies: (1) leveraging semantic information; (2) introducing a semantic guidance mechanism; and (3) proposing a prompt guidance mechanism.
However, I find the novelty of these points to be limited for the following reasons: (1) Regarding the use of semantic information, compared with GenSE, SenSE only employs a different implementation rather than introducing a fundamentally new idea. (2) For the semantic guidance mechanism, the authors adopt a random-mask-based condition enhancement strategy to ensure that semantic information is effectively utilized. However, a similar idea has already been proposed in DOSE, which is a strong baseline used in GenSE. I have a slight concern that the authors may be aware of DOSE but intentionally omitted it to emphasize novelty. I recommend including this related work to ensure completeness and uphold the integrity of academic writing. (3) The proposed prompt guidance mechanism provides additional contextual information to improve speech enhancement performance. However, similar strategies (e.g., encoding speaker or environmental information) are well-established in conventional speech enhancement research.
- **Efficiency.** I anticipate that the authors may argue their novelty lies in proposing a new model that achieves strong empirical performance. However, if this is the case, the contribution appears to be more engineering-oriented, essentially combining existing techniques in a practical way. I do not mean to undervalue such work—indeed, developing a well-performing model can often be more valuable than a purely theoretical contribution with limited real-world impact.
Nevertheless, for works of this nature, it is generally important to evaluate not only the performance metrics but also efficiency. The proposed framework seems to be computationally expensive, yet the paper lacks analysis and discussion of its parameter complexity and runtime efficiency compared with baselines. If the reported performance gains are achieved at the cost of substantially higher computational overhead, the practical value of the method—especially for industrial applications—would be limited.
As analyzed in the previous Novelty section, the paper’s academic contribution appears relatively limited compared to GenSE, making efficiency evaluation even more crucial to justify the overall significance of the work.

### Questions
- **Reliability of Semantic Representations.  (Lines 79–85)** I still speculate whether the proposed speech language model can produce reliable semantic representations under heavy noise or severe distortions. Have the authors conducted any experiments and theoretical analyses to verify this robustness?

- **Discriminative SE Methods. (Lines 49–52)** Are discriminative speech enhancement (SE) methods truly as ineffective as stated in the Introduction? From the reported results (Tables 1 and 2), discriminative models actually perform reasonably well and do not appear to be "bad" as the authors suggested. Moreover, since current diffusion- or flow-based SE models require significantly more training and inference time, it is worth questioning whether generative paradigm indeed represent a more promising future direction for the SE community. PS: I also find that Figure 5 lacks meaningful value. The authors present some qualitative samples, but given that the quantitative performance gap between methods is quite small, it is hard to believe that the selected examples are representative—more likely, they were selectively chosen to highlight differences.

- **Clarification of Table 3.** In Table 3, I would like to confirm whether the "short reference utterance" is from the same target speaker but a different utterance from the clean test sample, or if it is indeed the clean version of the test utterance itself.

It is possible that I have overlooked some aspects of the paper. I would appreciate it if the authors could clarify them during the rebuttal, and I remain open to updating my score based on their response.

### Soundness
2

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
This paper introduces SenSE, a semantic-aware universal speech enhancement framework. It consists of a Semantic-Aware Speech Language Model (SASLM) that generates purified semantic tokens from degraded speech, and a flow-matching-based enhancement module conditioned on both degraded acoustic features and semantic tokens. An additional prompt guidance mechanism allows preserving speaker similarity under severe distortions. Experiments on multiple DNS Challenge test sets and VCTK-GSR show that SenSE achieves strong semantic fidelity (SpeechBERTScore, dWER) and competitive perceptual quality compared to recent generative baselines.

### Strengths
1. The idea of explicitly incorporating semantic tokens into flow-based SE is conceptually appealing and well-motivated.
2. The system is carefully designed, combining semantic guidance and prompt guidance, and evaluated across diverse conditions.
3. The approach shows clear advantages on semantic-related metrics, suggesting potential benefits for ASR robustness and speech interfaces.

### Weaknesses
1. Improvements are uneven: SenSE leads on semantic metrics but only matches or slightly lags behind strong baselines (e.g., AnyEnhance, TF-GridNet) on perceptual quality and speaker similarity. The overall gains may not justify the added system complexity.
2. The pipeline introduces significant additional components (Whisper-based SASLM, ConvNeXt alignment, DiT flow model), but efficiency and runtime feasibility are not reported.
3. Prior works such as SELM, GenSE, and LLaSE-G1 have already integrated semantic/ASR-aware representations into SE. The paper should better clarify how its semantic guidance differs and why it is more effective.

### Questions
1. What is the runtime overhead of SASLM + flow matching compared to AnyEnhance or FlowSE? Could SenSE be applied in real-time or low-resource scenarios?
2. How does SenSE compare to simpler joint SE–ASR multi-task training or feature consistency loss methods?
3. Are the reported improvements statistically significant (e.g., across multiple random seeds)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
•	Summary: In this paper, the authors propose SenSE, which integrates semantic tokens with LM-predicted speech embeddings and effectively incorporates them into the flow-matching-based speech enhancement framework. In the first stage, a semantic-aware speech language model is adopted to estimate the clean speech tokens in an auto-regressive manner, with the tokens estimated by an audio encoder (Whisper-large-v3) as input. A speech tokenizer (S3) is further adopted to extract the semantic tokens. These two domains' tokens, together with the degraded and reference Mel-spectrogram, are adopted to deliver flow-matching-based speech enhancement in the second stage. A vocoder is finally adopted to estimate the final speech enhancement.
•	Contributions: This paper proposes a novel combination of features (semantic tokens) and continuous features (speech embeddings and Mel-spectrogram). A two-stage architecture is proposed to deliver semantic-guided speech enhancement with flow matching. The semantic tokens enable the model to reconstruct high-fidelity enhanced speech. A masking strategy is applied to the Mel-spectrogram in the second stage's input, motivating the model to learn the alignment of semantic tokens between clean and degraded speech. The proposed model achieves outstanding performance over popular discriminative/generative models on the Interspeech 2020 DNS Challenge dataset.

### Strengths
•	This paper is moderately original, with a proposed two-stage speech enhancement approach combining a language model and flow matching. The semantic tokens help to preserve key semantic information while improving alignment with text. The adoption of the language model also enables the model to connect with other language-related tasks, showing strong potential.
•	The paper is moderately clear and well-written, helping readers at least to easily follow the core ideas. Minor mistakes or mismatches exist, however, which will be illustrated in the following paragraphs.
•	The paper can be a good example of cross-domain generative speech enhancement, showing its considerable significance.

### Weaknesses
•	The algorithm, model architecture, and hyperparameters are not clearly explained, including but not limited to the setups of the semantic-aware speech language model in Fig. 1, the DiT Blocks in Fig. 2, the flow matching and its ODE solver in Fig. 2, the final modulation in Fig. 2, the Conv2NeXT v2 blocks in Fig. 2, etc, which will hinder readers' understanding and reproducibility of the model. Note: By publishing code, the reproducility issue might be relaxed, but the expectation must be that also the figures and description allow a reproduction of the model to a large extent. This is not given in the current form. 
•	The enhancement performance of the semantic-aware speech language model's output (the first stage's output) is not analyzed. Thus, the function and performance of the semantic-aware speech language model is not clear. Furthermore, it is unclear whether the estimated tokens from the first stage or the Mel-spectrograms contribute more to the second stage. The functionality and performance of the first and second stages need to be decoupled.
•	The reviewer is doubtful about the function of the speech tokenizer. If the reference speech is the clean speech from the same speaker as the noisy speech, then the tokens estimated should be regarded more as the speaker information, instead of the semantic information. If the reference speech is the noisy speech itself, then it can be regarded as the semantic information. According to line 96, the former is what the authors do. Thus, the reviewer would believe it is speaker information, instead of semantic information, that is estimated as the input to the semantic-aware speech language model. Then, a further study on using a speaker embedding extractor to substitute the speech tokenizer should be delivered.
•	Some mistakes in equations and figures, which will be illustrated in the following paragraphs.

### Questions
•	In eq. (1), index $i$ does not appear when representing the index of semantic tokens and speech embeddings. Substituting $i$ with $k$ should be correct. 
•	In Fig. 1(a), to the reviewer's understanding, the word "embedding" should represent a continuous representation, while the word "token" should represent a discrete representation. However, both speech embedding (pink cube) and clean token (yellow cube) are treated as the input to the language model, which should only take continuous representation as input. The same problem appears in Fig. 2(a). I would suggest adding an "embedding layer" block between the S3 tokenizer and language model.
•	In Fig. 1(b) and Fig. 2(b), an adder is used between the Mel-spectrogram and Conv2NeXT v2 blocks output. But in line 273, the word "concatenating" is used, leading to a mismatch. Please clarify.
•	In Fig. 1(b), the degraded Mel's original sequence length is $L_1$, while the ref Mel's original sequence length is $L_2$, as stated in lines 296 and 297. Thus, the padding length of degraded Mel and ref Mel should be $L_2$ and $L_1$, respectively. However, in lines 298 and 299, the situation is the opposite. Please clarify.
•	In Fig. 2(b), the masked clean speech and training target are all represented as $m_1x_1$, but to the reviewer's understanding, these two shouldn't be the same. The spectrum in Fig. 1(b) also shows their difference. In eq. 2, the training target is $(1-m_1)(x_1-x_0)$, which is another mismatch. Please clarify.
•	In line 271, semantic tokens are padded with filler tokens to match the length of the Mel spectrogram. Is interpolation a better choice? According to previous publications, interpolation is commonly adopted to deal with this kind of sampling rate or frame/hop length mismatch (AnyEnhance: A Unified Generative Model with Prompt-Guidance and Self-Critic for Voice Enhancement, Zhang et.al, TASLP 2025).
•	In line 286, the masking ratio of random temporal masking is not clear. Please clarify.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a semantic-aware generative speech enhancement method. The main idea is to integrate high-level semantic information, e.g., semantic tokens from an S³ tokenizer, into a flow-matching based TTS system to construct a generative speech enhancement model. Experimental results on several benchmark datasets show improved perceptual quality, speech fidelity, and robustness compared to existing methods. However, as a universal speech enhancement method, the evaluation should cover a wider range of corruptions, e.g., reverberation, clipping, bandwidth limitation, codec artifacts, and packet loss, to more comprehensively validate the generalization capability.

### Strengths
1. The paper tackles an interesting and underexplored problem of incorporating semantic priors into generative speech enhancement.
2. The conceptual motivation is strong and intuitively appealing, as human perception indeed leverages semantic context to interpret noisy speech.
3. The proposed integration of a language model into a flow-matching framework is novel and timely, which aligns with current trends in multimodal and semantically grounded generative models.

### Weaknesses
1. The technical details of the semantic-aware speech language model and the semantic guidance mechanism are insufficiently described. It is unclear how the semantic tokens are aligned or conditioned within the flow process, and there is no clear illustration of how semantic information is utilized for the speech enhancement task.
2. The novelty appears limited in implementation, as it is mainly a combination of existing components, language model + flow matching, without a clear theoretical insight or architectural innovation.
3. The experimental validation lacks depth. It is unclear whether the reported improvement are statistically significant or primarily due to additional conditioning information. Moreover, the performance gain over compared methods appears limited.
4. The integration of the language model seems heuristic and task-specific, raising concerns about the generalizability and scalability of the approach to other types of corruptions.

### Questions
1. Which specific language model architecture serves as the backbone for the proposed semantic-aware speech language model?
2. Could the authors clarify what the DNS Challenge GSR and VCTK GSR test sets refer to, and how they differ from the standard DNS and VCTK evaluation test sets?
3. How do the authors verify that the proposed method truly leverages semantic information, rather than simply benefiting from additional conditioning signals or larger model capacity?

### Soundness
3

### Presentation
2

### Contribution
2
