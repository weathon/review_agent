# MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
MambaVoiceCloning (MVC) asks whether the conditioning path of diffusion-based TTS can be made fully SSM-only at inference—removing all attention and explicit RNN-style recurrence layers across text, rhythm, and prosody—while preserving or improving quality under controlled conditions. MVC combines a gated bidirectional Mamba text encoder, a Temporal Bi-Mamba supervised by a lightweight alignment teacher discarded after training, and an Expressive Mamba with AdaLN modulation, yielding linear-time $\mathcal{O}(T)$ conditioning with bounded activation memory and practical finite look-ahead streaming. Unlike prior Mamba--TTS systems that remain hybrid at inference, MVC removes attention-based duration and style modules under a fixed StyleTTS2 mel--diffusion--vocoder backbone. Trained on LJSpeech/LibriTTS and evaluated on VCTK, CSS10 (ES/DE/FR), and long-form Gutenberg passages, MVC achieves modest but statistically reliable gains over StyleTTS2, VITS, and Mamba--attention hybrids in MOS/CMOS, F$_0$ RMSE, MCD, and WER, while reducing encoder parameters to 21M and improving throughput by $1.6\times$. Diffusion remains the dominant latency source, but SSM-only conditioning improves memory footprint, stability, and deployability. 
Code available at: \url{https://github.com/sahilkumar15/MVC}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents MambaVoiceCloning (MVC), a text-to-speech framework that removes all attention and recurrent components from the encoder and conditioning path at inference, relying solely on SSMs. The proposed system includes three Mamba structured modules. 
The key claimed is that MVC is the first fully SSM-only conditioning stack for TTS, distinguishing it from prior Mamba–attention hybrids that retain recurrence or attention in duration/prosody modeling. 
The MVC is trained based on LibriTTS, LJSpeech while tested based on VCTK and CSS10 dataset. The results in this paper show consistent improvements over StyleTTS2 and Mamba–attention baselines in terms of MOS, CMOS, F0 RMSE, MCD, and WER, while achieving a smaller parameter count and higher encoder throughput. 
Overall, the paper offers a technically report by demonstrating the practical benefits of fully SSM-based conditioning in TTS. The writing of this article is obscure and not well-structured, failing to effectively demonstrate its innovativeness.

### Strengths
This article applies the mamba block to the innovation of the TTS model structure, replacing the transformers based structure. By leveraging the advantages of the mamba block, it designs and implements the transfer of voice characteristics and style effects in speech synthesis.

### Weaknesses
1. The writing logic of the article needs to improve. Starting from the introduction section, there is no clear explanation of the specific reasons for using the Mamba block to replace the transformer structure.
2. Regarding the calculation symbols and abbreviations in the method module, there is no unified explanation. They are scattered throughout the article and seem disorganized.
3. In the experimental module, the baseline for comparison does not provide a detailed explanation of their training configuration. Especially for the VITS and JETS comparison schemes, no detailed description is given of the adopted structure and training situation.

### Questions
1. Due to the improvement in the synthesis effect of long sentence input brought by the use of mamba block, have tests been conducted on longer inputs? The current definition seen is 10 seconds, but 10 seconds is not considered a long sentence for the audiobook scenario mentioned in the paper. Nowadays, there are many zero-shot large language model based speech synthesis solutions that can support longer text input for synthesis, such as cosyvoice series . What are the advantages of the Mamba-based model solution?
2. In Section 4.1 "DATASETS AND PREPROCESSING", “LJSpeech Ito & Johnson (2017) (13k, ∼24 h, 1 spk.), LibriTTS Zen et al. (2019) (∼245 h, 1,151 spk.), VCTK Veaux et al. (2017) (∼44 h, 109 spk.; zero-shot), and CSS10(ES/DE/FR) Park & Mulc (2019) (1 spk./lang.).” What do the contents in the brackets represent? And why aren't they unified?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The author proposed a Text to speech(TTS) model, MembaVoiceCloning(MVC). MVC is the first state-space model throughout the entire encode/conditioning stack at inference for TTS, by removing attention and RNN modules. The model is consists of 3 components. It applies noval Bidirectional Mamba encoder with AdaLN(adaptive gating and layer normalization). It also applies Expressive Mamba Encoder that uses gated fusion and AdaLN-conditioned modulation within an SSM to enable expressive control in linear time while disentangling style from phonetic content. 

The model has 21M parameters and was finetuned standalone and then joint-trained with diffusion decoder. Compared to baseline models, MVC shows the best MOS and Out-of-Distributed(OOD) robustness. Detailed ablation study shows the effectiveness of all three components.

Overall the paper is well structured, easy to follow and shows good vision, novelty and results.

### Strengths
1. The model has good novelty on model architecture by replacing attention and RNN blocks by SSM-only blocks at inference time. The components are innovative on their ideas and implementations. For example, it introduces learnable gated fusion into previous bi-mamba encoders and it enhances prosodic disentanglement and long-form stability. Expressive mamba encoder also uses gated fusion and AdaLN-conditioned modulation.

2. Benefits from the SSM-only blocks, the model operates in linear time and bounded memory, which is optimized for efficiency. This is good for scalability. During inference, the proposed encoder model is not dominating the runtime. 

3. Experiments show that the model has SOTA performance on both subjective and objective results compared to baselines. Ablation study is comprehensive and detailed.

### Weaknesses
1. The LJSpeech and LibriTTS are standard TTS datasets in english, but it might be good to also extend the experiments on other languages -  this has been mentioned in the conclusion already

### Questions
The baselines are focused on transformer based SOTA. Are there any comparisons to other Mamba-based TTS models?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed to use MAMBA style SSM blocks in all conditioning encoders of a diffusion-based TTS model.

### Strengths
1. The paper replaces all encoders with MAMBA layers and show some improvements in latency, RTF and peak memory usage.
2. The paper open sourced the code for inspection, which helps with reproducibility.

### Weaknesses
1. The method is only compared against a few older TTS models like VITS and StyleTTS. There are many newer models (NaturalSpeech/CozyVoice/Higgs Audio etc) that are not compared against. While the authors didn't claim SOTA, I think the paper is weaker without comparison with SOTA methods.
2. The ablation study section is short and lacking details. By "removing expressive MAMBA" what does the author mean exactly? Is it replacing the expressive MAMBA encoder with some attention/RNN based encoder or remove the encoder all together? Also the authors said one of the contribution is the gated fusion for the bi-MAMBA layer, but it is not properly ablated in the experiments.

### Questions
see weakness.

### Soundness
2

### Presentation
2

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
MambaVoiceCloning proposes a TTS system whose entire deployed conditioning path (text, rhythm/duration, prosody) is state–space only at inference, paired with a diffusion decoder for quality, yielding linear-time encoding, bounded activation memory, long‑form stability, and feasible streaming with finite look‑ahead. It uses three SSM modules: (i) a gated bidirectional Mamba text encoder, (ii) a Temporal Bi‑Mamba trained with a lightweight aligner used only during training, and (iii) an Expressive Mamba with AdaLN conditioning.

The proposed method shows statistically reliable quality and efficiency gains over StyleTTS2 and Mamba-attention hybrids under a matched mel–diffusion–vocoder setup, while confirming the diffusion decoder remains the main latency bottleneck. 

All encoder modules run in O(T) with bounded activations and no global attention maps, reducing peak memory, improving encoder throughput, and enabling longer inputs/batches; the diffusion decoder still dominates latency.

### Strengths
1. The proposed method provides linear‑time scans, bounded activations, lower peak memory, and higher encoder throughput (≈1.6×).

2. The proposed method offers modest improvements over StyleTTS2 and Mamba‑attention hybrids in MOS/CMOS and objective metrics

### Weaknesses
1. Since the decoder model dominates the latency, the linear-time encoder does not translate into large overall RTF gains.

2. On LJSpeech and LibriTTS, differences in metrics as compared to the baselines are small; the paper frames MVC as a practical encoder-side refinement, not a paradigm shift.

3. The system lacks explicit emotion or fine-grained style controls beyond AdaLN conditioning, limiting targeted prosody manipulation compared to methods with richer style token or reference-attention mechanisms.

4. Claims of long-form stability are positive, but focus on MOS/WER and pitch drift; cross-chunk smoothing and boundary artifacts under streaming conditions are not deeply analyzed.

### Questions
1. How are Mamba selective-scan hyperparameters (state size, convolution kernel, gating temperature) chosen for each encoder, and how sensitive are results to these choices across datasets and lengths?

2. The Bi‑Mamba text fusion uses a learned gate plus AdaLN; is the gating stable on very long sequences or under domain shift, and are there failure cases where one direction collapses?

3. Why is fusion in the Temporal Bi‑Mamba kept linear (no gating) while gating is used elsewhere; are there quantitative trade-offs or counterexamples where temporal gating would help long-form speech?

4. The lightweight attention aligner is used only during training; what architecture and capacity does it have, and how robust are learned alignments if the aligner is partially mis-specified or noisy?

5. How long must the reference be for stable similarity in zero-shot cloning?

6. The paper claims finite look‑ahead streaming with Uni‑Mamba; what look‑ahead L is used in experiments, and how does WER/MOS vary with L for 2–6 minute inputs?

7. MVC is trained on English but tested on CSS10 ES/DE/FR with shared vocab and <lang> tags; what failure modes occur in cross-lingual phoneme inventories and stress patterns, and does alignment degrade on languages with different syllable timing?

### Soundness
3

### Presentation
2

### Contribution
3
