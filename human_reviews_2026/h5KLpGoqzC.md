# Hierarchical Semantic-Acoustic Modeling via Semi-Discrete Residual Representations for Expressive End-to-End Speech Synthesis

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4, 6

## Abstract
Generative models for speech synthesis face a fundamental trade-off: discrete tokens ensure stability but sacrifice expressivity, while continuous signals retain acoustic richness but suffer from error accumulation due to task entanglement. This challenge has driven the field towards multi-stage pipelines that rely on pre-trained discrete speech tokenizers, but these create a semantic-acoustic divide, limiting holistic and expressive speech generation.  We resolve these dilemma through hierarchical semantic-acoustic modeling with semi-discrete residual representations.Our framework introduces a differentiable quantization bottleneck that induces natural specialization: a Text-Semantic Language Model (TSLM) generates semantic-prosodic plans, while a Residual Acoustic Model (RALM) recovers fine-grained acoustic details.This hierarchical semantic-acoustic representation guides a local diffusion-based decoder to generate high-fidelity speech latents. 
Critically, the entire architecture is trained end-to-end under a simple diffusion objective, eliminating dependency on external discrete speech tokenizers. Trained on over 1 million hours of speech, our 0.5B-parameter model achieves state-of-the-art zero-shot TTS performance among open-source systems, demonstrating that our approach delivers expressive and stable synthesis. Audio samples are available at: https://voxcpm.github.io/VoxCPM-demopage/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a cascaded TTS model that predicts semantic tokens, refine with added residual latents and finally decoded into speech latents with a latent diffusion model.

### Strengths
1. While there are many similar cascaded architecture that predicts semantic tokens, refine with acoustic tokens/latents and finally decode into speech latents/waveform, there is some (but limited) novelty in using FSQ as a bottleneck layer directly after LLM output, rather than as in a pre-trained speech tokenizer. This does give some stability and flexibility.
2. There is ablation study that investigate the impacts of the design choices.

### Weaknesses
1. While the model does seem to perform better in term of WER, the speech naturalness performance seem to be a mixed bag. In Table 2, the model does not perform as well in DNSMOS metrics. Also in Table 3 the model underperforms/perform on-par on naturalness. In the evaluation section the authors tends to spent more time on the positive results but discuss little about the negative results. I also want to ask why some of the models evaluated in Table 2 are not presented in Table 3?
2. The paper doesn't evaluate the expressiveness nor the controllability of the proposed model. Some competitors (e.g. CozyVoice and Higgs Audio) are focusing on these aspects, and it's an important direction in the field of TTS. I wonder if the authors have some results in this area?

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents VoxCPM, a hierarchical model for zero-shot TTS. The authors point out that existing discrete token approaches, while stable, tend to lose subtle acoustic nuances, whereas fully continuous models preserve detail but often become unstable over longer utterances. VoxCPM addresses this by introducing a semi-discrete differentiable bottleneck that separates stable discrete, potentially linguistic and prosodic information, from finer acoustic details, avoiding the scalability issues of large discrete codebooks while keeping the system trainable end-to-end.

The model generates speech in stages: a text-semantic language model first captures content and prosody, this representation is regularized through a scalar quantization layer to form a coarse conditioning, and a residual acoustic module restores missing details such as timbre and micro-prosody. A local diffusion transformer then synthesizes the final latent audio segments, conditioned on both text and previously generated context. The latents are derived from a causal VAE.
Training is done jointly with a flow-matching objective.

The proposed method is technically sound and well-motivated, addressing a central bottleneck in hierarchical speech synthesis. The proposed semi-discrete, differentiable quantization + residual refinement framework feels like a natural evolution of current VALL-E-style architectures toward more continuous, end-to-end differentiable systems of hierarchical speech synthesis.

### Strengths
1. Cohesive hierarchical design: The decomposition into semantic (TSLM+FSQ), residual (RALM), and generative (LocDiT) components is conceptually elegant and grounded in recent hierarchical TTS models.

2. Innovative FSQ usage: Using scalar quantization as a differentiable inductive bias (rather than discrete target) is novel—addresses VQ scalability while maintaining stability.

3. End-to-end differentiability: Gradients flow through quantization and all components, unlike most multi-stage systems.

4. Strong performance across benchmarks

5. The t-SNE plots of the TSLM and RALM show a clear division of roles of the two stages.

### Weaknesses
1. Complex multi-stage hierarchy:
The full pipeline—LocEnc → TSLM → FSQ → RALM → LocDiT—adds several interacting components and training dependencies. Despite being end-to-end, this design could be challenging to scale or fine-tune efficiently.

2. Limited analysis of latency and efficiency:
The paper emphasizes real-time feasibility via a causal VAE but provides no measured latency or throughput results. It remains unclear how the diffusion-based LocDiT performs under streaming constraints.

3. FSQ interpretability and scalability:
While the ablation results show that the FSQ layer improves stability, plays a distinct role from the RALM, and remains largely speaker- and acoustics-agnostic, the claim that FSQ representations correspond to linguistic content is not directly supported by experimental evidence. This interpretation is plausible, but could be strengthened through probing analyses—such as phoneme prediction or ASR accuracy tests—to confirm that the FSQ indeed captures linguistic or prosodic structure. In addition, the paper does not discuss how well these semi-discrete representations transfer or scale when fine-tuned on different datasets, even within the same language domain.

4. The diagram should include the mathematical notation to make it clearer which token/color corresponds to which notation in the equation and text.

### Questions
See weaknesses above.

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
3

### Summary
The paper introduces VoxCPM, a framework that addresses speech expressivity and intelligibility challenges. Codec-based methods enable stable autoregressive generation but lose fine-grained acoustic details due to quantization, while continuous approaches accumulate errors over long sequences. VoxCPM combines a Text-to-Speech Language Model (TSLM) for generating semantic and prosodic plans with a Residual Acoustic Language Model (RALM) that restores fine acoustic details. Guided by these components, a local diffusion-based decoder produces high-fidelity speech latents. Experiments on two datasets show that VoxCPM achieves better results with fewer model parameters.

### Strengths
- The paper proposes a differentiable FSQ bottleneck that divides semantic–prosodic planning from acoustic rendering, maintaining full end-to-end trainability and overcoming the limits of both discrete and continuous methods.

- The paper reduces computational cost by using a VAE-based model that works in a compact latent space rather than on raw waveforms. A causal design allows for low-latency, real-time streaming synthesis.

- The paper shows that VoxCPM outperforms other open-source models on the experimented dataset using 0.5B parameters, outperforming larger models such as IndexTTS2 (1.5B) and HiggsAudio-v2 (3B).

### Weaknesses
- The paper combines existing components (pre-trained LLM, FSQ, diffusion decoder) without explaining why this combination works beyond T-SNE visualizations. What specific features do TSLM vs. RALM capture? Missing are probing experiments, attention analysis, layer-wise studies, or comparisons with alternative quantization methods. The paper should provide a few fundamental insights to verify their claims.

- The model does not use any explicit alignment mechanism, such as attention or duration modeling, to synchronize text tokens with acoustic frames. Instead, it depends on implicit correlations learned during training, which can result in mispronunciations, timing instability, or prosodic drift in longer utterances.

- The ablation studies don’t justify critical design decisions, including the 24-layer/6-layer split between TSLM and RALM, the contribution of pre-trained LLM initialization quality (e.g., comparing MiniCPM-4 vs. other pre-trained models), and the necessity of the FSQ bottleneck itself by comparing against simpler hierarchical architectures without quantization.

### Questions
- Regarding the acoustic embeddings (E<i), what specific acoustic features (pitch variations, timbre, voice quality) does it learn?

- Could the author clarify how the proposed framework manages long audio sequences?

- RALM is reported to recover “speaker identity” implicitly, yet the model has no explicit conditioning on speaker embeddings or style tokens. Could the authors clarify how speaker identity is actually captured?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a hierarchical AR framework based on semi-discrete residual representations to mitigate the trade-off between stability and expressivity observed in recent LLM-based TTS systems using speech tokenizers. The proposed architecture comprises a Text-Semantic Language Model that captures high-level linguistic and prosodic structure, and a Residual Acoustic Language Model that reconstructs fine-grained acoustic details. It further incorporates a differentiable FSQ bottleneck with LocDiT and a causal VAE to achieve unified and high-fidelity speech generation.

### Strengths
- The paper presents a clear and well-structured framework that effectively balances stability and expressivity through hierarchical semantic-acoustic modeling. The proposed combination of TSLM, RALM, and the FSQ bottleneck is conceptually coherent and technically sound. 
- The writing is clear and easy to follow, and the experimental results are well organized and convincingly support the main claims.

### Weaknesses
- The paper introduces FSQ as a semi-discrete bottleneck, but its function appears closer to latent feature regularization than true discrete representation learning. Because FSQ is not used for token prediction, it mainly constrains feature space rather than modeling discrete structure. A comparison with VAE-based latent regularization would help determine whether FSQ provides meaningful representational advantages or simply smoother feature constraints. 
- The authors claim the system as an  end-to-end framework, but it cannot be considered truly  end-to-end because it relies on a pre-trained audio VAE to extract continuous speech latents.
- The paper provides limited details about the Causal Audio VAE, even though the design of latent speech tokens can significantly affect overall performance.

### Questions
Q1. Does the observed separation in Appendix F.5 - Figure 2 truly result from the FSQ bottleneck, or is it primarily due to the pre-trained initialization of the semantic LLM? How would the t-SNE visualization change if the TSLM were trained from random initialization?

Q2. In most recent LLM-based TTS systems, increasing model capacity generally improves both speaker similarity and intelligibility. However, in the ablation study (Table 4), deeper TSLM layers slightly increase SIM but noticeably degrade WER. It would be helpful if the authors could clarify the reason for this. observation. Specifically, how can the effects of structural disentanglement be distinguished from capacity related trade-offs, and might the observed degradation reflect limitations in training dynamics rather than an inherent need for the RALM module?

Q3. The paper states that the TSLM uses MiniCPM-4-0.5B as its backbone, while the overall model size is also reported as 0.5B parameters. Could the authors clarify how this total is computed? Specifically, how many parameters are allocated to the AudioVAE and the RALM modules?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This manuscript proposes a novel end-to-end TTS framework, VoxCPM, to address the fundamental trade-off between discrete and continuous speech representations, where discrete representations maintain stability but sacrifice expressivity, while continuous representations retain acoustic richness but are prone to error accumulation. The hierarchical semantic-acoustic modeling framework is well-motivated and addresses some limitations in current TTS systems. 

The proposed VoxCPM is with the hierarchical semantic-acoustic architecture, consisting of a Text-Semantic Language Model (TSLM), a Residual Acoustic Language Model (RALM), and the Local Diffusion Transformer (LocDiT). The key innovation is the differentiable Finite Scalar Quantization bottleneck, which serves not as a prediction target, but as an intermediate regularization mechanism to induce natural task separation. 

The manuscript demonstrates that their 0.5B-parameter model achieves state-of-the-art zero-shot TTS performance with extensive experiments and ablations.

### Strengths
1. The major contribution, using a differentiable FSQ bottleneck as an internal inductive bias rather than a predictive target, is a elegant and well-motivated solution to the task entanglement problem. By regularizing the hidden state, it forces the TSLM to focus on stable semantic-prosodic structures while offloading fine-detail modeling to the RALM.

2. The paper provides compelling quantitative results. VoxCPM achieves state-of-the-art performance among open-source models on two challenging benchmarks, SEED-TTS-EVAL and CV3-EVAL. The ablation studies conclusively validate the authors' core hypotheses.

3. While the best-performing model (VoxCPM) is trained on a large, proprietary dataset , the authors have responsibly run all critical ablation studies on the public Emilia dataset.

### Weaknesses
1. While the empirical results are strong, the paper lacks theoretical justification for why FSQ specifically induces semantic-acoustic disentanglement

2. No formal analysis of the "quantization ceiling" or error accumulation mechanisms, and how the proposed architechture can effectively address them.

3. The claim that FSQ acts as an "inductive bias" needs more rigorous theoretical grounding and mathematical proof.

4.  The main SOTA results are from the VoxCPM model trained on a 1-million-hour internal, bilingual dataset. While the authors provide VoxCPM-Emilia, trained on a public dataset, the top-line performance is dependent on this massive, inaccessible dataset. This limits the direct reproducibility of the primary claims, although the architectural insights from the Emilia-based ablations remain sound. Some baselines (F5-TTS, GPT-Sovits) may use significantly less data, making direct comparison problematic.

5. The paper doesn't discuss whether the performance gains are primarily from architecture or data scale

6. The entire model generates latents from a pre-trained Causal VAE. This VAE is itself a form of compression. The paper criticizes discrete tokenizers for their "quantization ceiling", but doesn't discuss the potential information bottleneck or quality ceiling imposed by its own VAE. The quality of this VAE is critical to the final output, but it's only briefly detailed. 
The author also claimed that the voxCPM do not rely on a external tokenizer. However the VAE is itself a tokenizer.

### Questions
1. You sum TSLM and RALM outputs. Did you experiment with other fusion strategies (concatenation, gating, attention)?

2. The paper states FSQ is "analogous to the first layer of Residual Vector Quantization (RVQ)". Did you experiment with using a learned vector quantizer (e.g., a single VQ layer) as the bottleneck instead of the non-learned FSQ? It seems a learned codebook might create an even more efficient semantic-prosodic skeleton. And what happens if you replace FSQ with VQ-VAE or other quantization methods?

3. Please discuss the potential quality ceiling imposed by their pre-trained causal VAE  and how it relates to the "quantization ceiling" they criticize in discrete tokenizers.

4. Please try your best to provide more theoretical analysis as mentioned in "Weaknesses".

4. The VAE is also external tokenizer. 

5. For citation formatting, please ensure consistent citation style (some entries lack year information)

### Soundness
3

### Presentation
3

### Contribution
3
