# VoiceBridge: Designing Latent Bridge Models for General Speech Restoration at Scale

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Bridge models have recently been explored for speech enhancement tasks such as denoising, dereverberation, and super-resolution, while these efforts are typically confined to a single task or small-scale datasets, with constrained general speech restoration (GSR) capability at scale.
In this work, we introduce VoiceBridge, a GSR system rooted in latent bridge models (LBMs), capable of reconstructing high-fidelity speech at full-band (\textit{i.e.,} 48kHz) from various distortions. 
By compressing speech waveform into continuous latent representations, VoiceBridge models the \textit{diverse LQ-to-HQ tasks} (namely, low-quality to high-quality) in GSR with \textit{a single latent-to-latent generative process} backed by a scalable transformer architecture.
To better inherit the advantages of bridge models from the data domain to the latent space, we present an energy-preserving variational autoencoder, enhancing the alignment between the waveform and latent space over varying energy levels. Furthermore, to address the difficulty of HQ reconstruction from distinctively different LQ priors, we propose a joint neural prior, uniformly alleviating the reconstruction burden of LBM. At last, considering the key requirement of GSR systems, human perceptual quality, a perceptually aware fine-tuning stage is designed to mitigate the cascading mismatch in generation while improving perceptual alignment. 
Extensive validation across in-domain and out-of-domain tasks and datasets (\textit{e.g.}, refining recent zero-shot speech and podcast generation results) demonstrates the superior performance of VoiceBridge.
Demo samples can be visited at: \url{https://VoiceBridgedemo.github.io/}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes using a latent-space bridge model for the task of speech enhancement. The bridge is trained between the latent representations of clean and degraded speech recordings, and the latent space is constructed using VAE. They authors propose several techniques to make VAE more suitable for encoding waveforms. Overall, the paper is quite solid and well-written.

### Strengths
The paper has the following key strengths:

1. Paper is well-structured and easy to read. The authors provide a detailed description of the training setup, and the training procedure is presented clearly. 
2. The authors propose a set of novel techniques that enhance the training of VAEs for constructing latent audio representations. They also provide an ablation study to validate the importance of each design choice.
3. The authors show that the proposed method performs very well on several benchmarks for speech enhancement, as well as on other tasks (codec artefact removal, denoising)

### Weaknesses
However, the following weaknesses can be identified:

1. Some relevant speech enhancement methods, though cited in the paper, are not included in the benchmark comparisons. For instance, the authors could consider adding FINALLY [1], the popular HiFi-GAN-2 [2] baseline, and other methods like MIIPHER [3] and Genhanser [4] to the comparisons.
2. In many speech enhancement applications, inference speed plays a crucial role. However, bridge methods typically require multiple steps to obtain the final samples. It would be interesting to see the inference time comparison between the presented model and the other benchmarks.

### Questions
I have the following suggestions for the authors to consider:

- Since the presented model supposedly solves the Schrödinger bridge problem, it would be interesting to compute Path KL, which is the estimate of how good the model solves optimal transport. 
- Moreover, other papers, e.g. FINALLY, claim that speech enhancement necessitates finding such an enhanced audio $\hat{x}_0$ that is the closest (in some meaningful sense) to the original noisy audio $x_1$. It would also be interesting to compare the baselines and the presented model regarding "how far" the enhanced audio is from the noisy input.


#### **References:**

[1] Babaev et al., "FINALLY: fast and universal speech enhancement with studio-like quality"

[2] Su et al., "HiFi-GAN-2: studio-quality speech enhancement via generative" 

[3] Koizumi et al., "MIIPHER: a robust speech restoration model integrating self-supervised speech and text representations"

[4] Yang et al., "Genhancer: high-fidelity speech enhancement via generative modeling on discrete codec tokens"

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
This paper introduces a latent bridge model (LBM) for speech restoration and enhancement. The authors introduce an extensive training scheme for a variational autoencoder (VAE) in order to have both an energy preserving latent space as well as a joint prior for both high-quality and low-quality samples which have been arbitrarily degraded from the same sample. The authors show that this makes the latent diffusion easier to solve leading to an improved bridge model down the line. The LBM itself is a transformer with 544M parameters. In the end, the authors also introduce a joint fine-tuning stage for the bridge model and the decoder to generate perceptually better samples using human feedback and a descriminator-based adversarial and feature-matching loss.

### Strengths
- The proposed system was explained in detail and in a very concise and understandable way.
- The experiments and ablations support the claim that their approaches improve their own baseline system.

### Weaknesses
- The overall novelty of the proposed improvements is low:
	- Learning a joint neural prior in a VAE which is used in an LQ to HQ speech enhancement/restoration system is not novel (see Fang et al., "Variational Autoencoder for Speech Enhancement with a Noise-Aware Encoder", ICASSP 2021)
	- Ensuring the energy level stays consistent in the latent space is not a novel idea in itself. While the specific learning target using a joint scaling in the loss formulation might be novel, Karras et al. "Analyzing and Improving the Training Dynamics of Diffusion Models", CVPR 2024, already showed that diffusion models benefit from magnitude preserving layers in the neural network architecture. More recently, Richter et al., "Normalize Everything: A Preconditioned Magnitude-Preserving Architecture for Diffusion-Based Speech Enhancement", WASPAA 2025, showed that this also holds for Schrödinger-bridge-based speech enhancement models. However, we acknowledge that the latter was published within the last two months and was previously only available on arXiv.
	- LBMs and transformer based speech enhancement models using diffusion, Schrödinger bridges or flow matching are not novel as well.
- There was no comparison with recent universal speech enhancement models, namely
	- Ku et al. "Generative Speech Foundation Model Pretraining for High-Quality Speech Extraction and Restoration", ICASSP 2025
	  This work is cited. However, no comparison is done with regards to their results. This is especially important as their results on the speech denoising task indicate significantly higher performance when compared to the same baselines.
	- Liu et al., "Generative Pre-Training for Speech with Flow Matching", ICLR 2024
	  This work was not cited but the same holds for the results here, how does the proposed approach compare to SpeechFlow?

### Questions
- How does the model compare to the recent SOTA approaches:
	- Ku et al. "Generative Speech Foundation Model Pretraining for High-Quality Speech Extraction and Restoration", ICASSP 2025
	- Liu et al., "Generative Pre-Training for Speech with Flow Matching", ICLR 2024

- The clean speech VAE reconstructed, i.e., $\hat{\bf{x}}_0=\mathcal{D}(\mathcal{E}^{\text{np}}(\bf{x}_0))$, metrics were reported. However, the degraded and then only using the VAE as enhancement module output was not investigated. As the joint neural prior should make it possible to reconstruct the clean sample in the best case, the metrics for $\hat{\bf{x}}_0=\mathcal{D}(\mathcal{E}^{\text{np}}(\bf{x}_1))$ would be interesting to report.

- Please consider renaming the fine-tuning loss which should optimize for perceptual quality as the term "human feedback" implies something differently. From our understanding the $\mathcal{L}_\text{hf}$ loss is a perceptual loss based on PESQ and UTMOS. Thus, it should be named accordingly.

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a general speech restoration system based on latent bridge models that unifies diverse degradation-to-clean speech tasks in a single framework, achieving high-fidelity 48 kHz restoration through many innovations: energy-preserving VAE for structured latent space, a joint neural prior that fine-tunes the encoder to map degraded inputs closer to clean speech embeddings, and crucially perceptual-aware fine-tuning for human-aligned quality.

### Strengths
Results are strong, adequate empirical validation on multiple datasets and showing generalization (on reasonably matched domains though: only on unseen codec artifacts and from dns synthetic training to real). 
Comparison with strong baselines. 
Good writing and structuring.

### Weaknesses
A minor weaknes is that it is mainly incremental/engineering work as the core components ideas are already been proposed 
(Schrodinger bridge for SE, and transformer backbone and VAE are adapted from existing architectures ).
However there is still much value in engineering and incremental work IMHO. 

The model's total number of parameters is 544M in the transformer plus 156M each in the VAE encoder and decoder, totaling 856M parameters. This is substantially larger than baseline models such as Unverse++ (\~100M) or AnyEnhance (\~300M). As such, the comparisons in Table 1, for example, are not entirely fair. The better performance of the proposed method may stem from the increased model capacity rather than the novel methodology itself.

The use of UTMOS and PESQ losses calls some of the results into question. The comparisons in Table 1 are not entirely fair, at least regarding PESQ and UTMOS, as some baselines did not optimize directly for these metrics. The only way to rule this out, unfortunately, is by conducting larger-scale listening tests (as per Goodhart's law, PESQ and UTMOS become unreliable as evaluation metrics once they are used as objective functions e.g. see [1]).
To their credit, the authors discuss this issue in the Appendix and in the ablation section, and I find the discussion quite insightful. Looking at Table 15, the performance without these losses appears much weaker, which calls into question many of the results and the much of the effort and some design choices presented in the paper. 
This paper contains numerous contributions and represents a substantial engineering effort, but it is sometimes difficult to extract clear scientific insights about which components are most important. I believe the ablation results in the Appendix (Table 15) are among the most significant findings, as they suggest that what truly makes the model better is the combined use of GAN and perceptual losses. As such, I think these results should be presented more prominently in the main manuscript (e.g. included in some results Tables maybe ?)

Unfortunately, in its current form, the manuscript would benefit from refocusing on its most impactful contributions before publication.

[1] de Oliveira, D., Welker, S., Richter, J., & Gerkmann, T. (2024). The PESQetarian: On the Relevance of Goodhart's Law for Speech Enhancement. Interspeech 2024

### Questions
You train two separate encoders: E for HQ and E^np for LQ. Have you tried using a single conditional encoder E(x, is_lq)?

### Soundness
2

### Presentation
4

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
This paper proposes to tackle the general speech restoration task from the latent space with a Schrödinger Bridge model. To improve the performance of latent diffusion models in tackling diverse distortions in audio signals, an energy-preserving variational autoencoder and a joint neural prior are proposed. Furthermore, a refinement stage is applied for perceptual enhancement. Extensive experiments are conducted to verify the effectiveness of the proposed methods on diverse benchmarks.

### Strengths
1. The proposed joint neural prior, which uniformly reduces the distance between each degradation prior and the target, offers a promising approach for general speech generation and holds potential for advancing related domains.

2. This paper presents novel experiments on codec-artifact removal and post-processing for TTS systems.

### Weaknesses
1. The motivation for performing general speech restoration in the latent space rather than the data space is not clearly stated in the paper.
Additionally, latent diffusion has been widely applied in other domains (e.g., computer vision and audio generation); its novelty in the context of speech restoration remains limited. Clearer technical distinctions or improvements over prior work should be provided.

2. Based on the experimental results, the proposed VoiceBridge does not bring promising improvements compared with other baseline models, providing insufficient evidence of the necessity of adopting latent diffusion. 

3. The organization of the manuscript has minor structural issues, where the VoiceBridge training pipeline is only briefly described in the experimental setup section rather than being presented comprehensively in the methodology part.

4. The proposed perceptual-aware generation process has been a common practice applied in the speech restoration task and audio generation task, thus lacking sufficient novelty.

### Questions
1. Based on the experimental results, the adopted joint neural prior, which aims to uniformly reduce the distance between each degradation prior and the target, contributes to the overall improvements. However, it remains unclear to readers whether the adopted joint neural prior benefits all distortions equally or actually excels for certain distortions while underperforming for others. It would be more convincing to provide the results on each degradation to verify the proposed method.

2. The author could provide auxiliary ablation studies on performing latent diffusion from the data space to verify the effectiveness of the latent diffusion for the general speech restoration task. 

3. The formatting of the references is inconsistent. I recommend unifying the citation style and ensuring accurate, case-sensitive capitalization of all titles.

### Soundness
2

### Presentation
2

### Contribution
2
