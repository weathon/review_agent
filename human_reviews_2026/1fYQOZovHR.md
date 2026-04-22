# MOLM: Mixture of LoRA Markers

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Generative models can generate photorealistic images at scale. This raises serious concerns about the ability to detect synthetically generated images and attribute these images to specific sources. While watermarking has emerged as a possible solution, existing methods remain fragile to realistic distortions, susceptible to adaptive removal, and expensive to update when the underlying watermarking key changes. We propose a general watermarking framework that formulates the encoding problem as key-dependent perturbation of the parameters of a generative model. Within this framework, we introduce  Mixture of LoRA Markers (MOLM), a routing-based instantiation in which binary keys activate lightweight low-rank adapters (LoRA) inside residual and attention blocks. This design avoids key-specific re-training and achieves the desired properties such as imperceptibility, fidelity, verifiability, and robustness. Experiments on Stable Diffusion and FLUX show that MOLM preserves image quality while achieving robust key recovery against distortions, compression and regeneration, averaging attacks, and black-box adversarial attacks on the extractor. Code is available at [https://github.com/Samar-Fares/MOLM-Watermark](https://github.com/Samar-Fares/MOLM-Watermark)

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
MOLM repurposes multiple LoRA adapters as watermark “markers” and uses key-dependent routing (MoLE-style) to embed a bitstring while keeping the backbone frozen. A learned extractor recovers bits. Experiments on SD-1.5 and FLUX report near-zero inference overhead, small FID changes, and robustness to common distortions and selected white-box attacks.

### Strengths
1.Practicality. Lightweight LoRA adapters, no architecture changes, negligible runtime overhead.
2.Coverage. Evaluation on SD-1.5/FLUX with ~28-bit default capacity and higher-capacity variants; ablations on placement/configuration.
3.obustness. High bit accuracy under JPEG/crop/resize and diffusion regeneration; PGD-style white-box results show resilience.

### Weaknesses
1. The method is a straightforward combination of existing techniques (LoRA + MoE routing).
2. Fails to compare with recent state-of-the-art watermarking methods.

### Questions
1.Novelty. The core idea of MOLM is to repurpose LoRA adapters as watermark carriers and use key-driven routing to activate specific adapters for embedding. This is essentially a composition of mature techniques: 1) LoRA is a well-established parameter-efficient fine-tuning method; MOLM merely reinterprets it as a watermark carrier. 2) The routing logic follows MoLE, replacing data-gated expert selection with key-gated selection.
2.Baselines. Include recent watermarks or clearly justify exclusion.
3.Presentation. Fix typos (e.g., “Deployment”; “~8 days”), align the Table-3 title with its contents ('inference' is not listed, although it is mentioned in the paper), and unify notation/spacing across equations and tables.

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
3

### Summary
The paper proposes MOLM, a watermarking framework for text-to-image diffusion models that views watermarking as key-dependent parameter perturbations of a *frozen* generator. Concretely, the method installs LoRA adapters in selected residual/attention blocks and routes activation through one adapter per block according to chunks of a binary key. Thus, a key defines a deterministic execution path (the perturbation), while the backbone weights remain unchanged. A lightweight extractor network recovers the key from generated images. The training objective balances a perceptual loss for imperceptibility with a bitwise BCE for verifiability. Experiments on Stable Diffusion v1.5 (512×512) and FLUX (1024×1024) show high key recovery, small FID deltas (≤ ~1.5), and robustness to a range of common image distortions, learned compression, diffusion-based regeneration, averaging, and white-box PGD attacks. Default configuration routesL=14 decoder ResNet blocks with P=4 adapters per block, yielding M=28-bit keys; larger capacities are discussed via more blocks or choices

### Strengths
1. Casting watermarking as key-conditioned parameter routing over a frozen backbone is elegant, modular, and orthogonal to model architectures. The idea neatly ties together LoRA efficiency with watermarking needs.
2. Keys correspond to routing masks; adapting capacity does not require retraining the backbone. The paper reports ~1 GPU-day one-time training and no additional inference cost beyond activating chosen LoRA paths.

### Weaknesses
1. The paper frames both detection and attribution but largely evaluates bit recovery / TPR against a fixed key/threshold. How MOLM behaves with large key databases (collisions, nearest-neighbor attribution errors, false-match rate under heavy post-processing) is not fully quantified.
2. The white-box attacks target the extractor in image space. Stronger adversaries could attempt prompt-, noise-, or sampler-level optimization to suppress/flip bits while staying on-distribution (e.g., plug-and-play adversaries).
3. While MOLM avoids per-key retraining, operational details (e.g., revoking/rotating keys, multi-tenant key assignment, rate-limited leakage scenarios, collusion of users averaging many outputs with heterogeneous keys) are not empirically explored. 
4. The table compares to Stable Signature, AQuaLoRA, WOUAF (bit recovery) and Tree-Ring/ROBIN/Gaussian Shading (detection). Some strong recent defenses/attacks (esp. post-2024 SoK/benchmarks) may warrant a tighter apples-to-apples setup (identical prompts/seeds, extractors retrained with the same aug sets), though the paper cites them. 
5. The robust baselines for watermarking arbitrary images are lacking, such as TrustMark [1], VINE [2], StegaStamp [3]. Incorporating those methods would make the comparison more thorough and robust.


[1] TrustMark: Universal Watermarking for Arbitrary Resolution Images

[2] Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances

[3] StegaStamp: Invisible Hyperlinks in Physical Photographs

### Questions
1. If a key leaks, how quickly can you rotate without retraining? Is there support for *soft revocation* (e.g., attenuating adapters) versus installing fresh adapter banks?
2. You evaluate averaging removal/forgery with the same messages; what about heterogeneous-key collusion across many users? Can mixed-key averaging partially cancel the watermark?
3. Have you tried prompt/noise optimization targeting the extractor gradient via a differentiable proxy, or classifier-free guidance tuning to hide bits while maintaining content?

### Soundness
2

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
3

### Summary
This paper addresses verifying and attributing AI-generated images. The authors propose a watermarking framework inspired by the Mixture of LoRA Experts. They formulate the problem as key-dependent perturbations of frozen generative models, where binary keys are used to route among a mixture of LoRA adapters inserted into the generative model. In addition, the authors consider imperceptibility, fidelity, verifiability, and robustness by incorporating the L_imp​ and L_ver​ losses.

### Strengths
The underlying ideas are reasonable.

The evaluation demonstrates imperceptibility, fidelity, verifiability, and robustness.

The paper includes ablation studies supported by experiments and visualization.

The experimental details are clearly presented.

### Weaknesses
The watermarking results do not clearly demonstrate state-of-the-art performance; they are only competitive with previous methods.

The description of the proposed method is generally ok, but some paragraphs are poorly written and fragmented, which affects readability.

### Questions
The authors mention that the proposed method is inspired by Mixture of LoRA Experts (MoLE), but the motivation behind this choice is unclear. It would be helpful to explain the intuition for why MoLE's structure is beneficial for the watermarking problem, beyond the fact that it can be adapted to it.

Diffusion version 3.5 Large (8B) has already been released on Hugging Face. However, this paper still evaluates on version 1.5. It would be more convincing to include results from a more recent version to demonstrate scalability and relevance.

In Table 1, the bolded values appear to indicate the best and second-best results within each column, but this should be explicitly explained in the table caption. The meaning of the orange-highlighted values should also be clarified.

It would strengthen the paper if more baseline methods were reimplemented and evaluated on the FLUX architecture and LAION dataset to provide a fair and comprehensive comparison with the proposed approach.

### Soundness
2

### Presentation
2

### Contribution
2
