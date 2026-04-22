# Next-Scale Autoregressive Models are Zero-Shot Single-Image Object View Synthesizers

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 8, 4, 6

## Abstract
Learning to synthesize novel views without explicit 3D representations or hand-crafted 3D inductive bias has recently gained attention: it is simpler, more formally direct, and better aligned with the lesson that scalable learning paradigms with less assumptions built into architectural design (e.g., regarding geometry) often win. However, the current dominant solutions are diffusion-based, which typically suffer from problems like slow inference. We introduce ArchonView, the first autoregressive model for zero-shot single-image, object-centric novel view synthesis (NVS), achieving substantially faster inference, higher accuracies, and notably not relying on fine-tuning of 2D generative checkpoints (challenging the common assumption that 2D priors are required in diffusion-based NVS). We design innovative methods of both global and local conditioning to suit characteristics of the NVS task. Crucially, a naïve application of next-scale autoregression fails; we identify two design choices that unlock performance: local conditioning pre-filling, and removing global AdaLN at the classifier head. ArchonView delivers state-of-the-art zero-shot results across six standard benchmarks (GSO, ABO, OmniObject3D, RTMV, NeRF-Synthetic, ShapeNet), while being several times faster than diffusion baselines (e.g., 0.22s v.s. 1.7–1.8s per view at matched parameter count). It consistently improves synthesis accuracy, and scales predictably with both model size (135M–2B) and data size, exhibiting clear scaling-law-like trends. Our findings suggest a paradigm shift and challenge an existing assumption: first, for object-centric NVS, next-scale autoregression can be faster, simpler, and more accurate than diffusion; and second, priors obtained from fine-tuning 2D-pretrained models may not be necessary for generative NVS. Our code is open-sourced at https://anonymous.4open.science/r/ArchonView/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper extends the next-scale autoregressive generation paradigm from 2D image modeling to zero-shot 3D novel view synthesis (NVS). The proposed ArchonView model autoregressively predicts higher-resolution scales conditioned on a single input view. It achieves competitive or superior results compared to diffusion-based baselines such as Zero-1-to-3 and EscherNet, with significantly faster inference and no reliance on pretrained 2D diffusion models.

### Strengths
Interesting and timely exploration of applying next-scale generation to 3D NVS, challenging the diffusion-based dominance in this field.

Empirically strong results on multiple object-level benchmarks, with improved efficiency and competitive quality.

The overall design remains simple yet effective, providing a clean baseline for future autoregressive extensions in NVS.

### Weaknesses
- Conceptual clarity on “next-scale” in 3D

The paper straightforwardly brings the 2D next-scale paradigm into 3D, but it remains unclear what the “scale” represents in this context.
In 3D, is the next-scale process simply applied independently within each view, or could there exist a more native 3D notion of scale that aligns across multiple views?
The current adaptation feels like a direct transplant from 2D without strong 3D intuition or justification.
Some conceptual discussion or visualization would help clarify how “next-scale” manifests in multi-view geometry.

- Limited supporting of view numbers

EscherNet supports flexible N-to-M view synthesis, while this work only shows 1-to-1 generation.
Can this method extended to flexible multi-view inputs or outputs? If so, how would the computational complexity scale with N and M?
This discussion is important to position the method within the broader NVS landscape.

- Result discrepancies

The GSO performance reported for Zero123 differs from the original paper (Table 1).
Please clarify whether this comes from dataset versions, evaluation protocols, or re-implementation differences.

- Lack of visualization for 3D consistency

The paper mainly presents single novel views. For a model claiming 3D-aware generation, visualizing the synthesized 360° trajectory from a single input would be essential to evaluate consistency and geometry preservation.
Without this, it remains unclear how coherent the generated views are across different poses.

- Lack of diversity in generation

Does the model produce diverse outputs given the same input view and target pose, as seen in diffusion-based models like Zero123 (e.g., Fig. 8 in their paper)? 

- Compute scale and more diverse examples

The model is trained on 32 H200 GPUs, much larger than prior works like EscherNet or Zero123 (8 A100 GPUs).
The paper doesn't show any diverse qualitative “in-the-wild” examples. The other baselines are trained under the same data scale but show very impressive diverse examples.


- Reuse of pretrained components

The model reuses the pretrained next-scale VQVAE but trains the transformer backbone from scratch.
Why not or can't also reuse or partially initialize the pretrained transformer with zero-init?
Will this improve generalization?

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ArchonView, a model for zero-shot, single-image novel view synthesis based on a next-scale autoregressive paradigm. This paper conducts an in-depth investigation into efficient and high-quality novel view synthesis within the VAR framework. For example, since the conditions for novel view synthesis do not satisfy pixel alignment, the authors analyze different conditioning methods, including Prefilling, Causal Conditioning, and Cross-Attention. Their results show that simple prefilling significantly outperforms the other two approaches.

### Strengths
1. For the task of novel view synthesis, this paper provides a thorough analysis of various design choices and conditioning strategies within the VAR framework. The discussion covers conditioning methods, insights from The Devil is in the Classifier Head, Semantic Global Pose Conditioning, and Multi-Scale Local Conditioning.
2. The paper demonstrates the scalability of the proposed method through extensive experiments. By systematically increasing both the model size and the dataset size, the authors plot performance curves that clearly illustrate the improvements achieved.
3. For the task of novel view synthesis, the proposed method achieves faster and better performance compared to baseline approaches.

### Weaknesses
1. The baselines compared in this paper are relatively outdated, and many newer baselines have been introduced since. It is necessary to discuss and compare the proposed method with these more recent approaches.
2. The statements in the discussion section are somewhat confusing. The paper mentions, “In contrast, we only use the ‘fine-tuning’ dataset of previous works (with 800k 3D objects) and achieved significantly superior results.” However, it appears that the method also relies on a pretrained checkpoint for fine-tuning?

### Questions
The paper offers limited discussion on the practical applications of the proposed method, such as whether it is intended for 3D reconstruction or 3D generation, or how it could be integrated into scene understanding or combined with existing VLMs. These aspects warrant further exploration and discussion.

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
This paper address generative novel view synthesis with one single input image. This problem is highly proabilistic, and the author adopted the visual autoregessive model based approach for such generative task.  The author used the residual VQVAE from the original VAR paper, then  trained the generative model (next-scale prediction with block-wise causal transformer) from scratch on Objectverse dataset.  To improve results,  the author also applied CFG on the pose embedding.  Also the author tweakes the architecure a bit, mostly by removing the adaptive LN layer befire the classifier head.  The author compared their methods with a few diffusion based baseline, including Zero 1-to-3, Zero 123-XL, EscherNet.  These baselines finetuens a pretrained image diffusion model on Objectverse for generative novel view synthesis. The author shows improves results, e.g. on GSO (17.44 PSNR vs. 16.77).  The author showed ablation study on its architecture design, showed quite clear improvements.

### Strengths
1. The method is overall relative new for the field of novel view synthesis, and the author made several archteicture change that significantly improves the results. Which is quite solid. 
2. The author submitted code at the time of submission, which is a good practice.

### Weaknesses
I don't agree with a few of arguments (or maybe just wordings) from the author

Training from scratch is cool and quite impressive, but I don't agree that these diffusion based approach relying on pretrained checkpoints is a big drawbacks.  (as argued in the second paragraph of the introduction).  Also, this does not seem to be the problem of diffusion model. Diffusion model is just an algorithm for generative modelling,  previous baseline relies on pretraining and then finetuning does not mean that training a well-designed diffusion model for NVS task from scratch would not work.  (e.g. the first sentence in the 3rd paragraph of the intro goes:  such downside of diffusion calls for xxxx).  

The point is you don't need to over criticize the diffusion based approaches in your paper. VAR for NVS is already quite interesting and impressive. 



Another weakness of the paper is that the metric is so low to be indicative. Table 1 shows PSNR of 10-19.42.  For PSNR in this range, it's really hard to tell if the improvements are useful. I would suggest using a subset with nearby camera viewpoints (larger view overlap) for evaluation.  For multiview PSNR on GSO, people already got PSNR over 30.

### Questions
I would be curious, if the author train a diffusion baseline from scratch with similar recipes, how would that perform? And also how would that perform if the author adds architecure improvements like global and local conditioning.

### Soundness
3

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
4

### Summary
The paper adopts a next-scale autoregressive model—specifically VAR—for single-image novel view synthesis. Relative to diffusion, the VAR backbone offers much faster per-view generation (no multi-step denoising), a simpler inference pipeline, competitive or better fidelity, and predictable scaling with model and data size. To enable NVS, the authors augment VAR with: (i) a global “posed” start token that fuses CLIP semantics with the target relative pose; (ii) local multi-scale “prefilling,” which prepends the source image’s VQVAE tokens at every scale under a causal/block-triangular mask to guide generation; and (iii) an architectural fix—removing AdaLN at the classifier head—to preserve local correspondences. Experiments on six object-centric benchmarks (GSO, ABO, OmniObject3D, RTMV, NeRF-Synthetic, ShapeNet) in a zero-shot setting show state-of-the-art accuracy with several-times faster inference than diffusion baselines.

### Strengths
* Originality: To the best of my knowledge, this is the first work that adopts VAR for single-image novel view synthesis.

* Quality:

  * Both the qualitative and quantitative results are significant, demonstrated across multiple object-centric benchmarks.
  * The major components—global pose conditioning, local prefilling, and classifier-head AdaLN removal—are well supported by aligned ablation studies.
  * Efficiency and scaling capabilities are also demonstrated through experiments.

* Clarity: 
  - The paper is in general well structured.
  - The source code is given.

* Significance:

  * The proposed solution demonstrates the potential of VAR for novel view synthesis in both quality and efficiency, and can inspire follow-up research on this alternative (and potentially superior) model paradigm.
  * The solution is backed by actionable insights, which could transfer to areas beyond novel view synthesis.

### Weaknesses
* While the solution itself is clear, the motivation and insights behind the design need more elaboration:

  * Local attention. Provide a deeper investigation of how attention behaves in this model—e.g., whether generated patches attend to the intended input patches at the desired locations. Concretely, add attention visualizations across scales, quantify attention mass within pose-consistent neighborhoods, and report correspondence accuracy vs. pose gap.
  * Attention design (lines 249–265). Since VAR also uses prefilling techniques, clarify what is additionally novel here. Distinguish your contribution from VAR via ablations isolating token prepending vs. causal/block-triangular masking vs. cross-attention, and report any compute/latency trade-offs introduced by your variant.
  * AdaLN claim (lines 283–296). The claim is somewhat ambiguous and currently supported only by empirical results. Please add a more theoretical or mechanistic explanation (with a simple formulation, if possible), and consider alternatives (e.g., scaled/gated/partial AdaLN or layer-wise removal) with diagnostics such as layer-wise gradients/feature norms to substantiate the hypothesis.

* Beyond single-view fidelity, the paper should investigate cross-view consistency of synthesized views. Recommend a multi-target protocol (e.g., 8–16 target poses per source) and report consistency metrics (cycle/epipolar consistency, normal/depth agreement, or reconstruction consistency via a downstream NeRF fit), alongside qualitative failure cases.

### Questions
- First, please refer to the weaknesses.
- Second, I would like to ask whether the proposed solution in this paper can be extended to text to 3D model (multi-view images) generation with minimum efforts, if so what are the potential efforts.

### Soundness
3

### Presentation
2

### Contribution
3
