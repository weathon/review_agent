# Dress&Dance: Dress up and Dance as You Like It

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We present Dress&Dance, a video diffusion framework that generates high-quality virtual try-on videos of users wearing desired garments while performing complex motions. Our approach is the first to achieve high-resolution (1152×720) at high-FPS (24 FPS), with support for various try-on modes, including simultaneous try-on of tops and bottoms. At the core of our framework is CondNets, a novel attention-based conditioning architecture that unifies heterogeneous multi-modal inputs – including garment images, user images, motion videos, and text prompts – into a single homogeneous token sequence. To prevent strong pre-trained text priors from overshadowing garment inputs, we introduce garment-aware target steering, a guidance mechanism that enforces accurate garment placement. To further address both data scarcity and computational demands of training high-quality video models, we propose a synthetic triplet generation strategy for producing paired training data and a multi-stage training curriculum that progressively scales resolution and frame rate. Our framework outperforms existing open-source and commercial solutions, enabling flexible, high-quality try-on experiences that faithfully preserve garment details, user identity, and complex motions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Dress&Dance, a video diffusion framework enabling high-resolution virtual try-on videos, given a single user image, desired garment(s), and a reference motion video. The core innovation is the CondNets attention-based conditioning mechanism, designed to unify heterogeneous inputs—garments, users, motion references, and text—into a single token sequence suitable for scalable DiT-based video diffusion. The authors also address data scarcity and efficiency with a synthetic triplet data creation process and a multi-stage curriculum training strategy, facilitating training for large-scale video generation with limited paired try-on data. Extensive experiments are conducted with both quantitative and qualitative results, showing Dress&Dance outperforms strong baselines—including recent diffusion-based, commercial, and open-source methods—on metrics of fidelity and visual quality, and in support for complex motions and composite try-ons.

### Strengths
1. Scalable, Efficient Training Pipeline: The combination of synthetic triplet data creation (Section 3.2) and curriculum learning, where complexity increases along both task and resolution axes, is both pragmatic and generally applicable. The video-image hybrid training design (page 7) wisely utilizes abundant image data while aligning with the final inferential target.
2. Comprehensive, Robust Evaluation: The paper conducts both quantitative and qualitative comparisons, including commercial (Kling, Ray2) and recent strong open-source baselines, as well as ablative studies. Metrics span PSNR/SSIM/LPIPS and GPT4-based human/evaluator scoring, with cross-checks on Internet and captured datasets.
3. State-of-the-art Results and Strong Visuals: On core quantitative metrics, Dress&Dance surpasses all open-source and commercial baselines in garment and user appearance fidelity, and offers visually plausible, temporally consistent motion as attested by the sequences in the supplementary videos.
4. Practical System Contributions: Fast inference, user-facing modes, text-only try-on, and garment transfer (including multiple garments without explicit label) illustrate the system’s practicality and flexibility.

### Weaknesses
1. Overstated Claims and Weakness in Theoretical/Algorithmic Rigor: Much of the novelty is architectural and procedural, not fundamentally theoretical; e.g., Section 3’s CondNets is a nontrivial compositional application of attention blocks (as detailed in Figure 4 and G.1), but does not introduce fundamentally new attention or conditioning mechanisms—aware reviewers may view this as strong engineering and system design, not a conceptual breakthrough. E.g., equations governing training loss, attention fusion, and effect of garment-aware steering are described informally, and there are no explicit equations for the loss/objective (other than hints in the architecture description).
2. Lack of Ablation Study of CondNets: I think just encoding all the condition images (garment image, user image, skeleton images) into tokens and then input them into the DiT architecture should be able to deal with the try-on task (the subtask of video editing). The paper should include more explanation of the novelty of the proposed CondNets and add correponding ablation study.

### Questions
1. The fast inference speed (5-second clips at 24FPS and 1152×720resolution) and strong visual results come from the engineering techniques and rather than the proposed CondNets. This paper reads more like a technical report.
2. To foster further advancement in video try-on, it is hoped that the authors could open-source their dataset.
3. The details of CFG distillation. Necessary fomulas and related works should be included the appendix.

### Soundness
2

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
3

### Summary
This paper introduces Dress&Dance, a video diffusion framework that generates high-resolution (1152×720, 24 FPS) virtual try-on videos, allowing users to visualize various garments during complex motions. 
The proposed CondNets architecture effectively integrates multi-modal inputs, while garment-aware target steering ensures precise garment placement. 
Additionally, a synthetic triplet generation and multi-stage training strategy address data and computational challenges. 
Experimental results demonstrate that Dress&Dance surpasses existing open-source and commercial solutions in garment detail preservation, user identity, and motion fidelity.

### Strengths
- The paper is clearly written and easy to follow.
- The proposed scenario is novel and highly relevant to real-world applications.
- The three-phase Garment-Aware Target Steering Guidance is innovative and experimentally validated to be effective.
- Experimental results demonstrate that Dress&Dance achieves state-of-the-art performance, outperforming existing methods.

### Weaknesses
- The proposed scenario requires solving two key tasks: garment transfer and motion synthesis. There are two straightforward baseline approaches: (1) performing garment transfer on the reference image followed by motion transfer; (2) applying motion transfer first and then garment transfer on the resulting video. The authors only compare with the first approach and do not provide analysis or comparison with the second approach.
- The proposed method employs text as a guiding condition. However, in real-world applications, users typically only provide a reference image, garment image, and motion video, without any textual input. Many pose-guided generation tasks do not consider text conditions; the rationale for including text guidance in this work is unclear.
- The framework diagram in Figure 4 is confusing, whereas Figure G.1 is much easier to understand. Since each condition is encoded by forwarding through a model with the same architecture as the main diffusion model, there are concerns about the efficiency of this encoding approach during both training and inference.
- The authors claim that CondNets provide a unified approach for encoding multi-modal control signals, but it is unclear why such a unified encoding is necessary for this specific task. The motivation for this design is not well-justified, nor is its extensibility experimentally validated. Using separate, lightweight encoders for different conditions could be more efficient, especially for conditions with lower complexity.
- In Figure 4, the reason for applying VAE encoding to the noise input and the role of the conditional output are not clearly explained.
- The multi-stage training curriculum is not novel, as it is a standard approach in training multi-resolution generative models.

### Questions
See Weaknesses.

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
Dress&Dance focuses on generating virtual try-on videos given a person image, garment image(s), video showcasing the desired motion, and optionally text. This paper sets out to address the challenges of (1) high-quality input person/garment fidelity, (2) fine-grain motion preservation from input video, and (3) challenging multi-modal, non-pixel-aligned inputs.

Their contributions include:
- SOTA performance on synthesizing videos up to 5s at 1152x720px resolution.
- CondNets: An attention-based conditioning architecture to unify multimodal inputs into a single token sequence.
- Garment-aware target steering: A guidance mechanism to better balance text conditioning with other conditionings
- Synthetic triplet generation: Leveraging pretrained models to generate paired training data.
- Multi-stage training curriculum: Progressively increasing frame rate and spatial resolution during training

### Strengths
- The video results are compelling and clearly preserve sharp garment details.
- The authors are obviously well-versed in recent literature, and leverage many state-of-the-art strategies (e.g. cfg distillation, LoRA finetuning, etc.) and models (e.g. DiT) for their method.
- Extensive SOTA comparisons, including to recent open-source, commercial models, such as Kling Video 1.6 and and Ray2. These show the superiority of Dress&Dance for detail preservation and motion fidelity.
- Although the methods are not the most novel in my opinion, the overall framework achieves state-of-the-art results in video virtual try-on, which is a laudable contribution.

### Weaknesses
- Given similar strategies in numerous papers, such as Matryoshka Diffusion Models and Fashion-VDM, I suggest scaling down claims that multi-stage training curriculum a novel contribution.
- The Method section was missing many key details (see Questions below)
- Classifier-free guidance distillation training is not new, it is utilized in many popular SOTA models, including Flux. See https://arxiv.org/pdf/2210.03142. Missing citation for this or explanation of why it is different.
- The figures/tables could use some work. Specifically, The fonts in most of the figures (e.g. Fig. 4) are quite small. Additionally, Fig. 5 is very, very small and it is hard to see the frames.
- The ethics statement does not mention consent, compensation, or conditions provided by/for the 183 persons hired for their dataset generation.
- At ~1:21 of the video, I observe the buttons re-appearing in the back of the garment. This Janus problem is not mentioned as a limitation of the method.

### Questions
1. Lines 239-241: Along which dimension are the conditional sequences concatenated with the video/text sequences? 
2. How does CondNets differ from past methods that finetune DiT with LoRA for new conditioning inputs? For example, EasyControl, IC-LoRA, various Flux pipelines, etc.
3. Are the VAE encoders fine-tuned or frozen for the new conditioning inputs?
4. Garment-Aware Target Steering Guidance: Why do phase 1 and 2 need to be the layflat garment? Why not just gray-out the non-garment parts of the target frames with decreasing opacity for phases 1-3?
5. 299-300: Which pretrained model(s) is used for synthetic triplet generation? What were the prompts?
6. How are bottoms handled for top try-on (or tops for bottom try-on)? I notice in the video that the bottom is not preserved.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Dress&Dance, a video diffusion framework for video virtual try-on with motion guidance. It introduces CondNets for multi-modal input unification, garment-aware target steering to address text-prior dominance, and synthetic data/training curriculum to solve data scarcity and computational issues.

### Strengths
1. CondNets effectively integrates heterogeneous inputs (garment images, user images, motion videos, text) via attention-based tokenization, overcoming the limitations of pixel-aligned methods like ControlNet for unaligned garment data.

### Weaknesses
1. The authors did not provide quantitative comparisons with existing video virtual try-on methods, such as GPD-VVTO, ViViD, and Tunnel Try-on. Only qualitative comparison is not sufficient to demonstrate the superiority of Dress&Dance against these SOTA methods.
2. GPT-based evaluation lacks transparency and may be biased by prompt design; no user study validates practical usability.
3. Directly comparing Dress&Dance with the baseline methods may not be entirely fair, given that Dress&Dance benefits from a substantial amount of extra training data. This raises uncertainty regarding whether the observed improvements stem from the added data or the unique architecture of Dress&Dance. To ensure a thorough and impartial assessment, it is recommended that the authors re-evaluate their approach using publicly accessible datasets like VVT and ViViD.
4. Understanding the computational resources required to run and train this model is crucial. Given that each condition necessitates a DiT, it appears that the model demands substantial GPU memory.
5. Synthetic triplet generation relies on external image try-on models, introducing potential errors that propagate to training. This dependency also limits generalization if pre-trained models perform poorly on rare garment types.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
