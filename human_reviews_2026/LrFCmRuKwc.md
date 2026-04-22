# Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 6, 4, 4

## Abstract
CLIP has become a cornerstone of multimodal representation learning, yet improving its performance typically requires a prohibitively costly process of training from scratch on billions of samples. We ask a different question: *Can we improve the performance of open-weight CLIP models across various downstream tasks using only existing self-supervised datasets?* Unlike supervised fine-tuning, which adapts a pretrained model to a single downstream task, our setting seeks to improve general performance across various tasks.  However, as both our experiments and prior studies reveal, simply applying standard training protocols starting from an open-weight CLIP model often fails, leading to performance degradation.  In this paper, we introduce **TuneCLIP**, a self-supervised fine-tuning framework that overcomes the performance degradation. TuneCLIP has two key components: (1) a warm-up stage of recovering optimization statistics to reduce cold-start bias, inspired by theoretical analysis, and (2) a fine-tuning stage of optimizing a new contrastive loss to mitigate the penalization on false negative pairs. Our extensive experiments show that TuneCLIP consistently improves performance across model architectures and scales. Notably, it elevates leading open-weight models like SigLIP (ViT-B/16), achieving gains of up to +2.5\% on ImageNet and related out-of-distribution benchmarks, and +1.2\% on the highly competitive DataComp benchmark, setting a new strong baseline for efficient post-pretraining adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces TuneCLIP, a two-stage self-supervised fine-tuning framework that boosts the performance of existing open-weight CLIP models by first addressing cold-start bias through an Optimizer Statistics Recovery (OSR) warm-up phase, and then mitigating the impact of false negatives during fine-tuning with a novel Hinged Global Contrastive Loss (HGCL). By recovering accurate optimizer states before training and then using a margin-based loss to avoid over-penalizing semantically similar pairs, TuneCLIP consistently improves general-purpose capabilities across classification, retrieval, and other benchmarks, establishing an effective and compute-efficient method for enhancing pre-trained foundation models.

### Strengths
1.	The proposed self-supervised fine-tuning task for pre-trained models is novel.
2.	The OSR method is well-motivated, novel, and interesting.
3.	The experiments are thorough and comprehensive and the results are strong and impressive.
4.	The paper provides theoretical analysis for the observed phenomena, which further substantiates the underlying causes.
5.	The paper is well-written and clear.

### Weaknesses
1.	The proposed HGCL appears to adopt an overly simplified approach to managing false negatives.
2.	The paper fails to sufficiently compare the merits of applying OSR versus directly resuming from the optimizer state provided by a pre-trained model.
3.	The effectiveness of OSR has only been validated on specific model categories (e.g., vision-language models), and its generalizability remains unproven.

### Questions
1.	While a simple thresholding strategy might be beneficial for retrieval tasks, could it concurrently undermine the performance of discrimination-based tasks like classification?
2.	When a pre-trained model provides its optimizer state checkpoint, which approach is preferable: directly resuming from this state or applying OSR? Moreover, can OSR further enhance performance when initialized from the pre-trained optimizer state?
3.	Is OSR also applicable to pre-trained models beyond vision-language models, such as self-supervised models like DINO?

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
3

### Summary
This paper identifies two critical failure modes in self-supervised fine-tuning (SSFT) of open-weight CLIP models: an optimization cold-start bias from uninitialized optimizer statistics, and over-penalization due to false negatives in large web corpora. To address these problems, the authors propose TuneCLIP, a two-stage self-supervised fine-tuning (SSFT) recipe. Empirically, TuneCLIP consistently demonstrates improved zero-shot ImageNet classification, retrieval performance, and DataComp mean scores across multiple CLIP checkpoints compared to existing baselines.

### Strengths
1. **Clear practical problem with simple remedy:** The cold-start observation is concrete and likely familiar to practitioners; OSR is simple and easy to implement (run updates with frozen weights). The intuition is compelling and backed by analysis.
2. **Thorough experiments across models and data scales:** Results include multiple model checkpoints, DFN-12M/60M fine-tuning, ImageNet variants, retrieval (MSCOCO/Flickr30k), and the DataComp 38-dataset suite. Ablations isolate the contribution of transferred statistics $(m_t, v_t, u_t)$.
3. **Grounded theoretical support:** The paper provides a comprehensive theoretical analysis that connects initial-stat errors (m_0, u_0) to optimization difficulty, which motivates OSR. That makes the contribution more principled than an empirical trick.

### Weaknesses
1. **Lack of practical GPU-cost analysis**: OSR requires running several epochs (paper reports E=5) over the full fine-tuning corpus to estimate statistics. For DFN-12M/60M, this is non-trivial. The paper lacks a clear breakdown of wall-clock cost (GPU-hours) of OSR + TuneCLIP versus, e.g., a baseline “just run FastCLIP for 5 epochs”. Provide explicit compute numbers (GPU-hours, FLOPs), and a cost/benefit curve (OSR epochs vs gain).

2. **More baselines and alternative fixes:** The paper compares against OpenCLIP and FastCLIP, but it would be stronger to include simple, realistic alternatives that researchers might try to mitigate cold-start bias, such as: 1) Using SGD with momentum for a few iterations instead of Adam; 2) Warm-starting with gradient estimates computed from large batches or model ensembling.
Show that OSR outperforms these cheaper fixes or provide guidance when those suffice.

3. **Data bias and DFN choice:** DFN is a filtered web corpus; authors should discuss whether TuneCLIP’s gains rely on properties of DFN (filtering quality). Would the method behave differently on noisier corpora? An experiment fine-tuning on an unfiltered subset would be informative.

### Questions
N/A

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
The paper introduces TuneCLIP, a self-supervised fine-tuning framework designed to mitigate the performance degradation commonly observed when adapting pretrained CLIP models. The authors provide a systematic analysis of this degradation and propose two complementary components: Optimizer Statistics Recovery (OSR), a warm-up stage that restores optimizer states to alleviate cold-start bias, and Hinged Global Contrastive Loss (HGCL), a fine-tuning objective that reduces the impact of false negatives. Extensive experiments across multiple CLIP variants demonstrate that TuneCLIP consistently improves performance over baseline fine-tuning methods.

### Strengths
The paper offers a novel and practical perspective on improving CLIP models by focusing on efficient post-pretraining enhancement rather than costly retraining. It systematically analyzes the causes of performance degradation during early fine-tuning and provides clear theoretical justification for the proposed solution. The two-stage optimization framework is conceptually sound and empirically validated through extensive experiments across multiple CLIP architectures.

### Weaknesses
The experiments primarily focus on single-object datasets, limiting the evaluation of TuneCLIP’s effectiveness on more complex, multi-object benchmarks such as COCO. All results are reported on ViT-B/16 models, making it difficult to assess scalability across larger or smaller architectures. Additionally, while the method is more efficient than full pretraining, the added warm-up stage introduces extra computation, and the paper does not quantify the actual wall-clock or resource savings.

### Questions
1. It would be valuable to evaluate TuneCLIP on multi-object retrieval benchmarks such as COCO to better assess its generalization to more complex visual scenes.
2. How well does the proposed method generalize across different model scales, such as ViT-B/32 or ViT-L/14?
3. Could the authors provide a more detailed comparison of training efficiency, including wall-clock time or computational cost relative to standard fine-tuning baselines?

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
4

### Summary
The paper introduces TuneCLIP, a two-stage self-supervised fine-tuning framework for improving pre-trained CLIP models without expensive retraining from scratch. The method addresses two key challenges: (1) cold-start bias during fine-tuning initialization through Optimizer Statistics Recovery (OSR), and (2) over-penalization of false negatives through a Hinged Global Contrastive Loss (HGCL). The authors demonstrate consistent improvements across multiple CLIP architectures on ImageNet variants, retrieval benchmarks, and the DataComp benchmark, achieving gains of up to +2.5% on ImageNet and +1.2% on DataComp for SigLIP ViT-B/16.

### Strengths
- Addresses the important problem of improving existing CLIP models without expensive retraining
-  Provides formal analysis of cold-start bias with convergence guarantees
- Tests across multiple architectures (OpenAI, SigLIP, LAION) and diverse benchmarks
- The two-stage approach is well-motivated with good empirical evidence of the problems being solved

### Weaknesses
- Both OSR and HGCL are relatively straightforward modifications of existing techniques. OSR is essentially warmup with frozen parameters, while HGCL applies a standard hinge loss variant.
-  For SigLIP ViT-B/16, improvements are only +0.11% on retrieval and +1.15% on DataComp, raising questions about practical significance.
- No comparison with other fine-tuning strategies like LoRA or prompt tuning
- Limited ablation on hyperparameters (only margin m is studied)

### Questions
- How sensitive is the method to the margin parameter m? The paper shows m=0.1 works well, but how was this determined? What happens with adaptive margin selection?
- All experiments use only 5 epochs. What happens with longer fine-tuning? Does the advantage persist or diminish?
- What is the computational cost of OSR? How does the total training time compare to standard fine-tuning approaches?

### Soundness
3

### Presentation
3

### Contribution
2
