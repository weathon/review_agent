# Learning Neural Representations From Publicly Available Model Hubs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
The weights of neural networks have emerged as a novel data modality, giving rise to the field of weight space learning. 
A central challenge in this area is that learning meaningful representations of weights typically requires large, carefully constructed collections of trained models, typically referred to as model zoos. These model zoos are often trained ad-hoc, requiring large computational resources, constraining the learned weight space representations in scale and flexibility. 
In this work, we drop this requirement by training a weight space learning backbone on arbitrary models downloaded from large, unstructured model repositories such as Hugging Face. Unlike curated model zoos, these repositories contain highly heterogeneous models: they vary in architecture and dataset, and are largely undocumented. To address the methodological challenges posed by such heterogeneity, we propose a new weight space backbone designed to handle unstructured model populations. We demonstrate that weight space representations trained on models from Hugging Face achieve strong performance, often outperforming backbones trained on laboratory-generated model zoos. 
Finally, we show that the diversity of the model weights in our training set allows our weight space model to generalize to unseen data modalities. By demonstrating that high-quality weight space representations can be learned in the wild, we show that curated model zoos are not indispensable, thereby overcoming a strong limitation currently faced by the weight space learning community.  Code, pre-trained weights, and model collections can be found on redacted.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a new approach for training encoder-decoder based weight-space algorithms. Instead of training on lab curated models the authors propose to train on publily available models from huggingface with heterogeneous architectures and tasks. They make adjustments to the SANE architecture for learning on large scale and mixed architectures by adding a loss normalization, modified tokenization and sinusoidal positional encoding. The experimental section tests the new approach against SANE on several weight generation tasks.

### Strengths
- The paper targets an important and emerging task of learning from a collection of model weights to generalize to newly created models. 

- The idea of learning from heterogeneous model collections at once is interesting.

- The paper is clearly written and easy to follow.

### Weaknesses
- **Experimental design and results.** 
  To my understanding of the experimental section, some of the empirical findings are odd. Specifically, I am having trouble to understand how the model samples reasonable classifiers for tasks without knowing for which downstream task the sampled model is aimed for. To give an example, in Tab. 2, a metnetworkd is trained on the HF collection have seen models from variety of tasks (ImageNet, CIFAR10, CIFAR100). But, when sampling a new network, the metanetwork does not know if the sampled calssifier would be used for ImageNet, CIFAR10 or CIFAR100. Hence, since there is not additional fine-tuning, how can the metanetwork sample reasonable calssifiers for all tasks if the sampling process stays the same? This also affects the other training collections in Tab. 2 which are also composed of models trained for differnt tasks. 

  Moreover, this issue is even more prominent in the experiments of Fig. 2, where there are more downstream tasks including ones that have not been seen by the training models at all. For example, consider the SANE baseline which was trained only on the CIFAR10 and CIFAR100 model zoos. No model in its training set have been trained on the EuroSAT dataset. So, how can it generate accurate classifiers for EuroSAT (without further fine-tuning) if it doesnt even know the EuroSAT classes order (which can be chosen arbitrarily)?

  Could the authors please elaborate on these issues? If I misunderstood some parts of the experimental design please let me know so I can reconsider.

- **Generalization or memorization.** A recent work by Zeng et al. [1] demonstrated that weight generation experiments should be carefully examined, as metanetworks are prone to memorization instead of generalization. Therefore, to estimate the merit of the proposed approach for weight generation, its generated networks should be compared to the best pre-trained network in its training collection. For example, in Tab. 1 an additional baseline should be the best networks for CIFAR10, CIFAR100 and TIN from the training datasets. I believe this affects most of the generative results presented in the main menuscript including Tab. 1 as well as Fig. 3 where these baseline networks might provide better initialization for further fine-tuning. 

- **Differences from SANE.** The modifications of the proposed approach over SANE seem relatively small. Tab. 1 also shows that in some cases these modifications can decrease performance when applied to SANE.


[1] Zeng, Boya, et al. "Generative Modeling of Weights: Generalization or Memorization?." arXiv preprint arXiv:2506.07998 (2025).

### Questions
- In Fig. 4 did the authors try to initialize the GPT model weights from randomly selected weight matrices from the training data?

### Soundness
1

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
This paper proposes to use HuggingFace (HF) as a source of trained neural networks for weight space learning (WSL). Previous papers had previously designed model zoos for WSL. To do this, the paper proposes a new method to train a shared weight space representation from arbitrary models downloaded from HF, using a novel method that is capable of learning from models with arbitrary architectures.

### Strengths
Strengths of this paper include: 

**(S1)**: This paper proposes to leverage heterogeneous datasets of pre-trained models instead of curated model zoos, and proposes a new dataset curated from HuggingFace

**(S2)**:  This paper also proposes a new method to adapt existing works (i.e. SANE) to the challenge of heterogeneous model zoos using a new tokenization method and new normalization scheme

**(S3)**: The paper is well-written and clear, with many of the details of the method provided

### Weaknesses
**Major Weaknesses**

**(W1) Clarity**: The paper could be more clear in the following areas:
1. **Single weight space representation**: does this mean that there is a shared representation space for all models, regardless of architecture/training dataset, but each model has a unique representation in this space? Or does this mean that multiple unique models share the same representation?
2. **Evaluation protocol**: I think that the paper could be made more self-contained by putting more details on both the experimental procedure and evaluation and how models were generated in the main text of the paper.

**(W2) Limited diversity of experiments**: The experiments are limited to examining only the accuracy of generated models. 

**(W3) Claim of generalization to new modalities/OOD tasks**: I understand that it may be difficult to validate how all the different HF models were trained if the details weren’t provided, but I don’t think it’s fair to make this claim without checking to see if any of the models were trained with language embeddings or only training with models whose training protocol is clearly stated, such as curated model zoos. Currently, there is no baseline showing that a lab-produced model zoo does not also have this behavior. Also, there are no comparisons to other weight-initialization methods. 

**(W4) KDE bandwidth**: The influence of the KDE bandwidth parameter for model generation (Appendix B.2.2) is not examined, and the values of this parameter are not reported in the paper. 

**Minor Weaknesses**
Typos:
1. Line 220: “with a focus processing” -> “with a focus on processing”
2. Table 2: At the intersection of the “all” row and the “Num models” columns, it seems that it should be 2666 and not 2000?

### Questions
**(Q1)**: Random Fourier Features (RFF) positional encodings (PE) do not require additional trainable parameters and in many cases are better than sinusoidal positional encodings. Would they work better in this case as well?

**(Q2)**: Using MLN instead of LWLN and sinusoidal PE instead of learnable PE leads to lower performance on CIFAR-10, is there a reason why this is?

**(Q3)**: Is “single weight space representation” synonymous with having a shared representation space among all models, regardless of architecture/training dataset?

**(Q4)**: For each dataset, how does the accuracy of generated models compare to the accuracy of the HuggingFace models they were trained on?

**(Q5)**: What are the performance for curated vision model zoos on the GPT-2 task (e.g. Section 4.4.1, Figure 4)

**(Q6)**: The performance of generated models is much lower than what you would expect from just training a model on, say, CIFAR-10. What is the reason for this? What are the failure modes of these models?

**(Q7)**: Is there any structure to the latent representation space of the proposed model? Are the latent spaces of this model vs SANE meaningfully different? Do model with similar architectures or similar training datasets get mapped to similar regions of latent space?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors propose to learn weight-space representations directly from arbitrary models scraped from large public hubs (Hugging Face) rather than from curated, homogeneous “model zoos.” Concretely, the authors adapt a SANE-style encoder–decoder transformer to be architecture- and dataset-agnostic by introducing 

(i) Masked Loss Normalization (MLN) that normalizes reconstruction loss per token at runtime while respecting token masks, 

(ii) a dense tokenization scheme that reduces padding across heterogeneous layers/architectures, and 

(iii) sinusoidal positional encodings to avoid massive learned position tables that would otherwise explode with long weight sequences. 

Trained on ~2,000 HF models (≈171B parameters total), a single backbone is used for multiple downstream tasks, mainly generative initialization of diverse architectures/datasets. It often outperforms backbones trained on lab model-zoos, and shows OOD modality generalization by initializing GPT-2 better than training from scratch after short finetuning. The authors provide ablations on MLN, tokenization, positional encodings, and training-set composition, plus comparisons against SANE baselines.

### Strengths
- Heterogeneous-hub training at scale. Training a single model on ~2k HF models (≈171B params) while remaining architecture and dataset agnostic is a practical contribution that lowers the barrier to WSL research. 

- Method tweaks that matter. MLN effectively replaces per-layer normalization and is demonstrated to stabilize learning and improve generation on harder datasets; dense tokenization and sinusoidal PEs cut padding/params and enable scaling to larger, varied architectures. 

- Broad generative evaluation. Results span multiple datasets (CIFAR-10/100, Tiny-ImageNet, SVHN, EuroSAT) and many architectures (ResNet/ConvNeXt/EfficientNet/ViT/Swin/BeiT/GPT-2), generally showing faster/better finetuning vs. scratch, and gains over SANE zoo-trained baselines in most cases. 

- Ablations and implementation detail. Clear hyperparameters, runtime footprint, and ablations (MLN vs. LWLN, tokenization, PEs, dataset fraction and composition).

### Weaknesses
- Zero-shot quality seems weak; benefits hinge on finetuning. For many larger models, initial accuracy is near random, with benefits materializing only after a few epochs. This limits plug-and-play use compared to stronger weight generators or data-driven pretraining. I would recommend the authors to report more stringent “few-step” regimes and to compare sample-efficiency curves, which can help strengthen the paper further. 

- The reported results hider limited discriminative validation. Encoder embeddings trained on HF are worse for property prediction than zoo-trained baselines. The authors hypothesize higher variance and non-linearity but do not probe fixes (nonlinear heads, better pooling, contrastive variants). This somewhat weakens the “single backbone” versatility claim. 

- Data curation and licensing clarity. The download protocol excludes remote-code models and malformed checkpoints, but licensing/provenance and potential duplication/fine-tune lineage (e.g., many HF models fine-tune of the same base) aren’t deeply analyzed. These could bias and hurt the learned representation while raising reuse concerns. (The Discussion notes possible selection bias but does not quantify it.)

- Comparative baselines. The main comparison is with respect to SANE (under the authors modifications). It would strengthen the case to include diffusion-based and/or graph/functional backbones (even on smaller subsets), or to report hybrid (e.g., diffusion in latent-Z) to show trade-offs. The current evidence supports feasibility but not superiority across the method classes. 

- Metrics/uncertainty. Several plots lack error bars. The tables sometimes report ± but figures do not, making it hard to assess variance across seeds and models. More consistent reporting would help.

### Questions
- MLN vs. alternatives. Did the authors try token-wise standardization in the forward path (like a learnable norm layer) instead of normalizing only in the loss? Any effect on stability/quality? Would be nice to see/include a short ablation. 

- Are there any dense tokenization trade-off here?. Dense tokens mix channels across layers. Do the authors observe any loss of locality that harms small models? Would be nice that the authors report sparsity vs. quality vs. memory curves contrasting dense vs. sparse across 2–3 architectures? 

- Zero-shot vs. few-shot. How do results change under N optimizer steps (N∈{0,10,50,100}) with fixed batch size across all architectures? A unified “steps-to-X%” metric would clarify sample-efficiency gains over scratch. 

- Encoder utility. In the reported results, the HF-trained encoder underperforms for property prediction. Have the authors tried non-linear heads, token-attention pooling, or contrastive objectives tuned for discriminative tasks? Would be nice that the authors elaborate/explain on that more (ideally would be nice that they provide a small table).

- Cross-modality support. For GPT-2, could the authors report exact init metrics (loss/perplexity at step 0) and early convergence deltas vs. scratch across 3 seeds? Also, could the authors explan more about do CLIP-like weights in the HF set confound the “vision-only” claim?

- Dataset hygiene. Can you quantify duplication (identical or near-identical checkpoints), license coverage, and model lineage (fine-tunes of the same base) in your 2k-model set? A brief audit would help assess bias and legal reusability.

- Backbone scale. Seems like HF-Large (\~900M) improves results over HF-Small (\~456M). Could the authors sat more about where are the diminishing returns? Any evidence that position encodings or window size become the bottleneck next?

### Soundness
3

### Presentation
3

### Contribution
3
