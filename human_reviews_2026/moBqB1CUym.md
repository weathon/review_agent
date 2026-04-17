# Robust and Interpretable Adaptation of Equivariant Materials Foundation Models via Sparsity-promoting Fine-tuning

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Pre-trained materials foundation models, or machine learning interatomic potentials, leverage general physicochemical knowledge to effectively approximate potential energy surfaces. However, they often require domain-specific calibration due to physicochemical diversity as well as mismatches between practical computational settings and those used in constructing the pre-training data. To address this, we propose a sparsity-promoting fine-tuning method that selectively updates model parameters by exploiting the structural properties of E(3)-equivariant materials foundation models. On energy and force prediction tasks across molecular and crystalline benchmarks, our method matches or surpasses full fine-tuning and equivariant low-rank adaptation while updating only ~3 \% of parameters, and in some cases as little as \~0.5 \%. Beyond energy and force calibration, we further demonstrate task generalizability by applying our method to magnetic moment prediction and magnetism-aware total energy modeling. Finally, analysis of sparsity patterns reveals physically interpretable signatures, such as enhanced $d$-orbital contributions in transition metal systems. Overall, our results establish sparsity-promoting fine-tuning as a flexible and interpretable method for domain specialization of equivariant materials foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a method that enables pre-trained equivariant models for materials and molecules to be fine-tuned for new downstream tasks (e.g. different level of theory, different elemental compositions, etc) with very minimal parameters updates (< 5%) while maintaining equivariance. They use soft threshold weight reparameterization (STR) to learn the adaptation weights in a manner that promotes sparseness, so the original weights are only updated when necessary. Results are presented on LAM, MD17, TM-O-Spin, and MP-mag, in nearly all cases the method is reported to be equivalent or better than full fine-tuning.

### Strengths
- The work addresses an important and timely topic.
- A notable strength is the exceptional parameter efficiency: the model adapts to new tasks with surprisingly few parameter updates while maintaining good performance.
- The experimental design is rigorous and well-motivated, with appropriate baselines and comprehensive ablation studies.

### Weaknesses
- While reporting the fraction of parameters updated is great, it doesn’t help understand what the practical benefits of the methods are i.e. how much less memory is required?, how much faster is training than full fine-tuning?, can you store different adopters for different tasks?, etc. It would make the paper much stronger with this information included and emphasized. 
- The paper does not discuss in-depth why this fine-tuning method is consistently more accurate than full fine-tuning. This seems like a counter initiative result and is worth digging into more. 
- In particular, the MD17 benchmark is quite low signal i.e. small tweaks in hyperparameters can change the outcome.

### Questions
1. How much hyperparameter tuning was done for your method vs that done for full fine-tuning or ELoRA?
2. Was the schedule-free AdamW optimizer used for all experiments?
3. The percentage of parameters that get updated is impressively small, is there precedent for this (or examples of this) in other fields? 
4. Are the adaptation weights just added to the original weights? Or is something else happening?
5. I was confused by line 168, does the method only adapt path weights on scalar features (ignoring higher order features) or is it saying that path weights are scalar values? Also, what weights are being adapted in the linear layers?
6. What are the limitations of the method? 
7. Quick and easy fine-tuning is great to have, but as the foundational models get bigger distillation will also be important as well. Does any of the work here carry over to distillation?
8. The S in the table 1. makes me think of stress not sparsity.

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
The paper proposes a sparsity-promoting finetuning method for E(3)-equivariant MLIPs (e.g., MACE). The authors enforce the finetuned weights ΔW as a sparse format by using a Soft Threshold Reparameterization (STR) method, which only updates a small subset of interaction paths. This method can preserve equivariance during adaptation. Experiments on rMD17 (molecules), LAM subsets (crystals), and two magnetic datasets (TM-O-Spin, MP-mag) show that this sparse fine-tuning method can match or outperform full updating and ELoRA while only 0.5% to 3% of parameters are updated.

### Strengths
Employing STR with a learnable threshold per layer and decoupled optimization for ΔW vs. τ is simple, and the implementation appears lightweight.
Experimental Results: This paper covers molecules, multiple crystal subsets, and magnetic datasets. The results show that both low and high sparsity scenarios (meaning very few updated parameters), sparse finetuning gets lower error than full finetuning and ELoRA.

### Weaknesses
Evaluation: Sparse training is emphasized, but the training/inference time speed/footprint gains (if any) aren’t measured (e.g., wall-clock per step, peak memory usage). In Eq. (2), ΔW is initialized from a normal distribution and then sparsified via STR. I suspect this operation will not reduce training time and memory. Please provide memory profiles vs full finetuning and ELoRA.
Limited Benchmarks: The experiments mainly focus on MACE. It would be better to include Nequip-OAM-L to demonstrate the effectiveness of the sparse finetuning method.

### Questions
On the reproducibility of rMD17 dataset: In MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields (Table 1), the aspirin setting reports Etot=2.2 meV (original metric), whereas Table 1 in the author’s paper reports 0.6 meV/atom. Please provide a MACE-aligned reproduction (mean±std over seeds) and, ideally, attach the exact training config/script in the rebuttal for community verification.

The baseline results appear inconsistent with those reported in prior work. For instance, in the case of aspirin, MACE [1] reports errors of 2.2 meV for energy and 6.6 meV/Å for forces when trained from scratch. In contrast, this paper reports 0.60 meV/atom (equivalent to ~12.6 meV per molecule) and 25.55 meV/Å. Such a significant discrepancy raises the question of whether the baseline models in this study were suboptimally configured or tuned. It is recommended that the authors either realign their baselines with established reference values or provide a clear explanation for these differences.

### Soundness
2

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
2

### Summary
This paper proposes a sparsity-promoting fine-tuning method for equivariant materials foundation models, using Soft Threshold Weight Reparameterization (STR) to reduce the number of updated parameters. On molecular and crystalline benchmarks, it updates only a small fraction of parameters while matching or surpassing full fine-tuning.

### Strengths
+Strong fine-tuning accuracy across molecules, inorganic crystals, and magnetic tasks.
+Extends force-field foundation models to magnetic systems for magnetic moment prediction.
+Provides a novel analysis linking valence electronic structure with sparse update patterns.

### Weaknesses
-Limited novelty: mainly applying STR to equivariant models.
-Experimental results are not aligned with prior work and inconsistent across sections.
-The interpretability analysis is not fully convincing.

### Questions
1. STR is applied to \Delta W rather than W (Sec. 3.2). Does this double the number of parameters during training? Without structured sparsity, are these computations still dense?
2. In Eq. (3), the mask uses \Delta W > 0, which blocks gradients for negative weights. Is this a typo or an intentional design choice?
3. The definition of sparsity is confusing. Lines 316-317 state “Total sparsity measures the fraction of updated parameters,” implying that higher sparsity means more updates. Yet Table 1 shows 80-100\% “sparsity,” while the abstract claims only 0.5-3\% updated. Please unify the definition.
4. Baselines are misaligned with prior work. Taking aspirin as an example, MACE [1] reports 2.2 meV (energy) and 6.6 meV/Å (force) from scratch, while this paper reports 0.60 meV/atom (12.6 meV) and 25.55 meV/Å. Does this indicate suboptimal tuning for baselines? Please align baselines or explain the discrepancies.
5. TM-O-Spin: Appendix Fig. I shows force MAE ≥ 70 meV/Å in ablations, but Table 2 reports 48.75 meV/Å for Ours (L). Why the mismatch?
6. Interpretability: Why does ELoRA show updates for elements outside the training set? With one-hot element embeddings in MACE, absent elements should not activate or receive gradients. Official MACE implementations typically remove such parameters before fine-tuning.
[1] Batatia, I., Kovacs, D. P., Simm, G., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in neural information processing systems, 35, 11423-11436.

### Soundness
2

### Presentation
3

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
The paper proposes sparsity-promoting fine-tuning for E(3)-equivariant architectures. The sparsity is achieved by introducing learnable parameters tau that control sparsity. The method is applied to the MACE architecture and benchmarked on the Inorganic crystals, Revised MD17, TM-O-Spin, and MP-mag datasets. For these datasets, the accuracies of the non-fine-tuned model, the model fitted from scratch, the fully fine-tuned model, the model fine-tuned by ELoRA, and the model fine-tuned by the proposed methods with two sparsity levels are presented. Additionally, the applicability of the presented approach for model's interpretation was studied.

### Strengths
The method produces the most accurate models compared to other approaches most of the time. Furthermore, the method is competitive with very high reported sparsity. 

The benchmarking setups are diverse, including small molecules, crystals, and extension to other targets over energies and forces, along with magnetic degrees of freedom. 

The method helps with the interpretability of fine-tuned models.

### Weaknesses
It seems that the method doesn't reduce computational cost and memory requirements during finetuning as the achieved sparsity would suggest. During training, it maintains the coefficients tau, which, to the best of my understanding, mirror each of the dense parameters. So, their total number equals the number of parameters in all layers that undergo finetuning. 

The equivariance is preserved not by incorporating the equivariance constraints into the finetuning method itself, but instead by applying it only to scalar path weights of the model. In other words, any finetuning technique applied to these layers would preserve equivariants. Therefore, I don't think that the positioning of the method for equivariant models, e.g., in the title of the paper, is fully justified.

### Questions
Is it correct that the memory requirements match those of the full finetuning because of the need to store dense tensors of the tau coefficients? 

Could you provide practical finetuning times and memory requirements for full finetuning, ELoRA, and your method for the reported experiments?

Is it correct that the method is also equally applicable for invariant and unconstrained architectures?

### Soundness
3

### Presentation
3

### Contribution
3
