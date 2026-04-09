====================================================================================================
SPECIFIC WEAKNESS PATTERNS FROM HUMAN REVIEWS FOR SFT PAPER
====================================================================================================


## HYPERPARAMETER TUNING FAIRNESS
Total instances: 8
Description: Concerns about hyperparameter tuning fairness
Applies to SFT paper: SFT paper must ensure learning rate and training settings are fairly optimized for all method comparisons
----------------------------------------------------------------------------------------------------

### Example 1
**Paper ID:** XCugWIuHR8
**Title:** Convex Distillation: Efficient Compression of Deep Networks via Convex Optimization
**Category:** fine_tuning_memorization
**Exact Quote:** "Experimental results are not satisfactory to justify the efficacy of the proposed methods. Only small datasets are included. Meanwhile, the ResNet18 baseline seems not well tuned (with low acc less than 90%)."

### Example 2
**Paper ID:** E4Fk3YuG56
**Title:** Cut Your Losses in Large-Vocabulary Language Models
**Category:** transformer_analysis
**Exact Quote:** "I think there could have been additional experiments to explore how CCE performs relative to baselines as different hyperparameters vary (e.g., relative size of vocabulary vs sequence length vs. hidden dim, sparsity of S, etc.)."

### Example 3
**Paper ID:** cojJ2s1e35
**Title:** Inter-Environmental World Modeling for Continuous and Compositional Dynamics
**Category:** generalization
**Exact Quote:** "No discussion of hyperparameter selection"

### Example 4
**Paper ID:** RBp0x7rkMO
**Title:** Vector Grimoire: Codebook-based Shape Generation under Raster Image Supervision
**Category:** transformer_analysis
**Exact Quote:** "Despite the claim about lack of vector data, datasets like SketchGraph and DeepCAD offer more than enough data in vector/discrete formats, and datasets like Quick, Draw! (as cited briefly in this paper as well) already has a vector representation. Moreover, there are so many off the self approachs for vectorizing drawings (e.g. Deep Sketch Vectorization via Implicit Surface Extraction). With the vast abundance of such data, it just doesn't make sense to use the pseudo "SVG" representation in the paper, which has few of the benefits of actual vector drawings."

### Example 5
**Paper ID:** iXbUquaWbl
**Title:** End-to-end Learning of Gaussian Mixture Priors for Diffusion Sampler
**Category:** parameter_efficient
**Exact Quote:** "As acknowledged in the paper, there is no clear criterion for determining the appropriate number of mixture components needed for optimal performance. Furthermore, IMR relies on heuristic rules, such as a fixed number of iterations, to add new components, introducing additional hyperparameter tuning requirements for model optimization."


## LACK THEORETICAL JUSTIFICATION
Total instances: 4
Description: Lack of theoretical justification
Applies to SFT paper: SFT paper should provide theoretical explanations for why FNNs vs attention differ in memorization
----------------------------------------------------------------------------------------------------

### Example 1
**Paper ID:** XCugWIuHR8
**Title:** Convex Distillation: Efficient Compression of Deep Networks via Convex Optimization
**Category:** fine_tuning_memorization
**Exact Quote:** "Experimental results are not satisfactory to justify the efficacy of the proposed methods. Only small datasets are included. Meanwhile, the ResNet18 baseline seems not well tuned (with low acc less than 90%)."

### Example 2
**Paper ID:** 7ZToWPWUlO
**Title:** Solving Normalized Cut Problem with Constrained Action Space
**Category:** sft_vs_rl
**Exact Quote:** "it is not explained why the grouping in inner circles and outer wedges is a good modeling. Is it a pattern observable in other city map grouping algorithms? Does this pattern apply to all cities? There is a drop in writing quality in section 5, which raised some doubts:"

### Example 3
**Paper ID:** FbZSZEIkEU
**Title:** Adaptive Circuit Behavior and Generalization in Mechanistic Interpretability
**Category:** generalization
**Exact Quote:** "The S2 hacking hypothesis is quite vague and the author do not present any deep understanding that would explain the mechanisms by which certain attention heads pay extra attention to the S2 token."

### Example 4
**Paper ID:** hNjCVVm0EQ
**Title:** MamKO: Mamba-based Koopman operator for modeling and predictive control
**Category:** generalization
**Exact Quote:** "In the introduction, the computational time of other koopman based approaches is discussed. However, there is no proper discussion/evaluation/theory on the training/inference time with the proposed framework."


## LIMITED EVALUATION BENCHMARKS
Total instances: 4
Description: Limited evaluation on benchmarks
Applies to SFT paper: SFT paper needs to evaluate on diverse tasks and settings to demonstrate OOD generalization claims
----------------------------------------------------------------------------------------------------

### Example 1
**Paper ID:** XCugWIuHR8
**Title:** Convex Distillation: Efficient Compression of Deep Networks via Convex Optimization
**Category:** fine_tuning_memorization
**Exact Quote:** "Experimental results are not satisfactory to justify the efficacy of the proposed methods. Only small datasets are included. Meanwhile, the ResNet18 baseline seems not well tuned (with low acc less than 90%)."

### Example 2
**Paper ID:** CpQegoH1Fn
**Title:** Human-in-the-loop Neural Networks: Human Knowledge Infusion
**Category:** transformer_analysis
**Exact Quote:** "Poor evaluation with limited experiments and even more limited comparisons. The proposed method is validated only in one medical dataset. I would suggest to test it against other datasets too. Regarding the comparisons I understand that this is more difficult but you need to figure out a good ablation study at least."

### Example 3
**Paper ID:** RBp0x7rkMO
**Title:** Vector Grimoire: Codebook-based Shape Generation under Raster Image Supervision
**Category:** transformer_analysis
**Exact Quote:** "both the problem studied and the technique used are far more well studied than what is depicted in this paper. I am aware of many approaches that are similar in terms of core ideas (autoregressive generation of 2D/3D content represented as a set of discrete instructions/parameters), many datasets that are far more complex than what is used in this paper, as well as multiple works that seem to provide better solutions to this problem (and not compared):"

### Example 4
**Paper ID:** XwibrZ9MHG
**Title:** PokeFlex: A Real-World Dataset of Deformable Objects for Robotics
**Category:** transformer_analysis
**Exact Quote:** "Controlled Interaction Protocols: The poking and dropping protocols are specific and may not capture the full spectrum of possible deformations in less controlled or more complex environments. Additional manipulation actions could enrich the dataset."


## MISSING BASELINES COMPARISONS
Total instances: 4
Description: Missing baselines or incomplete method comparisons
Applies to SFT paper: SFT paper should compare against all relevant SFT variants, RL methods, and selective fine-tuning approaches
----------------------------------------------------------------------------------------------------

### Example 1
**Paper ID:** xsELpEPn4A
**Title:** JudgeLM: Fine-tuned Large Language Models are Scalable Judges
**Category:** fine_tuning_memorization
**Exact Quote:** "The authors did not provide clear evidence that this model is able to maintain good performance across tasks not in the training set. I suspect that the comparison to the PandaLM test set is showing this to some extent, but I did not see any prose on *how* these two datasets differ. What tasks are seen in PandaLM that arent seen in the JudgeLM dataset? If the authors can show that the task distribution is significantly different from the training set I would be satisfied"

### Example 2
**Paper ID:** ih3BJmIZbC
**Title:** Representational Similarity via Interpretable Visual Concepts
**Category:** transformer_analysis
**Exact Quote:** "but I would rather have seen more validation of the usefulness of the tool in this paper. It would be wonderful if the paper could use the tool to tell us something about neural representations we didn't know before, like, say, that resnets focus more on background elements than transformers, or, ideally, something deeper than that. To make this super compelling, I would like a demonstration that this new way of identifying differences has advantages over other ways of identifying representational differences, in a head to head comparison. But I'm not sure how to do that. There are also a few "

### Example 3
**Paper ID:** RBp0x7rkMO
**Title:** Vector Grimoire: Codebook-based Shape Generation under Raster Image Supervision
**Category:** transformer_analysis
**Exact Quote:** "both the problem studied and the technique used are far more well studied than what is depicted in this paper. I am aware of many approaches that are similar in terms of core ideas (autoregressive generation of 2D/3D content represented as a set of discrete instructions/parameters), many datasets that are far more complex than what is used in this paper, as well as multiple works that seem to provide better solutions to this problem (and not compared):"

### Example 4
**Paper ID:** i3f2N3iHl0
**Title:** Adaptive Tensor Attention Networks with Cross-Domain Transfer for Drug-Target Interaction Prediction
**Category:** generalization
**Exact Quote:** "The experiments were far from adequate: there should have been multiple runs (for uncertainty quantification) and more "modern" baselines to compare with."


## NARROW EVALUATION SCOPE
Total instances: 2
Description: Narrow scope of evaluation
Applies to SFT paper: SFT paper evaluates on limited domains; needs broader task variety to support claims about attention/FNN differences
----------------------------------------------------------------------------------------------------

### Example 1
**Paper ID:** 2RNGX3iTr6
**Title:** Tabby: Tabular Adaptation for Language Models
**Category:** transformer_analysis
**Exact Quote:** "applying MOE to a narrow problem (table generation). And the results are not all that strong. * It's not easy from the presentation what exactly do the tasks require, what exactly are the baselines and model variations."

### Example 2
**Paper ID:** CpQegoH1Fn
**Title:** Human-in-the-loop Neural Networks: Human Knowledge Infusion
**Category:** transformer_analysis
**Exact Quote:** "Poor evaluation with limited experiments and even more limited comparisons. The proposed method is validated only in one medical dataset. I would suggest to test it against other datasets too. Regarding the comparisons I understand that this is more difficult but you need to figure out a good ablation study at least."


## UNFAIR EXPERIMENTAL SETUP
Total instances: 2
Description: Unfair or biased experimental setup
Applies to SFT paper: SFT paper may have unfair comparisons if baselines are not optimized equally or use different hyperparameters
----------------------------------------------------------------------------------------------------

### Example 1
**Paper ID:** xsELpEPn4A
**Title:** JudgeLM: Fine-tuned Large Language Models are Scalable Judges
**Category:** fine_tuning_memorization
**Exact Quote:** "While the validation set was manually checked and corrected by the authors, it does still rely on GPT generated outputs. This provides somewhat of an unfair evaluation as JudgeLM is trained on GPT generated judgements as well. Even with the human validation, there is a reasonable chance that if this dataset where annotated by a different LLM and produced different judgements, humans checking responses would also consider them reasonable. An unbiased way of annotating is for humans to provide judgements *without* knowing what the GPT judgement is. If the agreement between humans and the GPT jud"

### Example 2
**Paper ID:** FhBT596F1X
**Title:** Learning Equivariant Non-Local Electron Density Functionals
**Category:** generalization
**Exact Quote:** "Since the proposed method requires computing DFT, comparing it to ML force field might not be fair due to the higher computational cost."