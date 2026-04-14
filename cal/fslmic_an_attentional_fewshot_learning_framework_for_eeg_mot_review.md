=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary

FSL-MIC proposes a two-way K-shot learning framework for cross-subject EEG motor imagery (MI) classification, combining a 1D CNN embedding module, a channel-wise self-attention mechanism, and a Relation Network–style relation module. The method is motivated by the practical challenge of collecting large labeled EEG datasets for new users, and is evaluated on BCI Competition IV 2a, 2b, and a newly collected 64-channel dataset under leave-one-subject-out cross-validation. The paper's central claim is that few-shot learning with minimal labeled data from a new subject, combined with cross-subject meta-training, can produce useful BCI classifiers without full-session calibration.

---

## Strengths

- **Practically relevant few-shot evaluation protocol.** The cross-subject leave-one-out meta-learning setup—where K support samples from a previously unseen subject are used at test time—directly mirrors the real-world BCI calibration bottleneck. This framing is appropriate and the experimental protocol (9-fold and 7-fold CV repeated 10 times per condition) is methodologically sound for the stated goal.

- **Multi-dataset validation with varying channel configurations.** Evaluating on BCI 2a (22 channels, no feedback), BCI 2b (3 channels, with neurofeedback), and a newly collected 64-channel dataset meaningfully tests the model's scalability across EEG setups. The consistent shot-count trend (accuracy rising monotonically from 1 to 20 shots across all three datasets) is a reliable empirical observation.

- **FSL meaningfully outperforms its natural comparison point.** Against CNN-attention-Few—the supervised baseline that has access to the same 40 test-subject samples as the 20-shot FSL model—RelationNet-attention achieves higher accuracy across all datasets (e.g., 72.6% vs. 62.8% on BCI 2a, 73.2% vs. 71.3% on BCI 2b, 68.2% vs. 69.2% on the experimental set). This demonstrates the FSL paradigm extracts useful cross-subject knowledge beyond what a directly supervised model trained on the same limited data can achieve.

---

## Weaknesses

### Fatal
None that singlehandedly invalidate the entire claim; however, the combination of Major weaknesses below collectively make this paper significantly below ICLR acceptance bar in its current form.

### Major

- **No comparison with any published FSL baseline or published EEG method on the same benchmarks.** The paper's only baseline is variants of the authors' own CNN architecture. There is no comparison against Prototypical Networks, Matching Networks, MAML, or An et al. (2020/2023) with published numbers. The conclusion states the model "outperforms An et al. (2020)" but provides no numerical entry for that method in Table 1. Without an external reference point, it is impossible to assess whether the performance represents progress or falls behind the existing literature. This is a critical omission for any ICLR submission claiming state-of-the-art or competitive performance.

- **"DA accuracy" metric is never defined.** This metric is reported alongside standard accuracy in every table and appears in the abstract, yet the paper never formally defines what "DA accuracy" means—whether it refers to accuracy when data augmentation is applied during training, at test time, or via test-time augmentation. Without a definition, these numbers cannot be interpreted, reproduced, or compared against other work.

- **The claim of reduced training time is completely unsubstantiated.** Both the abstract and conclusion explicitly list faster training as a key benefit ("reduces training time," "faster training cycles"). No training time measurements appear anywhere in the paper. This claim, as stated, is not supported by any evidence.

- **Deferred results are inappropriate for a submitted paper.** Section 3.2 explicitly states: *"the full results, including data from multiple subjects, will be presented in our next paper."* Promising key results in a future paper and submitting incomplete content is not acceptable for a peer-reviewed venue; the submission should stand alone.

- **Abstract framing is misleading.** The abstract claims the framework "significantly outperforms traditional methods," but CNN-attention-All (fully supervised, all training data) consistently and substantially outperforms RelationNet-attention by 12–17 percentage points across all datasets. While the FSL model appropriately outperforms CNN-attention-Few, claiming the method "significantly outperforms" without acknowledging that the full-supervision baseline dominates misrepresents the results.

### Minor

- **No ablation study.** The paper provides no experiment isolating the contribution of the self-attention module. It is unknown whether the gains over a plain Relation Network come from the attention module, the focal loss, data augmentation, or the overall meta-learning setup. Without at least one ablation (RelationNet without attention), the architectural contribution cannot be validated.

- **Non-standard attention formulation without justification.** The paper uses S = QK^T without the standard 1/√d_k scaling from Vaswani et al. (2017), which it cites. The output O = tanh(M) is also non-standard (standard attention uses softmax-weighted values directly). These deviations may be intentional design choices but are neither acknowledged nor motivated; the absence of scaling can cause training instability for large d_k.

- **Focal loss in a balanced binary classification setting is unexplained.** Focal loss is designed to address class imbalance, but the 2-way setup involves balanced classes. The rationale for using focal loss instead of standard binary cross-entropy or MSE (as in Sung et al., 2018) is not provided.

- **Batch size of 164 is unexplained.** This is an unusual value (not a power of 2, nor an obvious multiple of the number of classes or subjects). No justification is given.

### Tiny

- **Reproducibility placeholder links.** Both the dataset link and code repository are listed as "[xxx]," preventing verification. These must be replaced with anonymized links for a submission.

- **Learning rate decay described ambiguously.** "0.033% with each iteration" is not a standard parameterization; it is unclear whether this is multiplicative, additive, or per-epoch.

---

## Nice-to-Haves

- **Add t-SNE visualization of the embedding space** to verify that the relation module produces separable representations between left- and right-hand MI classes across subjects.
- **Provide attention topography heatmaps across multiple subjects** to validate that the model focuses on sensorimotor areas (C3/C4) rather than noise or artifactual channels.
- **Report inference latency** to substantiate the claim of real-time BCI suitability.
- **Include confusion matrices or per-subject accuracy breakdown** to reveal whether failure cases are driven by specific subjects or class ambiguities.
- **Extend to N > 2 way** classification (e.g., include foot and tongue imagery from BCI 2a) to demonstrate the claimed N-way extensibility.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Baseline construction is unfair"** (Harsh Critic): The critic argues CNN-attention-Few is not a proper few-shot baseline because it also uses all other subjects' training data. However, the RelationNet-attention model likewise uses all other subjects' data during meta-training. The only difference is *how* that data is used (episodic meta-learning vs. supervised fine-tuning). The comparison is valid and appropriately controlled. REMOVED.

- **"Missing prominent supervised baselines from related works"** (Harsh Critic, referring to EEGNet etc.): Per review instructions, claims about missing related works are not included since external references cannot be verified. REMOVED.

- **"FSL protocol contradiction: unseen subject vs. within-subject few-shot"** (Spark Finder): The protocol is internally consistent. The model is meta-trained on other subjects and tested using K support samples drawn from the held-out subject. This is a standard and well-defined cross-subject FSL evaluation. Not a contradiction. REMOVED.

- **"Unfair comparison because baseline has access to more data than the FSL model"** (paraphrase of Harsh Critic): CNN-attention-All uses all data including the full test-subject dataset, which is unavailable to the FSL model at test time. This comparison is intentionally asymmetric in favor of the baseline and demonstrates an upper bound. It does not make the FSL results look better than they are; it is an honest presentation. REMOVED as a standalone weakness (though the misleading abstract framing is retained as Major).

- **"Small subject count N=7 and N=9"** (Harsh Critic): For BCI 2a and 2b, the subject counts are fixed by the public datasets themselves (N=9 each). Criticizing the authors for the size of canonical benchmarks is not meaningful. The N=7 custom dataset is small, but for a self-collected pilot EEG study this is within the norm of the field. REMOVED as a formal weakness; the small custom dataset is acknowledged in context.

- **"No theoretical proof for meta-learning generalization"** (implicit in Harsh Critic): This is an empirical systems paper; demanding theoretical guarantees is not standard for this area. REMOVED.

---

## Novel Insights

The three reviewers collectively surface one noteworthy analytical point beyond the paper's own claims: the appropriate comparison for a few-shot learning paper is the supervised model trained on *the same* number of samples as the support set (CNN-attention-Few), not the fully supervised model trained on all available data. When evaluated this way, the FSL model's advantage is modest but real (particularly on BCI 2a), suggesting that the cross-subject meta-training does convey generalizable relational structure. However, this advantage plateaus at 20 shots and narrows on more complex datasets (e.g., 64-channel experimental set: 68.2% FSL-20 vs. 69.2% CNN-Few), raising the question of whether the FSL paradigm's benefit degrades as channel count and signal complexity increase—an analysis the paper does not perform.

---

## Suggestions

1. **Add published baselines to Table 1**: Include numerical results from An et al. (2020/2023) and at least one standard FSL method (e.g., Prototypical Networks using the same backbone) on BCI 2a and 2b. This is the single most important fix for establishing the paper's contribution.

2. **Formally define "DA accuracy"**: Add a precise one-paragraph definition explaining what transformations are applied, at what stage (train/test), and how it differs from standard accuracy.

3. **Run at least one ablation**: Train RelationNet *without* the attention module (plain Relation Network with the same 1D CNN backbone) and report its accuracy in Table 1. This directly validates the attention module's contribution.

4. **Remove or substantiate the training time claim**: Either provide wall-clock training time comparisons or remove all language about training speed from the abstract, introduction, and conclusion.

5. **Remove the deferred-results language**: Either include the multi-subject attention visualizations (even in the appendix) or remove the statement that they will appear in a future paper.

6. **Justify or correct the attention formulation**: Explicitly acknowledge the missing scaling factor and provide empirical motivation (e.g., show that scaled dot-product attention performed worse on a validation set), or add the scaling to align with standard practice.

---

**Evaluation summary:**
- *Novelty*: Low. The architecture is an incremental combination of Relation Networks, 1D CNNs, and standard self-attention already applied to EEG by prior work (An et al., 2020/2023); the specific contributions over those papers are not clearly differentiated.
- *Technical soundness*: Weak. Key architectural choices deviate from standard practice without justification; the undefined DA metric undermines reproducibility.
- *Empirical support*: Weak. The absence of any external FSL or published EEG baseline makes it impossible to assess competitive performance; ablations are absent.
- *Significance*: Limited in current form. The FSL motivation is genuine, but the empirical results do not establish a compelling case that the proposed approach advances the state of the art.
- *Clarity*: Below standard. The undefined DA metric, deferred results, and placeholder links reflect an incomplete submission.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 1.0, 3.0, 1.0]
Average score: 2.2
Binary outcome: Reject
