# Neuro-MoBRE: Exploring Multi-subject Multi-task Intracranial Decoding via Explicit Heterogeneity Resolving

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Neurophysiological decoding, fundamental to advancing brain-computer interface (BCI) technologies, has significantly benefited from recent advances in deep learning. However, existing decoding approaches largely remain constrained to single-task scenarios and individual subjects, limiting their broader applicability and generalizability. Efforts towards creating large-scale neurophysiological foundation models have shown promise, but continue to struggle with significant challenges due to pervasive data heterogeneity across subjects and decoding tasks. Simply increasing model parameters and dataset size without explicitly addressing this heterogeneity fails to replicate the scaling successes seen in natural language processing. Here, we introduce the Neural Mixture of Brain Regional Experts (Neuro-MoBRE), a general-purpose decoding framework explicitly designed to manage the ubiquitous data heterogeneity in neurophysiological modeling. Neuro-MoBRE incorporates a brain-regional-temporal embedding mechanism combined with a mixture-of-experts approach, assigning neural signals from distinct brain regions to specialized regional experts on a unified embedding basis, thus explicitly resolving both structural and functional heterogeneity. Additionally, our region-masked autoencoding pre-training strategy further enhances representational consistency among subjects, complemented by a task-disentangled information aggregation method tailored to effectively handle task-specific neural variations. Evaluations conducted on intracranial recordings from 11 subjects across five diverse tasks, including complex language decoding and epileptic seizure diagnosis, demonstrate that Neuro-MoBRE surpasses prior art and exhibits robust generalization for zero-shot decoding on unseen subjects.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose Neural Mixture of Brain Regional Experts (Neuro-MoBRE) as a general-purpose neurophysiological decoding framework, which uses intracranial data collected from 11 subjects across five distinct decoding tasks. Neuro-MoBRE incorporates a brain-regional-temporal embedding mechanism within a decoder-only transformer architecture to effectively handle structural heterogeneity. It addresses the critical concern of data heterogeneity and low signal-to-noise ratio (SNR) of neurophysiological signals.

### Strengths
1. The paper concludes two valuable and unresolved challenges: (1) Ubiquitous data heterogeneity in neurophysiological modeling and (2) Semantic vagueness and low signal-to-noise ratio of neurophysiological signals.

2. Neuro-MoBRE outperforms compared methods in two tasks. The authors also conduct ablation study to show the effectiveness of each part of the model.

### Weaknesses
1. Typos: Figure A1 and Table A1 in page 6 link to wrong figure and table.

2. The paper lacks experiments on publicly available datasets. For example, Labram conducts experiments on TUAB and TUEV.

3. While many existing works have achieved sentence-level language decoding (eeg-to-text, fmri-to-text, etc.), the proposed Neuro-MoBRE still only focuses on character-level decoding.

4. The paper lacks a case analysis part to show the detailed content of decoding results.

5. While the authors claim to perform multi-task decoding, the experiment only cover Language decoding task and Epileptic seizure diagnosis tasks, which weakens the effectiveness of the framework.

### Questions
1. I notice that there's no submitted code, which largely affects the potential influence of this paper. Will the authors open-source this framework?

2. In section 3.1, while the order of the framework is Brain-regional-temporal Tokenization, Task-disentangled Information Aggregation, and Brain-regional MoE according to figure 1, why the authors introduce Brain-regional MoE part before Task-disentangled Information Aggregation?

3. Can Neuro-MoBRE outperforms other compared models when the setting is not multi-subject multi-task?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Neuro-MoBRE, a novel modular mixture-of-experts (MoE) framework designed for neural decoding across multiple brain regions and tasks. The method assigns region-specific experts to distinct brain areas and employs a gating mechanism to adaptively integrate their outputs for robust behavioral prediction. The authors evaluate the model on intracranial recordings from 11 subjects across five decoding tasks under both within-subject and cross-subject settings, demonstrating improved within-subject performance over baseline models and the potential for cross-subject generalization.

### Strengths
1. The proposed framework achieved the best within-subject performance across all decoding tasks compared with the baselines.
2. The proposed Neuro-MoBRE framework is conceptually interesting, especially its modular mixture-of-experts (MoE) design that allocates region-specific experts for decoding across multiple brain areas.

### Weaknesses
Major concerns:
1. In the introduction, the authors identify low SNR in non-invasive neurophysiological recordings as one of the main challenges motivating this work (line 70). However, the paper states:
***“To circumvent the limitations posed by low SNRs in non-invasive neurophysiological recordings, we rigorously evaluate Neuro-MoBRE using intracranial data collected from 11 subjects across five distinct decoding tasks.”*** If the primary goal is to enhance robustness to low-SNR conditions, evaluating exclusively on high-SNR intracranial data does not convincingly demonstrate such robustness or the model’s applicability to non-invasive modalities. While using intracranial data as a clean benchmark to establish an upper bound of performance is reasonable, the paper should make this rationale explicit and avoid implying that this setup “circumvents” the low-SNR limitation, as the current experiments do not address or validate performance under low-SNR conditions. The authors could either include experiments using low-SNR modalities (e.g., EEG or MEG) to empirically test robustness, or clearly reframe the motivation to indicate that the intracranial evaluation serves only as a controlled, high-SNR benchmark. Based on the current paper, I don't see the proposed architecture/framework addressing the low SNR issue.

2. The paper claimed they achieve “zero-shot generalization". Typically, zero-shot refers to scenarios involving unseen datasets probably with different demographics, acquisition setups, or task domains. For instance, directly testing the model on unseen data collected from a separate study. The scenario described in the paper appears to involve unseen subjects within the same dataset, which would be more accurately characterized as cross-subject generalization rather than zero-shot generalization?

3. The model seems heavily tailored to the specific dataset (with specific tasks) used in the study, which raises questions about scalability. The use of subject-wise models for RMAE sessions may become computationally prohibitive as the dataset or population size increases. The term “generalization” should therefore be used with caution, especially since no external or cross-dataset experiments are provided.

4. For the unseen subject decoding performance, the paper didn't compare with other baselines.


Minor suggestions:
- Some tables (e.g., Table 3 and 4) are missing standard deviation (std) values, while Tables 1 and 2 include them. Consistent reporting of mean ± std would improve clarity and comparability.
- The description of the proposed method is sometimes unclear, and several components would benefit from improved clarity and organization.

### Questions
1. If I understand correctly, each decoding task is said to have its own set of classification tokens within this multi-task model. How does the model handle new, unseen tasks during inference? Is there a mechanism for extending the task token space without retraining the entire model?
2. When comparing against baselines such as PopT that are also trained on intracranial recordings, are these baselines fine-tuned from their pre-trained weights, or trained from scratch? 
3. Given that prior works such as PopT have already tackled the challenge of inter-subject electrode placement variability, what is the key novelty or improvement introduced by the proposed method in this aspect?
4. What distinguishes the proposed brain-regional MoE from a standard MoE architecture (from the architecture aspect)?
5. Will the dataset be released publicly if the paper is accepted?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Neuro-MoBRE, a novel neural decoding framework designed to explicitly resolve the pervasive data heterogeneity in multi-subject, multi-task intracranial neurophysiological decoding. The framework integrates a brain-regional-temporal embedding mechanism, a Mixture of Brain Regional Experts (BrMoE) module, and a task-disentangled information aggregation mechanism. It is further enhanced by a region-masked autoencoding (RMAE) pre-training strategy to improve generalization. Evaluated on intracranial recordings from 11 subjects across five diverse tasks, including language decoding and seizure diagnosis, Neuro-MoBRE demonstrates superior performance over existing methods.

### Strengths
1. Well-Motivated and Novel Methodology: The paper accurately identifies "data heterogeneity" as a fundamental challenge in neural decoding and proposes a systematic solution. The core ideas, particularly the brain-regional MoE and task-disentangled aggregation, are innovative.

2. Multi-Subject, Multi-Task Modeling: The framework successfully unifies data from multiple subjects and tasks within a single model. Its demonstrated zero-shot generalization to unseen subjects have some advantages with practical value.

3. Rigorous Experimental Design: The evaluation covering challenging tasks like Mandarin phonological decoding (initials, finals, tones) and clinical epilepsy diagnosis (detection and prediction) using real sEEG data, making the results credible.

4. Modular Design and Thorough Ablation Studies: The model is decomposed into key components (BrMoE, TIA, RMAE), and extensive ablation studies are conducted to validate the contribution of each, solidifying the methodological claims.

### Weaknesses
1. Limited Generalization Evidence: The model is evaluated solely on one private iEEG dataset. Its generalization capability remains unverified on any public iEEG benchmarks with varying experimental paradigms and recording parameters [1,2], raising questions about its robustness across broader data distributions.

2. Brain Region Modeling: The current approach models neural activity at the level of entire brain regions, potentially overlooking the functional complexity and finer-grained functional sub-divisions or dynamic network interactions within these regions.

3. Limited Applicability: While the framework is innovative, the absolute performance for language decoding remains low (e.g., ~29% top-1 accuracy for initial decoding in Table 3). This level of accuracy is far from sufficient for practical clinical applications in assistive communication, highlighting a significant gap towards immediate real-world impact.

4. Lacks In-Depth Comparison with State-of-the-Art Baselines: Comparisons with recent foundational models for neural decoding [3,4] are relatively limited. The paper does not fully establish its superior advantage in unified multi-task modeling against these strong contenders.

**References:**

[1] Wang, C., Yaari, A., Singh, A., Subramaniam, V., Rosenfarb, D., DeWitt, J., ... & Barbu, A. (2024). Brain treebank: Large-scale intracranial recordings from naturalistic language stimuli. *Advances in Neural Information Processing Systems, 37*, 96505-96540.

[2] Zheng, H., Wang, H., Jiang, W., Chen, Z., He, L., Lin, P., ... & Liu, Y. (2024). Du-IN: Discrete units-guided mask modeling for decoding speech from Intracranial Neural signals. *Advances in Neural Information Processing Systems, 37*, 79996-80033.

[3] Singh, A., Thomas, T., Li, J., Hickok, G., Pitkow, X., & Tandon, N. (2025). Transfer learning via distributed brain recordings enables reliable speech decoding. *Nature Communications, 16*(1), 8749.

[4] Chen, X., Wang, R., Khalilian-Gourtani, A., Yu, L., Dugan, P., Friedman, D., ... & Flinker, A. (2024). A neural speech decoding framework leveraging deep learning and speech synthesis. *Nature Machine Intelligence, 6*(4), 467-480.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a general-purpose neural decoding framework designed to handle the pervasive heterogeneity in multi-subject, multi-task intracranial recordings (ECoG/sEEG). The motivation stems from the limitations of existing neurophysiological decoding models, which often perform well only in single-task or single-subject settings and fail to generalize across heterogeneous datasets. The proposed framework, Neuro-MoBRE, explicitly models the variability across brain regions, subjects, and tasks to achieve robust generalization in brain decoding. Evaluations on intracranial recordings from subjects across diverse tasks show the performance of the proposed framework.

### Strengths
1. The paper presents a framework to address cross-subject and cross-task heterogeneity. By using a brain-regional mixture-of-experts mechanism, a brain-regional-temporal tokenizer, and task-disentangled aggregation, the framework separates regional, temporal, and functional variability of EEG data.

2. The authors curate and unify one of the most comprehensive intracranial EEG datasets to date, spanning 11 subjects and five heterogeneous tasks, including speech decoding, movement execution, and epileptic activity classification. 

3. The authors benchmark Neuro-MoBRE against a range of recent baselines (e.g., BIOT, LaBraM, NeuroLM) and demonstrate consistent improvements in accuracy and cross-subject generalization.

### Weaknesses
1. While the paper emphasizes that Neuro-MoBRE is designed to “explicitly resolve multi-subject and multi-task heterogeneity,” the empirical evidence for this claim is qualitative and indirect. The results show performance gains across subjects and tasks, but it remains unclear how much of that improvement is attributable to reduced heterogeneity versus general over-parameterization or better representation learning.

2. The paper claims robustness to low-SNR neural recordings, but the evidence remains unquantified. Although masked pretraining and expert specialization are helpful, there is no experiment showing that the model’s performance degrades less severely under noisy or limited-channel conditions than baselines.

3. The evaluation focuses on accuracy and ablation gains but provides limited insight into the learned representations or biological plausibility. There is no quantitative measure of region-expert correspondence, no evaluation of representational disentanglement.

4. The references to Figure A1 and Table A1 in lines 312 and 313 are incorrect. They actually refer to Figure 1 and Table 1, not the ones in the appendix.

### Questions
1. How can the authors demonstrate that Neuro-MoBRE truly resolves heterogeneity across subjects and tasks, rather than simply benefiting from larger capacity or better feature sharing? Would quantitative analyses such as inter-subject representational similarity, variance reduction, or expert-routing ablations help substantiate this claim?

2. Can the authors provide evidence on solving the low-SNR issue? Like the robustness under low-SNR conditions?

3. What interpretability or representational analyses can clarify what each regional expert learns?

### Soundness
3

### Presentation
2

### Contribution
2
