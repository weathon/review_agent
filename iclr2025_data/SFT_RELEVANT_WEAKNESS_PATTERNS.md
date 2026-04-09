# Relevant Weakness Patterns for SFT Paper
## "SFT WITHOUT OVERFITTING: ANALYZING THE TRAINING DYNAMICS OF SUPERVISED FINE-TUNING"

Based on analysis of 8 key human reviews from ICLR 2025, the following weakness patterns emerge that are applicable to a paper on selective fine-tuning and OOD generalization:

---

## 1. LIMITED BENCHMARK EVALUATION & GENERALIZATION CONCERNS
**Pattern**: Papers claiming improvements on limited task sets fail to demonstrate broad applicability.
- **Review Evidence**:
  - "Experiments on the Long Range Arena are not exactly like-for-like, as the ToST hyper-params are tuned individually for each task" (Transformer architecture paper)
  - "The method is evaluated on low-to-mid complexity image classification tasks only. Detailed analysis on higher complexity classification tasks on ImageNet is missing" (Continual learning paper)
  - "Multi-domain data is almost only in English. For multilingual data, data of each language should also contain various domains. Therefore there are confounding variables." (DEPT paper)
- **Risk to SFT paper**: If GeneralPoints and V-IRL benchmarks are the only evaluation datasets, reviewers may question whether selective attention-only SFT truly generalizes beyond these specific rule-based reasoning tasks.

---

## 2. INSUFFICIENT ANALYSIS OF FAILURE MODES & DOMAIN SPECIFICITY
**Pattern**: Papers don't adequately investigate when/why their method works or fails in different settings.
- **Review Evidence**:
  - "There is an absence of performance metrics for base methods such as PEFT on LLMs" (Domain adaptation paper)
  - "There could be more analyses on the data and model scaling... It remains unclear why certain tasks perform better or worse on different environments" (GUI agent paper)
  - "How different are the task vectors learned with ITA vs IEL?" (Model merging paper)
- **Risk to SFT paper**: The paper should clearly explain WHY attention-only SFT works better for OOD generalization. Without mechanistic insight, results appear task-specific rather than principled.

---

## 3. MEMORIZATION VS. GENERALIZATION NOT CLEARLY DISTINGUISHED
**Pattern**: Papers studying fine-tuning/learning dynamics conflate observed improvements with actual generalization or fail to measure memorization separately.
- **Review Evidence**:
  - "The observed correlation between example learning speed and catastrophic forgetting is empirical, with no theoretical analysis provided" (Learning dynamics paper)
  - "Most evaluations, including the ablation studies, focus on downstream tasks, which are interesting and practically relevant but do not provide clear insight into what the algorithm is doing" (Shift invariance paper)
  - "The generalization guarantee applies to the implicit reward model, but what we actually use is the LLM... there could not exist such a guarantee for π_θ" (DPO generalization paper)
- **Risk to SFT paper**: Need to explicitly measure memorization (e.g., test on identical in-distribution examples) vs. generalization (OOD shift). Claims about avoiding overfitting need direct evidence.

---

## 4. HYPERPARAMETER SENSITIVITY & REPRODUCIBILITY ISSUES
**Pattern**: Methods appear to require task-specific tuning, undermining claims of general applicability.
- **Review Evidence**:
  - "Transformers are amazing because they require little tuning from task to task... if the method needs tuning for each specific task that definitely takes quite a bit away from its appeal" (Transformer architecture paper)
  - "The algorithm may rely on selecting hyperparameters (e.g. s and q) for removing slowest and fastest examples. It might be unclear how that parameter varies across different datasets" (Continual learning paper)
  - "If choosing hyperparameters via repetitive experiments, then it may defeat the premise of continual learning" (Continual learning paper)
- **Risk to SFT paper**: If selective attention/feedforward tuning requires per-dataset hyperparameter selection, this limits practical impact. Needs clear guidance on hyperparameter selection across different tasks.

---

## 5. INCOMPLETE ABLATION STUDIES & MISSING BASELINES
**Pattern**: Papers lack comprehensive comparisons or ablations that would reveal whether improvements come from the proposed mechanism or other confounding factors.
- **Review Evidence**:
  - "An ablation study or an explanatory paragraph isolating factors that contribute to divergence would also help" (DEPT paper)
  - "An important additional baseline would be models trained on individual data sets" (DEPT paper)
  - "L_KDL is not ablated to show its usefulness in this work" (Domain adaptation paper)
  - "More importantly, what motivation or significant proof do these ablation studies provide for the design of this method?" (Adapter paper)
- **Risk to SFT paper**: Need ablations showing: (a) attention-only vs. feedforward-only vs. both, (b) different selective fine-tuning percentages, (c) interaction with model size, (d) comparison to standard full fine-tuning and other efficient FT methods (LoRA, prefix tuning, etc.).

---

## 6. LIMITED SCOPE & SCALABILITY QUESTIONS
**Pattern**: Methods tested only on small models or datasets; unclear if they scale to realistic settings.
- **Review Evidence**:
  - "Tripling the number of parameters yields similar performance to GPT2-Base. The running time would be considerably higher... not exactly re-assuring" (Transformer paper)
  - "Experiments are performed only on the image classification domain... it remains unclear how the approach performs on other modalities" (Domain shift paper)
  - "Data sources of 54K for desktop is imbalanced... not similar for each platform" (GUI agent paper)
  - "Llama 2-7B is relatively old, experiments on 3/3.1/3.2 would be better" (DPO generalization paper)
- **Risk to SFT paper**: If experiments use small models or specific architectures, reviewers will ask: Do results hold for larger models? Different transformer variants? Modern instruction-tuned models?

---

## 7. THEORETICAL JUSTIFICATION OR MECHANISTIC UNDERSTANDING MISSING
**Pattern**: Empirical findings lack theoretical grounding; claims about "why" something works are not substantiated.
- **Review Evidence**:
  - "The foundational hypothesis that 'PLMs encapsulate multiple pieces of knowledge as subnetworks' lacks supporting references or verification experiments" (Domain adaptation paper)
  - "The authors did not provide an independent evaluation of the guidance network's specific performance in estimating the phase angle φ... Most evaluations focus on downstream tasks, confounding factors" (Shift invariance paper)
  - "No explanation or intuition is provided as to why medium learning speed items are the most useful" (Continual learning paper)
- **Risk to SFT paper**: Paper should provide mechanistic explanation for WHY selective attention SFT improves OOD generalization. Is it about reducing overfitting to spurious task-specific patterns? Reducing gradient interference? Needs theoretical or empirical validation.

---

## 8. OOD EVALUATION & GENERALIZATION GAPS NOT RIGOROUSLY MEASURED
**Pattern**: Papers claiming improved generalization don't properly test OOD robustness or the measurement methodology is unclear.
- **Review Evidence**:
  - "How well does the guidance network perform in OOD settings where the ground truth shifts are known but the time series were not part of training data?" (Shift invariance paper)
  - "The LLM experiments need baselines to compare to, like in Table 3" (Domain shift paper)
  - "Since IN-R, C-100 and CUB are very much in-distribution w.r.t pre-training on ImageNet, I wonder whether simple fine-tuning of the final classification layer can be sufficient?" (Model merging paper)
- **Risk to SFT paper**: Need rigorous OOD evaluation protocol: (a) clear definition of distribution shift (length, complexity, rule variation), (b) comparison to strong baselines (full FT, LoRA, standard fine-tuning), (c) analysis of which types of OOD shifts benefit from selective SFT.

---

## 9. INSUFFICIENT ERROR ANALYSIS & EDGE CASE HANDLING
**Pattern**: Papers don't investigate when methods fail or perform worse than baselines.
- **Review Evidence**:
  - "It seems that all the fine-tuning methods fail to improve average accuracy" (Fairness paper)
  - "Sometimes results show a fair overlap between prior methods and yours in standard deviation; the advantage in those cases is not clear" (Shift invariance paper)
  - "High variance/noise in results makes it unclear whether improvements are statistically significant or due to random fluctuation" (Multiple papers)
- **Risk to SFT paper**: Need confidence intervals/statistical significance tests. Identify when selective attention SFT underperforms (e.g., small training sets, certain reasoning types). Explain failure modes.

---

## 10. PRACTICAL APPLICABILITY & COMPUTATIONAL TRADE-OFFS NOT ADDRESSED
**Pattern**: Methods improve some metrics but ignore computational costs, inference latency, or practical deployment challenges.
- **Review Evidence**:
  - "Will the results converge after longer learning times? The unidirectional flow increases output quality, but why?" (Adapter paper - no mechanistic explanation)
  - "During inference, will UniCon be similar or even worse in latency and memory due to the more complex structure?" (Adapter paper)
  - "What is the speed reduction in using this method? It must have some sort of slow-down associated with it" (Domain adaptation paper)
  - "The paper lacks discussion of computational overhead and how this method compares when accounting for total training/inference cost" (Multiple papers)
- **Risk to SFT paper**: Need to quantify: (a) training time for selective SFT vs. standard FT, (b) computational overhead of determining which modules to fine-tune, (c) memory usage during training, (d) inference cost (does selective SFT speed up inference?).

---

## 11. BENCHMARK DESIGN ARTIFACTS & LIMITED TASK DIVERSITY
**Pattern**: Benchmarks may be too simplistic, biased toward the proposed method, or not representative of real-world complexity.
- **Review Evidence**:
  - "The paper only explores ResNet and its smaller variants... do the same conclusions stand for transformers, VGG, etc.?" (Continual learning paper)
  - "More analysis should cover per-task and per-environment performance. Why do certain tasks perform better or worse?" (GUI agent paper)
  - "The experimental design may have confounding variables" (Multiple papers)
- **Risk to SFT paper**: Are GeneralPoints and V-IRL truly representative of reasoning tasks? Do they have specific properties that favor attention-selective tuning? Need analysis of task characteristics and diversity.

---

## Summary: Key Challenges to Address

To strengthen the SFT paper against these reviewer patterns, authors should:

1. **Expand evaluation** beyond two benchmarks; include diverse reasoning tasks and compare to strong baselines
2. **Provide mechanistic insight** into why selective attention SFT improves OOD generalization (ablations, analysis)
3. **Measure memorization vs. generalization** explicitly; show that improvements aren't just from smaller effective model capacity
4. **Include comprehensive ablations**: attention vs. feedforward, selective percentages, interaction with model architecture/size
5. **Test scalability**: larger models, different architectures, different data regimes
6. **Analyze failure modes**: when does selective SFT underperform? Which reasoning types benefit most?
7. **Quantify computational trade-offs**: training time, memory, inference cost
8. **Statistical rigor**: confidence intervals, significance tests, multiple random seeds
9. **Provide hyperparameter guidance**: how to set selection criteria for new tasks/domains
10. **Compare to related efficient FT methods**: LoRA, prefix tuning, adapter tuning, layer-wise fine-tuning

