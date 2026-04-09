# Detailed Weakness Extraction: Empirical Study Limitations in ICLR 2025

## Paper 1: Language Models Need Inductive Biases to Count Inductively
**File:** s3IBHTTDYl.txt  
**Authors:** Yingshan Chang & Yonatan Bisk (Carnegie Mellon University)

### Weakness Category 1: Extrapolation and OOD Generalization Failures
**Quote:** "Transformers do not learn to count inductively, e.g. when the model knows increment(50)=51, it still cannot output the length of a 51-symbol sequence as 51 if it has only been trained on up to 50-length sequences."  
**Location:** Abstract/Introduction (Line ~17)  
**Issue:** Fundamental failure to extrapolate beyond training distribution - model cannot apply learned increment function to longer sequences

**Quote:** "Generalization deteriorates as the ratio between MAX_OOD_SEQLEN and MAX_TRAIN_SEQLEN becomes greater."  
**Location:** Table A8 findings (Line ~608)  
**Issue:** Performance not robust; as OOD gap increases, accuracy drops dramatically (from 98.9% to 22.1% with 4x gap)

### Weakness Category 2: Model-Size and Architecture Dependency
**Quote:** "Shallow 1L or 2L Transformers struggle to generalize inductively. Successful generalization is observed with 4L Transformers, but requiring different positional embeddings for different forms of counting."  
**Location:** Introduction/Abstract (Line ~19)  
**Issue:** Results are model-depth dependent; solutions not universal across depths; positional embeddings must be tuned per task format

**Quote:** "modern RNNs also largely underperform traditional RNNs in generalizing counting inductively, hinting at the tradeoff modern RNNs struggle to balance between parallelized training and maintaining their recurrent nature."  
**Location:** Abstract (Line ~19)  
**Issue:** Findings don't transfer across RNN architectures; newer architectures (SSMs, RWKV) fail where older ones succeed

### Weakness Category 3: Task-Specific and Positional Embedding Sensitivity
**Quote:** "different PEs encode different inductive biases that facilitate counting in different task formats"  
**Location:** Abstract (Line ~7)  
**Issue:** No universal solution; each task format requires different PE strategy - findings are not generalizable

**Quote:** "RoPE fails [at first token recognition] because it only modifies queries and keys, leaving the values identical for a homogeneous sequence"  
**Location:** Line ~512  
**Issue:** Architecture-specific failure mode not captured in simpler analysis

### Weakness Category 4: Methodological Concerns
**Quote:** "Position shift augmentation... ensure[s] that all position embeddings will be trained"  
**Location:** Section 2.2 (Line ~57)  
**Issue:** Without this augmentation, models fail at OOD testing - finding is training-procedure dependent

**Quote:** "we adopt shifted PEs. We empirically find that randomized PEs perform much worse than shifted PEs."  
**Location:** Line ~57  
**Issue:** Data augmentation strategy is task-specific; arbitrary choices (shifted vs. randomized) dramatically affect results

---

## Paper 2: Exploring and Benchmarking Planning Capabilities of Large Language Models
**File:** koza5fePTs.txt  
**Benchmark suite paper**

### Weakness Category 1: Complexity-Dependent Performance
**Quote:** "as the instance complexity increases the model success rate decreases"  
**Location:** Line ~47  
**Issue:** Performance is complexity-bounded; findings only valid within tested complexity range; harder instances may show different patterns

**Quote:** "Moreover, as the instance complexity increases the model success rate decreases"  
**Location:** Repeated in conclusion/methods section  
**Issue:** Systematic degradation suggests findings don't generalize to harder problem variants

### Weakness Category 2: In-Context Learning Generalization Issues
**Quote:** "prompting with easier instances leads to better performance on hard instances compared to prompting with hard instances"  
**Location:** Line ~37  
**Issue:** Non-monotonic relationship suggests distribution-specific biases in training data; findings not robust to prompt composition

### Weakness Category 3: Failure Mode Heterogeneity
**Quote:** "We categorize the failure modes into three categories: failure to satisfy environmental constraints, failure to meet the goal and failure to generate legal actions in a given state. [However,] not all of these failure modes are present across all benchmarks and methods."  
**Location:** Line ~45  
**Issue:** Failure modes are benchmark-specific; results don't transfer uniformly; method effectiveness varies by task

### Weakness Category 4: Data Curation Dependency
**Quote:** "pinpoint failure modes that are result of biases in training data emphasizing the importance of data curation during training"  
**Location:** Line ~49  
**Issue:** Model behavior stems from data artifacts, not fundamental capabilities; findings dependent on dataset composition

### Weakness Category 5: Out-of-Distribution Generalization Limitations
**Quote:** "we also probe the performance of the proposed methods in out-of-distribution scenarios, assessing the ability to generalize to novel and unseen planning challenges"  
**Location:** Abstract  
**Issue:** Even with explicit OOD testing, generalization is not uniform (as shown by earlier findings); OOD setting produces different phenomena

---

## Paper 3: Report Cards: Qualitative Evaluation of Language Models
**File:** H25xduunIK.txt  
**Authors:** Anonymous, submitted to ICLR 2025

### Weakness Category 1: Evaluation Scope Limitations
**Quote:** "However, manual inspections of LLM outputs, although insightful, are labor-intensive and can be limited in scope"  
**Location:** Introduction (Line ~17)  
**Issue:** Automated evaluation still has scope limitations despite claims of improvement

**Quote:** "our experiments are limited to specific topics and datasets. Future work should consider applying Report Cards to a wider range of domains—including open-ended evaluation"  
**Location:** Limitations section (Line ~136-140)  
**Issue:** Generalization of methodology unknown; tested only on specific benchmark subsets

### Weakness Category 2: Model-Size Generalization Uncertainty
**Quote:** "We generate Report Cards for a diverse set of models, ranging from smaller models like Llama-3.1-8B-Instruct to larger models such as Mixtral-8×7B-Instruct and GPT-3.5/4o"  
**Location:** Experiments section (Line ~88-95)  
**Issue:** Range of model sizes tested, but no analysis of whether Report Card methodology generalizes uniformly across sizes

### Weakness Category 3: Incomplete Coverage
**Quote:** "We are not yet aware of [all evaluation factors]. Moreover, our experiments are limited to specific topics and datasets."  
**Location:** Limitations  
**Issue:** Authors acknowledge unknown unknowns; methodology may miss important factors not captured in tested domains

---

## Paper 4: Efficient Multi Agent Offline Coordination via Diffusion Based Trajectory Stitching
**File:** EpnZEzYDUT.txt  
**Authors:** Lei Yuan et al., Nanjing University

### Weakness Category 1: Theoretical Grounding Absent
**Quote:** "Diffusion-based generation 'might violate environment dynamics due to excessively prioritizing high returns'" requiring "bidirectional dynamics constraint mechanism to identify generated observations that violate dynamics"  
**Location:** Section 4.1 (Line ~59-82)  
**Issue:** Core method has inherent reliability issue; post-hoc constraint added to patch problem; no proofs that constraints are sufficient

### Weakness Category 2: Complex Multi-Component System
**Quote:** "Requires multiple learned models (diffusion model, forward/inverse dynamics, credit assignment)"  
**Location:** Method section  
**Issue:** Complex pipeline increases failure surface; error accumulation across components not analyzed

**Quote:** "Once the joint observation trajectory is generated... we discard the subsequent segment after if dynamics consistency is severely violated"  
**Location:** Line ~80-81  
**Issue:** Ad-hoc filtering threshold; no principled way to set threshold; trajectories discarded without analysis

### Weakness Category 3: Benchmark-Specific Results
**Quote:** "Evaluated on MPE, SMAC, SMACv2, MAMuJoCo"  
**Location:** Section introduction  
**Issue:** Limited to specific benchmark family; generalization to other multi-agent domains unclear

### Weakness Category 4: Dataset Composition Dependency
**Quote:** "method addresses 'temporal and spatial imbalances inherent in multi-agent datasets' [where] performance depends on dataset composition and balance"  
**Location:** Analysis  
**Issue:** Method specifically designed for imbalanced data; performance may degrade with balanced datasets

---

## Paper 5: Loss in the Crowd: Hidden Breakthroughs in Language Model Training
**File:** pK4Z6NZ2DB.txt  
**Authors:** Anonymous, ICLR 2025 submission

### Weakness Category 1: Unvalidated Core Assumption
**Quote:** "If we assume that the top eigenvectors of the aggregate Hessian maintain high curvature at other points in training and on individual datapoints, then the scaling factor in the second order Taylor term will be very large even at the datapoint level."  
**Location:** Section 3.1.2 (Line ~56-73)  
**Issue:** Critical assumption for method validity not formally proven; empirical validation only shows "difference between first and second order values is small"

### Weakness Category 2: Limited Empirical Validation Scope
**Quote:** "We find that empirically, the difference between the first and second order values is small (Appendix F), but compute the second-order approximation to achieve a better estimate."  
**Location:** Line ~73  
**Issue:** Validation limited to appendix; difference being small doesn't prove assumption holds; method may be sensitive to edge cases not tested

### Weakness Category 3: Clustering Dependency and Hyperparameter Sensitivity
**Quote:** "HDBSCAN outliers are labeled and shown as cluster 0 but excluded from remaining analysis. [We] set the minimum cluster size to be at least 20% of the total number of trajectories"  
**Location:** Experimental setup (Line ~91)  
**Issue:** Arbitrary hyperparameter choice (20% threshold); outlier removal biases results; clustering algorithm choice affects conclusions

### Weakness Category 4: Limited Natural Language Validation
**Quote:** "Validated on synthetic setting (arithmetic) and limited natural language experiments"  
**Location:** Experimental section  
**Issue:** Main results on synthetic arithmetic task; natural language findings limited in scope; transfer unclear

### Weakness Category 5: Task-Specific Success
**Quote:** "For <1>, the carrying skill is likely recovered because of the digit skill, as the carry cluster for <1> consists of token instances in the 1000s place"  
**Location:** Line ~94-96  
**Issue:** Recovery of skills sometimes due to task structure, not method; findings may be artifact of task design

---

## Cross-Paper Synthesis: Common Weakness Patterns

### Pattern 1: Extrapolation Beyond Training Distribution
- **Papers affected:** s3IBHTTDYl, koza5fePTs
- **Core issue:** Models fail to extrapolate when OOD distance increases or complexity increases
- **Quote from s3IBHTTDYl:** "the model knows increment(50)=51, it still cannot output the length of a 51-symbol sequence as 51"
- **Quote from koza5fePTs:** "as the instance complexity increases the model success rate decreases"

### Pattern 2: Findings are Architecture/Model-Dependent
- **Papers affected:** s3IBHTTDYl, H25xduunIK
- **Core issue:** Results don't transfer uniformly across model sizes and architectures
- **Quote:** "different PEs encode different inductive biases... [requiring] different positional embeddings for different forms of counting"

### Pattern 3: Methodological Choices Drive Results
- **Papers affected:** All five papers
- **Core issue:** Task formulation, hyperparameters, data augmentation heavily influence outcomes
- **Quote from pK4Z6NZ2DB:** "set the minimum cluster size to be at least 20% of the total number of trajectories"

### Pattern 4: Limited Scope Acknowledged
- **Papers affected:** H25xduunIK, koza5fePTs
- **Core issue:** Authors explicitly note limitations; findings not claimed to be fully general
- **Quote:** "our experiments are limited to specific topics and datasets. Future work should consider applying Report Cards to a wider range of domains"

### Pattern 5: Missing Theoretical Grounding
- **Papers affected:** EpnZEzYDUT, pK4Z6NZ2DB
- **Core issue:** Methods work empirically but lack formal guarantees or proofs
- **Quote from EpnZEzYDUT:** "bidirectional dynamics constraint mechanism [added] to identify [violations]" - post-hoc fix without theoretical justification

---

## Implications for Review

### Strong Concerns
1. **Generalization claims beyond tested scope** - need explicit acknowledgment of distribution shift limitations
2. **Model-dependent results** - need evidence of transfer across architectures
3. **Unvalidated assumptions** - need formal proofs or extensive empirical validation
4. **Hyperparameter sensitivity** - need ablation studies and robustness analysis

### Moderate Concerns
1. **Limited benchmark coverage** - acknowledge domain specificity
2. **Task-specific phenomena** - clarify what is general vs. artifact of task design
3. **Data dependency** - acknowledge data curation effects
4. **Computational cost barriers** - discuss reproducibility implications

### Areas for Reviewer Scrutiny
- Claims of "inductive" learning when findings are specific to certain formulations
- Generalization claims without explicit OOD testing
- Complexity-dependent phenomena presented as universal
- Ad-hoc solutions (constraints, filtering, augmentation) without theoretical justification
