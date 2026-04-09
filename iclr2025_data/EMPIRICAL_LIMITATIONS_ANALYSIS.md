# Empirical Study Limitations in ICLR 2025 Papers

## Summary
This document analyzes papers from ICLR 2025 dataset that explicitly discuss limitations of empirical studies, scaling methodology, curve fitting reliability, experimental design, and generalization concerns.

## Papers Analyzed

### 1. **Language Models Need Inductive Biases to Count Inductively** (s3IBHTTDYl.txt)

**Key Weakness - Generalization Failures:**
- "although the notion of length may vary across domains (e.g. sequence length, recursion depth, counter states for DSAs, stack sizes for PDAs), counting is always involved as a required component to successfully handle the task"
- **Critical finding:** "Transformers do not learn to count inductively, e.g. when the model knows increment(50)=51, it still cannot output the length of a 51-symbol sequence as 51 if it has only been trained on up to 50-length sequences."
- This demonstrates **fundamental out-of-distribution (OOD) generalization failure** - the model cannot extrapolate beyond training range

**Scope Limitations:**
- "Shallow 1L or 2L Transformers struggle to generalize inductively. Successful generalization is observed with 4L Transformers, but requiring different positional embeddings for different forms of counting."
- Finding is **model-size dependent** and **task-format dependent**
- Different positional embeddings needed for different counting formats (vanilla, modular, selective)

**Methodological Concerns:**
- Different positional embeddings (SinePE, APE, RoPE, SPE) show different performance - finding is **architecture-specific**
- Modern RNNs (SSMs, RWKV) "largely underperform traditional RNNs in generalizing counting inductively"
- Task design heavily influences results (vanilla vs. modular vs. selective counting)

**Generalization Degradation Pattern (Table A8):**
```
MAX_TRAIN_SEQLEN | MAX_OOD_SEQLEN | Accuracy
     50          |      100        |  84.3% (IND), 98.9% (OOD)
     50          |      200        |  100% (IND), 30.9% (OOD)
     50          |      200        |  100% (IND), 22.1% (OOD)
```
Performance "deteriorates as the ratio between MAX_OOD_SEQLEN and MAX_TRAIN_SEQLEN becomes greater"

---

### 2. **Exploring and Benchmarking Planning Capabilities of Large Language Models** (koza5fePTs.txt)

**Limited Scope - Complexity Boundary:**
- "As the instance complexity increases the model success rate decreases"
- Performance is **complexity-dependent** and findings may not generalize to harder instances

**Generalization Challenges:**
- Out-of-distribution generalization tested but not uniform: "prompting with easier instances leads to better performance on hard instances compared to prompting with hard instances"
- This suggests **task-specific biases** in the training data that don't generalize

**Failure Modes - Task-Specific Issues:**
- Authors "categorize the failure modes into three categories: failure to satisfy environmental constraints, failure to meet the goal and failure to generate legal actions in a given state"
- "Moreover, as the instance complexity increases the model success rate decreases"
- Some failure modes "are not present across all benchmarks and methods"

**Data Bias Concerns:**
- "Moreover, pinpoint failure modes that are result of biases in training data emphasizing the importance of data curation during training"
- Findings are **data-dependent** and domain-specific to particular benchmarks

---

### 3. **Report Cards: Qualitative Evaluation of Language Models Using Natural Language Summaries** (H25xduunIK.txt)

**Evaluation Limitations:**
- "manual inspections of LLM outputs, although insightful, are labor-intensive and can be limited in scope"
- "We are not yet aware of [all evaluation factors]"
- "our experiments are limited to specific topics and datasets"

**Scope Constraints:**
- "Future work should consider applying Report Cards to a wider range of domains—including open-ended evaluation"
- Evaluation methodology tested only on "a subset of topics from three datasets: MMLU, Anthropic Advanced AI Risk, and Chinese grammar"
- Model evaluation "ranges from smaller models like Llama-3.1-8B-Instruct to larger models such as Mixtral-8×7B-Instruct and GPT-3.5/4o"
- **Finding generalization unclear across model sizes**

---

### 4. **Efficient Multi Agent Offline Coordination via Diffusion Based Trajectory Stitching** (EpnZEzYDUT.txt)

**Scope Limitations:**
- Method tested on "imbalanced datasets of multiple benchmarks" but focuses on specific types
- Evaluated on MPE, SMAC, SMACv2, MAMuJoCo - **benchmark-specific results**

**Generalization Concerns:**
- Method addresses "temporal and spatial imbalances inherent in multi-agent datasets"
- Performance depends on dataset composition and balance
- Diffusion-based generation "might violate environment dynamics due to excessively prioritizing high returns"
- Had to introduce "bidirectional dynamics constraint mechanism to identify generated observations that violate dynamics"

**Theoretical Grounding:**
- Approach is **heuristic-based** without theoretical guarantees on trajectory quality or dynamics consistency
- Requires multiple learned models (diffusion model, forward/inverse dynamics, credit assignment) - **complex pipeline with multiple failure points**

---

### 5. **Loss in the Crowd: Hidden Breakthroughs in Language Model Training** (pK4Z6NZ2DB.txt)

**Limited Empirical Validation:**
- "we are still only considering breakthroughs that are general enough to be perceived in loss curves"
- Second-order Taylor approximation assumptions: "If we assume that the top eigenvectors of the aggregate Hessian maintain high curvature at other points in training and on individual datapoints"
- **This assumption is not formally validated**

**Scope of Findings:**
- Validated on synthetic setting (arithmetic) and limited natural language experiments
- "We find that empirically, the difference between the first and second order values is small (Appendix F), but compute the second-order approximation to achieve a better estimate"
- **Hessian-based analysis may not capture all breakthrough phenomena**

**Clustering Reliability:**
- "HDBSCAN outliers are labeled and shown as cluster 0 but excluded from remaining analysis"
- Results depend on clustering parameters (minimum cluster size set to 20% of trajectories)
- **Arbitrary hyperparameter choices affect conclusions**

**Task-Specific Results:**
- Arithmetic experiments show good cluster recovery, but "For <1>, the carrying skill is likely recovered because of the digit skill, as the carry cluster for <1> consists of token instances in the 1000s place"
- Natural language results limited; findings may not transfer to other tasks

---

## Common Themes Across Papers

### 1. **Generalization Failures at Distribution Shift**
- Transformers fail to extrapolate counting beyond training lengths
- Model performance degrades with increased OOD distance
- Task complexity directly impacts performance degradation

### 2. **Scope and Model-Size Dependency**
- Results often specific to particular model sizes (1L vs 4L transformers; 8B vs 70B models)
- Performance characteristics not consistent across architecture families
- Positional embedding choices critical but task-dependent

### 3. **Methodological Limitations**
- Findings often emerge from specific task formulations that may not generalize
- Data augmentation and curation heavily influences results
- Heuristic solutions (dynamics constraints, positional shifts) work but lack theoretical justification

### 4. **Incomplete Theoretical Grounding**
- Empirical methods (POLCA, trajectory stitching) lack formal guarantees
- Assumptions in methods (e.g., Hessian eigenvector stability) not thoroughly validated
- Linear decomposition and clustering assumptions work in practice but theoretical limits unknown

### 5. **Benchmark and Data Dependencies**
- Performance highly dependent on specific benchmark compositions
- Data imbalance and bias affect generalization
- Findings don't uniformly transfer across different benchmarks or domains

### 6. **Task-Specific Biases**
- Failure modes and successes often tied to particular task structures
- Easier examples help learn harder examples (suggesting distribution-specific phenomena)
- Feature importance varies across subtasks within same domain

---

## Computational Cost Concerns

Several papers mention but don't fully address:
- Hessian computation expensive (POLCA paper requires Hessian-vector products)
- Diffusion-based trajectory generation computationally intensive
- Fine-tuning and extensive hyperparameter search required for generalization
- **Reproducibility challenged by computational costs**

---

## Statistical Significance Gaps

- Most papers report best results from multiple seeds but don't discuss variance
- Some papers rely on qualitative clustering rather than statistical tests
- Confidence intervals and significance testing absent in several empirical claims
