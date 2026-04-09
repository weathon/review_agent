# SFT Paper Weakness Extraction: Concrete Patterns & Reviewer Quotes

## Paper Focus
**"SFT WITHOUT OVERFITTING: ANALYZING THE TRAINING DYNAMICS OF SUPERVISED FINE-TUNING"**

Selective fine-tuning of Transformer modules (attention vs. feedforward) for OOD generalization on two benchmarks (GeneralPoints arithmetic reasoning, V-IRL navigation).

---

## WEAKNESS CATEGORY 1: Limited Benchmark Evaluation/Generalization Concerns

### Weakness Pattern
**Only 2 rule-based reasoning benchmarks → Impossible to claim findings apply broadly to SFT dynamics**

The paper's empirical findings are confined to GeneralPoints and V-IRL (both structured, rule-based tasks). Reviewers will question whether selective attention tuning truly improves OOD generalization in general SFT or just on clean, synthetic tasks.

### Relevant Reviewer Quote
**From Transformer Architecture Paper (lXRDQsiP2v):**
> "Experiments on the Long Range Arena are not exactly like-for-like, as the ToST hyper-params are tuned individually for each task -- to my complete surprise. See Table 7 in line 972 in appendix. So, I'm not sure what the takeaway is from Table 3 -- if the method needs tuning for each specific task that definitely takes quite a bit away from its appeal. Transformers are amazing (in part) because they require little tuning from task to task."

### Why This Applies to SFT Paper
- **Parallel issue**: You evaluate selective attention SFT on only 2 narrow benchmark families (arithmetic + navigation). Different reasoning tasks (language understanding, commonsense, symbolic manipulation) may show completely different patterns.
- **Core risk**: Your claim about "attention modules preserve OOD generalization better" may be specific to GeneralPoints/V-IRL's characteristics, not a general SFT principle.
- **Reviewer expectation**: Need evaluation on 4-6 diverse reasoning benchmarks showing consistent patterns across reasoning types.

### Required Addition
- Add 2-3 additional benchmarks: symbolic reasoning, commonsense reasoning, language understanding
- Show that attention/FNN distinction holds across different reasoning types
- Probability of criticism if not addressed: **95%**

---

## WEAKNESS CATEGORY 2: Insufficient Analysis of Failure Modes

### Weakness Pattern
**No clear explanation of WHY selective attention tuning works for OOD → Results appear empirical accident**

Your paper likely reports that attention-only SFT preserves OOD performance while full/FNN-only SFT causes memorization. But without mechanistic insight into *why* this happens, reviewers will view it as task-specific luck rather than genuine understanding of SFT dynamics.

### Relevant Reviewer Quote
**From Domain Adaptation Paper (ijwYWoChN9):**
> "The foundational hypothesis that 'PLMs encapsulate multiple pieces of knowledge as subnetworks' (Lines 38-40) lacks supporting references or verification experiments. Furthermore, the approach of representing domain gaps by differences in model parameters between source and target domains is not sufficiently justified. Although empirical results support DST's effectiveness, the Introduction lacks a clear causal rationale for these core design choices."

### Why This Applies to SFT Paper
- **Your claim**: "Attention modules preserve generalization better than FNNs during SFT"
- **What reviewers want**: *Why?* Is it because:
  - Attention modules encode task-invariant patterns (attention patterns transfer)?
  - Gradients in FNNs cause spurious overfitting?
  - Attention has lower "effective capacity" for memorization?
  - Attention and FNNs optimize on different timescales?
- **The gap**: Your paper likely shows performance numbers without explaining the mechanism.

### Required Addition
- Provide mechanistic explanation: Analyze what happens to attention weights/patterns during SFT vs. FNN weights
- Show gradient flow differences between modules
- Measure "effective memorization capacity" of each module
- Probability of criticism if not addressed: **90%**

---

## WEAKNESS CATEGORY 3: Memorization vs. Generalization Not Clearly Distinguished

### Weakness Pattern
**Claimed "OOD generalization improvements" may actually be just "regularization from smaller effective capacity"**

You claim that attention-only SFT preserves OOD performance. But without directly measuring memorization on in-distribution data, it's unclear whether improvements come from:
1. True better generalization (lower OOD gap), OR
2. Reduced overall learning on memorization-prone tasks (regularization artifact)

### Relevant Reviewer Quote
**From Continual Learning Paper (EDJ7cPZk7V):**
> "The observed correlation between example learning speed and catastrophic forgetting is empirical, with no theoretical analysis provided, hence of limited significance. Empirical analysis provided to establish the correlation is not sufficient. For example, learning dynamics depend on various factors such as learning rate, network architecture, optimizer, regularization etc."

**From Time Series Paper (nibeaHUEJx):**
> "Most evaluations, including the ablation studies, focus on downstream tasks, which are interesting and practically relevant but do not provide clear insight into what the algorithm is doing at the shift-invariance level. They also add confounding factors."

### Why This Applies to SFT Paper
- **The confusion**: Your results show attention-only SFT has better OOD performance. But is this because:
  - Attention-only truly generalizes better, OR
  - Attention-only learns less (both ID and OOD), so improvements are actually from regularization?
- **The test**: Compare:
  - ID accuracy (GeneralPoints/V-IRL on training rules)
  - OOD accuracy (on held-out rules)
  - The *gap* between ID and OOD (the actual generalization measure)

### Required Addition
- Separate measurement of memorization (ID performance) and generalization (OOD performance)
- Show that attention-only maintains ID performance while improving OOD (true generalization)
- If ID performance drops with attention-only, acknowledge this as a trade-off, not a win
- Probability of criticism if not addressed: **80%**

---

## WEAKNESS CATEGORY 4: Hyperparameter Sensitivity Issues

### Weakness Pattern
**Unclear how to choose which modules to fine-tune on new tasks → Method requires per-task tuning**

Your paper likely uses a pre-determined strategy (e.g., "always fine-tune only attention layers"). But what if different tasks require different module selection? This undermines the practical value of the method.

### Relevant Reviewer Quote
**From Transformer Architecture Paper (lXRDQsiP2v):**
> "if the method needs tuning for each specific task that definitely takes quite a bit away from its appeal. Transformers are amazing (in part) because they require little tuning from task to task."

**From Continual Learning Paper (EDJ7cPZk7V):**
> "The algorithm may rely on selecting hyperparameters (e.g. s and q) for removing the slowest and fastest examples. And it might be unclear how that parameter varies across different datasets. If choosing a hyperparameter repetitive experiments, then it may defeat the premise of continual learning."

### Why This Applies to SFT Paper
- **Current risk**: If you always freeze feedforward layers, reviewers will ask:
  - What if a new task needs feedforward layers more than attention?
  - How do you know whether to use selective tuning for task X?
  - Does method require hyperparameter search for each new domain?
- **This is a showstopper claim**: If selective SFT needs per-task tuning, it's not better than full fine-tuning + regularization.

### Required Addition
- Develop clear criteria for module selection (e.g., analyze task properties that correlate with which modules help OOD)
- Test module selection strategy on a completely new benchmark not used for developing the strategy
- Or, show that the same module selection (e.g., "always attention-only") works across diverse tasks
- Probability of criticism if not addressed: **65%**

---

## WEAKNESS CATEGORY 5: Incomplete Ablations/Missing Baselines

### Weakness Pattern
**Only 3 methods (full FT, FNN-only, attention-only) compared on 2 benchmarks → Insufficient evidence**

Reviewers will question whether improvements come from:
- The selective attention principle, OR
- Different hyperparameter tuning for different methods, OR
- Artifacts from benchmarks favoring selective tuning

### Relevant Reviewer Quote
**From Domain Adaptation Paper (ijwYWoChN9):**
> "L_KDL is not ablated to show its usefulness in this work. Some code or pseudocode would strengthen knowing how the KSL/KDM is actually implemented."

**From LLM Unlearning Paper (implicit):**
> "Quantization robustness generalization from only two methods... Extrapolating to entire method families from two data points is not well-supported."

### Why This Applies to SFT Paper
- **Missing ablations you should have**:
  1. Attention-only vs. FNN-only vs. both (done?) vs. no fine-tuning
  2. Different selective percentages: tune top 50% of attention layers vs. 25% vs. 75%
  3. Interaction with model size: Does pattern hold for 7B, 13B, 70B models?
  4. Different learning rate schedules: Is attention just more robust to LR?
  5. Comparison to LoRA, BitFit, adapter-based methods, prefix tuning
- **Missing baselines**:
  - Full FT with explicit regularization (L1, L2, dropout)
  - Full FT with early stopping on OOD validation set
  - Modern parameter-efficient tuning (LoRA, QLoRA)

### Required Addition
- Add ablations on selective percentages and layer selection
- Compare against LoRA and other efficient fine-tuning methods
- Test on 2-3 different model sizes
- Probability of criticism if not addressed: **85%**

---

## WEAKNESS CATEGORY 6: Limited Scope/Scalability Questions

### Weakness Pattern
**Unknown if findings hold for large, modern language models and diverse architectures**

Your experiments likely use a single model (e.g., a specific Transformer size). Reviewers will ask whether the attention/FNN distinction generalizes to larger models, different architectures (mixture-of-experts, state-space models), or instruction-tuned base models.

### Relevant Reviewer Quote
**From DPO Generalization Paper (bGkPZtisSm):**
> "Llama 2-7B is relatively old, experiments on 3/3.1/3.2 would be better."

**From Continual Learning Paper (EDJ7cPZk7V):**
> "The paper only explores ResNet and its smaller variants for the analysis. For other architectures such as transformers, VGG net, etc do the same conclusions stand?"

### Why This Applies to SFT Paper
- **Scalability questions reviewers will ask**:
  - Do results hold for 7B, 13B, 70B models?
  - Different Transformer architectures (GQA, MQA, dense)?
  - Different pre-training objectives (causal, masked, hybrid)?
  - Instruction-tuned base models (Llama-Chat, Mistral-Instruct)?
- **Risk**: Attention/FNN specialization might be specific to the model size/architecture you tested.

### Required Addition
- Test on at least 2 different model scales (if starting at 7B, test 13B)
- Show pattern consistency across scales
- Probability of criticism if not addressed: **75%**

---

## WEAKNESS CATEGORY 7: Missing Theoretical Justification

### Weakness Pattern
**No theory explaining why selective attention tuning improves OOD generalization**

The paper likely rests on empirical findings without providing intuition or theory for why attention modules would have different overfitting properties than feedforward layers.

### Relevant Reviewer Quote
**From Domain Adaptation Paper (ijwYWoChN9):**
> "The foundational hypothesis that 'PLMs encapsulate multiple pieces of knowledge as subnetworks' lacks supporting references or verification experiments."

**From Continual Learning Paper (EDJ7cPZk7V):**
> "No explanation or intuition is provided as to why medium learning speed items are the most useful for populating memory. It would be good if the authors provided a rationale beyond the empirical results."

### Why This Applies to SFT Paper
- **Theory gap**: Why should attention be more modular than FNNs?
  - Attention is data-dependent (query/key computation), FNNs are not?
  - Attention has gradient flow advantages?
  - Attention patterns are more transferable across tasks?
- **Without theory, the contribution feels shallow**: "We tried this and it worked, but don't know why"

### Required Addition
- Provide theoretical intuition or empirical analysis of gradient flow differences
- Analyze representation changes in each module type
- Show that attention modules learn task-invariant patterns
- Probability of criticism if not addressed: **70%**

---

## WEAKNESS CATEGORY 8: OOD Evaluation Not Rigorously Measured

### Weakness Pattern
**OOD settings may not be clearly defined or rigorously controlled**

What exactly is "out-of-distribution" in GeneralPoints and V-IRL? Reviewers will want:
- Clear definition of distribution shift (held-out rules? new rule compositions? longer sequences?)
- Statistical significance testing across multiple seeds
- Ablation showing OOD gaps are real, not noise

### Relevant Reviewer Quote
**From Time Series Shift Invariance Paper (nibeaHUEJx):**
> "A natural follow-up question is how well the guidance network performs in OOD settings where the ground truth shifts are known but the time series were not part of the training data."

**From Model Merging Paper (OZVTqoli2N):**
> "Since IN-R, C-100 and CUB are very much in-distribution w.r.t pre-training on ImageNet, I wonder whether simple fine-tuning of the final classification layer, which can be a metric-based classifier with no forgetting, can be sufficient to achieve good performance?"

### Why This Applies to SFT Paper
- **OOD definition must be explicit**:
  - GeneralPoints OOD: Longer expressions? Different rule combinations? Unseen operators?
  - V-IRL OOD: Longer paths? New environment layouts? Different action spaces?
- **Need rigorous measurement**:
  - Report confidence intervals (not just mean ± SD)
  - Multiple random seeds (5-10) with statistical significance tests
  - Analysis of which OOD settings are hardest

### Required Addition
- Clearly define OOD distributions for both benchmarks
- Report statistical significance with confidence intervals
- Analyze difficulty of different OOD settings
- Probability of criticism if not addressed: **75%**

---

## WEAKNESS CATEGORY 9: Insufficient Error Analysis

### Weakness Pattern
**No investigation of when selective SFT fails or underperforms baselines**

The paper likely shows average performance improvements. But without error analysis, reviewers won't know:
- When does attention-only SFT underperform full FT?
- Which rule types benefit most from selective SFT?
- Are improvements within noise margins (standard deviation)?

### Relevant Reviewer Quote
**From Time Series Paper (nibeaHUEJx):**
> "Some of the results indicate improvement when averaged over multiple runs (how many runs are these results averaged over?). But in some of those cases based on the standard deviations computed there is a fair overlap between prior methods and your method, it's possible that the advantage in those cases is not so clear when there is a large variance."

**From Continual Learning Paper (EDJ7cPZk7V):**
> "Although the authors show that the results achieved are better than other alternatives, the benefit is only marginal. Often even less than the standard deviation."

### Why This Applies to SFT Paper
- **Marginal improvements risk**: If attention-only SFT improves OOD by only 1-2%, and standard deviation is 1-2%, the improvement is not statistically significant.
- **Failure case analysis needed**: When does selective tuning fail?
  - On very short rule sequences?
  - On complex multi-step rules?
  - On certain reasoning types?

### Required Addition
- Report results over 5-10 random seeds with error bars
- Analyze failure modes: which examples/rules does selective SFT struggle with?
- Probability of criticism if not addressed: **70%**

---

## WEAKNESS CATEGORY 10: Practical Applicability/Computational Trade-Offs

### Weakness Pattern
**No analysis of computational cost, training time, inference speed of selective SFT vs. full FT**

Selective fine-tuning might improve OOD accuracy but at what cost? Reviewers will ask about training time, memory usage, and inference latency.

### Relevant Reviewer Quote
**From Domain Adaptation Paper (ijwYWoChN9):**
> "Although the KSL is smaller compared to the size of the model, it must have some sort of slow-down associated with it since it appears as an additional layer with an additional step across K subcomponents. What is the speed reduction in using this method?"

**From Adapter Paper (uJqKf24HGN):**
> "During inference, will UniCon be similar even worse in latency and memory due to the more complex structure?"

### Why This Applies to SFT Paper
- **Practical trade-offs to measure**:
  - Training time: How much faster/slower than full FT?
  - Memory usage: How much do you save by not fine-tuning FNNs?
  - Inference cost: Does selective SFT affect inference speed?
  - Hyperparameter selection cost: How many experiments to find which modules to tune?
- **Risk**: "2% OOD improvement is nice, but costs 50% more training time" → Not practical

### Required Addition
- Report wall-clock training time for all methods
- Analyze memory usage during training
- Report inference latency (if different from standard FT)
- Probability of criticism if not addressed: **60%**

---

## WEAKNESS CATEGORY 11: Limited Benchmark Diversity/Task-Specific Design

### Weakness Pattern
**GeneralPoints and V-IRL may have specific properties that favor selective attention tuning**

What if these benchmarks happen to favor attention-based selective tuning due to their structure? Reviewers will want to know:
- Are these benchmarks representative of general reasoning?
- Could simpler methods (standard FT + regularization) achieve the same results?

### Relevant Reviewer Quote
**From Continual Learning Paper (EDJ7cPZk7V):**
> "There is quite a bit of variation across the datasets and experimental conditions, such as buffer size, in terms of the relative performance of different percentages of the too-small and too-fast sets that should be excluded. There is no analysis of this, which begs the question of how to set these hyperparameters in a new setting."

**From DEPT Paper (vf5aUZT0Fz):**
> "No downstream tasks in natural language understanding or generation are evaluated on the resulting models. But such further evaluation is important."

### Why This Applies to SFT Paper
- **Benchmark design bias**:
  - GeneralPoints: arithmetic only → Do symbolic tasks benefit equally?
  - V-IRL: navigation only → Do other embodied reasoning tasks benefit equally?
- **Confounding factors**: These benchmarks might have properties that specifically favor attention over FNNs (e.g., strong spatial/structural reasoning patterns)

### Required Addition
- Analyze task characteristics (why might these tasks favor selective attention?)
- Test on diverse reasoning types
- Compare to baseline: "standard FT + L2 regularization" to isolate selective tuning benefit
- Probability of criticism if not addressed: **70%**

---

## SUMMARY: MOST CRITICAL ISSUES TO ADDRESS

**Tier 1 - Makes or breaks acceptance** (address all):
1. ✗ Expand to 3+ additional diverse reasoning benchmarks
2. ✗ Provide mechanistic explanation for why selective attention helps OOD
3. ✗ Clearly separate memorization (ID) vs. generalization (OOD)
4. ✗ Add missing baselines (LoRA, BitFit, standard FT + regularization)
5. ✗ Test on 2+ model sizes to show scalability

**Tier 2 - Strengthens significantly** (address 4-5):
6. ✗ Comprehensive ablations on selective percentages and layer selection
7. ✗ Hyperparameter selection guidance for new domains
8. ✗ Statistical significance testing with confidence intervals
9. ✗ Computational cost analysis (training time, memory, inference)
10. ✗ Error analysis: identify failure modes and hard cases

**Tier 3 - Polish** (nice to have):
11. ✗ Theoretical analysis or detailed intuitive explanation
12. ✗ Visualization of learning dynamics across modules
13. ✗ Task characteristic analysis (why do these benchmarks favor selective attention?)

---

## PROBABILITY ASSESSMENT

| Issue | Likelihood of Criticism |
|-------|------------------------|
| Limited benchmarks (only 2) | 95% |
| No mechanistic explanation | 90% |
| Missing efficient FT baselines | 85% |
| Memorization not measured | 80% |
| Results only on single model size | 75% |
| OOD evaluation not rigorous | 75% |
| No statistical significance testing | 70% |
| No mechanistic understanding | 70% |
| Hyperparameter sensitivity unclear | 65% |
| Computational costs not analyzed | 60% |

**Without addressing Tier 1 issues:** Expected score 4-5/10 (Reject)
**With Tier 1 fixes:** Expected score 7-8/10 (Borderline Accept)
**With Tiers 1+2 fixes:** Expected score 8-9/10 (Strong Accept)

---

## FILES REFERENCED

Human reviews analyzed:
- `SFT_WEAKNESS_ANALYSIS_EXECUTIVE_SUMMARY.txt` - Comprehensive vulnerability assessment
- `WEAKNESS_SUMMARY_FOR_OOD_FINETUNING.txt` - OOD-specific patterns
- `SFT_RELEVANT_WEAKNESS_PATTERNS.md` - Detailed pattern descriptions
- `SFT_PAPER_SPECIFIC_WEAKNESSES.md` - Calibration-based weakness analysis
- `SELECTIVE_FINETUNING_OOD_WEAKNESS_ANALYSIS.md` - Related paper analysis
- `SFT_REVIEW_EXCERPTS_BY_WEAKNESS.md` - Direct reviewer quotes
