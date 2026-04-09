# Concrete Critique Points for SFT Paper
## Extracted from Related Papers on Fine-Tuning Dynamics

**Paper Being Reviewed:** "SFT Without Overfitting: Analyzing the Training Dynamics of Supervised Fine-Tuning"

---

## PAPER 1: TIC-LM (Continual Pretraining of Language Models)
**Source:** `/papers/MB53uAZKSc.txt`

### Evaluation Limitations that Apply:

1. **Domain-Specific Trade-offs (CRITICAL)**
   - TIC-LM Finding: Different domains evolve at different rates, requiring different optimization methods
   - SFT Paper Gap: Only evaluates on rule-based reasoning (GeneralPoints, V-IRL). No evidence that attention-only strategy generalizes to other domains with different adaptation characteristics
   - Specific Critique: "The paper assumes attention-only fine-tuning is universally superior, but TIC-LM demonstrates that method effectiveness varies by domain. No evidence provided that the proposed strategy works on temporally-evolving or rapidly-shifting distributions"

2. **Failure to Separate Forward Transfer from Forgetting Prevention**
   - TIC-LM Finding: Effective continual learning requires balancing new knowledge learning with old knowledge retention (explicitly measured as separate metrics)
   - SFT Paper Gap: Doesn't isolate performance gains from (a) adapting to new reasoning patterns vs. (b) preserving pre-training knowledge
   - Specific Critique: "Performance gains could stem from task-specific optimization rather than fundamental OOD preservation. The paper lacks metrics distinguishing forward transfer from stability"

3. **Insufficient Evaluation of Multi-Task Stability**
   - TIC-LM Finding: Data replay most effective (60% regret reduction) but degrades on fast-changing domains
   - SFT Paper Gap: Doesn't evaluate attention-only fine-tuning on sequential reasoning tasks where old task performance must be maintained
   - Specific Critique: "OOD evaluation limited to validation sets from same distribution. No evidence of performance on previously-learned reasoning types after fine-tuning on new types"

---

## PAPER 2: Catastrophic Forgetting via Function Vectors
**Source:** `/papers/gc8QAQfXv6.txt`

### Mechanism and Generalization Issues:

4. **Forgetting is Activation Bias, Not Parameter Overwriting (CRITICAL)**
   - Function Vector Finding: "CF in LLMs primarily stems from biases in function activation rather than the overwriting of task processing functions"
   - SFT Paper Assumption: Freezing FNN layers prevents overwriting of pre-trained knowledge
   - Specific Critique: "The paper's core assumption that freezing FNNs prevents forgetting is incorrect. Function Vector analysis shows the mechanism is activation bias shift, not parameter overwriting. Fine-tuning attention heads alone causes activation drift in frozen FNN responses, undermining the theoretical justification"

5. **Model-Dependent and Task-Dependent Results (CRITICAL)**
   - Function Vector Finding: "Model forgetting is influenced by both the specific training tasks and the models themselves"
   - SFT Paper Gap: How many models tested? How many task types? Generality unsupported
   - Specific Critique: "The finding (attention > FNN > full) may be specific to the reasoning tasks and models tested. No systematic study across model sizes, architectures, or task types. Claims of superiority are not generalizable"

6. **Missing Mechanistic Analysis of Fine-Tuning**
   - Function Vector Finding: FV framework enables tracking which attention heads shift and how activation patterns change
   - SFT Paper Gap: Provides zero mechanistic analysis. Claims about "preserving OOD generalization" are unvalidated
   - Specific Critique: "The paper provides no analysis of: (a) which specific attention heads change during fine-tuning, (b) whether pre-trained heads become inactive, (c) actual activation distribution shifts. Without this, the claim of 'preservation' is unfounded"

---

## PAPER 3: Knowledge Editing and Attention Drift
**Source:** `/papers/4l3AH8Bhmt.txt`

### Specificity Failure and Unintended Effects:

7. **Selective Fine-Tuning Creates Attention Drift (CRITICAL)**
   - Knowledge Editing Finding: When selectively editing parameters, attention heads develop "Attention Drift" - excessive focus on modified entities leading to systematic failures on related queries. Specificity Failure occurs in 50%+ of cases
   - SFT Paper Gap: No measurement of whether fine-tuned attention heads develop drift, causing misalignment on OOD examples
   - Specific Critique: "Attention-only fine-tuning is a form of selective parameter editing. The paper provides no evidence that attention drift doesn't occur, or that fine-tuned heads don't overfit to task-specific patterns while misinterpreting OOD examples"

8. **Unvalidated OOD Robustness (CRITICAL)**
   - Knowledge Editing Finding: Standard metrics miss specificity failures that appear in specific contexts (when edited knowledge appears in context)
   - SFT Paper Gap: Uses standard benchmarks but doesn't test edge cases, paraphrased versions, or adversarial examples
   - Specific Critique: "Performance on V-IRL and GeneralPoints test sets doesn't guarantee OOD robustness. The evaluation doesn't test: (a) paraphrased reasoning queries, (b) out-of-domain applications of reasoning patterns, (c) counterexamples where new reasoning conflicts with pre-training"

9. **No Measurement of Downstream Task Corruption**
   - Knowledge Editing Finding: Specificity failures appear in unrelated tasks after selective editing
   - SFT Paper Gap: Only tests on reasoning domain; doesn't evaluate whether fine-tuning corrupts other capabilities
   - Specific Critique: "No evidence provided that fine-tuned attention patterns don't degrade performance on language understanding, generation, or retrieval tasks. Selective fine-tuning of attention has broad effects not captured by task-specific benchmarks"

---

## PAPER 4: Implicit Layer Independence Assumption
**Source:** Derived from mechanistic interpretability literature referenced in Function Vector paper

### Architectural Mismatch:

10. **Attention-FNN Coupling Ignored**
    - Mechanistic Finding: Attention and FFN layers form integrated key-value memories; FNN outputs feed into attention, attention routing affects FNN queries
    - SFT Paper Assumption: Fine-tuning attention while freezing FNNs treats them as independent
    - Specific Critique: "The paper treats layer fine-tuning as independent, but fine-tuned attention heads will query frozen FNN states differently. This creates a mismatch: new attention patterns expect different FNN representations that can't adapt. Either (a) frozen FNNs become bottlenecks, (b) attention reverts to pre-training patterns, or (c) hidden states drift incompatibly. The paper doesn't analyze these failure modes"

11. **No Empirical Validation of Independence**
    - Missing Evidence:
      - Attention head output distribution doesn't diverge from original
      - FNN layer input distributions don't shift during fine-tuning
      - Cross-layer interaction terms don't degrade performance
    - Specific Critique: "The claim that attention-only fine-tuning 'preserves' functionality rests on an unvalidated assumption that frozen FNN layers still function correctly when queried by different attention patterns. No analysis provided"

---

## SYNTHESIS: 11 Specific, Evidence-Based Critiques

### Critical (Directly Invalidate Main Claims):
1. Forgetting mechanism assumption is incorrect per Function Vector analysis
2. Attention drift phenomenon not addressed despite being empirically documented
3. Model and task generality unsupported by results
4. Mechanistic understanding missing (which attention heads change?)

### Major (Undermine Evaluation Validity):
5. Domain specificity not tested across diverse task distributions
6. Forward transfer vs. forgetting prevention not separated
7. Multi-task stability not evaluated
8. OOD robustness not properly validated (specificity failures unmeasured)
9. Downstream task corruption not assessed

### Significant (Indicate Missing Related Work/Baselines):
10. Cross-layer dependencies (attention-FNN coupling) ignored
11. No comparison to theoretically-grounded selective methods (LoRA, adapters)

---

## Evidence Locations in Found Papers:

| Critique # | Paper | Specific Location |
|-----------|-------|-------------------|
| 1-3 | TIC-LM | "different domains evolve at different rates"; "forward transfer" vs "forgetting" metrics |
| 4-6 | Function Vector | "biases in function activation"; "model-dependent"; FV extraction methodology |
| 7-9 | Knowledge Editing | "Attention Drift phenomenon"; specificity failure metrics; "Distract Neighborhood" task |
| 10-11 | Multiple | Mechanistic interpretability literature; key-value memory architecture |
