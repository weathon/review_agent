# Weakness Analysis for SFT Paper
## Selective Fine-Tuning of LLMs for Attention-Only Layer Updates

**Reviewed Paper:** "SFT Without Overfitting: Analyzing the Training Dynamics of Supervised Fine-Tuning"
- **Main Claims:** Attention-only SFT preserves OOD generalization better than FNN-only or full fine-tuning
- **Evaluation:** Rule-based reasoning tasks (GeneralPoints, V-IRL)
- **Key Findings:** Attention-only SFT achieves comparable performance to RL methods

---

## 1. TIC-LM: Multi-Year Benchmark for Continual Pretraining
### Paper: `/home/wg25r/review_agent/iclr2025_data/papers/MB53uAZKSc.txt`

### Critical Weakness Patterns Applicable to SFT Paper:

#### A. **Limited Time-Horizon Evaluation**
- **Finding:** Different domains evolve at different rates, requiring different optimization trade-offs
- **Application to SFT:** The SFT paper only evaluates on rule-based reasoning tasks (GeneralPoints, V-IRL), which may have consistent semantic patterns
- **Critique Point:** No evidence that attention-only SFT strategy generalizes across multiple domains with different rates of knowledge evolution. The paper may be overfitting the solution to the specific task characteristics rather than discovering a general principle
- **Specific Evidence:** TIC-LM shows "different domains favor different trade-offs between mixing old and new data" - suggesting selective fine-tuning effectiveness may be task-specific, not universal

#### B. **Evaluation Bias Toward Simple Forward Transfer**
- **Finding:** Data replay is most effective (60% regret reduction) but hurts on rapidly-changing domains
- **Application to SFT:** The SFT paper's claim that attention-only fine-tuning preserves OOD generalization is not tested on:
  - Tasks with high distributional shift from pre-training data
  - Sequential learning scenarios where knowledge accumulation is required
- **Critique Point:** The evaluation domains (reasoning tasks) may represent "stable" knowledge distributions where selective updates appear effective, but the paper doesn't test on time-evolving or rapidly-shifting domains

#### C. **Methodological Limitation: No Forward Transfer Assessment**
- **Finding:** TIC-LM distinguishes between forward transfer (learning new data) and forgetting (retaining old knowledge)
- **Application to SFT:** The SFT paper compares to "full fine-tuning" and "FNN-only" but doesn't explicitly separate:
  - How much performance improvement comes from adapting to new data (forward transfer)
  - How much comes from preserving pre-trained knowledge (preventing forgetting)
- **Critique Point:** Attention-only updates might preserve knowledge at the cost of insufficient plasticity for genuinely novel reasoning patterns not in pre-training

---

## 2. Catastrophic Forgetting in Continual Instruction Tuning (Function Vector Analysis)
### Paper: `/home/wg25r/review_agent/iclr2025_data/papers/gc8QAQfXv6.txt`

### Critical Weakness Patterns Applicable to SFT Paper:

#### A. **Forgetting Mechanism Mischaracterization**
- **Finding:** Catastrophic forgetting in LLMs stems from "biases in function activation rather than overwriting of task processing functions"
- **Application to SFT:** The SFT paper assumes that freezing FNN layers prevents overwriting pre-trained knowledge
- **Critique Point:** By the Function Vector analysis, the problem is not parameter overwriting but activation bias shifts. Freezing FNNs does NOT address the core mechanism - attention heads can still develop biased activation patterns when fine-tuned
- **Specific Implication:** The paper's reasoning that "attention fine-tuning is safer than FNN fine-tuning" is based on an incomplete understanding of forgetting mechanisms. Both attention and FNN updates can cause activation drift

#### B. **Model and Task Dependence Ignored**
- **Finding:** "Model forgetting is influenced by both the specific training tasks and the models themselves" - not universal
- **Application to SFT:** The SFT paper doesn't establish that attention-only fine-tuning is universally better across:
  - Different model architectures (tested on how many?)
  - Different types of instruction tuning tasks
- **Critique Point:** The finding (attention > FNN > full) might be specific to reasoning tasks and the particular models tested. Generalizations are unsupported by systematic ablation

#### C. **Incomplete Mechanistic Analysis**
- **Finding:** Function Vector framework reveals which attention heads encode task-relevant information and which shift during fine-tuning
- **Application to SFT:** The SFT paper provides no analysis of:
  - Which specific attention heads are being modified during selective fine-tuning
  - Whether some pre-trained attention heads become inactive (degradation) vs. reused
  - The actual activation patterns before/after fine-tuning
- **Critique Point:** Without mechanistic analysis, the claim that attention-only fine-tuning "preserves" OOD generalization is unvalidated. Attention heads could be repurposed rather than preserved

---

## 3. Knowledge Editing and Attention Drift Problem
### Paper: `/home/wg25r/review_agent/iclr2025_data/papers/4l3AH8Bhmt.txt`

### Critical Weakness Patterns Applicable to SFT Paper:

#### A. **Selective Fine-Tuning Creates Attention Drift**
- **Finding:** When editing model parameters (even with careful localization), attention heads develop "Attention Drift" - excessive focus on edited entities leading to specificity failure
- **Application to SFT:** Attention-only fine-tuning is a form of selective parameter editing
- **Critique Point:** Fine-tuning attention layers creates exactly the conditions for Attention Drift. The paper provides no evidence that:
  - Attention heads don't develop drift during fine-tuning
  - The model doesn't overly focus on reasoning patterns from the new task
  - OOD examples aren't misinterpreted due to attention miscalibration
- **Key Risk:** Empirically, 50%+ of edited models show specificity failure even with careful methods. Selective attention fine-tuning faces the same risk

#### B. **Unintended Downstream Effects Not Measured**
- **Finding:** Specificity Failure in knowledge editing occurs when "the model's attention heads assign excessive attention scores to edited entities" - causes appear in unrelated tasks
- **Application to SFT:** The SFT paper doesn't evaluate OOD generalization on:
  - Tasks semantically related to the reasoning domain
  - Tasks that might rely on attention patterns modified during fine-tuning
- **Critique Point:** The paper's OOD evaluation (V-IRL, GeneralPoints) may not cover domains sensitive to attention pattern corruption. True OOD generalization requires testing on tasks minimally impacted by attention fine-tuning

#### C. **Evaluation Metrics Insufficient for Specificity**
- **Finding:** Standard fine-tuning metrics (accuracy, loss) miss specificity failures that appear in specific contexts
- **Application to SFT:** The SFT paper uses standard benchmarks but doesn't test:
  - Cases where edited knowledge appears in the context (counterexamples)
  - Whether reasoning applies correctly to edge cases outside the training distribution
  - Paraphrased versions of reasoning tasks
- **Critique Point:** Performance gains on V-IRL/GeneralPoints don't guarantee true OOD generalization if specificity failures go unmeasured

---

## 4. Implicit Assumption: Layer Independence
### Derived from Catastrophic Forgetting Literature

#### A. **Cross-Layer Knowledge Integration Assumption**
- **Assumption in SFT paper:** Freezing FNN layers while fine-tuning attention layers allows independent specialization
- **Contradiction:** Recent mechanistic work shows attention and FNN operate as integrated key-value memories
  - Attention outputs depend on FNN hidden states (from previous layers)
  - FNN outputs depend on attention context routing
- **Critique Point:** The paper treats attention fine-tuning as isolated, but attention patterns fine-tuned on new tasks will expect different FNN representations, creating a mismatch:
  - Fine-tuned attention heads learn to query frozen FNN states differently
  - This creates implicit pressure toward FNN change (but constrained), degrading adaptation
  - Frozen FNNs can't respond to new attention patterns, causing bottlenecks

#### B. **No Validation of Independence Assumption**
- **Evidence Needed:** Analysis showing:
  - Attention head outputs don't diverge from their original distribution
  - FNN layer inputs don't shift significantly despite frozen parameters
  - Interaction terms between fine-tuned and frozen layers don't degrade performance
- **Gap:** Paper provides none of this. Claims about "preserved OOD generalization" rest on an unvalidated assumption

---

## 5. Evaluation Limitations

### A. **Narrow Task Domain**
- **Tasks:** GeneralPoints (rule-based), V-IRL (value function reasoning)
- **Missing:**
  - Symbolic reasoning tasks requiring novel attention patterns
  - Knowledge-intensive tasks where attention to correct context is crucial
  - Adversarial OOD examples testing robustness
  - Cross-domain transfer (e.g., pre-trained on code, fine-tuned on math)
- **Critique:** Rule-based reasoning is a "sweet spot" for selective fine-tuning - tasks with stable attention patterns. The paper doesn't demonstrate generality

### B. **No Comparison to Layer-Scaling Baselines**
- **Missing:** Comparison to:
  - Low-rank adaptation (LoRA) applied selectively to layers
  - Layer-wise learning rate scaling
  - Adapter modules (which also selectively fine-tune)
- **Why It Matters:** These baselines might achieve the same OOD preservation with more principled trade-offs

### C. **Insufficient OOD Definition**
- **Assumption:** Held-out test set from same task distribution = OOD
- **Reality:** True OOD requires distributional shift (domain shift, syntax shift, semantic shift)
- **Critique:** V-IRL and GeneralPoints validation sets may be in-distribution relative to training. The paper conflates test generalization with OOD robustness

---

## 6. Alternative Explanations Not Ruled Out

### A. **Scale-Dependent Phenomena**
- **Alternative:** Attention-only fine-tuning works because:
  - Model size is below where FNN fine-tuning becomes beneficial
  - Task complexity is below threshold where full fine-tuning needed
- **Not tested:** Does the result hold for larger models? More complex tasks?

### B. **Task-Specific Attention Patterns**
- **Alternative:** Reasoning tasks inherently change attention patterns more than feed-forward representations
- **Not tested:** Do FNN-only and attention-only trade-offs reverse on language generation tasks?

### C. **Learning Rate Artifacts**
- **Alternative:** Results reflect learning rate tuning for different components, not fundamental differences
- **Not tested:** Do carefully tuned (full fine-tuning with adjusted learning rates) baselines match selective approaches?

---

## Summary of Concrete Critique Points for Review

| Issue | Evidence | Specific Critique |
|-------|----------|-------------------|
| **Limited generality across domains** | TIC-LM: different domains need different methods | No evidence attention-only strategy works on time-evolving or rapidly-shifting domains |
| **Forgetting mechanism misunderstood** | Function Vector analysis: forgetting is activation bias, not parameter overwriting | Freezing FNNs doesn't prevent activation drift in attention heads |
| **Attention drift not addressed** | Knowledge editing: selective edits cause attention drift in 50%+ cases | Fine-tuning attention creates specificity failures unmeasured in paper |
| **Cross-layer dependencies ignored** | Mechanistic work: attention-FNN integration through key-value memories | Independent layer tuning violates transformer architecture assumptions |
| **Insufficient OOD evaluation** | Catastrophic forgetting: OOD requires true distributional shift | Test sets may be in-distribution; V-IRL/GeneralPoints don't test specificity |
| **No mechanistic validation** | Function Vector framework available | Paper provides no analysis of which attention heads change or how activation patterns shift |
| **Narrow evaluation** | Only rule-based reasoning | No evaluation on language generation, code, or adversarial examples |
| **Missing baselines** | LoRA, adapters exist | Doesn't compare to other selective fine-tuning methods with better theoretical grounding |

