## Summary
This paper proposes SpikeZIP, an ANN-to-SNN conversion framework aimed at improving the accuracy/latency Pareto frontier for low-timestep SNN inference. Its two main ingredients are Paths-Ensemble Training (PET), which trains a quantized ANN across multiple quantization levels using shared parameters, and a claimed model-level equivalence result between the trained QANN and the converted SNN under specific architectural and encoding conditions.

## Strengths
- **Strong empirical results on standard ANN-to-SNN conversion benchmarks.** On ImageNet, the paper reports 73.92% with VGG-16 at 7/9 timesteps and 74.21% with ResNet-34 at 11 timesteps, improving over prior reported operating points in Table 3. The paper also evaluates on CIFAR100 and includes both VGG and ResNet families.
- **PET is a meaningful practical contribution.** The multi-path training setup with shared weights, path-specific BN statistics, and shared/scaled quantization parameters is well motivated and backed by nontrivial ablations in Table 5 and Fig. 6. This is the most convincing methodological contribution in the paper.
- **The ablation section is fairly thorough.** The paper examines PET vs. RCR, quantization levels, parameter-sharing strategies, loss weighting, and label choices. These experiments help establish that the final recipe is not arbitrary.
- **The paper addresses an important problem.** Reducing ANN-to-SNN latency while preserving accuracy is a relevant and practically important objective for neuromorphic inference.
- **The writing is reasonably clear at the high level.** The overall pipeline is understandable, and the paper makes a real effort to connect theory, architecture changes, and empirical results.

## Weaknesses

###: Fatal
- **The core theorem claiming whole-model QANN-SNN equivalence is not actually established at the level claimed.**  
  This is the paper's most serious issue because one of the two headline contributions is a “mathematically equivalent conversion algorithm between the whole QANN and SNN.” However, Proof 3.1 only shows a local correspondence by matching one fused block to the closed-form ST-BIF response:
  > “By setting \( \mathbf{V}^{in} = \tilde{\mathbf{W}}_l \cdot \mathbf{X}_{l-1} + \tilde{\mathbf{b}}_l \), \(V_{t=0}=0.5V_{thr}\), \(S_{max}=n\), \(V_{thr}=s\), eq. (7) and eq. (9) are equivalent. By extending the equivalence between blocks to the network, the eq. (8) is proven.”
  
  That last sentence skips the actual hard part. In the SNN, deeper layers receive temporally distributed spike sequences from earlier layers, not directly the static ANN activation \( \mathbf{X}_{l-1} \). The proof does not rigorously show, layer by layer, that the accumulated input arriving at each downstream SNN block equals the corresponding QANN pre-activation under the proposed encoding schedule, nor does it formally handle residual paths in the network-level induction. As written, the proof supports a block/neuron-level mapping under stated settings, but not a fully rigorous network-level theorem of exact model equivalence.

### Major:
- **The paper overstates the empirical support for exact equivalence.**  
  Section 4.2 presents Figure 5 as experimental confirmation of equivalence, but the evidence is narrow: one architecture family, 100 ImageNet samples, one feature-map metric, and the reported L1 gap is still 0.5 rather than 0. The paper writes:
  > “only SpikeZIP shows the equivalence both theoretically ... and experimentally e.g., L1 distance is 0.5.”
  
  That is too strong. At best this is evidence of close agreement on one selected setting, not a demonstration of exact whole-model equivalence.
- **The practical headline results emphasize low-timestep inference at \(T < T_{eq}\), whereas the equivalence claim is about equilibrium.**  
  The paper itself notes:
  > “The peak accuracy of SNN is not achieved at the \(T_{eq}\) but a time-step \(T < T_{eq}\)”  
  This creates an unresolved gap between the theorem and the main practical claim. The theory is meant to hold at equilibrium, but the reported Pareto-front improvements rely on earlier operating points before equilibrium. The paper does not adequately analyze this relationship or report the key quantity directly: whether SNN accuracy at \(T_{eq}\) matches QANN accuracy across models.
- **The state-of-the-art / Pareto-front comparison is somewhat confounded by architectural and training changes beyond conversion.**  
  SpikeZIP does not only change the conversion rule. It also performs “SNN-friendly morphing,” including replacing max-pooling with average-pooling and, for ResNets, residual connection re-routing (RCR), plus PET-based retraining. The paper does ablate PET and RCR, but it does not fully isolate how much of the claimed gain over prior methods comes from these architecture/training modifications versus the conversion/equivalence mechanism itself. This weakens the breadth of claims like “better Pareto frontier with SpikeZIP” when interpreted as a pure conversion-method superiority claim.
- **The novelty framing around model-level equivalence is broader than what the paper itself substantiates.**  
  The abstract/introduction says existing work fails to provide theoretical equivalence between QANN and converted SNN, but Table 2 marks several prior works as having model equivalence. The body later narrows the claim to analog encoding with their particular assumptions, which is more plausible. The paper should be much more precise here; otherwise the novelty claim reads broader than the paper's own framing supports.

### Minor
- **Comparison to learning-based SNN methods is of limited interpretive value.**  
  These methods solve a related but different problem, and SpikeZIP benefits from pretrained ANN initialization and conversion. The training-cost comparison in Fig. 4 is interesting, but it is not a clean apples-to-apples comparison of methodology.
- **The object detection experiment is only modest evidence of generality.**  
  The experiment is encouraging, but it uses a modified YOLOv3 backbone and compares against a small set of prior conversion results on different architectures/settings. This is supportive, not conclusive, evidence of broader applicability.
- **Hyperparameter/recipe complexity is nontrivial.**  
  PET introduces several coupled design choices: path quantization levels, parameter-sharing schemes, loss weighting \(\alpha\), and label type. The ablations help, but the method is still fairly recipe-heavy and practical guidance is limited.
- **The claimed explanation for the nonzero L1 gap (“intrinsic computing error of GPU hardware”) is not verified.**  
  This is not a major flaw by itself, but it is an overconfident attribution that would need stronger validation.

### Trivial
- None.

## Nice-to-Haves
- Report \(T_{eq}\) explicitly for all main models and show whether SNN accuracy at \(T_{eq}\) matches QANN accuracy.
- Add a cleaner decomposition of gains: morphed architecture only, +RCR, +PET, +equivalence conversion.
- Test at least one larger or more modern architecture to strengthen generality claims.
- Provide a more careful discussion of when architectural morphing may or may not preserve the original ANN's useful inductive biases.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Requests for confidence intervals / repeated runs on close benchmark margins.**  
  Single-run reporting is common in this benchmarking setting, so while variance would be helpful, its absence is not a core flaw here.
- **Complaints that baseline comparisons are “unfair” because SpikeZIP has more favorable architectural changes.**  
  This criticism must be handled carefully: the comparisons can still be informative even if asymmetric. The real issue is not unfairness per se, but that the paper sometimes attributes gains too broadly to the conversion framework without fully disentangling architecture/training effects.
- **Doubts about existence, availability, or reproducibility of cited systems or prior results.**  
  These are not valid criticisms under the review rules.
- **Pure scope-creep requests to evaluate transformers/LLMs because the primary area is foundation models.**  
  The paper is clearly an SNN conversion paper; lack of LLM evaluation is not itself a flaw.

## Novel Insights
The paper is strongest when viewed not as a definitive theory paper, but as a practical conversion-and-training recipe with one especially useful idea: PET effectively regularizes the major-path QANN toward lower quantization regimes that correspond to lower SNN timesteps. That practical contribution is real and well supported. The paper becomes much weaker when it elevates this into a rigorous whole-network equivalence claim, because the proof as written does not bridge the temporal dynamics of intermediate spike propagation across layers. In short: the empirical method looks publishable-quality; the theorem, in its current form, does not.

## Suggestions
- **Rewrite the theory claim more narrowly unless a full inductive proof is added.** Either provide a rigorous network-level proof handling temporal propagation and residual structures, or weaken the claim to block-level equivalence plus conditions under which network behavior is expected to align.
- **Report \(T_{eq}\) and accuracy at \(T_{eq}\) for all main models.** This is the most direct validation of the claimed theorem and is currently missing.
- **Disentangle empirical gains more cleanly.** Add results for: morphed QANN without PET, morphed + PET, morphed + RCR, and full SpikeZIP, so readers can see which component is responsible for what.
- **Temper the language around “only SpikeZIP shows equivalence.”** The current empirical evidence does not justify that level of certainty.
- **Clarify novelty relative to prior “model equivalence” works.** If the intended novelty is equivalence under analog encoding and the specific SpikeZIP conditions, state that precisely in the abstract/introduction.
- **Strengthen the discussion of scope and limitations.** In particular, explain where RCR / pooling replacement may be necessary, and where the method may be less applicable.

## Score and Decision
**Assessment across axes:**  
- **Originality:** Moderate. PET is a solid practical contribution; the theoretical angle is less convincing in its current form.  
- **Importance:** Good. Low-latency ANN-to-SNN conversion is a worthwhile problem.  
- **Claims supported?:** Mixed. The empirical improvement claims are fairly well supported; the central whole-model equivalence claim is overstated relative to the proof and evidence.  
- **Experimental soundness:** Good but not airtight. The experiments are broad for this subarea and ablations are strong, but the decomposition of gains is incomplete.  
- **Clarity:** Reasonably clear overall.  
- **Community value:** Moderate to high if framed primarily as a practical conversion method rather than a rigorous theory result.

**Calibration:** I compared this paper against lower-scoring human-reviewed papers with theory/claim mismatches or overreach, such as:
- `D4sQzdMvcG.md` (QAC: Quantization-Aware Conversion for Mixed-Timestep SNNs; human scores 5/6/6/6, reject): similar area, practical value but limited by how strongly claims are established.
- `CwAY8b8i97.md` (Spike Accumulation Forwarding; scores 5/3/3/5, reject): weaker overall than this submission, with less compelling empirical support.
- `mtmqwhQiaG.md` (Canonic Signed Spike Coding; scores 3/6/6/6, reject): also in SNNs, with meaningful ideas but notable concerns.

I also anchored against stronger accept-range papers in the database that had clearer support for their main claims and fewer theory-evidence gaps. Relative to those, this paper falls short because one of its two headline contributions is not convincingly established. Relative to weaker rejects, this paper is stronger due to its substantial experiments and useful PET contribution. That places it in a borderline-to-weak-reject range rather than a hard reject.

**Final score: 4.5 / 10 — Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>