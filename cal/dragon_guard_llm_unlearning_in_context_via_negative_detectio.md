=== CALIBRATION EXAMPLE 67 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is mostly accurate: the paper is about a prompting-based unlearning framework (“DRAGON”) that uses detection plus in-context reasoning. However, “guard LLM unlearning in context” slightly overstates the scope because the method is not unlearning in the parameter-update sense; it is a deployment-time mitigation/behavioral steering approach.
- The abstract clearly states the problem, the broad method, and the intended setting (black-box, no retain data, continual unlearning). It also names the proposed metrics.
- A key concern is that the abstract claims “extensive experiments” and “strong unlearning capability, scalability, and applicability in practical scenarios,” but the evidence is mostly on a limited set of benchmarks and relies heavily on synthetic or idealized components. The claim of “no retain data” is true for the method, but some baselines and evaluation setups still depend on retain information, and the paper sometimes compares against very favorable or partially idealized baselines.
- The abstract’s promise of “unlearning performance” is somewhat misleading for an ICLR audience: the method does not remove knowledge from the model, but filters and redirects outputs at inference time. That distinction should be made more explicitly in the abstract.

### Introduction & Motivation
- The motivation is good and relevant for ICLR: practical unlearning is hard when retain data is unavailable, model access is black-box, and requests arrive continuously.
- The gap in prior work is identified reasonably well: training-based methods often need retain data or parameter updates, while training-free methods are underexplored.
- The main contribution is clearly stated, but the paper sometimes conflates unlearning with runtime refusal/steering. For example, the introduction says the framework “protects the model through stepwise reasoning instructions” and “guard[s] deployed LLMs before inference,” which is useful, but it is not the same as unlearning in the standard sense.
- The introduction slightly over-claims novelty around metrics. The proposed RQ, DDS, and DUS are useful as diagnostics, but they are not obviously conceptually new enough to justify a strong methodological claim without stronger validation.
- For ICLR standards, the motivation is timely, but the paper needs to better delimit what problem it solves: output suppression and request filtering versus actual model unlearning.

### Method / Approach
- The method is described in enough detail to understand the pipeline: build an unlearn store, detect whether a query is forget-related, retrieve policy, generate CoT via a guard model, and prepend it to the input.
- The main weakness is conceptual: the method is not truly “unlearning” the model; it is a prompt-time intervention system. This matters because many claims and comparisons are framed as if the model has forgotten knowledge, when in fact it is being conditioned to refuse or redirect.
- The detection formulation in §4.1 is somewhat ad hoc. For TOFU/private-record unlearning, the confidence score combines exact name matching, embedding similarity, and a classifier probability (Eq. 4). For harmful knowledge, it combines a classifier and similarity metrics (Eq. 5). This is plausible, but the paper does not justify why these components should be combined additively, how thresholds are set, or how robust this is to paraphrase, obfuscation, and benign mentions of the same entities.
- The “unlearn store” is synthetic and created by paraphrasing forget prompts. That helps with privacy, but it also raises a circularity concern: the detector is evaluated on data derived from the same prompt distribution it was trained to recognize.
- The guard model is trained on synthetic CoT data produced by GPT-4o and paraphrases of TOFU prompts. This is a major dependency. The paper does not sufficiently justify why this trained guard model generalizes beyond the specific benchmarks used.
- The paper does not provide a convincing theoretical argument that CoT improves unlearning. The discussion in §E (“On the Theory Gap”) is heuristic only. That is acceptable for an empirical paper, but then the claims should be toned down accordingly.
- Failure modes are under-discussed. In particular:
  - If a harmful or private prompt is not detected, the base model answers normally.
  - If a benign prompt is falsely detected, the model may refuse unnecessarily.
  - The system’s security depends on the detector, which is itself trained and can be attacked or distribution-shifted.
  - Since the method does not alter model weights, any “forgetting” is only as strong as the prompt wrapper.
- The paper should be clearer that this is a safety/behavioral filter, not a replacement for true unlearning when the objective is removal of information from model parameters.

### Experiments & Results
- The experiments do test the paper’s claims to some extent: WMDP for harmful knowledge, TOFU for privacy record unlearning, and MUSE for copyright content.
- However, several comparisons are not fully fair or are too favorable to DRAGON:
  - Baselines like ICUL+ are explicitly evaluated in an “idealized setting” with knowledge of forget data, and the paper itself notes this. That makes the comparison less informative as a practical benchmark.
  - For prompting baselines, the paper often uses near-perfect classifiers or strong detector variants, which may strengthen those baselines in ways that are not always symmetric across methods.
  - Some methods are compared using checkpoints, others via tuned hyperparameters, others via synthetic data. The evaluation protocol is not fully standardized across methods.
- The reported results do show that DRAGON often achieves very low WMDP accuracy while preserving MMLU, and strong DS/MU on TOFU. But the strength of the conclusions is limited by the nature of the task:
  - On WMDP, if the detector catches the query, refusal is easy.
  - On TOFU, the model can often preserve utility because it is not being updated.
- The new metrics are the most questionable part of the experimental section:
  - RQ combines template similarity, refusal classifier output, and a gibberish score. This may reward superficial refusal style rather than robust, semantically appropriate behavior.
  - DDS and DUS are straightforward summary statistics of DS/utility trajectories, but their interpretation as principled continual-unlearning metrics is not established. They are useful diagnostics, but the paper overstates them as “novel metrics” without demonstrating that they correlate with meaningful real-world notions beyond the benchmarks.
- There are missing ablations that would materially matter:
  - No ablation cleanly isolates the contribution of the detector versus the CoT guard versus the safety policy retrieval.
  - No ablation on false-negative behavior: what happens when detection fails?
  - No calibration/ROC analysis of the detector, despite thresholding being central to the method.
  - No comparison to a simpler “refuse if detector fires” pipeline without CoT, beyond some partial ablations.
  - No report of variance/error bars or repeated-run statistical stability.
- The robustness results are encouraging, but some are weakly informative because they remain within synthetic perturbation families. For ICLR-level confidence, I would want more adversarial evaluation and stronger out-of-distribution tests.
- The MUSE/copyright section is especially underdeveloped. Table 14 is truncated in the parsed text, but based on the narrative, it seems less central and less thoroughly validated than TOFU/WMDP. If copyright unlearning is part of the claim, the paper needs a more complete and rigorous experimental presentation there.

### Writing & Clarity
- The paper is generally understandable, but several parts are hard to follow because the method mixes concepts from unlearning, safety filtering, and in-context refusal.
- The most confusing issue is the repeated conflation of “unlearning” with “refusal generation” and “response redirection.” Readers may not know whether the system is meant to remove knowledge or just intercept queries.
- The metric definitions in §3.2 and Appendix C are not always easy to parse, and the rationale for combining sub-scores is not fully explained.
- Some figures and tables likely suffer from parser/OCR artifacts, so I am not penalizing those. Ignoring those artifacts, the presentation still needs better conceptual organization. In particular:
  - §4 interleaves detection, store creation, policy retrieval, and guard-model training in a way that makes the architecture harder to reason about.
  - The relation between the main paper’s metric claims and the appendix details is not always transparent.
- The best parts are the high-level pipeline descriptions and the ablation sections, which do help readers understand how the system is supposed to work.

### Limitations & Broader Impact
- The paper does acknowledge some limitations, especially in the discussion section: black-box access, the fact that the method is not meant for fine-grained editing, multilingual limitations, and latency overhead.
- Still, it misses the most important limitation: the method does not actually erase the target knowledge from the underlying model. It enforces behavior at inference time. This is a fundamental limitation and should be foregrounded much more clearly.
- Another key limitation is detector fragility. The whole framework depends on correctly recognizing forget-worthy prompts. False negatives are a direct safety failure; false positives harm usability. This is a central weakness that deserves more explicit discussion.
- Broader societal impact is addressed, including the risk that unlearning could be abused to erase inconvenient facts. That is good. But there is little discussion of operational risks: logging, store maintenance, data governance, and whether synthetic paraphrases might still leak private content.
- The privacy and ethics story is mixed. The system uses external LLMs like GPT-4o to synthesize CoT data and paraphrases, which may be acceptable for harmful-knowledge tasks but is more delicate for privacy tasks. This deserves a sharper treatment.

### Overall Assessment
DRAGON is an interesting and practically motivated attempt to handle unlearning-like requests without retraining the base LLM, and it could be useful as a deployment-time safety layer. The strongest parts are the black-box friendliness, the continual-request framing, and the empirical evidence that a detector-plus-refusal pipeline can preserve utility while reducing harmful or private outputs. However, for ICLR standards, the central concern is that the paper overstates “unlearning”: the system does not remove knowledge from the model, it filters and steers responses. Many of the claimed gains therefore reflect prompt-time refusal rather than true forgetting. The detector, synthetic paraphrase store, and CoT guard are all plausible, but the method’s robustness and generality are not established as strongly as the paper suggests. I think the contribution is promising as a practical safety/prompt-guard framework, but not yet fully convincing as a substantive unlearning method.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DRAGON, a training-free, in-context unlearning framework for LLMs that combines prompt detection, an unlearn-store of paraphrased/synthetic forget prompts, and a fine-tuned guard model that generates CoT-style intervention prompts at inference time. The authors evaluate it on sample unlearning, concept unlearning, continual unlearning, and copyright-related settings, and also introduce new metrics for refusal quality and dynamic stability. Their central claim is that DRAGON achieves strong forgetting while preserving utility, without requiring retain data or modifying base-model weights.

### Strengths
1. **Addresses a genuinely practical setting that ICLR reviewers care about.**  
   The method targets black-box or deployment-constrained scenarios where retain data is unavailable and base-model fine-tuning is expensive or impossible. This is an important and timely problem in LLM unlearning, especially for proprietary models and continual deletion requests.

2. **Training-free intervention on the base model is a useful design choice.**  
   Rather than updating the target LLM, DRAGON performs detection plus prompt-level intervention, which improves compatibility with closed models and reduces deployment friction. The paper also reports low detector latency on some settings and emphasizes reuse across models and unlearning requests.

3. **The framework is modular and spans multiple unlearning regimes.**  
   The paper evaluates private-record unlearning, harmful-knowledge unlearning, continual unlearning, and copyrighted content unlearning. This breadth is valuable, and the same overall pipeline is reused across tasks, which suggests some degree of generality.

4. **The authors provide extensive empirical evaluation and ablations.**  
   The paper includes comparisons against several baselines, robustness checks, model-size sensitivity, CoT ablations, detection ablations, and threshold sensitivity studies. This is stronger than many unlearning papers that only show a narrow benchmark win.

5. **The paper attempts to improve evaluation, not just methods.**  
   Introducing Refusal Quality, Dynamic Deviation Score, and Dynamic Utility Score shows an effort to evaluate aspects of unlearning that static metrics may miss, particularly refusal coherence and stability under repeated unlearning.

### Weaknesses
1. **The core technical novelty is somewhat incremental and the framing is broad.**  
   DRAGON combines prompt filtering, similarity matching, a classifier, and CoT-based refusal prompting—each of which is individually familiar from prior work in detection, guardrails, and in-context prompting. The paper claims systematic unlearning, but the main mechanism appears to be a careful composition of known ingredients rather than a clearly new algorithmic principle.

2. **The evaluation appears heavily tailored to the method and its metrics.**  
   Several metrics, especially Refusal Quality, are custom-designed and depend on template similarity, a refusal classifier, and a gibberish detector. This raises concerns about whether improvements reflect genuine unlearning or better alignment with the paper’s scoring pipeline. For ICLR, reviewers will likely ask for stronger evidence that the method improves truly task-relevant behavior rather than metric-specific outputs.

3. **The paper’s claim of “unlearning” may be overstated for a prompt-based approach.**  
   DRAGON does not remove information from model weights; it intercepts prompts and steers outputs at inference time. That can be useful as a guardrail, but it is conceptually different from unlearning in the stricter sense of erasing knowledge from the model. The paper discusses this partially, but the main claims sometimes blur the distinction.

4. **Dependence on a separately trained guard model and detector adds hidden complexity.**  
   Although the base model is frozen, DRAGON still requires training a scoring model and a guard model, synthetic data generation, and a maintained unlearn store. This is not cost-free, and the operational burden may be understated relative to “training-free” claims. The need to retrain/adapt the detector and guard across task families also limits simplicity.

5. **Generalization and robustness claims are not fully convincing from the evidence shown.**  
   The paper reports robustness to certain paraphrases, typos, and language mixing, but the detection strategy may still be brittle in broader adversarial settings or under semantic drift. The detector’s reliance on paraphrased prompts and similarity thresholds also suggests potential recall/precision trade-offs not fully explored.

6. **Several comparisons are hard to interpret fairly across settings.**  
   The method is evaluated against strong baselines, but some baselines seem to operate under different assumptions, with access to retain data or idealized prompt construction. The paper does acknowledge this in places, but the comparison space is uneven, making it difficult to isolate the contribution of DRAGON itself.

7. **The reproducibility story is only partial.**  
   The appendix gives substantial implementation detail, but a reader would still need to reconstruct multiple synthetic-data generation pipelines, detector training regimes, threshold choices, and guard-model prompts. Because performance depends on generated paraphrases and LLM-produced CoT traces, exact replication may be sensitive to prompt wording and model versions.

### Novelty & Significance
**Novelty: Moderate.** The paper’s novelty lies more in the end-to-end packaging of detection + guard prompting for unlearning, plus the continual-unlearning metrics, than in a fundamentally new learning algorithm. The idea of using in-context prompts to suppress sensitive responses is related to prior in-context unlearning and guardrail work, though the dual-layer detector and CoT guard pipeline add a practical twist.

**Clarity: Moderate.** The high-level story is clear, but the paper sometimes overstates claims and uses many custom components and metrics, which makes the method harder to parse. The distinction between unlearning, refusal, and safety filtering should be sharper.

**Reproducibility: Moderate.** The appendix is fairly detailed, but the pipeline depends on multiple external models, synthetic data generation, and thresholded detection. That makes exact replication possible in principle, but somewhat brittle and prompt-sensitive.

**Significance: Moderate to good.** If the method works as claimed, it is practically relevant for black-box deployment settings where retraining is infeasible. However, because the base model is not actually altered, the significance to the core unlearning literature is somewhat narrower than the title suggests.

Overall, this is a solid and practically motivated paper, but on ICLR standards it likely sits near the border: strong empirical breadth and useful engineering, but only moderate algorithmic novelty and some unresolved concerns about what is being measured as “unlearning.”

### Suggestions for Improvement
1. **Sharpen the conceptual scope.**  
   Clearly distinguish “unlearning” from “prompt-time refusal/guardrail behavior.” If the method is primarily an inference-time safety wrapper, say so explicitly and frame claims accordingly.

2. **Strengthen evidence that improvements reflect real forgetting.**  
   Add evaluations that are less susceptible to refusal-style gaming, such as canary-style memorization probes, leakage tests under paraphrasing, or attack suites where the detector is bypassed without obvious lexical clues.

3. **Ablate the detector and guard components more thoroughly.**  
   Show the incremental contribution of each signal in the detector, the effect of synthetic vs. human-derived prompts, and whether the guard model alone or detector alone accounts for most gains.

4. **Compare more directly against strong inference-time safety baselines.**  
   Since DRAGON is training-free at the base-model level, comparisons should include stronger refusal and filtering baselines, not only unlearning methods that assume different access or update regimes.

5. **Analyze cost and failure modes more honestly.**  
   Report total system cost including synthetic data generation, guard-model training, and maintenance of the unlearn store. Also discuss cases where the detector misses semantically novel or adversarially rephrased forget requests.

6. **Validate the new metrics more rigorously.**  
   For Refusal Quality, show correlation with human judgments and robustness across different refusal styles and detectors. For DDS/DUS, test whether they align with external assessments of stability and utility over time.

7. **Improve fairness and transparency of benchmark setup.**  
   Ensure all baselines are compared under clearly matched assumptions, and provide a concise table specifying which methods have access to retain data, auxiliary models, or idealized forget prompts.

8. **Provide a simpler version of the method.**  
   An ablation that removes one or more moving parts could clarify whether the full pipeline is necessary, or whether a smaller subset of components achieves most of the benefit.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to the strongest recent LLM unlearning methods on the same benchmarks, especially methods explicitly designed for the data-limited setting (e.g., ECO/embedding-corrupted prompts, SNIP/negative-instruction style methods, and any black-box unlearning baselines). Without these, the claim that DRAGON is state-of-the-art for practical black-box unlearning is not convincing under ICLR standards.

2. Evaluate on a benchmark that tests forget-retain overlap or near-neighbor leakage, not just clean TOFU/WMDP splits. ICLR reviewers will expect evidence that the detector/intervention still works when forget and retain examples are semantically close, since that is where prompt-level unlearning usually fails.

3. Add end-to-end ablations isolating detection, retrieval, and CoT generation separately on all tasks. Right now it is unclear whether the gains come from the detector, the prompt template, or simply using a stronger guard model; without this decomposition, the method’s actual contribution is not identifiable.

4. Report false negative rates, attack success rates, and utility on the full distribution of non-forget prompts, not only aggregate accuracy/RQ. A detector with perfect recall on the benchmark but poor precision under realistic traffic would undermine the claim that DRAGON is deployment-ready.

5. Include a direct comparison against simply using the same guard model as a refusal/prompting baseline without the unlearn store and similarity retrieval. That baseline is the closest test of whether the extra machinery is needed at all.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze detector calibration and threshold sensitivity with ROC/PR curves, not just a few threshold points. The paper claims robustness to distribution shift and paraphrases, but without calibration analysis it is impossible to know whether the detector meaningfully generalizes or is just tuned to the benchmark.

2. Quantify how much the method depends on the quality of synthetic paraphrases generated for the unlearn store. If paraphrase generation is a hidden bottleneck, the claimed “data-limited” advantage weakens substantially.

3. Analyze whether the CoT guard model is actually reasoning about policy or merely learning a refusal script. The current evidence does not separate true contextual reasoning from pattern matching, which is critical because the paper’s core claim is “negative detection and reasoning.”

4. Provide statistical significance and variance across runs/models. The paper reports many single-point scores, but ICLR reviewers will expect confidence intervals or multiple seeds to judge whether improvements are stable or just benchmark noise.

5. Clarify the validity of the new metrics, especially RQ, DDS, and DUS, by correlating them with human judgments or established unlearning metrics. As written, these metrics could be optimizing artifacts rather than actual unlearning quality, which weakens the paper’s evaluation contribution.

### Visualizations & Case Studies
1. Show detector decision traces on clean, paraphrased, adversarial, and OOD prompts, including both true positives and false positives. This would expose whether the detector is learning the intended unlearning concepts or just exploiting superficial lexical cues.

2. Provide qualitative side-by-side generations for failure cases where DRAGON refuses incorrectly or misses a forget prompt. The paper currently shows only success examples, which does not reveal the real boundary conditions of the method.

3. Add a trajectory plot for continual unlearning showing utility and forget metrics over time for DRAGON and baselines. This is necessary to support the claimed stability advantage behind DDS/DUS.

4. Visualize nearest-neighbor matches in the unlearn store for detected prompts. That would reveal whether similarity retrieval is genuinely semantic or whether the method relies on brittle overlap with paraphrases.

### Obvious Next Steps
1. Test DRAGON on a true black-box API model without access to logits or internal embeddings. The paper claims black-box applicability, but the strongest evidence should come from a setting where only text input/output is available.

2. Extend evaluation to multilingual and cross-domain unlearning benchmarks with held-out concept families. The current language-mix perturbation tests are not enough to justify broad multilingual robustness.

3. Evaluate adaptive adversaries that optimize prompts specifically against the detector and guard prompt. ICLR reviewers will expect an adversarially trained or gradient-free attack analysis if the system is intended as a safety mechanism.

4. Compare against a minimal “refusal-only” system using the same refusal templates but no CoT or retrieval. This is the simplest next step to verify that DRAGON’s additional complexity is necessary.

5. Release a standardized benchmark for continual unlearning with paired detection and utility measurement. The paper introduces new metrics but does not yet establish a reproducible benchmark setting that the community can reuse.

# Final Consolidated Review
## Summary
This paper proposes DRAGON, a training-free, inference-time framework for LLM unlearning that first detects whether a prompt targets forgotten content and then injects a chain-of-thought guard prompt to induce refusal or safe redirection. The paper evaluates the system on privacy, harmful-knowledge, continual-unlearning, and copyright settings, and introduces new metrics for refusal quality and temporal stability.

## Strengths
- The paper targets a real deployment problem: black-box or access-limited LLMs where retain data is unavailable and retraining is impractical. The overall design is practical and modular, and the authors do show the framework can be reused across several model families and unlearning categories.
- The experimental breadth is decent for a prompt-level unlearning paper: TOFU, WMDP, continual unlearning, robustness tests, and a copyright-related benchmark are all covered, with ablations on CoT prompting, detection, thresholding, and guard-model choice. This is more complete than many papers in this area.

## Weaknesses
- The paper repeatedly overstates “unlearning” for what is fundamentally an inference-time guardrail / refusal system. DRAGON does not erase target information from model weights; it filters inputs and steers outputs at inference time. This distinction is central, and the current framing makes the core claim stronger than the method actually supports.
- The detector-and-guard pipeline is the whole method, yet its actual necessity is not isolated cleanly enough. The paper does not provide a truly minimal refusal-only baseline using the same guard model, nor a full component-by-component ablation showing how much comes from similarity retrieval, the classifier, the synthetic unlearn store, and CoT generation separately. Without that, it is hard to tell whether the “systematic” part is adding substantive value or just stacking heuristics.
- The evaluation is heavily benchmark- and metric-shaped. Refusal Quality, DDS, and DUS are custom metrics built from template similarity, classifier outputs, and trajectory summaries; they may reward surface-form refusal behavior rather than robust semantic unlearning. The paper does not validate these metrics against human judgments or strong external probes, so their substantive meaning is still weak.
- Robustness and generalization claims are not sufficiently stress-tested. The paper shows some paraphrase, typo, language-mix, and OOD results, but there is still little evidence against an adaptive attacker who actively tries to bypass the detector, or against harder forget-retain overlap settings where prompt-level unlearning usually struggles.
- Fairness of comparison remains imperfect. Several baselines rely on retain data, idealized forget prompts, or stronger access assumptions, and the paper acknowledges this. That makes some wins informative but not fully decisive for practical black-box unlearning.

## Nice-to-Haves
- A cleaner exposition that explicitly distinguishes “true unlearning” from “inference-time suppression/refusal.”
- Human validation of the new metrics, especially RQ, and correlation analysis for DDS/DUS.
- A refusal-only baseline using the same guard model, but without the unlearn store and similarity retrieval.

## Novel Insights
The most interesting aspect of the paper is not the claim of unlearning itself, but the attempt to turn unlearning into a prompt-routing problem: detect likely forget-related inputs, then condition the model with policy-aware CoT instructions. That is a plausible and potentially useful deployment pattern for closed models, especially in continual request settings where repeated retraining is expensive. However, the paper’s own results also reveal the limitation of this framing: once the detector fires, much of the observed success comes from controlled refusal behavior rather than any genuine removal of knowledge, so the method is best understood as a practical safety wrapper rather than a substantive unlearning algorithm.

## Potentially Missed Related Work
- ECO / embedding-corrupted prompts — relevant because the paper itself is close in spirit to prompt-level suppression and should be compared against stronger training-free unlearning baselines.
- Snap / negative-instruction style unlearning — relevant as another prompt-level, data-limited unlearning approach that should be a direct baseline.
- Guardrail baselines for unlearning in LLMs — already cited, but a tighter comparison to the strongest refusal-only variants would be especially important here.

## Suggestions
- Add a minimal baseline that uses the same guard model and refusal policy but removes the detector and unlearn-store retrieval, to quantify the value of the full pipeline.
- Report human judgments for refusal quality and stability, and correlate them with RQ, DDS, and DUS.
- Evaluate against a harder benchmark with explicit forget-retain overlap or semantically near-neighbor prompts, and include an adaptive detector-bypass attack suite.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0, 6.0]
Average score: 5.5
Binary outcome: Accept
