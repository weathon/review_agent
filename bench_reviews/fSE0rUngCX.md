## Summary
This paper introduces **Multimodal Policy Internalization (MPI)**: training a multimodal model to follow complex multimodal policies **without** supplying the policy at inference time. To support this setting, the authors contribute two benchmarks—**ClevrPolicy** for controlled, reasoning-heavy visual decision policies and **GTAPolicy** for multimodal tool-use rules—and propose **TriMPI**, a three-stage pipeline combining visually-masked continual pretraining, CoT SFT, and RL with **PolicyRollout**.

The work is clearly novel in scope: prior internalization/alignment work has largely focused on text-only settings or simpler prompt compression, whereas this paper targets multimodal, reasoning-intensive policies. Empirically, the paper shows large gains over its own no-policy internalization baselines, especially once RL is added, and it usefully evaluates not just task accuracy but also policy override, forgetting, and inference-time prompt savings.

## Strengths
- **The paper defines a genuinely new problem setting and supports it with task-specific benchmarks rather than only proposing an algorithm.** MPI is not just “prompt compression” in a new name: the policies here include multimodal content and reasoning-intensive rules, and the paper operationalizes this with two substantially different datasets:  
  - **ClevrPolicy**, which varies policy complexity by decision-tree depth and includes a multimodal-policy variant with image demonstrations inside the policy;  
  - **GTAPolicy**, which encodes tool descriptions plus versioning/user-conditional business rules for tool selection.  
  This benchmark construction is a meaningful contribution on its own.

- **ClevrPolicy is particularly well-designed for analysis of policy complexity.** The use of synthesized decision trees converted into natural-language policies gives unusually clean control over policy difficulty (e.g., \(N=2,4,6\)), and Table 1/Table 8 show that complexity systematically affects both in-context following and internalization performance. That makes the paper stronger scientifically than many agent papers that only evaluate on messy end tasks.

- **The proposed training decomposition is thoughtful and empirically informative.** The three stages are not redundant in the reported results: the paper ablates VM-CPT, RL, and PolicyRollout separately, and the results support the claim that **RL carries much of the gain**, while **VM-CPT and PolicyRollout add further improvement**, especially on harder settings. Even if one debates some design choices, the decomposition yields useful insight into what actually helps internalization.

- **The evaluation goes beyond end-task accuracy in ways that are specific to the paper’s claims.** In particular:  
  - **Policy Override** tests whether an internalized model can still follow updated in-context policy rules;  
  - **Policy In-Context** tests whether TriMPI improves policy following even when the policy is later supplied again;  
  - **catastrophic forgetting** is checked on MMMU-Pro, MMLU-Pro, and WildGuardTest;  
  - **efficiency** is quantified via prompt token reduction and prefill latency reduction.  
  These evaluations are aligned with the paper’s intended deployment story.

- **The efficiency motivation is concretely substantiated at inference time.** Figure 6 reports up to **93.9% prompt token reduction** and **85.7% prefill inference time reduction** once the policy is removed from the prompt. That is a specific and relevant systems-level benefit of internalization.

- **The paper surfaces a potentially useful RL idea.** PolicyRollout—augmenting the rollout pool with policy-aware responses while only optimizing the no-policy path—is a simple modification that appears empirically beneficial over vanilla GRPO/DAPO in their setting. Whether fully principled or not, it is the kind of practical training trick that could matter in future work on policy/internalization tasks.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper does not sufficiently disentangle “policy internalization” from ordinary task-specific adaptation.**  
  A central headline comparison is the large gain over the **in-context** setting, but the in-context numbers in Table 1 are from **zero-shot off-the-shelf models**, while TriMPI and even CoT SFT are trained on thousands of task examples. That makes the strongest “better than in-context” claim less persuasive as evidence for the necessity of the proposed internalization framework.  
  Concretely, the paper’s own results already show that much of the gain comes from supervised/RL adaptation to the benchmark tasks: e.g., on ClevrPolicy-M \(N=6\), **CoT SFT + DAPO = 74.40** while **TriMPI + PoRo-DAPO = 85.00** (Table 8). This suggests TriMPI is a meaningful improvement over task-tuned baselines, but the much larger comparison to zero-shot in-context prompting overstates what has been established.  
  The core contribution is still valid, but the paper should frame it more carefully as improving **internalization training** over strong no-policy baselines, rather than as a fully decisive superiority claim over prompted policy use.

- **Evidence for true policy abstraction remains limited, especially on GTAPolicy.**  
  GTAPolicy has only **451 training instances and 106 test instances** (Appendix C.2.2, Table 7). In such a small-data regime, strong performance could reflect a mix of policy learning and narrow task-specific fitting. The paper does include useful probes—especially **Policy Override**—but the current evaluation still falls short of decisively showing that the model has learned generalizable policy reasoning rather than memorizing policy-specific behavior patterns.  
  This matters because the paper’s conceptual claim is stronger than “we improved benchmark accuracy”: it argues that the model has **internalized policy knowledge**. More structurally novel rule tests or held-out policy families would make that claim much stronger.

- **PolicyRollout is not analyzed rigorously enough given its nonstandard objective.**  
  From the paper text, PolicyRollout concatenates no-policy and policy-conditioned rollouts into the same rollout space for group-based advantage estimation, while applying policy gradient only to the no-policy path. This is an interesting heuristic, but the paper does not provide enough analysis of its optimization behavior, variance, or potential bias.  
  The issue is not that the method is obviously invalid—the paper clearly states the intended mechanism in §4.3 and Eq. (3)—but that for a key algorithmic novelty, the empirical support is stronger than the conceptual explanation. The paper would benefit from deeper analysis of how mixed rollout groups affect reward normalization and whether the gain comes from better exploration, better ranking signals, or something else.

- **The efficiency argument is one-sided because training cost is not quantified.**  
  The paper convincingly measures inference-side savings, but TriMPI adds **three stages**, including full-parameter CPT and RL with large rollout batches on H100s (Appendix B/Table 5). For a paper motivated partly by efficiency, the omission of any training-cost accounting is noticeable.  
  This is not a contradiction—deployment may still amortize the upfront cost—but without a train-vs-infer tradeoff analysis, the practical case is incomplete.

### Minor
- **The “Policy Referral” evaluation is only weak evidence because it relies on LLM-as-a-judge scoring of reasoning traces.**  
  The paper is transparent about this setup (§5.3, Appendix I), and uses it only as an auxiliary probe rather than the main metric, which is appropriate. Still, it remains subjective and somewhat vulnerable to stylistic alignment effects rather than pure policy understanding.

- **The real-world side of the benchmark suite is still relatively narrow.**  
  ClevrPolicy is intentionally synthetic and analytically useful; GTAPolicy is more realistic but small and derived from a specific tool-use benchmark. As the authors themselves note in §7, broader real-world multimodal policy settings would strengthen external validity.

- **The method is only tested on one base model family (Qwen2.5-VL, 3B/7B).**  
  The scaling across 3B and 7B is useful, but architectural diversity is limited. This weakens claims about generality somewhat, especially for an algorithmic contribution centered on training dynamics.

- **VM-CPT’s visual masking is empirically motivated but only lightly justified.**  
  The paper acknowledges this simplicity (“it has shown empirical success despite its simplicity”), but for multimodal policies that can themselves contain visual components, it would be helpful to better understand what cross-modal knowledge this stage does and does not inject.

### Trivial
- **Some of the strongest claims in the abstract/introduction are broader than what the evidence fully supports.**  
  The paper is strongest when comparing TriMPI to other **trained no-policy internalization baselines**, less so when using zero-shot in-context prompting as the marquee contrast.

## Nice-to-Haves
- Add a **training cost vs. inference savings** analysis (FLOPs, GPU-hours, or wall-clock), ideally showing break-even points under different deployment volumes.
- Add a more targeted test of **policy abstraction**, e.g., held-out policy structures, unseen rule templates, or new decision-tree families rather than only modified policy content.
- Provide a deeper analysis of **PolicyRollout**: how mixed rollout groups are normalized, how reward statistics differ between policy/no-policy samples, and whether separate-baseline variants change results.
- Evaluate on at least one additional **natural-image / larger-scale tool-use** benchmark to strengthen claims of real-world applicability.
- Report **run variance or seed sensitivity** for RL stages, especially since Table 6 shows instability/early stopping differences across DAPO runs.
- Compare against a **lighter-weight parameter-efficient internalization alternative** (e.g., adapter/LoRA-only internalization) to better justify the full training overhead.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The method is unreproducible because some cited models/tools may not be available or verifiable.”**  
  Removed under instruction: if cited in the paper, these entities are assumed to exist and be available.

- **Generic reproducibility complaints about omitted trivial implementation details.**  
  Removed because the paper actually provides substantial implementation detail in Appendix B/Table 5/Table 6, including learning rates, epochs, batch sizes, rollout batch sizes, KL coefficient, and hardware.

- **Claims that the paper lacks any baselines or any ablations.**  
  Factually incorrect. The paper includes Direct SFT, CoT SFT, CoT SFT + GRPO, CoT SFT + DAPO, and ablations over VM-CPT and PolicyRollout in Table 2/Table 8.

- **Formatting/style issues from PDF extraction.**  
  Removed as parser artifacts rather than paper weaknesses.

- **Criticism that Table 4 itself proves the method is not internalization because models improve when given policy in-context.**  
  Overstated. Table 4 only shows that TriMPI improves policy-following competence even when policies are later reintroduced; it does not invalidate internalization. At most, it suggests some gains may reflect broader task competence in addition to policy embedding.

## Novel Insights
The most interesting synthesis across the reviews and the paper is that **the strongest empirical story here is not “internalization beats prompting,” but rather “RL-based no-policy training can recover policy-following behavior far better than SFT alone, and explicit access to policy during training further improves that recovery.”** In other words, the paper’s real contribution may be less about proving a strict replacement for in-context policies and more about identifying a workable recipe for turning long, reasoning-heavy multimodal policies into latent behavior. The results also suggest a useful qualitative distinction: **ClevrPolicy diagnoses policy-complexity scaling cleanly, while GTAPolicy exposes the brittleness of low-data internalization**, making the two datasets complementary in a way that strengthens the benchmark contribution.

## Suggestions
- Reframe the headline claims to emphasize **improvement over trained no-policy baselines** rather than large gains over zero-shot in-context prompting.
- Add a controlled experiment where the policy is changed **structurally**, not just via content override, to better separate memorization from policy reasoning.
- Analyze PolicyRollout with an alternative variant that computes group statistics separately for policy-conditioned and no-policy rollouts, to test whether the current mixed grouping is essential.
- Include a quantitative **training-cost amortization** discussion to support the efficiency motivation.
- Strengthen the real-world evidence with either a larger GTAPolicy-style dataset or another benchmark involving natural images and richer tool-use policies.
- Report multi-seed results, or at least variance for RL stages, to clarify robustness of the claimed gains.

