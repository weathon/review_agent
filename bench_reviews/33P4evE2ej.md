## Summary
This paper proposes **DynaMer Adapter**, a parameter-efficient adaptation method that jointly uses a general-domain ViT and a medical-domain ViT by dynamically merging their token representations through a **shared gated MoE adapter** and a **layer-wise skipping router**. Empirically, the method delivers consistently best average results across Med-VTAB, patient-split evaluations, and some general-domain transfer benchmarks, while using fewer tunable parameters than prior dual-expert adapter variants.

## Strengths
- **The central design is more specific than a routine adapter tweak:** rather than adapting one backbone, the paper explicitly targets the practically relevant case where **general and medical pretraining provide complementary strengths**, and operationalizes this via **token-level cross-model fusion** with a shared MoE adapter plus per-domain gating. This is a concrete architectural idea, not just “add another adapter.”
- **The paper shows unusually broad benchmark coverage within its target setting.** Results are reported across **23 medical datasets** spanning color, X-ray, and OCT/CT/MRI tasks (Tables 1–3), plus **patient split evaluations** (Tables 8–9) and **general-domain transfer** on FGVC/VTAB-1K (Table 10). The wins are not isolated to one modality.
- **DynaMer is consistently stronger than the closest adapter baselines while using fewer added parameters than prior dual-model MoE variants.** For example, compared to GMoE-Adapter, DynaMer improves nearly every reported dataset while using **1.21X total params vs. 1.35X** in Tables 1–3 and 10. That supports the claim that the shared adapter design is parameter-efficient.
- **The ablations do verify that some proposed components matter.** Tables 4–6 show gains from adding gates on both streams and applying them across more layers; Table 7 shows a meaningful internal speed/accuracy tradeoff, with **50% token processing reducing batch inference time from 0.165s to 0.086s** while slightly improving accuracy on the color-image suite.
- **The generalization beyond purely medical evaluation is at least partially supported.** Table 10 shows that the same machinery remains competitive and slightly best on FGVC and VTAB-1K, which strengthens the paper’s claim that the token-merging principle may extend beyond the immediate medical setting.

## Weaknesses
### Fatal
- None.

### Major:
- **The paper’s efficiency claims are only partially substantiated, because the reported timing evidence is not comparative and does not account for the cost of using two backbones.**  
  The paper repeatedly makes strong efficiency claims: e.g., the abstract says the skipping router “optimiz[es] inference time,” and the introduction claims DynaMer “achieves few costs in both training and inference time.” However, the only explicit wall-clock evidence is **Table 7**, which reports inference time **only for DynaMer at different token-retention ratios**. There is **no direct runtime/FLOPs/memory comparison against baselines** such as Adapter, MoE-Adapter, or GMoE-Adapter, despite Figure 1 visually positioning the method as comparatively efficient. This matters especially because the system runs **both a general ViT and a medical ViT**, which is a real architectural cost not quantified in the paper. Parameter-count savings alone are not sufficient to support the broader efficiency narrative.
- **Several headline claims are stronger than the empirical evidence warrants, especially for patient-OOD and few-sample performance.**  
  The abstract and introduction claim DynaMer “particularly excel[s] in patient out-of-distribution settings and tasks with only few samples.” The paper does show improvements in Tables 8–9 and Figure 1(c), but the margins over the strongest baselines are generally **small** (often only a few tenths of a point), and there is **no uncertainty reporting** anywhere. Likewise, the few-shot/data-efficiency evidence is concentrated in **Figure 1(c)** without a corresponding detailed table or protocol description. Given the strength of the wording (“particularly excelling”), the empirical support is not fully commensurate with the claim.
- **The method description is underspecified in a few places that matter for interpretation of the architecture.**  
  This is not a minor implementation-detail complaint; several core aspects are genuinely unclear from the current text:
  - Sec. 3.2 introduces top-\(k\) expert routing but does not clearly specify the actual values of **number of experts \(n\)** and **selected experts \(k\)** in the paper body.
  - The interaction between the two streams is not fully transparent: the method assumes token-wise pairing \((\mathbf{x}_{\text{gen},i}, \mathbf{x}_{\text{med},i})\), but the exact alignment assumptions across the two frozen ViTs are not discussed.
  - Sec. 3.3 claims the skipping router can reduce the number of tokens processed in deeper layers, but **Eq. (4)** only explicitly partitions/reorders tokens into selected and skipped subsets. The text says skipped tokens “will skip the adapter” and that processed and skipped tokens are concatenated and sent to the next layer, which makes the exact computational savings mechanism somewhat ambiguous unless one infers that skipping applies only to the adapter path, not the full transformer.
  - There is also a genuine inconsistency between Sec. 3.3 (“They are optimized end-to-end”) and Sec. 4.1 (“Each expert within the MoE architecture was optimized individually before the gating mechanism was trained”). Those two descriptions suggest different training procedures and should be reconciled.
- **The paper does not test whether the benefit really comes from the proposed dynamic token fusion, versus simpler ways of combining two frozen experts.**  
  The comparisons are mostly against prior prompt/adapter methods, including prior MoE-style adapters, but the paper omits some highly relevant simple baselines for its central claim: e.g., **feature averaging, concatenation plus linear fusion, or logit ensembling** of the general and medical models. Since the conceptual contribution is specifically “dynamic merging” of two experts, it is important to show that the elaborate routing/gating machinery outperforms straightforward two-expert fusion strategies, not only prior adapter families.

### Minor
- **The absolute gains over the strongest baseline are often modest.**  
  Tables 1–3 and 10 show consistent improvements, which is positive, but many are incremental relative to GMoE-Adapter. This does not negate the contribution, but it does reduce the significance of claims framed as a large advance.
- **The choice of medical expert is narrow relative to the breadth of evaluated modalities.**  
  Sec. 4.1 states that the medical expert is a ViT-B/16 pre-trained on **1.6M cell images**, yet the downstream evaluation spans retinal, skin, X-ray, CT, and MRI tasks. The paper’s thesis is that combining general and medical priors is broadly useful, but it does not analyze whether this result depends on this specific medical checkpoint or whether other medical pretraining choices would materially change the conclusions.
- **The analysis of the skipping router is incomplete, particularly because 50% token retention outperforms 100%.**  
  Table 7 is interesting, but also raises a question the paper does not answer: why does discarding half the tokens slightly improve every reported metric on that benchmark slice? This could reflect useful regularization or denoising, but without analysis it is hard to know whether the router is learning meaningful saliency or whether the full-token configuration is simply less well tuned.
- **The qualitative evidence is weaker than the text suggests.**  
  Figure 3 only visualizes DynaMer attention maps, yet the surrounding discussion makes comparative claims about “previous methods” and about mitigating “spatial and prompt forgetting.” Without side-by-side baseline visualizations under identical conditions, those interpretability claims are suggestive rather than demonstrated.
- **There is a naming inconsistency that hurts clarity.**  
  Table 9 refers to “GL-MoF Adapter (ours)” rather than DynaMer Adapter, which looks like a leftover name/version mismatch and should be corrected.

### Trivial
- None.

## Nice-to-Haves
- Report **mean ± std over multiple seeds** for the main benchmark tables, especially because many gains over the strongest baseline are small.
- Add **comparative compute reporting**: total FLOPs, wall-clock inference/training time, and peak memory for DynaMer versus Adapter/GMoE-Adapter, including the cost of running two frozen ViTs.
- Include **simple two-expert fusion baselines** such as feature averaging, concatenation with a learned linear head, and logit ensembling.
- Analyze the learned **gating weights** and **expert-routing patterns** across modalities to verify that the method is actually learning meaningful domain specialization.
- Test at least one additional **medical pre-trained checkpoint** from a different modality family to assess how much the result depends on the current cell-image expert.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper relies too heavily on limited medical benchmarks / lacks external datasets.”**  
  Removed because it is not supported by the paper text. The paper evaluates on a broad Med-VTAB suite spanning many datasets and modalities, plus patient-split and general-domain benchmarks. One can still question specific claims, but not dismiss the evaluation as narrowly confined.
- **Pure writing/style issues** such as grammar problems (“four four folds”) or generic presentation complaints.  
  Removed per instruction as formatting/style nitpicks, except where a wording problem reflects overclaiming or genuine ambiguity.
- **Broad criticism that ablations are absent or wholly inadequate.**  
  Removed in that absolute form because the paper does include several real ablations (Tables 4–7). The fairer retained criticism is narrower: the ablations do not test some of the most relevant simple fusion alternatives or fully explain the skipping behavior.
- **Any concern doubting the existence/availability of cited models or benchmarks.**  
  Removed by rule.

## Novel Insights
The most interesting aspect of the paper is not just that two experts are combined, but that the **adapter itself is shared across the two backbones while the gates/routers remain stream-specific**. That is a potentially elegant compromise: it encourages a common fusion mechanism while still allowing domain-specific control over how much adaptation to apply. The empirical pattern in Table 7 is also more revealing than the paper makes it seem: the fact that moderate token skipping can slightly improve accuracy suggests that the model may benefit from **selective suppression of low-value cross-domain interactions**, not merely from saving compute. If validated with routing analysis, that could become a stronger conceptual contribution than the current presentation emphasizes.

## Suggestions
- Reframe the abstract/introduction claims to better match the current evidence, especially around **efficiency**, **patient-OOD**, and **few-sample** advantages.
- Add a table reporting **full compute costs** versus baselines: runtime, FLOPs, and memory, including the overhead of the second frozen backbone.
- Clarify the training procedure so Sec. 3.3 and Sec. 4.1 are consistent: is the method trained **end-to-end**, or are experts pretrained separately before gate training?
- Explicitly specify the key architectural hyperparameters in the main paper: **number of experts, top-k routing choice, adapter placement, and whether the skipping router is used during training, inference, or both**.
- Add **simple dual-expert fusion baselines** to isolate the value of the proposed dynamic token-merging mechanism.
- Provide a short analysis of **why 50% token retention beats 100%**, ideally with token-selection visualizations or per-layer selection statistics.
- If space permits, test one more **medical expert checkpoint** from a non-cell modality to strengthen the claim of broad medical applicability.