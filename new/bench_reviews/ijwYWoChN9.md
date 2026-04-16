## Summary

Domain Shift Tuning (DST) proposes a framework for adapting pre-trained language models to target domains by conceptualizing the domain gap as differences in "knowledge weights" over latent subnetworks. The method introduces a Knowledge Steering Layer (KSL) that routes tokens through K affine transformations via a discrete latent indicator z, and a Knowledge Distribution Modeling (KDM) objective that aligns similarity structure in the z-space with hidden representation similarities. DST is applied by inserting KSL on top of the Transformer layers while freezing the PLM, and is evaluated on topic discovery (NYT) and text generation (Amazon, arXiv) using GPT-2, BLOOM, and Llama-3.

## Strengths

- **Simple and practical architectural design.** The KSL is a lightweight top-layer intervention (~5.9M parameters for GPT-2 medium with K=10) that preserves the PLM structure entirely. This makes it easy to implement and compatible with other PEFT methods, as the paper correctly notes.

- **Consistent empirical improvements on GPT-2.** Table 3 shows that DST outperforms baselines (COCON, Prefix, NRP, LoRA, AdaMix, ReFT) across perplexity, Dist-4, BLEU-4, METEOR, and ROUGE-L on both Amazon and arXiv datasets. The ablation over K and transformation type (addition, multiplication, affine) is informative, confirming affine as the best variant.

- **Model-agnostic applicability.** The method is conceptually applicable to both encoder-style (BERT) and decoder-style (GPT-2, BLOOM, Llama-3) architectures, as demonstrated across experiments, lending some generality to the approach.

- **Intrinsic interpretability metric.** Eq. (8) defines r_KSL to measure the fraction of tokens where non-residual knowledge is selected, providing a useful diagnostic that correlates with improved generation quality.

## Weaknesses

### Major

- **The "knowledge subnetworks" framing significantly overstates what the implementation actually does.** The paper's core conceptual claim — that DST "conceptualizes domain gaps as differences in knowledge encapsulated within multiple subnetworks of PLMs" (Abstract) and "partitions a PLM into knowledge-equivalent subnetworks" (§3.1) — is not matched by the mechanism. In reality, Eq. (4) shows that KSL applies K affine transformations to the *final* hidden state h_{L,t}, selecting one per token via z. The "subnetworks" are newly added parameters (W_Z, W_{az}, b_z), not structures identified *within* the frozen PLM. No evidence is provided that z values correspond to meaningful, reusable, or interpretable partitions of PLM knowledge. The paper itself acknowledges that "knowledge is considered a latent and relative concept, not as concretely defined as topics" (§3.1), which undermines the precision of the central claim. In effect, DST is a top-layer mixture-of-affine-experts adapter — a valid PEFT contribution, but one whose conceptual framing is misleading relative to the implementation.

- **No ablation of KDM, leaving its contribution entirely unverified.** The KDM loss (Eq. 6-7) is a core component, yet all reported results include it. There is no KDM-off baseline (training with only L_{MLM}), making it impossible to attribute improvements to KDM rather than to the KSL gating mechanism alone. Additionally, the KDM formulation is under-specified: it is unclear what norm is used, how the "min over pairs" is aggregated in practice, and how ε=0.2 relates to the formula. The rationale — that aligning z-space and TID-space similarities helps domain adaptation — is asserted without theoretical or empirical justification.

- **No direct evaluation of domain shift or catastrophic forgetting despite these being the central claims.** The paper motivates DST as a method to "bridge the domain gap" and "prevent catastrophic forgetting" (§1, §3.2), yet all experiments are single-domain fine-tuning tasks. There is no measurement of source-domain performance before/after adaptation, no sequential multi-domain evaluation, and no established domain adaptation benchmark (e.g., cross-domain sentiment). Claims about mitigating catastrophic forgetting are stated but never tested: the paper says "In principle, DST avoids catastrophic forgetting by using the residual in Eq (3) and freezing PLMs" (§5.3), but this is an architectural argument (freezing prevents forgetting), not an empirical demonstration that DST retains source-domain knowledge *while* adapting to a target domain.

- **LLM experiments (BLOOM, Llama-3) lack baseline comparisons and absolute metric values.** Table 4 reports only improvement percentages for BLOOM and Llama-3, without showing absolute baseline scores or any PEFT baselines (LoRA, ReFT, etc.) on these models. This makes it impossible to assess whether DST's improvements over the base model are competitive with standard PEFT methods at this scale, a critical gap given that DST's primary claim is effectiveness as a PEFT alternative.

### Minor

- **The learned z variables are never analyzed or interpreted.** The paper proposes that z captures "knowledge," but provides no probing, visualization, or qualitative analysis of what different z values represent. Without evidence that z encodes meaningful semantic partitions, the "knowledge" interpretation remains an unverified metaphor.

- **Topic discovery experiment (Table 2) is tangential to the core claim.** DST is motivated for domain adaptation in generative PLMs, but the NYT topic discovery experiment uses BERT with no domain shift setting — just single-corpus topic modeling. The improvements over TopClus are marginal (NMI 0.47 vs 0.45), and the paper itself acknowledges DST "aims to discover differences between linguistic and semantic knowledge...rather than coherent and meaningful topics," which contradicts evaluating on topic coherence metrics.

- **K is a critical hyperparameter without systematic analysis or automatic selection.** The paper uses K∈{10,20,30} and acknowledges "automatic determination of K" as future work. The parameter cost scales linearly with K (K×d_h×d_h for W_{az}), and computational cost increases as well, but no wall-clock time comparison against LoRA/AdaMix is provided despite claims of "lower computational cost" (Abstract).

- **Human evaluation is limited.** Only fluency (1–5 scale) is evaluated by screened colleagues, with no domain-relevance or content-quality assessment and no inter-annotator agreement reported.

### Trivial

- Table 4's notation is confusing: the caption states "improvement (+%)" but some values (e.g., Flu scores) appear quasi-absolute, making interpretation difficult.

## Nice-to-Haves

- Ablate KDM entirely (train with only L_{MLM}) to determine whether it contributes beyond the KSL mechanism.
- Probe the learned z representations by correlating z assignments with interpretable features (topic labels, token frequency bands, domain membership) to validate the "knowledge" interpretation.
- Evaluate on a standard domain adaptation benchmark (e.g., cross-domain sentiment classification) with explicit measurement of catastrophic forgetting (source-domain performance before and after adaptation).
- Report wall-clock time and memory usage comparisons against LoRA/AdaMix to substantiate the "lower computational cost" claim.

## Removed Points

- **Demand for continual pretraining baselines (DAPT/TAPT).** While comparing against continual pretraining would strengthen the paper, the paper scopes its comparison to PEFT methods and fine-tuning baselines, which is a reasonable choice. Continual pretraining requires substantially different infrastructure and is not the same methodological family. → *Moved to Nice-to-Have*

- **Complaint that baselines are "outdated."** The baselines (LoRA, AdaMix, ReFT, Prefix, NRP, COCON) are all published and widely recognized PEFT methods. This is a standard set for this venue and time. No specific newer method is identified as a missing critical baseline. → *Removed*

- **Demand for varying target corpus sizes to validate the "small corpus" claim.** The paper does claim effectiveness on "small target corpora," and testing different sizes would be informative, but the experiments already include Amazon (210K reviews) and arXiv (1.5M papers), which vary in scale. The claim is partially addressed. → *Removed*

- **Formatting and notation nitpicks** about inconsistent notation (g_z appears but is not consistently used) or the ε=0.2 parameter. These are minor presentation issues, not substantive methodological flaws. → *Removed*

- **Criticism that "subnetwork" should be replaced with "latent expert routing."** While the terminology is indeed imprecise (addressed as a Major Weakness above), suggesting alternative terminology is a presentation preference rather than a methodological issue. The substantive problem is the overclaiming, not the specific word choice. → *Removed*

## Novel Insights

The most interesting aspect of this work is the explicit mixture formulation (Eq. 2) that decomposes the language model probability into a mixture of knowledge-conditional distributions weighted by a learned discrete router — formally connecting domain adaptation to latent topic models. However, the disconnect between this appealing mathematical framework (which implies partitioning the PLM into distinct computational pathways) and the actual implementation (which only adds top-layer affine transforms) means the core insight remains unvalidated. The r_KSL metric is a practical contribution for diagnosing how much "new" versus "old" knowledge is being used per token, though it measures routing frequency, not knowledge quality.

## Suggestions

1. **Ablate KDM** by training DST with only the L_{MLM} loss (set λ_KDM=0). Report the same metrics to determine whether KDM is a necessary component or whether KSL alone accounts for the gains.

2. **Reduce the scope of conceptual claims.** Reframe the contribution as a top-layer mixture-of-experts adapter for domain adaptation rather than claiming to identify and reweight "knowledge subnetworks" within PLMs. This honesty will strengthen the paper by aligning claims with evidence.

3. **Add PEFT baselines (LoRA, ReFT) to the BLOOM and Llama-3 experiments** and report absolute metric values alongside improvement percentages, so readers can assess DST's competitiveness at larger scales.

4. **Evaluate catastrophic forgetting directly** by measuring source-domain performance (e.g., general language modeling perplexity on a held-out corpus) before and after DST adaptation, providing the missing evidence for a core claim.

## Evaluation

**Originality:** The mixture-language-model formulation connecting domain adaptation to topic-style latent variables is a novel conceptual angle, but the implementation reduces to a standard top-layer MoE adapter. The gap between framing and mechanism dilutes the originality.

**Importance of research question:** Domain adaptation for PLMs/LLMs is an important and active area, making the problem setting well-chosen.

**Claims well supported:** Only partially. Empirical improvements on GPT-2 are demonstrated, but the core claims about knowledge subnetworks, domain shift mitigation, and catastrophic forgetting are not substantiated. The KDM component is un-ablated, and LLM results lack baselines.

**Soundness of experiments:** The GPT-2 experiments are reasonably thorough with multiple baselines and ablations over K and transformation type, but critical gaps remain (no KDM ablation, no forgetting evaluation, no LLM baselines, no domain-adaptation-specific benchmarks).

**Clarity of writing:** The paper is dense and the mathematical notation (Eqs. 2-6) requires careful reading. The connection between "knowledge" and the implementation is unclear, and some definitions (KDM loss, ε) are imprecise.

**Value to the research community:** As a lightweight PEFT variant, DST may be of practical interest, but the overclaimed "knowledge subnetwork" framing without evidence could mislead future work. The value depends on whether the gains are truly attributable to the knowledge-routing mechanism or simply to adding trainable parameters at the output layer.

## Score and Decision

Calibration papers:
- **Mkdwvl3Y8L** (Knowledge subnetworks) — similar issues with vague "knowledge" concepts, limited baselines; scores: 5,3,3,5 (avg ~4.0), Reject
- **t5mpbfpZuF** (DEAL domain adaptation) — insufficient baselines, unclear experimental methodology; scores: 3,3,5,3 (avg ~3.5), Withdrawn
- **c3rfGbXMBE** (Neural topic modeling with LLMs) — topic evaluation methodology issues, scalability concerns; scores: 3,3,3,5 (avg ~3.5), Withdrawn
- **rWui9vLhOc** (MoLEx) — outdated models, efficiency claims unsubstantiated; scores: 5,6,8 (avg ~6.3), Accept Poster

This paper has more empirical substance than DEAL or the topic modeling paper (consistent gains on GPT-2 across multiple baselines, ablations, and datasets), making it stronger than the ~3.5 range. However, it shares the same core weakness as the knowledge subnetworks paper (Mkdwvl3Y8L): overclaimed conceptual contributions that outpace the evidence. The disconnect between "knowledge subnetworks" framing and the actual top-layer MoE adapter is a significant conceptual issue, and the absence of KDM ablation, forgetting evaluation, and LLM baselines leaves key claims unverified. This is somewhat weaker than Mkdwvl3Y8L (which at least analyzed subnetwork behavior) because DST provides no analysis at all of what z captures. Below the MoLEx paper (~6.3) which had better evaluation despite model currency concerns.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>