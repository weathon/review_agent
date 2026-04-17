---
job_id: 4caa5192-2cbe-4222-b311-776b9f3876d7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: GMP1S4R6Ke.pdf
paper: LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is about parameter-efficient multi-task adaptation of LLMs via a LoRA-based MoE routing mechanism, squarely within ICLR’s core topics of representation learning, optimization, and large-scale language models.

## Minimum Quality
Pass ✅.  
The paper has all required sections (Abstract, Introduction, Related Work, Method/Approach, Experiments/Results, Conclusion). The method is technically nontrivial, experiments span multiple benchmarks and baselines, and there are no glaring methodological or theoretical errors that would justify immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find hidden prompts or attempts to manipulate automated reviewing systems in the main paper content.

---

# Expected Review Outcome:

## Summary

The paper proposes **LoRA-Mixer**, a modular mixture-of-experts framework that routes multiple LoRA adapters at the *projection layers* (e.g., attention input/output projections) of Transformers and SSMs, instead of replacing full attention/FFN blocks or adding parallel branches. Each LoRA is treated as an expert and a lightweight router selects experts per token using a sparse top‑k scheme.

To address over-uniform routing from standard auxiliary MoE losses, the authors introduce **Routing Specialization Balance Loss (RSL)**, which augments the usual load-balancing term with an entropy-penalization term to encourage token-aware, peaked routing while maintaining global load balance. Experiments on 15 benchmarks across several domains, cross-model transfer, and plug‑and‑play usage of Internet LoRAs indicate consistent gains over baselines such as LoRA, LoRAHub, MixLoRA, MoLE, and routing‑loss variants (GMoE, Ds‑MoE, AESL) under similar or lower trainable parameter budgets.

## Strengths

1. **Clear architectural idea and practical applicability.**  
   - The key design choice, illustrated in **Figure 1** (Page 2) and elaborated in **Figure 2** (Page 3), is to attach MoE *at the linear projections* inside attention / SSM modules rather than at FFNs or via parallel branches. This is a crisp, implementable idea that is plausibly better aligned with how LoRA is commonly used in practice (on Q/K/V/O and similar projections).  
   - The framework is explicitly architecture-agnostic (Transformers and Mamba-based SSMs) and uses standard LoRA parametrization (Eq. (4)), making it very easy to adopt in existing codebases.

2. **RSL is a well-motivated, mathematically grounded modification of standard MoE auxiliary loss.**  
   - Section 3.3 formalizes RSL (Eq. (5)–(10)) as the sum of a conventional global load term and a negative-entropy term on the routing distribution. The gradient derivation in Eq. (7)–(9) is correct and clearly shows how the entropy piece introduces token-level curvature via the \(\log p_i(x)\) term.  
   - The appendices (A.1–A.2) give reasonably careful optimization and generalization analyses, showing that the negative-entropy term makes the smoothed objective \(\lambda\)-strongly convex (Lemma 1, Lemma 3) and leads to stability-based generalization bounds (Theorem 2). While not groundbreaking theory, this is a nice, consistent story tying the loss design to optimization stability and data efficiency.

3. **Strong empirical coverage with varied models and regimes.**  
   - The paper evaluates on three base models (Falcon‑Mamba‑7B, Mistral‑7B, LLaMA3‑8B), plus Flan‑T5 for Internet LoRAs, and across a reasonably broad benchmark suite (GLUE tasks, ARC-E/C, GSM8K, MedicalQA, HumanEval, BoolQ, HellaSwag, PIQA, plus custom cross-domain QA).  
   - **Table 2** (Pages 6–7) is central: it shows that LoRA-Mixer consistently improves over single-task LoRA and over compositional baselines (LoRAHub, MoLE, MixLoRA) on almost every task and base model, often by 1–3 points absolute. For example, on LLaMA3‑8B, LoRA-Mixer improves GSM8K from 65.14 (LoRA) to 65.53, ARC‑C from 82.15 to 83.24, and HumanEval from 55.61 to 57.32 while using fewer trainable parameters than MixLoRA.  
   - **Table 8** (Page 8) isolates the effect of the routing loss under fixed data (2k) and experts, showing RSL beating GMoE, DS‑MoE, and AESL on all five tasks; the HumanEval jump from 50.46 (AESL) to 57.32 (RSL) is particularly compelling.

4. **Modularity and plug‑and‑play use of off‑the‑shelf LoRAs is convincingly demonstrated.**  
   - Section 4.3 and **Table 3** (Page 7) show that freezing Internet LoRAs on Flan‑T5 and training *only the router* on 2k mixed samples already yields strong performance: LoRA-Mixer improves over single-task LoRA on CoLA (80.54 → 82.14), MRPC (83.76 → 85.15), and RTE (83.47 → 85.31).  
   - The cross-model transfer experiment in **Table 5** (Page 7) is a nice touch: routing parameters trained on Mistral‑7B are transplanted to LLaMA3‑8B, giving nontrivial gains on GSM8K across 0/2/5-shot CoT settings and modest improvements on ARC‑C without any adaptation.

5. **Evidence that routing actually specializes and balances experts.**  
   - **Figure 3** (Page 8) shows that the average load per expert on 1K mixed examples stays roughly between 15–18%, suggesting no expert collapse.  
   - **Figure 4** (Page 9) compares expert activations with and without RSL across Medical, GSM8K, and HumanEval; with RSL, “relevant” experts show clearly higher activation bars per task, while the no‑RSL condition yields more uniform, less interpretable patterns. This directly supports the paper’s claim that RSL induces input-aware specialization while preserving balanced load.

6. **Reasonable efficiency analysis and practical trade-offs.**  
   - The parameter and memory breakdown in Appendix A.4 and A.7, together with **Table 12** and **Table 14**, shows that LoRA-Mixer adds modest overhead over LoRA/LoRAHub and is competitive with MoLE/MixLoRA in memory and inference time (0.574 s vs 0.597 s for MixLoRA on LLaMA3‑8B).  
   - **Table 9** (Page 9) empirically supports the “low-data” claim: with only 1–2k routing examples, RSL outperforms the auxiliary-loss baseline by 1–2 points on average over seven tasks, and achieves near-saturation performance with about half the data.

7. **Clarity and organization.**  
   - Despite some rough edges in English, the core narrative is coherent. Figures 1 and 2 are well annotated and give a clear mental model of where LoRA-Mixer sits in the stack and how hard vs. soft routing stages work.  
   - The paper repeatedly grounds abstract claims (data efficiency, specialization, transferability) in specific experiments and tables; this is appreciated.

## Weaknesses

1. **Positioning and novelty relative to very close LoRA‑MoE work is underdeveloped.**  
   - The method is conceptually close to several existing efforts that compose many LoRAs via routing or gating. The related work section mentions MixLoRA, MoLE, LoRAHub, HMoRA, MoLA, LoRAMoE, etc., but the *mechanistic* differences are not dissected in enough depth, especially regarding:  
     - where exactly experts are inserted (projection layers vs FFNs vs LoRA branches);  
     - whether experts are trained jointly or independently;  
     - how routing is parameterized and trained.  
   - For instance, **Ostapenko et al., 2024 (“Towards Modular LLMs by Building and Reusing a Library of LoRAs”)** and **Sheng et al., 2023 (S‑LoRA)** are both about modular reuse / serving of multiple LoRAs. The paper’s claimed advantages in modularity and plug‑and‑play use should be explicitly contrasted with those. Currently, this leaves some ambiguity regarding how fundamentally different LoRA-Mixer is vs simply gating between pre-trained LoRAs at standard insertion points.

2. **RSL analysis depends heavily on unverified assumptions and smoothing, which are not clearly tied back to the implemented system.**  
   - The theoretical development in A.1–A.2 relies on **Assumption 1**, which postulates convexity and L‑smoothness of \(\sum_i \widehat{p}_i \widehat{s}_i\) in \(\{p(x_j)\}\) after replacing discrete top‑1 usage with soft surrogates \(s_i^\tau\). There is no justification that this holds for the specific router architecture and logits used in the experiments; in practice the mapping from parameters to \(p(x)\) is highly non-convex.  
   - The convergence theorem is thus about optimizing *probability vectors* \(p(x_j)\) directly via entropic mirror descent, not about the actual neural router parameters updated via SGD. This gap is not acknowledged in the main text, and can mislead readers into interpreting Theorem 1 as a guarantee for the implemented training.  
   - Similarly, the generalization bound in A.2 assumes strong convexity in the averaged \(\ell_1\)-norm and a Lipschitz surrogate loss, but there is no discussion of whether the real training regime (mini-batch SGD with Adam, non-smooth top‑k routing at inference, etc.) respects those conditions. The bound is therefore more of a qualitative motivation than a rigorous guarantee; the paper should make this limitation explicit.

3. **Routing mechanics and implementation details are underspecified in the main text.**  
   - Equation (4) describes \(\mathbf{y} = W\mathbf{x} + \mathcal{F}_{\text{route}}(\{\alpha_e(\mathbf{x})\Delta W^{(e)}\mathbf{x}\}_{e=1}^E)\) but does not clearly state what \(\mathcal{F}_{\text{route}}\) is in practice. From context, it seems to be a top‑k sparsification followed by a weighted sum, but several important questions are left vague:  
     - Is the router shared across all projection matrices in a layer, or separate per Q/K/V/O (and per layer)?  
     - What is the input to the router (token embedding, hidden state, pooled sequence, task id)? **Figure 2** hints at per-token routing but does not assert it.  
     - How is “differentiable hard–soft top‑k” implemented during training (e.g., straight-through, Gumbel tricks, or purely soft with top‑k masking)? The text briefly mentions “soft expert fusion” with softmax scores \(\mathbf{p}_{b,t}\), but the concrete algorithm is not described.  
   - This lack of specificity makes it hard to reproduce the method from the main paper without relying on the code.

4. **Two-stage training pipeline and data usage are somewhat confusing.**  
   - The method conceptually involves: (i) training individual LoRAs per task; (ii) freezing them and training the router with RSL on a multi-task mixture; and optionally (iii) joint fine‑tuning of experts + router with \(\mathcal{L}_{\text{preserve}}\). However, Section 4 devotes very little space to clarifying which stage(s) are used in which experiments.  
   - For example, **Table 2** reports LoRA-Mixer performance but does not state whether experts are independent single-task LoRAs or jointly trained under hard routing. The description “LoRA” in the rows is also ambiguous: is it single-task fine-tuning per dataset or a multi-task LoRA trained on a mixed corpus?  
   - Appendix A.6 mentions “Stage 1: 40k data; Stage 2: 2k or 4k”, but the relationship between those 40k and the datasets in Table 11 (Page 17) is not spelled out. Without a clear protocol, it is hard to assess the fairness of comparisons with baselines that may have different total data or training regimes.

5. **Some experimental comparisons do not fully isolate the contribution of RSL vs architectural changes.**  
   - While **Table 8** nicely fixes the LoRA experts and compares routing losses, many of the headline gains in **Table 2** conflate several changes: (i) using projection-layer experts vs FFN experts; (ii) using top‑k sparse routing vs continuous weighting; and (iii) using RSL instead of standard auxiliary losses.  
   - For a clean story about RSL itself, one would like to see:  
     - A comparison of LoRA-Mixer with and without RSL **on the same architecture and hyperparameters**, per task, not just an averaged score in **Table 9**. The average gap is small at larger data sizes, and it is unclear on which tasks RSL helps or hurts.  
     - A control where standard auxiliary loss is used but experts are still placed on projection layers, to separate the effect of “where to put MoE” from “how to regularize routing”.

6. **Evaluation breadth is good but depth on key tasks is limited.**  
   - For reasoning/complex tasks like GSM8K and HumanEval, the reported improvements are relatively modest (e.g., GSM8K LLaMA3‑8B: 65.14 → 65.53; Falcon‑Mamba: 56.27 → 57.87). The paper claims LoRA-Mixer is particularly suitable for complex cross-domain reasoning, yet there is little qualitative or error analysis explaining what kinds of instances actually benefit from adaptive expert routing.  
   - **Figure 4** uses aggregated activation per “task” expert, but there is no per-example or per-subskill breakdown (e.g., which GSM8K categories get routed to which experts). As a result, the reader is asked to take “improved specialization” largely on faith, beyond a few aggregate histograms.

7. **Some notational and textual inconsistencies and minor math issues.**  
   - In **Equation (3)**, the definition of \(L_{\text{aux}}\) is given as \(\alpha\sum_i \bar{p}_i \bar{f}_i\), but later in Appendix A.17 the “balancing loss” is defined as \(\alpha \sum_i \bar{p}_i^2\). The relationship between these two formulations should be clarified explicitly.  
   - In **Equation (9)**, \(\nabla_{p_i(x)}\mathcal{L}_{\text{RSL}} = \alpha\cdot\frac{\partial \bar{p_i}}{\partial p_i(x)}\cdot\bar{f}_i + \lambda(\log p_i(x)+1-\mu)\). However, from Eq. (5) one would expect a minus sign in front of the entropy gradient if the original loss is \(\alpha \sum_i \bar{p_i}\bar{f_i} - \lambda \mathbb{E}[\mathcal{H}(p(x))]\). The sign flips again in A.1, where \(F_S\) is defined with \(+\frac{\lambda}{n}\sum p\log p\). The distinction between minimizing RSL vs minimizing its negative‑entropy-regularized surrogate is not clearly spelled out in the main text and can be confusing.  
   - Small issues like repeated “Table 2” caption text, stray typos, and duplicated references (the long block of “Lin, Han, and Li, Hsin” entries on Pages 12–13) detract from presentation quality.

8. **Limited ablation on where in the network LoRA-Mixer is applied.**  
   - The paper claims a key benefit is applying MoE only on “core projection layers”, improving parameter efficiency. However, there is no experiment that varies layer depth or projection types (e.g., only Q/K vs Q/K/V/O vs attention+FFN projections) beyond a short statement in the conclusion that uniform use across all layers may be redundant.  
   - Some insight into which layers or projections contribute most to the gains would make the architectural claims more convincing.

## Potentially Missing Related Work

1. **Ostapenko, Su, Ponti, “Towards Modular LLMs by Building and Reusing a Library of LoRAs”, 2024.**  
   This work directly tackles creating a library of reusable LoRA modules and composing them for new tasks, which is highly aligned with the plug‑and‑play LoRA reuse story in Section 3.2 and experiments in Section 4.3. It should be discussed in the related work section (Page 2–3) and contrasted with LoRA-Mixer in terms of routing, specialization, and whether composition is dynamic (per token) or static per task.

2. **Luo, Lei, Lei, “MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models”, 2024.**  
   MoELoRA combines MoE with LoRA for parameter‑efficient fine‑tuning using a contrastive loss to guide expert specialization. This is very close to the paper’s setting (LoRA‑MoE with specialized routing) and should be cited in Section 2 and empirically compared against, at least in discussion, as an alternative specialization mechanism to RSL.

3. **Sheng, Cao, Li, “S‑LoRA: Serving Thousands of Concurrent LoRA Adapters”, 2023.**  
   S‑LoRA studies the system-level problem of serving many LoRA adapters concurrently, which is conceptually related to LoRA-Mixer’s goal of using many LoRA experts in a modular fashion. It would be helpful to acknowledge this in Related Work and clarify how LoRA-Mixer’s routing differs from S‑LoRA’s serving architecture, especially in practical deployment scenarios.

## Questions

1. **Router architecture and sharing.**  
   - Is there a single router per layer, shared across all projection matrices (e.g., Q/K/V/O and SSM projections), or do you use separate routers per projection? What is the exact input feature used for routing (per-token hidden state, pooled state, task ID)? A concrete schematic or formula for \(G(x)\) in Eq. (1) would be helpful.  
   - How many parameters does the router have per layer, and how does that contribute to the “0.04% routing parameters” claim in A.4?

2. **Training protocol for the main comparisons in Table 2.**  
   - For the LLaMA3‑8B results in **Table 2**, are the LoRA experts single-task adapters trained on each dataset independently, or a shared multi-task LoRA?  
   - Does LoRA-Mixer always use the two-stage pipeline (expert training then router training), or is any joint fine-tuning with \(\mathcal{L}_{\text{preserve}}\) used in those numbers? Please clarify per model.

3. **Effect of RSL per task and per data regime.**  
   - **Table 9** only reports average performance across seven tasks at different routing data sizes. Could you provide per-task breakdowns (or at least per-domain) for w/ vs w/o RSL at 2k data? This would help understand on which tasks RSL is most beneficial.  
   - Have you observed any tasks where RSL *hurts* performance relative to auxiliary loss even at low data, and if so, what is your interpretation?

4. **Top‑k routing details and stability.**  
   - In Section A.3 and **Figure 5**, you show ablations over top‑k values for SST‑2 and CoLA. Are these ablations done with fixed RSL hyperparameters, or are \(\alpha,\lambda\) re-tuned per k?  
   - During training, do you use a straight-through estimator or continuous relaxation for top‑k, or is routing entirely soft (no masking) until inference? This also affects the validity of the entropy-based analysis.

5. **Layerwise application of LoRA-Mixer.**  
   - Do you apply LoRA-Mixer to all layers and all projection matrices of the base models, or to a subset (e.g., only higher layers)? If all layers, have you tried restricting to, say, upper 1/3 of layers, and how does performance vs parameter-efficiency trade-off change?

Clarifications on these points could strengthen the paper’s reproducibility and sharpen the understanding of where the gains are coming from.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is conceptually sound and the empirical evidence is reasonably strong, but the theoretical analysis rests on idealized assumptions not clearly connected to the implemented system, and some experimental protocols and routing details are under-specified.

## Presentation Rating

3: good.  
The overall narrative and figures/tables are clear, but there are several notational inconsistencies, duplicated references, and missing implementation details that hurt clarity and reproducibility.

## Contribution Rating

3: good.  
The idea of projection-layer LoRA‑experts plus RSL is a meaningful contribution with demonstrated practical value, though it builds on a rapidly growing body of closely-related LoRA‑MoE work and could be better positioned against that prior art.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper presents a coherent architectural tweak with a well-motivated routing loss, backed by solid experiments across several models and tasks, and shows convincing gains in modular multi-LoRA composition with good parameter efficiency. The main concerns are incomplete positioning vs very close LoRA‑MoE work, under-specified routing implementation, and somewhat idealized theory. With clearer exposition and stronger comparative analysis, this would be a solid ICLR contribution; as is, I lean positive but see room for sharpening.

## Reviewer Confidence

4: confident.  
I am familiar with MoE and LoRA literature, checked the core math and experimental claims in detail, and feel reasonably certain about this evaluation, though some implementation specifics remain unclear without code.