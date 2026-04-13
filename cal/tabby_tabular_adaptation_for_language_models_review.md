=== CALIBRATION EXAMPLE 31 ===

# Final Consolidated Review
## Summary
This paper proposes **Tabby**, a post-training modification of transformer language models for tabular data synthesis in which selected MLP layers and/or the LM head are replaced by **column-specific mixture-of-experts components**. The main empirical finding is that the **MH variant** (MoE at the LM head) is often the strongest LLM-based configuration, and under the paper’s evaluation protocol it performs very competitively, sometimes reaching the same downstream utility as real data and outperforming prior LLM-based tabular synthesis approaches on several datasets.

## Strengths
- **A specific architectural bias that matches the tabular setting well.** The key idea is not just “use MoE,” but to assign experts **by column identity** so that each column has dedicated parameters inside the LM. This is a clean and plausible inductive bias for fixed-schema tables, where columns have heterogeneous semantics and distributions.
- **The MH variant shows a real empirical signal.** In Table 2, Plain-trained **Tabby MH** is the strongest LLM configuration overall and improves over Plain NT on several datasets, notably Travel (87.7 vs. 85.5), Rainfall (0.58 vs. 0.41), and House (0.75 vs. 0.70), while remaining competitive on Diabetes and Adult.
- **The paper surfaces an unexpectedly strong baseline that is genuinely informative.** A notable and useful result is that simple **Plain** sequential training of Distilled-GPT2 is already very strong on several datasets, often beating more elaborate LLM tabular training schemes. That is a concrete empirical takeaway for the community, not a generic strength.
- **The comparison against multiple synthesis families is broader than many narrowly scoped LLM papers.** The experiments include GAN/VAE/diffusion baselines (CTGAN, TVAE, Tab-DDPM) alongside several LLM-based setups (Plain, GReaT, GTT), which makes the empirical picture more useful than a purely within-family comparison.
- **The per-column loss decomposition is a practical byproduct of the training formulation.** Section 3.3 and Figure 4 show that Tabby’s training setup naturally yields per-column validation losses, which can help identify columns that remain difficult to model during finetuning.

## Weaknesses
### Major:
- **The central empirical attribution is confounded by a large parameter increase.** The paper argues that the gains come from a tabularly appropriate architectural modification, but the main Tabby models are substantially larger than the non-Tabby baselines. Table 3 makes this explicit: Distilled-GPT2 grows from **80M to 270M** parameters, and Llama from **8B to 10.5B**. There is no parameter-matched non-MoE baseline, so the paper does not isolate whether gains come from **column-specific specialization** or simply **added capacity**. This weakens both the mechanistic claim and especially Claim 2.
- **The headline claims are stronger than the evidence supports.** The abstract and introduction frame the method as broadly improving tabular synthesis quality and reaching “near or equal” real-data performance, but the evidence is mixed. The main positive result is strongest for **Plain-trained MH on this benchmark**, not for Tabby in general. On the six datasets, non-Tabby or non-LLM baselines remain best on important cases, especially regression: **Tab-DDPM beats Tabby on Abalone**, and is very competitive elsewhere. The paper’s own table shows substantial variation across datasets and Tabby variants.
- **The evaluation is too narrow to support broad claims about synthetic data fidelity.** The paper relies primarily on **MLE with a random forest downstream model** and a **random-forest discriminator**, with one target column per dataset. These are useful metrics, but not sufficient to justify stronger statements such as the synthetic data being “capable stand-ins for real data” in a broad sense. Under this protocol, some synthetic methods even exceed the “Original” row, so treating that row as a strict “upper bound” is not fully justified by the metric itself.
- **The architectural story is incomplete because the weaker Tabby variants are not explained.** The results are not monotonic with more MoE structure: **MMLP** often hurts badly, and **MMLP-MH** is frequently worse than **MH** alone. Examples in Table 2 include Adult and House, where MMLP and MMLP-MH can underperform NT substantially. Since the paper motivates Tabby through increased expressivity and column specialization, this inconsistent behavior needs analysis; otherwise the evidence supports a narrower conclusion: **one specific variant (MH) often helps**, not that the proposed architectural family is broadly validated.
- **Claim 2 is under-supported.** The “smaller models approach larger models” story in Section 4.2 is based on **one subset of one dataset**, with different finetuning setups across model families (LoRA for Llama, apparently standard finetuning for Distilled-GPT2), and the Llama improvement from NT to MH is negligible (0.560 → 0.562). This is an interesting pilot result, but not a strong claim at ICLR scope.

### Minor
- **The paper does not quantify efficiency tradeoffs.** Because Tabby materially increases parameters and involves per-column processing, the lack of training/inference cost, memory, or throughput reporting makes it hard to assess practical value relative to the observed gains.
- **The gating/routing description is underspecified.** The paper refers to “Gated Mixture-of-Experts layers,” but in the method description the effective behavior appears to be deterministic **column-to-expert assignment** (“The i-th column in the dataset is modeled by \(L_{a,i}\)”). This should be described more explicitly to avoid ambiguity about whether there is learned token-level routing or fixed expert selection by column identity.
- **Some claims around interpretability/understanding are overstated.** Figure 4 demonstrates that per-column losses can be tracked, which is useful. But the stronger language that this leads to better understanding of model behavior or supports distinctions such as aleatoric vs. epistemic error is not established by the presented evidence.
- **The instability/failure cases are noted but not analyzed.** Table 2 marks several configurations with failed valid-sample generation on Rainfall. That is useful to report, but the paper stops at observation rather than analyzing why some training schemes fail and whether Tabby systematically improves reliability.

### Trivial
- **The “up to 7%” claim should be tied to a precise comparison.** It would help to explicitly state which dataset/metric/baseline this refers to, since the average picture is more mixed than the best-case framing suggests.

## Nice-to-Haves
- Add a **parameter-matched non-MoE baseline** to separate specialization effects from capacity effects.
- Expand the analysis of **why MH helps while MMLP and MMLP-MH often hurt**, e.g., by inspecting expert similarity, utilization, or optimization behavior.
- Report **efficiency metrics**: wall-clock training time, memory footprint, and generation speed.
- Strengthen Claim 2 by evaluating the model-size comparison on **more than one dataset**.
- Include **reliability statistics** such as the fraction of invalid generations, not just whether at least one run failed.
- Add more direct fidelity diagnostics beyond the current utility/discriminator pair, such as per-column distribution matching or multivariate statistics.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **“The paper is generally readable / well-organized.”** Removed as a generic strength.
- **Requests for unspecified additional related work.** Removed per instruction not to speculate about missing references.
- **Strong criticism of limited dataset count as a standalone flaw.** The paper already evaluates on six datasets with several baseline families; asking for still more datasets is reasonable as a nice-to-have, but on its own this is too generic unless tied to a concrete overclaim. The real issue is that the paper’s **claims are stronger than what this benchmark and evaluation protocol can justify**, which is retained above.
- **Pure reproducibility nitpicks about implementation details/hyperparameters.** The paper gives the main training protocol, averaging over three runs with early stopping and grid search.
- **Concern about the prompt in Section 3.3 ending with `<EOS>` being necessarily erroneous.** This may reflect a notation issue or parser artifact; it is not reliable enough as a substantive weakness from the extracted text alone.

## Novel Insights
The most interesting synthesis across the reviews and the paper is that the submission actually supports a **more specific story than the abstract claims**: the contribution is not that “MoE-for-tabular LLMs” broadly wins, but that **putting column-specific experts in the LM head is a surprisingly effective and simple intervention**, while pushing the same idea deeper into the MLP stack can be counterproductive. Coupled with the unexpectedly strong Plain baseline, this suggests that for tabular synthesis the main bottleneck may lie closer to **output distribution shaping** than in wholesale architectural restructuring of the transformer body. That narrower conclusion is still valuable, but it is more modest than the paper’s current framing.

## Suggestions
- **Reframe the claims more narrowly and accurately.** Emphasize that the strongest result is for **Plain-trained MH**, and avoid implying that all Tabby variants or tabular synthesis broadly are improved.
- **Add a parameter-matched baseline** as the highest-priority revision; without it, the core causal interpretation remains unresolved.
- **Analyze variant behavior**, especially why **MH > MMLP >? MMLP-MH** does not hold monotonically and why some variants collapse on regression tasks.
- **Temper the “real-data parity” language** unless supported by stronger evaluation; describe parity as holding under the paper’s specific MLE setup rather than as a general statement about fidelity.
- **Report efficiency and reliability metrics** so practitioners can judge whether the quality gains justify the extra parameters and complexity.
- **Clarify the routing mechanism** in the method section so the reader can tell exactly how expert selection occurs during training and generation.

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 1.0]
Average score: 3.0
Binary outcome: Reject
