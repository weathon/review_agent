## Summary
This paper identifies a "priming vulnerability" specific to Masked Diffusion Language Models (MDLMs), where injecting affirmative tokens at intermediate steps of the iterative denoising process can steer safety-aligned models to generate harmful responses. The authors propose a novel defense, Recovery Alignment (RA), which trains models to recover and produce safe outputs even when starting from such adversarially contaminated states. Experiments show RA significantly mitigates this vulnerability and improves robustness against conventional jailbreak attacks while largely preserving general task performance.

## Strengths
- **Identifies a novel, DLM-specific safety weakness:** The work provides clear, quantitative evidence (e.g., Attack Success Rate jumps from 2% to 21% with a single token intervention via the "anchoring attack") of a critical vulnerability inherent to the parallel, iterative denoising process of MDLMs, differentiating it from prior work on autoregressive models.
- **Effective and tailored mitigation:** The proposed Recovery Alignment (RA) is well-motivated and directly addresses the core issue by training models to generate safe responses from intentionally contaminated intermediate states. Empirical results are strong, showing RA drastically reduces ASR across multiple priming-based attacks and outperforms baseline alignment methods (SFT, DPO, MOSA) across three MDLMs.
- **Comprehensive and rigorous evaluation:** The paper validates its claims through extensive experiments on two datasets (JBB-Behaviors, AdvBench) using three safety evaluators (GPT-4o, guardrail model, keyword matching), multiple attack families (including proposed First-Step GCG), and ablation studies on scheduling and generation length. General capability is preserved across 11 diverse benchmarks.

## Weaknesses
- **Performance degrades under very strong attacks:** As shown in Table 2, for very late intervention steps (e.g., `t_inter=32`), where many harmful tokens are anchored, RA's Attack Success Rate remains high (50–79%). The paper notes generating a safe response from many fixed anchors is "practically impossible," but a deeper analysis of these failure modes (e.g., does the model output gibberish, partial harm, or a different unsafe response?) is missing.
- **Theoretical assumption's scope is not fully characterized:** Theorem 4.1, which enables the efficient First-Step GCG attack, relies on a monotonicity assumption. While empirically validated in Appendix C.2 for the studied models and attack states, a more formal discussion of the conditions under which this assumption may fail (e.g., for highly unnatural sequences) would strengthen the theoretical contribution.
- **Computational cost of alignment:** RA, as an RLHF-style method, incurs higher training cost (~16 hours on 4 H100 GPUs) compared to supervised baselines like SFT or DPO (Appendix C.4). While reasonable for the study, this may impact scalability and practicality for very large models.

## Nice-to-Haves
- A qualitative analysis of denoising trajectories for RA versus baseline models to illustrate the hypothesized "recovery" mechanism in action.
- A preliminary exploration of a supervised (e.g., DPO-style) variant of RA to assess if similar robustness can be achieved with lower training cost, as suggested in the Limitations section.
- Reporting the inference-time latency/throughput of RA-aligned models compared to originals to assess any deployment overhead.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness about reward model justification:** The paper uses DeBERTaV3 as a reward model, citing its use in prior work (Köpf et al., 2023). This is a reasonable, established choice, and an ablation is not required for the core claim.
- **Weakness demanding comparison against adapted ARM jailbreak methods:** The paper's scope is DLM-specific vulnerabilities and defenses. Requiring adaptation and benchmarking of all state-of-the-art ARM attacks is tangential and not a core evaluation flaw.
- **Weakness about over-reliance on LLM-as-judge:** The paper employs three distinct evaluation metrics (GPT-4o, guardrail model, keyword matching), which is a robust and standard practice in the field. Consistency across metrics is discussed in the appendix.
- **Weakness about potential safety degradation from training on harmful data:** The paper evaluates general capability and shows no substantial degradation (Table 4). A specific "poisoning" evaluation is beyond the standard scope and is not required to validate the method's efficacy.
- **Weakness about testing RA against intervention steps > 96:** The paper systematically evaluates up to `t_inter=32` (25% of total steps) and shows a clear trend. Testing even later steps, while interesting, is not necessary to establish the core contribution—identifying the vulnerability and providing a significant mitigation.

## Novel Insights
The paper provides a novel and important insight: the iterative, parallel denoising process of MDLMs introduces a unique safety vulnerability where early affirmative tokens can irrevocably bias the generation trajectory toward harmful content, a phenomenon distinct from attacks on autoregressive models. Furthermore, it demonstrates that standard alignment, which only trains models from a fully masked state, is fundamentally insufficient to defend against this, necessitating alignment that explicitly conditions on and recovers from contaminated intermediate states—a principle that also generalizes to improve robustness against conventional jailbreak attacks.

## Suggestions
- Conduct a qualitative analysis of model outputs in high-ASR failure cases (e.g., for `t_inter=32`) to better characterize the failure mode and inform potential complementary defenses.
- Provide a more formal discussion or empirical bounds on the monotonicity assumption in Theorem 4.1, clarifying the types of sequences or model states where it may not hold.