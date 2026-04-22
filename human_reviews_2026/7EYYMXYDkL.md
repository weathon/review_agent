# PLaID++: A Preference Aligned Language Model for Targeted Inorganic Materials Design

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a promising approach to improve correctness in LLMs, however, in many scientific problems, the objective is not necessarily to produce *the* correct answer, but instead to produce a diverse array of candidates which satisfy a set of constraints. We study this challenge in the context of materials generation. To this end, we introduce PLaID++, an LLM post-trained for stable and property-guided crystal generation. We find that performance hinges on our crystallographic representation and reward formulation. First, we introduce a compact, symmetry-informed Wyckoff text representation which improves computational efficiency and encourages generalization from physical priors. Second, we demonstrate that temperature scaling acts as an entropy regularizer which counteracts mode collapse and encourages exploration. By encoding symmetry constraints directly into text and guiding model outputs towards desirable chemical space, PLaID++ generates structures that are thermodynamically stable, unique, and novel at a $\sim$ 50\% greater rate than prior methods and conditionally generates structures with desired space group properties. Our work demonstrates the potential of adapting post-training techniques from natural language processing to materials design, paving the way for targeted and efficient discovery of novel materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents PLaID++, a preference-aligned large language model for inorganic crystal generation. The model fine-tunes a pretrained Qwen-2.5 7B LLM using a new Wyckoff-based textual representation that encodes symmetry and lattice parameters compactly, then post-trains it via Direct Preference Optimization (DPO).

The authors adapt DPO into a scientific alignment framework called RLIP (Reinforcement Learning from Interatomic Potentials): generated crystals are evaluated with machine-learning interatomic potentials (eqV2, eSEN) for stability, novelty, and space-group correctness, and ranked into preference pairs (e.g., stable > metastable > unstable). Iterative DPO updates align the model toward physically stable and diverse structures, while a dynamic-temperature schedule prevents mode collapse.

On the MP-20 dataset, PLaID++ achieves 97.3 % stability and a 7.7 % S.U.N. (Stable-Unique-Novel) rate, roughly 50 % higher than prior methods such as FlowLLM and ADiT. Joint training on unconditional and space-group-conditioned prompts also improves targeted generation (↑ 47 % S.S.U.N.). Ablations show that the Wyckoff representation, tiered stability rewards, and dynamic temperature are key to these gains.

### Strengths
- Clear problem framing and relevance. The paper targets a high-impact task: generating novel, thermodynamically stable inorganic crystals, potentially conditioned on structural symmetry. This is central to materials discovery and of broad scientific interest.

- They check their reward model.
They do attempt to show that MLIP-based energy ranking is not totally hallucinated. They correlate MLIP-based energy above hull with true DFT energy above hull on 1,000 samples, and report R² ~0.84 for eSEN near the stability threshold. They also report classification precision/recall for “stable vs metastable vs unstable,” which matters because the entire RLIP pipeline depends on the surrogate reward being aligned with physics.

- Computational throughput / practicality.
They emphasize that PLaID++ can sample 10,000 crystals in ~23 minutes on a single H100 and achieves ~5× higher throughput than FlowLLM. High-throughput suggests this may actually be usable for screening, not just a pretty table.

### Weaknesses
- No new neural architecture; novelty is mostly procedural.
The backbone is Qwen-2.5 7B with LoRA. There is no new model class, no new attention mechanism, no new equivariant layer, no new geometry-aware transformer. The “innovation” is:
- - a structured text format (Wyckoff),
- - a preference-alignment / DPO loop using physics-derived rewards,
- - and some engineering heuristics (tiered rewards, temperature ramp).
This is valuable, but reviewers will ask: is this ICLR-level novelty, or is it an application of known LLM post-training techniques to a new domain?

- All training/evaluation is on MP-20 only.
The model never leaves this relatively standard ~45k-material dataset with ≤20 atoms. There is no demonstration on larger or more diverse datasets (Alexandria, LeMat-Bulk, etc.) and no zero-shot transfer experiment. So we don’t know if PLaID++ is general, or just very tuned to MP-20 chemistry space.


- Prompt robustness is not explored.
The model is always prompted with essentially the same English template (“Below is a description of a bulk material… The spacegroup number is X… Generate…”). There is no ablation on prompt phrasing, no test of whether the generation is brittle to different wording, no demonstration of controllability beyond “give me space group N.”
For a claimed “LLM interface,” this is under-explored. We don’t learn how instruction-like this really is vs how template-locked it is.

- No multi-objective targeting beyond symmetry.
They only condition on space group (as a proxy for symmetry class). They mention other properties (band gap, etc.) in the prompt template during SFT, but the reported experiments don’t actually demonstrate controllable generation for properties like band gap or conductivity. For discovery workflows, that’s critical. Right now, controllability is still narrow.



- Figures could be more readable.
Several figures (e.g., Figure 3, Figure 4, Figure 8) have very small font on axes and legends. For a paper that leans heavily on ablation plots to support its claims (temperature schedule, S.U.N. over iterations, MLIP-vs-DFT correlation), readability matters. Increasing font size / contrast in these plots would make the empirical story much easier to audit.

### Questions
- How sensitive is performance to the exact wording / structure of the prompt template? Can the model handle natural variations in phrasing, or is it essentially bound to the provided template?

- You frame PLaID++ as “preference aligned” via RLIP. In normal RLHF, preferences come from humans and implicitly encode high-level goals. Here, preferences come from MLIPs and symmetry filters. Do you observe any reward hacking behavior (i.e., crystals that look unphysical to a human crystallographer but score as “stable” under eqV2/eSEN)?

- Can you report transfer? For example, train on MP-20 but prompt for a space group that is rare in MP-20, or generate crystals with >20 atoms, or test on a different dataset split. Right now it’s unclear how generalizable the method is beyond MP-20.

- The font size / legend readability in key figures (3, 4) is quite small. Please enlarge axis labels and legends so that stability / S.U.N. trends and correlation plots can be evaluated without zooming.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PLaID++, a preference-aligned LLM framework for inorganic crystal structure generation. PLalD++ first cast the crystal design into a text generation problem using a symmetry-aware Wyckoff-based representation, trained on MP-20 with SFT on top of Qwen2.5-7B. Then it introduces RLIP (Reinforcement Learning from Interatomic Potentials): the model samples candidate crystals, an ML interatomic potential assigns stability-related scores (stable/metastable/unstable), these scores are converted into preference pairs, and the model is iteratively updated with DPO. A key engineering choice is a temperature-increasing sampling schedule during iterative DPO to prevent diversity collapse.

### Strengths
1. Treating crystal design as conditional/unconditional text generation and then doing preference-based post-training is a clean and extensible design that the broader LLM-for-science community can reuse.

2. Using ML interatomic potentials to create automatic preference pairs and feeding them to DPO is a nice instantiation of “RLHF without humans” for material science domains. It avoids training a separate value model and keeps the pipeline relatively simple.

3. The paper gives evidence that doing iterative preference optimization on a naive coordinate representation leads to mode/dictionary collapse, while the Wyckoff-based representation maintains S.U.N. This is a useful empirical observation for future work on structure-generators.

### Weaknesses
1. The reward of preferences is narrow and model-based. All large-scale preference pairs come from MLIPs (eqV2 for training, eSEN for evaluation). Even though they separate train-time and eval-time models to reduce reward hacking, the model may still be aligned to this family of potentials rather than to actual DFT or experimental reality. The 1k DFT sanity check helps, but is too small to fully validate the main claim.


2. Targeted design is only partially demonstrated. The title says “targeted inorganic materials design,” but the actual conditional task is mainly space-group conditioning on 7 groups. That is narrower than what “targeted” typically implies in materials (e.g., band gap, stability window, mechanical property). The method may extend, but the paper does not show it.


3. Some pipeline details are underspecified. For iterative DPO, the paper does not clearly tabulate per-iteration: number of samples, number of preference pairs, positive/negative distribution across space groups, and exact temperature schedule. That makes it harder to reproduce and to judge whether the gains come from DPO itself or just from resampling at a higher temperature.


4. The main ablation shows that the coordinate representation collapses under DPO, but we do not see ablations that separately test: (i) a more structured prompt without Wyckoff; (ii) Wyckoff but no DPO; (iii) entropy regularization without temperature scheduling. So we cannot fully isolate what contributes most to the S.U.N. gain.


5. Given that this is a design paper, an example of “top-N generated structures sent for expert or database validation” would have made the story more convincing.

### Questions
1. Are all preference pairs derived purely from model-generated samples, or do you also form pairs that include real MP-20 structures as positives? If yes, what is the ratio of synthetic–synthetic vs. real–synthetic pairs per iteration?


2. You use eqV2 for training and eSEN for evaluation to mitigate reward hacking. Have you tried a third MLIP/evaluator (even a weaker one) to test whether the performance gains generalize across potentials, or do you observe overfitting to the two chosen models?

3. Please provide the exact per-iteration temperature schedule. Also, can you compare “iterative DPO + temperature schedule” against “iterative SFT (no DPO) + the same temperature schedule”? This would clarify whether the diversity preservation comes from DPO or simply from sampling at higher T.


4. Your conditional experiments focus on 7 space groups. How would the pipeline change if the target were an observable (e.g., band gap > 2 eV, or formation energy below some threshold) instead of a space group? Can MLIP scores be turned into preferences for such targets without retooling the whole pipeline?


5. For the 1k DFT relaxations, were the candidates sampled only from the final aligned model, or proportionally from all DPO iterations? If only from the last one, could this selection bias exaggerate the agreement between MLIP and DFT?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents PLaID++ which is a language model-based approach for generating stable inorganic crystal structures by combining a Wyckoff position-based text representation with Direct Preference Optimization (DPO) training. The work uses Reinforcement Learning from Interatomic Potentials (RLIP). The model uses the eqV2 MMLIP for structure relaxation and evaluation. Then they use a tiered preference pairing system to create preference pairs for DPO optimization. The authors report sota performance on the MP-20 benchmark with a 22.27% stability rate and a 7.74% S.U.N. rate.

### Strengths
1. Multiple MLIPs are employed for rigorous validation. They use separate MLIPs for training (eqV2) and evaluation (eSEN) prevents reward hacking, and the inclusion of DFT validation on a 1000-sample subset provides ground-truth verification. 

2. The dual strategy for diversity is an innovative practice. The combination of a symmetry-aware representation with Wyckoff positions and dynamic temperature scheduling during DPO training is innovative for encouraging diversity and preventing potential mode collapse. 

3. The reported sampling speed shows advantage over diffusion-based methods.

### Weaknesses
1. The tiered preference pairing scheme is a key contribution of the work and crucial design choice. The design are claimed valid but. rather intuitive and lacks rationale. The work would benefit from a thorough ablation study for e.g. the tiering threshold, pairing strategy, sample portions, etc.

2. The dynamic temperature schedule is described as critical for maintaining diversity. But the work lacks the analysis of sensitivity to temperature e.g. tracking how temperature changes affect exploration patterns or causally lead to the design of S.U.N. materials. The research gap remains to understand the improvements reported in the paper.

### Questions
1. How was the metastable threshold (0.08 eV/atom) determined? Why you choose the specific tiered preference pairing scheme? Have you performed ablation experiments, e.g. use only the (metastable, unstable) pairs, considering the often noised predictions for the 'stable' crystal structures with MLIPs? In addition, how does the sampling ratio contributes to the final performance gain?
2. How sensitive is the final S.U.N. rate to the hyperparameters of the temperature schedule? 
3. I believe some samples would fail the relaxation, are they regarded as unstable or disgarded?
4. It is mentioned that the exploration "expands into areas underrepresented in the base model" and shows increased generation of P-block elements. Have you validated whether this represents valid exploration or bias? Have you compared generation patterns between the base pre-trained Qwen before SFT and PLaID++ to determine if the P-block favoritism comes from pre-training bias, SFT, or DPO learning?

### Soundness
4

### Presentation
4

### Contribution
3
