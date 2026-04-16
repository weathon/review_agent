The paper content is already in my context. Let me now compose the final review based on my thorough analysis.

## Summary

DyGPrompt proposes a prompt learning framework for dynamic (continuous-time) graphs that introduces dual prompts (node and time) to bridge task and temporal gaps between pre-training and downstream tasks, and dual condition-nets (time condition-net generating time-conditioned node prompts, node condition-net generating node-conditioned time prompts) to capture evolving node-time patterns. The framework is evaluated on four benchmark datasets for temporal node classification and link prediction, showing improvements over baselines.

## Strengths

- **Novel and well-motivated problem**: Extending prompt learning from static to dynamic graphs is a genuinely important and underexplored direction. The identification of both task objective gaps and temporal variation gaps in pre-training→downstream transfer on dynamic graphs is clearly articulated and goes beyond what prior static graph prompting methods address (Sect. 1).

- **Elegant, modular framework design**: The dual prompt + dual condition-net design is principled and conceptually clean. The hypernetwork-style condition-nets generate context-dependent prompts rather than parameterizing per-node/per-timestamp prompts, which is a sensible parameter-efficient approach. The framework plugs into arbitrary DGNN backbones (Sect. 4, Table 3).

- **Strong empirical results in the evaluated setting**: DyGPrompt consistently improves over baselines across 4 datasets, 3 task types, and 6 different DGNN backbones (Tables 1 and 3). The backbone-agnostic nature and consistent gains are a genuine strength.

- **Comprehensive ablation study**: Table 2 systematically varies each component, demonstrating that node prompts, time prompts, and condition-nets each contribute. Figure 3 provides qualitative evidence.

## Weaknesses

### Major

- **Asymmetric evaluation protocol clouds baseline comparisons**: The paper compares DyGPrompt (frozen backbone + lightweight prompt tuning on 30 events) against baseline methods that appear to use different training regimes. For conventional DGNNs (TGAT, TGN etc.), the paper states they are "continually trained on the same training data" (Sect. 5.2, the Remark), which is fine-tuning all parameters rather than prompt tuning, providing an advantage in terms of model capacity but a disadvantage in data scarcity. For DDGCL/CPDG (pre-train/fine-tune), it is unclear whether they are constrained to the same 30-event budget. For GraphPrompt/ProG (static graph prompting), it is unclear if they share the same pre-training objective. The paper's core claim of "outstanding performance" over "state-of-the-art approaches" hinges on these comparisons, yet the training regimes are not uniformly controlled. This does not invalidate the overall finding of DyGPrompt's effectiveness, but it does undermine the strength of the claim of superiority over all baselines. The self-comparison (Table 3, where the same backbone is used with and without DyGPrompt) is more trustworthy and already demonstrates the framework's value.

- **Extreme few-shot evaluation regime with limited generalization analysis**: Downstream tasks are constructed from only ~30 events (≈0.01% of the dataset). While data scarcity is a valid and important setting for prompt learning, the paper's broad claims (Section 6: "significantly outperforms various state-of-the-art baselines") lack the important caveat that this is demonstrated only in this very extreme few-shot regime. No experiments examine how performance changes with increasing downstream data (e.g., 0.1%, 1%, 5%), which would establish the practical boundaries of when prompting is preferred over fine-tuning. On MOOC link prediction and Genre node classification, several methods (including DyGPrompt) achieve near-random AUC (~50–53%), which goes unremarked.

- **Core conceptual claim about "mutual characterization" is undersubstantiated**: The paper's central novelty is the dual condition-net mechanism for node-time mutual characterization (Sect. 1, 4.4). However: (a) the condition-nets are small MLPs with bottleneck structures—a fairly generic feature modulation approach, similar to FiLM conditioning; (b) the ablation (Table 2) does not include the obvious simpler alternative of a single joint conditioner that concatenates node and time features, making it unclear whether the specific dual, asymmetric design is necessary; (c) in the ablation, Variant 4 (dual prompts, no condition-nets) underperforms Variant 2 (node prompt only) on Wikipedia node classification (72.25 vs. 72.59), suggesting that adding the time prompt alone can hurt—yet this anomaly is unaddressed. This weakens the claim that both dual prompts and dual condition-nets are jointly necessary.

### Minor

- **The motivation figure (Fig. 1a) illustrates rich temporal context (morning/professor/noon/student), but the method only uses scalar timestamps via sinusoidal time encoding and MLP-based modulation**: There is no mechanism that explicitly captures periodic or semantic temporal patterns. The gap between motivation and mechanism is acknowledged in passing but not discussed.

- **Equation (9) notational ambiguity**: The left and right sides both use the symbol $\tilde{\mathbf{p}}_{t,v}^{time}$ for different quantities (the node-conditioned time prompt on the left, and the result of element-wise multiplication on the right). Similarly, in Eq. (10), $\tilde{\mathbf{p}}^{time}_{t',v}$ in the neighbor aggregation should presumably be $\tilde{\mathbf{p}}^{time}_{t',u}$ conditioned on neighbor $u$, not $v$, to be consistent with the node-conditioning described in Sect. 4.4. These notational issues make the dataflow harder to verify.

- **Prototype computation in Eq. (11) under extreme few-shot**: Class prototypes $\bar{\mathbf{h}}_{t_i,y}$ are defined as "mean embeddings of examples in class $y$ at time $t_i$", but with only ~30 events total (possibly 1 example per class per timestamp), this averaging is over extremely small samples, which is not discussed.

### Trivial
- None worth listing.

## Nice-to-Haves

- Scaling analysis showing DyGPrompt's performance at different downstream data ratios (0.1%, 1%, 5%, 10%) vs. fine-tuning and parameter-matched fine-tuning baselines, to establish when prompting is genuinely advantageous.
- Comparison against simpler conditioning alternatives (single joint MLP over concatenated [x_t, TE(t)] to justify the dual condition-net design.
- Temporal analysis: report performance stratified by how far downstream test events are from pre-training data (early vs. late), to validate the time prompt's effectiveness for bridging temporal gaps.
- Report total tunable parameter counts alongside results, to substantiate the parameter-efficiency claim.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Missing comparison with LoRA/adapters"**: The paper does not compare against parameter-efficient fine-tuning alternatives like LoRA. While this could be informative, LoRA is not standard in the graph prompt learning literature and the paper's stated scope is prompt-based adaptation. This is a nice-to-have, not a required comparison. (From Spark and Human Finder)

- **"Scalability concerns"**: The human finder raised scalability concerns about condition-nets for large graphs. However, the condition-nets are lightweight MLPs with bottleneck structures (parameterized by α), and the paper explicitly discusses parameter efficiency. The concern is speculative without evidence of actual computational issues on the evaluated benchmarks. (From Human Finder)

- **"Limited dataset diversity"**: The four datasets (Wikipedia, Reddit, MOOC, Genre) are standard benchmarks for dynamic graph methods. While more diverse datasets would be better, this is a generic concern that applies to most papers in the area. (From Human Finder)

- **"Missing analysis of pre-training data relationship"**: How pre-training data quantity/quality affects downstream performance is an interesting direction but falls outside the paper's stated scope, which focuses on prompt-based adaptation given a pre-trained model. (From Human Finder)

- **"The similarity function sim() is unspecified"**: The harsh critic noted this, but it is a standard choice (likely cosine similarity given the contrastive setup and temperature), and its exact form does not affect the core claims.

- **"Overclaiming novelty of condition-nets"**: The human finder argued the condition-nets are "straightforward" and "incremental." However, the application of hypernetwork-style conditioning to generate node-time mutual prompts on dynamic graphs is a specific design contribution, even if individual MLP components are standard. The novelty claim-level is appropriate for the contribution.

## Novel Insights

The key insight emerging from the reviews is that DyGPrompt's empirical story is strongest when evaluated in a controlled, same-backbone comparison (Table 3), where the framework consistently improves across 6 DGNN backbones—but weakest when claimed as a wholesale "state-of-the-art" over heterogeneous baselines (Table 1) where training regimes differ. The conceptual contribution of dual conditioning (node↔time) is appealing but undersubstantiated relative to simpler alternatives. The paper's overall value proposition—prompt-based parameter-efficient adaptation for dynamic graphs—is sound and timely, but the evidence for the specific architectural mechanism vs. simpler modulation schemes remains incomplete.

## Suggestions

- **Unify baseline training regimes**: Run at least one strong baseline (e.g., TGN or TGAT) under exactly the same frozen-backbone + 30-event protocol to isolate the effect of prompting vs. fine-tuning. Table 3 partially addresses this ("—" rows) but explicitly showing a parameter-matched fine-tuning baseline would be more convincing.
- **Add data-scaling curves**: Performance vs. downstream data fraction (0.01%, 0.1%, 1%, 5%, 10%) would establish the practical regime where DyGPrompt excels and where fine-tuning catches up, and is easy to run from the existing setup.
- **Add a "single joint conditioner" ablation**: Replace the dual condition-nets with one MLP that takes concatenated [x_{t,v}, TE(t)] and outputs both prompt vectors. This is the most natural alternative to test whether the dual, mutual conditioning is necessary.

## Score and Decision

Calibration: I compared against several graph prompt learning papers:
- IA-GPL (Instance-Aware Graph Prompt Learning): similar in spirit (instance-level prompts on graphs), scored 5-6 by reviewers, Withdrawn/Reject. Had concerns about scalability, novelty of conditioning mechanism, and unfair parameter comparisons.
- PromptST (Simple Yet Effective Spatio-Temporal Prompt Learning): similar in extending prompts to temporal settings, scored 3-5, Withdrawn/Reject. Core issues were incremental novelty, unfair baseline comparisons, and overclaiming.
- GPromptShield: more niche (adversarial robustness of graph prompts), scored 6, Accept (Poster).

DyGPrompt is stronger than PromptST in that it introduces a more principled design (dual condition-nets for mutual node-time characterization) and demonstrates backbone-agnostic gains. It is comparable to IA-GPL in the conditioning mechanism space, but IA-GPL had similar concerns about whether the added complexity (codebook, PHM layers) justified marginal improvements. DyGPrompt's evaluation protocol issues and the undersubstantiation of its core novelty (dual conditioning vs. simpler alternatives) are significant but not fatal—it does show consistent improvements across multiple backbones and tasks. The paper makes a genuine contribution in extending prompt learning to dynamic graphs, but the claims slightly overreach the evidence.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>