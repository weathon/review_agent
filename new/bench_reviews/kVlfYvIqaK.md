The paper content was already provided in the user message. Let me verify the key claims from the harsh critic directly from that content.

Looking at Table 2:
- Variant 1: ✓ node prompt, ✗ time prompt, ✗ NCN, ✗ TCN → Wikipedia NC: 67.00, Reddit NC: 53.64, MOOC NC: 59.27
- The paper states TGAT gets: Wikipedia NC: 67.00, Reddit NC: 53.64, MOOC NC: 59.27 (from Table 1, TGAT row)
- So Variant 1 WITH a node prompt gives the EXACT same results as TGAT WITHOUT a node prompt. This is indeed suspicious - it means the node prompt has zero effect when used alone. The harsh critic is right about this inconsistency.

Actually wait - looking more carefully at Table 2, Variant 1 has ✓ for node prompt. But its results exactly match the TGAT baseline from Table 1. This is confusing. Let me check if Variant 1 might be "no prompts at all" (mislabeled) or if the node prompt truly has zero effect.

Looking at Variant 2 (✗ node, ✗ time, ✗ NCN, ✗ TCN): 72.59, 61.82, 63.50 for NC on Wikipedia, Reddit, MOOC. This performs BETTER than Variant 1 (which supposedly has node prompt). This is very strange - a variant with NO prompts outperforms one with a node prompt by 5+ points.

This suggests that the variants might have different pre-training or fine-tuning setups, or there's a labeling issue. Let me look at the paper text more carefully about what Variant 1 and 2 are.

Actually, re-reading Section 4.3, the node prompt modifies features BEFORE they go into the DGNN, and then Eq. 11 is the downstream loss using similarity-based classification with prototypes. Variants 1-6 all appear to use the same pre-trained DGNN. But Variant 1 seems to also use the similarity-based task template while having a node prompt. If Variant 1 = TGAT (exact same numbers), then either:
1. The node prompt with the node prompt alone without the prototype/similarity classification approach gives no benefit, OR
2. There's a labeling/inconsistency issue

Looking at the paper's discussion of Table 2: "Variant 2 (with node prompt) and Variant 3 (with time prompt) outperform Variant 1 (without these prompts)." But per the table, Variant 1 HAS a node prompt (✓) while Variant 2 does NOT (✗). So either the paper text is wrong about what the checkmarks mean, or the checkmarks are wrong.

This is indeed a real inconsistency that needs to be flagged.

Now let me think about the inductive evaluation concern. The paper says: "For inductive link prediction, we remove instances from the test set if their nodes have appeared in the pre-training or downstream training data." So they remove test instances involving nodes seen during pre-training. This means the pre-training phase DOES see all nodes (in the first 80% of events), but at test time, only edges involving novel nodes (ones not seen in pre-training or training) are evaluated. This is consistent - it's a standard inductive setup for dynamic graphs. The pre-trained model doesn't necessarily have node embeddings for unseen nodes, but the time encoder and structural patterns may still generalize.

Actually wait - for DGNNs like TGAT/TGN that compute node embeddings based on interactions, when you encounter a truly novel node at test time, the model has no prior interaction history for it. So having high performance on inductive nodes would actually be impressive. The concern about "information leakage" from pre-training doesn't apply as strongly here - the inductive test specifically filters out nodes seen in pre-training.

However, the harsh critic's concern about transductive link prediction is partially valid: with 80% pre-training data on the same graph, the model has already learned rich representations of the test nodes. The temporal gap is real (last 20% events), but the model isn't adapting to truly new nodes.

Let me now compose the final review.</think>Now I have sufficient information to write the final consolidated review. Let me verify a few more specific claims from the harsh critic about Eq. 9 and Eq. 10 from the paper text:

Looking at the paper content again:

Eq. (9): "$\tilde{\mathbf{p}}_{t,v}^{\text{time}} = \tilde{\mathbf{p}}_{t,v}^{\text{time}} \odot \tilde{\mathbf{f}}_t^{\text{time}}$"

This does indeed have $\tilde{\mathbf{p}}_{t,v}^{\text{time}}$ on both sides. Based on context, the left side should be a modified time feature, like $\tilde{\mathbf{f}}_{t,v}^{\text{time}}$.

Eq. (10): "$\mathbf{h}_{t,v} = \text{DGE} \left( \text{Fuse}(\tilde{\mathbf{x}}_{t,v}^{\text{node}}, \tilde{\mathbf{p}}_{t,v}^{\text{time}}), \left\{ \text{Fuse}(\tilde{\mathbf{x}}_{t',u}^{\text{node}}, \tilde{\mathbf{p}}_{t',v}^{\text{time}}) : (u, t') \in \mathcal{N}_v \right\} \right)$"

The neighbor term uses $\tilde{\mathbf{p}}_{t',v}^{\text{time}}$ (v's time prompt) instead of $\tilde{\mathbf{p}}_{t',u}^{\text{time}}$ (u's time prompt). This could be intentional (using v's node-conditioned time prompt throughout aggregation) or a typo. Given the context of message-passing, using the target node v's time prompt for all neighbors could be a design choice rather than an error, but it's noteworthy.

Now let me write the final review.

## Summary
DyGPrompt introduces a prompt learning framework for dynamic graphs that addresses both task objective gaps and temporal gaps between pre-training and downstream tasks. It proposes "dual prompts" (a node prompt and a time prompt) to modify node and time features, and "dual condition-nets" (time condition-net and node condition-net) that generate context-dependent prompts conditioned on node/time features to capture evolving node—time patterns. Experiments on four benchmarks demonstrate improvements over baselines in few-shot node classification and link prediction tasks.

## Strengths
- **Novel and well-motivated problem:** The paper identifies a meaningful gap in existing graph prompt learning, which has focused on static graphs, and tackles the unique temporal dynamics of continuous-time dynamic graphs. The observation that node and time patterns mutually characterize each other in evolving interaction networks (e.g., students vs. professors commenting at different times) is intuitive and well-illustrated (Figure 1).
- **Coherent technical design:** The dual prompts (node/time) and dual condition-nets form a principled, modular framework. Using lightweight bottleneck MLPs as hypernetwork-style condition-nets is a parameter-efficient way to generate context-dependent prompts rather than storing per-node/per-timestamp parameters, which is a sensible design choice.
- **Backbone-agnostic validation:** Table 3 demonstrates that DyGPrompt consistently improves over 6 different DGNN backbones (DyRep, JODIE, TGAT, TGN, TREND, GraphMixer), providing strong evidence that the proposed prompt mechanism is not tied to a specific architecture and is broadly applicable.
- **Comprehensive baselines:** The paper compares against four categories of methods—conventional DGNNs, dynamic graph pre-training, static graph prompting, and a contemporary dynamic graph prompting method (TIGPrompt)—across three tasks (node classification, transductive/inductive link prediction) on four datasets.

## Weaknesses

### Fatal
None.

### Major

- **Ablation study has a significant labeling inconsistency that undermines interpretation.** In Table 2, Variant 1 is marked as having a node prompt (✓), yet its results exactly match the vanilla TGAT baseline from Table 1 (e.g., Node Classification Wikipedia: 67.00, Reddit: 53.64, MOOC: 59.27). This either means the node prompt has zero effect when applied alone, or the table is mislabeled. Further, Variant 2 (marked as having ✗ for all prompts) substantially outperforms Variant 1 on node classification (72.59 vs. 67.00 on Wikipedia). If Variant 2 truly has no prompts, it is unclear what mechanism produces these gains over Variant 1 (which supposedly has a node prompt). The paper's text states "Variant 2 (with node prompt)" while the table marks Variant 2 as ✗ for node prompt—a direct contradiction. This inconsistency makes it impossible to reliably assess the contribution of each component, which is critical because the condition-nets are a central novelty claim.

- **The parameter-efficiency claim for prompt tuning is unsubstantiated.** The paper repeatedly emphasizes that DyGPrompt is "parameter-efficient" (e.g., "significantly reducing the number of learnable parameters," "lightweight condition-nets"). However, no parameter count comparison is provided between DyGPrompt's tunable parameters (prompt vectors p^node, p^time plus MLPs for TCN and NCN) and the baselines. The condition-nets are MLPs, and their total parameter count depends on dimension d and bottleneck factor α. Without explicit parameter counts, it is unclear whether the additional capacity from condition-nets—rather than the prompt design itself—drives the improvements. This is especially relevant because the closest comparator (TIGPrompt) also uses a prompt generator.

- **Ablation inconsistencies on link prediction are not adequately explained.** In Table 2, adding both prompts (Variant 4) sometimes performs worse than a single prompt (Variants 2/3) for transductive link prediction on Wikipedia (68.36 vs. 68.12). Adding both condition-nets (Variant 6 → full DyGPrompt) underperforms Variant 5 (NCN only) on multiple link prediction metrics. The paper attributes the MOOC anomaly to "less dynamic" interactions, but does not explain the broader pattern where more components sometimes hurt. This challenges the narrative that all components are consistently beneficial.

### Minor

- **Equation (9) has a notation error:** $\tilde{\mathbf{p}}_{t,v}^{\text{time}}$ appears on both the left and right sides. Based on context, the left side should represent a modified time feature (e.g., $\tilde{\mathbf{f}}_{t,v}^{\text{time}}$). While likely a typo, this creates ambiguity in interpreting the downstream equations.

- **The inductive evaluation protocol could be more precisely defined.** The paper states that "we remove instances from the test set if their nodes have appeared in the pre-training or downstream training data" for inductive link prediction. Since pre-training uses the first 80% of events (which includes most nodes in the graph), it is unclear how many test instances actually remain for the inductive setting. The surprisingly high inductive AUCs (e.g., 95.69% for TGN-DyGPrompt on Reddit) warrant clarification about what proportion of test instances survive filtering and whether these are a biased subset.

- **The paper evaluates only with 30 training events per task in data-scarce scenarios** but does not include experiments varying the number of downstream training events (e.g., 10, 50, 100, 500). Showing how DyGPrompt's advantage scales with data availability would strengthen the core claim that prompt learning is specifically beneficial in low-data regimes.

- **No statistical significance tests** are reported despite 500 runs (100 tasks × 5 seeds), though standard deviations are provided and generally small.

### Trivial
- Minor text repetition in Section 2: "(Liu et al., 2023a; 2023a)" should be "(Liu et al., 2023a; 2023b)" or similar.

## Nice-to-Haves
- Vary the downstream training data size to demonstrate that DyGPrompt's advantage is specific to data-scarce regimes and not simply from adding learnable parameters.
- Visualize the condition-net outputs over time for the same node (and across nodes at the same time) to validate that TCN/NCN actually learn meaningful node—time interactions rather than acting as additional capacity.
- Compare DyGPrompt against fine-tuning the entire model with the same 30-event budget to isolate the benefit of prompt design from the parameter budget.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that the inductive evaluation is fundamentally broken because pre-training sees all nodes.** The paper explicitly states that inductive test instances involving nodes seen during pre-training are removed. Dynamic graph models like TGAT/TGN compute embeddings from interaction histories; if a node has no prior interactions, the model must rely solely on node features and structural generalization. The high inductive AUCs are plausible because some test nodes may appear only in the last 20% of events. The description, while terse, is consistent with standard inductive evaluation in dynamic graph literature.

- **Harsh Critic's claim that the "objective gap" is smaller than portrayed for node classification because link prediction pre-training already encodes user behavior.** This is speculative and does not constitute a methodological flaw. Pre-training on link prediction and adapting to classification with 30 labeled examples is still a meaningful adaptation challenge, even if the pre-trained representations capture some behavioral patterns. The gap in task objectives (contrastive link prediction vs. prototype-based classification) is real.

- **Harsh Critic's demand for "an analysis showing that the time prompt meaningfully re-aligns downstream timestamps" or "a comparison to fine-tuning TE with a small learning rate."** While these would strengthen the paper, they are beyond its stated scope. The paper provides ablation studies showing that the time prompt contributes to performance, and demanding theoretical validation of the specific mechanism goes beyond what is standard for empirical contributions in this area.

- **Harsh Critic's claim that Variant 2 (no prompts) outperforming Variant 1 (node prompt) is "unexplained" and therefore suspicious.** Upon reflection, Variant 2 likely represents a different downstream evaluation protocol (using the similarity-based task template from Eq. 11) without any prompt modification, while Variant 1 adds a node prompt on top. The similarity template itself could explain gains over the raw TGAT. The inconsistency is real (and noted as a Major weakness above), but the explanation that a different evaluation protocol drives Variant 2's gains is plausible.

- **Demand for experiments on larger/newer benchmark datasets from TGB.** The four datasets used are standard benchmarks in the dynamic graph literature. Requesting additional datasets is a nice-to-have rather than a core weakness.

- **Request for statistical significance tests with 500 runs.** While this would strengthen the paper, single-run evaluation with standard deviations is the norm in this area; this is a nice-to-have rather than a weakness.

## Novel Insights
The most insightful observation from the reviews is that the ablation table (Table 2) contains a fundamental labeling inconsistency: Variant 1 (purportedly with a node prompt) exactly matches the vanilla TGAT baseline, and Variant 2 (purportedly without any prompts) substantially outperforms it. The paper text contradicts its own table header regarding which variant has which component. This is not merely a cosmetic issue—it directly undermines the paper's ability to make causal claims about the contribution of each proposed component, which is the foundation for validating the central novelty of conditional prompt generation via dual condition-nets.

## Suggestions
- **Clarify and correct Table 2's variant labels and ensure they match the text discussion.** If Variant 1 truly includes a node prompt but shows zero gain, explain why. If the checkmarks are wrong, correct them and re-explain the ablation results.
- **Report tunable parameter counts** for DyGPrompt (prompts + condition-nets) versus baselines like TIGPrompt, GraphPrompt, and ProG, to substantiate the parameter-efficiency claim.
- **Add a simple experiment** comparing DyGPrompt to full fine-tuning of the backbone with the same 30-event budget, to isolate the contribution of the prompt architecture from the restricted parameter budget.
- **Clarify the inductive setting** by reporting how many test instances remain after filtering and what percentage of nodes are truly unseen during pre-training.

## Evaluation on Key Axes
- **Originality:** Moderate-to-high. Prompt learning for dynamic graphs with dual prompts and condition-nets is novel. The mutual node—time characterization is a meaningful insight.
- **Importance of research question:** High. Bridging pre-training and downstream adaptation on dynamic graphs is an important and timely problem.
- **Claims well-supported:** Partially. The main empirical results show strong improvements, but the ablation study has a labeling inconsistency that weakens the ability to attribute gains to specific components. The parameter-efficiency claim is unverified.
- **Soundness of experiments:** Moderate. The evaluation protocol is reasonable but the ablation table undermines component-level claims. The 30-event training budget is quite restrictive and only one data budget is tested.
- **Clarity of writing:** Generally good. The paper is well-structured and the technical approach is clearly described, though Eq. 9 has a notation error and the ablation table has labeling issues.
- **Value to research community:** Moderate-to-high. The framework is backbone-agnostic and addresses a real gap, but the current evidence for the condition-net contribution needs repair.

## Score and Decision

Calibration: 
- IA-GPL (instance-aware graph prompt learning on static graphs, similar topic but without condition-nets): scores 5-6, treated as reject/withdrawn. Key weaknesses: unfair parameter comparison, limited improvement.
- Multi-modal prompt learning for GNNs: scores 3-5, rejected. Weaker novelty and validation.
- GPromptShield (graph prompt tuning for adversarial robustness): scores 6, accepted poster.
- FreeDyG (dynamic graph link prediction): scores 5-8, accepted poster. Novel frequency-based approach but with evaluation gaps.
- OFA (one graph model for all tasks, graph prompt): scores 6-10, accepted spotlight. Strong novelty and broad scope.

This paper has real novelty (prompt learning for dynamic graphs, condition-nets for node—time mutual characterization) and strong empirical results, but suffers from a significant ablation inconsistency that undermines the core novelty validation. The parameter-efficiency claim is unsubstantiated. These are not fatal but are genuinely major issues. Compared to IA-GPL which received 5-6, this paper has a similar novelty profile but also has the ablation table problem. Compared to FreeDyG (5-8, mean ~6.75), this paper addresses a similar area with somewhat stronger results but a more notable evidential gap.

Given the ablation inconsistency (Major) and unsubstantiated efficiency claim (Major), balanced against the genuine novelty and backbone-agnostic results, I place this at a borderline score.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>