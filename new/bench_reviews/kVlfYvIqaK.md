Let me also check how Variant 1 is defined. Looking at Table 2 header again:

Variant 1: ✓ node prompt, ✗ time prompt, ✗ NCN, ✗ TCN — wait no. Let me re-read:

| VARIANT 1 | ✓ | ✗ | ✗ | ✗ | 67.00 | ...

This means Variant 1 has only the node prompt (✓ for node prompt, ✗ for time prompt, ✗ for NCN, ✗ for TCN). But its AUC is 67.00, which is the same as plain TGAT (67.00 in Table 1). That is suspicious — it says Variant 1 has node prompt only but achieves the same performance as plain TGAT without any prompting.

Wait, let me re-read. In Table 1, TGAT achieves 67.00 on Wikipedia NC. In Table 2, Variant 1 (with node prompt only) also achieves 67.00. The harsh critic noted this. But actually, looking at the table header columns:

| Methods | Node prompt | Time prompt | NCN | TCN | Node classification | ...

Variant 1: ✓ (node prompt), ✗ (time prompt), ✗ (NCN), ✗ (TCN) — so it HAS a node prompt but NO time prompt, NO condition-nets. The fact that it achieves exactly the same number as plain TGAT suggests it might be the backbone evaluated with the same protocol (just using prototype matching for classification), or the node prompt alone doesn't help.

Actually, looking at the footnote: "-" in Table 3 refers to fine-tuning or continual training. Also, the baselines in Table 1 (TGAT, TGN etc.) are "continually trained" on the downstream data. So Variant 1 in Table 2 is the TGAT backbone evaluated with node prompt but getting the same score — this suggests the node prompt by itself doesn't help, or Variant 1 is actually just frozen backbone with prototype matching (GraphPrompt-style task template).

Let me check: in GraphPrompt and this paper's prompt tuning approach, only the prompts are tuned while the backbone is frozen. The baselines (TGAT, TGN etc.) in Table 1 are "continually trained." So Variant 1 with ✓ for node prompt — it would use the frozen backbone + node prompt for task template matching. And it gets 67.00, same as TGAT (which is continually trained with 30 events). So this suggests that even a single prompt on TGAT backbone doesn't improve over continual training in NC.

But looking at Variant 2 (no node prompt, no time prompt, no NCN, no TCN) — all ✗ — gets 72.59. Wait, that means variant 2 has NO prompts at all? But it gets 72.59 which is HIGHER than Variant 1 (67.00) with node prompt.

That's confusing. Let me re-examine. Wait - looking again at Table 2 headers vs the actual column alignment. Let me reread carefully:

```
| VARIANT 1 | ✓ | ✗ | ✗ | ✗ | 67.00 | 53.64 | 59.27 | 55.78 | 62.43 | 51.49 | 48.21 | 57.30 | 51.42 |
| VARIANT 2 | ✗ | ✗ | ✗ | ✗ | 72.59 | 61.82 | 63.50 | 68.12 | 88.59 | 51.24 | 51.89 | 74.84 | 51.37 |
```

So Variant 2 has ALL ✗ — meaning no prompts at all and no condition-nets? But Variant 1 has ✓ for the first column (Node prompt). If Variant 1 has node prompt and gets 67.00 while Variant 2 has nothing and gets 72.59, that's strange.

Wait, maybe the columns don't mean what I thought. Let me look more carefully. The headers are:

| Methods | Node prompt | Time prompt | NCN | TCN |

So Variant 1: ✓ Node prompt, ✗ Time prompt, ✗ NCN, ✗ TCN
Variant 2: ✗ Node prompt, ✗ Time prompt, ✗ NCN, ✗ TCN

Variant 2 has NO prompts and NO condition nets, yet achieves 72.59? That would mean the prompt-free backbone with similarity-based task template achieves 72.59? But then how does TGAT in Table 1 only get 67.00?

I think the key is: Variant 1-6 and DyGPrompt all use the frozen-backbone + similarity-based task template approach (same as GraphPrompt). The TGAT baseline in Table 1 is continually trained (fine-tuned). So:

- TGAT (continually trained on 30 events): 67.00 → this overfits on 30 events
- Variant 1 (frozen TGAT + node prompt + task template): 67.00 → Same as fine-tuned TGAT!
- Variant 2 (frozen TGAT + task template, NO prompts): 72.59 → Much better!

Wait, that doesn't make sense. If Variant 2 has no prompts, it's just frozen backbone + task template. But that gets 72.59 (higher than Variant 1 with node prompt at 67.00)?

Hmm, actually Variant 1 has ✓ for Node prompt, which means it has the static node prompt p^node applied. Maybe the static node prompt actually hurts? That would be explained by Variant 2 not having it. But then, why does DyGPrompt with node prompt ✓ get 82.09?

Actually, I think I need to reconsider. Perhaps the ✓/✗ pattern works differently. Let me look at Variant 4: ✓ node, ✓ time, ✗ NCN, ✗ TCN = 72.25. And Variant 2 (all ✗) = 72.59. So having both prompts WITHOUT condition nets actually slightly hurts. And Variant 3 (time only, no condition nets) = 73.22, time prompt alone slightly better than nothing.

Then Variant 5: ✓ node, ✗ time, ✓ NCN, ✗ TCN = 81.40 — big jump. This means adding NCN to the node prompt gives a massive improvement. And DyGPrompt: ✓ all = 82.09.

But wait, looking at Variant 6: ✗ node, ✓ time, ✗ NCN, ✓ TCN = ??? Let me check:

| VARIANT 6 | ✓ | ✓ | ✗ | ✓ | ... wait, that doesn't match. Let me reread:

| VARIANT 6 | ✓ | ✓ | ✓ | ✗ | 80.34 ...

No wait:
| VARIANT 5 | ✓ | ✗ | ✓ | ✗ | 
| VARIANT 6 | ✗ | ✓ | ✗ | ✓ | 

Hmm, actually from the paper text: "the node condition-net is advantageous... Variant 5 outperforms Variant 2" and "the time condition-net is beneficial... Variant 6 outperforms Variant 3."

So Variant 5 adds NCN on top of Variant 2 (which had node prompt only? or nothing?). Wait the text says "Variant 5 (NP+NCN)" but the table shows Variant 5 has ✓ for Node prompt. Actually let me re-check the table structure:

Looking at the headers again carefully:
```
| Methods | Node prompt | Time prompt | NCN | TCN | Node classification | ...
```

And:
- Variant 1: ✓ | ✗ | ✗ | ✗  — Node prompt only
- Variant 2: ✗ | ✗ | ✗ | ✗  — Nothing (frozen backbone + task template only)
- Variant 3: ✗ | ✓ | ✗ | ✗  — Time prompt only
- Variant 4: ✓ | ✓ | ✗ | ✗  — Both prompts, no condition nets
- Variant 5: ✓ | ✗ | ✓ | ✗  — Node prompt + NCN
- Variant 6: ✗ | ✓ | ✗ | ✓  — Time prompt + TCN

This seems odd: Variant 5 has NCN (node condition net) but no time prompt. NCN takes already-masked node features and generates time prompts. But the NCN output is a time prompt (node-conditioned). So even without a static time prompt, the NCN can generate time-conditioned prompts... but wait, Eq. 8 shows that NCN takes $\mathbf{x}_{t,v}^{node}$ as input. Without a node prompt (Variant 5 has ✓ node prompt, so $\mathbf{x}_{t,v}^{node} = \mathbf{p}^{node} \odot \mathbf{x}_{t,v}$). Hmm, actually Variant 5 has ✓ for node prompt.

OK but the key issue remains: Variant 2 with no prompts at all gets 72.59 on Wikipedia NC, which is much higher than continual-training TGAT (67.00). This suggests that even frozen backbone + task template without any prompts is already pretty good. Then adding node prompt (Variant 1, 67.00) actually hurts compared to no prompts (72.59). Node prompt is actually harmful without condition-nets.

This is actually an important observation. The node prompt p^node is a simple static element-wise mask, and when applied alone it hurts performance. The condition-nets are what actually make the prompts effective. The harsh critic noted that margin of condition-nets was small, but actually the ablation shows condition-nets provide the big jumps: Variant 2 → Variant 5 goes from 72.59 to 81.40, a ~9 point gain.

Now let me think about the Eq. 10 indexing issue more carefully. In Eq. 10:

$$\mathbf{h}_{t,v} = \text{DGE} \left( \text{Fuse}(\tilde{\mathbf{x}}_{t,v}^{\text{node}}, \tilde{\mathbf{p}}_{t,v}^{\text{time}}), \left\{ \text{Fuse}(\tilde{\mathbf{x}}_{t',u}^{\text{node}}, \tilde{\mathbf{p}}_{t',v}^{\text{time}}) : (u, t') \in \mathcal{N}_v \right\} \right)$$

The neighbor representation uses $\tilde{\mathbf{p}}_{t',v}^{\text{time}}$ — the time prompt conditioned on node $v$ (the target node) instead of node $u$ (the neighbor). This is actually by design: the NCN generates node-conditioned time prompts. If the DGE is computing the embedding for node $v$, then for each neighbor $(u,t')$, the time prompt used is conditioned on the *target* node $v$'s features at time $t'$. This could be intentional as an "ego-centric" design where all temporal prompts in the aggregation are conditioned on the target node $v$'s perspective.

But the harsh critic has a point that this contradicts the stated motivation that "different nodes may evolve differently over the same time interval" — if that's the case, shouldn't each neighbor's time prompt be conditioned on that neighbor's features? The paper says NCN generates "a series of node-conditioned time prompts" (Eq. 8 generates $\tilde{\mathbf{p}}_{t,v}^{\text{time}}$ conditioned on $\mathbf{x}_{t,v}^{\text{node}}$). So it's conditioned per target-node not per source-node, which is somewhat ego-centric. This is a valid design question but could be intentional.

Now let me consolidate everything and write the review.</think>Now I have all the information needed. Let me construct the final review.

## Summary

The paper proposes DyGPrompt, a prompt learning framework for dynamic graphs that introduces dual prompts (node prompt and time prompt) and dual condition-nets (time condition-net and node condition-net) to bridge both the task gap and temporal gap between pre-training and downstream tasks. The condition-nets generate input-dependent prompts via lightweight MLP hypernetworks, capturing node-time mutual characterization. Experiments on four datasets with a few-shot (30 events) protocol show improvements over fine-tuning baselines, static graph prompting methods, and the dynamic graph prompting baseline TiGPrompt.

## Strengths

- **Clear and well-motivated problem decomposition.** The paper identifies two distinct gaps—task objective gap (via node prompts) and temporal gap (via time prompts)—and provides an intuitive motivation in Figure 1, making the dual-prompt design conceptually clean and easy to follow.

- **Condition-nets provide substantial empirical gains.** The ablation (Table 2) shows that the node condition-net (NCN) drives the largest improvement: Variant 2 (no prompts, no CN) → Variant 5 (+NCN) jumps from 72.59 to 81.40 on Wikipedia NC, and 61.82 to 73.12 on Reddit NC. These are meaningful margins, not marginal ones—contrary to the comparison against the full model that yields small increments.

- **Backbone-agnostic evaluation across six DGNN architectures.** Table 3 demonstrates improvements on DyRep, JODIE, TGAT, TGN, TREND, and GraphMixer backbones, showing the framework generalizes beyond a single encoder—a meaningful robustness check.

- **Parameter-efficient design.** The condition-nets generate prompts from input features via bottleneck MLPs rather than directly parameterizing per-node/per-timestamp prompts (Eqs. 6–9), which scales well and is a principled design choice.

- **Transparent evaluation design discussion.** The Remark in Section 5.2 explicitly acknowledges that TGAT/TGN scores are lower than in their original papers due to the changed data split, showing honesty about the experimental setup.

## Weaknesses

### Fatal
None.

### Major

- **Eq. 10 uses the target node's conditioned time prompt for all neighbors, which creates an inconsistency with the stated motivation.** In the message-passing step (Eq. 10), neighbor representations use $\tilde{\mathbf{p}}_{t',v}^{\text{time}}$—conditioned on the **target node** $v$'s features—rather than $\tilde{\mathbf{p}}_{t',u}^{\text{time}}$ conditioned on each **neighbor** $u$'s features. This ego-centric aggregation directly contradicts the paper's central claim that "different nodes may evolve differently over the same time interval" (Section 4.4) and that node-time mutual characterization should be per-node. If intentional (an ego-centric design choice), it requires explicit justification, as it means all neighbors' temporal signals are filtered through the target node's perspective rather than their own. If a bug, the implementation may deviate from the conceptual framework. Either way, this design choice weakens the alignment between motivation and method.

- **The ultra-low-data regime (30 events) structurally favors prompt methods over fine-tuning baselines, making headline comparisons hard to interpret.** With only ~30 training events (0.01% of the dataset), full fine-tuning methods (TGAT, TGN, etc.) are expected to overfit, and indeed many achieve near-random performance (e.g., TGN at 50.61% on Wikipedia NC). The paper's strongest claims—large margins over baselines—are largely driven by this regime. The critical missing comparison is a **frozen-backbone + linear probe** baseline (using pre-trained embeddings with a simple classifier on 30 events), which would isolate whether the gains come from the prompt design per se or simply from not fine-tuning the backbone. Variant 2 in Table 2 (no prompts, no CN, frozen backbone + task template) already gets 72.59% on Wikipedia NC without any prompts—far above fine-tuned TGAT's 67.00%—suggesting much of the benefit comes from freezing the backbone. A proper linear probe baseline would cleanly separate these factors.

### Minor

- **The static dual prompts (Eqs. 4–5) add limited value without condition-nets, as shown in the ablation.** Variant 4 (both prompts, no CN) at 72.25% actually slightly underperforms Variant 2 (no prompts at all) at 72.59% on Wikipedia NC, suggesting the fixed masks alone can be counterproductive. The paper's text does not acknowledge this nuance when discussing Variant 4.

- **Per-timestamp prototypes with sparse data may be noisy.** Eq. 11 computes $\bar{\mathbf{h}}_{t_i,y}$ as the mean embedding of class $y$ examples at **specific time** $t_i$. With 30 training events across multiple classes, there could be as few as 0–2 examples per class at any given timestamp, making time-specific prototypes extremely noisy. The paper does not clarify how this is handled in practice (e.g., whether prototypes are aggregated across time), which could significantly affect results.

- **No scaling analysis with training data size.** The evaluation uses only 30 events per task, and no results are shown for 50, 100, or 200 events. Without this, it is unclear whether DyGPrompt's advantages persist beyond the extreme scarcity regime, or whether fine-tuning methods would become competitive with more data.

### Trivial
None.

## Nice-to-Haves

- An ablation removing the fixed prompts (keeping only condition-nets) would determine whether the static $\mathbf{p}^{node}$ and $\mathbf{p}^{time}$ are redundant when condition-nets already generate input-dependent prompts.

- Statistical significance testing (paired t-tests across the 500 runs) for the ablation, particularly for the small gaps between Variant 5 and full DyGprompt.

- Visualization of how condition-net outputs change over timestamps for specific nodes, to directly validate that the nets capture meaningful temporal variation.

## Removed Points

- **"Genre dataset results are near-random."** This is an observation about dataset difficulty, not a methodological flaw. Near-chance performance across all methods shows the task is hard, not that the method fails. Kept out of weaknesses.

- **"The time prompt (Eq. 5) is just a fixed mask that cannot capture temporal evolution."** This is partially addressed by the condition-nets—Eq. 9 shows the NCN generates node-conditioned time prompts. The static prompt is just the base, not the full mechanism. The claim that condition-nets are what matter is supported by the ablation. Moved to a minor note about static prompts being limited.

- **"Variant 1 matches TGAT exactly (67.00), making the ablation unclear."** This is actually informative: Variant 1 with a static node prompt achieves the same as continually-trained TGAT on 30 events, suggesting the prompt alone doesn't help. This is consistent with the finding that condition-nets drive the improvement. Not a flaw in the ablation.

- **"Missing comparison with established few-shot benchmarks."** The paper uses its own protocol. While comparing to existing protocols would help, the evaluation design is self-consistent and the protocol is clearly described. Moving to nice-to-have.

- **"Frozen-backbone + linear classifier baseline."** This was promoted to a Major weakness because it is the single most important missing baseline for interpreting the results. The specific demand remains as Major.

- **"Paper overclaims 'significantly outperforms'" when condition-net margins are marginal.** This was partially addressed: the NCN provides a large jump (72.59→81.40), but the TCN's added contribution (81.40→82.09, +0.69) is indeed small. The claim of TCN's significance is somewhat overstated in the paper text.

- **"Indexing issue in Eq. 10 suggests a bug."** Promoted to Major weakness since it creates a conceptual inconsistency with the stated motivation.

## Novel Insights

The ablation reveals an interesting pattern that the paper underemphasizes: the static dual prompts alone (Variant 4) actually slightly hurt performance compared to no prompts at all (Variant 2) on node classification, and the entire empirical advantage comes from the condition-nets. This suggests the real innovation is the input-conditioned hypernetwork mechanism, not the static prompt masks that the paper frames as an equal pillar of the contribution.

## Suggestions

- Add a frozen-backbone + linear probe baseline to cleanly isolate the contribution of the prompt mechanism from the benefit of not fine-tuning.
- Conduct a scaling analysis with varying numbers of training events (30, 50, 100, 500) to show whether DyGPrompt's advantages persist beyond extreme scarcity.
- Clarify whether the indexing in Eq. 10 ($\tilde{\mathbf{p}}_{t',v}^{\text{time}}$ for neighbors rather than $\tilde{\mathbf{p}}_{t',u}^{\text{time}}$) is intentional and provide justification, or fix it and report updated results.

## Score and Decision

Calibration anchors:
- **U3ROVRTKTa** (Prompting-based Temporal DG, avg 4.5, Reject): Similar overclaim concerns, weak ablation, questioned prompt motivation. DyGPrompt has a stronger ablation (NCN provides 9+ point gains) and clearer problem formulation.
- **4IT2pgc9v6** (OFA, avg 7.0, Accept Spotlight): Strong graph prompting framework with cross-domain generalization. DyGPrompt is more focused but methodologically sound within its scope, though missing a key baseline.
- **OKOjkFrhSs** (Prompt-guided Dynamic Network for SR, avg 3.0, Reject): Unfair baseline comparison flagged. DyGPrompt's comparison issue (favoring prompt methods in low-data regime) is similar but less severe since the paper includes prompt baselines.
- **MiMxv6ijvC** (CAReNet, avg 2.5, Reject): Unfair comparison to data-hungry models, no ablation. DyGPrompt has better ablation and comparison scope.
- **Hf54sNeeBM** (Knowledge Accumulating Contrastive Prompt, avg 4.75, Reject): Marginal improvements over SOTA flagged.

DyGPrompt presents a solid and well-motivated framework for dynamic graph prompting. The core contribution—the dual condition-nets—provides substantial empirical gains (9+ AUC points from NCN alone), and the evaluation is comprehensive. However, the paper has a meaningful structural issue: the headline comparisons against fine-tuning baselines are inflated by the ultra-low-data regime, and the missing frozen-backbone + linear probe baseline makes it impossible to fully attribute gains to the prompt design. The Eq. 10 inconsistency with stated motivation is also concerning. These are real but not fatal issues—the method clearly works and the condition-net idea is valuable. The paper is above the borderline of rejection-quality papers but below the strongest accept-quality papers.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>