## Summary

This paper empirically investigates why State Space Models (SSMs) underperform Transformers on associative recall and copying tasks, arguing that the gap is primarily driven by optimization instability (extreme learning rate sensitivity) rather than fundamental expressivity limitations. Through extensive hyperparameter sweeps (~3,000 runs, ~20,000 GPU hours), the authors show that SSMs require a much narrower learning rate window than Transformers, that prior benchmarks confounded expressivity with suboptimal tuning, and that the two architectures exhibit opposite scaling behaviors (SSMs favor width, Transformers favor depth). Targeted ablations identify 1D convolution as the critical enabler for single-layer recall in both Mamba and augmented Transformers.

## Strengths

- **Corrective empirical contribution with rigorous scale:** The paper convincingly demonstrates that prior MQAR evaluations (Arora et al., 2023) were confounded by coarse learning rate grids. Figure 1 and Figure 2 together provide strong evidence: the dashed vertical lines from Arora et al.'s grid clearly miss the narrow success windows of Mamba and Hyena, and the finer grid (solid orange) recovers performance. This is a substantive corrective finding backed by 3,000+ runs with 5 seeds, directly challenging how the community evaluates SSM capabilities.

- **Clean ablation identifying convolution as the key architectural driver:** Table 2 is a high-value result. Showing that `Mamba - conv1d` drops to 2% while `Attention + Conv` reaches 99% isolates local mixing as the critical component for single-layer recall, reframing the debate from "Attention vs. SSM recurrence" to "local vs. global mixing." This is the kind of mechanistic insight that shifts how researchers think about architectural design.

- **Contrasting scaling behaviors clearly demonstrated:** Figure 4 effectively shows that parameter count alone is insufficient—it is the scaling axis (width vs. depth) that matters, with SSMs benefiting from width and Transformers from depth. Table 1 provides a concrete parameter-matched demonstration (deeper Mamba fails, wider Mamba succeeds at the same parameter count), making the point unambiguous.

## Weaknesses

### Major:

- **Overstated central claim creates internal tension with own results:** The abstract states "Transformers differ from SSMs not in terms of expressive power but mainly because of their optimization dynamics." However, the paper's own Section 4 shows that 1-layer Transformers fundamentally fail MQAR regardless of tuning (Figure 3), while 1-layer SSMs succeed with sufficient width. This is a genuine expressivity difference in the shallow regime—not an optimization issue. The paper acknowledges this factually but the abstract's framing elides it. The more accurate claim, supported by the paper's own evidence, is: optimization instability is the primary barrier for *deep* SSMs, while architectural composition depth is the barrier for *shallow* Transformers. These are distinct phenomena, and conflating them weakens the paper's narrative coherence.

- **Claim of "induction head formation" in 1-layer Transformers lacks mechanistic support:** Section 6 interprets a loss bump in 1-layer Transformer training as "reminiscent of the induction head phenomenon" and hypothesizes an "attempt" to form them. However, induction heads as defined by Olsson et al. (2022) mechanically require two layers (one to copy previous-token information, one to attend based on that copy). A 1-layer model cannot implement this circuit. Observing a loss bump without any mechanistic evidence (e.g., attention pattern visualization showing the copy-then-attend structure) is insufficient to invoke the induction head framework. The bump could reflect any optimization transient. The paper uses hedging language ("reminiscent," "hypothesize"), but the claim still risks overinterpretation that could mislead readers familiar with the specific mechanistic definition.

### Minor:

- **Initial comparisons conflate two architectural differences:** The primary comparisons in Sections 3–5 pit standard Mamba (which includes a 1D convolution) against standard Attention (which does not). The convolution's critical role is only revealed in Section 7. This sequencing means early readers may attribute performance differences entirely to the recurrence mechanism when convolution is a confounding variable. A brief note in Section 3 flagging this architectural asymmetry would improve interpretability before the full ablation.

- **No quantification of learning rate window narrowness:** The core claim about narrow LR windows is presented qualitatively through visual inspection of figures. A simple metric—e.g., the ratio of successful learning rates to the total search range, or the width of the success interval in log-space—would make the comparison between architectures rigorous and reproducible rather than impressionistic.

- **Practical cost of SSM tuning not discussed:** The paper demonstrates that SSMs require much finer learning rate searches to find the narrow success window, but does not discuss what this implies for practical training budgets. If SSMs need 10× the hyperparameter search cost to achieve comparable performance, this partially offsets their per-step training efficiency advantage. This is directly relevant to the paper's stated goal of guiding practitioners.

### Trivial:

- The "relative max-min errors" metric used in several figures is less standard than accuracy percentage and could benefit from a clearer definition in the main text (it is explained in the appendix).

## Nice-to-Haves

- **Gradient norm / loss landscape analysis:** The paper hypothesizes that vanishing/exploding gradients drive the optimization brittleness, but never measures gradient statistics or loss curvature. Even a simple plot of gradient norms across training steps for SSMs vs. Transformers would substantiate the mechanistic explanation.

- **Optimizer ablation:** All experiments use AdamW. Testing whether the instability persists with alternative optimizers (e.g., Lion, different gradient clipping strategies) would clarify whether this is an architecturally inherent property or an optimizer-architecture interaction.

- **Downstream language modeling validation:** The paper acknowledges this limitation. A small-scale LM experiment (e.g., WikiText-103 or a C4 subset) testing whether the LR sensitivity observed on MQAR transfers to perplexity would significantly strengthen the practical relevance.

- **Investigation of initialization schemes:** The paper references Trockman et al. (2024) on mimetic initialization but does not test whether better initialization narrows the LR window gap. This is a practical intervention that could directly address the identified problem.

- **Hybrid architecture exploration:** Given that convolutions fix 1-layer Attention and that DeltaNet shows improved stability, testing whether hybrid SSM-Attention models inherit the best of both properties is the natural next step the paper hints at but does not pursue.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Instability appears inherent rather than a training artifact" (from harsh critic's transferable weaknesses).** This actually *supports* the paper's thesis rather than weakening it. The paper's central argument is precisely that SSMs have inherent optimization instability—not that it's a mere training artifact. Framing this as a weakness misunderstands the paper's position.

- **Weakness: "Incomplete treatment of state capacity limitations" citing "Stuffed Mamba."** The paper explicitly confirms that hidden state size matters (width scaling results) and does not claim state capacity is irrelevant—its argument is that optimization is the *primary* confounder in prior evaluations, not the *only* factor. Demanding a full treatment of state capacity is scope creep beyond the paper's stated contribution.

- **Weakness: "Limited discussion of architectural trade-offs between efficiency and learnability."** This asks the paper to address a broader design philosophy question that is outside its scope. The paper documents and characterizes the optimization gap; proposing architectural solutions to resolve the efficiency-learnability trade-off is a different research direction.

- **Weakness: "Proper baseline replication with original hyperparameters—the paper doesn't show what LR Arora et al. used."** This is factually wrong. Figure 1 explicitly shows Arora et al.'s learning rate grid as dashed vertical lines, making it visually clear that their grid missed the optimal region for SSMs.

- **Weakness: "Limited exploration of whether architectural modifications are sufficient long-term solutions" citing VMamba vision results.** This imports concerns from a different domain (vision) and a different paper. The current paper's scope is synthetic sequence benchmarks; extrapolating to vision is outside scope.

- **Strength: "Paper is well-written" and "topic is important."** These are generic strengths that apply to many papers at ICLR and are removed per the rules.

- **Nitpick: Grammar issues ("Attention exhibit" → "Attention exhibits").** Removed as a formatting/style nitpick per rules.

## Novel Insights

The most striking insight from the reviews and the paper itself is the **reframing of the SSM vs. Transformer debate from expressivity to learnability—and the way the paper's own results complicate this reframing.** The paper's strongest finding (Table 2) is that removing the 1D convolution from Mamba makes it perform identically to 1-layer Attention on MQAR, while adding convolution to Attention makes it perform identically to Mamba. This effectively reduces the "SSM advantage" in the single-layer regime to the presence of local mixing, not the recurrence mechanism itself. This suggests the community's expressivity comparisons may have been comparing not "attention vs. state-space recurrence" but "global-only mixing vs. global+local mixing"—a much less fundamental architectural distinction, and one that is straightforwardly addressable. Meanwhile, the persistent narrow LR window even in S6+MLP (which solves MQAR at 98%) indicates that the optimization brittleness is genuinely tied to the recurrent computation structure, not to auxiliary components. This creates a clean separation: convolutions explain the expressivity gap, while the recurrent structure explains the learnability gap.

## Suggestions

- **Rewrite the abstract and conclusion to distinguish the two distinct findings:** (1) For *deep* models, optimization instability is the primary differentiator. (2) For *shallow* models, architectural composition (depth for Transformers, convolution for SSMs) is the primary differentiator. Conflating these weakens the paper's otherwise strong empirical case.

- **Add a quantification of LR window width** (e.g., fraction of log-LR range yielding >90% accuracy) as a single number per architecture in Figure 1 or its caption, to make the "narrowness" claim precise and citable.

- **Soften the induction head claim in Section 6** by either (a) adding attention pattern visualizations before/after the loss bump, or (b) reframing as "an optimization transient whose mechanism we leave to future work" rather than invoking the induction head framework without the requisite mechanistic evidence.

- **Add a sentence in Section 3** noting that Mamba includes a causal convolution absent from the standard Attention baseline, flagging this as a known architectural asymmetry that is dissected in Section 7.