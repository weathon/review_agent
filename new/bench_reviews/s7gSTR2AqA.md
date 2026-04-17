Now I have a thorough understanding of the paper. Let me synthesize the final review.

## Summary

This paper investigates whether LLMs develop human-aligned semantic categories governed by the Information Bottleneck (IB) principle. Using color categorization as a testbed, the authors conduct two studies: (1) an English color-naming experiment across 39 LLMs showing wide variation in alignment, with larger instruction-tuned models performing best; and (2) an Iterated In-Context Language Learning (IICLL) experiment with 4 models, adapting iterated learning paradigms from cognitive science, showing that LLM chains restructure random pseudo-color-naming systems toward IB-efficient solutions, with only Gemini 2.0 recapitulating the full human-like range of IB tradeoffs. The paper concludes that "human-aligned semantic categories can emerge in LLMs via the same fundamental principle that underlies semantic efficiency in humans."

## Strengths

- **Strong theoretical grounding with direct human experimental comparison**: The use of the IB framework and iterated learning—both well-established in cognitive science—provides principled, quantitative predictions rather than ad-hoc metrics. The direct replication of Xu et al. (2013)'s human experiment conditions enables meaningful human-LLM comparison, which is a genuine methodological advance.

- **Novel IICLL paradigm**: Extending Zhu & Griffiths' I-ICL to iterated in-context *language* learning is a creative and well-motivated methodological contribution. The pseudo-term and feature-labeling design is a reasonable attempt to control for mimicry, and the rotation analysis and clustering baseline (Appendix M) provide some non-triviality checks.

- **Comprehensive model sweep for Study 1**: Testing 39 models across 6 families, varying in size, instruction tuning, and modality, is thorough and provides genuinely informative results. The finding that instruction tuning matters more than raw scale (e.g., Llama 3.3 70B inst. performs poorly) and the Olmo 2 learning trajectory analysis are valuable contributions.

- **Important negative results**: The finding that many state-of-the-art LLMs struggle with basic English color naming—despite massive training data—is itself a significant and informative result that counteracts optimism about LLM grounding.

- **Compelling qualitative patterns in IICLL**: The information-plane trajectories showing chains climbing toward the IB bound and then sliding along it, parallel to human IL dynamics, are visually compelling and suggest genuine learning dynamics rather than trivial convergence.

## Weaknesses

### Major:

- **Overclaiming from observed behavioral similarity to shared "fundamental principle"**: The paper's central claim—that LLMs exhibit a "human-like inductive bias toward IB-efficiency" reflecting "the same fundamental principle"—goes well beyond what the evidence supports. The iterated learning framework's theoretical guarantee (that stationary distributions reflect learners' priors) requires Bayesian agents sharing priors and likelihoods (as the paper itself acknowledges in §2.3). LLMs are not Bayesian learners in this sense, and no validation is provided that these conditions approximately hold. The observed behavioral convergence toward IB-efficient solutions could arise from multiple mechanisms: generic smoothing of noisy mappings in-context by transformers, distributional knowledge of color-space structure, instruction-tuning biases toward coherent responses, or indeed an inductive bias toward IB-efficiency. The paper does not adequately disentangle these alternatives, yet draws the strongest possible interpretation. The discussion frames the origin of the bias as "future work," but this uncertainty should temper—not just follow—the main claims.

- **IICLL evidence rests heavily on one model, with 3 of 4 models showing a different pattern**: The paper's strongest claim—that IICLL recapitulates the full range of human IB tradeoffs—holds robustly only for Gemini 2.0. The other three tested models (Gemma 3 27B, Qwen 2.5 32B, Llama 3.3 70B) converge to low-complexity solutions, which is qualitatively different from the human pattern. The paper attributes this to differential "in-context capabilities," but this confounds multiple factors: the models differ in architecture, training data, instruction tuning, and API-style (Gemini uses controlled generation vs. log-prob scoring for open-weight models). The claim that only Gemini "truly exhibits an emergent inductive learning bias" while others possess it "to varying degrees" is not well-supported when 75% of tested models show a qualitatively different pattern. The paper also only tests models that already performed well on the English naming task (selection bias), making it difficult to assess whether IB-efficient restructuring requires prior English-like color knowledge.

- **Mimicry vs. inductive bias remains under-controlled**: The pseudo-term and feature-labeling strategy is an improvement over naïve prompting, but it is not airtight. The stimuli are the exact WCS grid coordinates—a very specific 2D structure that any model with exposure to color-coding in web data could recognize. The paper acknowledges this but does not include controls such as permuted feature dimensions, shuffled coordinate grids, or explicitly telling models whether the stimuli are colors. Without such controls, one cannot rule out that the IB-efficient behavior emerges from pattern-completion over structurally familiar data rather than a domain-general inductive bias.

### Minor:

- **CIELAB vs. sRGB discrepancy is underexplored relative to its theoretical importance**: The finding that presenting colors in CIELAB (which captures human perceptual structure) *hurts* LLM performance directly challenges the "same principle" framing, since IB-efficient human color naming is grounded in perceptual geometry. The paper briefly notes this "reveals a key difference between how LLMs represent color and how humans do" but does not fully grapple with its implications: if LLMs achieve IB-like efficiency via statistical regularities in sRGB-encoded text rather than perceptual structure, the "same principle" claim is undermined as a mechanistic claim even if it holds as a descriptive one.

- **Shepard circles experiment is underdeveloped**: The domain-generalization claim rests on a single model (Gemini), a single condition (k=4), image input only (no text), qualitative analysis only, and no IB analysis. The authors acknowledge this as "preliminary investigation," but its inclusion adds minimal evidence for domain generality while inviting the question of why text encoding failed.

- **The "models struggle with English color naming" claim is narrower than stated**: The finding applies specifically to the constrained classification task with fixed term lists and numeric/patch input. Prior work (Marjieh et al., 2024; Abdou et al., 2021) showed similar models do encode English-like color structure under different probing. The paper's contribution is to show that under this specific elicitation setup, many models produce non-English systems—but the "struggle" framing overstates what has been shown.

### Trivial:
- None worth flagging.

## Nice-to-Haves

- A control condition using permuted or unstructured stimuli in IICLL would strengthen the inductive bias vs. mimicry distinction significantly.
- Testing at least one poorly-performing model in IICLL would clarify whether IB-efficient restructuring requires prior English-like knowledge.
- Computing IB efficiency for the Shepard circles domain would make the domain-generalization claim substantive.
- Systematic prompt ablations (e.g., explicitly identifying stimuli as colors vs. abstract features) would help isolate the source of the bias.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"IB bound may be too forgiving as a normative yardstick"**: The harsh critic argues that many "clusterable" systems would appear near the IB bound without being guided by an IB principle. However, the paper includes a rotation analysis (Appendix H) showing that rotated systems significantly decrease in efficiency/alignment for Gemini, and compares against a clustering baseline (Appendix M) showing Gemini outperforms it. These controls partially address the concern. While still a methodological caveat, it is not as unfounded as the critic implies. WEAKENED and partially addressed; remaining concern is that the clustering baseline and rotation analysis are only briefly described in the main text and are less conclusive for non-Gemini models.

- **"English color naming overclaims based on narrow elicitation setup"**: The critic's concern about format sensitivity and tokenization is partially addressed by the paper's own multimodal and CIELAB analyses. However, the paper does not systematically test alternative prompting or unconstrained generation, so the concern that the "struggle" claim is setup-specific remains partially valid—kept as a minor weakness.

- **"Confounding in-context capability with IB bias"**: This is partially a restatement of the claim that only Gemini shows the full pattern. Kept as a major weakness, but the framing that the paper doesn't acknowledge this at all is softened—the paper does briefly discuss "strong in-context capabilities" as a possible explanation, even if it doesn't go far enough.

- **"Prompt sensitivity and implementation details"**: Demands for complete prompt specifications and sensitivity analysis are reasonable but go beyond what is standard for this type of work. The paper provides prompts in the appendix. MOVED to Nice-to-Haves.

- **"Missing related works"**: Removed per instructions; I cannot verify whether specific works exist or were omitted.

- **"Reproducibility concerns about models/API access"**: Removed per hard rules; if the paper cites these models, they are treated as existing and available.

- **"Statistical significance and variance across chains"**: Requesting confidence intervals for large-scale experimental runs is reasonable for establishing robustness, but single-run evaluation without statistical testing is common in this area. WEAKENED and moved to Nice-to-Haves.

- **"Ecological validity of color naming task for LLMs"**: While there's a legitimate question about how natural these tasks are for LLMs, this is inherent to any cognitive-science-inspired evaluation of LLMs and not specific to this paper's methodology. The paper itself acknowledges the CIELAB discrepancy. WEAKENED.

## Novel Insights

The relationship between IB-efficiency in LLMs and in humans may be more appropriately framed as *convergent* rather than *shared*: two systems with very different architectures and learning mechanisms (embodied perception + cultural transmission vs. text-based pattern completion + in-context learning) may arrive at similar *outcomes* on the IB frontier, but via fundamentally different *processes*. The CIELAB finding actually supports this distinction—LLMs' efficiency is grounded in sRGB statistical regularities, not perceptual geometry. This reframing preserves the interest of the finding while avoiding mechanistic overclaim. The IICLL paradigm itself is the paper's most durable contribution; it could become a standard tool for eliciting inductive biases in LLMs across domains with rich human data.

## Suggestions

- Reframe the main claims from "the same fundamental principle" to "convergent IB-efficient outcomes via different mechanisms"—this preserves the significance of the result while being honest about what IICLL can and cannot establish about causality.
- Add a control IICLL condition with permuted input features to test whether IB-efficient restructuring requires structurally familiar stimulus spaces.
- Include at least one model that performed poorly on the English naming task in IICLL to assess whether prior color knowledge is necessary for IB-efficient convergence.
- Move the rotation analysis and clustering baseline comparison into the main text—they are critical for the paper's core claim and currently relegated to appendices.

## Score and Decision

Comparing against calibration papers:

- "Turning LLMs into cognitive models" (scores: 8,8,8,8): Strong methodology, clear claims well-supported by evidence, focused scope. Our paper is less rigorous in its causal claims and more dependent on a single model's behavior.

- "Does Spatial Cognition Emerge in Frontier Models?" (scores: 6,8,8,5): Comprehensive benchmark with interesting negative results, but overclaims about "cognitive" capabilities. Our paper shares a similar pattern—strong empirical work with some overclaiming.

- "When LLMs Play the Telephone Game" (scores: 5,8,8,3): Novel iterated-interaction paradigm, somewhat thin on controls. Our paper is more theoretically grounded but also more dependent on single-model results.

- "Visual LLMs Exhibit Human-Level Cognitive Flexibility" (scores: 5,3,3): Overclaims about "human-level" cognitive abilities based on limited evidence. Our paper has stronger controls and theoretical grounding but still overclaims.

The paper's theoretical framework and IICLL paradigm are genuinely novel and valuable. The empirical work is extensive and carefully executed. However, the overclaiming about "the same fundamental principle" and the reliance on a single model (Gemini) for the strongest results, combined with underexplored alternative explanations, are significant weaknesses that inflate the paper's perceived contribution. At the same time, the work is clearly above average—it provides real empirical insights and a useful methodological contribution.

I would place this paper in the 5.5-6 range: solid contribution with important weaknesses that should be addressed in revision.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>