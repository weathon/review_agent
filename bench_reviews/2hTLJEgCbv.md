## Summary
This paper presents an empirical study of how encoder/decoder architecture choices in a standard VAE affect optimization behavior, latent collapse, and reconstruction on MNIST. The main reported takeaway is that shallow dense encoders tend to work better than more complex encoders in this setup, while decoding benefits more from convolutional structure, especially with deeper convolutional decoders.

## Strengths
- The paper isolates a narrow question—encoder/decoder architectural asymmetry within an otherwise standard VAE—and explores it via a reasonably systematic grid over encoder type, decoder type, depth, and latent size. That focused ablation structure is more informative than a single preferred architecture.
- The paper explicitly separates reconstruction and KL-related behavior in the analysis (Figures 1–3), which is appropriate for VAEs and helps expose that some configurations reconstruct reasonably while still exhibiting latent collapse or near-collapse.
- One concrete empirical pattern does emerge from the sweep: among the better-performing configurations in this MNIST setting, shallow dense encoders recur more often, while convolutional decoders with multiple blocks appear advantageous on the decoding side. Even if limited in scope, this encoder/decoder asymmetry is the paper’s most useful practical observation.
- The paper deliberately studies simple building blocks rather than mixing in more advanced priors/objectives, which makes the architectural comparisons easier to interpret than studies where architecture changes are entangled with loss redesign.

## Weaknesses

###: Fatal

### Major:
- **The empirical scope is too limited to support the paper’s broad architectural conclusions.** All experiments are on MNIST (“All experiments are be conducted on the MNIST dataset”), which is too simple to justify general guidance such as “small dense networks are more effective for encoding” and “decoding benefits from architectures with structural processing capabilities.” On a dataset with such strong low-level regularity and low semantic complexity, observed trends may not transfer to more realistic image distributions.
- **The evaluation does not adequately support claims about generative quality, representation quality, or compression.** The paper mainly uses ELBO components, qualitative reconstruction comments, and 2D PCA plots of latent codes. But the abstract and conclusion make claims about “generative quality,” “representation quality,” and “compressive capacities.” The current pipeline does not directly measure those notions. In particular, PCA visualizations on MNIST are weak evidence for latent quality, and there are no quantitative sample-quality metrics or more direct representation diagnostics.
- **Architectural claims are confounded by uncontrolled model capacity and training setup.** Section 3 describes CNN and dense variants only at a high level and does not report parameter counts, matched-capacity comparisons, or other controls that would let one attribute differences to architecture type rather than raw capacity/depth/optimization effects. Since the conclusions hinge on whether dense vs. convolutional encoders/decoders are intrinsically preferable, this missing control matters substantially.
- **The “top 25% / top 50%” analysis introduces an unclear and potentially biased selection procedure.** The paper repeatedly analyzes “top 25%” and “top 50%” models, but it does not clearly define the ranking criterion in the text. Because the main architectural conclusions are partly drawn from these filtered subsets, the lack of a precise and justified selection rule weakens the evidential chain.
- **Several headline findings are weaker than the paper presents them.** For example, the paper emphasizes that “models with non-zero Kullback-Leibler Divergence (KLD) loss outperform collapsed latent space models.” That is directionally true in practice, but it is not a strong scientific insight by itself; avoiding posterior collapse is largely a prerequisite for the latent variables to carry information at all. As stated, this reads more as confirmation of expected VAE behavior than as a novel contribution.

### Minor
- **Posterior collapse is discussed somewhat informally and without stronger diagnostics.** Section 4.1 equates many runs with “collapsed latent spaces” and describes this as latent distributions becoming identical to a multivariate normal distribution, but the paper does not define a threshold or provide supporting diagnostics such as active units, mutual information proxies, or per-dimension KL statistics. This makes the collapse analysis less rigorous than it could be.
- **The paper’s notion of “compression levels” is underspecified.** The text says latent spaces of varying compression are studied, and Figure 4 refers to “compression size,” but the paper does not clearly formalize compression in relation to input dimensionality or provide a principled rate–distortion style analysis. As a result, the compression conclusions remain qualitative.
- **The significance and novelty are modest for an ICLR paper.** The observed trends largely align with known inductive biases: convolution helps image decoding via spatial structure, and smaller encoders may reduce overfitting or mismatch in simple settings. The paper is a useful exploratory sweep, but it stops short of a deeper mechanistic explanation or broader validation.

### Trivial

## Nice-to-Haves
- Add experiments on at least one more challenging image dataset to test whether the encoder/decoder asymmetry persists beyond MNIST.
- Report matched-parameter or matched-FLOP comparisons across architecture families.
- Include direct generative evaluations (sample grids plus quantitative metrics where appropriate) and more formal latent-space diagnostics.
- Run multiple seeds and report variance, especially if the architectural performance gaps are not large.
- Clarify exactly how models are ranked when defining the “top 25%” and “top 50%” subsets.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper misrepresents the literature by claiming architecture is underexplored.”** This is too dependent on external literature adjudication beyond what can be verified from the submission alone. The paper does cite prior architecture-focused work (e.g., NVAE, DGSN), so while the novelty framing may be somewhat overstated, this criticism was too strong as written.
- **Missing comparison to specific VAE variants such as NVAE or β-VAE as a core flaw.** The paper is framed as an internal architectural ablation within a standard VAE rather than a state-of-the-art method paper. Competitive baselines would help contextualization, but their absence is not by itself a decisive flaw for the stated scope.
- **Complaints purely about omitted optimizer / learning-rate / batch-size details.** The paper is indeed sparse on implementation details, but this falls under reproducibility nitpicks unless tied to a substantive claim. The more important issue is not the missing hyperparameters per se, but the lack of controls over KL weighting/capacity when drawing architectural conclusions.
- **Criticism that the paper should include transformers or modern architectures.** This is scope creep. The paper studies simple dense vs. convolutional VAEs; asking for transformer baselines is not necessary to evaluate whether it answered its chosen question.

## Novel Insights
The most interesting synthesis across the reviews and the paper is that the useful contribution here is not the broad claim that the paper has “solved” VAE architecture design, but the narrower empirical asymmetry it exposes: in a plain-VAE regime, encoder simplicity may be beneficial while decoder structure matters more. That is a potentially practical design heuristic. However, the current study does not yet distinguish whether this asymmetry is due to information bottleneck effects, optimization stability, parameter-count mismatch, or dataset simplicity. In other words, the paper’s strongest spark is a plausible architectural asymmetry, but the present evidence does not yet pin down its cause or generality.

## Suggestions
- Reframe the contribution more modestly as an MNIST-based exploratory study of architectural asymmetry in plain VAEs, rather than a general statement about VAE design.
- Add at least one nontrivial dataset and verify whether the “simple encoder / structured decoder” pattern still holds.
- Control for capacity explicitly: report parameter counts and include matched-capacity comparisons between MLP and CNN variants.
- Replace or supplement PCA-based latent analysis with more direct latent-usage diagnostics and clearer collapse criteria.
- Clarify the model-selection/ranking procedure for the “top 25%” and “top 50%” analyses, or avoid filtered-subset conclusions if the ranking criterion is not principled.
- Tone down the claim around non-zero KL being beneficial, or instead analyze it more rigorously via a controlled sweep over KL weighting / rate-distortion tradeoffs.
- Include representative unconditional samples, reconstructions, and latent traversals/interpolations to support the claims about generative and representational quality.
- Strengthen the discussion of novelty and significance: the current work is best positioned as an empirical heuristic study, not as a fundamentally new understanding of VAE behavior.