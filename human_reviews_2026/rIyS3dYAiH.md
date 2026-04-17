# Hyperbolic Music Representations

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Music is inherently hierarchical due to keys and variations of note sequences. These dependencies need to be captured by the metric of choice to learn an appropriate representation space. Although Euclidean geometry is frequently used to embed music, it is clearly unable to capture the hierarchical structures. In this paper, we propose to learn hyperbolic representation spaces for music using Variational Autoencoders with a Poincaré ball as a natural alternative to Euclidean geometry. The resulting latent space is interpretable, reflects keys and musical richness, and allows for meaningful interpolations due to a novel generalization of Spherical Linear Interpolation to Riemannian manifolds. Empirically, we compare our contribution to standard Euclidean representations and observe that the latter fall short in terms of interpretation and reconstruction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes learning hyperbolic representation spaces for music using a VAE with a Poncare ball as the latent space, capturing the hierarchical nature of musical structures (e.g., keys, chords, and variations) more effectively than Euclidean geometry.

It introduces a novel Riemannian Spherical Linear Interpolation (R-SLERP) method, enabling smooth and meaningful interpolations between musical pieces by respecting hierarchical distances and curvature in hyperbolic space.

Experiments on MIDI and raw audio datasets show that the proposed P-VAE achieves better interpretability and reconstruction performance than standard Euclidean VAEs.

### Strengths
The use of hyperbolic geometry allows the model to naturally capture the hierarchical structure of musical concepts (such as notes -> chords -> progressions -> full pieces), which is difficult for Euclidean embeddings to represent effectively.

By introducing R-SLERP in the hyperbolic latent space, the model produces smoother and more musically coherent transitions between pieces, leading to more interpretable and meaningful latent representations.

### Weaknesses
Hyperbolic latent spaces and VAEs have been explored in prior work (e.g., Nagano et al., 2019; Mathieu et al., 2019). The paper’s contribution mainly lies in applying it to music and introducing R-SLERP, so the core idea of hyperbolic representation is not entirely new.

While the paper argues that music is hierarchical, the intuition behind why hyperbolic embeddings are particularly suitable for musical structures (beyond theoretical tree-like properties) could be explained more clearly with musical examples or perceptual insights.

The experiments focus on VAEs with hyperbolic vs. Euclidean latent spaces, but the paper does not compare performance with modern large-scale transformer-based music models trained on massive datasets, which are state-of-the-art in generative quality.

It is not shown how the hyperbolic embeddings improve performance on downstream tasks such as style transfer, music recommendation, or generation diversity, limiting the practical impact beyond reconstruction and interpretability.

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper suggested that music’s structure is hierarchical and therefore better modelled in negatively curved latent spaces than in Euclidean ones. Therefore, it defines a VAE whose latent space is a Poincare ball; sampling uses a Wrapped Normal with log/exp maps and parallel transport, and the decoder consumes \log_0(z). The loss is an ELBO with a Monte-Carlo KL. A procedure called “R-SLERP” interpolates by SLERPing unit directions in T_0 and scaling the radius linearly before mapping back with \exp_0. Experiments on POP909, MIDICAPS (symbolic) and MAESTRO (audio) show statistically reconstruction gains over a Euclidean VAE and a latent organization consistent with the circle of fifths.

### Strengths
The paper offers a refreshing perspective by attempting to embed musical hierarchies, such as tonality, key relationships, and structural depth within a hyperbolic latent space, where the geometry itself mirrors how music naturally branches and resolves. In essence, the authors use the Poincare ball to represent tonal distance and harmonic tension in a way that feels musically intuitive: the deeper or more nested a tonal relationship, the farther it sits from the origin, much like how modulation in tonal space creates perceptual “depth.” Technically, the model extends the Variational Autoencoder (VAE) into this non-Euclidean space, defining the exponential and logarithmic maps correctly and applying the hyperbolic reparameterization trick to learn latent structures that curve outward.

### Weaknesses
a) A key methodological ambiguity arises from the decoding step described in Section 4.2.  After sampling a manifold point z using the Wrapped Normal distribution (Eq. 7), the paper states that “the result of \log_{0}(z) is then decoded in standard fashion to produce \hat{x}”. This means the decoder operates on \log_{0}(z), a vector in the tangent space at the origin T_{0}\mathcal{M}, rather than on the manifold point z itself. Because T_{0}\mathcal{M} is Euclidean, this choice effectively flattens the curvature for the likelihood p_{\theta}(x\mid z), making the decoder independent of the negative curvature that motivates the model.  But, the paper provides no justification for why this operation preserves the claimed benefits of hyperbolic geometry.  The authors should clarify whether decoding from \log_{0}(z) is theoretically equivalent to decoding from z or \log_{\mu}(z), and, if so, under what invariances or assumptions this equivalence holds.  If not, empirical evidence comparing these alternatives is needed to confirm that curvature has a meaningful influence on the generation process, rather than being restricted to the prior–posterior space.  In its current form, the use of \log_{0}(z) risks neutralising the negative-curvature effects in the decoder, undermining the central claim that the proposed Poincare-VAE benefits from a non-Euclidean latent geometry.

---

b) In Section 4.3, the paper introduces R-SLERP as an interpolation method that it claims is “guaranteed to stay within the hierarchy levels of its start and end point,” and that the distance to the origin “changes linearly with t, given that such behaviour holds for v_{\text{int}}(t)”. However, what this  section actually provides is a constructive recipe, that is, it maps the two endpoints z_1, z_2 to the origin’s tangent space with \log_0, performing SLERP on the unit directions, scaling the radius linearly \alpha(t)=(1-t)\|v_1\|+t\|v_2\|, and mapping back using \exp_0, but without a formal derivation or set of sufficient conditions to justify the “guarantee.”

Technically, this construction ensures only that the radius from the origin changes linearly if the tangent-space behaviour meets certain assumptions, because Eq. 4 earlier shows d(0,\exp_0(v))=\|v\|. However, the paper never proves that the resulting path always remains between the two endpoint radii or that it preserves the intended “hierarchy level” for all admissible points in the Poincare ball. In simple terms, the authors are asserting that the curve connecting two musical representations never overshoots or collapses outside their hierarchy range, but they do not actually prove it. They only describe the recipe that should, in theory, make it happen.

This is critical for me because the paper’s entire argument that hyperbolic interpolation captures smooth hierarchical transitions in music depends on this behaviour being guaranteed by geometry, not just observed empirically. To be convincing, the authors need to (a) provide a short lemma showing that d(0,z_{\text{int}}(t))=\alpha(t)\in[d(0,z_1),d(0,z_2)] for all t\in[0,1], or (b) explicitly state that the claim is heuristic. An empirical observation rather than a mathematical fact. Without that, the so-called “hierarchy-preserving” property remains an intuitive but unverified idea, leaving the interpolation step, and thus one of the paper’s main technical contributions, on uncertain theoretical ground. 

---
c) In Section 5, it compares only two models: the proposed P-VAE with a Poincare ball latent space and an E-VAE with Euclidean latent space. In the same subsection (“Models”), the authors explicitly state that prior music-VAE work on architectural optimisation and disentanglement (Roberts et al., 2018; Brunner et al., 2018; Li & Mandt, 2018; Yang et al., 2019; Wang et al., 2020b) is “orthogonal” to their goal of comparing hyperbolic vs Euclidean geometry and “may further improve results,” and is therefore not evaluated. They also implement a “standard Euclidean VAE” and note that it does not differ significantly from their E-VAE because the two are mathematically equivalent. As a result, all reported gains are of the form “P-VAE > this one Euclidean baseline,” and the experiments do not isolate curvature as the unique cause of improvement. To make the curvature claim testable, additional Euclidean and non-Euclidean baselines would be required. 

That is, all reported improvements hinge on outperforming one basic flat-space model. The omission of stronger Euclidean or alternative-geometry baselines makes it impossible to determine whether the observed gains genuinely arise from negative curvature or merely from unrelated architectural or training differences. For me, this restricted setup cannot substantiate the paper’s central claim that hyperbolic geometry itself improves musical representations. To isolate curvature as the causal factor, the study should expand its baselines to include at least some of below:
(i) advanced Euclidean controls such as β-VAEs, disentangled or hierarchical VAEs, or Euclidean VAEs with regularized Riemannian metrics (e.g., Chen et al., 2022);
(ii) alternative non-Euclidean geometries like spherical or mixed-curvature manifolds (Skopek et al., 2020); and
(iii) explicit ablations varying the curvature constant c while keeping all other parameters fixed.

### Questions
- Section 4.3 Header reads “Riemannnian SLERP” (extra ‘n’).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes learning hyperbolic representation spaces for music using variational autoencoders with a Poincaré ball latent space. The authors motivate this by claiming that musical structures are hierarchical and thus, better suited to hyperbolic geometry than to Euclidean space. They also introduce a Riemannian generalization of spherical linear interpolation (R-SLERP) to generate interpolations that vary linearly in hierarchy level along the path. Experiments on POP909, MIDICAPS, and MAESTRO show slightly improved reconstruction metrics over a Euclidean baseline and visually coherent interpolations.

### Strengths
- The paper provides a coherent geometric narrative linking hierarchical structures to negative curvature, mapping this intuition to music in an interpretable manner.

- Writing and visuals are clear. The figures and geometric explanations make the paper approachable for readers less familiar with hyperbolic geometry.

- R-SLERP addresses a real issue with geodesic interpolation. The observation that geodesics curve toward the origin (causing “over-simplified” midpoints) is valid. The proposed interpolation is a reasonable workaround. The qualitative results provided, though limited, support this claim.

- The experimental setup covers both symbolic and raw waveform music datasets, and the reported reconstruction improvements are small but statistically significant.

### Weaknesses
- Limited technical contribution. The Poincaré VAE and wrapped-normal sampling closely follow prior work (Nagano et al., 2019). R-SLERP is essentially a composition of log/exp maps and SLERP, with linear radial scaling. While it helps to avoid interpolation paths curving toward the origin, it does not constitute a clear theoretical advance or provide formal guarantees.

- Speculative motivation. The claim that music is hierarchical rests on the combinatorial growth of note sequences, which applies to any sequential data. The dependency that “each note constrains the next” is not unique to music. The connection between musical structure and tree-like geometry, or negative curvature, is not formally or empirically substantiated.

- Marginal empirical gains. The claim that hyperbolic representations are superior is not convincingly supported. Reconstruction improvements (Table 1) are small, yet the authors describe them as “impressively” demonstrating the superiority of hyperbolic latent spaces. Furthermore, the hyperbolic VAE increases runtime by 7–40%, and the latent dimensionality is identical to Euclidean baselines (128d for POP909, MAESTRO and 512d for MIDICAPS), undermining the expected efficiency benefits of hyperbolic representations.

- Conceptual tension around interpolation. The authors argue that hyperbolic geometry is a natural fit for musical hierarchy, yet their principal interpolation justification is that hyperbolic geodesics are a poor choice (they curve inward, producing sparse midpoints). The authors must explain why choose hyperbolic geometry in the first place. 

- Insufficient experimental validation. R-SLERP is the main technical novelty, yet there is only 1 qualitative experiment presented. In this experiment, the results seem to agree with the hypothesis that hyperbolic geodesic interpolation lead to sparse midpoints. However, no comparison was made to Euclidean interpolation (Euclidean VAE) and there's no quantitative results to conclude that R-SLERP produces better or more meaningful interpolations.

- Interpretability claims: The UMAP plot in Fig. 4 is interesting but there's no comparison against a Euclidean baseline to assess whether the angular separation and arrangement of keys is due to hyperbolic geometry. The same follows for the claim that radius encodes richness. There is a single example of this in Fig. 5c. Consequently, it's unclear whether the proposed approach is also better in terms of  interpretability. Quantitative validation (e.g., classifier for key from angular coordinate, correlation between radius and richness metrics) and comparison against Euclidean baselines is needed.

### Questions
- Can the authors provide a clearer justification for why hyperbolic geometry is the appropriate choice for modeling musical hierarchies, especially given that geodesics in hyperbolic space behave poorly for interpolation?

- Can the relationship between radial distance in the latent space and objective measures of musical richness (e.g., chord size, number of unique notes) be quantified?

- Did the authors experiment with lower-dimensional latent spaces to test whether hyperbolic representations preserve reconstruction quality better than Euclidean representations at smaller dimensionalities?

- Could the authors provide a UMAP (or other dimensionality reduction) visualization of the Euclidean VAE latent space for comparison with the hyperbolic latent space?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper suggests that there is a fundamental incongruity with the nature of music and the practice of embedding music in a Euclidean space.  The authors suggest that a Riemannian space is more appropriate, and then show how to adopt the widely-used VAE technique of embedding data into R^n to study this phenomenon using Poincare balls.  They show that they do better on reconstruction scores than traditionally-trained VAEs, and that they are able to meaningfully interpolate in the induced space.

### Strengths
This paper applies a SOTA realization from the literature - that tree-like structures are better embedded in Riemanniam than Euclidean spaces to the oft-neglected topic of music.  The UMAP projection very nicely demonstrates that there is real mathematical backing for the “circle of 5ths” hypothesized in music theory.  The SLERP VAE method is novel and could be impactful in other areas as well.

### Weaknesses
This paper is not particularly high-impact in terms of improving content in the application domain - there are much, much better symbolic transformers than the MusicVAE.  While I recognize that science needs basic experiments, I’m not convinced by these - the interpolation examples give didn’t look particularly meaningful (I honestly didn’t know what I was supposed to be looking for).  There could have been much more statistical rigor.

### Questions
1. Could this method be applied to more than just short snippets of music (which was an inherent limitation of VAE’s) in the past?
2. Could you explain what I should be looking for in the interpolation figures?
3. Are reconstruction losses between the riemannian and euclidean domain directly comparable like you seem to claim?
4. Have you been able to use VAE output to do any musical property classification? 
5. What would you say could be taken from your work and applied to music \textbf{generation} (not theory), if anything?

### Soundness
4

### Presentation
3

### Contribution
2
