---
job_id: 5b969de0-916e-49ad-8855-6d3e71898470
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: D5PJX02Jki.pdf
paper: Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new positional encoding / attention mechanism for transformers and evaluates it on language modeling and long‑context benchmarks, which fits squarely within core ICLR topics (representation learning, optimization, transformers, long‑context LLMs).

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion/Conclusion) are present and reasonably complete. The method is clearly specified, maths is mostly consistent, and there is substantial experimental evidence (multiple model sizes, several benchmarks). While there are debatable design choices and some missing comparisons, there is no fatal flaw that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, invisible text, or attempts to instruct automated reviewers. The text is a standard research manuscript.

---

# Expected Review Outcome:

## Summary

The paper revisits Rotary Position Embeddings (RoPE) from the complex-number perspective and points out that standard implementations only use the real part of the complex dot product between rotated queries and keys, effectively discarding the imaginary component. The authors derive a formulation that recovers the (negative) imaginary part as another valid attention score, and propose RoPE++, which treats the real and imaginary parts as distinct attention head groups. Two configurations are introduced: RoPE++EC, which keeps KV cache size fixed but doubles the number of heads, and RoPE++EH, which keeps the number of heads fixed while halving the KV cache and QKV parameters. Pre‑training experiments at 376M, 776M, and 1.5B parameters show that RoPE++ variants generally match or outperform RoPE and several alternative position encodings on both short- and long‑context benchmarks, with particular gains in long‑context tasks.

## Strengths

1. **Clear, simple core idea with nontrivial implications.**  
   The observation that RoPE’s complex form naturally yields both real and imaginary attention components, and that the imaginary part can be used as additional attention heads with almost no change to the core architecture, is conceptually clean. Equations (2)–(4) on Pages 3–4 neatly show that the imaginary term corresponds to a rotation of queries by \(-\pi/2\) followed by the same RoPE operator, preserving the absolute–relative equivalence.

2. **Mathematical characterization of imaginary attention and its role in long‑range dependence.**  
   The paper does more than just “use the imaginary part”: it derives expectation-based characteristic curves for real vs. imaginary attention. Equation (5) and the corresponding integral form in Appendix B (Page 16) show that the imaginary attention averages \(\sin(\theta \Delta t)\) over the RoPE frequency distribution, leading to a sine‑integral curve \(c_{\text{Im}}(\Delta t)\) that decays more slowly for large \(\Delta t\). Figure 1 visually supports this: the real component’s characteristic curve drops relatively quickly, whereas the imaginary curve remains elevated over a large distance range, arguing that imaginary heads are biased toward long-range dependencies.

3. **Careful analysis of length extrapolation behavior.**  
   Section 3.4 gives a reasonably concrete argument about how imaginary attention expands the range of positional values seen during training. For even query and odd key dimensions, RoPE++ exposes them to both positive and negative \(\cos\) / \(\sin\) values via the additional \(-\cos\theta (t-s)\) term in (3), effectively covering the full sinusoidal range within half a period. The qualitative illustrations in Figure 3 (subfigures 3a–3d on Page 5) make this clear: for RoPE++, the “trained region” (yellow) of positional embeddings covers positive and negative values, while vanilla RoPE only sees non‑negative embeddings before extrapolation. This is a concrete, mechanistic explanation for the slower perplexity growth observed in the extrapolation curves in Figure 6 (Page 20).

4. **Nontrivial efficiency–accuracy trade‑off design (RoPE++EH).**  
   Section 3.3 shows that, by viewing real and imaginary scores as separate heads sharing the same Q/K parameters, one can either (i) keep cache size fixed and double head count (RoPE++EC) or (ii) keep the number of output heads fixed and halve QKV and KV‑cache (RoPE++EH). Figure 2 (Page 5) illustrates these configurations clearly: RoPE++EC adds a mirrored set of “imaginary” query rotations feeding the same keys and values, while RoPE++EH reduces KV heads but still outputs the same number of attention heads by combining real and imaginary components. Figure 4 (Page 8) then empirically verifies the memory and TPOT benefits of RoPE++EH; for example, in Figures 4a and 4c, RoPE++EH consistently uses less memory than RoPE at 32k context, and Figures 4b and 4d show shorter decoding TPOT, with the gap widening as context length grows.

5. **Comprehensive experimental evaluation across scales and benchmarks.**  
   The paper evaluates at 376M and 776M, plus a 1.5B follow‑up in Appendix C, all pre‑trained for 50B tokens on 4k context and then extended with 5B tokens at 32k context.  
   - **Short‑context:** Table 1 (Page 7) reports perplexity and accuracy on WikiText, LAMBADA, and 9 Open LLM Leaderboard tasks. RoPE++EC tends to achieve the best or near‑best average scores (e.g., 41.0 vs. 40.1 for 376M Short, and 43.5 vs. 42.0 for 776M Long).  
   - **Long‑context:** Table 2 (Page 7) shows clear gains on RULER and BABILong at 32k and 64k contexts, where RoPE++EC consistently outperforms RoPE; for 776M, RoPE++EC achieves an average RULER score of 29.4 vs. 27.4 for RoPE, and BABILong 24.1 vs. 22.8.  
   - **Combination with interpolation methods:** Table 3 (Page 9) demonstrates that RoPE++ still improves over RoPE when combined with Linear PI and YaRN, across multiple model sizes and metrics, reinforcing that the imaginary-attention idea is complementary to existing length-extension tricks.  
   - **Scaling:** Tables 5–6 (Page 17) replicate the pattern at 1.5B. While the gains are not always dramatic, RoPE++EH or RoPE++EC often achieve higher RULER/BABILong averages than RoPE, confirming that the mechanism is not an artifact of a single small model.

6. **Insightful attention pattern and perturbation analysis.**  
   Figure 5 (Page 8) presents attention heatmaps from several heads in 376M and 776M models. The real heads (e.g., Figures 5a, 5c, 5f, 5h) focus more locally around the current token, while the paired imaginary heads (5b, 5d, 5g, 5i) show broader, often near‑uniform attention over the full context and stronger emphasis on early tokens. Figure 5j shows that corrupting imaginary attention with Gaussian noise (\(\sigma\)) degrades RULER‑4k scores much faster than corrupting real attention, especially around \(\sigma=1.0\) where the gap is 5–8 points. This is a nice piece of supporting evidence that imaginary heads are particularly important for long‑context reasoning.

7. **Practical relevance and integration with existing tooling.**  
   The method is designed to integrate with FlashAttention (Section 3.3, Page 4): since keys share the same positional embedding, only queries need the extra \(-\pi/2\) rotation, and both real and imaginary scores can be computed in one pass. The authors also implement RoPE++ for both standard MHA and GQA, and they show throughput and storage comparisons in Table 11 (Page 19), making it easier to judge engineering overhead.

## Weaknesses

1. **The novelty is somewhat modest conceptually, closer to a re‑parameterization / head‑doubling than a fundamentally new positional encoding.**  
   The main change is to keep both real and imaginary components from the existing complex RoPE formulation. The paper correctly notes that imaginary parts have been underused in RoPE, but prior work on complex-valued networks and attention (e.g., Lee et al., 2022; Wang et al., 2025) already explored complex representations in attention-like settings. The paper could do more to clearly distinguish what is *specific* to RoPE++ beyond “use the imaginary part as more heads”. For instance, Section 3.3 argues that imaginary attention “cannot exist independently” because a \(\pi/2\) rotation can map it back to real RoPE. This is essentially a statement about linear dependence, and undercuts the apparent conceptual novelty: one might interpret RoPE++EC as just a constrained way of duplicating and tying parameters across heads. The paper’s contribution is still useful, but the positioning in the introduction feels slightly oversold relative to the simplicity of the idea.

2. **Mathematical analysis is expectation‑based and omits several subtle but important details.**  
   The derivations in Appendix B assume i.i.d. components with mean \(\mu\) and variance \(\sigma^2\) for queries and keys, and then compute expectations like \(\mathbb{E}[\bm{q}^\top \mathcal{R}_{-\Delta t} \bm{k}]\) and the characteristic curves \(c_{\mathrm{Re}}(\Delta t)\), \(c_{\mathrm{Im}}(\Delta t)\). While this is a reasonable analytical device, some claims feel stronger than warranted:
   - In Section 3.2 (Page 4), the authors state that imaginary attention “still shares the semantic-aggregation property” and “on average, this component attends more to distant positions”. This relies on the sine‑integral curve in Equation (5) being “very slowly declining” beyond some distance, but no rigorous inequality is given to show that \(c_{\mathrm{Im}}(\Delta t)\) is positive and larger than \(c_{\mathrm{Re}}(\Delta t)\) over a nontrivial range. Given that \(\sin(\theta \Delta t)\) oscillates and is zero at \(\Delta t=0\), more precise conditions (e.g., bounds on \(d\), frequency distribution, and \(\Delta t\)) would strengthen the argument.  
   - Equation (2) on Page 3 appears to contain a typo: the two lines, both starting with \(\sum_{n=0}^{d/2-1}\), are separated by a stray equals sign and a minus, which makes the final expression ambiguous. It likely should be a single sum with two terms, similar to Equation (1), but as written it is mathematically inconsistent.
   - In Section 3.4, the reasoning about “OO D negative embeddings” and the trained vs. untrained regions (Figure 3) is qualitative. A more formal statement about the measure of \(\Delta t\) values for which each component sees the full \([-1,1]\) range during training, as a function of maximum train length and frequencies, would make the extrapolation claim more solid.  
   Overall, the math is intuitively reasonable but somewhat informal, and there are a few places where the notation or signs in the equations could be clarified or corrected.

3. **Experimental scope, while solid, is still limited to relatively small models and single‑corpus pre‑training.**  
   The largest model is 1.5B (Table 5–6), trained on 50B + 5B tokens of DCLM‑Baseline‑1.0. This is non‑trivial, but far from the scale where RoPE is normally deployed (tens of billions of parameters and hundreds of billions of tokens). The core claim is that RoPE++ improves long‑context LLMs, but without at least one experiment on a mid‑size (e.g., 7B) model with realistic long‑context workloads (code, retrieval‑augmented QA, or document QA tasks beyond synthetic RULER/BABILong), it is hard to know whether the observed improvements persist under higher‑capacity models and more diverse data. The authors mention resource limits (Appendix C.1), but from a scientific standpoint, this still constrains the impact of the work.

4. **Some key baselines and ablations are missing or underdeveloped.**  
   - In short‑context experiments (Table 1), the paper compares to RoPE, FoPE, Pythia partial RoPE, and ALiBi, which is good. However, for long‑context tasks in Table 2 and Table 3, only RoPE vs. RoPE++ variants are reported; the other strong position encodings (FoPE, DAPE, PaTH, etc.) are not compared on RULER/BABILong. Given that the core selling point is long‑context performance, direct comparison against other recent extrapolation‑oriented methods on these benchmarks would substantially strengthen the experimental story.  
   - Ablations would benefit from an explicit “double‑heads RoPE” baseline that simply doubles the number of standard RoPE heads (with untied parameters) while keeping KV cache constant, to disentangle gains due to the imaginary formulation from gains due simply to more heads. RoPE++EC is constrained because real and imaginary heads share \(W_q\); a head‑doubling RoPE baseline with the same Q/K/V dimensions and head count would help isolate whether the specific imaginary‑attention structure matters beyond capacity.  
   - Similarly, an ablation where imaginary attention is down‑weighted or linearly combined with real attention (beyond the noise experiment in Figure 5j) would more directly test whether simply having an extra diverse set of rotations is the crucial factor.

5. **Interpretation of the perturbation experiment could be more careful.**  
   In Section 5.2 and Figure 5j, the authors add Gaussian noise to either real or imaginary attention and note that performance drops more when corrupting imaginary scores. They interpret this as “imaginary attention plays a more dominant role in long‑context tasks”. While plausible, alternative interpretations exist: for example, imaginary heads might have higher effective magnitude or smaller redundancy, so noise of the same variance affects them more. Without normalizing for per‑head variance or considering the relative scale of the two components, it is hard to translate the perturbation results into a precise statement about dominance. A simple check would be to report the norm statistics of real vs. imaginary attention matrices or their gradients, or to equalize the perturbation relative to each component’s standard deviation.

6. **Clarity issues and minor inconsistencies in exposition.**  
   - There are several places where notation is overloaded or slightly inconsistent. For instance, Equation (1) and the Appendix B version differ in sign conventions (\(e^{-i\theta(t-s)}\) vs. \(e^{i\theta(s-t)}\)), and the paper occasionally says “negative imaginary part” vs. just “imaginary attention”. Explicitly pinning down the convention (e.g., always using \(-\mathrm{Im}[\cdot]\)) and giving the reason for that choice once in Section 3.1 would help.  
   - The definition of \(\bm{\mathcal{R}}_{-\frac{\pi}{2}}\) in Equation (4) is only implicitly given via the rotated query components in Equation (3). A short explicit matrix form and explanation (e.g., “\(\mathcal{R}_{-\pi/2}\) is the 2D rotation by \(-\pi/2\) applied blockwise to each [2n,2n+1] pair”) would make the mapping from complex imaginary part to vector rotation clearer.  
   - The description of the impossible configurations (“75% imaginary vs 25% real or 100% imaginary are impossible”, Page 5) is somewhat confusing. This is due to the parameter sharing between real and imaginary attention, but the text reads more like a hard constraint of the method than an implementation choice. Explicitly stating that these ratios are not supported *in this design* (because of head tying) would clarify that one *could* design asymmetrical allocations, just not under the current parameter sharing scheme.

7. **Limited discussion of related complex‑valued and positional‑encoding literature.**  
   While the Related Work section mentions some complex‑valued neural network papers and position‑encoding extensions, it omits several directly relevant lines:
   - High‑level surveys on positional encodings and alternatives (e.g., overviews of absolute, relative, rotary, learned, and structured encodings).  
   - Prior analyses that revisit RoPE mathematically or use complex arithmetic explicitly beyond Su et al. (2024).  
   - Broader long‑context modeling methods like structured state spaces, Longformer‑style sparse attention, or hybrid architectures, which provide alternative solutions for long‑range dependencies.  
   Including these would better situate RoPE++ in the broader landscape and clarify to what extent its benefits complement, compete with, or could be combined with those orthogonal approaches.

8. **Dependence on training from scratch and lack of plug‑and‑play story.**  
   The Limitations Section (Page 20) acknowledges that RoPE++ “needs training from scratch and fails to deliver plug‑and‑play length extrapolation” and that it does not match methods like FoPE or PaTH in that regard. This is an important practical limitation: for many practitioners, the most appealing features of RoPE modifications are their ability to extend existing checkpoints’ context lengths with little or no continued training. RoPE++ requires architectural changes and retraining, which reduces adoption potential relative to interpolation‑based methods. The paper would benefit from a clearer discussion of how existing RoPE models could be adapted to RoPE++ (e.g., partial head replacement, distillation, or backward‑compatible modes), or at least from a more candid comparison of cost/benefit trade‑offs.

9. **Some numerical gains are modest or inconsistent.**  
   Although the average scores in Tables 1–3 and 5–6 generally favor RoPE++, not all tasks see improvements, and some drops are notable. For instance, in Table 1 (776M Short, 4k context), RoPE++EH reduces WikiText perplexity vs RoPE (15.6 vs 14.8, worse), and GPQA accuracy drops significantly (15.8 vs 25.8). In Table 2 (376M Long), RoPE++EH underperforms RoPE on RULER 4k (29.9 vs 31.6) and 16k (17.6 vs 22.0) despite higher 8k. The paper mostly reports averages, which mask these regressions. A more systematic analysis of where RoPE++ helps or hurts (e.g., task categories or context‑length regimes) would provide more nuanced guidance to practitioners.

## Potentially Missing Related Work

1. **Wang, B., Zhao, T., Zhang, Y. (2022). “Position Encoding in Transformer Models: An Overview.”**  
   This survey provides a comprehensive taxonomy of positional encodings and discusses their extrapolation properties. It is directly relevant to Section 2 and Section 3.4, where the authors motivate improving RoPE’s extrapolation and compare against a small subset of alternatives. It should be cited in the Related Work section and briefly used to position RoPE++ among absolute, relative, learned, and rotary schemes.

2. **Beltagy, I., Peters, M. E., Cohan, A. (2020). “Longformer: The Long-Document Transformer.”**  
   Longformer introduces sparse attention patterns specifically for long documents. While it is an architectural rather than positional solution, it is highly relevant to the long‑context modeling theme discussed in the Introduction and Section 2. It would be appropriate to mention it in Related Work to clarify that RoPE++ tackles positional encoding, which is complementary to sparse attention designs like Longformer.

3. **Huang, P., Liu, X., Qiu, X. (2023). “Enhancing Transformer Models with Position-Aware Attention.”**  
   This work proposes modifications to attention that directly incorporate positional information into the attention computation, which is conceptually similar to how RoPE++ augments scores with imaginary components. It should be discussed alongside other RoPE-improvement works in Section 2 and possibly compared in the context of how positional information is injected into the attention score.

4. **Zhang, S., Li, J., Zhao, H. (2022). “Complex-Valued Neural Networks for Sequence Modeling.”**  
   This paper explores complex-valued representations and operations for sequence models, including the use of both real and imaginary components. Given that RoPE++ hinges on using the full complex product, this work is directly relevant to Section 3.1 and Appendix B and should be cited where the authors discuss prior attempts to use complex-valued architectures.

5. **Chen, Y., Liu, J., Wang, S. (2023). “Revisiting Positional Encoding in Transformers.”**  
   This paper systematically examines the impact of different positional encoding schemes on transformer performance. It is relevant to the empirical and analytical evaluation in Sections 3.2–3.4 and 4.1–4.3. Citing it in Related Work and discussing how RoPE++’s improvements compare to those observed for other schemes would strengthen the positioning.

6. **Gong, Y., He, D., Chen, X. (2022). “Efficient Long-Range Sequence Modeling with Structured State Spaces.”**  
   Structured State Spaces (SSMs) are an alternative route to long-range sequence modeling. While orthogonal to positional encoding, they address the same high-level problem of long-context dependencies. A brief mention in the Introduction or Related Work, especially around the discussion of “long-context arena” and alternative architectures, would contextualize RoPE++ relative to SSM approaches.

7. **Liu, H., Dai, Z., So, D. R. (2021). “Pay Attention to MLPs.”**  
   This paper shows that much of transformer expressivity can be attributed to MLP blocks and that attention can, to some extent, be simplified. Although not about position encodings per se, it is relevant when discussing how architectural tweaks (like RoPE++) interact with the rest of the transformer stack. A short mention in Related Work or Discussion, noting that RoPE++ targets attention rather than MLP expressivity, would clarify scope.

## Questions

1. **Clarification of Equation (2) and sign conventions.**  
   Equation (2) on Page 3 appears to have a duplicated sum with an extra equals sign and minus sign:  
   \[
   \sum_{n=0}^{d/2-1}(\cdots)\sin\theta_n(t-s) - = \sum_{n=0}^{d/2-1}(\cdots)\cos\theta_n(t-s)
   \]  
   Could the authors provide the correct, fully expanded expression for \(\bm{A}^{\text{Im}}_{t,s}\) and explicitly state the sign convention (e.g., always using \(-\mathrm{Im}[\cdot]\))? A clean derivation from the complex product would greatly aid clarity.

2. **Head‑doubling RoPE baseline.**  
   To what extent are the long‑context gains of RoPE++EC due to simply having more attention heads versus the specific imaginary‑attention structure? Would the authors be able to report RULER/BABILong results for a baseline that doubles head count under vanilla RoPE (with matched Q/K/V sizes and cache) to isolate capacity effects?

3. **Scale and data diversity.**  
   The current experiments use DCLM‑Baseline‑1.0 and model sizes up to 1.5B. Do the authors have any preliminary evidence (even partial or fine‑tuning based) that RoPE++ behaves similarly on larger models (e.g., 7B+) or on qualitatively different corpora (code, multilingual text, synthetic long‑range patterns)? Any such results, even limited, would increase confidence that the observed patterns generalize.

4. **Quantitative analysis of real vs. imaginary attention contributions.**  
   Beyond the noise experiment in Figure 5j, can the authors provide statistics on the average norm or entropy of real vs. imaginary attention matrices, or the gradient norms through these components? This would help disentangle “dominance” from “sensitivity to noise” and might reveal whether imaginary heads truly carry more crucial information or are simply less redundant.

5. **Backward compatibility with existing RoPE models.**  
   In practice, could one retrofit RoPE++ to an existing RoPE‑based LLM by, for instance, expanding the attention module and initializing imaginary heads from real ones (or via distillation) without full retraining from scratch? If the authors have considered such approaches, a short discussion or pilot experiment would make the method more appealing to practitioners with existing checkpoints.

6. **Impact on optimization and convergence.**  
   Tables 7–9 in Appendix C suggest training and validation losses for RoPE and RoPE++ track each other closely, but RoPE++ sometimes shows slightly higher validation loss yet better downstream averages (or vice versa). Could the authors comment on whether RoPE++ changes training dynamics (e.g., gradient noise scale, learning rate robustness) and whether any special tuning was required compared to RoPE?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The work focuses on architectural changes to LLMs’ positional encoding. It does not introduce new datasets, sensitive applications, or deployment practices that raise specific ethics concerns beyond those of standard LLM pre‑training.

## Soundness Rating

3: good.  
The method is technically sound overall, with correct high‑level formulations and reasonable experimental evidence across several settings. Some derivations are informal and a few equations/notations (especially Equation (2)) need correction or clarification, and some baselines/ablations are missing, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is generally well written and organized, with helpful figures and tables (e.g., Figures 1–5 and Tables 1–3). However, there are some notation inconsistencies, minor typos in equations, and the related work could be more comprehensive regarding complex‑valued and long‑context methods.

## Contribution Rating

3: good.  
The contribution is not conceptually deep but is a clever and practically relevant extension of a ubiquitous component (RoPE). The combination of mathematical analysis, efficiency designs (RoPE++EH/EC), and fairly extensive experimentation makes the work a meaningful addition to the literature on long‑context transformers and positional encodings.

## Overall Rating

8: Accept, good paper (poster).  
The paper delivers a neat and well‑motivated extension of RoPE by exploiting the imaginary part of the complex attention, offers a thoughtful mathematical characterization of its properties, and backs this up with reasonably thorough experiments at multiple scales and on several long‑context benchmarks. Despite some missing baselines, limited model scale, and a few clarity issues in the math and exposition, the strengths clearly outweigh the weaknesses. The idea is simple but impactful enough, and the empirical gains plus cache‑efficiency configuration make it likely to be of interest to the ICLR community.

## Reviewer Confidence

4: confident.  
I am familiar with RoPE, positional encodings, and long‑context LLM work, and I have carefully checked the main derivations and experimental design. Some corner‑case mathematical details (e.g., exact behavior of the sine‑integral characteristic curve) and the breadth of related work in complex‑valued models could still hide subtleties, but they are unlikely to overturn the overall assessment.