---
job_id: a9e9697a-c74b-48e2-bd95-b0258995f1d7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: WhO6Km5Rku.pdf
paper: QUBITCache: Quantum-Inspired Probabilistic Attention Preservation for KV-Cache Compression
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is clearly about KV-cache compression for transformer-based LLMs, using quantum-inspired representations, squarely within representation learning / optimization / hardware-aware ML topics appropriate for ICLR.

## Minimum Quality
Pass ✅.  
All major sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present; the paper is in English and reports nontrivial new methodology plus extensive experiments. While I find substantial technical and conceptual issues, they do not rise to a level that justifies desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, instructions to reviewers, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper proposes QubitCache, a KV-cache compression framework that treats attention patterns, rather than tokens themselves, as the primary information carrier in transformers. The method keeps about 15% of tokens (anchors, recent, and “critical” tokens) in standard KV-cache form, while encoding attention distributions over the remaining tokens via a quantum-inspired amplitude encoding into low-dimensional “quantum states”, which are then used to reconstruct soft attention contributions during inference. The authors claim theoretical guarantees on preservation of rank‑$r$ attention structure and empirically report around $7\times$ memory reduction with 92–97% of baseline performance across several LLMs and long-context benchmarks, outperforming existing token-eviction and quantization baselines.

## Strengths

1. **Interesting conceptual shift (relational vs token-centric)**  
   The paper emphasizes that attention patterns and relational structures may be more important than individual token retention. This is well-motivated in Section A.2, where the authors connect to prior interpretability and graph-theoretic analyses of attention, and it is concretely instantiated by encoding attention distributions into a compact latent representation rather than just keeping “important” tokens.

2. **Hybrid architecture is clearly articulated at a high level**  
   Figure 1 provides a reasonably clear overview of the pipeline: partitioning tokens into anchor / recent / critical / non‑critical, encoding attention over non‑critical tokens into a 9‑qubit segment state, and combining classical attention over preserved tokens with probabilistic contributions from compressed segments. This figure helps the reader understand the intended data flow and where compression actually occurs.

3. **Nontrivial empirical evaluation across multiple models and tasks**  
   The experiments cover five different 4–8B models and a variety of long-context tasks. Table 1 is substantial: it reports seven metrics (PG19 F1?, PIQA accuracy, HotpotQA F1, TriviaQA F1, GovReport ROUGE, Contract accuracy, SummScreen ROUGE) for each of five baselines plus QubitCache, on five different models. QubitCache is typically close to the Full KV baseline and consistently above ScissorHand, H2O, and StreamingLLM; it is often competitive with or slightly better than GEAR despite a higher claimed compression ratio. This suggests the method is at least empirically viable in the tested regime.

4. **Clear demonstration of memory savings, at least at the KV level**  
   Table 3 makes the claimed memory benefit explicit for Llama‑8B on 8K‑token sequences: QubitCache reduces KV memory to 0.55 GB (a 7.0× compression) compared to 3.91 GB for Full KV, and is slightly more memory-efficient than GEAR’s 0.59 GB (6.7×). The asymptotic complexity row clarifies that the method scales with $0.15 S$ preserved tokens plus an additional $\log N$ term for the quantum state, which is easy to reason about.

5. **Component ablations partly support the central hypothesis**  
   Table 4 (“Component ablation: validating attention-based selection”) shows that removing attention-selected “critical” tokens causes a large performance drop (0.491 → 0.391), whereas removing anchors or recent tokens causes only a minor drop. The comparison to “Random + Quantum” (0.335 vs 0.491) empirically underscores that attention-based token selection matters much more than preserving an equivalent fraction of random tokens, which aligns with the paper’s relational-information narrative.

6. **Some quantitative study of quantum parameters**  
   Figure 3(a,b) explores F1 vs. number of qubits and F1 vs. circuit depth, showing monotonic improvements with more qubits and saturation around depth 15. While these are limited, they at least give a first-order sense of how the quantum-encoding configuration affects performance and make the “NISQ‑feasible” claim more concrete.

7. **Appendix provides useful interpretative context**  
   Appendix A.2 is quite thorough in arguing, via prior work, why attention graphs and their spectral structure might be more important than token embeddings. Appendix A.3 provides reconstruction-focused ablations (Table 5) that, although not task metrics, do help clarify which parts of the quantum construction actually matter for KV reconstruction (e.g., hybrid vs. fully quantum).

## Weaknesses

1. **Core “quantum advantage” / information-theoretic claim is misleading and not justified**  
   The abstract and Section 2 say QubitCache “achiev[es] logarithmic compression beyond classical information-theoretic limits” and that “classical methods remain bounded by $H(X)\ge\log_2|X|$ bits for distinguishable states $|X|$”. In the actual implementation, however, everything is simulated classically (Section 3.2.2, Section A.1.1). A 9‑qubit *statevector* has $2^9$ complex amplitudes; storing it in FP16 or FP32 is a *classical* $O(2^n)$ representation. There is no evidence that the proposed encoding violates or surpasses classical Shannon bounds, nor that the method would be representable with $O(\log N)$ *classical* bits. The claim conflates quantum notation with classical storage and risks misinforming readers about achievable compression. At minimum, the authors should very clearly distinguish: (i) theoretical qubit-level information capacity (which is relevant only on actual quantum hardware), and (ii) the real memory footprint of their classical simulation. Right now the narrative overstates “beyond classical limits” in a way that is scientifically incorrect.

2. **Theoretical “rank‑$r$ attention preservation with bounded error” claim is unsubstantiated in the main paper**  
   The abstract and Introduction mention a theorem guaranteeing preservation of rank‑$r$ attention structures with bounded reconstruction error, but no theorem, lemma, or proof actually appears in the main text. Equations (1)–(7) define the encoding and reconstruction heuristics but there is no formal result of the form  
   \[
   \|\tilde{A}-A\|_F \le \varepsilon(r,\text{compression})
   \]  
   or similar. The closest we get is the low-rank / spectral motivation in Appendix A.2, which cites other work but does not derive guarantees for *this* method. For an ICLR main track paper that leans heavily on such a guarantee in its narrative, this is a substantial gap. If the main claims rest on “we can preserve rank‑$r$ attention with bounded error,” those claims must be explicitly stated and at least sketched in the main paper.

3. **Mathematical formulation is often underspecified or inconsistent**

   - **Equation (1) vs. later usage**: Eq. (1) defines $|\psi\rangle = \sum_i \sqrt{\alpha_i}\,|i\rangle$ with $\alpha_i$ normalized from $a_i$ attention weights, ostensibly over “non-critical tokens”. In Eq. (2) and Eq. (7), coefficients $\alpha_i$ reappear for preserved tokens, but it is never made clear whether these are the same normalized attention scores, recomputed per step, or something else. There is also no head‑ or layer‑specific indexing in Eq. (2), despite $A^{(l,h)}$ in Eq. (3); it is unclear whether the same $\alpha_i$ is shared across layers and heads at inference, or if per-head states are used.

   - **Equation (7) indexing is confusing**:  
     \[
       p_j(|\psi\rangle) = | (j \bmod n_s | \psi_{S_{j/n_s}}) |^2
     \]  
     This is syntactically odd and ambiguous. It seems you intend  
     \[
       p_j = |\langle j \bmod n_s \,|\, \psi_{S_{\lfloor j/n_s\rfloor}}\rangle|^2,
     \]  
     but the current formula does not clarify the use of $\lfloor\cdot\rfloor$ vs. integer division, and units (segment index vs. position-within-segment) are mixed. This is a critical mapping, because it determines how compressed segments map back to absolute token positions.

   - **Interpolation in Eq. (6) is not well-motivated mathematically**:  
     \[
       \hat{V}_j = \frac{d_{j,\text{left}}}{d_{j,\text{left}}+d_{j,\text{right}}} V_{\text{left}(j)} +
                   \frac{d_{j,\text{right}}}{d_{j,\text{left}}+d_{j,\text{right}}} V_{\text{right}(j)},
     \]
     with $d_{j,k}=|j-k|^{-1}$. This gives weights proportional to the *inverse* distance, but then normalized by the sum; effectively, the closer preserved neighbor dominates. This is plausible, but there is no analysis of whether this preserves particular properties (e.g., Lipschitz continuity of attention outputs) or how sensitive performance is to this choice vs. e.g. exponential or learned interpolation kernels. Given that $\hat{V}_j$ is central to how compressed tokens contribute, a more principled justification or at least an ablation on the interpolation kernel is needed.

   - **Choice of $\lambda = \sqrt{|\mathcal{I}_p|/N}$**: Eq. (7) sets the mixing weight between preserved and reconstructed contributions to this ad‑hoc formula, but there is no derivation or sensitivity study. Why square root instead of linear scaling? How does this interact with sequence length or segment size? Without either theory or thorough ablations, this looks like a tuned heuristic rather than a principled design.

4. **Quantum mechanics is mostly veneer over a classical low-dimensional encoding**

   Despite the “quantum-inspired” branding, the method boils down to storing a normalized vector $\alpha$ of length 512 per segment and then using its squared entries as probabilities. The amplitude encoding and circuit in Figure 2(a,b) are mathematically equivalent to preparing a normalized vector and later reading its squared magnitudes; since everything runs on a classical simulator, this is just a fancy reparameterization of a classical distribution. You are not using interference, complex phases, or nontrivial entanglement in any demonstrable way. Table 5 (“w/o Entanglement” vs. “Full QubitCache”) shows virtually no difference in reconstruction metrics (MSE 0.0124 vs 0.0124, cosine 0.943 vs 0.943). This strongly suggests that the “quantum” layer could be replaced with a simple learned or fixed low-rank mapping from attention weights to soft selection coefficients, without any reference to qubits or circuits. The paper does not explore or compare against such classical parameterizations, nor does it isolate any benefit unique to genuine quantum operations.

5. **Computational and runtime costs of the classical quantum simulation are not analyzed**

   While Table 3 focuses on *memory* for the KV cache, there is no quantitative treatment of *latency* or flops associated with encoding/measurement. Section 3.4 claims amortized $O(\log n)$ per-token update for quantum states, but this ignores the constant factor of simulating 9‑qubit statevectors with controlled‑$R_y$ gates for each segment. On GPUs, simulating $2^9$ amplitude vectors is cheap, but the method scales with the number of segments and layers, and the cost of repeated measurement (shots) is nontrivial. The experiments all cap sequence lengths at 8K (Section 4.1.2), whereas the Introduction motivates 100K+ contexts. Without wall-clock comparisons to baselines, it is unclear whether QubitCache is actually efficient in *time* for the touted compression regimes.

6. **Experimental inconsistencies and limited metric justification**

   - **Metric choices and naming**: In Table 1, PG19 is reported as “F1(↑)”, which is not standard (PG19 is used for perplexity / language modeling). The text in Section 4.2 mentions “PG19 language modeling” but not how the F1 is defined. This raises questions about whether metrics are correctly implemented and comparable to prior work.

   - **Context length mismatch with motivation**: The introduction emphasizes 100K‑token contexts and 122 GB KV caches, yet all experiments use 2K–8K token sequences (Section 4.1.2 and Table 3). There is no experiment approaching the highly long-context regime where KV memory is the bottleneck; as a result, the claimed practical relevance for “repository-scale” and “multi-document” long contexts is not convincingly demonstrated.

   - **No variance / significance reporting**: All performance numbers in Tables 1 and 2 are point estimates, with no standard deviations, confidence intervals, or multiple seeds. Given that some improvements over GEAR and other baselines are small (e.g., GovReport ROUGE 0.837 vs 0.840 for Llama‑8B, Table 1), it is impossible to tell which differences are statistically meaningful.

7. **Ablations insufficient to disentangle contributions of individual design elements**

   The component ablation in Table 4 uses a single F1 score (presumably on one task / model, unspecified) and lumps several different design aspects together. For instance, “No Quantum” still preserves critical tokens but removes quantum encoding for non-critical ones; “Random + Quantum” keeps similar fraction but with random preserved tokens. There is no ablation where the “quantum” part is replaced with a classical low-dimensional embedding of attention weights (e.g., PCA, learned autoencoder) under the same retention ratio. Given that Table 5 shows entanglement and “noise dropout” have almost no effect on reconstruction, this missing baseline is particularly important: it would clarify whether any improvement over “No Quantum” is due to the specific amplitude encoding, or simply due to adding a smoothed attention prior over discarded tokens.

8. **Related work on KV compression is substantially incomplete**

   The Related Work section focuses on ScissorHand, H2O, StreamingLLM, and a bit on KV quantization (AWQ, KVQuant). However, there is now a rapidly growing body of work on KV cache compression and allocation with provable guarantees or sophisticated allocation schemes. None of the following, all highly relevant, are cited or compared against:
   - Low-rank / spectral KV compression with attention fidelity guarantees (e.g., KQ‑SVD).
   - Training-free, provably exact or near-lossless compression schemes such as Q‑Filters or GaugeKV.
   - Context reconstruction and query-agnostic eviction (KVzip, KV‑Distill).
   - Layer-aware and allocation-optimized approaches (GUI‑KV, CAKE, XKV, and similar).
   - Transform coding based approaches with high compression ratios (e.g., KV transform coding methods).  
   Many of these target exactly the same problem (KV cache reduction at inference) and some explicitly aim to preserve attention structure or provide provable fidelity bounds, which overlaps with the central claims of QubitCache. Not engaging with them weakens the positioning and undercuts the novelty claims.

9. **Claims around “graceful degradation” and “catastrophic failure in baselines” are not systematically supported**

   The text repeatedly asserts that discrete token-eviction baselines suffer catastrophic failures and that QubitCache offers graceful degradation. While some qualitative evidence is given in Tables 6–8 (XSum samples) and summarized in Table 9, this analysis is based on 3 samples and highly anecdotal. Quantitatively, many baselines in Table 1 are not catastrophically bad (e.g., Llama‑8B + H2O on HotpotQA: 0.502 vs 0.537 Full KV vs 0.510 QubitCache). The paper would need more systematic error-type statistics over a substantial test set to justify these strong claims; otherwise, they come across as cherry-picked.

10. **Notational and presentation issues that affect clarity**

    - There are several typos and small inconsistencies (e.g., “qquantum” in Section 4.5, “No Anchor / No Recent” in Table 4 lack precise definitions in the main text; “le f t” and “r i g h t” spacing issues in Eq. (6)); while minor individually, they accumulate.
    - Sections 4.5.2 and A.3 refer to “associative memory” and “noise dropout” as if they were well-defined components, but they are only loosely described post-hoc. For instance, “associative memory” corresponds to using the measured distribution vs random sampling, which is just standard probabilistic selection; the terminology adds confusion without clear technical distinction.
    - Figure 2(b) depicts a circuit with CNOTs and single-qubit rotation/measurement boxes, but the text never formalizes what these entanglement operations compute beyond the already‑encoded amplitudes. Given that ablations show no effect from removing entanglement, the figure risks over-emphasizing a component that is not empirically relevant.

Collectively, these weaknesses substantially reduce my confidence in the claimed theoretical contributions and the uniqueness of the “quantum-inspired” aspect, even though the empirical results themselves are reasonably solid.

## Potentially Missing Related Work

All of the following appear highly relevant to KV-cache compression and are not cited anywhere in the paper; they should be discussed, likely in Section 2 and in relation to Tables 1–3:

1. **Godey et al., “Q-Filters: Cache Compression and Quantum Applications”, 2025**  
   This work uses projection-based methods for KV-cache compression and explicitly connects to quantum-inspired designs, targeting high compression ratios (up to 32×) without retraining. It is directly related both in problem (KV compression) and in the “quantum” framing; it should be compared conceptually to QubitCache’s amplitude encoding and ideally included as a baseline in the compression/memory tables if feasible.

2. **Wang & Wang, “GaugeKV: Composable Exact KV Cache Compression”, 2025**  
   GaugeKV exploits gauge symmetries to achieve *exact* KV cache compression compatible with various attention mechanisms. Since QubitCache also claims attention-structure preservation, the paper should discuss how its approximation guarantees (if formally provided) compare to exact schemes like GaugeKV, and whether hybrid use is possible.

3. **Kim et al., “KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction”, 2025**  
   KVzip compresses KV caches in a query-agnostic manner while enabling contextual reconstruction, which is conceptually close to QubitCache’s soft reconstruction from compressed representations. It should be cited in the Related Work section and, if feasible, added to Tables 1 and 2 as a baseline to better contextualize QubitCache’s performance.

4. **Lesens et al., “KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity”, 2025**  
   KQ‑SVD introduces low-rank decompositions with explicit attention fidelity guarantees. Since QubitCache emphasizes “rank‑$r$ attention structure preservation” with bounded error, a side‑by‑side conceptual comparison is essential, and KQ‑SVD should be referenced after the claims around fidelity and rank preservation (e.g., near Eqs. (3)–(5)).

5. **Hosseini et al., “InnerQ: Hardware-aware Tuning-free Quantization of KV Cache for Large Language Models”, 2026**  
   InnerQ provides hardware-aware quantization of KV caches with strong empirical results. It is relevant to Table 3’s discussion of memory vs performance, and should be cited as part of the quantization-focused prior art, possibly discussed alongside GEAR and KVQuant.

6. **Staniszewski & Łańcucki, “KV Cache Transform Coding for Compact Storage in LLM Inference”, 2026**  
   This method uses transform coding to reach up to 20× KV compression while maintaining long-context and reasoning accuracy. Given the similar goal of high-ratio compression with minimal degradation, it should be explicitly compared to QubitCache’s 7× claims in terms of both performance and computational overhead.

7. **Chari et al., “KV-Distill: Nearly Lossless Context Compression for Transformers”, 2024**  
   KV‑Distill compresses context representations while preserving accuracy, often in a nearly lossless fashion. Since QubitCache claims 92–97% retention under 7× compression, it should be situated relative to KV‑Distill in terms of both achievable ratios and whether it requires retraining.

8. **Huang et al., “GUI-KV: Layer-aware Compression for KV Caches in Transformers”, 2025**  
   GUI‑KV introduces layer-aware allocation of KV budgets. QubitCache currently uses a uniform 15% retention and does not adapt per-layer; discussing GUI‑KV would highlight potential extensions, and it should be cited in the discussion of memory budget strategies (Section 4.4).

9. **Qin et al., “CAKE: Cache Allocation for Key-Value Eviction in Transformers”, 2025**  
   CAKE formalizes cache allocation as a utility-maximization problem under memory constraints. Given that QubitCache also chooses which tokens to preserve vs compress, CAKE is directly relevant and should be discussed when motivating the token categorization and critical-token selection strategy.

10. **Li et al., “XKV: Personalized Per-layer Schedules for KV Cache Management”, 2024**  
    XKV models layer-wise importance of KV values as a knapsack problem, providing personalized management strategies. This is relevant to QubitCache’s global 15% retention ratio and could inspire or benchmark a more refined per-layer variant; the paper should acknowledge and compare to this line of work.

## Questions

1. **Where is the formal theorem on rank‑$r$ attention preservation?**  
   You repeatedly state that QubitCache “preserves rank $r$ attention structure with bounded reconstruction error,” but there is no formal theorem in the main paper. Can you provide the precise statement (assumptions, norm, and dependence on compression ratio) and at least a sketch of the proof in the rebuttal, or clarify if this claim should be weakened?

2. **What is the *actual* classical memory footprint of the “quantum state” in your implementation?**  
   For each 512‑token segment, do you store a 512‑dimensional amplitude vector (in FP16/FP32) or something more compact? How many segments per layer are active at 8K and at, say, 100K tokens? Could you provide a precise breakdown of: (i) KV-cache memory, (ii) attention/quantum-state memory, and (iii) overhead for metadata, so we can verify the end-to-end 7× compression claim?

3. **Can you report wall-clock latency and FLOP overhead vs. baselines?**  
   Given that you simulate quantum circuits classically, what is the per-token decode latency relative to FullKV, GEAR, and StreamingLLM for Llama‑8B on 8K inputs? It would be very helpful to see a table similar to Table 3 but for runtime (both encoding and inference), to judge whether the memory savings are achieved at an acceptable computational cost.

4. **How sensitive is performance to the choice of $\lambda$ and the interpolation kernel in Eq. (6)?**  
   Have you tried $\lambda$ as a tunable hyperparameter, or different scaling (e.g., $\lambda = |\mathcal{I}_p| / N$)? Likewise, have you compared inverse-distance interpolation to exponential kernels $e^{-|j-k|/\tau}$ or to learned interpolation weights? Some small ablation or at least a sensitivity plot would significantly strengthen the case that the current choices are not arbitrary.

5. **What happens if you replace the quantum amplitude encoding with a classical low-rank or learned mapping?**  
   For instance, you could store a 32‑dimensional latent per segment, learned via a small autoencoder over attention vectors, and then derive soft weights from that. Do you expect similar or worse performance? Such a baseline would clarify whether the benefit is due to the specific amplitude encoding formalism or simply due to having *any* smooth prior over discarded tokens.

6. **Clarify the metric and setup for PG19 and the F1 scores in Tables 1 and 2.**  
   PG19 is normally evaluated via perplexity; how exactly are you computing “PG19 F1”? Also, for Table 2 (NarrativeQA), what is the evaluation setup (extractive vs. generative?). Providing more detail would help the community reproduce and interpret the results.

7. **Scalability beyond 8K tokens**  
   Do you have any preliminary experiments or sanity checks at longer context lengths (e.g., 16K, 32K) even if not across all benchmarks? Given the central long-context motivation, it would be reassuring to see at least one model+task evaluated beyond 8K with reported memory and performance.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The empirical evaluation is reasonably thorough and suggests the method can work, but several key theoretical claims (rank‑$r$ preservation, beyond-classical limits) are unsupported or overstated, and the mathematical formulation leaves critical design choices heuristic and under-analyzed.

## Presentation Rating

2: fair.  
The paper is readable and figures/tables are informative (e.g., Figure 1, Tables 1–3), but there are multiple notational inconsistencies, unclear definitions (e.g., Eq. (7), “associative memory”), and oversold claims about quantum advantages that harm clarity.

## Contribution Rating

2: fair.  
Reframing KV compression in terms of relational/attention preservation is valuable, and the empirical results are decent, but the actual technical novelty over classical low-dimensional attention encodings is not convincingly isolated, and the connection to genuine quantum computation is mostly cosmetic in the current classical implementation.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The work presents an interesting and somewhat original perspective on KV-cache compression and has solid empirical evidence that the proposed scheme is competitive with existing baselines at moderate sequence lengths. However, the theoretical claims are underdeveloped or overstated, the “quantum-inspired” aspect does not appear to yield a real advantage over straightforward classical schemes, and important related work is missing. With more rigorous theory, clearer mathematical specification, honest treatment of the classical nature of the implementation, and stronger baselines (especially non-quantum low-rank / transform-based compression), this line of work could become impactful. In its current form, I see it as promising but not yet at ICLR main-track standards.

## Reviewer Confidence

4: confident.  
I am familiar with KV-cache compression literature and quantum-inspired ML, and I have carefully checked the equations and experiments as presented. Some details (e.g., omitted theorem, missing runtime data) limit certainty, but the main points of critique are robust.