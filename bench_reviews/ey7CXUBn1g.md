## Summary
This paper proposes AdaSVD, a method for compressing Large Language Models via Singular Value Decomposition. It introduces two core components: **adaComp**, an iterative procedure that compensates for SVD truncation error by alternately updating the singular matrices using a stabilized Moore-Penrose pseudoinverse solution, and **adaCR**, a heuristic that assigns layer-specific compression ratios based on a simple input-output similarity metric. The method demonstrates consistent performance improvements over existing SVD-based baselines across multiple model families and high compression ratios.

## Strengths
- **Comprehensive and convincing empirical evaluation.** The paper validates its method across multiple LLM families (LLaMA, OPT, Mistral, Vicuna), compression ratios (40%-80%), and task types (perplexity, QA, VLM captioning). The gains are particularly pronounced at higher compression ratios (e.g., 60%), which is a critical target for deployment.
- **Effective ablation studies and practical engineering.** The ablation studies in Tables 3a-3d cleanly isolate the contributions of each component, showing that adaComp is crucial at high ratios and adaCR provides consistent gains. The "stack-of-batch" strategy for calibration data is a clever, practical solution to a real memory constraint.
- **Orthogonality to other compression techniques.** The paper demonstrates that AdaSVD can be effectively combined with weight quantization (GPTQ), outperforming the quantized version of the strongest baseline (SVD-LLM+GPTQ), which enhances its practical utility.

## Weaknesses
- **Lack of computational cost analysis.** The paper claims efficiency but provides no quantification of the computational overhead of the iterative adaComp procedure or the layer importance estimation. The wall-clock time and memory cost of the compression process itself (not inference) compared to baselines like SVD-LLM are critical for assessing practical deployment trade-offs and are absent.
- **Weak justification and analysis for the adaptive compression ratio (adaCR) heuristic.** The core importance metric—cosine similarity between a layer's input and output—is intuitive but simplistic. The paper does not justify why this is a good proxy for a layer's importance to the *final model performance* under SVD compression, nor does it analyze its sensitivity or compare it to other possible metrics (e.g., gradient-based). This leaves the foundation of adaCR somewhat under-supported.
- **Severely hindered clarity due to parsing artifacts.** While not a flaw of the scientific content, the extracted text contains garbled tables, broken equations, and misplaced text/figure references (e.g., Table 1, Section 3.1 derivations). This significantly obstructs a detailed assessment of the methodology and results. For a conference submission, the authors must ensure a clean, readable PDF.

## Nice-to-Haves
- A more thorough discussion relating the adaCR importance metric to prior work on layer importance in pruning and compression.
- A sensitivity analysis of the adaComp procedure to the amount and quality of calibration data.
- Exploration on even larger-scale models (e.g., 70B parameters) to further stress-test the method's scalability.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Evaluate on larger-scale LLMs (e.g., 13B, 70B parameters)."** The paper already evaluates models up to 13B parameters (LLaMA-13B). Demanding evaluation on a 70B model is a "larger model" generic request and is not required to validate the core contribution.
- **Weakness: "Benchmark on diverse downstream tasks beyond perplexity and QA" and "Compare with non-SVD compression baselines (e.g., pruning)."** These are requests for scope creep. The paper's contribution is an improvement within the SVD-based compression paradigm, evaluated against SVD baselines on standard LM/QA benchmarks. Requiring comparisons to fundamentally different compression families (pruning) or a broader suite of generation tasks is not standard for establishing a SOTA advance in this specific area.
- **Weakness: "Convergence and stability analysis for adaComp."** While interesting, demanding a theoretical convergence analysis for an empirical, post-training compression method is imposing a rigor requirement not standard in this applied field. The empirical ablation (Table 3c) shows the procedure works effectively.
- **Weakness: "Report inference speed or latency measurements."** The paper's focus is on reducing memory footprint via compression, a valid goal independent of immediate inference speed measurements. Furthermore, SVD compression inherently changes the matmul structure, making direct latency comparison complex and hardware-dependent; its absence is not a core flaw.
- **Strength/Weakness about generic writing or topic importance.** Removed as per instructions.

## Novel Insights
The paper's primary novel insight is the integration of a stabilized, alternating update mechanism (adaComp) to directly minimize the *task-relevant* SVD truncation error (\(||U_k^\sigma V_k^{\sigma\top}X - WX||_F^2\)), moving beyond the standard Frobenius norm on weights. The use of the Moore-Penrose pseudoinverse within a least-squares formulation provides a numerically stable solution, and the "stack-of-batch" calibration strategy is a simple but effective tactic to maximize data utility under memory constraints. The combination of this compensation with a layer-adaptive compression scheme (adaCR) within a single, training-free SVD framework is also a new synthesis.

## Suggestions
- **Fix presentation completely.** Provide a clean, properly formatted PDF with legible tables, correctly rendered equations, and clear figure/text alignment for the final submission.
- **Add a computational cost analysis.** Include a table or subsection reporting the wall-clock time and peak GPU memory consumption required to compress a standard model (e.g., LLaMA2-7B) at a few key compression ratios, comparing AdaSVD directly to SVD-LLM.
- **Strengthen the discussion of the adaCR importance metric.** Justify the choice of input-output similarity more deeply, discuss its potential limitations, and optionally provide a brief comparison against another simple baseline (e.g., uniform or random allocation) to better isolate its contribution.