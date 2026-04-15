## Summary
This paper proposes a preprocessing approach for federated learning that aligns clients’ image data before training: each client computes channel-wise Wasserstein barycenters over its local RGB images, the server aggregates these into a global RGB barycenter, and clients project their images into that shared target space before running standard FL. The core idea is modular and potentially useful because it intervenes at the data level rather than modifying the optimizer, and the paper reports large gains over its own FedAvg baseline on CIFAR-10.

## Strengths
- **A genuinely distinct FL intervention point:** rather than designing another aggregation rule or personalization objective, the paper tackles heterogeneity through a preprocessing pipeline that is external to the learner. This is a concrete and fairly uncommon design choice in FL, and the pipeline is clearly specified in Algorithm 1 and Figures 2–3.
- **The method is interpretable rather than black-box:** the use of local and global channel-wise Wasserstein barycenters gives a transparent mechanism for what is being aligned. The construction is simple enough to understand and, in principle, easy to combine with existing FL pipelines.
- **There is at least a real empirical signal on the authors’ setup:** Table 1 and Figure 4 consistently show improvements over the paper’s own FedAvg baseline across several \(N/P\) configurations, including partial participation. Even if the broader claims are overstated, the method appears to have some effect in the tested setting.

## Weaknesses

### Fatal
- **The comparative claims against prior work are not supported by valid evidence.**  
  The paper explicitly claims things like “we have demonstrated the best generalization of current standing work” and that its results are “undoubtedly comparable” to numbers taken from other papers despite “not using the exact same hyperparameters” (Sec. 5, Table 2). This is not a valid basis for superiority claims: Table 2 mixes results from different papers with different models, data partitions, training budgets, and protocols. Since these cross-paper comparisons are used to support broad claims of outperforming other methods, a central empirical conclusion of the paper is currently unsubstantiated.

### Major:
- **The paper overstates what kind of heterogeneity it addresses.**  
  The method aligns **channel-wise RGB marginals**, but the framing repeatedly speaks much more broadly about “dataset imbalance,” “different generating distributions,” and minimizing “distributional discrepancy” in FL. The actual mechanism shown in Sec. 4 is much narrower: it acts on color-channel distributions. The paper does not show that this reduces FL-relevant heterogeneity in the broader sense usually meant in non-IID FL (e.g., label skew, semantic feature shift, class-conditional mismatch). This matters because the central narrative is that the method tackles the heterogeneity problem in FL, while the evidence only supports a narrow image-level alignment operation.
- **The empirical support is too narrow for the strength of the claims.**  
  The main paper evaluates only on CIFAR-10 with FedAvg. There is no controlled evaluation across multiple datasets or heterogeneity regimes in the visible paper, yet the conclusions are broad (“can be leveraged in any FL paradigm,” “paired with any learning algorithm,” “superior results than … other comparable work”). For a paper whose contribution is entirely empirical and whose gains are extremely large, the evidence base is too limited.
- **The data partition protocol is underspecified and not convincingly tied to standard non-IID FL heterogeneity.**  
  Sec. 5 says the data are distributed by “uniformly sampling, without replacement,” which ensures varying dataset sizes and that clients are “not completely homogenous.” But this does not clearly instantiate a standard, strong non-IID partition, and the paper does not quantify the resulting client heterogeneity. Given that the whole motivation is to handle imbalanced / discrepant client distributions, the lack of a precise heterogeneity specification makes it difficult to assess what problem is actually being solved.
- **The paper does not validate the claimed mechanism behind the gains.**  
  The abstract and conclusion argue that alignment “facilitates the learning process” by reducing discrepancy/variance, but the paper does not measure any quantity that would support this explanation: no pre/post client distance analysis, no client drift or gradient disagreement, no class-conditional preservation analysis, and no ablation against simpler color-alignment baselines. As written, the paper shows that a particular image transform helps in one setup, not why that help occurs or whether it is specifically due to OT-based alignment rather than generic color normalization.
- **The privacy claims are too strong for the evidence shown.**  
  Sec. 4 states that the method uses information from all agents “without losing privacy because WBs obfuscate the data in an irreversible fashion,” and the abstract similarly says this happens “without breaking privacy concerns.” In the visible paper, there is no formal privacy analysis and no empirical leakage analysis. At most, the paper demonstrates that raw data are not directly shared. That is much weaker than the privacy language currently used.

### Minor
- **The “model-/algorithm-agnostic” claim is only conceptual, not empirically demonstrated.**  
  It is reasonable to say the preprocessing is external to the learning algorithm, but the paper only evaluates it with FedAvg in the main text. Statements such as “allowing any learning algorithm to be paired with our method” are plausible as design intent, but they are not established experimentally.
- **The paper lacks controlled preprocessing baselines.**  
  Because this is a preprocessing paper, comparisons only against “no preprocessing” are insufficient. The paper itself mentions standard normalization/scaling techniques in Related Work, but does not empirically compare against simple alternatives that also align low-level statistics. Without such controls, it is hard to isolate what is uniquely gained by OT.
- **The complexity discussion is incomplete relative to the efficiency claims.**  
  Sec. 6 gives asymptotic costs for barycenter computation and projection, but there is no wall-clock accounting and no comparison to the total training cost or to alternative alignment methods. Since the paper also emphasizes faster convergence in communication rounds, the absence of end-to-end cost analysis weakens any practical efficiency claim.
- **Some technical exposition is imprecise.**  
  In Sec. 3.2, the paper says one is “looking for a permutation matrix \(P\)” in the Kantorovich formulation. In general, \(P\) is a transport plan/coupling, not necessarily a permutation matrix. This is not central to the contribution, but it is a mathematical imprecision in the preliminaries.
- **The channel-wise factorization may discard cross-channel structure, but the paper does not discuss this limitation.**  
  Since the method independently aligns red, green, and blue channels, it may not preserve full color correlations. That does not invalidate the method, but it is a real modeling limitation that should be acknowledged, especially because the method’s scope is already narrow.

### Trivial
- None.

## Nice-to-Haves
- Add a direct comparison against at least one stronger FL heterogeneity method under the **same** setup, rather than via copied numbers.
- Include one additional dataset and one standard non-IID partition protocol to show that the approach is not specific to the current CIFAR-10 setup.
- Quantify pre/post alignment using an inter-client discrepancy metric and visualize a few aligned images to make the transformation’s effect more transparent.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The appendix is missing, so implementation details are unavailable.”**  
  Removed as a reproducibility nitpick. The main paper already points to Appendix A.2 for details; lack of appendix text in the extracted content is not itself a valid criticism of the submission.
- **“The method should compare against additional recent related work.”**  
  Removed in this form due to the instruction not to mention missing related works. The valid underlying point is retained as a narrower criticism that the paper’s comparative claims are unsupported because its cited cross-paper comparisons are invalid.
- **“FedAvg baseline seems too low compared to what one might expect externally.”**  
  Removed because it relies on external expectations rather than verifiable evidence from the paper. What can be said, and is kept, is that the evaluation is undercontrolled and the partition protocol is underspecified.
- **Generic praise such as “the paper is well-written” or “the topic is important.”**  
  Removed as non-specific strengths.

## Novel Insights
The paper’s most interesting aspect is also its main limitation: by choosing a purely preprocessing-based route, it offers a modular and interpretable way to modify client data without touching the FL optimizer, but this very modularity makes it essential to precisely delimit what is being aligned. In the current form, the work is best understood not as a general solution to federated heterogeneity, but as an image-level color/distribution alignment heuristic that happens to help under the tested CIFAR-10 setup. That reframing would make the contribution more modest but also more technically honest, and it would clarify what evidence is still needed to elevate the idea into a strong FL claim.

## Suggestions
- Remove or sharply tone down all cross-paper superiority claims unless competing methods are rerun under the exact same experimental protocol.
- Reframe the contribution more precisely as **channel-wise image distribution alignment**, not a broad solution to FL heterogeneity or dataset imbalance.
- Specify and quantify the client partitioning protocol in detail, and evaluate at least one standard non-IID split.
- Add preprocessing control baselines that test whether OT is genuinely needed beyond simpler low-level alignment methods.
- Soften the privacy language to “does not require sharing raw data” unless a formal or empirical privacy analysis is included.
- If the authors want to keep the “agnostic to model/algorithm” claim, support it with at least one additional FL algorithm in the same setup.

## Score and Decision
**Novelty:** moderate. The idea of FL preprocessing via OT barycenter-based alignment is distinct and interesting, but the actual operation is fairly narrow.  
**Technical soundness:** below the bar. The method itself is coherent, but several claims are materially overstated relative to what is shown.  
**Empirical support:** weak for ICLR standards. The evidence is limited to one benchmark family, one main learner, unspecified heterogeneity severity, and an invalid cross-paper comparison used for strong claims.  
**Significance:** currently limited. The approach could become useful, but the present paper does not establish broad impact.  
**Clarity:** mixed. The pipeline is understandable, but the framing conflates narrow color-space alignment with broad FL heterogeneity reduction.

Overall, this is an interesting idea with some promising signal, but the submission in its current form does not meet the evidentiary standard for acceptance. The biggest issue is not that the method fails outright; it is that the paper claims substantially more than the experiments justify.

**Score:** 4.6  
**Decision:** Reject

MY FINAL SCORE: <pineapple>4.6</pineapple>
MY FINAL DECISION: <orange>Reject</orange>