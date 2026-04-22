# From Invariant Representations to Invariant Data: Provable Robustness to Spurious Correlations via Noisy Counterfactual Matching

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Models that learn spurious correlations from training data often fail when deployed in new environments. While many methods aim to learn invariant representations to address this, they often underperform standard empirical risk minimization (ERM). We propose a data-centric alternative that shifts the focus from learning invariant representations to leveraging invariant data pairs---pairs of samples that should have the same prediction. We prove that certain counterfactuals naturally satisfy this invariance property. Based on this, we introduce Noisy Counterfactual Matching (NCM), a simple constraint-based method that improves robustness by leveraging even a small number of \emph{noisy} counterfactual pairs---improving upon prior works that do not explicitly consider noise. For linear causal models, we prove that NCM's test-domain error is bounded by its in-domain error plus a term dependent on the counterfactuals' quality and diversity. Experiments on synthetic data validate our theory, and we demonstrate NCM's effectiveness on real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper tackles domain generalization problems with a shift in focus from learning invariant representations to leveraging invariant data, where an invariant pair is a pair of samples that should share the same prediction under the optimal hypothesis. The authors first show that counterfactual pairs are invariant and analyzes when such pairs improve robustness and how many are needed. The paper then introduces Noisy Counterfactual Matching, which adds a constraint to ERM to improve robustness to spurious correlations using a small set of (noisy) invariant pairs. The scope of distribution shift considered is restricted to domains that only intervene on spurious latent variables that are non‑ancestors of the target variable.

### Strengths
* The paper is clearly written and easy to follow.
* The problem is well motivated, with an interesting insight and thorough theoretical analyses for the setting considered.
* Empirical results are presented to support the theoretical claims.

### Weaknesses
* Theoretical results are only applicable to linear SCMs, which is rather simple and restrictive, relative to real‑world scenarios. 

* The collection of invariant data pairs is only feasible given some knowledge about the spurious features. While the authors argue that such knowledge may be available in practice, I think that the knowledge of some may be feasible, but knowledge of all possible spurious features is hard to achieve, even with domain knowledge. For data like images, spurious features may simply be related to the background or style of the images. However for tabular data in arbitrary more complex domains, humans are often deprived of the full knowledge of all possible spurious factors. 

* There are other technical concerns that requires further clarification. See the questions below.

### Questions
1. The logic of the proposal seems problematic to me. The paper proves that counterfactual pairs improve robustness and then uses the fact that counterfactual pairs are invariant to justify collecting invariant pairs instead. Are counterfactual and invariant pairs equivalent? From my understanding of its proof, Proposition 1 shows that counterfactual pairs are invariant, but not all invariant pairs are counterfactual. If the model gives the same prediction to different inputs, collecting such pairs may not yield the desired robustness effect. Could the authors clarify this? 

2. The examples about the collection of invariant pairs, in the Introduction and Appendix A.2, seem to rely on knowing that pairs share the same label, whereas invariance is defined with respect to the optimal hypothesis. True labels and optimal predictions can differ. How does this mismatch affect your results? If invariant pairs can be collected using training labels, why not define invariance directly with respect to the true label and perform the alignment/subspace objective accordingly, rather than relying on the robust model’s prediction?

3. What is the output space of the robust classifier $h$? Does it output a hard label or a label distribution?

4. Definition 1 is stated for classifiers, but Theorem 1 discusses linear and logistic regression losses. Is Theorem 1 applicable to common losses like cross‑entropy for classification tasks?

5. The authors also claim that one feasible way to collect invariant pairs is to perform augmentation. So at a high level, I do not understand how the proposed approach differs from data augmentation or in other words directly using augmented data for alignment.  

6. A major limitation is that the constraint in Eq. (1) reduces to Eq. (2) only under linear models. Once we have the counterfactual data pairs, why don’t we directly align them using some distance metrics, as often done in machine learning literature e.g., contrastive learning? 

7. I understand that theoretical analyses in non-linear settings are challenging. However, could the authors at least provide empirical results on standard benchmark like DomainBed to at least verify whether the idea carries over empirically?

My current rating reflects the above concerns. I will consider updating the score once they are resolved.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper presents Noisy Counterfactual Matching (NCM), a method for domain generalization. NCM uses pairs of samples that differ only in spurious aspects (features) to enforce consistent predictions and filters out spurious directions with truncated SVD. The authors provide theoretical support showing that few diverse pairs are sufficient for invariance and report experiments on ColoredMNIST, Waterbirds-CF, and PACS where NCM matches or slightly outperforms established baselines such as IRM, GroupDRO, SWAD, and LISA.

### Strengths
The paper presents a an approach to domain generalization based on counterfactual matching. Its originality lies in reformulating invariant representation learning without relying on explicit domain labels, supported by a well-developed theoretical framework. The analysis is technically sound, with proofs that connect counterfactual consistency to invariance and empirical results that align with the theory. The experimental setup is appropriate, using standard benchmarks.

### Weaknesses
While the paper is well-executed, its novelty is somewhat limited relative to prior work on counterfactual and invariant learning, such as MatchDG (ICLR 2021) . The distinction between NCM and these methods, beyond the use of formal proofs, could be explained more clearly. The empirical evaluation, though thorough on benchmark datasets, depends on synthetic or curated counterfactuals (e.g., Waterbirds-CF), which limits evidence of real-world applicability. Demonstrating how counterfactuals could be obtained or approximated in realistic settings would strengthen the paper. The study would also benefit from testing on additional or more recent benchmarks.

### Questions
1. The experiments primarily rely on synthetic or semi-synthetic datasets where counterfactuals are artificially constructed. How would NCM perform on real-world datasets where counterfactual pairs are unavailable or difficult to define or noisy? 
2. Would the authors consider including results for baselines on the synthetic dataset to confirm that NCM’s advantages are not specific to its own simulation setup?
3. Please include 2023, 2024, and 2025 methods in Related Work and, where feasible, add a conceptual and empirical comparison.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a data-centric approach to domain generalization called Noisy Counterfactual Matching (NCM). Instead of enforcing representation-level invariance, the authors suggest learning invariance directly from invariant data pairs. The method approximates counterfactual sample pairs and constrains the classifier so that its predictions remain consistent across them. Theoretical analysis shows that when these counterfactual pairs approximate the true invariant directions, the classifier achieves provable robustness to spurious correlations. Experiments on synthetic data show modest improvements over baselines.

### Strengths
1. Addressing spurious correlations and domain generalization remains an important and open challenge.
2. The paper is well-written and conceptually easy to follow, with a consistent flow from motivation to theory to experiments.
3. Experimental results show modest but consistent improvements on benchmark datasets.

### Weaknesses
1. The core idea of enforcing invariance through matching or orthogonality constraints is not entirely new. Previous works such as MatchDG, IRMv1 have already explored related ideas. NCM’s main distinction, using noisy counterfactual pairs, is interesting but incremental and mostly limited to linear settings.
2. The theoretical guarantees hold only for linear models with idealized “counterfactual pairs.” In realistic nonlinear cases (e.g., deep networks), the method lacks justification. The authors claim empirical robustness, but this cannot be directly attributed to the theory presented.
3. The approach assumes access to approximate counterfactual pairs or noisy invariant pairs, which may not be available in practice. The paper does not discuss how these pairs could be obtained in realistic settings or how sensitive the method is to poor-quality matches.
4. While the results show slight gains, they are relatively small, raising questions about whether the added complexity is justified compared to simpler ERM or representation-based baselines.
5. The extension of the orthogonality constraint via SVD projection to deep models is mentioned but not clearly demonstrated or validated experimentally.

### Questions
See Weaknesses Part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a data‑centric approach to robustness under domain shift that leverages invariant data pairs (instead of learning invariant representation)—pairs of inputs that should receive the same prediction under a robust classifier. The authors formalize the  spurious counterfactuals in a causal perspective, and prove that spurious counterfactuals are invariant pairs. They introduce Noisy Counterfactual Matching (NCM): augment ERM with a linear constraint that projects predictions onto the orthogonal complement of the spurious subspace estimated via the truncated SVD of pairwise differences. For linear and logistic regression, the authors provide theoretic guarantees of NCM . Empirically, synthetic experiments validate the theory’s trade‑offs, and linear probing on CLIP features shows gains on ColoredMNIST, Waterbirds‑CF, and PACS.

### Strengths
- The proposed method, Noisy Counterfactual Matching (NCM), is well-motivated and intuitive—it estimates a “spurious” subspace from pairwise differences and removes it via projection.
- NCM is sample efficient. This is often easier to obtain in practice.
- The implementation of NCM is simple, and has potential to be applied to different applications.
- The method addresses the biggest practical challenge—noisy counterfactual pairs—and remains robust through a simple truncated-SVD design with a provable error bound.
- The theoretical analysis is solid in the linear setting, and synthetic results nicely match the theory.
- Beyond the linear case, real-data experiments show consistent gains for CLIP linear probes, suggesting the approach has broader practical potential beyond its theoretical contribution.

### Weaknesses
- The projection removes only observed linear directions of domain shift, leaving potential nonlinear or unseen correlations unaddressed.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
