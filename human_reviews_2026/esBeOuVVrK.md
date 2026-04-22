# ReciNet: Reciprocal Space-Aware Long-Range Modeling for Crystalline Property Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Predicting properties of crystals from their structures is a fundamental yet challenging task in materials science. Unlike molecules, crystal structures exhibit infinite periodic arrangements of atoms, requiring methods capable of capturing both local and global information effectively. However, current works fall short of capturing long-range interactions within periodic structures. To address this limitation, we leverage \emph{reciprocal space}, the natural domain for periodic crystals, and construct a Fourier series representation from fractional coordinates and reciprocal lattice vectors with learnable filters. Building on this principle, we introduce the reciprocal space-based geometry network (\textbf{ReciNet}), a novel architecture that integrates geometric GNNs and reciprocal blocks to model short-range and long-range interactions, respectively. Experimental results on standard benchmarks JARVIS, Materials Project, and MatBench demonstrate that ReciNet achieves state-of-the-art predictive accuracy across a range of crystal property prediction tasks. Additionally, we explore a model extension to multi-property prediction with the mixture-of-experts, which demonstrates high computational efficiency and reveals positive transfer between correlated properties. These findings highlight the potential of our model as a scalable and accurate solution for crystal property prediction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes ReciNet, which integrates geometric GNNs and reciprocal blocks to model short-range and long-range interactions, for crystalline property predictions. In contrast to existing work that struggles in capturing long-range interactions within periodic structures, this work leverages reciprocal space and constructs a Fourier series representation from fractional coordinates and reciprocal lattice vectors with learnable filters to address the challenge. The model is further extended to a multi-task variant, enabling more efficient and scalable multi-property prediction. Given the experimental results, ReciNet achieves state-of-the-art predictive accuracy across benchmarks and shows positive transfer between correlated properties.

### Strengths
1. ReciNet effectively models long-range interactions in crystalline materials by combining geometric message passing in real space with learnable Fourier representations in reciprocal space. The idea is physically grounded and demonstrates solid scientific insight.

1. The manuscript is well-written, intuitively explaining how reciprocal blocks complement conventional GNN-based local modeling.

1. The experimental section is comprehensive, covering multiple benchmarks (Materials Project, JARVIS, and MatBench) and comparing against various baselines, including iComFormer, M3GNet, and CrystalFramer.

### Weaknesses
1. Though achieving the SOTA performance on various evaluation datasets, the improvement by ReciNet compared to the strongest baselines is not significant. 

1. The idea of using reciprocal space to model long-range interactions is physically well-motivated. However, the experiments in the paper are limited to conventional crystalline property prediction tasks, where long-range effects may not play a dominant role. As a result, the work does not clearly demonstrate the practical significance or unique benefits of explicitly learning long-range interactions in reciprocal space.

### Questions
1. The paper proposes to learn long-range information in the reciprocal space for better property prediction performance. However, properties like formation energy and band gap often depend primarily on local atomic environments. Could the authors identify specific properties or datasets where long-range information is critical to performance?

1. The multi-task variant ReciNet-MT is briefly mentioned. What motivates including MTL in this paper, and does it provide any clear scientific insight or practical benefit besides efficiency?

1. The reciprocal block models long-range interactions through learnable Fourier filters. Is there any interpretability analysis to validate that these filters capture meaningful long-range interactions rather than serving as generic global attention?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
ReciNet is a hybrid architecture that combines a geometric GNN for short-range interactions with a ReciprocalBlock that learns long-range interactions directly in reciprocal space. It builds a Fourier-series representation from fractional coordinates and reciprocal lattice vectors, using learnable filters rather than fixed, hand-crafted kernels. It benchmarks on the Materials Project, JARVIS-DFT, and MatBench, and achieves SOTA for most of the tasks.

### Strengths
1. A clear mechanism for long-range physics. ReciNet’s learnable ReciprocalBlock operates directly on fractional coordinates and reciprocal lattice vectors, preserving periodicity and space-group symmetries without supercells or fixed analytic kernels as a clean solution to long-range interactions.
2. Consistent SOTA across major benchmarks. It beats strong baselines on MP, on JARVIS, and leads MatBench (e_form and jdft2d).
3. The paper well explains why fixed, hand-crafted long-range potentials/kernels (PotNet/Crystalformer) or grid-based long-range projections (EwaldMP) are misaligned with crystal symmetry.

### Weaknesses
1. Is the primary difference from EwaldMP the avoidance of k-space grid discretization? Please clarify how operating directly with reciprocal lattice vectors captures long-range information better than a grid-based approach theoretically or empirically.
2. The reported gains are modest but consistent; in my view, this does not diminish the significance of the authors’ contribution.

Typos: 
1. Line 124: Remove space before ","
2. Line 229: Incomplete sentence around "are"
3. Line 426: "crystal-naive" -> "crystal-native"
4. Line 474: "shown in inference time as domination" -> "dominate its inference time"

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ReciNet, a crystal property prediction model that combines geometric GNNs with a reciprocal space representation to capture both local and long-range interactions in periodic structures. It achieves state-of-the-art results on major benchmarks and shows efficient multi-property prediction through a mixture-of-experts design.

### Strengths
(1) The paper presents clear motivations, and this reviewer agrees on the importance of capturing long-range interactions.

(2) The experiments are relatively comprehensive, covering diverse benchmarks and baselines, which effectively support the proposed approach.

(3) The use of reciprocal space is well-motivated, as it naturally aligns with the periodic nature of crystal structures.

### Weaknesses
(1) A major concern lies in the relatively marginal improvement over baselines. As shown in Tables 1 and 2, ReciNet only slightly outperforms existing models, and the gain may not be statistically significant. In particular, the ablation study shows that with only three blocks, ReciNet underperforms baseline approaches in 3 out of 5 metrics on the Materials Project dataset, which weakens the claimed contribution of the long-range module.

(2) If the main contribution is indeed the long-range interaction module, it should be tested on other representative architectures that currently lack such modeling, and compared against alternative long-range strategies. The current evaluation is not sufficiently ablative to establish the effectiveness of the proposed module.

(3) The receptive field range of the long-range module is not clearly explained. If it depends on cutoff or truncation, an efficiency analysis should be included to show how these parameters affect computational complexity and how the trade-off is managed. While the proposed method may achieve higher efficiency than Transformer architectures with O(N^2) complexity, this claim remains unsubstantiated without quantitative analysis.

### Questions
(1) Would increasing the depth of the strong baseline models also lead to significant improvements?

(2) Does the long-range module require computing all pairwise distances between atoms? If so, how does the proposed approach achieve the claimed efficiency advantage?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present ReciNet for predicting properties of crystalline materials. The main motivation is that existing GNN-based methods may suffer from a locality bias by using a fixed cutoff radius, failing to capture the infinite periodic nature of crystals and the associated long-range interactions that affect many key properties (e.g., bulk modulus, band gap).

ReciNet proposes a hybrid architecture that models interactions at two scales: one is the short-range geometric GNN that captures the local bonding environment and the other is the long-range ReciprocalBlock that captures the global periodic interactions in the reciprocal space.

### Strengths
1. The key motivation in the problem —the locality bias of GNNs —is a well-known and significant barrier in the field. Its use of learnable filters  (Eq. 9) is a significant step beyond models like PotNet, which rely on fixed, hand-crafted analytical functions (e.g., Ewald sums, Gaussian kernels).

2. Achieving SOTA on multiple properties across three standard, large-scale benchmarks (MP, JARVIS, MatBench) is a strong empirical validation.

3. The efficiency analysis (Table 5) is an important addition. Demonstrating that the proposed method is not only more accurate but also significantly faster to train than other recent SOTA models

### Weaknesses
1. The paper lacks physical insight or interpretability. This is the main concern. The paper only proves empirically that the learnable filter in Eq.(9) works, but it makes no attempt to explain why it works or what it has learned exactly. Without this analysis, the ReciprocalBlock remains a "black box." 

2. The details regarding the ReciprocalBlock are not clear. For example, Eq.(8) computes a sum over $k_m$, which are described as "basis reciprocal lattice vectors". Do these $k_m$ mean only the three basis vectors $(b_1, b_2, b_3)$? This seems like a very sparse and incomplete sampling of the reciprocal space. Or, does it mean a set of $k$-vectors generated by those basis vectors, as suggested by Eq.(17)?

3. For some properties relevant to long-range interactions, the performance of the proposed model does not achieve the SOTA, like Bandgap(OPT).

### Questions
1. Can you provide any in-depth theoretical analysis of the learned filter $W$ in Eq.(9)? How will it vary for different properties? This would provide invaluable physical insight.

2. Can you clarify precisely how the set of reciprocal vectors used in Eq.(8) is selected? 

3. Why does the proposed model fail to achieve SOTA on Bandgap(OPT)?

4. If you paired the proposed ReciprocalBlock component with a stronger short-range model like Matformer, would you expect even more significant performance gains?

### Soundness
2

### Presentation
3

### Contribution
2
