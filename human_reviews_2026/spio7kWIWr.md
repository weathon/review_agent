# Calibrated Spiking Messages for Emergent Multi-Agent Communication

- Decision: Reject
- Scores: 0, 2, 6, 2

## Abstract
We study emergent communication in multi-agent reinforcement learning (MARL) via a calibrated, bandwidth-aware framework that exchanges spiking messages built on a pretrained perceptual code. Agents share a spiking encoder (COMMSMOD) trained with a prototype–contrastive–sparsity objective, and use independent attention-based decision heads (DECISIONMOD) trained using calibration-aware Q-Learning. In referential games on Fashion-MNIST, with agents alternating sender/receiver roles, we assess protocol quality using within vs between class similarity, temporal attention consistency, and calibration; spike count serves as a bandwidth proxy. Experiments demonstrate the spiking channel yields accurate and sample-efficient communication, improves protocol discriminability, and reduces synaptic operations versus a matched continuous ANN baseline. Ablations show that (i) the shared pretrained encoder, (ii) temporal attention, and (iii) calibration terms are each necessary. Overall, semantically anchored, calibrated spiking communication offers a favourable accuracy–robustness–bandwidth trade-off and a practical route to neuromorphic deployment.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper uses spiking neural networks (SNNs) for communication in the emergent communication field and proposes to calibrate confidence to reduce the number of spikes. The proposed method is evaluated on the Fashion-MNIST dataset and shows improved learning performance with fewer messages.

### Strengths
Overall, I found the idea of using spiking neural networks for communication to be interesting.

### Weaknesses
1. The writing is unclear and lacks professionalism.
2. There are too many references, and the connection between them and the paper is not well explained. Many technical terms, such as “ECE/MCE” and “neuromorphic,” are introduced without explanation. Moreover, many relevant works are missing. 
3. The authors start discussing “single-agent deep learning” in the related work section which is conceptually nonsense—this suggests a lack of understanding of reinforcement learning fundamentals.
3. The paper only reports results on the Fashion-MNIST dataset, making it difficult to assess whether the proposed method and its claims generalize to other data or environments.

### Questions
- **Line 41:** The authors neglect several works on limited bandwidth communication, such as IMAC, NDQ, and DDCL. Moreover, the paper does not consider noisy communication channels in the experiments.
- **Line 48:** The reference to Pina et al. is unclear. That paper studies whether to use parameter sharing rather than ad-hoc teamwork with new partners. Could you clarify how your focus relates to that work?
- **Line 52:** The LIF dynamics are not properly referenced.
- **Line 83:** What are the coding schemes and temperature mentioned here?
- **Line 125:** The work by Guo investigates general deep learning. There is no such “single-agent” concept in that reference.
- **Section 3:** What are _K_, _T_, and _d_? Does the sender select messages for all steps at the start of the episode? It’s also unclear why the proposed method is placed in Section 3, which otherwise seems to describe background material.
- **Section 4.1:** The spike encoders appear similar to using binary messages. While this paper also considers temporal horizons, the formulas in Equations (6) and (7) are very specific and hard to generalize to broader multi-agent settings. Additionally, how are the coefficient parameters in Equation (8) chosen?
- **Equation 14:** It’s unclear how the constraints are incorporated into the expectation.
- **Table 1:** Are language tokens not discrete symbols?
- **Table 2:** The results are interesting. Compared to using discrete (Gumbel) or continuous symbols, spiking messages improve performance (CSR) while reducing message count (SynOps). However, I would not consider this an ablation study, but rather a comparison across different formats of messages. One question is why using discrete symbols (Gumbel) still results in such a high communication load.
- Finally, the agents are not adapted to new partners but rather to “COMMSMOD,” as the authors mention.


Related work such as:

- Mathieu Rita, Corentin Tallec, Paul Michel, Jean-Bastien Grill, Olivier Pietquin, Emmanuel Dupoux, Florian Strub. Emergent Communication: Generalization and Overfitting in Lewis Games. NeurIPS 2022
- Shengchao Hu, Li Shen, Ya Zhang, Dacheng Tao: Learning Multi-Agent Communication from Graph Modeling Perspective. ICLR 2024
- Zhuohui Zhang, Bin He, Bin Cheng, Gang Li: Bridging Training and Execution via Dynamic Directed Graph-Based Communication in Cooperative Multi-Agent Systems. AAAI 2025: 23395-23403.
- Rundong Wang, Xu He, Runsheng Yu, Wei Qiu, Bo An, Zinovi Rabinovich. _Learning Efficient Multi-Agent Communication: An Information Bottleneck Approach._ In Proceedings of the 37th International Conference on Machine Learning (ICML 2020).

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a novel approach that integrates *spiking neural networks (SNNs)* into emergent communication (EmComm), emphasizing *calibration* as a measure of agent reliability and robustness. The study aims to bridge neuromorphic learning with multi-agent reinforcement learning (MARL), showing how spike-based representations can affect communication performance. While the direction is innovative and thought-provoking, the paper’s contributions remain underdeveloped, with unclear empirical benefits, missing definitions, and insufficient quantitative validation.

### Strengths
- **Novel direction:**  
  The paper pioneers the use of *spiking neural networks (SNNs)* in the EmComm setting. This direction is a rarely explored but seems conceptually valuable for bridging the gap  between neuromorphic and communicative learning paradigms.

- **Calibration focus:**  
  Introducing *calibration* as an evaluation criterion in multi-agent systems is a meaningful extension to the typical focus on coordination or task success. It encourages the community to consider reliability and uncertainty-awareness as part of communication quality.

- **Potential for robustness and efficiency:**  
  The proposed spiking approach could contribute to advances in **energy-efficient** and **noise-tolerant communication**. Moreover, its event-based and structured nature may align with key traits of natural language such as **productivity**, **compositionality**, and **systematicity**. This direction also resonates with current trends in **biologically inspired computation** and **low-power neural architectures**, bridging insights from cognitive science and neuromorphic engineering.

### Weaknesses
- **Clarity and accessibility:**  
  The introduction assumes familiarity with spiking neuron dynamics (e.g., LIF, surrogate gradients) and does not provide enough background for readers from the EmComm community. A short background section on *spike encoding, LIF neurons, and surrogate gradients* would make the work much more approachable.

- **Unclear contributions:**  
  The manuscript does not clearly articulate its core contributions. While it references both *calibration* and *spiking communication*, it never isolates what is novel beyond combining the two. A dedicated “Contributions” paragraph at the end of the introduction would improve clarity.

- **Weak empirical evidence:**  
  The experiments are limited to *Fusion-MNIST* with *k=3*, which simplifies the problem substantially. There is no evidence that the proposed system generalizes or outperforms existing protocols. Quantitative comparisons are minimal and lack statistical grounding.

- **Ambiguous calibration concept:**  
  Although calibration is highlighted as central, the paper does not define or quantify it clearly (e.g., ECE, MCE, temperature scaling). Without showing calibration metrics or comparisons, the argument for its relevance remains conceptual.

- **Compositionality not evaluated:**  
  Table 1 references compositionality metrics, but none are computed or analyzed. Since compositionality is a key benchmark in EmComm, its omission weakens the paper’s analytical depth.

- **Incomplete related work:**  
  Important prior works in discrete and quantized communication are not cited, such as:  
  - Carmeli et al., *Emergent Quantized Communication*  
  - Tucker et al., *Trading Off Utility, Informativeness, and Complexity in Emergent Communication*  
  Including them would situate this work more concretely within existing EmComm literature.

### Questions
- **Abstract:**  
  The abstract uses domain-specific terms (*spiking messages, pretrained perceptual code*) without context. A short intuitive framing, e.g., “We use event-based neural dynamics to model communication between agents and assess their calibration and robustness,” would make it more accessible.

- **Terminology:**  
  - Line 52: Define LIF as *Leaky Integrate-and-Fire*.  
  - Lines 38–39: Note that quantized/discrete communication protocols are already common in EmComm.  
  - Lines 43–45: Clarify what “calibration” means in this context, as this differs from the traditional EmComm linguistic.

- **Formalisms:**  
  - Lines 177–180 and Eq. 3: Symbols such as *E* are undefined and equestion is difficult to follow.  
  - Lines 208–209: Appears to contain a fragment (“Spike encoders. Over normalised pixe…”). Likely a formatting or copy error.

- **Experiments and Results:**  
  - Header of Section 4.3 appears incomplete or truncated
  - Lines 303–304: Assertions should be empirically demonstrated or cited.  
  - Lines 360–362 (*Baselines*): Add *quantized communication* as an additional baseline.  
  - Lines 422–423: Table 2 does not appear to reflect all discussed results. May need correction.  
  - Lines 449–459: The comparison and ablation discussion lacks numerical support. Adding a summary table with accuracy, calibration error, and energy metrics would clarify the findings.

- **Presentation and Readability:**  
  Consider reorganizing the paper so that:  
  1. Background on SNNs and calibration precedes the method.
  2. The *proposed architecture* is visualized in a clear, labeled figure.
  3. The results section directly connects back to the stated hypotheses.
  4. Ablation results are clearly presented.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a neuromorphic framework for emergent communication in multi-agent reinforcement learning (MARL), where agents exchange temporal spiking messages rather than dense continuous vectors.

The system factorizes communication into three modules:

1 a pretrained spiking perceptual encoder that transforms sensory input (e.g., Fashion-MNIST images) into temporal spike trains;

2 an attention-based decoder that interprets these spike sequences and outputs calibrated confidence estimates; and

3  a reinforcement-learning wrapper that optimizes task return under explicit penalties for bandwidth (spike count) and miscalibration (ECE/MCE).

Training employs a calibration-aware deep Q-learning objective integrating reliability losses into the reward. Evaluation on a referential game (K = 3) demonstrates that:

Shared pretrained encoders stabilize communication and reduce sender–receiver co-adaptation;

Temporal attention and calibration terms improve both reliability and interpretability;

Spiking messages achieve ≈ 97 % communication success at roughly 87 % lower synaptic-operation cost than dense continuous baselines; and

Protocol diagnostics show high discriminability (δ ≈ 0.85–0.9) and strong temporal consistency (> 0.997).

### Strengths
1. Introduces a biologically grounded and bandwidth-aware communication mechanism using spiking neural codes, extending emergent communication research beyond symbolic or continuous channels.
2. Separates perception (COMMSMOD) and decision (DECISIONMOD) via a frozen shared encoder, reducing sender–receiver co-adaptation and stabilizing learned protocols.
3. Incorporates explicit calibration losses (ECE/MCE) into the learning objective—an unusual and valuable addition for multi-agent reliability.
4. Evaluates not just task performance but also geometric and temporal properties of communication, including discriminability (δ), attention consistency, and reliability diagrams.
5. Demonstrates high accuracy under limited spike budgets and noisy channels, with strong sample efficiency and large reductions in synaptic operations versus dense baselines.
6. Removing pretraining, temporal attention, or calibration terms substantially degrades both CSR and calibration quality, confirming the design’s necessity.

### Weaknesses
1. All experiments are conducted on Fashion-MNIST with K=3 candidates. It remains unclear how the approach scales to naturalistic tasks, continuous control, or communication among more than two agents.

2. While the paper introduces channel operators (drop/flip/jitter), the quantitative effects of each perturbation on CSR and calibration are not systematically presented—only summarized qualitatively.

3. Reported ECE/MCE values use five bins by default; sensitivity to bin count or adaptive calibration methods is deferred to the appendix, leaving the main results potentially unstable.

4. The only baselines are a continuous ANN and a discrete Gumbel-Softmax channel. Broader comparisons to recent structured communication models (graph or language-based) would strengthen the empirical claim.

5. Although the shared encoder is intended to improve ad-hoc cooperation, the paper does not evaluate cross-play between independently trained agents.

6. The methodological contribution lies mainly in system integration and diagnostics rather than in algorithmic or theoretical advances. Depending on ICLR track priorities, this could be seen as engineering-heavy.

### Questions
Can you provide quantitative curves of CSR/ECE vs. drop, flip, and jitter rates to substantiate the noise-robustness claim?

How stable are calibration metrics when using 10 or 15 bins (ECE/MCE)?

Have you tested cross-play performance between independently trained agents sharing only the frozen COMMSMOD?

Can you show a Pareto curve (CSR & ECE vs. spike budget) to illustrate the bandwidth–accuracy trade-off?

Does freezing COMMSMOD ever hinder adaptability to novel perceptual domains (e.g., CIFAR or DVS datasets)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a new method for MARL communication using a spiking network. The core idea is to use a pretrained, frozen spiking encoder and train an RL agent to select the correct image based on the resulting spike train.

### Strengths
1. Interesting idea of using spiking networks for communication. 
2. Extensive benchmark with multiple ablations.

### Weaknesses
1. The limitation discussion is missing or extremely brief, failing to address significant methodological weaknesses.
2. Proposed neuromorphic benefits are purely theoretical, and omitting components like neuron resetting/decaying in computational comparison does not seem fair.
3.  Evaluation is limited to a single Fashion-MNIST with k=3, which sounds like a very toyish problem for such scenario.
4. The paper's use of deep Q-learning is unnecessarily complex and poorly justified. The problem is a standard supervised classification task where the receiver has access to the true labels. The claim that RL is needed to add a calibration loss is false; a hybrid loss (e.g., cross-entropy + $\mathcal{L}_{cal}$) on a standard classifier would have achieved the same goal with far less complexity. Furthermore, by freezing the sender's encoder, the system is reduced to a single-agent problem, making the entire MARL framework a layer of abstraction that obscures, rather than clarifies, the method."

### Questions
1. Could you please give more information on the weaknesses/limitations of the method? 
2. How would computation/benefits change if we factor in (2) for neuromorphic hardware?
3. Could you address (3/4) from the previous section?

### Soundness
2

### Presentation
3

### Contribution
2
