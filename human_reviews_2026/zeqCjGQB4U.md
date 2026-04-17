# Why Keep Your Doubts to Yourself? Trading Visual Uncertainties among Vision-Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Vision-Language Models (VLMs) enable powerful multi-agent systems, but scaling them is economically unsustainable: coordinating heterogeneous agents under information asymmetry often spirals costs. Existing paradigms, such as Mixture-of-Agents and knowledge-based routers, rely on heuristic proxies that ignore costs and collapse uncertainty structure, leading to provably suboptimal coordination.
We introduce Agora, a framework that reframes coordination as a decentralized market for uncertainty. Agora formalizes epistemic uncertainty into a structured, tradable asset (perceptual, semantic, inferential), and enforces profitability-driven trading among agents based on rational economic rules. A market-aware broker, extending Thompson Sampling, initiates collaboration and guides the system toward cost-efficient equilibria. Experiments on five multimodal benchmarks (MMMU, MMBench, MathVision, InfoVQA, CC-OCR) show that Agora outperforms strong VLMs and heuristic multi-agent strategies, e.g., achieving +8.5% accuracy over the best baseline on MMMU while reducing cost by over 3×. These results establish market-based coordination as a principled and scalable paradigm for building economically viable multi-agent visual intelligence systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces "Agora," a novel framework for multi-agent coordination designed to address the high cost and inefficiency of multi-agent systems (MAS) built with VLMs.

The authors' core argument is that existing coordination paradigms, such as Mixture-of-Agents (MoA) or knowledge-based routers like KABB, are "Cost-Agnostic" and "Uncertainty-Structure-Agnostic." They rely on heuristic proxies (e.g., consensus, semantic similarity) for decision-making, which leads to provably suboptimal coordination and spiraling economic costs.

To solve this, the Agora framework reframes coordination as a "decentralized market for uncertainty." Its core mechanisms include: Assetization, trading and brokerage.

Comprehensive experiments are conducted on five multimodal benchmarks. The results demonstrate that Agora significantly outperforms strong VLM baselines and heuristic multi-agent strategies in both accuracy and cost-effectiveness. For example, on MMMU, Agora (79.2%) achieves a +8.5% absolute accuracy gain over the best baseline in its agent pool, while reducing costs by over 3 times.

### Strengths
**Highly Novel and Well-Grounded Problem Formulation:** The paper's main strength is its core idea: "market-based uncertainty trading." This provides a principled, economic alternative to heuristic routing in MAS. The theoretical formulation, which defines "Agnostic Coordination" and proves the "Inefficiency Theorem for Agnostic Coordination", provides an exceptionally strong motivation and foundation for the work.

**Sound and Constructive Methodology:** The Agora framework is well-designed. The decomposition of uncertainty into three types (perceptual, semantic, inferential)  is a key, intuitive design choice that enables fine-grained cost management. The "profitability-driven" trading protocol directly and elegantly optimizes the core economic objective.

**Extremely Comprehensive and Rigorous Experiments:** The experimental validation is a significant strength.

### Weaknesses
**Performance Claims:** Thank authors for the comparison in Table 1 against gemini-2.5-pro (81.7%). Does this imply that Agora's primary value is "cost-saving" (achieving near-SOTA with a non-SOTA pool) rather than "SOTA-beating"? If gemini-2.5-pro were included in Agora's agent pool, would you expect the Agora score to surpass 81.7%?

**Uncertainty Quantification:** I found the uncertainty vector interesting. 

1. Could you provide a concrete example of how the uncertainty vector $[u_{perc}, u_{sem}, u_{inf}]$ is calculated from a VLM response? For instance, for the query in Figure 359, "Is this place crowded?", if an agent replies "It seems somewhat busy," how are the numerical values for these three uncertainty dimensions estimated? 
2. Could we change the dimension of the vector? What would happen if we design the uncertainty vector $[u_{observation}, u_{answer}]$ or $[u_{perc}, u_{sem}, u_{inf}, u_{answer}]$​? 
3. How could we determine the weight of each dimension?

**Framework Overhead:** The ablation study (Table 3) shows the "Strategic Uncertainty Index" ($U_{strategic}$) is the most critical component. As defined in Appendix C, computing this index seems complex, requiring an estimation of expected system-wide cost savings for all potential trades. Is this computation tractable in a real-time system? What is the computational overhead (e.g., in latency or FLOPS) of the Agora broker and trading mechanism itself, separate from the VLM inference costs?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Agora, a market‑based way to coordinate multi‑agent VLM systems. Instead of heuristic routing or consensus, Agora uses epistemic uncertainty in a tradable asset with three dimensions—perceptual, semantic, and inferential—and lets agents “trade” packets of uncertainty to reduce total cost. A market‑aware broker (an extension of Thompson Sampling) initializes collaboration; trades are executed only if the cost delta ΔC is negative (Eq. 4), yielding a locally cost‑efficient equilibrium (Algorithm 1). Authors use extensive experiments to demonstrate the Agora's effectiveness in comparison with single LLMs, SOTA, and other benchmarks.

### Strengths
1. The paper formalizes why common coordinators are cost‑agnostic and uncertainty‑structure‑agnostic (Theorem 1), then builds a mechanism that is explicitly cost‑aware and structure‑aware (Eq. 4; uncertainty portfolio on p. 5). Fig. 2 (p. 3) empirically shows lower final epistemic uncertainty.
1. The trade condition ΔC<0 with capacity constraints is intuitive and easy to implement
1. The paper measures COI, Ufinal, and UAPS (uncertainty‑aware performance), offering a fuller view of coordination quality

### Weaknesses
1. The market phase performs greedy descent and is guaranteed to reach only a local equilibrium (Algorithm 1). The bandit section explicitly notes that classic regret bounds are hard and provides only directional convergence intuition (Appendix C.4.1), leaving global‑optimality and regret guarantees open.
1. The broker’s utility (Eq. 6) multiplies several factors (distance, synergy, strategic uncertainty, decay). Ablations (Table 3) show sensitivity (e.g., removing Ustrategic costs ~3% accuracy), suggesting tuning burden and potential dataset‑specific overfitting.
1. Entropy/dispersion‑based scores (and their mapping to Uperc/Usem/Uinf) can be model‑calibration–dependent; without shared calibration or normalization, one model’s probabilities may look more or less “uncertain” than another’s even on identical behavior. The paper formalizes the decomposition but does not detail cross‑model calibration procedures.
2. Total uncertainty fuses dimensions with fixed weights (e.g., wperc=0.4, wsem=0.3, winf=0.3); if those weights were tuned on setups where the proposed method reduces the dominant dimension, Ufinal_epis could favor it. (Weights are reported as defaults; sensitivity within Agora is ablated, but cross‑method sensitivity is not shown.)
1. In the main body, head‑to‑head comparisons with alternative routing/multi‑agent strategies are only shown on MMBench V11 Test (Figure 4). For the other datasets (MMMU, MathVision, InfoVQA, CC‑OCR), the main table contrasts Agora vs. single models / external SOTAs, not against routing/MAS baselines—so readers can’t verify cross‑dataset superiority of the coordination strategy from the main text alone.
1. The writing should be enhanced, especially regarding some details and accuracy. For example, the color description in Figure 1 is wrong. The Mathvision gap in Table 1 is wrong. The readability of Figure 5 is poor (the font could be adjusted to make it easier to read).

### Questions
1. How do you do cross‑model calibration or normalization for Entropy/dispersion‑based scores?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces Agora, an framework for efficient coordination among VLM agents. It conceptualizes epistemic uncertainty as a tradable asset which is decomposed into perceptual, semantic, and inferential components. The authors also design a profitability-driven trading protocol where agents exchange uncertainty based on rational market rules. The system employs a market-aware broker extending Thompson Sampling method to initiate and manage trades, aiming to achieve cost-efficient equilibria. Experiments on 5 multimodal benchmarks show performance gains about 8.5% accuracy while reducing cost by over 3-times compared to SOTA multi-agent and routing strategies.

### Strengths
I'm not an expert in the field of VLM, but I appreciate the idea of treating uncertainty as a tradable commodity, which is novel seems to me.  And the introduced Broker algorithm that operationalizes the market through Thompson Sampling looks solid. The empirical study is reasonably sufficient and supportive. And the writing is easy to follow in general.

### Weaknesses
1. While the authors prove the inefficiency of heuristic coordination, the optimality guarantees of Agora’s decentralized equilibrium are not clearly established. Is the equilibrium globally optimal or locally optimal? More formal convergence or even a broader discussion would be appreciated.
2. The framework are limited to visual-linguistic benchmarks, and I would be curious to see if Agora scales to pure language, control, or multi-round dialogue tasks. Can you include some discussion in this regard? If it cannot be easily adapted to other domains, what could be the reason?
3. What is exactly the theoretical result regarding the regret bound in Appendix C.4.1? I get lost there and did not identify any concrete result.

Minor issues and typos:
1. P2: “Gale & and, 1962”
2. Duplicated references of the paper 'Are more llm calls all you need? towards scaling laws of compound inference systems'.

### Questions
1. How sensitive is Agora’s performance to the parameters in Eq. 6? 
2. Can Agora handle non-stationary uncertainty distributions, for example, what if agents adaptively reprice uncertainty in evolving tasks?
3. Since the framework could converge to suboptimal equilibria, are we able to observe such phenomenons in experiments? Or are there any useful heuristics to help escape local minima?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Agora, a novel market-based framework for coordinating heterogeneous Vision-Language Models (VLMs) in multi-agent systems. The key idea is to treat epistemic uncertainty as a tradable asset across perceptual, semantic, and inferential dimensions. Agents trade uncertainty based on a profitability-driven protocol, guided by a market-aware Thompson Sampling broker, to minimize system-wide cost while maintaining task performance. The authors provide both theoretical and empirical evidence that Agora outperforms strong baselines (e.g., Mixture-of-Agents, KABB) in terms of accuracy (+8.5% on MMMU) and cost-efficiency (3× reduction).

### Strengths
- The core concept of a decentralized market for uncertainty is a powerful and novel metaphor that drives the entire paper.
- The work tackles a foundational challenge (economic scalability) with a comprehensive solution that includes theoretical formulation, algorithm design, and empirical validation.
- The framework demonstrates consistent and sometimes substantial gains in accuracy and cost-efficiency across multiple benchmarks.

### Weaknesses
-  The framework introduces significant complexity with its multi-dimensional uncertainty quantification, trading ledger, and sophisticated broker. While the paper claims scalability, the experiments are limited to ≤15 agents. It’s unclear how the market mechanism would behave with hundreds of agents or high-frequency trading.
- The paper overlooks key transaction costs (coordination latency, broker computation, and communication overhead), which are excluded from its cost metrics. A system that is 3x cheaper in API calls but 10x slower due to coordination may not be viable for real-time applications.
-  While numerous baselines are included, the adaptation of language-model-focused routers (like FrugalGPT, RouteLLM) to the multi-modal, multi-agent setting is only briefly mentioned in the appendix.

### Questions
- How would Agora perform in the presence of adversarial or strategic agents that intentionally misreport uncertainty or cost to exploit the market? For instance, how does the framework handle strategic manipulation or collusion that could distort trading dynamics or market equilibrium?
- Could you clarify the computational complexity of the broker and trading protocol, particularly their overhead within a more realistic analysis?
- Would Agora still work if agents have overlapping expertise or non-stationary cost profiles?

### Soundness
4

### Presentation
4

### Contribution
3
