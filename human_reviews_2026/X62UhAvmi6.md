# Budgeted Broadcast: An Activity-Dependent Pruning Rule for Neural Network Efficiency

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Most pruning methods remove parameters ranked by impact on loss (e.g., magnitude or gradient). We propose Budgeted Broadcast (BB), which gives each unit a local traffic budget—the product of its long-term on-rate $a_i$ and fan-out $k_i$. A constrained-entropy analysis shows that maximizing coding entropy under a global traffic budget yields a selectivity–audience balance, $\log\tfrac{1-a_i}{a_i}=\beta k_i$. BB enforces this balance with simple local actuators that prune either fan-in (to lower activity) or fan-out (to reduce broadcast). In practice, BB increases coding entropy and decorrelation and improves accuracy at matched sparsity across Transformers for ASR, ResNets for face identification, and 3D U-Nets for synapse prediction, sometimes exceeding dense baselines. On electron microscopy images, it attains state-of-the-art F1 and PR-AUC under our evaluation protocol. We further implement BB for large language models using both unstructured and structured one-shot pruning.BB is easy to integrate and suggests a path towards learning more diverse and efficient representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Different from the previous pruning methods based on the magnitude or gradient, this paper proposed Budgeted Broadcast (BB) based on the inspiration from the metabolic cost in biological neural networks. They formalize this cost as a neuron's traffic, which is the product of a neuron's activity and its number of connections. The rule of BB is derived from constrained-entropy optimization. By applying to for domains, ASR, face identification, synapse detection, and change detection, it shows better or comparable performance with other pruning methods.

### Strengths
- This method reinterprets biological phenomena mathematically and applies it to prune artificial neural networks. 
- It has lower computational complexity because only local statistics (ai, ki) are used. It can be integrated into previous networks.
- It shows good results on various domains, especially for long-tail data.
- Common units have fewer connections, while rare units maintain more connections, increasing expressive diversity.
- It is based on the theoretical basis, "selectivity-audience balance' relationship.

### Weaknesses
- This paper deals with unstructured sparsity. This research has limitations because it is not easy to reduce actual computation compared to structure pruning, and specialized hardware is required for the acceleration.
- The application for large-scale models, such as LLMs or LMMs, is necessary to show the effectiveness of this method. 
- The performance depends on various hyperparameters, for example, \beta, \tau, etc. 
- The proposed method should be compared more recent weight pruning methods.
- In the derivation, this paper assumed that AWGN, weak correlation, and bounded energy. In the actual ANN, these conditions may not be valid.

### Questions
- It is not clear that this model can actually be put to practical use.
- I wonder why this paper did not apply to database that is widely used for pruning, such as imagenet or cifar.
- It would be better to explain how the entire network structure can be organized efficiently based on the proposed local budget rule.

### Soundness
2

### Presentation
2

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
This paper proposes Budgeted Broadcast, a biologically inspired pruning approach that shifts focus from neuron’s utility to metabolic constraints. Specifically, the paper defines neuron traffic and derives a selectivity-audience balance from constrained entropy maximization. Intuitively, this approach enforces a tradeoff: neurons can speak loudly to a small audience or quietly to a large one. Experiments on ASR, face identification, and synaptic prediction demonstrate that BB demonstrate the effectiveness of the proposed method.

### Strengths
- The proposed pruning criteria that combines long-term activation and the number of fan-out is novel, as far as I know.
- The relationship between neurosience, learning theory and practical AI algorithms is interesting.
- The proposed method is evaluated on a wide-range of tasks including encoder-decoder transformers, and CNNs.

### Weaknesses
- The pruning mechanism appears to rely on sparse activations (e.g., ReLU) to estimate EMA on-rates. However, many state-of-the-art architectures, including modern Transformers, predominantly use smoother and less sparse activations such as GELU or Swish. This may constrain the practical applicability of the method unless the definition of on-rate can be adapted for dense activation functions.
- Comparative analysis pruning methods which uses activation signals (e.g., WANDA) is missing. A discussion and empirical comparison would help clarify how the proposed approach differs in principle and performance when activations drive the pruning signal.
- Lack of empirical evidence for hardware-aligned pattern. While the authors acknowledge the need for structured sparsity projection (Sec 2), empirical validation of this deployment strategy would strengthen the practical claims. Specifically, demonstrating that N:M projection preserves the model performance would be valuable.

### Questions
- Could the authors clarify whether the proposed method can be applied to state-of-the-art models such as large language models, which predominantly use dense activations like GELU? If so, empirical evidence in such settings would strengthen the claim of architectural generality.
- What is the definition of $\bar{p}$ in the figure 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes budgeted broadcast defining a neuron’s cost as traffic. By imposing a budget on this traffic, the proposed method forces neurons to specialize, either be highly active to a few neurons or quietly active to many neurons, but not both. This aims to create sparse networks by preventing any single neuron from dominating information flow, thus protecting potentially important but rarely active neurons.

### Strengths
1. This paper propose a new axis of pruning, cost (traffic), rather than just looking at the importance of each neuron as in existing works. This drives neural networks to develop in to structures that play a more efficient and diverse role through simple traffic budget rules.
2. Effect of protecting low active but highly important neuron: Some neuron which is barely activate but giving output on some rare and important property can be pruned in existing saliency aware pruning. However, in budgeted broadcast, since this type of neurons have low traffic cost, it can be retained with high fan out.
3. The effect of budgeted broadcast to information of the network: The total traffic can be viewed as information that the network process. As a result, constraining the total traffic can be seen as limiting the amount of information which leads the network to learn core information while restricting redundant information.

### Weaknesses
See questions below.

### Questions
1. While cost (traffic) constrained pruning can give an opportunity to low activity and high fan out neuron, in constrast to saliency based pruning, it can ignore the important neuron that can affect to the performance of the neural network. For example, in budgeted broadcast, the case of neurons having high activity and high fan-out cannot be considered due to tradeoff provided by local budget.
2. How can we set optimal budget threshold? Do we have to sweep to find the optimal threshold for every new case?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Budgeted Broadcast introduces a new axis for network efficiency, traffic-based pruning, grounded in biological energy constraints and constrained-entropy theory.
The authors claim to achieve competitive or superior accuracy at matched sparsity across diverse tasks, to improves rare-event performance, and to offers a principled way to link neuron activity and connectivity.

### Strengths
The paper reframes pruning from a parameter-importance problem into a resource-allocation problem, grounded in biology. Instead of deciding which weights are useful, it asks: how much “energy” can each neuron afford to broadcast? This shift is conceptually powerful, it introduces metabolic efficiency as a first-class design principle for artificial networks.
The resulting “traffic budget” $t_i = a_i * k_i$ creates a concrete, interpretable tradeoff between neuron activity (functional demand) and connectivity (structural cost).
This perspective unifies biological realism, efficiency, and interpretability in a way few pruning methods attempt. It doesn’t just compress models - it offers a mechanistic explanation for how efficient representations might self-organize.
The derivation from constrained entropy maximization is elegant and self-consistent.
The selectivity–audience balance $log\frac{1-a_i}{a_i} = \beta k_i$ emerges as a measurable equilibrium - not an arbitrary heuristic.
This gives pruning a predictive theory (you can test whether a trained network obeys the balance line) rather than merely an empirical recipe. The connection between traffic and mutual information provides an intuitive justification: controlling broadcast capacity limits redundant information flow. 
Most sparsification techniques are justified post hoc. Here, the authors build the method top-down from first principles, giving it unusual conceptual coherence.
This balance of theory and practicality makes BB accessible - it’s implementable without modifying training objectives or requiring gradient-based importance metrics.
Overall, BB stands out for its theoretical depth, conceptual originality, and empirical coherence. It combines biological realism with computational pragmatism, providing both a mechanistic theory and a working algorithm. Its main strength is not just performance, but explanatory power - it shows why efficient networks might evolve toward diverse, energy-balanced representations.

### Weaknesses
The theoretical derivation relies on simplifying assumptions - weak activation correlations, bounded edge energies, and Gaussian noise - that seldom hold in deep networks with normalization layers, residual paths, or nonstationary activations. The resulting selectivity-audience balance is therefore plausible but not guaranteed to emerge under realistic nonlinear dynamics.
There is no formal convergence proof showing that the iterative pruning–regrowth process minimizes the proposed entropy-constrained objective.
While the mathematics is elegant, its validity in modern deep architectures remains heuristic. The theoretical link between entropy maximization and the actual mask updates could break in practice.
Critical parameters such as the traffic threshold $\tau$, the inverse-temperature $\beta$, and the refresh interval $\Delta$ require manual tuning per task. There is no adaptive or theoretically principled mechanism for setting these values. Small changes to these hyperparameters can affect sparsity patterns and final accuracy. It undermines the self-organizing spirit of the approach - a rule meant to represent automatic homeostasis still depends on careful manual calibration.
The connection between total traffic $\sum a_ik_i$ and mutual information $I(Z;Y)$ is qualitative, based on a very loose upper bound. No empirical estimates or ablations directly measure how BB affects actual information flow, entropy, or redundancy between layers.
Since the core justification is information efficiency, the lack of empirical validation of that claim leaves a theoretical gap.

Although 4domains are tested, all benchmarks are medium-scale. No large-scale or high-capacity models are evaluated. Reported gains (often 1-3%) are promising but within statistical noise; no significance testing or error bars are provided. The efficiency gains are reported in terms of sparsity, not actual speedups on hardware. Without large-scale or runtime evidence, claims about “efficiency” remain conceptual rather than practical.
The dual pruning mechanisms (SP-in for dendritic, SP-out for axonal pruning) are theoretically motivated, but experiments emphasize only SP-in. There is no detailed analysis of how the two interact, nor whether combining them improves or destabilizes training. The dual-controller mechanism is central to the biological analogy but remains underexplored empirically, weakening the claim of symmetry between input and output homeostasis.

The "natural Top-k reselection" for regrowth is ad hoc and not theoretically tied to the entropy objective. The dynamics of pruning-regrowth cycles are not studied; it’s unclear whether BB reaches a stable equilibrium or oscillates around one. Without analyzing these dynamics, it’s difficult to assert that the pruning process is truly "self-balancing" rather than just stochastic.

Finally, the conceptual clarity is occasionally overshadowed by mathematical compression. The paper proposes unstructured pruning; hence, real-world speedups on GPUs or edge hardware remain minimal. Authors mention potential mapping to structured or N:M sparsity, but this is speculative. Without hardware-aware results, it’s unclear how much of the claimed "efficiency" translates into deployable gains.

### Questions
1) How biologically realistic is the "traffic budget" model $t_i = a_ik_i$?. Does it reflect metabolic constraints observed in neural circuits, or is it mainly a conceptual analogy?
2) Can the selectivity-audience balance $log\frac{1-a_i}{a_i} = \beta k_i$ be derived without strong independence and Gaussian assumptions? How robust is this relationship in deep nonlinear networks?
3) Is there any theoretical or empirical evidence that the pruning–regrowth process converges to the predicted equilibrium, or does it oscillate over time?
4) How is the global budget parameter $\beta$ determined in practice, and could it be learned automatically rather than tuned manually?
5) To what extent does BB improve real computational efficiency (runtime or energy use), given that it currently produces unstructured sparsity?
6) How sensitive is the method to hyperparameters such as the threshold $\tau$, refresh period $\Delta$, and degree limits $d_0, D$?
7) What is the distinct contribution of the traffic rule itself compared to existing dynamic pruning or sparse training methods?
8) Why does BB particularly benefit rare or long-tail features? Is this explicitly enforced by the balance rule or an emergent effect of local regulation?
9) Can the traffic-budget principle be extended to structured or N:M sparsity to achieve hardware-level acceleration without losing its homeostatic behavior?
10) Does controlling traffic actually maximize information efficiency as claimed by the mutual-information bound, and can this be empirically verified?

### Soundness
3

### Presentation
3

### Contribution
3
