## Summary
ELMUR is a transformer architecture augmented with layer-local external memory tracks, bidirectional token-memory cross-attention, and an LRU-based update rule. It demonstrates exceptional long-term retention, solving a synthetic T-Maze task with corridors up to one million steps while using a context window of only 10, and significantly improves performance on visual robotic manipulation and diverse memory-intensive benchmarks.

## Strengths
- **Demonstrates extreme long-horizon retention.** ELMUR achieves 100% success on the T-Maze task with inference corridors up to 1 million steps, extending effective memory horizons by 100,000x beyond its attention window (Figure 3). This directly validates the core claim of scalable long-term memory.
- **Strong, broad empirical gains.** On the MIKASA-Robo benchmark of sparse-reward visual manipulation tasks, ELMUR nearly doubles the aggregate success rate of the strongest prior method (RATE) and ranks first on 21 of 23 tasks with non-zero success (Table 1, Appendix Table 8). It also achieves the top aggregate score on the diverse POPGym-48 suite.
- **Rigorous mechanistic analysis.** The paper provides a theoretical analysis of memory retention (exponential forgetting, half-life) and stability (Proposition 1, 2). Extensive appendix analyses—including memory probing, PCA visualizations, update patterns, and attention maps (Figures 9-15)—convincingly show that performance gains stem from functional use of the external memory, not simply increased capacity.

## Weaknesses
- **Missing comparisons to key architectural baselines.** While compared to RATE and DT, the paper does not benchmark against other prominent long-context or memory-augmented transformers (e.g., Transformer-XL, Compressive Transformer, Memorizing Transformer) or state-space models like Mamba. This omission makes it difficult to fully situate ELMUR's novelty and performance within the architectural landscape.
- **Limited analysis of failure cases and task-type sensitivity.** ELMUR does not win on 24 of the 48 POPGym tasks. The paper does not analyze these cases—whether they are reactive tasks where memory is less needed, or if ELMUR has specific weaknesses—which would provide a more nuanced understanding of its capabilities and limitations.
- **Detached memory limits gradient-based credit assignment for writing.** Training uses detached memory (`sg(m^{i-1})`) between segments (Algorithm 1), preventing gradients from flowing through memory across segment boundaries. This design choice stabilizes training but may restrict the model's ability to learn *how* to write to memory based on very long-term credit assignment. The implications of this are not sufficiently discussed.
- **Exclusive focus on offline imitation learning.** All experiments are conducted in an offline IL setting. The method's efficacy and sample efficiency in online reinforcement learning—where exploration and long-horizon credit assignment are fundamental—remain untested, limiting assessment of its broader impact for RL.

## Nice-to-Haves
- A more detailed computational complexity analysis comparing FLOPs, memory footprint, and latency against baselines as context length and memory size scale, to better substantiate claims of efficiency.
- Exploration of adaptive or learned memory update policies (e.g., a learned blending factor λ) instead of the fixed LRU rule, which could be a natural extension.
- Testing on environments with multiple, interleaving long-term dependencies (beyond remembering a single cue) to more rigorously stress-test memory capacity and management.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **"The paper does not analyze the actual content and usage of memory slots."** → Extensive memory probing, PCA, and attention analyses in Appendix A.9-A.11 directly address this.
- **"The paper should evaluate on established long-horizon benchmarks like D4RL."** → Appendix A.5 (Table 4) includes results on D4RL MuJoCo tasks.
- **"The theoretical analysis is not a major advance."** → While straightforward, it provides necessary formal grounding for the empirical claims and is kept as a strength.
- **"The use of MoE-FFN is not justified."** → The ablation (Table 3) shows it is not essential for the memory mechanism; this is noted but not a core flaw.
- **Formatting nitpicks and generic strengths (e.g., "well-written")** are removed per the instructions.

## Novel Insights
The paper's most novel insight is that a simple, structured, layer-local external memory with an LRU-based update rule can enable transformers to retain information over horizons up to 100,000 times longer than their native attention window, solving extreme long-horizon POMDPs that defeat standard architectures. The combination of bidirectional cross-attention (mem2tok/tok2mem) and temporal grounding via relative bias creates a coherent read-write interface that, coupled with the LRU policy, yields bounded yet persistent storage. The extensive appendix analyses further reveal that the model learns to perform sparse, one-shot writes of task-critical variables into dedicated memory slots and preserves them with high fidelity until retrieval.

## Suggestions
- Add comparisons to Transformer-XL and/or other contemporary long-context architectures on the same benchmarks to clarify ELMUR's relative advantages.
- Include a brief analysis of the POPGym tasks where ELMUR does not achieve top performance, discussing whether the shortfall relates to task type (e.g., reactive vs. memory-intensive).
- In the limitations section, expand the discussion of the implications of using detached memory during training and the method's current restriction to offline IL.