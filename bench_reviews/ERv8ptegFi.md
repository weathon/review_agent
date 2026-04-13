## Summary

GPUDrive is a GPU-accelerated, multi-agent driving simulator built on the Madrona Game Engine that achieves over one million simulation steps per second by parallelizing hundreds of environments with hundreds of agents each. The simulator integrates real-world data from the Waymo Open Motion Dataset, supports multiple sensor modalities (including LiDAR and human-like vision cones), provides Gymnasium-compatible Python interfaces, and includes pre-trained reinforcement learning agents that achieve 95% goal-reaching rates on 1000 training scenarios.

## Strengths

- **Exceptional throughput with strong empirical validation.** The paper demonstrates peak throughput of 2.3 million Agent Steps Per Second (ASPS) and introduces the Controlled ASPS metric to account for variable agent counts. Figure 3 shows credible scaling curves comparing against Nocturne (CPU-based) and includes comparisons across consumer (RTX 4080) and datacenter (A100) GPUs. The 200-300× speedup over CPU-based Nocturne is well-supported by the wall-clock training time comparisons in Figure 5.

- **Thoughtful systems engineering for real-world data at scale.** The Bounding Volume Hierarchy for collision detection, Visvalingam-Whyatt polyline decimation achieving 10-15× reduction in road points, and memory allocation proportional to actual rather than maximum agent counts (Section 3.1) demonstrate careful optimization for the unique challenges of driving simulation.

- **Transparent acknowledgment of limitations.** Section 3.2 explicitly documents simulator sharp edges: absence of a lane graph, convex-only collision objects, ~2% unsolvable goals due to dataset labeling errors, and initialization modes that filter out stationary agents. This candor is commendable and helps users understand appropriate use cases.

- **Strong reproducibility and accessibility.** The paper provides Dockerfiles, pre-trained agents, Gymnasium environments for both PyTorch and JAX, and training loops via PufferLib. The claim that experiments can be reproduced on a single A100 in 16 hours is specific and testable.

## Weaknesses

- **No evaluation on held-out test scenarios.** All reported results (95% goal-reaching, Figure 5 training curves, Figure 6 amortized costs) are on training scenarios. The Waymo Open Motion Dataset has official train/test splits, but the paper does not report performance on held-out data. For a simulator intended to enable research on learned driving policies, this leaves generalization entirely unvalidated. The 98% ceiling attributed to mislabeled road edges is based on informal analysis, not empirical measurement.

- **Collision rates and safety metrics are not reported.** Section 3.2 explicitly states "collision penalties... are not used in the experiments reported in this work," and no collision rates are provided. For a simulator explicitly positioned for autonomous driving and safety-critical settings, this omission makes it impossible to assess whether the trained agents are learning safe behaviors or simply reaching goals through aggressive driving. The qualitative description of agents as "extremely aggressive about reaching their goals" (Section 3.2) underscores this concern.

- **The Waymax comparison lacks hardware specification and may be unfair.** Section 4.1 states "we could not run more than 16 environments in parallel due to Out of Memory (OOM) issues" for Waymax, which becomes a key pillar of the scalability comparison. However, the paper does not specify the GPU memory configuration used for Waymax, whether the Waymax authors' recommended settings were followed, or whether memory optimization was attempted. Without this information, readers cannot determine whether the OOM reflects a fundamental limitation of Waymax or a configuration issue.

- **The headline "1 Million FPS" claim relies on ASPS rather than CASPS.** While the paper clearly defines both metrics, the abstract and title emphasize the million-step figure, which counts all agents including parked cars. The more practically relevant metric (CASPS) peaks at ~200,000 for typical WOMD scenarios. This is still impressive, but the 5× gap between headline and learning-relevant throughput should be explicit in the abstract.

- **Multi-agent coordination capabilities are claimed but not demonstrated.** The introduction frames GPUDrive as enabling "multi-agent learning" and "self-play" research, but all experiments use Independent PPO with sparse goal-reaching rewards. There is no experiment testing emergent coordination, negotiation, or game-theoretic properties. The paper would benefit from at least one experiment that demonstrates multi-agent interaction benefits over single-agent baselines or scripted traffic.

## Nice-to-Haves

- **Lane compliance or rule-following metrics.** Although the simulator lacks a formal lane graph, proxy metrics for driving quality (e.g., distance from road centerline) would contextualize the 95% goal-reaching rate against actual driving behavior.

- **Ablation of observation types on learning performance.** Figure 4 shows LiDAR is faster than radial filter, but the paper does not analyze whether learned policy quality differs between observation modalities—critical for researchers choosing configurations.

- **Analysis of failure modes.** Qualitative examination of the 5% of scenarios where agents fail (collision vs. stuck vs. unreachable) would provide actionable insights for algorithm development.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **CARLA multi-agent table entry.** The claim that CARLA should have a multi-agent checkmark is a minor table formatting dispute. CARLA does support multi-agent scenarios via TrafficManager, but the distinction between full multi-agent RL support and scenario scripting is nuanced. This is not a substantive issue.

- **Equation units complaint.** The criticism that Equation (1) has inconsistent units misunderstands the simplified bicycle model formulation—the term in parentheses gives a distance, and the steering command s has units of curvature (rad/m), making the units consistent. This is standard in vehicle dynamics literature.

- **LiDAR vs. radial filter implementation asymmetry as a flaw.** The observation that LiDAR is faster because it's GPU-accelerated while radial filter uses linear scan is accurate but reflects design tradeoffs, not a paper defect. Both observation types are available to users.

- **"Simulator" baseline in Figure 5 label.** The figure could be clearer about what "Simulator" refers to, but this is a minor clarity issue, not a fundamental methodological concern.

## Novel Insights

The amortized per-scene training cost decreasing as the scenario dataset grows (Figure 6) is a genuinely interesting finding: at 1024 scenarios, solving an additional scene costs ~15 seconds compared to minutes per scene when training on fewer scenarios. This suggests strong positive transfer across scenarios and implies that larger training sets may be more efficient overall—counter to the intuition that more scenarios require proportionally more computation. This finding has implications for how researchers should structure large-scale RL training for driving.

## Suggestions

- Add a test-set evaluation using WOMD's official validation split to demonstrate that learned policies generalize beyond training scenarios.
- Report collision rates alongside goal-reaching rates, even for agents trained without collision penalties—this is essential for assessing whether GPUDrive-trained agents are suitable for downstream safety research.
- Specify the GPU memory and configuration used for all baselines (especially Waymax) and, if possible, attempt memory optimization or report the maximum batch size achieved before OOM.