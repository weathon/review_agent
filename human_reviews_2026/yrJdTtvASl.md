# Diffusion-Enhanced GFlowNet for Solving Vehicle Routing Problems

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
Traditional neural solvers for solving vehicle routing problems (VRPs) often suffer from limited solution diversity, motivating the development of Generative Flow Network (GFlowNet)–based models. However, the effectiveness of these models is frequently constrained by insufficient flow expansion in high-reward regions, limiting their ability to distribute probability across promising solution routes, the deeper exploration could yield superior results. Diffusion models, in contrast, provide stronger structural guidance for exploration. These two paradigms are naturally complementary: GFlowNet can supply edge-level signals for diffusion to embed, while diffusion can guide broader exploration. Leveraging this synergy, we propose Diffusion-Enhanced GFlowNet (DEG), a novel framework that integrates GFlowNet with diffusion model to encourage richer flow expansion toward high-reward regions and derive higher-quality solutions. Specifically, DEG exploits GFlowNet’s inherent diversity to generate edge-specific backward signals, applies the stochastic noise schedule of diffusion to perturb these signals, and then denoises them within the GFlowNet paradigm. To further improve scalability, we introduce a specialized decoder capable of dynamically adapting to diverse problem scales. Extensive experimental evaluations on synthetic and real-world datasets, including instances with up to 10,000 nodes, demonstrate that DEG consistently achieves favorable performance compared to baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Diffusion-Enhanced GFlowNet (DEG) to solve vehicle routing problems. Specifically, its key idea is to integrate GFlowNet with diffusion models to facilitate flow expansion in high-reward regions. Meanwhile, it introduces a simple but effective scale adapter for instances to achieve better generalization performance on scales. Experiments on synthetic data and real-world dataset (TSPLib and CVRPLib) demonstrate that DEG can achieve competitive performance over previous methods, especially on large-scale instances.

### Strengths
1.	The integration of GFlowNet with diffusion models seems novel. The idea of utilizing the exploration ability of diffusion models to facilitate effective flow expansion for GFlowNet seems to be technically sound.
2.	The proposed method DEG demonstrates good performance, especially on large-scale instances. The ablation studies are clear and sufficient.

### Weaknesses
1.	As discussed in the conclusion section of this paper, the performance of DEG on small-scale still fall short of AR methods like LEHD, which limits its practical application where the instance scale can be various.
2.	While the empirical results are good, a more rigorous theoretical analysis of how the diffusion process facilitates exploration, and why edge-specific signals outperform trajectory-level signals would strengthen the work a lot.

### Questions
Is the proposed graph-scale adapter available for other methods to enhance their performance? Can you provide some discussions on it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Diffusion-Enhanced GFlowNet (DEG), a framework that injects a diffusion process into GFlowNet training to encourage flow expansion toward high-reward regions for TSP/CVRP. Concretely, the method constructs edge-specific backward signals by aggregating trajectory rewards, normalizes them (z-score + Gaussian CDF), perturbs them with a diffusion noise schedule and then denoises within the GFlowNet paradigm by using the noised backward policy in the trajectory balance loss. The authors also introduce a Graph-Scale Adapter (GSA) to align spatial statistics across train/test graph sizes, claiming improved scalability up to 10k nodes. Empirically, DEG reports consistent improvements over GFlowNet- and diffusion-based baselines on synthetic TSP/CVRP and shows favorable gaps on TSPLib/CVRPLib-style instances, often with strong runtime.

### Strengths
1. **Methodological clarity**. The core mechanism—edge-level reward estimation (Eq. 3–5), diffusion corruption (Eq. 6), and its integration into TB training (Eq. 10–12)—is described with reasonable detail, and the learning-target interpretation is intuitively argued.

2. **Originality**. Using GFlowNet-induced edge-specific signals as the substrate for diffusion in VRP appears new and interesting; most prior diffusion solvers are TSP-only and supervised, whereas DEG claims a more general, reward-driven integration.

3. **Empirics**. Tables indicate strong results across sizes, including large-scale TSP up to 10k nodes with favorable runtime, and competitive CVRP gaps vs LEHD/POMO and GFlowNet baselines. However, some comparisons (e.g., CVRPLib subsets and enormous gaps for some baselines) merit closer scrutiny and clarified protocols (datasets, post-processing, hardware).

### Weaknesses
1. **Lack of empirical, quantitative evidence for the proposed method**. The paper's central and most interesting claim is that the diffusion-based noise schedule creates a natural learning curriculum, shifting the learning objective from exploitation (proportional to R(τ)²) to exploration (proportional to R(τ)). This idea, however, hinges on a critical assumption presented in lines 222-224: "Early in training... the noised backward policy ... closely approximates the reward distribution." While this assumption is intuitive, the paper does not provide any empirical or quantitative evidence to validate it. The entire justification for using a diffusion model rests on this premise, but it is left as a claim rather than a verified fact. To significantly strengthen the paper's contribution, the authors should provide evidence to support this. 

2. **Single-scale training**.  All models are trained only on 100-node synthetic instances, and scalability to larger graphs is achieved mainly via GSA rescaling at test time. This makes it unclear whether the method itself is scale-robust, or whether the rescaling step is doing most of the work. The authors should show results where models are also trained on other node sizes and evaluated with/without rescaling, to demonstrate that performance is not tied to the 100-node training setup.

### Questions
1. Could you provide empirical or quantitative evidence to support the key assumption stated in lines 222–224 that “early in training, the noised backward policy closely approximates the reward distribution”?

2. What is the computational and memory overhead of maintaining and denoising the edge-level backward maps compared to a vanilla GFlowNet? Is the added complexity justified by the performance gain across all problem sizes, or only for large graphs?

3. Since GSA addresses training–inference scale mismatch and not a DEG-specific issue, it seems reusable. Can you show its impact when applied to at least other VRP baselines?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Diffusion-Enhanced GFlowNet (DEG), a novel framework that synergistically integrates GFlowNet with diffusion models to address the challenge of insufficient flow expansion in high-reward regions for solving Vehicle Routing Problems (VRPs). The core innovation lies in using GFlowNet to generate edge-specific backward signals, which are then perturbed and denoised via a diffusion process to guide richer exploration. The proposed method, complemented by a Graph-Scale Adapter for enhanced scalability.

### Strengths
1.	The idea of injecting a diffusion-like noise mechanism into GFlowNet training is novel and interesting. It effectively enhances the model's exploration ability within high-reward regions, addressing a key limitation of existing GFlowNet-based solvers.

2.	Extensive experiments on large-scale TSP and CVRP datasets convincingly demonstrate the superior performance and strong generalization ability of DEG. The results on very large CVRP instances (N ≥ 5000) are particularly impressive, showing that the method can even outperform the well-known LKH solver.

### Weaknesses
1.	The experiments lack a comprehensive evaluation on small-scale problems (N ≤ 500). While tackling large-scale instances is a crucial goal for neural solvers, demonstrating robust performance across diverse problem scales is also important, especially for a method that aims to improve general training framework.

2.	The application domain is currently limited to routing problems (TSP and CVRP). Given that DEG appears to be a general training framework for neural combinatorial optimization, evaluating its effectiveness on other related problems (e.g., scheduling, maximum independent set) would significantly broaden the paper's impact.

### Questions
The presented experiments primarily focus on generalization performance. I am curious about the fundamental optimization ability of DEG compared to the original GFlowNet method, standard Reinforcement Learning (RL) methods, and newly developed preference optimization methods [1]. To provide a clearer comparison of learning efficiency, could the authors conduct experiments on a fixed problem scale and distribution while aligning the number of training instances/samples across all compared methods?

[1] Preference Optimization for Combinatorial Optimization Problems. ICML 2025.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Diffusion Enhanced GFlowNet, a combination of GFlowNet with diffusion technique, to encourage exploration of higher reward regions. Although DEG achieves promising results on several dataset, it still lags behind more recent baselines in terms of speed and performance.

### Strengths
- The idea of using diffusion to encourage exploration in GFlowNet is interesting.

### Weaknesses
**Major issues**

1. The paper is hard to read for people without a solid background in GFlowNet (like me). I struggled to understand how the network is trained and how the solution is constructed.
2. The main motivation is not properly backed by theoretical and empirical evidence. Questions:
- Adding noise to the backward policy changes the distribution. How can we ensure that DEG samples from the target distribution?
- I would like to see more theoretical and empirical evidence for the claim ''DEG improves the diversity''. The second part of section 4.1 only offers a vague intuition on how diffusion might encourage exploration. The experiment also did not illustrate this point.
3. The performance is weak compared to more recent baselines. For example, [1] achieved much stronger results at even faster speed.

**Minor issues**

- Missing citation to original work on diffusion model (Sohl-Dickstein et al., 2015) (line 60).

[1] Learning to Reduce Search Space for Generalizable Neural Routing Solver. Changliang Zhou, Xi Lin, Zhenkun Wang, Qingfu Zhang

### Questions
Please address the weaknesses mentioned above.

### Soundness
2

### Presentation
2

### Contribution
2
