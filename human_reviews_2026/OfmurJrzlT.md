# Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Crystal property prediction, governed by quantum mechanical principles, is computationally prohibitive to solve exactly for large many-body systems using traditional density functional theory. While machine learning models have emerged as efficient approximations for large-scale applications, their performance is strongly influenced by the choice of atomic representation. Although modern graph-based approaches have progressively incorporated more structural information, they often fail to capture long-range atomic interactions due to finite receptive fields and local encoding schemes. This limitation leads to distinct crystals being mapped to identical representations, hindering accurate property prediction. To address this, we introduce PRDNet that leverages unique reciprocal-space diffraction besides graph representations. To enhance sensitivity to elemental and environmental variations, we employ a data-driven pseudo-particle to generate a synthetic diffraction pattern. PRDNet ensures full invariance to crystallographic symmetries. Extensive experiments are conducted on Materials Project, JARVIS-DFT, and MatBench, demonstrating that the proposed model achieves state-of-the-art performance. The code is openly available at \url{https://github.com/Bin-Cao/PRDNet}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a new architecture for processing crystalline graphs. The architecture makes use of Fourier/diffraction space features to enable capturing long-range dependencies. Empirical results validate this.

### Strengths
- The paper provides a principled approach to modeling crystalline materials in ML. The use of diffraction-space features is well-motivated from the hundred years of crystallography which have relied on such features.
- Empirical results show the improvement over many strong baselines.

### Weaknesses
see questions

### Questions
- Is it possible to add on your diffraction modules to the existing GNNs to provide a fairer ablation? I'm interested to see if incorporating this information to the GNNs would improve performance. As a baseline, you could just add some more generic parameters (e.g., equivariant layers) to the GNNs and compare to that.
- Can you match the number of parameters in evaluations? Basically, if you can show that for a given number of parameters, it is better to use diffraction modules (or some mix of diffraction modules and attention layers), the results would be convincing.
- How is the diffraction module different from the Fourier transform employed in [1]? I think yours is probably more elaborate, but you should take some paragraphs to explain this, since Fourier features have been used for a while now.
- You could consider generative tasks, although this could be tight given the rebuttal timeline. I don't expect such experiments to be done within the rebuttal period, but if they were to be done, I think the audience would be more interested.

[1] Jiao, Rui, et al. "Crystal structure prediction by joint equivariant diffusion." Advances in Neural Information Processing Systems 36 (2023): 17464-17497.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
PRDNet augments a standard crystal graph encoder with a reciprocal-space “pseudo-particle diffraction” pathway. Instead of fixed X-ray form factors that can’t tell identical elements in different local environments apart, the model learns per-atom form factors that depend on |Q|, element type, and a graph-encoded local environment, making the diffraction signature itself discriminative for crystal property prediction. Across Materials Project, JARVIS-DFT, and Matbench benchmarks, PRDNet delivers superior or competitive performance.

### Strengths
1. Reciprocal-space encoding captures long-range periodic interactions that finite-receptive-field graph encoders miss, which can otherwise collapse distinct crystals to the same embedding.
2. It’s SOTA/competitive across MP, JARVIS-DFT, and multiple Matbench tasks, with best-in-table errors/accuracy on MP and strong wins on JARVIS/MB.

### Weaknesses
Majors:
1. The paper discusses ReGNet/ReciNet and critiques its reciprocal block, but it does not cite or position against other highly related long-range approaches that also reason in reciprocal/Ewald space (e.g., EwaldMP [1] and PotNet [2]). Please add these citations and discuss and compare to them in the main text.
2. You evaluate on your own MP/JARVIS selections and Matbench tasks, but we cannot directly compare against the MP/JARVIS setup used by all your other baselines, e.g., Crystalformer [3]. Please add a table on that benchmark or clearly justify why not.
3. The message-passing and “reciprocal head + fusion” pipeline reads very close to ReGNet/ReciNet [4] as cited. The paper argues ReciNet’s reciprocal block omits key physical dependencies, but eventual formulation and implementation of this work is incremental to theirs. Please show the ReciNet's result over you benchmark to justify your novelty and contribution.

Minors:
1. Use "long-range" & "short-range" instead of "long-term" & "short-term".
2. Use "Matformer", "Crystalformer" and "Crystalframer" in Appendix D.
3. Line 161: "on" -> "in the"
4. Line 205: "task" -> "tasks"
5. Line 208: Remove ")"


[1] Ewald-based Long-Range Message Passing for Molecular Graphs. Arthur Kosmala, Johannes Gasteiger, Nicholas Gao, Stephan Günnemann.

[2] Efficient Approximations of Complete Interatomic Potentials for Crystal Property Prediction. Yuchao Lin, Keqiang Yan, Youzhi Luo, Yi Liu, Xiaoning Qian, Shuiwang Ji.

[3] Crystalformer: Infinitely Connected Attention for Periodic Structure Encoding. Tatsunori Taniai, Ryo Igarashi, Yuta Suzuki, Naoya Chiba, Kotaro Saito, Yoshitaka Ushiku, Kanta Ono.

[4] ReciNet: Reciprocal Space-Aware Long-Range Modeling for Crystalline Property Prediction. Jianan Nie, Peiyao Xiao, Kaiyi Ji, Peng Gao.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes PRDNet, a novel approach for crystal material property prediction. PRDNet integrates graph representation of materials with a learned diffraction module, which is invariant to crystallographic symmetries. Benchmark experiments show that PRDNet achieves better property prediction accuracy than baselines.

### Strengths
- Propose a novel idea of introducing learnable diffraction module into the crystal graph neural network model. The good property of achieving crystallographic symmetries and discriminating distinct crystal structures is very useful.
- Achieve state-of-the-art performance on benchmark datasets.
- Generally good and well-organized paper writing.

### Weaknesses
- The idea of using learnable diffraction module is good, but authors could clarify more clearly about its advantages over existing approaches. Satisfying crystallographic symmetries is also focus of many existing approaches, but they may fail to distinct crystal structures (as mentioned in line 87). So authors are encouraged to discuss how existing approaches fail to do it and why the proposed learnable diffraction module succeed.
- While the proposed PRDNet adopts a graph transformer architecture as backbone, could the learnable diffraction module be integrated into other backbone models, or even existing crystal material property prediction model (e.g., CGCNN and Matformer)? Authors are encouraged to discuss this generalizability and conduct experiments about integrating diffraction module into more different model architectures.

### Questions
No additional questions.

### Soundness
3

### Presentation
3

### Contribution
3
