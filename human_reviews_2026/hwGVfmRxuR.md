# Principled Latent Diffusion for Graphs via Laplacian Autoencoders

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Graph diffusion models achieve state-of-the-art performance in graph generation but suffer from quadratic complexity in the number of nodes---and much of their capacity is wasted modeling the absence of edges in sparse graphs. Inspired by latent diffusion in other modalities---a natural idea is to compress graphs into a low-dimensional latent space and perform diffusion there. However, unlike images or text, graph generation demands near-lossless reconstruction, since even a single error in decoding an adjacency matrix can invalidate the entire sample. This challenge has remained largely unaddressed. We propose a latent graph diffusion framework that directly overcomes these obstacles. A permutation-equivariant autoencoder maps each node into a fixed-dimensional embedding from which the full adjacency is provably recoverable, enabling near-lossless reconstruction for both undirected graphs and DAGs. The latent representation scales linearly with the number of nodes---eliminating the quadratic bottleneck and making it feasible to train larger and more expressive models. In this latent space, we train a Diffusion Transformer with flow matching, enabling efficient and expressive graph generation. Our approach is competitive results against prior graph diffusion models, while achieving speed-up ranging from $50\times$ to $1000\times$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This article proposes a nearly lossless latent graph diffusion method in view of the high complexity limitations of the existing graph diffusion models due to the discrete paradigm. Specifically, considering the strict requirements of graph data for reconstructing encoder-decoder, the authors proposed a Laplacian autoencoder, which was proven to be able to compress graph data into low-dimensional vectors nearly lossless. Then, the author placed it in the framework of stream matching and trained it using DIT. It is worth noting that this method has also been extended to the research in the field of directed graphs generation.

### Strengths
1.I recognize the contribution of this paper in the field of graph generation. Unlike the mature autoencoders in the field of computer vision, the field of graph learning still lacks such an effective lossless encoding method.

2.This paper extends the framework of graph generation to directed graphs, which is of great significance. Previous work mainly focused on molecular data or synthetic undirected graph structure data. Generating directed graphs is equally important.

3.The paper provides theoretical guarantees for the method.

4.The paper is well-written and the introduction of the motivation is very convincing.

### Weaknesses
1.I think that compared with the design in graph generation, the author's contribution mainly focuses on a powerful autoencoder. I think this is very important, but there is a lack of sufficient research and innovation on the generation method. 

2.Regarding the autoencoder, the author achieved nearly lossless results in the experiment. However, as the author mentioned, their assumption is that the node order of the graph remains unchanged, but this can cause problems in some specific scenarios (tree-like structure). Their processing method is to use wl-test to color the nodes. I think this point is worthy of exploration and analysis. For instance, if appropriate position encodings are assigned to the graph, is there a possibility of a solution?

3.One concern about the autoencoder is that although this method can be extended to large graphs by virtue of its complexity advantage, the number of samples in large graphs is equally scarce. Then, when only a limited number of large graphs are used for training, can lossless compression still be achieved? I suggest that the author could attempt to train on the sbm datasets(you can generate by yourself) using only 10 graphs each containing 10,000 nodes to train and verify the reconstruction performance of the autoencoder on them.


4.Another concern is that this method achieves a lossless effect on the encoder, but it does not have a significant advantage in terms of generation effect. However, the author lacks profound analysis and discussion. Is this due to the error of the generative model or the inherent distribution difference brought about by the training set and the test set?

5.Furthermore, it would be great to be able to see the comparison of the effects of different generation strategies. For instance, the effect differences between ddpm and flow matching, as well as the influence of the selection of different compression dimensions on the reconstruction and generation effects. I hope the author can increase experiments on different diffusion paradigms and, most importantly, compare the impact of choices in different dimensions.

### Questions
All my questions are shown in Weakness. I will consider improving my score if the authors can answer my questions.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a latent diffusion framework for graph generation that uses an autoencoder with Laplacian based adjacency identifying positional encodings. The decoder applies bilinear attention style scoring and a row wise DeepSet, and the approach extends to DAGs via a magnetic Laplacian. The authors claim that after freezing the autoencoder, a Diffusion Transformer is trained in latent space with conditional flow matching, which yields mostly linear per node complexity with only the final adjacency decoding quadratic. Experiments report near lossless reconstruction on several datasets, competitive generation quality.

### Strengths
- The experiments show competitive reconstruction performance while substantially reducing the runtime.
- Efficient yet near-lossless graph generation is an important topic.

### Weaknesses
- Since the model uses only the lowest Laplacian eigenvalues/eigenvectors (rather than the full spectrum), it leans toward global/low-frequency structure and may miss high-frequency details. It might potentially hurt the exact reconstruction of fine structures.
- The approach depends on computing Laplacian eigenvectors for positional encodings, which is costly and can dominate preprocessing on large graphs.
- The promise of near-lossless reconstruction appears to rely on having sufficiently large training corpora of graphs, and performance can degrade on smaller or more complex datasets.

### Questions
- Recent graph representation learning works [2] suggest that emphasizing low-frequency components can improve downstream representations (e.g., node classification), so selecting the k smallest eigenpairs is often reasonable in that context. However, for generation, where accurately reconstructing fine structural details could be crucial, restricting to the lowest eigenpairs may suppress high-frequency information. If I am wrong, please correct me.
- In (e.g., line 174), writing $(\phi:\mathbb{R}^2\to\mathbb{R}^d)$ can suggest the input itself is 2-dimensional, whereas the actual input per node is an $(n\times 2)$ table formed by concatenating two vectors row-wise (each row is a 2-vector ($(U_{ij},\lambda_j{+}\epsilon_j))$). Could you explicitly state that $(\phi)$ is applied row-wise and annotate the shapes (input $(n\times 2)$ → output ($n\times d)$) to avoid confusion?
-  In the notation section, node features $X$ are defined as discrete labels. Does the current framework support graphs with dense real-valued node attributes (e.g., Cora dataset $n\times d$ features)? 
- Several recent works also build graph learning/generation models from spectral perspectives (e.g., using the smallest (k) eigenpairs). Could the authors discuss the key differences between your design and [1,2]?

[1] "Generating Graphs via Spectral Diffusion." The Thirteenth International Conference on Learning Representations, 2025.

[2] "SDMG: Smoothing Your Diffusion Models for Powerful Graph Representation Learning." The Forty-second International Conference on Machine Learning, 2025.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
ML graph generation models suffer from poor scalability due to quadratic cost of graph representation and hardness to compress such graphs into revertible latent representation. LG-VAE poses itself as a variational autoencoder that uses the graph laplacian spectrum to bring provable reversibility and enable fast graph generation. LG-VAE is shown to generate graphs up to 1000x faster than state-of-the-art while retaining generative quality.

### Strengths
- The proposed method brings outstanding improvements under inference cost compared to the state-of-the-art
- The proposed method brings comparable generative performance than the state-of-the-art 
- The proposed method covers an extensive set of graphs, including attributed, non-attributed, and directed acyclic graphs

### Weaknesses
- The main contribution of the paper is scalable graph generation. However, the experiments mostly cover the generation of small graphs.
- How evaluation is conducted is partly unclear, making the validity of the paper’s result debatable. For instance, it is not clear whether LG-- VAE has been tested with the same batch size as the baselines, a detail that could invalidate a significant part of the paper contribution.
- Experiments on directed graphs seem to be insufficient. First, LG-VAE is tested on one dataset only. Secondly, inference time of the SOTA baseline is inferred by their paper, meaning that it has also been run on a different hardware.
- The experiment section could cover more ablation studies to properly assess the impact of each of the proposed components.
- Minor: methodology figure could be improved for clarity and making LG-VAE’s components easier to understand.

### Questions
Altogether, I think the paper makes a reasonable amount of contributions and that the experiments showcase that LG-VAE is a superior choice for graph generation with diffusion models. However, I have concerns about how the evaluation of LG-VAE has been performed. Here are my questions for the authors:

- The main contribution of LG-VAE is scalable generation. However, the experiments mostly cover datasets of small graphs. Could the authors perform experiments on datasets with larger graphs? Proteins [1] could be a good choice also used in several papers in this field. If the authors have preference for any other dataset with large graphs over Proteins would also be fine.
- Have the baselines and LG-VAE tested with the same batch size? If not, what batch sizes have been used in the experiments? This is not clearly explained in the paper content. From what I understand, the reason why LG-VAE is much faster than the SOTA in Table 1 is because the inference has been run with one batch only for LG-VAE but not for the other models. If this is the case could the author measure memory cost and time cost of LG-VAE and the baselines with the same batch sizes?
- The paper would benefit from more ablation studies to assess the impact of each of the components. For instance, experiments to test the impact of the choice of k as the number of used eigenvectors for encoding on computational costs and generation quality. Could the authors provide such experiments?
- Could the authors run the experiments of Table 5 on the same hardware to verify the computation efficiency improvement wrt Directo?
- Could the authors find at least one more DAG dataset to perform experiments on?
- Fu et al. [2] already provided a method for graph diffusion with cost linear in the number of nodes. The paper cites this work but does not compare LG-VAE performance with this model. Could the authors compare LG-VAE computation time and generation quality with the one of HypDiff?

[1] Dobson, P.D. and Doig, A.J., 2003. Distinguishing enzyme structures from non-enzymes without alignments. Journal of molecular biology, 330(4), pp.771-783.
[2] Fu, X., Gao, Y., Wei, Y., Sun, Q., Peng, H., Li, J. and Li, X., 2024. Hyperbolic geometric latent diffusion model for graph generation. arXiv preprint arXiv:2405.03188.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Graph generation suffers from quadratic complexity in the number of nodes stemming from the adjacency matrix, which, however, contains mostly zeros for most datasets. This hinders the generation of large graphs. The paper proposes a permutation-equivariant autoencoder that maps each node into a fixed-size latent space, leveraging the graph Laplacian spectrum to reduce complexity. The latents are then used to train a generic diffusion model.

### Strengths
- Strong generation speedup compared to state-of-the-art baselines.
- Quality-wise, the method achieves the best or second-best performance against the baselines.
- Experiments on a large variety of different types of graphs, both synthetic and real-world, e.g., DAG, molecules, planar, trees.
- Good reconstruction results.

### Weaknesses
- Motivation stems from limitations affecting the generation of large graphs, but evaluation only covers standard graph benchmark datasets of the same size as tackled by related work. Memory and time might not be the only limits when generating graphs of larger sizes. It is unclear if quality can be maintained.
- Unclear how comparable the baseline results are on inference time: e.g., how optimal/equal were the used batch sizes and if the same hardware was used. Especially DAG runtimes are inferred from the original authors' paper, which might not have been using the same setup.
- No ablation studies to better highlight the impact of the different components.

Minor:
- Table 2: Sample acc. column: wrong row is highlighted in bold

### Questions
- How is the generation quality on larger graphs?
- A sensitivity analysis on the number of eigenvectors used
- What batch size is used, and what hardware is used for inference (appendix mentions 2x L40 for training)?
- How similar is the hw form Directo? Can the experiments be run on the same hardware?

### Soundness
2

### Presentation
3

### Contribution
3
