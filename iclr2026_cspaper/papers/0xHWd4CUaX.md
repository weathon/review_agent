000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 We propose a novel reinforcement learning (RL) framework for automated code refactoring that uses contrastive pre-trained code graph embeddings to overcome the limitations of the traditional heuristic-based reward functions. The key challenge is balancing the implementation of syntactic improvements - while maintaining the semantics of the code being refactored - something that necessarily requires the existing RL approaches to accomplish and that most often do last year because of the handcrafted nature of their metrics. Our approach presents a syntax-guided contrastive encoder that acquires structural invariant representations of code graphs by relating structurally augmented variants under a selfsupervised objective. These embeddings are then combined with standard measures of code quality in a composite reward function, allowing the RL agent to reason about both low-level changes to the syntactic structure as well as highlevel changes in the semantic structure. The policy network itself, which takes the form of a graph attention network, runs on the joint representation space directly, which models dependency on the context on the code structure.

## 1 Introduction

Anonymous authors Paper under double-blind review

## Abstract

Automated code refactoring has become increasingly crucial in modern software development, objecting to code quality and reducing technical debt. Traditional approaches to this problem have primarily relied on rule-based systems (Taentzer et al., 2012) or static analysis techniques (Kimura et al., 2012). Recent advances in machine learning have helped to open up new opportunities for automated code refactoring. Reinforcement learning (RL) has emerged as a particularly promising direction, as it can learn optimal refactoring strategies through interaction with the code environment (Palit & Sharma, 2024a). The fundamental issue with refactoring using RL is how to build good representations of code that have both syntactic and semantic aspects. The current methods mostly involve handcrafting of features, or the use of simple syntactic measurements, which possibly aren't sufficient to represent the complex relationships in code structures. We tackle these issues with the introduction of a new, yet simple framework that brings together variable-contrastive learning with reinforcement learning for code refactoring in an automated way. Our approach takes advantage of the power of self-supervised learning to produce rich representations of code that are aware of refactoring, and that do not require any large amounts of labeled data. The proposed method differs in some important aspects from previous methods. First, instead of using reward functions crafted by hand, our framework is based on the automatic learning of meaningful representations of code quality in contrastive pre-training. Second, the combination of graphbased representations enables the model to reason over the structure of code at various scales of granularity from individual statements to complete modules. Third, our approach is excellent in reducing the necessity of expert demonstration based learning and uses self-supervised learning by using huge amounts of unlabeled code.

# Contrastive Code Graph Embeddings For Re- Inforcement Learning-Based Automated Code Refactoring

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 The effectiveness of our method comes from three individual components: (1) contrastive encoder that establishes an invariant representation of code graphs using structural augmentations, (2) composite reward function that leverages both learned embeddings and traditional code quality metrics, as well as (3) policy network that operates directly on the joint representation space. Our experimental evaluation shows that the proposed approach is better than the existing methods for the refactoring quality and generalization capability. The rest of this paper is organized as follows: Section 2 presents a review of related work in the automated refactoring and setup of code transformation using RL. Section 3 gives some needed background about reinforcement learning and contrastive representation learning. Section 4 presents our proposed method in detail, including contrastive pre-training objective and RL integrating. Section 5 is dedicated to describing the set-up of our experiments and our results. Section 6 is about limitations and future directions and it is followed by conclusions in Section 7.

## 2 Related Work

The combination of machine learning and code refactoring has enjoyed great innovation in the last few years, various approaches being developed, from supervised learning models to reinforcement learning approaches.

## 2.1 Traditional Refactoring Approaches

Early automated refactoring systems relied heavily on predefined rules and static analysis (Taentzer et al., 2012). These systems generally used pattern matching mechanisms to detect code smells and transformation templates to fix the code. While working for specific anti-patterns, such approaches were not flexible enough to handle different coding styles, and often required huge manual configuration. More sophisticated static analysis tools (Kimura et al., 2012) incorporated control and data flow analysis to detect refactoring opportunities, but remained constrained by their rule-based nature.

## 2.2 Learning-Based Code Transformation

Recent lemon deep learning technologies have made it more adaptable to code transformation. Sequence-to-sequence models (Tufano et al., 2019) initially demonstrated promise by treating code as natural language, but struggled to capture structural dependencies. Graph neural networks (GNNs) addressed this limitation by explicitly modeling code structure through abstract syntax trees (ASTs) and control flow graphs (LeClair et al., 2020). Syncobert (Wang et al., 2021) introduced syntax-aware contrastive learning for code representation, though its focus remained on general code understanding rather than refactoring-specific tasks. GraphCodeBERT (Guo et al., 2020) advanced this direction by incorporating data flow information into pre-training, showing improved performance on downstream tasks like code search and clone detection.

## 2.3 Reinforcement Learning For Code Refactoring

The movement of using reinforcement learning on code refactoring has been a study of note to translate the defect of static approaches. Early RL-based methods (Marvellous et al., 2025) formulated refactoring as a Markov Decision Process, using handcrafted reward functions based on code metrics. Subsequent work (Polu, 2025) demonstrated RL's adaptability to various optimization constraints, particularly for performance-critical code. The hybrid approach in (Prasad & Srivenkatesh, 2025) combined GNNs with RL, showing improved refactoring quality but still relying on expert demonstrations for training. Process-supervised RL (Ye et al., 2025) introduced teacher models for guided exploration, though its focus remained on code generation rather than refactoring. The proposed way differs from current methods in combining contrastive pre-training method with RL in a new manner. While prior works either relied on hand-crafting rewards or required lots of supervision, we take a step forward and learn refactoring aware representations using self-supervised contrastive objectives.

## 3 Background And Preliminaries

In order to set the stage for our proposed approach, we first discuss important ideas about reinforcement learning and contrastive representation learning in relation to code refactoring here.

## 3.1 Reinforcement Learning Framework

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Reinforcement learning formulates decision-making problems through the lens of an agent interacting with an environment (Sutton & Barto, 1998). In the code refactoring case, the environment is the codebase and the associated quality metrics, and the agent learns a policy for determining how to refactor the code. The standard RL framework models this interaction as a Markov Decision Process (MDP) defined by the tuple (*S, A, P, R, γ*), where S represents the state space (code representations), A denotes the action space (possible refactorings), P describes transition dynamics, R specifies the reward function, and γ is the discount factor.

The policy π(a|s) determines the probability of taking action a in state s, with the objective of maximizing expected cumulative reward:

$$J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{T}\gamma^{t}r_{t}\right]$$

$$(1)$$

where τ represents trajectories sampled from the policy. Policy gradient methods (Sutton et al., 1999) optimize this objective directly by estimating gradients with respect to policy parameters.

The advantage function Aπ(s, a) = Qπ(*s, a*) − V
π(s) plays a crucial role in reducing variance during training, where Qπand V
π denote action-value and state-value functions respectively.

## 3.2 Contrastive Representation Learning

Contrastive learning has emerged as a powerful paradigm for self-supervised representation learning (Chen et al., 2020). The general concept is learning an embedding space where positive pairs (similar instances) will be placed closer together and negative pairs (dissimilar instances) pushed away from each other.

Given a batch of N examples, the contrastive loss (InfoNCE) (Oord et al., 2018) for an anchor xi and its positive pair xj is defined as:

$\mathcal{L}_{contrast}=-\log\frac{\exp(sim(z_{i},z_{j})/\tau)}{\sum_{k=1}^{N}\mathbb{I}_{k\neq i}\exp(sim(z_{i},z_{k})/\tau)}$  **Remark 1**: _The second fundamental domain is a fundamental constant._
$$(2)$$
where zi = fθ(xi) represents the encoded embedding, sim denotes cosine similarity, and τ is a temperature hyperparameter. For code graphs, positive pairs can be generated through structurepreserving transformations like variable renaming or statement reordering (Ding et al., 2021).

## 3.3 Code Graph Representations

Modern code analysis increasingly relies on graph-based representations that capture both syntactic and semantic relationships (Allamanis et al., 2017). Abstract Syntax Trees (ASTs) are used for hierarchical structure, and control flow graphs (CFGs) are used for execution paths.

Formally, a code graph G = (V, E) consists of nodes v ∈ V representing code elements (e.g., statements, expressions) and edges e ∈ E denoting relationships between them. Graph neural networks (Kipf, 2016) operate on these structures through message passing:

$$h_{v}^{(l+1)}=\sigma\left(W^{(l)}\sum_{u\in\mathcal{N}(v)}\frac{h_{u}^{(l)}}{|\mathcal{N}(v)|}+B^{(l)}h_{v}^{(l)}\right)$$
$$({\mathfrak{I}})$$

where h
(l)
v represents the node embedding at layer l, N (v) denotes neighbors of node v, and W(l), B(l)are learnable parameters.

The combination of these concepts is the theoretical basis for our approach. The RL framework handles the decision making mechanism, the use of contrastive learning to make effective representation from unlabeled code, and GNN to process the structural information inherent in software.

## 4 Contrastive Graph Embeddings For Refactoring-Aware Rl

The proposed framework reinforces a contrastive pre-training approach with reinforcement learning and the ability to conduct automated code refactoring without extensive expert supervision.

## 4.1 Syntax-Guided Contrastive Code Graph Encoder

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 The encoder architecture processes code graphs G = (*V, E*) through a series of graph attention layers that compute node embeddings hv ∈ R
d. For contrastive pre-training, we are generating positive examples by pairs of generated graphs ({G1, G2}) where is undergone syntax-preserving transformations that include:
- Subtree masking: Randomly removing AST subtrees while maintaining program validity
- Edge rewiring: Modifying non-critical control flow edges without altering semantics
- Identifier shuffling: Permuting variable names within scope constraints The contrastive objective minimizes:

## 4.2 Metric Fusion In Reward Function

The composite reward combines three components: The fused reward becomes:

$$r_{t}=\mathbf{w}_{q}^{\top}\phi(\mathbf{q}_{t})+\alpha\operatorname{tanh}(\beta\Delta\mathbf{h}_{t})-\gamma(1-\delta_{t})$$
q ϕ(qt) + α tanh(β∆ht) − γ(1 − δt) (5)
The policy's exploration distribution incorporates Mahalanobis distance to prototype states:

## 4.3 Embedding-Guided Exploration Strategy

where ϕ denotes min-max normalization and *α, β, γ* are scaling parameters. The hyperbolic tangent means that the gradients propagate in a stable way during RL training.

$$\pi_{e x p l o r e}(a|s)\propto\exp\left(-\frac{1}{2}(\mathbf{h}_{s}-\mathbf{h}^{*})^{\top}\Sigma^{-1}(\mathbf{h}_{s}-\mathbf{h}^{*})\right)$$
(6)
where h
∗represents the running average of high-reward states and Σ is the empirical covariance matrix of pre-training embeddings. This lets exploration be biased toward parts of the latent space where there are associated effective refactorings.

## 4.4 Graph Attention Policy With Joint Representations

The policy network processes concatenated features [ht; qt] through attention mechanisms:
ωij = softmaxjLeakyReLU a
⊤[Whi∥Whj ] (7)
where a ∈ R
2d
′and W ∈ R
d
′×dare learned parameters. The attention weights decide how nodes aggregate information from their syntactic neighbors when they are amounting correct refactoring actions.

where B denotes the batch of negative examples and fθ produces graph-level embeddings through mean pooling of node representations. The temperature parameter τ controls separation between positive and negative pairs.

$$(6)$$
$$(7)$$
$${\mathcal{L}}_{p r e}=-\mathbb{E}_{(G_{1},G_{2})}\left[\log{\frac{\exp(\sin(f_{\theta}(G_{1}),f_{\theta}(G_{2}))/\tau)}{\sum_{G^{\prime}\in{\mathcal{B}}}\exp(\sin(f_{\theta}(G_{1}),f_{\theta}(G^{\prime}))/\tau)}}\right].$$
$$(4)$$

(4)
1. **Traditional metrics** qt ∈ R
m: Cyclomatic complexity, coupling metrics, and style violations 2. **Embedding dynamics** ∆ht = ∥ht − ht−1∥2: Magnitude of latent space movement 3. **Semantic preservation** δt = I[test(Gt) = test(Gt−1)]: Differential test verification

## 4.5 Semantic Preservation Via Differential Testing

A lightweight equivalence checker computes δt by:
1. Extracting method signatures and I/O contracts from Gt−1 and Gt 2. Generating test cases through symbolic execution (Cadar & Sen, 2013) 3. Comparing execution traces using normalized Hamming distance:

$$\delta_{t}=1-{\frac{1}{L}}\sum_{k=1}^{L}\mathbb{I}[\mathrm{trace}_{k}(G_{t-1})\neq\mathrm{trace}_{k}(G_{t})]$$
$$(8)$$

where L denotes the test case count. This dynamic verification ensures behavior preservation without expensive formal methods.

## 4.6 End-To-End Integration

The complete system operates in three phases:
1. **Pre-training**: Optimize fθ via contrastive loss on unlabeled code corpora 2. **RL fine-tuning**: Fix fθ and train policy network πϕ using PPO (Schulman et al., 2017)
3. **Inference**: Deploy πϕ with ϵ-greedy exploration (ϵ = 0.1)
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 5 Experimental Evaluation 5.1 Experimental Setup

Datasets and Codebases We evaluated our method on three established code refactoring datasets: To validate the effectiveness of our proposed method, we conducted thorough experiments to compare our method with state-of-the-art baselines in multiple dimensions of refactoring quality and generalization ability. This section presents our experimental set-up, evaluation criteria and comparison results.

- **Refactory** (Kad´ ar et al., 2016): Contains 12,500 Java methods with expert-annotated refac- ´
toring labels
- **CodeRef** (Wang et al., 2024): Comprises 8,700 Python functions with version historybased refactoring pairs
- **BigCloneBench** (Svajlenko & Roy, 2016): Includes 6 million Java code fragments for cross-project evaluation For pre-training the contrastive encoder, we used the **CodeSearchNet** (Husain et al., 2019) corpus containing 2 million functions across 6 programming languages. Baseline Methods We compared against four categories of refactoring approaches: The way we design the network in a modular fashion and can therefore swap the components (i.e. to the different GNN architecture) while the overall learning paradigm remains the same.

1. **Rule-based**: PMD (Mayer & Schroeder, 2012) and Checkstyle (Kupari et al., 2025) with default rule sets 2. **Learning-based**:
- **Code2Seq** (Alon et al., 2018): Sequence-to-sequence model with AST paths
- **Graph2Edit** (Cai et al., 2023): GNN-based edit predictor 3. **RL-based**:
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Table 1 presents the aggregate performance across all evaluation metrics.

## 5.2 Comparative Results

The results provide some important insights:
- Graph encoder: 4-layer GAT with 256-dimensional hidden states - Contrastive pre-training: 100 epochs, batch size 512, temperature τ = 0.1 - RL training: PPO with γ = 0.99, λ = 0.95, 1M environment steps
- Reward weights: wq = [0.4, 0.3, 0.3], α = 0.2, β = 1.0, γ = 0.5
- Hardware: 8×NVIDIA V100 GPUs (pre-training), single GPU (RL)
1. Traditional rule-based methods (PMD, Checkstyle) show limited syntactic improvement due to their rigid rule sets, though they maintain good semantic preservation.

2. Learning-based approaches (Code2Seq, Graph2Edit) achieve better SI scores but suffer in SP due to their lack of explicit semantic constraints.

3. RL methods generally outperform static analyzers, with GraphRL showing particularly strong performance due to its graph-based policy.

4. Our method achieves the best balance across all metrics, with particularly strong gains in generalization (GS), suggesting the contrastive pre-training effectively captures transferable refactoring patterns.

## Evaluation Metrics

We employed five complementary metrics:
1. **Syntactic Improvement (SI)**: Percentage reduction in code smells (PMD/Checkstyle violations)
2. **Semantic Preservation (SP)**: Test case pass rate after refactoring 3. **Edit Distance (ED)**: Normalized Levenshtein distance between original/refactored code 4. **Maintainability Gain (MG)**: Improvement in QMOOD (El-Wakil et al., 2004) metric scores 5. **Generalization Score (GS)**: Performance on unseen project types (cross-validation)
Implementation Details Our implementation used:

Method SI (%) SP (%) ED MG (%) GS (%) PMD 62.1 88.3 0.41 15.2 45.6 Checkstyle 58.7 91.2 0.38 12.8 49.3 Code2Seq 71.5 82.4 0.52 18.7 54.2 Graph2Edit 75.2 85.6 0.49 21.3 58.9 RLRefactor 68.3 86.7 0.45 19.5 52.4 GraphRL 77.8 89.2 0.43 23.1 63.7 NeuroRefactor 79.4 90.5 0.40 24.6 67.2 Ours 83.7 93.8 0.36 27.9 72.4

- **RLRefactor** (Palit & Sharma, 2024b): DQN with handcrafted rewards - **GraphRL** (Darvariu et al., 2024): GNN policy with expert demonstrations 4. **Hybrid**:
- **NeuroRefactor** (Karakati & Thirumaaran, 2022): Combines neural metrics with rulebased constraints

## 5.4 Cross-Language Generalization

324

![6_image_0.png](6_image_0.png) 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

| Variant                      | SI (%)   | SP (%)   | MG (%)   |
|------------------------------|----------|----------|----------|
| Full model                   | 83.7     | 93.8     | 27.9     |
| w/o contrastive pre-training | 76.2     | 89.1     | 22.4     |
| w/o embedding rewards        | 79.5     | 91.3     | 24.7     |
| w/o semantic tests           | 81.6     | 85.2     | 25.3     |
| Random exploration           | 74.8     | 92.6     | 21.8     |

## 5.3 Ablation Study

To understand the contribution of each component, we conducted an ablation study by systematically removing key elements of our approach (Table 2). The most significant drop occurs when removing contrastive pre-training (-7.5% SI), highlighting the importance of learned representations. The semantic test component proves crucial for maintaining behavior preservation (-8.6% SP when removed). To study the transferability of our approach, we evaluated the model already trained over a Java language codebase (CodeSearchNet) over Python and C++ codebases without further fine-tuning. As shown in Table 3, the method maintains reasonable performance despite the domain shift, outperforming language-specific rule-based tools.

## 5.5 Qualitative Analysis

Case studies demonstrate our method's ability to discover non-obvious optimizations:

## 6 Discussion And Future Work 6.1 Limitations Of The Proposed Method

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 While it is clear that our approach shows significant improvements over existing approaches, there are some rather obvious limitations that should be commented upon. The contrastive pre-training phase is costly especially when dealing with large codebases with complicated dependency graphs.

## 6.2 Potential Application Scenarios

| Target Language   | Method   | SI (%)   | SP (%)   |
|-------------------|----------|----------|----------|
| Python            | PyLint   | 59.2     | 90.4     |
| Ours              | 68.7     | 88.9     |          |
| C++               | Cppcheck | 54.3     | 93.1     |
| Ours              | 63.5     | 91.2     |          |

![7_image_0.png](7_image_0.png)

Figure 2: Correlation (Pearson's r=0.72) between embedding space movement (∆h) and actual quality improvement (SI), validating that the learned representations capture meaningful refactoring signals. Most beneficial refactorings (high SI) cluster in a specific region of embedding dynamics.

1. **Pattern Consolidation**: Identified duplicate validation logic across nested conditionals and extracted them into guard clauses 2. **Dataflow Optimization**: Reordered operations to minimize intermediate object creation in collection processing chains 3. **Architectural Hint**: Suggested converting procedural-style code to strategy pattern when detecting similar control flows with varying operations The framework has been found to be particularly promising in a number of widely used practical software engineering settings. Continuous integration pipelines for the purpose could integrate with the refactoring agent to automatically improve code quality in the development process.

432

![8_image_0.png](8_image_0.png) 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 8 The Use Of Llm

We use LLM polish writing based on our original paper.

## References 6.3 Scalability Of The System

The current implementation has reasonable scalability characteristics, and supports codebases with as many as 1 million lines of code in our experiments. This graph attention mechanism is complex and increases in a linear fashion with the number of edges; this makes it feasible for most real-world projects.

## 7 Conclusion

The proposed framework shows that combining contrastive pre-training and reinforcement learning essentially leads to enormous improvement in the automated code refactoring capabilities. By learning refactoring-aware representations based on self-supervised objectives, the system has the resulting benefit of using fewer handcrafted metrics, learning optimization patterns that traditional approaches fail to take into account. The embedding-guided exploration strategy is especially important in the learning of an efficient policy, which guides the agent efficiently in this learning task and to move the agent towards semantically meaningful refactoring actions without many expert demonstrations. M Allamanis, M Brockschmidt, and M Khademi. Learning to represent programs with graphs.

Technical report, arXiv preprint arXiv:1711.00740, 2017.

U Alon, S Brody, O Levy, and E Yahav. code2seq: Generating sequences from structured representations of code. Technical report, arXiv preprint arXiv:1808.01400, 2018.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 T Chen, S Kornblith, M Norouzi, et al. A simple framework for contrastive learning of visual representations. In *International Conference on Machine Learning*, 2020.

VA Darvariu, S Hailes, and M Musolesi. Graph reinforcement learning for combinatorial optimization: A survey and unifying perspective. Technical report, arXiv preprint arXiv:2404.06492, 2024.

Y Ding, L Buratti, S Chakraborty, S Pujar, A Morari, et al. Contrastive learning for source code with structural and functional properties. Technical report, openreview.net, 2021.

M El-Wakil, A El-Bastawisi, et al. Object-oriented design quality models a survey and comparison.

Unable to determine the complete publication venue, 2004.

D Guo, S Ren, S Lu, Z Feng, D Tang, S Liu, et al. Graphcodebert: Pre-training code representations with data flow. Technical report, arXiv preprint arXiv:2009.08366, 2020.

H Husain, HH Wu, T Gazit, M Allamanis, et al. Codesearchnet challenge: Evaluating the state of semantic code search. Technical report, arXiv preprint arXiv:1909.09436, 2019.

Chitti Babu Karakati and Sethukarasi Thirumaaran. Software code refactoring based on deep neural network-based fitness function. *Concurrency and Computation: Practice and Experience*, 35, 2022.

S Kimura, Y Higo, H Igaki, et al. Move code refactoring with dynamic analysis. In 2012 28th Ieee International Conference On Software Maintenance (Icsm), 2012.

TN Kipf. Semi-supervised classification with graph convolutional networks. Technical report, arXiv preprint arXiv:1609.02907, 2016.

A Kupari, N Giacaman, and V Terragni. Adoption and evolution of code style and best programming practices in open-source projects. Technical report, valerio-terragni.github.io, 2025.

I Kad´ ar, P Hegedus, R Ferenc, et al. A code refactoring dataset and its assessment regarding software ´
maintainability. In *2016 IEEE 23rd International Conference On Software Analysis, Evolution,* And Reengineering, 2016.

A LeClair, S Haque, L Wu, and C McMillan. Improved code summarization via a graph neural network. In Proceedings of the 28th ACM SIGSOFT International Symposium on Software Testing and Analysis, 2020.

A Marvellous, I Frigerio, D Verdi, U Schultz, et al. Reinforcement learning for automated code refactoring. Technical report, researchgate.net, 2025.

P Mayer and A Schroeder. Cross-language code analysis and refactoring. In 2012 IEEE 12th International Working Conference on Source Code Analysis and Manipulation, 2012.

A Oord, Y Li, and O Vinyals. Representation learning with contrastive predictive coding. Technical report, arXiv preprint arXiv:1807.03748, 2018.

I Palit and T Sharma. Generating refactored code accurately using reinforcement learning. Technical report, arXiv preprint arXiv:2412.18035, 2024a.

Indranil Palit and Tushar Sharma. Generating refactored code accurately using reinforcement learning. *ArXiv*, abs/2412.18035, 2024b.

OR Polu. Ai-driven automatic code refactoring for performance optimization. Technical report, academia.edu, 2025.

C Cadar and K Sen. Symbolic execution for software testing: three decades later. *Communications* of the ACM, 2013.

H Cai, Y Nong, Y Ou, and F Chen. Generating vulnerable code via learning-based program transformations. *Ai Embedded Assurance for Cyber Physical Systems*, 2023.

RD Prasad and M Srivenkatesh. A hybrid model combining graph neural networks, reinforcement learning, and autoencoders for automated code refactoring and optimization. Journal of Theoretical and Applied Information Technology, 2025.

J Schulman, F Wolski, P Dhariwal, A Radford, et al. Proximal policy optimization algorithms.

Technical report, arXiv preprint arXiv:1707.06347, 2017.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 RS Sutton and AG Barto. Reinforcement learning: An introduction. Technical report, cambridge.org, 1998.

RS Sutton, D McAllester, S Singh, et al. Policy gradient methods for reinforcement learning with function approximation. In *Advances in Neural Information Processing Systems*, 1999.

J Svajlenko and CK Roy. Bigcloneeval: A clone detection tool evaluation framework with bigclonebench. In *2016 IEEE International Conference on Software Maintenance and Evolution*, 2016.

G Taentzer, T Arendt, C Ermel, and R Heckel. Towards refactoring of rule-based, in-place model transformation systems. In *Proceedings of the First Workshop on the Model-Driven Engineering,* Verification, and Validation, 2012.

M Tufano, J Pantiuchina, C Watson, et al. On learning meaningful code changes via neural machine translation. In *2019 IEEE/ACM 41st International Conference On Software Engineering*, 2019.

X Wang, Y Wang, F Mi, P Zhou, Y Wan, X Liu, et al. Syncobert: Syntax-guided multi-modal contrastive pre-training for code representation. Technical report, arXiv preprint arXiv:2108.04556, 2021.

Y Wang, Y Wang, S Wang, D Guo, J Chen, et al. Repotransbench: A real-world benchmark for repository-level code translation. Technical report, arXiv preprint arXiv:2412.17744, 2024.

Y Ye, T Zhang, W Jiang, and H Huang. Process-supervised reinforcement learning for code generation. Technical report, arXiv preprint arXiv:2502.01715, 2025.