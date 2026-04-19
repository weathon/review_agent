# On the Expressive Power of Sparse Geometric MPNNs

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Motivated by applications in chemistry and other sciences, we study the expressive
power of message-passing neural networks for geometric graphs, whose node
features correspond to 3-dimensional positions. Recent work has shown that such
models can separate generic pairs of non-isomorphic geometric graphs, though they
may fail to separate some rare and complicated instances. However, these results
assume a fully connected graph, where each node possesses complete knowledge
of all other nodes. In contrast, often, in application, every node only possesses
knowledge of a small number of nearest neighbors.
This paper shows that generic pairs of non-isomorphic geometric graphs can
be separated by message-passing networks with rotation equivariant features as
long as the underlying graph is connected. When only invariant intermediate
features are allowed, generic separation is guaranteed for generically globally
rigid graphs. We introduce a simple architecture, EGENNET, which achieves our
theoretical guarantees and compares favorably with alternative architecture on
synthetic and chemical benchmarks

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This study studies expressivity of message-passing neural networks (MPNNs) for 3D graphs. Unlike earlier research that assuming fully connected graphs, this work demonstrates that MPNNs with equivariant features can distinguish between generic non-isomorphic geometric graphs when the graphs are connected. For cases where only invariant features are used, the ability to differentiate graphs is ensured for those that are generically globally rigid. This work also proposes a new architecture called EGENNET, which utilizes equivariant features and performs well on both synthetic and chemical benchmarks.

### Strengths
1. Solid theoretic results. The "generic" describes degeneracies cases well.
2. Separation experiments are clear.

### Weaknesses
1. Considering generic geometric graphs only rather than all geometric graphs significantly weakens the significance of theoretic results. Though degenerated cases are rare, they can affect model's performance on generic graphs as neural networks are smooth. The inclusion of high-order irreducible representations solves these cases and boosts performance significantly.[1, 2]

2. Previous work also considers expressivity on geometric graph without assuming fully connection. [3] proves in that in non-degenerated cases, MPNN with equivariant feature can distinguish connected graphs, while MPNN with invariant features only cannot.

3. Experiment should includes graph force field tasks on qm9 and md17 as previous works [2].

[1] Nadav Dym, Haggai Maron. On the Universality of Rotation Equivariant Point Cloud Networks. ICLR 2021.
[2] https://www.nature.com/articles/s41467-022-29939-5
[3] Xiyuan Wang, Muhan Zhang. Graph Neural Network With Local Frame for Molecular Potential Energy Surface. LoG 2022.

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the expressive power of in- and equivariant message-passing neural networks (MPNNs) on geometric graphs is studied, where in contrast to prior results that assume full graphs, arbitrary graph topologies are considered. Here, the authors focus on *generic* separation, meaning that the analysis is restricted to graphs without certain symmetries. The analysis reveals that a graph can be generically identified by an invariant GNN iff it is generically globally rigid, and by an equivariant GNN iff it is connected. The authors then introduce a maximally expressive equivariant GNN, called EGenNet, which can generically separate all connected graphs. The expressivity is empirically verified on toy graphs, and significant improvements on benchmarks for chemical property prediction are reported.

### Strengths
- Analyzing the expressive power of sparse geometric GNNs is a relevant research direction, closing the gap between theory (where the attention has been mostly on full graphs) and practive (where typically sparse graphs are considered, e.g., via distance cutoffs).
- The characterizations for the expressivity of in- and equivariant GNNs are simple and practical, and are therefore relevant to the research community, in my opinion.
- The empirical results are quite strong, as EGenNet outperforms practically all graph ML methods and geometric GNNs on the Drugs-75k, Kraken, and BDE benchmarks.

### Weaknesses
Although I understand that the assumption of genericity makes the analysis tractable, this remains a significant limitation. Real-world geometric graphs, particularly in the chemical domain, can exhibit lots of symmetries that cannot be overlooked.

### Questions
- Did you conduct experiments on how EGenNet compares to the other benchmarks on non-generic geometric graphs? A step towards this could be looking at the levels of symmetry that the graphs in the chemical dataset exhibit (i.e., distances between nodes or graph automorphisms), and analyzing the performance of EGenNet vs. the other benchmarks on such graphs.
- In section 5.1, the EGenNet architecture comes a bit out of nowhere. Perhaps, a paragraph with an intuitive explanation why the update step is defined as it is in equation 5 and how it relates to maximal expressivity would add to the overall understanding of the method.
- The results for chemical property prediction could be discussed in more detail. Specifically, a further analysis of the number of parameters or training runtimes of the different models compared in Table 2 would be interesting to see.
- There are lots of presentation issues in the script that should be fixed, e.g., inconsistent usage of `\citep`/`\citet`, misalignment of tables (Table 2, Table 4, Table 5, Table 6), inconsistent caption alignment, Table 6 seeming to be a screenshot from its original source, or "todo" comments (line 451).

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
The present manuscript  investigates the expressive capabilities of MPNNs for sparse geometric graphs, and it focuses on applications in fields such as chemistry for benchmarking. The authors differentiate between Equivariant Geometric GNNs (E-GGNNs) and Invariant Geometric GNNs (I-GGNNs), claiming that E-GGNNs can separate connected graphs, while I-GGNNs require globally rigid graphs for separation.

They propose EGENNET, a simple E-GGNN model intended to maximize expressiveness in sparse settings, and benchmark its performance against alternative models.

### Strengths
1. The paper is clear on addressing the question of expressivity on the specific area of sparse graphs, which is relevant to applications like molecular modeling, where the computational cost of fully connected graphs is prohibitive.

2. The authors provide a theoretical analysis distinguishing the separation power of E-GGNNs and I-GGNNs based on graph connectivity and rigidity, which adds clarity to the strengths and limitations of each approach.

3. The proposed architecture seems to be fairly simple, providing a potentially expressive model that is easier to implement than complex alternatives like (d−1)-WL-based models, aligning with the need for practical applications in sparse graph settings.

### Weaknesses
- I find the connection of generic matrices with the use cases of graphs complicated. The authors do not show a clear motivation on defining expressivity over generic spaces for the use over graphs. Can the authors motivate better the analysis over generic spaces? What happens in the non-genericity direction?

- The tradeoff between efficiency and expressiveness is confusing. It seems that still if we want to achieve maximal expressiveness leads to quadratic complexity (G.2 Section), not having a clear understanding of the expressivity loss, once we set C < 2Nd + 1.
- The presentation is very poor. The theoretical analysis is hard to follow. There are a lot of typos, and inconsistencies. There are forgotten TODOs in the main text (check Section 6.2), and elements of the paper are not even visible (for example Table 5 does not fit the page).

Overall, the paper shows some elements of haste, and I would suggest the improvement of the writing and the presentation of the theoretical as well as the empirical results.

### Questions
Given the reliance of the paper's results on the genericity assumption, can the authors motivate clearly the need to study the genericity case? What can we say about non-generic spaces, and why are they important to their experimental results? e.g. for the molecular property prediction tasks.


UPDATE: Given the answer by the authors and their efforts to fix the presentation issues of the paper, I'm raising my rating.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method for distinguishing between geometric graphs based on the connectivity of the graph structure. Moreover they propose an equivariant architecture which they claim to be maximal in expressivity.

### Strengths
The paper utilizes global rigidity to develop a novel direction for developing geometric graph neural networks.

 EGenNet achieves state of the art performance and makes significant improvements on a few datasets.

### Weaknesses
I have two major concerns regarding the submitted work.

1) Incomplete Submission: The paper contains 'TODO' placeholders and formatting issues, indicating it is an unfinished draft. This severely hinders the ability to conduct a thorough and fair review and compromises the integrity of the review process and the conference's high standards.

2) Lack of Consistency and Discussion in Empirical Results: The empirical results lack completeness and coherence. Key sections are marked TODO, and there are inconsistencies between tables, making it difficult to interpret the significance of the findings. A more thorough discussion and clear presentation of results are necessary to support the paper's claims.

### Questions
Below is a partial list of incomplete work:

1) The paper is not formatted to the ICLR standard, e.g. it is missing the header **Under review as a conference paper at ICLR 2025**.

2) The inline citation format is incorrect.

3) Section 6.2 contains a "TODO" placeholder.

4) The second paragraph of Section 5 is two egregiously run on sentences.

5) Tables 2, 4, 5, and 6 extend beyond the page margins, with Table 5 exceeding the page to the point where the results are unreadable.

6) Table 4 uses an incorrect title for the model, referring to it as GenNet.

7) Table 6 is a low resolution image of a table and has discrepancies in the reported results by comparison to Table 2.

---

1) The definition of "identifies" appears to be equivalent to the definition of a complete invariant function F. Per Section 1 and Appendix B: "The main text discussed (sic) how generic completeness can be achieved using I-GGNN when the graph is globally rigid." Is there a reason that the terminology of completeness that appears in the introduction and appendix is dropped in the definitions of Section 2 in favor of identifies? Maintaining consistent terminology would significantly enhance clarity.

2) The cutoff tables 5 and inconsistencies between Table 6 and Table 3 make it difficult to interpret the significance of the reported results. Could the authors please explain the discrepancies between the Tables.

3) In Kraken BurL and BDE it appears that EGenNet significantly outperforms the previous state of the art, especially for BDE. Could the authors provide an explanation for why this major improvement occurs for the BDE dataset?

### Soundness
3

### Presentation
3

### Contribution
3
