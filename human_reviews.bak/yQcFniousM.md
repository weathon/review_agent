# Pacmann: Efficient Private Approximate Nearest Neighbor Search

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
We propose a new private Approximate Nearest Neighbor (ANN) search scheme
named Pacmann
that allows a client to perform ANN search
in a vector database 
without revealing the query vector to the server.
Unlike prior constructions that run encrypted search on the server side,
Pacmann carefully offloads limited computation and storage to the client,
no longer requiring computationally-intensive cryptographic techniques.
Specifically, clients run a graph-based ANN search, where in each hop on the graph, the client privately retrieves local graph information from the server. 
To make this efficient, we combine two ideas: 
(1) we adapt a leading graph-based ANN search algorithm to be compatible with private information retrieval (PIR) for subgraph retrieval;
(2) we use a recent class of PIR schemes that trade offline preprocessing for online computational efficiency. 
Pacmann achieves significantly better search quality than
the state-of-the-art private ANN search schemes,
showing up to 2.5$\times$ better search accuracy on 
real-world datasets than prior work and
reaching 90\% quality of a state-of-the-art 
non-private ANN algorithm.
Moreover on large datasets with up to 100 million vectors,
Pacmann shows better scalability 
than prior private ANN schemes
with up to 62\% reduction in computation time
and 22\% reduction in overall latency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces PACMANN, a private Approximate Nearest Neighbor (ANN) search scheme enabling clients to search vector databases without exposing their query vector to the server. Unlike previous methods relying on server-side encrypted searches, PACMANN offloads limited computation and storage to the client. It achieves up to 2.5× higher search accuracy on real-world datasets compared to prior private ANN schemes, approaching 90% of the quality of a state-of-the-art non-private ANN algorithm.

### Strengths
Query search mechanisms using fully homomorphic encryption (FHE) and multi-party computation (MPC) often suffer from high computational complexity. This paper addresses these challenges by introducing a localized search approach that performs iterative graph traversal on the client side. To minimize computation costs, it preprocesses Private Information Retrieval (PIR) to obtain private information from the server efficiently. Leveraging the Piano framework, the paper achieves a reduction in computation and communication costs to $O(\sqrt{n})$. For practical implementation, it further enhances efficiency with techniques such as beam search, fast start, and batched PIR queries.

### Weaknesses
The definition of privacy in the paper is somewhat vague. Could you clarify the attack model? Specifically, what are the attacker’s capabilities, and what information are they attempting to extract from the dataset? Providing these details would help make the definition of privacy more precise and formal.

In addition to FHE, MPC, and PIR, differential privacy (DP) can also provide formal privacy guarantees. While DP generally incurs low computational costs, it may reduce the model’s utility. Including a comparison with DP-related work would make the paper more comprehensive.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces PACMANN, a private approximate nearest neighbor (ANN) search method enabling clients to conduct privacy-preserving nearest neighbor queries across hundreds of millions of vectors. PACMANN utilizes state-of-the-art private information retrieval technique and customized searching algorithm to query vector database both privately and efficiently. PACMANN surpasses current leading private ANN search techniques in search quality and offers reduced latency for large-scale datasets. This approach could inspire more follow-ups applications in the area.

### Strengths
[S1] The motivation of this paper is clear and interesting, which is to address the privacy and efficiency challenges of remotely querying a vector database. 

[S2] This paper successfully identifies the and addresses the limitations of existing approaches. It introduces a novel application of private information retrieval (PIR) to perform ANN search in a vector database. Compared with the baseline Tiptoe [1], this work does not involve heavy clustering-based ANN algorithm and homomorphic encryption, acheving promising efficiency for private query.

[1] Henzinger, Alexandra, et al. "Private web search with Tiptoe." Proceedings of the 29th symposium on operating systems principles. 2023.

[S3] This paper is well-organized with necessary background knowledge and clear illustration of high-level ideas. The language and presentation of this paper are easy understand.

[S4] This paper provides fair discussion on the proposed approach and identifies the limitation of this work. For example, it is more suitable for scenarios with good network connection, which would lead further research on potential following works on more general network connection. 

[S5] This paper provides rigorous theoretical insights and illustrate the gap between theory and practice to enhance the understanding of graph-based ANN search.

### Weaknesses
[W1] The security analysis of the proposed approach is missing, which should be provided for a work specified in privacy-preservation. Although the core procedure PIR is invoked in black-box style, this paper should formally capture the potential information leakage during the workload. For example, the baseline of this paper Tiptoe provides a security analysis in Appendix D.

[W2] It would be better if the baseline Tiptoe can be implemented more completely in experiments. For example, this paper claims the efficiency limitation of Tiptoe comes from heavy homomorphic encryption and some other factors. As far as I know, Tiptoe just requires lightweight homomorphic encryption (i.e., homomorphic addition in LWE-style ciphertexts) and it is not that slow. For fair comparison, the simulation of Tiptoe should be equipped with homomorphic encryption to confirm the outperformance of this paper.

Typos: 
“We could potentially the verifiable” in Appendix C.1.

### Questions
[Q1] Could you provide security analysis of the proposed method?

[Q2] It would be better to compare efficiency with Tiptoe-based solution.

### Soundness
3

### Presentation
4

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
This paper uses proposes a secure protocol for approximate nearest neighbours (ANN). The protocol is based on the client traversing a graph of the database held by a server, using a Private Information Retrieval (PIR) protocol to query each part of the graph that the client traverses. As the PIR protocol they use from previous work has O(sqrt(n)) online cost per query, and they need o(n) queries the latency is sublinear in the database size. Though the whole database must be sent during a preprocessing phase only a O(sqrt(n)) fraction of it need be stored by the client.

They provide some optimizations to the bsic idea, taking advantage of beam search to use the fact that it is cheaper per query to make several at once. they provide a small ablation study to verify this is infact helpful.

They then compare the quality of their results to the quality of some linear latency algorithm finding it improves on the baseline (on quality/latency tradeoff) when the database has size 2m to 50m depending on the network connection.

### Strengths
This is an interesting idea to explore.

The scheme works well asymptotically, being the first scheme with sublinear latency, thus constituting a theoretical breakthrough.

The idea is not just naively applied with thought gone into possibilities like beam search.

### Weaknesses
The breakthrough in asymptotics here is mostly coming from previous PIR work, though exploring the applicaiton of it is still valuable.

For the largest dataset they consider, after all their optimizations the reduction in latency is between 1.3x and 2.2x depending on the network (down from about 4 seconds)  to achieve this they require 3GB of client storage and 60GB of preprocesing communication. I am not aware of and they do not point to any realistic application in which this is a good tradeoff in practice.

### Questions
Are you aware of any realistic applications?

Shouldn't the expansion of the PACMANN acronym include the word nearest to explain the double N?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents Pacmann, a new private approximate nearest neighbor search scheme. The main claim is that this scheme is more accurate and efficient than prior private ANN search schemes.

The design of Pacmann includes a number of features, which I will try to recount here:
* Pre-process the search graph to ensure that every vertex in the graph has a bounded out-degree of $C$. Then, when one performs a greedy walk on the graph (toward finding an approximate neighbor), the search for the next hop needs to inspect $C$ vertices at most.
* To ensure privacy, the client will perform the search algorithm themselves. However, it is unreasonable to store the graph on the client side. If the client sends queries to the server in plain text, this may pose another privacy risk. The solution is a private information retrieval scheme, which allows the client to retrieve data from the server without revealing the index of interest. The paper appears to use an off-the-shelf PIR scheme from prior work, which, after client-side pre-processing, can achieve sublinear communication and computation costs for information retrieval.

A number of experiments are provided to support the effectiveness of the approach.

### Strengths
* The problem of study seems to be important, and the solution is natural and clearly written.
* The experiment results look promising.

### Weaknesses
(Disclaimer: I might not be an expert in critiquing this paper, as I don't have much experience in this area.)

* Can you comment on the scalability of your approach? I am quite concerned with the $O(\sqrt{n})$ communication and computation cost, as I understand that $n$ is the total number of vertices in the database. I see you ran your experiments up to 100M vertices. What does this scale mean in practice? I imagine, e.g., Google search would have a way larger scale of data than this.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
