# DLDP-BF: A Differentiated Local Differential Privacy Bloom Filter for Membership Queries

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
In privacy-preserving data processing, Bloom filters are widely used for their efficiency and scalability. 
However, existing methods adopt a fixed number of hash functions for all elements, disregarding their varying importance or frequency within the dataset. 
This uniform treatment leads to a suboptimal trade-off between privacy and utility, as high-priority elements, such as frequent or critical data, require more precise encoding and finely tuned privacy protection, while less significant elements can tolerate greater uncertainty without severely affecting system performance.
To address this issue, we propose a Differentiated Local Differential Privacy Bloom Filter for Membership Queries (DLDP-BF). 
This method dynamically allocates hash functions based on the relative importance of elements, enabling configuration of differentiated Bloom filters. 
DLDP-BF allocates more resources to high-priority elements, improving their encoding precision and reducing perturbations, thereby ensuring query accuracy for critical data.
Furthermore, we design a novel local differential privacy (LDP) budget allocation algorithm based on differentiated Bloom filters that adaptively adjusts noise intensity based on element importance.
This algorithm ensures strict privacy protection while minimizing the impact on data utility.
We construct a mathematical model linking the importance of elements and privacy budget allocation, and theoretically demonstrate that our method maintains privacy while also balancing data utility.
Experimental results show that DLDP-BF significantly improves data utility while preserving privacy. Specifically, it achieves an average reduction in RMSE of 37.1\% and an average improvement in accuracy of 9.05\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes DLDP-BF, a Differentiated Local Differential Privacy Bloom Filter that addresses limitations in existing LDP Bloom filter methods by dynamically allocating hash functions and privacy budgets based on element importance. The authors introduce two algorithms: DHFA (Differentiated Hash Function Assignment) that assigns more hash functions to frequently queried elements, and PBA (Personalized Budget Allocation) that allocates privacy budgets proportionally to query frequency and membership probability. Experiments on three datasets demonstrate 49.0% RMSE reduction and 12.3% accuracy improvement compared to RAPPOR and DPBloomFilter, though the evaluation scope remains limited.

### Strengths
1. Well-motivated technical contribution with theoretical support. The paper identifies two genuine limitations in existing work: uniform hash function assignment regardless of data importance and fixed privacy budget allocation across all elements (lines 086-098). The proposed differentiated approach is intuitive and addresses a real practical need. The theoretical analysis (Section 4) provides formal privacy guarantees (Theorem 2) and utility bounds (Theorem 3), with the privacy proof properly accounting for the permanent randomized response mechanism.


2. New privacy budget allocation framework. The PBA algorithm (Algorithm 3, Equation 2) represents, to the authors' knowledge, the first local differential privacy budget allocation method that jointly considers membership probability and query frequency (lines 106-107). This personalized approach is more realistic than uniform allocation and the formulation elegantly balances privacy protection for critical elements while maintaining utility for frequently queried items.

### Weaknesses
1. Insufficient experimental evaluation. The experiments only compare against three methods (Non_Privacy, RAPPOR, DPBloomFilter) on three datasets (Section 5.1), which is limited for demonstrating broad applicability. Critically, the paper claims DPBloomFilter (Ke et al. 2025) is "a representative solution" (line 068) yet this is a very recent concurrent work that may not represent the state-of-the-art.

2. Critical methodological details are unclear. The paper does not explain how query frequency Fi and membership likelihood Li are obtained in practice. Are these estimated from historical data, provided by applications, or learned? This is crucial since the entire method depends on these values (Algorithm 2, line 258; Algorithm 3, line 272). The DHFA algorithm (Equation 1) computes optimal hash count h*i but line 268 states "Let the expected size of the set be n" without clarifying whether n is known beforehand or estimated. 

3. Privacy-utility trade-off analysis lacks depth and the privacy model has limitations. While Figure 3 and 4 show performance across different ε values, the paper does not analyze the fundamental trade-off: how much privacy is sacrificed for utility gains compared to uniform allocation? For instance, does allocating more budget to high-frequency elements disproportionately expose their membership? The privacy guarantee (Theorem 2) assumes the hash function family and number of hash functions are public (Definition 2, lines 146-149), but the differentiated assignment itself may leak information about element importance.

### Questions
1. Why was DPBloomFilter chosen as the primary comparison method rather than earlier established LDP Bloom filter works?

2. What happens in cold-start scenarios where no historical data exists for new elements?

3. For membership likelihood, how do you estimate the probability that an element belongs to a set before actually querying it?

### Soundness
3

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
2

### Summary
This paper studies privacy-preserving membership queries with Bloom filters under LDP. It argues that existing LDP-BF methods use uniform parameters (same number of hash functions, same privacy budget) for all elements, which can be inefficient when elements differ in importance (e.g., membership likelihood or query frequency). The authors propose DLDP-BF, which differentiates both (i) the number of hash functions and (ii) the local privacy budget per element according to importance, aiming to improve utility at fixed privacy.

### Strengths
S1: The problem studied in this paper is clearly defined and well-motivated. 

S2: The experimental results demonstrate performance gains, showing that the proposed approach outperforms four existing baseline methods across multiple datasets.

S3: The paper provides theoretical analysis covering computational complexity, privacy guarantees, and utility bounds.

### Weaknesses
W1: More clarification of the threat model is expected. The method assumes access to (or accurate estimation of) per-element membership likelihood and query frequency to drive DHFA/PBA, but it does not specify how to obtain these safely under privacy constraints or how robust the system is to estimation errors or distribution shift.

W2: Since the proposed method assigns a larger $h_i$ to high-importance items, a server can possibly infer an item’s importance class from the number/structure of touched bits. It is better to analyze or bound this ‘importance-label leakage,’ or propose a mitigation.

W3: Similarly, since the proposed method assigns larger privacy budgets to high-importance elements, a server may be able to infer an item’s importance level based on differences in noise magnitude or output variance.

W4: The DHFA logic leverages prior knowledge of frequency/likelihood distributions; however, the paper does not analyze worst-case mismatch. 

W5: Experiments report RMSE and accuracy across datasets and parameters, but there is no measurement of runtime/throughput, memory, or client/server communication overhead, which are important for Bloom-filter pipelines at scale.

W6: Algorithms 1–3 need clearer, step-by-step explanations.

### Questions
Q1: Could the authors clarify the assumed threat model in more detail? Specifically, how are per-element membership likelihood and query frequency obtained or estimated without violating privacy guarantees? 

Q2: How robust is the proposed framework if these statistics are inaccurate or shift over time?

Q3: Since the proposed method assigns a larger privacy budget/a larger number of hash functions (h_i) to higher-importance items, could a server or an attacker potentially infer an element’s importance level or class from observable Bloom filter updates?

Q4: Similarly, as the proposed method allocates larger privacy budgets to high-importance elements, could the server infer an item's importance level from the differing noise magnitude or output variance?

### Soundness
2

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
This paper investigates the problem of privacy-preserving membership queries over large-scale datasets. Traditional Bloom filter-based approaches typically employ a fixed number of hash functions for all elements, overlooking differences in their importance or frequency, which leads to a suboptimal balance between privacy and utility.

The authors propose a Differentiated Local Differential Privacy Bloom Filter (DLDP-BF), which dynamically allocates the number of hash functions and privacy budgets according to element importance, thereby improving query accuracy. The paper also designs a novel LDP budget allocation algorithm that adaptively adjusts noise intensity proportionate to element importance. Theoretical analysis proves that the method provides strict Local Differential Privacy (LDP) guarantees while enhancing data utility. Experimental results demonstrate the superior performance of the proposed approach in real-world scenarios, confirming its practical value.

### Strengths
1.  **Originality**: This paper innovatively incorporates element importance into privacy budgeting and hash function allocation to achieve differential privacy protection. The introduction of Personalized Budget Allocation (PBA) represents a novel direction in privacy-preserving membership query scenarios.

2.  **Clarity**: Figure 2 provides an intuitive illustration of the DLDP-BF workflow, effectively facilitating the understanding of inter-modular relationships within the system. The paper features a complete structure, with precise textual expression and coherent logical flow.

### Weaknesses
1. Using Bloom filter with LDP for privacy-preserving membership query is not well-motivated. The competitor RAPPOR is originally designed for statistical frequency queries.

2. The proposed method requires prior knowledge about importance/frequency of elements, which can be hard obtain in privacy-sensitive scenarios.
 
3. While the paper provides proofs for privacy guarantees and complexity analysis, it does not establish theoretical performance bounds under varying importance distributions .

4.  The adaptability of DLDP-BF requires further validation in dynamic query scenarios or under distribution drift conditions.

### Questions
1. Please provide motivation scenarios of Bloom filter with LDP for privacy-preserving membership query.

2. Please discuss the practicalness of requiring prior statistical profile about elements.

3. Please discuss the permances under distribution drift.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel variant of locally differentially-private Bloom filter in which the number of hash functions assigned as well as the privacy budget allocated are chosen in a data-dependent manner to improve on the utility.

### Strengths
-The authors have clearly introduced the background on Bloom filters, including their efficiency for performing membership queries but also the privacy issues associated to their uses. 

-The proposed approach has been validated experimentally on three different datasets and compare to two other local differentially-private versions of Bloom filters and seem to display a higher performance in terms of utility.

### Weaknesses
-The review of local differentially-private Bloom filters methods should be strengthen and detailed more. For instance, currently only two methods are cited but without providing any details on the underlying constructions. This issue is really important to be able to position the approach proposed and assess its novelty. Additionally, the literature review should also refer to papers that have studied how to integrate heterogenous or personalized privacy budget in a differentially-private mechanism.

-The dynamic adjustment of the hash function and the privacy budget requires to be able to store a significant amount of information on each element, which defeats the purpose of the Bloom filter that requires to have a highly efficient data structure that can be updated on the fly. More precisely, the choice of the hyper parameters of the proposed data structure should depend on a set of realistic assumptions about the distribution but not require to store an amount of data that is significantly larger than the size of the data structure. 

-The dynamic adjustment of the hash function assignment and the privacy budget based on an element importance is directly in tension with privacy as estimating one of these two parameters can directly leak information about the element. This aspect is crucial to the privacy analysis but is not discussed at all in the paper, which raises serious doubts about the privacy protection provided.

-The papers are not appropriately cited in the text. In particular, there are not between parentheses.

Minor typo :
-« the overall privacy security of the system »

### Questions
Please see the main points raised in the weaknesses section.

### Soundness
1

### Presentation
2

### Contribution
1
