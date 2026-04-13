

{0}------------------------------------------------

# COLLABORATIVE COMPRESSORS IN DISTRIBUTED MEAN ESTIMATION WITH LIMITED COMMUNICATION BUDGET

**Anonymous authors**

Paper under double-blind review

## ABSTRACT

Distributed high dimensional mean estimation is a common aggregation routine used often in distributed optimization methods (e.g. federated learning). Most of these applications call for a communication-constrained setting where vectors, whose mean is to be estimated, have to be compressed before sharing. One could independently encode and decode these to achieve compression, but that overlooks the fact that these vectors are often similar to each other. To exploit these similarities, recently Suresh et al., 2022, Jhunjhunwala et al., 2021, Jiang et al, 2023, proposed multiple *correlation-aware compression schemes*. However, in most cases, the correlations have to be known for these schemes to work. Moreover, a theoretical analysis of graceful degradation of these correlation-aware compression schemes with increasing *dissimilarity* is limited to only the  $\ell_2$ -error in the literature. In this paper, we propose four different collaborative compression schemes that agnostically exploit the similarities among vectors in a distributed setting. Our schemes are all simple to implement and computationally efficient, while resulting in big savings in communication. We do a rigorous theoretical analysis of our proposed schemes to show how the  $\ell_2$ ,  $\ell_\infty$  and cosine estimation error varies with the degree of similarity among vectors. In the process, we come up with appropriate dissimilarity-measures for these applications as well.

## 1 INTRODUCTION

We study the problem of estimating the empirical mean, or average, of a set of high-dimensional vectors in a communication constrained setup. We assume a distributed problem setting, where  $m$  clients, each with a vector  $g_i \in \mathbb{R}^d$ , are connected to a single server (see, Fig. 1a). Our goal is to estimate their mean  $g$  on the server, where

$$g \triangleq \frac{1}{m} \sum_{i \in [m]} g_i. \quad (1)$$

We use  $[m]$  to denote the set  $\{1, 2, \dots, m\}$ . The clients can communicate with the server via a communication channel which allows limited communication. The server does not have access to data but has relatively more computational power than individual clients.

This problem, referred to as *distributed mean estimation* (DME), is an important subroutine in several distributed learning applications. Two common scenarios for these applications are distributed training, when different clients correspond to different processors inside a datacenter or federated learning McMahan et al. (2016); McMahan & Ramage (2017), when different clients correspond to different edge devices, for instance mobile phones. In distributed training, the communication channel is the network inside the datacenter, while in federated learning, the communication channel can be the internet.

The typical learning task for DME is supervised learning via gradient-based methods Bottou & Bousquet (2007); Robbins & Monro (1951). The vectors  $g_i$  then correspond to the gradient updates for each client  $i$  computed on its local training data and  $g$  is the average gradient over all clients. On the other hand, distributed mean estimation is also used in unsupervised learning problems such as distributed KMeans Liang et al. (2013) and distributed PCA Liang et al. (2014) or distributed power iteration Li et al. (2021). In distributed KMeans and distributed power iteration,  $g_i$  corresponds to estimates of cluster center and the top eigenvector respectively, on the  $i^{\text{th}}$  client.

{1}------------------------------------------------

![Figure 1: Compression for Distributed Mean Estimation. (a) Independent Compression: A server at the top computes the mean estimate $\tilde{g} = \frac{1}{m} \sum_{i=1}^m \text{Decode}(\tilde{b}_i)$. Four clients (labeled Client $j$ and Client $i$) send their encoded vectors $\tilde{b}_i$ to the server. Each client $i$ has a vector $g_i$, encodes it into $\tilde{b}_i$, and sends it to the server. The server decodes each $\tilde{b}_i$ to get an estimate $\tilde{g}$. (b) Collaborative Compression: Similar setup, but the server computes $\tilde{g} = \text{Decode}(\tilde{b}_1, \tilde{b}_2, \dots, \tilde{b}_m)$, indicating a joint decoding process.](9ba3dc91984c80b96f217fb1bddd5c06_img.jpg)

Figure 1: Compression for Distributed Mean Estimation. (a) Independent Compression: A server at the top computes the mean estimate \$\tilde{g} = \frac{1}{m} \sum\_{i=1}^m \text{Decode}(\tilde{b}\_i)\$. Four clients (labeled Client \$j\$ and Client \$i\$) send their encoded vectors \$\tilde{b}\_i\$ to the server. Each client \$i\$ has a vector \$g\_i\$, encodes it into \$\tilde{b}\_i\$, and sends it to the server. The server decodes each \$\tilde{b}\_i\$ to get an estimate \$\tilde{g}\$. (b) Collaborative Compression: Similar setup, but the server computes \$\tilde{g} = \text{Decode}(\tilde{b}\_1, \tilde{b}\_2, \dots, \tilde{b}\_m)\$, indicating a joint decoding process.

Figure 1: Compression for Distributed Mean Estimation

The naive strategy of clients sending their vectors  $g_i$  to the server for DME incurs no error, however, has a high communication cost, rendering it useless in most of the real-world network applications. A principled way to tackle this is to use compression: each client  $i \in [m]$  compresses its vector  $g_i$  into an efficient encoding  $\tilde{b}_i \in \mathcal{B}_i$  which can then be sent to the server; The server forms an estimate  $\tilde{g}$  of the mean  $g$  using the encodings  $\{\tilde{b}_i\}_{i \in [m]}$ . We can then compute the error of the estimate  $\tilde{g}$  and the number of bits required to communicate  $\tilde{b}_i$  (i.e.,  $\log_2 |\mathcal{B}_i|$ ) to analyze the efficiency of the compression scheme. As opposed to distributed statistical inference Braverman et al. (2016); Garg et al. (2014), we do not assume that  $g_i$  are sampled from a distribution, and instead the estimation error of these schemes is computed in terms of  $g_i$ .

One way to approach this compression paradigm is when each client compresses its vector oblivious to others, and the server separately decodes the vectors before aggregating (Figure 1a). We call this *independent compression* and several existing works Konečný & Richtárik (2018); Suresh et al. (2017); Safaryan et al. (2021); Gandikota et al. (2022); Vargafik et al. (2021) use such a compression scheme. The simplest example of this scheme is RandK Konečný & Richtárik (2018), where each client sends only  $K \in \mathbb{N}$  coordinates as  $\tilde{b}_i$ , and the server estimates  $\tilde{g}$  as the average of  $K$ -sparse vectors from each client. As  $K < d$ , this scheme requires less communication than sending the full vector  $g_i$  from each client  $i \in [m]$ . Note that independent compressors are a specific class among the more general possible compressors.

However, independent compressors suffer from a significant drawback, especially when the vectors to be aggregated are similar/not-too-far, which is often the case for gradient aggregation in distributed learning. Consider the case when two distinct clients  $i, j \in [m]$  have different vectors  $g_i \neq g_j$ , but they differ in only one coordinate. Then, independent compressors like RandK will end up sending  $\tilde{b}_i$  and  $\tilde{b}_j$  which are very similar (in fact, same with high probability) to each other, and therefore wasting communication.

Collaborative compressors Suresh et al. (2022); Szlendak et al. (2021); Jhunjhunwala et al. (2021); Jiang et al. (2023) can alleviate this problem. Figure 1b describes a collaborative compressor, where the encodings  $\{\tilde{g}_i\}_{i \in [m]}$  may not be independent of each other and a decoding function *jointly* decodes all encodings to obtain the mean estimate  $\tilde{g}$ . Clearly, this opens up more possibilities to reduce communication - but also the error of collaborative compressors can be made to scale as the variance of the vectors instead of their norms. Whereas, in independent compression a lot of communication is also spent in figuring out their norms separately.

The amount of required communication also depends on the metric for estimation error. Among the existing schemes for collaborative compressors, most provide guarantees on the  $\ell_2$  error  $\|\tilde{g} - g\|_2^2$  Suresh et al. (2022); Szlendak et al. (2021); Jhunjhunwala et al. (2021); Jiang et al. (2023). Also, in collaborative compressors, the error must ideally be dependent on *some measure of correlation/distance* among the vectors, which is indeed the case for all of these schemes. In this paper, the measure of such a distance is denoted with  $\Delta$ , with some subscript signifying the exact measure; the vectors in question have high similarity as  $\Delta \rightarrow 0$ . The estimation error naturally grows with the dimension  $d$ , and decays with the number of clients  $m$  (due to an averaging). One of our major contributions is to design a compression scheme that has significantly improved dependence on the number of clients  $m$  to counter the effect of growing dimension  $d$ .

If one were to estimate the unit vector in the direction of the average vector  $\frac{1}{m} \sum_{i=1}^m g_i$ , which is often important for gradient descent applications, using an estimate of the mean with low  $\ell_2$  error can be

{2}------------------------------------------------

| Compressor | Error metric | Error | # Bits/client |
|-|-|-|-|
| NoisySign<br>(Algorithm 1) | $\ \tilde{g} - g\ _\infty$ | $\left(1 - \frac{\Delta_\Phi + \sqrt{\frac{\log m}{m}}}{\alpha(\ g\ _\infty)} (\sqrt{\Delta_\Phi + \alpha(\ g\ _\infty)})\right)^{-1} - 1$ | $d$ |
| HadamardMultiDim<br>(Algorithm 3) | $\mathbb{E}[\ \tilde{g} - g\ _\infty]$ | $\frac{B}{2^{m-1}} + \Delta_{\text{Hadamard}}$ | $d$ |
| SparseReg<br>(Algorithm 4) | $\mathbb{E}[\ \tilde{g} - g\ _2^2]$ | $B^2 \exp\left(-\frac{2m \log L}{d}\right) + \Delta_{\text{reg}}$ | $\log L$<br>( $L \geq 1$ tunable) |
| OneBit<br>(Algorithm 5) | $\arccos(\tilde{g}, g)$ | $\pi(\Delta_{\text{corr}} + \frac{d}{m t})$ | $t$<br>( $t \geq 1$ tunable) |

Table 1: Theoretical results for our proposed collaborative compression schemes.  $\Delta_\Phi, \Delta_{\text{Hadamard}}, \Delta_{\text{reg}}$  and  $\Delta_{\text{corr}}$  are measures of average dissimilarity between vectors  $\{g_i\}_{i \in [m]}$  defined in Theorems 4, 1, 2 and Lemma 1 respectively. For NoisySign,  $\alpha(x) = 1 - \Phi_\sigma(x)$  for any  $x \in \mathbb{R}$ , where  $\Phi_\sigma(x) = \text{erf}(\frac{x}{\sqrt{2}\sigma})$  with erf being the error function Glaisher (1871) and  $\sigma > 0$  is an algorithm parameter. For HadamardMultiDim, we assume  $\|g_i\|_\infty \leq B, \forall i \in [m]$ . For SparseReg, we assume  $\|g_i\|_2 \leq B, \forall i \in [m]$  and  $L$  is an algorithm parameter. For OneBit,  $g$  is the unit vector along the average  $\frac{1}{m} \sum_{i=1}^m g_i$  and  $\tilde{g}$  is also a unit vector.

highly sub-optimal as the  $\ell_2$  error might be large even if all the vectors point in the same direction but have different norms. For this the cosine distance  $\arccos(\frac{\langle \tilde{g}, g \rangle}{\|\tilde{g}\| \|g\|})$  is a better measure, which has not been studied in the literature. We also give a compression scheme specifically tailored for this error metric. Another interesting metric is the  $\ell_\infty$ -error which has also not been studied except for in Suresh et al. (2022). There as well, we give an improved dependence of the estimation error on  $m$ .

Further drawback of existing collaborative compressors such as, Jhunjunwala et al. (2021); Jiang et al. (2023) is that they require the knowledge of correlation between vectors before employing their compression. Without this knowledge, their error guarantees do not hold.

**Notation.** Let  $[n] \equiv \{1, 2, \dots, n\}$ . We use  $g^{(j)}$  to denote the  $j^{\text{th}}$  coordinate of a vector  $g \in \mathbb{R}^d, j \in [d]$ . For a permutation  $\rho$  on  $[m]$ ,  $\rho^{(i)}$  denotes mapping of  $i \in [m]$  under  $\rho$ .

**Our contributions.** We provide four different collaborative compressors, which are communication-efficient, give error guarantees for different error metrics ( $\ell_2$  error,  $\ell_\infty$  error and cosine distance), and exhibit optimal dependence on the number of clients  $m$  and the diameter of ambient space  $B$ . To see the advantage of collaboration, we define few natural similarity metrics. All our schemes show graceful degradation of error with the similarity metric between different clients. Our schemes have three subroutines: Init which corresponds to initial steps, Encode which is performed individually at each client to obtain their encoding  $\tilde{b}_i$  and Decode which is performed at the server on all the encodings to obtain estimate of mean  $\tilde{g}$ .

We now provide our main contributions. The theoretical guarantees for our algorithms are summarized in Table 1.

1. We provide a simple collaborative scheme based on the popular signSGD Bernstein et al. (2018a) scheme, NoisySign (Algorithm 1), where sign of each coordinate of a vector is sent after adding Gaussian noise. An advantage of this scheme, compared to others is that we can infer the vector  $g$  with an  $\ell_\infty$  error guarantee increasing with  $\|g\|_\infty$  and decreasing with  $m$ , without the knowledge of  $\|g\|_\infty$  itself. The dissimilarity is  $\Delta_\Phi = \mathcal{O}(\frac{1}{m\sigma} \sum_{i=1}^m \|g - g_i\|_\infty)$ , where  $\sigma$  is the variance of the noise added (Theorem 4). The details of this scheme is delegated to Appendix A.
2. ( **$\ell_\infty$ -guarantee**) For vectors with  $\ell_\infty$  norm bounded by  $B$ , we propose a collaborative compression scheme, HadamardMultiDim (Algorithm 3) which performs coordinate-wise collaborative binary search. We obtain the best dependence on  $m$  and  $B$  for the  $\ell_\infty$  error ( $\mathcal{O}(B \cdot \exp(-m))$ ) while suffering from an extra error term  $\Delta_{\text{Hadamard}}$ , which is a measure of average dissimilarity between compressed vectors.  $\Delta_{\text{Hadamard}}$  lies in the range  $[\Delta_\infty, \Delta_{\infty, \max}]$  where  $\Delta_\infty = \max_{j \in [d]} \frac{1}{m} \sum_{i=1}^m |g_i^{(j)} - g^{(j)}|$  and  $\Delta_{\infty, \max} = \max_{j \in [d], i \in [m]} |g_i^{(j)} - g^{(j)}|$  (Theorem 1). In Section 2.3, we provide a practical example where value of  $\Delta_{\text{Hadamard}}$  can be approximated and use it compare theoretical guarantees of HadamardMultiDim with those of baselines in Table 2.

{3}------------------------------------------------

3. ( **$\ell_2$ -guarantee**) For vectors with  $\ell_2$  norm bounded by  $B$ , we provide a collaborative compression scheme SparseReg (Algorithm 4) based on Sparse Regression Codes Venkataramanan et al. (2014b;a). We obtain the best dependence on  $B$  and  $m$  for the  $\ell_2$  error ( $\mathcal{O}(B \exp(-m/d))$ ) while compressing to much less than  $d$  bits (in fact, to a constant number of bits) per client. The error consists of a penalty for the dissimilarity,  $\Delta_{\text{reg}}$ , the average dissimilarity between compressed vectors which lies in the range  $[\Delta_2, \Delta_{2, \max}]$  where  $\Delta_2 = \frac{1}{m} \sum_{i=1}^m \|g - g_i\|_2^2$  and  $\Delta_{2, \max} = \max_{x \in [m]} \|g - g_i\|_2^2$  (see, Theorem 2).

4. (**cosine-guarantee**) For unit norm vectors  $\{g_i\}_{i \in [m]}$ , we estimate the unit vector  $g$  in the direction of the average  $\frac{1}{m} \sum_{i=1}^m g_i$ . For this, motivated by one-bit compressed sensing Boufounos & Baraniuk (2008), our collaborative compression scheme, OneBit (Algorithm 5), sends the sign of the inner product between the vector  $g_i$  and a random Gaussian vector. By establishing an equivalence to halfspace learning with malicious noise, we propose two decoding schemes: the first one is based on Shen (2023) which is optimal for halfspace learning but harder to implement and a second one, based on Kalai et al. (2008) which is easy to implement. Both schemes are computationally efficient, and have an extra dissimilarity term in the error,  $\Delta_{\text{corr}} = \frac{1}{m\pi} \sum_{i=1}^m \cos^{-1}(\langle g, g_i \rangle)$ , which is the appropriate dissimilarity between unit vectors (see Theorem 3).

5. (**Experiments**) We perform a simulation for DME with our schemes as the dissimilarities vary and compare the three different error metrics from above with various existing baselines (Fig 2a-2c). We also used our DME subroutines in the downstream tasks of KMeans, power iteration, and linear regression on real (and federated) datasets (Fig 2d-2i). Our schemes have lowest error in all metrics for low dissimilarity regime.

#### --- **Algorithm 1** NoisySign ---

```

Encode ( $g_i$ )
Sample  $\xi_i \sim \mathcal{N}(0, \sigma^2 \mathbb{I}_d)$ 
 $\tilde{b}_i = \text{sign}(g_i + \xi_i)$ 
return  $\tilde{b}_i$ .
Decode ( $\{\tilde{b}_i\}_{i \in [m]}$ )
 $\tilde{g}^{(j)} \leftarrow \Phi^{-1}(\frac{1}{m} \sum_{i=1}^m \tilde{b}_i^{(j)}), j=1, \dots, d$ 
return  $\tilde{g}$ 

```

---

#### --- **Algorithm 2** Hadamard1DEnc ---

```

Input: Scalar  $s$ , Level  $K$ 
 $S_{\bar{K}} = \bigcup_{k=0}^{K-1} [-B + \frac{2kB}{2^{2k-1}}, -B + \frac{(2k+1)B}{2^{2k-1}}]$ 
return  $-1$  if  $s \in S_{\bar{K}}$  else  $+1$ 

```

---

#### --- **Algorithm 3** HadamardMultiDim ---

```

Init()
Clients and server share  $\rho$ , a random permutation on  $[m]$ .
Encode ( $g_i$ )
for  $j \in [d]$  do
   $\tilde{b}_i^{(j)} \leftarrow \text{Hadamard1DEnc}(g_i^{(j)}, \rho^{(j)})$ 
end for
return  $\tilde{b}_i$ 
Decode ( $\{\tilde{b}_i\}_{i \in [m]}$ )
for  $j \in [d]$  do
   $\tilde{g}^{(j)} = \sum_{i=1}^m \tilde{b}_i^{(j)} \cdot \frac{B}{2^{\rho^{(j)}-1}}$ 
end for
return  $\tilde{g}$ 

```

---

**Organization.** In the next subsection, we present related works in distributed mean estimation. The NoisySign algorithm is given in Algorithm 1, and its analysis can be found in Appendix A. In Section 2, we present the two schemes obtaining optimal dependence on  $m$ , HadamardMultiDim in Subsection 2.1 and SparseReg in Subsection 2.2. In Section 3, we analyze the OneBit compression scheme. Finally, in Section 4, we provide experimental results for our schemes.

### 1.1 RELATED WORKS

**Compressors in Distributed Learning.** Starting from Konečný et al. (2016) most compression schemes in distributed learning involve either quantization or sparsification. In quantization schemes, the real valued input space is quantized to specific levels, and each input is mapped to one of these quantization levels. A theoretical analysis for unbiased quantization was provided in Alistarh et al. (2017). Subsequently, the distributed mean estimation problem with limited communication was formulated in Suresh et al. (2017) where two schemes, stochastic rotated quantization (SRQ) and variable length coding, were proposed. These schemes matched the lower bound for communication and  $\ell_2$  error in terms of  $\bar{B}^2 = \frac{1}{m} \sum_{i=1}^m \|g_i\|_2^2$ . Performing a coordinate-wise sign is also a quantization operation, introduced in Bernstein et al. (2018b). Further advances in quantization include multiple quantization

{4}------------------------------------------------

| Compressor | Error | # Bits/client | Notes |
|-|-|-|-|
| RandK Konečný & Richtárik (2018) | $\mathcal{O}(\frac{d}{K} \bar{B}^2)$ | $32K + K \log d$ | Independent |
| SRQ Suresh et al. (2017) | $\mathcal{O}(\frac{\log d}{m(K-1)^2} \bar{B}^2)$ | $Kd$ | Independent |
| Kashin Safaryan et al. (2021) | $\mathcal{O}(\frac{10\sqrt{\Delta_2}}{\sqrt{K}})^4 \bar{B}^2$ | $31 + \lambda d$ | Independent |
| Drive Vargafitik et al. (2021) | $\mathcal{O}(\bar{B}^2)$ | $32 + d$ | Independent |
| PermK Szlendak et al. (2021) | $\mathcal{O}((1 - \max\{0, \frac{m-d}{m-1}\}) \Delta_2)$ | $32K + K \log d$ | Collaborative |
| RandKSpatial Jhunjhunwala et al. (2021) | $\mathcal{O}(\frac{d}{mK} \Delta_2)$ | $32K + K \log d$ | Needs Correlation |
| RandKSpatialProj Jiang et al. (2023) | $\mathcal{O}(\frac{d}{mK} \Delta_2)$ | $32K + K \log d$ | Needs Correlation |
| Correlated SRQ Suresh et al. (2022) | $\mathcal{O}(\frac{1}{m} \min\{\frac{\sqrt{d\Delta_2}}{K} B, \frac{d\bar{B}^2}{K^2}\})$ | $2d \log K + K \log d$ | $\ g_i\ _2 \leq B, \forall i \in [m]$ |

Table 2: Comparison of existing independent and collaborative compressors in terms of  $\ell_2$  error and bits communicated.  $K$  is the number of coordinates communicated for sparsification methods(RandK, PermK, RandKSpatial, RandKSpatialProj) and the number of quantization levels for quantization methods (SRQ, vqSGD, Correlated SRQ). The constant  $\lambda$  is a parameter of the Kashin scheme. Further,  $\bar{B}^2 = \frac{1}{m} \sum_{i=1}^m \|g_i\|_2^2$ ,  $\Delta_2 = \frac{1}{m} \sum_{i=1}^m \|g_i - g\|_2^2$ , and  $\Delta_\infty = \max_{j \in [d]} \frac{1}{m} \sum_{i=1}^m |g_i^{(j)} - g^{(j)}|$ . It is also assumed that a real is equivalent to 32 bits, which is an informal norm in this literature.

levels Wen et al. (2017), probabilistic quantization with noise Chen et al. (2020); Jin et al. (2021); Safaryan & Richtarik (2021), vector quantization Gandikota et al. (2022), and applying structured rotation before quantization Vargafitik et al. (2021); Safaryan et al. (2021). Sparsification involves selecting only a subset of coordinates to communicate. Common examples include RandK Konečný & Richtárik (2018), TopK Stich et al. (2018) and their combinations Beznosikov et al. (2022). Note, for all independent compressors, the  $\ell_2$  error scales as  $\bar{B}^2$ .

**Collaborative Compressors.** PermK Szlendak et al. (2021) was the first collaborative compressor, where each client would send a different set of  $K$  coordinates. Their error scales with the empirical variance,  $\Delta_2 = \frac{1}{m} \sum_{i=1}^m \|g_i - g\|_2^2$ . If  $\Delta_2$  is known, or one of the vectors  $g_i$  is known, the lattice-based quantizer in Davies et al. (2021) and correlated noise based quantizer in Mayekar et al. (2021) obtains  $\ell_2$  error in terms of  $\Delta_2$ . Further, RandKSpatial Jhunjhunwala et al. (2021) and RandKSpatialProj Jiang et al. (2023) utilize the correlation information to obtain the correct normalization coefficients for RandK with rotations, obtaining guarantees in terms of  $\Delta_2$ . In absence of correlation information, they propose a heuristic. A quantizer also based on correlated noise, was proposed in Suresh et al. (2022) which achieves the lower bound for scalars. However, for  $d$ -dimensional vectors of  $\ell_2$ -norm at most  $B$ , their dependence on dimension  $d$  and number of clients  $m$  can be improved by our schemes.

We provide a summary of existing compressors in Table 2, along with their error guarantees.

## 2 OPTIMAL DEPENDENCE ON $m$

If  $\|g\|_\infty$  or  $\|g\|_2$  is bounded, we can obtain an almost optimal exponential decay with  $m$ . We provide two schemes that obtain optimal  $\ell_\infty$  (by modifying the sign compressor) and  $\ell_2$  error dependence in terms of  $m$  and the diameter of the space  $B$ .

### 2.1 HADAMARDMULTIDIM

When the vectors have bounded  $\ell_\infty$  norm, instead of obliviously using the sign compressor on every coordinate on every client, one may be able to divide their range and cleverly select bits to encode the most information. We call our algorithm Hadamard scheme, because the binary-search method involved is akin to the rows of a Hadamard-type matrix.

**Assumption 1** (Bounded domain).  $\|g_i\|_\infty \leq B, \forall i \in [m]$ .

This would imply that for any  $j \in [d]$ ,  $g_i^{(j)} \in [-B, B], \forall i \in [m]$ . Now, consider the  $i^{\text{th}}$  client and the scalar  $g_i^{(j)}$  and assume that we are allowed to encode this using  $m$  bits. The best error that we can achieve is  $\frac{B}{2^{m-1}}$ , by performing a binary search on the range  $[-B, B]$  for  $g_i^{(j)}$ , sending one bit per level of the binary search. However, this scheme is not collaborative. To obtain a collaborative scheme, for some permutation  $\rho$  on the set of clients  $[m]$ , the  $i^{\text{th}}$  client can perform binary search until level  $\rho^{(i)}$

{5}------------------------------------------------

and sends its decision at level  $\rho^{(j)}$ . In this case, each client sends only 1 bit per coordinate. To decode  $\tilde{g}^{(j)}$  we take a weighted sum of the signs obtained from different clients weighed by their coefficients  $\frac{B}{2^{\rho^{(j)}-1}}$ . This is the core subroutine (Algorithm 2). The full compression scheme for  $d$  coordinates applies this coordinate-wise in Algorithm 3. Note that, the clients and the server should share the permutation  $\rho$  before encoding and decoding, which need not change over different instantiations of the mean estimation problem. To understand the core idea of the scheme, consider the case when all vectors  $g_i = g$ . Then, sending a different level from a different client is equivalent to doing a full binary search to quantize  $g$ . As long as  $g_i$ s are close to  $g$ , we hope that this scheme should give us a good estimate of  $g$ . Suppose,  $\tilde{b}_{i,k}^{(j)}$  denotes the encoding of  $g_i^{(j)}$  at level  $k \forall i, k \in [m], j \in [d]$ .

**Theorem 1** (HadamardMultiDim Error). *Under Assumptions 1, the estimation error for Algorithm 3 is*

$$\mathbb{E}[\|\tilde{g} - g\|_\infty] \leq \frac{B}{2^{m-1}} + \min\{\Delta_{\text{Hadamard}}, \Delta_{\infty, \max}\}, \quad (2)$$

where  $\Delta_{\text{Hadamard}} \equiv \max_{r \in [d]} \sqrt{\frac{1}{m^2} \sum_{1 \leq i \neq j \leq m} \sum_{k=1}^m \left( \frac{B(\tilde{b}_{i,k}^{(r)} - \tilde{b}_{j,k}^{(r)})}{2^{k-1}} \right)^2}$ , and  $\Delta_{\infty, \max} \equiv \max_{r \in [d], i \in [m]} |g_i^{(r)} - g^{(r)}|$ .

We provide the proof for this theorem in Appendix D.1. The first term corresponds to the error for binary search, and has an exponential decay with number of clients. In contrast, all previous schemes give  $\text{poly}(1/m)$  dependence (see, Table 2). The second term is the price we pay for dissimilarity between the vectors. The term  $\Delta_{\text{Hadamard}}$  is the average of the pairwise difference between the encodings at each level. As long as vectors  $g_i$  and  $g_j$  are similar and their encodings do not differ on a lot of levels,  $\Delta_{\text{Hadamard}}$  is small. The following is an interpretable bound on  $\Delta_{\text{Hadamard}}$ .

$$\Delta_{\text{Hadamard}} \geq \frac{1}{\sqrt{3}} \Delta_{\infty} - \sqrt{\frac{2(m-1)}{m}} \frac{B}{2^{m-1}}, \quad (3)$$

where  $\Delta_{\infty} \equiv \max_{r \in [d]} \frac{1}{m} \sum_{i=1}^m |g_i^{(r)} - g^{(r)}|$ . The proof of this is provided in Appendix D.2. As we allow full collaboration between clients, in the worst case, we might have to incur a cost  $\Delta_{\infty, \max}$  which is the worst case dissimilarity among clients. However, if client vectors are close, we might end up paying a much lower cost.

#### Algorithm 4 SparseReg

---

```

Init()
Clients and server share  $A \in \mathbb{R}^{mL \times d}$ , and  $\rho$ , a
random permutation on  $[m]$ 
Encode ( $g_i$ )
 $g_i' \leftarrow g_i$ 
for  $j \in [\rho^{(i)}]$  do
   $b_{i,j} \leftarrow \text{argmax}_{r \in [L]} \langle A_{(j-1)L+r}, g_i' \rangle$ 
   $g_i' \leftarrow g_i' - c_j A_{(j-1)L + b_{i,j}}$ 
end for
 $\tilde{b}_i \leftarrow \tilde{b}_{i, \rho^{(i)}}$ 
return  $\tilde{b}_i$ 
Decode ( $\{\tilde{b}_i\}_{i \in [m]}$ )
 $\tilde{g} \leftarrow \sum_{i \in [m]} c_{\rho^{(i)}} A_{(\rho^{(i)}-1)L + \tilde{b}_i}$ 

```

---

$$c_i = B \sqrt{\frac{2 \log L}{d^2} \left( 1 - \frac{2 \log L}{d} \right)^{i-1}} \quad (4)$$

#### Algorithm 5 OneBit

---

```

Init()
Clients and server share unit vectors  $\{z_i\}_{i \in [m]}$ .
Encode ( $g_i$ )
 $\tilde{b}_i \leftarrow \text{sign}(\langle g_i, z_i \rangle)$ 
return  $\tilde{b}_i$ 
Decode ( $\{\tilde{b}_i\}_{i \in [m]}$ )
 $g' \leftarrow \begin{cases} (\text{Shen, 2023, Algorithm 1})(\text{Tech. I}) \\ \frac{1}{m} \sum_{i=1}^m z_i \tilde{b}_i (\text{Tech. II}) \end{cases}$ 
 $\tilde{g} \leftarrow g' / \|g'\|_2$ 

```

---

### 2.2 SPARSE REGRESSION CODING

In this part, we extend the coordinate-wise guarantee of the HadamardMultiDim to  $\ell_2$  error between  $d$ -dimensional vectors of bounded  $\ell_2$ -norm.

**Assumption 2** (Norm Ball).  $\|g_i\|_2 \leq B, \forall i \in [m]$ .

To extend the idea of binary search and full collaboration from HadamardMultiDim, we first need a compression scheme which performs binary search on  $d$  dimensional vectors with  $\ell_2$  error guarantees.

{6}------------------------------------------------

Sparse Regression codes Venkataramanan et al. (2014b;a), which are known to achieve rate-distortion function for a Gaussian source, fit our requirements. Let  $A \in \mathbb{R}^{mL \times d}$  for some parameter  $L > 0$ , where each element of  $A$  is sampled iid from  $\mathcal{N}(0, 1)$  and  $A_k$  denotes the  $k$ th row of  $A$ . The full algorithm SparseReg is presented in Algorithm 4. To compress a single vector  $g$  using  $m \log L$  bits, we find the closest vector to  $g$  in the first  $L$  rows of  $A$ ; say the index of this vector is  $b_1$ . Similar to binary search, we subtract  $c_1 A_{b_1}$  from  $g$ , where  $c_1$  is given in (4) to obtain an updated  $g$ . We repeat the process using the next set of  $L$  rows. Here, each set of  $L$  rows corresponds to a single level of binary search, with the coefficients  $c_2$  obtained from Eq (4) having a decaying exponent. By carefully selecting the parameters in the proof of (Venkataramanan et al., 2014b, Theorem 1), we can show that this scheme obtains  $\ell_2$  error  $B \exp(-m)$ . We extend this scheme to all clients to allow full collaboration in a manner similar to HadamardMultiDim. Each client  $i \in [m]$  encodes at level  $\rho^{(i)}$  where  $\rho$  is a permutation on  $[m]$  and the server computes the weighted sum of the encodings from each client with corresponding coefficients  $c_{\rho(i)}$ .

**Theorem 2 (SparseReg Error).** *Under Assumption 2, there exists a matrix  $A$  and constants  $\delta_1, \delta_2 > 0$ , such that the estimation error of Algorithm 4 is*

$$\mathbb{E}_\rho [\|g - \tilde{g}\|_2^2] \leq B^2 \left(1 + \frac{10 \log L}{d}\right) \exp\left(\frac{m \log L}{d}\right) (\delta_1 + \delta_2)^2 \left(1 - \frac{2 \log L}{d}\right)^m + \min\{\Delta_{\text{reg}}, \Delta_{2, \max}\}$$

$$\text{where, } \Delta_{\text{reg}} \equiv \frac{1}{m^2} \sum_{i, j \in [m], i \neq j} \sum_{k=1}^m c_k^2 \|A_{(k-1)L + b_{i,k}} - A_{(k-1)L + b_{j,k}}\|_2^2, \quad \Delta_{2, \max} \equiv \max_{i \in [m]} \|g - g_i\|_2^2.$$

In fact, a Gaussian matrix  $A$  satisfy this with probability  $1 - 2m^2 L \exp(-d\delta_1^2/8) - m \left(\frac{L^{2\delta_2}}{\log L}\right)^{-m}$ .

For  $d = \Omega(\log m)$ , the probability above can be made arbitrarily close to 1 for large  $m$ . The proof is provided in Appendix D.3. Similar to HadmardMultiDim, the first term has an exponential dependence in  $m$  and is obtained from the existing results of Sparse Regression Codes from Venkataramanan et al. (2014b). In terms of  $\ell_2$  error this dependence on  $m$  is better than all the prior methods.

The dissimilarity term  $\Delta_{\text{reg}}$  has a similar structure to  $\Delta_{\text{Hadamard}}$  as it is the pairwise difference between encodings of two different vectors at all levels. As long as the vectors are close to each other, this term is not large. Similar to Equation (3), we can interpret  $\Delta_{\text{reg}}$  with the following lower bound for Gaussian matrices with the probability given above.

$$\Delta_{\text{reg}} \geq \frac{1}{3} \Delta_2 - 2B^2 \left(1 + \frac{10 \log L}{d}\right) \exp\left(\frac{m \log L}{d}\right) (\delta_1 + \delta_2)^2 \left(1 - \frac{2 \log L}{d}\right)^m, \quad (5)$$

where  $\Delta_2 \equiv \frac{1}{m} \sum_{i=1}^m \|g_i - g\|_2^2$ . The proof of this is provided in Appendix D.4. If the vectors are close to each other we might incur the worst possible error  $\Delta_{2, \max}$ , but if they are close, we will pay an average price in terms of  $\Delta_{\text{reg}}$ .

While both the HadmardMultiDim and SparseReg schemes achieve very low communication rate, that comes at the price of  $O(m)$  computing in the Encode step. This higher cost in computing is to be expected when one wants to exploit the full potential of collaborative compression (e.g., Jiang et al. (2023), where the Decode step takes  $O(m^2)$  time).

### 2.3 MOTIVATING EXAMPLE

We now provide an example to show that for practical scenarios, the error terms  $\Delta_{\text{reg}}$  and  $\Delta_{\text{Hadamard}}$  are much smaller than their worst case values. Consider the scenario of Theorem 1 ( $\ell_\infty$  error) and set  $d=1$ . Assume that the first  $c$  vectors are  $g'_1$  and the remaining  $m-c$  vectors are  $g'_2$ , for some constant  $c \ll m$ . In this case,  $\Delta_{\infty, \max} = (1 - \frac{c}{m}) |g'_1 - g'_2| \approx |g'_1 - g'_2|$ , while  $\Delta_{\infty} \approx \frac{c}{m} |g'_1 - g'_2|$ . In this scenario, if the compressed values  $\tilde{b}$  for  $g'_1$  and  $g'_2$  according to the HadamardMultiDim differ at  $k \in \mathcal{K} \subseteq [m]$  levels, then,  $\Delta_{\text{Hadamard}} \approx \sqrt{\frac{c}{m} \sum_{k \in \mathcal{K}} (B/2^{k-1})^2} \leq \sqrt{\frac{c}{m} \min_{k \in \mathcal{K}} \frac{B}{2^{k-1}}}$ . As  $\Delta_{\text{Hadamard}}$  averages over all machines, it decreases with  $m$  similar to  $\Delta_2$  and should be much smaller than  $\Delta_{\infty, \max}$ . The only case when it is not smaller than  $\Delta_{\infty, \max}$  is when  $g'_1$  and  $g'_2$  are very close, so that  $\Delta_{\infty, \max} = O(\sqrt{m^{-1}})$ , but the first level where they differ ( $\min_{k \in \mathcal{K}} k$ ) is very small. One such example is when the quantized values of  $g'_1$  in the set  $\mathcal{K}$  sorted by the levels in increasing order are  $(+1, -1, -1, -1)$  and that of  $g'_2$  are  $(-1, +1, +1, +1)$ . As the vectors are extremely close in this case, the estimation error with  $\Delta_{\infty, \max}$

{7}------------------------------------------------

is not very large. Further, if we assume a distributional assumption on the vectors  $g_i$ , similar to how we generate Figure 2b, obtaining vectors where  $\Delta_{\text{Hadamard}} > \Delta_{\infty, \max}$ , happens with low probability. Note that a similar example can be constructed for the SparseReg scheme.

We use this example to further compare the error of our proposed schemes to baselines mentioned in Table 2. Consider any  $\ell_2$  compressor whose error is either proportional to  $\Lambda \tilde{B}^2$  or  $\Lambda \Delta_2$  and it sends  $\lambda$  bits/client for some  $\lambda, \Lambda > 0$ . The  $\ell_2$  error is defined as  $\mathbb{E}[\|\tilde{g} - g\|_2^2]$  and the  $\ell_\infty$  error is defined as  $\mathbb{E}[\|\tilde{g} - g\|_\infty]$ , therefore the corresponding  $\ell_\infty$  error of these compressors is  $\sqrt{\Lambda \tilde{B}}$  or  $\sqrt{\Lambda \Delta_2}$ . Now, consider the example which we just presented with  $d > 1$  and all coordinates being equal for each vector. Therefore,  $\Delta_2 \approx \frac{cd}{m} |g'_2 - g'_1|^2$ , and plugging this in, the  $\ell_2$  error of the schemes is  $\sqrt{\Lambda \tilde{B}}$  or  $\sqrt{\Lambda \frac{cd}{m} |g'_2 - g'_1|}$ . HadamardMultiDim sends  $d$  bits/client, therefore, to compare with any of these schemes, we set  $\lambda = d$ .

For RandK, this would mean setting  $K = \frac{d}{32 + \log d}$ . Now, if  $|g'_1|, |g'_2| \approx B$  but  $|g'_2 - g'_1| \ll B$ , then  $\tilde{B} \approx \sqrt{dB}$ . Using these approximations, the error of RandK is  $\sqrt{(32 + \log d)dB}$ , as  $\Lambda = 32 + \log d$ . This is much larger than the  $\ell_\infty$  error of HadamardMultiDim, as the first term is  $B \cdot 2^{m-1}$  and the second term  $\Delta_{\text{Hadamard}} \approx \sqrt{\frac{c}{m} |g'_2 - g'_1|}$ . A similar argument holds for all independent compression schemes, as their  $\ell_\infty$  error scales as  $\tilde{B}$  which in the worst case is  $\sqrt{dB}$ .

For compressors whose error scales as  $\Lambda \Delta_2$  (PermK, RandKSpatial, RandKSpatialProj), by setting  $K = \frac{d}{32 + \log d}$ , we obtain the same number of bits/client as HadamardMultiDim scheme. Consider RandKSpatialProj, where  $\Lambda = \frac{32 + \log d}{m}$ , and the error for our example is  $\sqrt{c \frac{(32 + \log d)d}{m^2} |g'_2 - g'_1|}$ . As long as  $d > m$ , this error is larger than  $\Delta_{\text{Hadamard}}$  by constant terms. A similar argument holds for RandKSpatial and PermK. Additionally, note that the theoretical guarantees for RandKSpatial and RandKSpatialProj do not hold if the correlation is not known, as it is required in the algorithm. Without this information, the heuristics they use do not result in theoretical guarantees and their error might become similar to the error of RandK.

The CorrelatedSRQ compressor achieves the lower bound for collaborative compressors for  $d = 1$ , and is based on a coordinate-wise scheme, hence the  $\Delta_\infty$  in its error guarantees. However, for  $d \gg 1$ , its error scales poorly. For the example described above,  $\|\tilde{g}_i\|_2 \leq \sqrt{dB}$ , therefore, the  $\ell_\infty$  error for CorrelatedSRQ is  $\sqrt{\frac{1}{m} \min\{\frac{d\Delta_2}{K}, \frac{dB^2}{K^2}\}}$ . Note that even for  $K = 2$ , correlated SRQ requires double the number of bits/client as HadamardMultiDim. Note that the first term of HadamardMultiDim is  $B \cdot 2^{m-1}$  which is much smaller than any of these terms, while  $\Delta_{\text{Hadamard}} \approx \sqrt{\frac{m}{c} \Delta_\infty}$  for our example. Therefore, as long as  $\left(\frac{m^2 K}{cdB}\right)^{1/(2d-1)} < \Delta_\infty < \frac{\sqrt{cdB}}{mK}$ ,  $\Delta_{\text{Hadamard}}$  is smaller than  $\ell_\infty$  error of CorrelatedSRQ. The size of this interval for  $\Delta_\infty$  increases as  $d$  increases.

With the above example and analysis, we have specified the exact scenarios when HadamardMultiDim outperforms baselines and this can be easily extended to SparseReg.

## 3 ONE-BIT SCHEMES

In this section, our vectors are assumed to belong on the unit sphere  $\mathbb{S}^{d-1}$ . Further, our goal is to recover the unit vector in the direction of the average vector  $g = (\frac{1}{m} \sum_{i \in [m]} g_i) / \|\frac{1}{m} \sum_{i \in [m]} g_i\|_2$ .

**Assumption 3** (Unit vectors).  $g_i \in \mathbb{S}^{d-1}, \forall i \in [m]$ .

Consider the collaborative compressor where each client has sample  $z_i \sim \text{Unif}(\mathbb{S}^{d-1})$  (which are also available to the server a priori). Client  $i$  sends the single bit  $b_i = \text{sign}(\langle g_i, z_i \rangle)$  to the server. To recover  $g$ , consider the trivial case when all vectors  $g_i$ s were equal. Then, each  $b_i = \text{sign}(\langle g, z_i \rangle)$ , and to recover  $g$ , the server needs to learn the halfspace corresponding to  $g$  from a set of  $m$  labeled datapoints. Applying the same method to when  $g_i$ s are not all the same, we can estimate  $g$  by solving the following optimization problem.

$$\min_{\tilde{g} \in \mathbb{S}^{d-1}} \frac{1}{m} \mathbf{1}(\tilde{b}_i \neq \text{sign}(\langle z_i, \tilde{g} \rangle)). \quad (6)$$

Here,  $\mathbf{1}(\cdot)$  denotes the indicator function. We can intuitively view (6) as a halfspace learning problem with a groundtruth  $g$ , but in the presence of noise, as  $g_i \neq g$ . Learning halfspaces in the presence of

{8}------------------------------------------------

noise is hard in general Guruswami & Raghavendra (2006). In our setting, if we sample  $z_i$  from the intersection of the halfspaces with normal vectors  $g$  and  $g_i$ , then the label is  $\text{sign}(\langle g, z_i \rangle)$ , otherwise, it is  $-\text{sign}(\langle g, z_i \rangle)$ . We can consider this to be under the malicious noise model, wherein a fraction of datapoints are corrupted.

**Lemma 1 (Malicious Noise).** *If  $z_i \sim \text{Unif}(\mathbb{S}^{d-1})$  and  $\tilde{b}_i = \text{sign}(\langle z_i, g_i \rangle)$ ,  $\forall i \in [m]$ , then, with probability  $1 - \mathcal{O}(\exp(-m\Delta_{\text{corr}}))$ ,  $\zeta$ , the fraction of the set of datapoints  $\{(z_i, \tilde{b}_i)\}_{i \in [m]}$  satisfying  $\text{sign}(\langle z_i, g_i \rangle) \neq \text{sign}(\langle g, z_i \rangle)$  is equal to  $\Theta(\Delta_{\text{corr}})$ , where  $\Delta_{\text{corr}} \triangleq \frac{1}{m\pi} \sum_{i=1}^m \arccos(\langle g_i, g \rangle)$ .*

The proof of the lemma is provided in Appendix E.1. Our methods will use  $\Delta_{\text{corr}}$  to measure the deviation between clients. For small  $\Delta_{\text{corr}}$ , we obtain better performance. If  $\langle g, g_i \rangle \geq 0, \forall i \in [m]$ , then

$$\cos(\pi\Delta_{\text{corr}}) \geq \sqrt{\frac{1}{m} + \frac{2}{m^2} \sum_{1 \leq i < j \leq m} \langle g_i, g_j \rangle}. \quad (7)$$

The proof of the above remark is provided in Appendix E.3.

As long as the corruption level,  $\zeta < \frac{1}{2}$ , we can hope to recover the halfspace  $g$ . We provide two techniques – Techniques I and II, to recover  $g$ , thus yielding two corresponding Decode procedures.

The first decoding procedure (Technique I) is a linear time algorithm for halfspace learning in the presence of malicious noise (Shen, 2023, Theorem 3) that provides obtaining optimal sample complexity and noise tolerance.

**Theorem 3 (Error of Technique I).** *If  $\zeta$  defined in Lemma 1 is less than  $\frac{1}{2}$ , after running Algorithm 5 with Technique I, with probability  $1 - \delta - \mathcal{O}(\exp(-m\Delta_{\text{corr}}))$ , we obtain a hyperplane  $\tilde{g}$  such that,  $\langle \tilde{g}, g \rangle \geq \cos(\pi(\Delta_{\text{corr}} + \frac{d}{m}))$ .*

The algorithm itself is fairly complicated. It assigns weights to different points based on how likely they are to be corrupted. The algorithm proceeds in stages, wherein each stage decreases the weights of the corrupted points and solves the weighted version of (6). The key technique is to use matrix multiplicative weights update (MMWU) Arora et al. (2012) to yield linear time implementation of both these steps, instead of Awasthi et al. (2017) which used polynomial time linear programs for this purpose.

Technique II is the simple average algorithm of Servedio (2002), which obtains suboptimal error guarantees. We defer the details of this to Appendix B and the proofs are provided in Appendix E.

## 4 EXPERIMENTS

**Setup.** To compare the performance of our proposed algorithms, we perform DME for three different distributions which correspond to the three error metrics covered by our schemes –  $\ell_2, \ell_\infty$  and cosine distance. Then, we run our algorithms as the DME subroutine for three different downstream distributed learning tasks – KMeans, power iteration and linear regression. KMeans and power iteration are run on MNIST LeCun & Cortes (2010) and FEMNIST Caldas et al. (2018) datasets and we report the KMeans cost and top eigenvalue as the metrics. For linear regression, we run gradient descent on UJIndoorLoc Torres-Sospedra et al. (2014) and a Synthetic mixture of regressions dataset, with low dissimilarity between the mixture components, and report the test MSE. We compare against all baselines in Table 2 for 3 random seeds and report the methods which perform the best in Fig 2. Additional details for our experimental setup are deferred to Appendix F.

**Results.** *Distributed Mean Estimation.* From Fig 2a and 2b, HadamardMultiDim and SparseReg, whose error is optimal in  $m$ , obtain the best performance in terms of  $\ell_\infty$  and  $\ell_2$  error for low dissimilarity. Especially, for HadamardMultiDim in Fig 2b, the gap in  $\ell_\infty$  error to next best scheme is very large. NoisySign obtains competitive performance to other baselines as we use a large  $\sigma$ . The performance of OneBit for cosine distance metric (Fig 2c) shows that compressors with  $\ell_2$  error guarantees perform poorly in terms of cosine distance. For all collaborative compression schemes, including our proposed schemes, performance degrades as dissimilarity increases. From Fig 2a and 2b, the rate of this decrease is more severe for SparseReg than HadamardMultiDim. For large dissimilarity, HadamardMultiDim and SparseReg can perform worse than certain baselines.

*KMeans and Power iteration.* For MNIST dataset, where dissimilarity is low, HadamardMultiDim performs best for KMeans and close to the best baseline for power iteration (Fig 2d and 2e). Most of

{9}------------------------------------------------

![Figure 2: Performance of DME, KMeans, Power iteration, and linear regression for various datasets and error metrics. The figure consists of nine subplots (a-i) showing the performance of different communication schemes (NoisySign, OneBitAvg, RandK, SparseReg, HadamardMultiDim, QuantSparsify, QuantAvg, QuantReg, QuantCosine, QuantHadamard, QuantLinear, QuantSpectral, QuantOrthogonal, QuantDiagonal, QuantHadamardOrthogonal, QuantHadamardDiagonal, QuantHadamardSpectral, QuantHadamardOrthogonalDiagonal) across different datasets (MNIST, FEMNIST, UJIIndoorLoc, Synthetic) and error metrics (L2, L-infinity, cosine distance).](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 2 displays the performance of various communication schemes across different datasets and error metrics. The subplots are arranged in a 3x3 grid:

- (a) DME with  $\ell_2$  error: Performance on Gaussian vectors. The y-axis is  $\ell_2$  error (log scale,  $10^{-1}$  to  $10^1$ ) and the x-axis is Dissimilarity ( $10^{-1}$  to  $10^1$ ).
- (b) DME with  $\ell_\infty$  error: Performance on vectors from hypercube. The y-axis is  $\ell_\infty$  error (log scale,  $10^{-1}$  to  $10^1$ ) and the x-axis is Dissimilarity ( $10^{-1}$  to  $10^1$ ).
- (c) DME for cosine distance: Performance on a dataset. The y-axis is Cosine distance (log scale,  $10^{-1}$  to  $10^1$ ) and the x-axis is Dissimilarity ( $10^{-1}$  to  $10^1$ ).
- (d) KMeans on MNIST: Performance on MNIST dataset. The y-axis is KMeans cost (log scale,  $1.2 \times 10^7$  to  $2.0 \times 10^7$ ) and the x-axis is Iterations (0 to 25).
- (e) Power iteration on MNIST: Performance on MNIST dataset. The y-axis is Top Eigenvalue (0 to 5) and the x-axis is Iterations (0 to 30).
- (f) Lin. Reg. on UJIIndoorLoc: Performance on UJIIndoorLoc dataset. The y-axis is Loss (log scale,  $10^0$  to  $10^2$ ) and the x-axis is Iterations (0 to 50).
- (g) KMeans on FEMNIST: Performance on FEMNIST dataset. The y-axis is KMeans cost (log scale,  $0.90 \times 10^7$  to  $1.30 \times 10^7$ ) and the x-axis is Iterations (0 to 30).
- (h) Power iteration on FEMNIST: Performance on FEMNIST dataset. The y-axis is Top Eigenvalue (0 to 1.2) and the x-axis is Iterations (0 to 30).
- (i) Lin. Reg. on Synthetic: Performance on Synthetic dataset. The y-axis is Loss (log scale,  $1.00 \times 10^0$  to  $1.50 \times 10^0$ ) and the x-axis is Iterations (0 to 50).

Figure 2: Performance of DME, KMeans, Power iteration, and linear regression for various datasets and error metrics. The figure consists of nine subplots (a-i) showing the performance of different communication schemes (NoisySign, OneBitAvg, RandK, SparseReg, HadamardMultiDim, QuantSparsify, QuantAvg, QuantReg, QuantCosine, QuantHadamard, QuantLinear, QuantSpectral, QuantOrthogonal, QuantDiagonal, QuantHadamardOrthogonal, QuantHadamardDiagonal, QuantHadamardSpectral, QuantHadamardOrthogonalDiagonal) across different datasets (MNIST, FEMNIST, UJIIndoorLoc, Synthetic) and error metrics (L2, L-infinity, cosine distance).

Figure 2: Performance of DME(Distributed Mean Estimation), KMeans, Power iteration and linear regression for the same communication budget. For each experiment, we report the best compressors. Lin. Reg. refer to Linear Regression. For power iteration, higher top eigenvalue is better. For all other experiments, we report the error, so lower is better.

our collaborative compression schemes do not perform as well as RandK on FEMNIST, due to higher client dissimilarity. OneBit is very communication-efficient, so running it for the same communication budget as our baselines ensures that it still remains competitive for KMeans(Fig 2g).

**Linear Regression.** From Fig 2f and 2i, all collaborative compressors perform better than independent compressors as UJIIndoorLoc and synthetic datasets have low dissimilarity among clients as compared to FEMNIST. Our schemes can take full advantage of this low dissimilarity, so HadamardMultiDim and OneBit outperform baselines on both datasets. As the Synthetic dataset has lower dissimilarity than UJIIndoorLoc, even the NoisySign performs better than other baselines, and SparseReg obtains best performance.

## 5 CONCLUSION

We proposed four communication-efficient collaborative compression schemes to obtain error guarantees in  $\ell_2$ -error (SparseReg),  $\ell_\infty$ -error (NoisySign, HadamardMultiDim) and cosine distance (OneBitAvg). The estimation error of our schemes improves with number of clients, and degrades with dissimilarity between clients. Our schemes are biased and our dissimilarity metrics ( $\Delta_{\text{reg}}$ ,  $\Delta_{\text{Hadamard}}$ ) depend on the quantization levels. However, these can be improved by using existing techniques for converting biased compressors to unbiased ones Beznosikov et al. (2022) and adding noise before quantization Tang et al. (2023); Chzhen & Schechtman (2023). Lower bounds for collaborative compressors in terms of their dissimilarity metrics will allow us to assess the optimality of our schemes.

 Rest of paper (reference and Appendix) is removed.