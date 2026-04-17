# Towards Learning Pomdps Without Full Observability

Anonymous authors Paper under double-blind review

## Abstract

We are interested in enabling autonomous agents to learn and reason about systems with hidden states, such as furniture with hidden locking mechanisms. We cast this problem as learning the parameters of a discrete Partially Observable Markov Decision Process (POMDP). The agent begins with knowledge of the POMDP's actions and observation spaces, but not its state space, transitions, or observation models. These properties must be constructed from actionobservation sequences. Spectral approaches to learning models of partially observable domains, such as learning Predictive State Representations (PSRs), are known to directly estimate the number of hidden states. These methods cannot, however, yield direct estimates of transition and observation likelihoods, which are important for many downstream reasoning tasks. Other approaches leverage tensor decompositions to estimate transition and observation likelihoods but often assume full state observability and full-rank transition matrices for all actions. To relax these assumptions, we study how PSRs learn transition and observation matrices up to a similarity transform, which may be estimated via tensor methods. Our method learns observation matrices and transition matrices up to a partition of states, where the states in a single partition have the same observation distributions corresponding to actions whose transition matrices are full-rank. Our experiments suggest that these partition-level transition models learned by our method, with a sufficient amount of data, meets the performance of PSRs as models to be used by standard sampling-based POMDP solvers. Furthermore, the explicit observation and transition likelihoods can be leveraged to specify planner behavior after the model has been learned.

## 1 Introduction

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 When planning and acting in the real world, intelligent agents must learn and reason about hidden information. Of great inspiration to us is the work of Baum et al. (2017), which shows that a real autonomous robot can infer a cabinet's locking mechanism from a hypothesis set of possible mechanisms through interaction. We are interested in a more general problem where autonomous agents must learn, through interaction, the dynamics of a system with hidden states, without any knowledge of the system state and transitions beforehand. The agent should also compute explicit estimates of transition and observation likelihoods to support downstream operations that manipulate the model, such as the specification of tasks to direct agent behavior. Our problem is modeled as learning the parameters of a discrete Partially Observable Markov Decision Process (POMDP) from a sequence of actions and observations acquired through random exploration. One common approach to learning a representation of a probabilistic latent-variable models like a POMDP is to apply a spectral decomposition to a matrix that contains estimates of the joint likelihoods of the observable random variables (Hsu et al., 2012; Balle et al., 2014). In particular, singular-value decompositions (SVD) give a way of estimating the number of hidden variables of the system by truncating the singular values under a certain threshold. For POMDPs, spectral methods may be applied to a *Hankel matrix*, which stores the joint likelihood between past and future observations conditioned on a sequence of past and future actions. The decomposition of this matrix can be used to derive a (linear) Predictive State Representation, which can be interpreted as an automaton with real-valued transition matrices (Boots et al., 2011; Balle et al., 2014). The 'state' of the PSR is a sufficient statistic that can be used to predict the likelihood of future observations given a 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 possible sequence of actions. This prediction capability allows PSRs to be used as black-box models for reinforcement learning (Liu et al., 2022; Zhan et al., 2022), but since transition and observation likelihoods cannot be directly read from a PSR, these models are difficult to manipulate after they are learned. There are other POMDP-learning algorithms that yield estimates of observation and transition likelihoods, but under assumptions that ultimately restrict the class of POMDPs that can be learned. Approaches introduced by Azizzadenesheli et al. (2016) and Guo et al. (2016) utilize tensor decompositions to recover observation distributions for each action whose transition matrix is full-rank. To recover the transitions, however, these approaches must also make the assumption that for each action, the corresponding observation distribution must be unique for every state. Full-rank transitions are common when modeling many real-world POMDPs, especially when actions may 'fail' with some probability, causing the system state to self-transition. Many systems, however, do not have distinct observation distributions associated with every action, like the locking mechanisms of Baum et al. (2017) or many standard POMDPs in the literature, like Tiger (Kaelbling et al., 1998). We investigate the relationship between PSRs and tensor decomposition methods to learn a broader class of POMDPs than existing tensor methods. To connect the two approaches, we extend a result established by Carlyle & Paz (1971) and Balle et al. (2014) that states that PSRs learn transitions and observation matrices up to an unknown basis. We then reformulate tensor decomposition methods to estimate the unknown basis to recover the original basis. Our modification of tensor decomposition methods for hidden state inference allows us to simultaneously leverage all observation distributions from all actions with full-rank transition methods all at once, rather than a per-action basis like previous approaches (Azizzadenesheli et al., 2016; Guo et al., 2016). Should the collection of observation distributions of all full-rank actions be unique for each state, like Tiger, we may recover the full POMDP. Should there exist states that share the same set of observation distribution when aggregated across actions, we learn transitions between partitions of states, where states in a single partition share the same observation distributions over all actions. Learning explicit transition and observation models is valuable because they enable model-based reasoning over environment dynamics. Whereas black-box PSRs only provide predictive likelihoods of observation sequences, access to explicit transition matrices and observation matrices allows for the specification of rewards after the model has been learned to direct planner behavior. Our experimental results suggest that our method can correctly learn partition-level transitions and observations and that these likelihoods are necessary to correctly direct agent behavior in POMDPs with very noisy observations.

## 2 Problem Setting

We assume that the ground truth system can be described as a discrete POMDP, represented by a 8-tuple (S, T , A, O, Z, b0*, R, γ*). The set S = {s 1, s2*, . . .* } is a discrete set of states, A is a discrete set of actions, and O = {o 1, o2*, . . .* } is a discrete set of observations. T = {T
a: a *∈ A}* denotes a set of row-stochastic state transition matrices. The element T
a ij = P(st+1 = s j|st = s i, at = a)
denotes the probability of transition to state s jfrom state s iafter taking action a at time t. The set Z = {Oao : (a, o) *∈ A × O}* describes a collection of diagonal matrices, where Oao ii = P(ot =
o|st = s i, at = a) denotes the emission probability of o under action a when leaving state s i. The distribution b0 ∈ ∆(S) describes the distribution over the initial state. The constant γ ∈ (0, 1) is the reward discount factor.

The agent begins acting in a POMDP with access to the action and observation spaces A and O. Under a uniform, memoryless random exploration policy at ∼ Unif(A) for all t ≥ 1, the agent collects a dataset D, which is a long string of actions and observations D = (a1, o1, a2, o2*, . . .*). From this data, we wish the agent to estimate the number of hidden states |S|, transition matrices Tˆ = {Tˆa: a *∈ A}*, and observation matrices Zˆ = {Oˆao : (a, o) *∈ A × O}*. We may also require the agent to learn a tabular reward R function by including rewards as observations (Izadi & Precup, 2008). One way we evaluate the approach is by measuring the error of the estimated model parameters against those of the groundtruth POMDP. Another is by evaluating the performance agent behavior under a planning algorithm after the POMDP is inferred from D. The last is by evaluating the behavior of a planner at a task designated by a user after the model has been learned.

## 3 Learning Predictive State Representations

3.1 FORWARD, BACKWARD, AND HANKEL MATRICES
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

$$(\mathbb{I})$$
$\eqref{eq:walpha}$. 
The backward matrix Back is a *|S| × ∞* matrix whose columns are indexed by 'future' actionobservation sequences. Following the PSR literature (Singh et al., 2004), we call these sequences
tests. An ordering must be decided to determine the test indices in the same manner as the histories. Should *test* = (at+1, ot+1*, . . . , a*t+n, ot+n), then the entry Backi,*test* expresses the likelihood of observing the test conditioned on beginning in state s
i. The entire column of *test* can also be derived
through forward multiplication of the POMDP matrices
$\mathbf{Back}_{i,test}=\mathrm{P}(o_{t+1},\ldots,o_{t+n}|a_{t+1}\ldots,a_{t+n},s_{t+1}=s^{i})$  $\mathbf{Back}_{:,test}=O^{a_{t+1}o_{t+1}}T^{a_{t+1}}\cdots O^{a_{t+n}o_{t+n}}T^{o_{t+n}}\cdot\mathbf{1}$
i) (3)
where 1 is the vector where all entries are set to one. The product of the forward and backward matrices results in the Hankel matrix, which we denote as H. The matrix multiplication unconditions and then marginalizes out the intermediate hidden
state. Given a history *hist* = (a1, o1, . . . , ak, ok) and a test *test* = (ak+1, ok+1, . . . , an, on), a
Hankel matrix entry is the corresponding joint likelihood of receiving a full string of observations conditioned on taking a full string of actions, e.g.
Hhist,*test* = P(o1, . . . , on|a1*, . . . a*n). (5)
The Hankel matrix does not refer to the underlying hidden state of the POMDP and can be estimated from action-observation trajectories. If we had a long string of actions and observations Dn = (a1, o1, . . . , an, on), the matrix H could be estimated by the *suffix-history approach*, taking frequency counts of subsequences of increasing lengths (Wolfe et al., 2005; Boots et al., 2011):

$$\begin{array}{l}{(3)}\\ {(4)}\end{array}$$
$${\mathcal{H}}_{h i s t,t e s t}=\mathrm{P}(o_{1},\ldots,o_{n}|a_{1},\ldots a_{n}).$$
$$(S)$$

$${\hat{\mathcal{H}}}_{h i s t,t e s t}={\frac{\sum_{i=1}^{n-L}\mathbb{I}_{(a_{i},o_{i},\ldots,a_{i+L},o_{i+L})=h i s t\oplus t e s t}}{\sum_{i=1}^{n-L}\mathbb{I}_{(a_{i},\ldots,a_{i+L},)=\mathrm{acts}(h i s t\oplus t e s t)}}}$$
$$(6)$$

This factorized construction is inspired by the construction of a related matrix, called the System Dynamics Matrix, by Singh et al. (2004).1It is important to note that expressing the Hankel matrix as a factorization of Forw and Back represents the system under a memoryless policy where future actions are independent of previous observations. To correctly estimate the matrix via Eq. 6, the data must also be collected under a memoryless policy, such as the uniform exploration policy as introduced in Sec. 2 (Bowling et al., 2006).

1The entries of the System Dynamics Matrix (SDM) are the likelihood of a given test *conditioned* on a history. Each row of the Hankel matrix is the same as the SDM except scaled by a constant (the likelihood of the history that indexes the row; Bacon et al. (2015)).

where acts(hist⊕*test*) is the action sequence associated with *hist*⊕Dtest and L = |hist⊕test| < n.

To estimate systems of hidden state, a natural place to start is to form an array that expresses the joint likelihoods between the observable random variables. A *Hankel matrix* is an instance of these arrays that encodes the joint likelihoods of past and future action-observation trajectories. In this section, we derive the Hankel matrix given knowledge of the ground truth POMDP. Our construction starts with two intermediate factors, called the *forward* and *backward* matrix, which we will multiply together to form the Hankel matrix. The forward matrix Forw is a *∞ × |S|* matrix whose rows are indexed by histories of actionobservation sequences. The row indices are determined by choosing an ordering that enumerates all possible history sequences. A sensible ordering is to enumerate the action-observation sequences of length one first, then of length two, etc. In practice, while there are an infinite number of actionobservation sequences, enumerating sequences up to a certain length is sufficient for computation. If hist = (a1, o1, . . . , at, ot), then an entry Forw*hist*,i expresses the *joint likelihood* of observing the history and landing in state s i. The entire row of *hist* may be computed by 'forward multiplying' the POMDP matrices that correspond to the action and observations in *hist*

  **Forw${}_{hist,i}=\mathrm{P}(o_{1},\ldots,o_{t},s_{t+1}=s^{i}|a_{1},\ldots,a_{t})$,**  **Forw${}_{hist,:}=b_{0}\cdot O^{a_{1}o_{1}}T^{a_{1}}\cdot\cdot\cdot O^{a_{n}o_{n}}T^{a_{n}}$.**

## 3.2 Transforming Predictive State Representations

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

## 3.3 Assumptions

Before we discuss how we estimate the similarity transform P, we must introduce a few important assumptions. First, we assume that under a memoryless random exploration-policy a ∼ πexp(A), π ∈ ∆(A)
2(which, in this paper, we take to be uniform), the induced Markov chain
(st, at, ot)t≥0 is ergodic. The visitation distribution over states will converge to a stationary distribution bπ, which has nonzero support over the entire state space. Second, we also assume that Forw, when limited to indices corresponding to one fewer than the maximum sequence length, has the same rank as the number of states (e.g. is full-rank), and that Back is also full-rank. This assumption is required to exclude POMDPs that have been shown to be computationally intractable to learn (Jin et al., 2020; Liu et al., 2022). For further discussion on the realism of systems learnable by our method, see Sec. 4.1.1. Our assumptions have a few consequences on the estimated Hankel matrix. We only sample the starting state distribution b0 at the start of the problem, so b0 has little influence over the Hankel matrix. Instead, the Hankel matrix will take on the stationary distribution bπ as the initial distribution instead. Furthermore, the rank of the resulting Hankel matrix will be *equivalent* to the number of states of the POMDPs in the restricted class that adhere to our assumptions, as opposed to the lower bound as is the case for general POMDPs. We formalize these consequences in Appendix A.

2∆(A) denotes the set of distributions over the discrete set A.

Suppose we have a Hankel matrix H, estimated in the limit of infinite data. Since the Hankel matrix H can be computed by multiplying two low-rank factors together, a natural first step of our method (and learning a PSR) is to compute a rank factorization of H (Boots et al., 2011; Balle et al., 2014).

One way to achieve this factorization is to compute a singular-value decomposition of the Hankel matrix H = UΣV
T, where singular values under a specified threshold (and their corresponding orthogonal vector components) are dropped. The truncated SVD is converted into a *rank factorization* by computing A = UΣ to be the left factor and V
Tto be the right factor. Crucially, since A · V
T
and Forw · Back both form rank factorizations of H (according to assumptions in Sec 3.3), there must exist some invertible transformation P such that A = Forw · P and P
−1· Back = V
T(see Appendix A.2). Moving one step earlier in the Hankel construction (Sec. 3.1), we can relate transitions, observations, and initial distributions with the rank factors and the Hankel matrix using Eqs. 1-4. Let *hists*ao denote an ordered set of all history indices that end in action-observation pair ao, and *hists*−ao denote the same set with the same ordering but without the ending pair ao. From Eqs. 2 and 4, we observe for each a ∈ A, o ∈ O:
H*hists*ao,: = Forw*hists*−ao,:
· O
aoT
a· Back = A*hists*−ao,:
· P
−1O
aoT
aP · V
T(7)
Hε,: = b T
0· Back = b T
0 P · V
T(8)
H:,ε = Forw · 1 = A · P1 (9)
After applying the Moore-Penrose inverse of A and V
Tto solve Eqs. 7-9, we obtain the observationtransition product, initial belief, and final summation vector up to a similarity transform. The transformed initial belief m0 = b T
0 P is called the *initial vector* and the transformed summation vector m∞ = P1 is called the *final vector*. The product Mao = P
−1OaoT
aP is called a linear PSR
update matrix. Together, this collection of matrices and vectors forms a *linear PSR model* (Littman & Sutton, 2001; Boots et al., 2011). A PSR can be used to compute the likelihood of observations o1, o2*, . . . , o*n under actions a1*, . . . , a*n by computing the product P(o1, . . . , on|a1*, . . . , a*n) =
mT0 Ma1o1· · · Manon m∞ = b0P P −1Oa1o1 T
a1· Oanon T
an P
−1P1, with appropriate normalizations for conditional calculations. With a few more details, the argument sketched above is a proof of a result of Carlyle & Paz (1971) and later Balle et al. (2014). The original result given by the authors was for probabilistic automata.

Proposition 1. [Carlyle & Paz (1971); Balle et al. (2014)] Let H = AV T be a rank factorization of a Hankel matrix H *with* rank(H) = r formed from a POMDP with initial state bπ*, transition* matrices {T
a} and observation matrices {Oao}. Suppose m0, {Mao, ∀(a, o) ∈ A × O}, m∞ are computed as in Eqs. 7, 8, and 9. Then there exists a nonsingular matrix P ∈ R
r×r*such that* P
−1MaoP = OaoT
afor all a ∈ A, o ∈ O, mT
0 P = bπ*, and* P
−1m∞ = 1.

## 4 Computing The Similarity Transform

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

![4_image_0.png](4_image_0.png)

Figure 1: Sense-Float-Reset. Edges are labeled with transition probabilities, and nodes are labeled with observations and reward received upon *leaving* the state. The reward of a state is zero unless specified otherwise. Observability partitions are represented by node shade. In many problem scenarios, P is nontrivial (e.g., not the identity matrix). The rank factors computed via SVD are orthogonal matrices, which is not always the case for Forw and Back. Furthermore, since rank factors may take real number values, m0 and {Mao} are real-valued as well, and we cannot interpret their entries as state, transition, or observation emission likelihoods. These likelihoods are essential for downstream operations that direct agent behavior. The key, then, is to recover the similarity transformation P. Once we know P, we can recover the original POMDP parameters by inverting the transform as expressed in Prop. 1. The observation and transition matrices can be recovered from products OaoT
a by computing the sums of the rows to form the diagonal of Oao and then normalizing the rows to form T
a. Our approach can recover P up to a certain partition of states, which we introduce with an example. A running example. Consider the POMDP illustrated in Figure 1, which is modified from the Float-Reset domain introduced by Littman & Sutton (2001). Like the original, the float action transitions the state up and down a line graph and will always emits an observation of 0. The reset action, also identical to the original, deterministically sets the state to the left end of the graph. This action emits an observation of 1 if the state is already in the leftmost state and 0 otherwise. The observations of the sense action are the same as reset, except each state of the system does not change. We also augment the system a reward function; the agent obtains +1 reward for executing any action in the state adjacent to the reset state and zero reward otherwise. This system is challenging to learn due to its nontrivial partial observability. Aside from the two leftmost states (when treating rewards as observations), all other states in this POMDP have the same observation distributions, regardless of the action. Furthermore, the transition matrix corresponding to the reset action is singular since it is zero everywhere except for a single column of ones. We wish to capture the difficulties of Sense-Float-Reset to discuss the main output of our algorithm in general terms. For arbitrary POMDPs, we group states that have the same observation distribution to form a *partition* of states. We call this grouping an *observability partition*. Of particular importance is the collection of observation distributions that correspond to actions with full-rank transition matrices. For the purpose of abbreviation, actions associated with full-rank transitions will be called *full-rank actions*, and we denote the entire set of full-rank actions as Afull ⊆ A. We call this alternate grouping restricted to action in Afull a *full-rank observability partition*.

## 4.1 Recovery Up To A Full-Rank Observability Partition

Our algorithm can estimate the similarity transform P up to the *full-rank observability partition*,
which we formalize in Theorem 1. Our statement is given in the regime of infinite data; for parameters introduced for finite data, see Appendix B.1.

Theorem 1. Let H be a Hankel matrix of POMDP (S, T , A, O, Z, bπ, R, γ) that adheres to the assumptions in Sec. 3.3, where bπ *is the stationary distribution under a uniform random policy* a ∼ Unif(A)*. Let* SΠ ⊂ 2 S be the full-rank observability partition of the POMDP. Let A and V
T
be a rank factorization of H, and m0, {Mao : a ∈ A, o ∈ O}, and m∞ *be the linear PSR model as* computed via Eqs. 7-9. Then there exists an algorithm on inputs A, V
T, m0, {Mao}, and m∞ that

![5_image_0.png](5_image_0.png)

then 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 computes a nonsingular matrix P˜*, such that if we let*

 $\tilde{b}^T_{\pi}=m^T_0\tilde{P}=b_{\pi}P^{-1}\tilde{P}$  $\tilde{O}^{ao}\tilde{T}^a=\tilde{P}^{-1}M^{ao}\tilde{P}=\tilde{P}^{-1}PO^{ao}T^aP^{-1}\tilde{P}$  $\tilde{b}_{\infty}=\tilde{P}^{-1}m_{\infty}=\tilde{P}^{-1}P\mathbf{1}$
$$(10)^{\frac{1}{2}}$$
$$(11)$$
$$\begin{array}{l}{{\sum_{s^{i}\in S}\hat{b_{\pi\,i}}=\sum_{s^{i}\in S}b_{\pi\,i}}}\\ {{\sum_{s^{i}\in S}(\bar{b_{\pi}^{T}}\bar{O}^{a_{1},o_{1}}\bar{T}^{a_{1}}\dots\bar{O}^{a_{n},o_{n}}\bar{T}^{a_{n}})_{i}=\sum_{s^{i}\in S}(b_{\pi}^{T}O^{a_{1}o_{1}}T^{a_{1}}\dots O^{a_{n}o_{n}}T^{a_{n}})_{i}}}\\ {{\tilde{b}_{\infty}={\bf1}}}\end{array}$$
$$(12)$$
$$(13)$$
an )i (14)
$$(14)$$
$$(15)$$
for all a1, . . . , an ∈ A, o1, . . . , on ∈ O, and integer n > 0 *and every partition set* S ∈ SΠ.

What Theorem 1 states is that we must sum over indices of the initial 'belief vector' to compute the likelihood the system is in a particular partition (Eq. 10). The same remains true when computing joint likelihoods between observations and the current state partition (Eq. 14; see Fig. 2 for a worked example for Sense-Float-Reset.). For POMDPs that have unique observation distributions across all actions, each state is in its own singleton partition, and we can recover the full similarity transform. Otherwise, we recover P up to the full-rank observability partition. We note it is possible for us to recover some POMDPs that have fewer observations than states, since the collection of *distributions* over emitted observations across all actions must be distinct (see Appendix C.5.3 for examples).

## 4.1.1 On The Restrictiveness Of Learnable Systems

To benefit from the result of Theorem 1, the systems to be learned must satisfy the assumptions stated in Sec. 3.3 and contain full-rank actions. Here, we discuss when these assumptions are satisfied. Full-Rank Transitions. In automated manipulation, robot actions have a desired transition state but may also *fail* (a gripper misses a grasp, slips of a drawer handle, etc.). One way these actions have been modeled in robot planning systems is to designate a successful 'desired state' with some success likelihood p*succ*, and have the system state 'fail' with some likelihood (causing a self-transition) (Kaelbling & Lozano-Perez, 2013; Garrett et al., 2020). In POMDP terms, these types of actions can ´
be simply modeled as the convex combination psuccT + (1 − psucc)I, where T is a matrix with rows containing all zeros except for a single entry of 1 (the desired states), the identity I indicates selfloop failure dynamics, and p*succ* the likelihood of an action succeeding. Under mild assumptions (e.g. psucc ̸= 1/2, 1), these actions are full-rank (see Appendix A.6). Ergodic Systems. Since we stipulate that POMDPs must be learned from a single trajectory, it is reasonable that the robot must be able to explore every state to correctly learn transition dynamics and observation emissions. One condition of ergodicity, *irreducibility*, ensures that the system does not get trapped in a subset of states. Furthermore, in many robot manipulation scenarios, robots are given a passive 'sensing' action that only obtains an observation sample without causing a change to the system state (Kaelbling & Lozano-Perez, 2013). The presence of these actions break any ´ periodic cycles, the other condition of ergodicity.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

$$(16)$$

With a slight abuse of notation, we denote P
−1T
aP as the matrix Ma. The next step of our procedure continues with transitions that are full-rank, which can easily be determined by a threshold
test on the singular value decomposition on all matrices Ma. Let Mfull = {P
−1T
aP : a ∈ A*full*}
be the set of full-rank transitions. Next, we compute the observation matrices associated with the
full-rank actions. For each Ma ∈ M*full* and o ∈ O we compute
$$M^{a o}\cdot M^{a-1}=P O^{a o}T^{a}P^{-1}(P T^{a}P^{-1})^{-1}=P O^{a o}P^{-1}.$$
−1. (17)
Since we know that all matrices Oao are diagonal, the eigenvalues of the matrices MaoMa−1 will be the diagonal entries of Oao. If the entries of a particular Oao are unique, then the eigenvectors computed from an eigendecomposition of MaoMa−1 will recover the columns of P up to a scalar factor. However, it is common to have repeated observation likelihoods across states for a single action (like all of Sense-Float-Reset), and an eigendecomposition may produce any spanning set of the invariant space corresponding to the repeated eigenvalue.

To reduce ambiguity, we wish to compute a *joint diagonalization* of all matrices MaoMa−1, which attempts to diagonalize each matrix with the same similarity transform. We apply a method of He et al. (2024). Their method exploits the fact that *sums* of matrices {MaoMa−1: a ∈ Afull , o *∈ O}*
do not change the invariant spaces spanned by eigenvectors of each matrix MaoMa−1. Suppose
{w ao : a ∈ Afull , o *∈ O}* is a set of weights, then the weighted sum

$$\sum_{a\in{\mathcal{A}}_{f i n l,o}\in O}w^{a o}M^{a o}M^{a-1}=P{\bigg(}\sum_{a\in{\mathcal{A}}_{f i n l,o}\in O}w^{a o}O^{a o}{\bigg)}P^{-1}$$
$$(17)^{\frac{1}{2}}$$
$$(18)$$

is still diagonalizable by P. Should we choose *random* weights wao, then the eigenvalues will be distinct up to states that share the same observation distribution almost surely. He et al. (2024)
recommends sampling these weights from the unit sphere S
|Afull*|·|O|−*1.

Lemma 1. Let weights {wao : a ∈ Afull , o ∈ O} *be sampled i.i.d. with respect to* Unif(S
|Afull|·|O|−1) and Λ = Pa∈A*full*,o∈O w aoOao*. Then* Λii = Λjj *with prob. 1 if and only* if Oao ii = Oao jj for all o ∈ O and all a ∈ A*full* .

When multiple states have the same observation distribution for all actions, the eigenvalues corresponding to those states will be the same, so their eigenvectors cannot be uniquely determined. Thus, the similarity transform P
′is nonunique when we have a nontrivial full-rank observability partition, the consequences of which we discuss in the next session.

## 4.3 Recovering Partition-Level Belief State Likelihoods And Transitions

We now introduce an algorithm that computes the similarity transform P˜. Our approach is a reformulation of the tensor decomposition method (Anandkumar et al., 2012; Azizzadenesheli et al., 2016) for linear PSR models.
Our procedure begins by marginalizing out the observations in matrices Mao, yielding the transitions
T
a up to similar transform P. This marginalization can be done by summing all matrices Mao over
all o ∈ O for some fixed a ∈ A:
X o∈O Mao = P X o∈O O aoT a P −1 = P T aP

## −1(16) 4.2 Recovering Observation Distributions From Full-Rank Actions

The recovered similarity transform P
′formed by the eigenvectors of the random sum in Equation 18, but not the partition-level transitions. When the full-rank observability partition is nontrivial, the matrix Q = P
−1P
′is block-diagonal, with invertible blocks that correspond to states within the same partition (see Appendix A.4 for a proof). This matrix Q prevents us from using P
′as the transform promised in Theorem 1. For example, when applying P
′as a similarity transform to the PSR vector m0, a restriction to the subindices of the partition S1 yields [m0P
′]S1 = [b T
0 P
−1P
′]S1 =
[b T
0]S1Q1, so the sum of the entries is not a proper likelihood, violating Eq. 13 of Theorem 1.

To recover partition-level likelihoods and transitions, we look to the final vector of the linear PSR after applying the transform P
′, e.g. Pm0 = P
′−1P1. Intuitively, by applying diag(Pm0) as a similarity transform, we transform the final vector back to 1, recapturing a marginalization of the latent 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png) 

state variable. To avoid scenarios where P
′−1m0 has entries of zero, we perform a pre-processing step by multiplying the system with a random block-diagonal rotation matrix R, whose blocks correspond to the full-rank observability partition. We take the transform diag(RP′−1m∞)RP′−1as the similarity transform P˜ that satisfies Theorem 1 (see Appendix A.5 for proof of correctness).

## 5 Experiments

Our experiments evaluate the fidelity of the learned POMDP models and explore the advantages of estimating transition and observation likelihoods. We seek to know, empirically, how quickly the learned observation matrices (and fine-grained transition matrices, if available) converge to ground truth values. We also wish to know whether the performance of the planning model is impaired by errors in the estimated similarity transform. Lastly, we evaluate whether the transition and observation likelihood estimates can be leveraged to specify a reward function to elicit desired behavior from a planner. All experiments are compared against linear PSRs and an Expectation-Maximization (EM) baseline (Rabiner, 1989; Shatkay & Kaelbling, 2002) with a number of states determined by the number of components of the truncated SVD when learning a linear PSR.

For our planning experiments, we verify our approach on several standard POMDPs: Tiger (Kaelbling et al., 1998), T-Maze (with a truncated corridor) (Bakker, 2001), and Sense-Float-Reset. To allow the agent to collect an arbitrary-length string of data in all domains, we modify T-Maze to choose the next state randomly from the initial state distribution instead of terminating the sequence of interactions. Appendices B.1 and C contain details on the parameters of the learning algorithm and planner. Rewards of the original POMDPs have been learned as observations for planning.

![8_image_0.png](8_image_0.png) 
For our reward-specification experiments, we introduce two novel domains (*noisy hallway* and directional hallway) whose observation and transition matrices can be fully recovered by our method. The domains share transition dynamics on a three-state 'hallway,' in which the actions include noisily translating left and right, choosing to deterministically stay in the current cell, or performing a reset to a uniform random state. The observation space is also shared; the agent may noisily observe whether it is on the left-end or right-end of the hallway. The domains differ in the observation distributions of the middle states. In *noisy hallway*, the agent noisily observes the end of the hallway in the direction of commanded movement ('directional' observations). In directional hallway, the agent observes left-end or right-end with probability 1/2 ('noisy' observations). For more details, see Appendix C.5.3.

Convergence to true POMDP parameters. In Figure 3, our results suggest that our method successfully recovers the underlying observation models through the L1 error of learned observation and partition-level transition likelihoods against ground truth. EM consistently converges to a local minimum and does not obtain correct observation or transition likelihoods. Planning performance with the learned model. To evaluate the performance of the planning model, we apply a standard sampling-based POMDP solver to the original ground truth POMDPs, learned PSRs, and learned POMDPs and compare the average yielded rewards. We use the samplingbased planning approach PO-UCT of Silver & Veness (2010) with the correction described by Shah et al. (2022). Ideally, planning performance should be the same across ground truth models, PSRs, and the learned partition-level POMDPs (see Appendix C.3 for discussion on rollout strategies for each model). Performance as a function of the number of action-observations collected is shown in Fig. 3, which we find to be similar across all models learned. Planning performance on specified rewards. We explore whether the likelihoods and observations yielded by our algorithm can be leveraged to direct agent behavior after the model is learned. One case, motivated by automated planning in robotics, is to direct the agent to drive a system into a set of states as determined by the states' emitted observations by specifying a reward function (Boots et al., 2011). After learning a POMDP, we can analyze the learned observation matrices to find the states to emit positive reward. In the past, if a PSR did not learn a reward model, then rewards were determined solely by observations (Boots et al., 2011). Otherwise, the entire model must be relearned to estimate a reward model that depends on state (Izadi & Precup, 2008). Our evaluations of this experiment are carried out on the two noisy hallway domains, where we attempt to direct the agent to drive the POMDP to the 'middle' hallway state with ambiguous observations. We compare the strategies of assigning rewards to observations and assigning rewards to states. In the directional domain, we assign +1 reward to action-observation pairs (left, end-left) and (right, end-right) for the former strategy, and assign +1 reward to the state whose maximum likelihood observations under left and right are their corresponding hallway ends for the latter. For the noisy environment, we reward the same action-observation 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 pairs as the directional environment and also (left, end-right) and (right, end-left) for the former strategy, and add +1 reward to the state that maximizes the sum of entropy of observation distributions across all actions for the latter. The former strategy is evaluated on PSRs and POMDPs, whereas the latter is evaluated on learned POMDP models only. Performance is judged on the number of timesteps the agent spends in the desired states. Results can be found in Figure 4. In the directional domain, models that use the first strategy allow the planner to drive the middle state because it is easily identified by the observations received under left and right. The second strategy performs poorly due to slow convergence of transition matrices (see Appendix C.4). In the noisy domain, the uniform belief state and belief state that places all mass on the middle of the hallway yield the same mixture observation distribution weighted by the belief, which does not elicit the correct behavior from the planner. The planner that uses the rewards emitted from the highest-entropy state performs well after the transition matrices begin to converge. This additional flexibility highlights that learning POMDPs maintains all the advantages of PSRs and obtains the flexibility to exploit observation and transition likelihood models.

## 6 Related Work

Spectral methods are a common technique for learning partially-observable dynamics in the theory of RL literature (Liu et al., 2022; Zhan et al., 2022). These models are sufficient to serve as blackbox models for model-based RL but do not afford likelihood estimates for other computations that require general inference operations related to the latent state. Spectral methods to model learning have been applied to related settings, including linear time-invariant system identification (Ho &
Kalman, 1966; Oymak & Ozay, 2022). Other alternate approaches for learning POMDPs have also been explored. Toro Icarte et al. (2019) applies mixed-integer linear programming to learn an automaton to describe transition data. The learning problem has been framed as an automaton learning problem by Angluin (1987) to accept a particular *language* with strings given as data (Brafman & De Giacomo, 2019; Ronca et al., 2022). Others have resorted to inductive logic schemes (Amir & Chang, 2008; Silver et al., 2021). These methods usually assume that transitions or observations made by the agent are deterministic. Other approaches have relaxed assumptions to stochastic observations but still assume deterministic transitions (Dean et al., 1995). Recurrent deep-learning-based architectures (Wang et al., 2023; Allen et al., 2024), can learn to make future predictions of system behavior from histories of actions and observations. Recurrent neural nets perform particularly well, unlike Transformers, which represent a fixed circuit that cannot maintain memory internally (Lu et al., 2024). These recurrent models are use specialized training objectives that encourage networks to learn how to summarize histories observed by the agent (Agarwal et al., 2021; Allen et al., 2024) or have access to privelaged full-state observable information during training (Wang et al., 2023). Like PSRs, the representation of the hidden state learned by these models is , and cannot readily provide likelihood models for probabilistic inference.

## 7 Conclusion And Future Work

We present a method that learns discrete POMDP parameters from an action-observation sequence gathered under a random exploration policy up to a partition of the state space. Our approach applies tensor decomposition methods to estimate a similarity transform to transform a PSR model to a basis where observation and transition likelihoods can be recovered. In domains where each state has a unique observation distribution aggregated across all full-rank actions, we recover the true POMDP. Otherwise, we learn the transitions between full-rank observability partitions of the state space. In the future, we intend to improve our method to scale to larger problems, but in class and scale. Removing the restriction of learning observations from full-rank actions, or learning full transitions despite the presence of a nontrivial observability partition would be desirable. Another direction is to improve our approach to scale to larger POMDPs. Matrix-completion methods under low-rankness assumptions could help the algorithm infer *missing* entries in the Hankel matrix. Additional future work is to expand the theoretical foundations of our algorithm. Carefully studying our algorithm under a PAC-learning framework would contribute to our understanding of the computational complexity of learning POMDPs in general.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Reproducibility Statement: All omitted proofs to derive the theoretical claims in this paper have been included in Appendix A. Discussion regarding algorithm parameters selected (Hankel matrix size, rankness thresholds, UCT constants, etc.) and experimental domains can be found in Appendices B and C. Code, including environments and the learning algorithm, will be made available should our work be accepted for publication. Ethics Statement: This paper discuses the derivation of a novel learning algorithm and application in toy experimental planning domains, so we do not believe there is any cause for ethical concern of our work.

## References

Anish Agarwal, Abdullah Alomar, Varkey Alumootil, Devavrat Shah, Dennis Shen, Zhi Xu, and Cindy Yang. PerSim: Data-Efficient Offline Reinforcement Learning with Heterogeneous Agents via Personalized Simulators, November 2021.

Cameron Allen, Aaron Kirtland, Ruo Yu Tao, Sam Lobel, Daniel Scott, Nicholas Petrocelli, Omer Gottesman, Ronald Parr, Michael Littman, and George Konidaris. Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy. Advances in Neural Information Processing Systems, 37:62988–63028, December 2024.

E. Amir and A. Chang. Learning Partially Observable Deterministic Action Models. Journal of Artificial Intelligence Research, 33:349–402, November 2008. ISSN 1076-9757. doi: 10.1613/ jair.2575.

Anima Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, and Matus Telgarsky. Tensor Decompositions for Learning Latent Variable Models:. Technical report, Defense Technical Information Center, Fort Belvoir, VA, December 2012.

Dana Angluin. Learning regular sets from queries and counterexamples. Information and Computation, 75(2):87–106, November 1987. ISSN 0890-5401. doi: 10.1016/0890-5401(87)90052-6.

Michael Artin. *Algebra*. Pearson, 2011. Kamyar Azizzadenesheli, Alessandro Lazaric, and Animashree Anandkumar. Reinforcement Learning of POMDPs using Spectral Methods. In *Conference on Learning Theory*, pp. 193–256.

PMLR, June 2016.

Pierre-Luc Bacon, Borja Balle, and Doina Precup. Learning and Planning with Timing Information in Markov Decision Processes. In UAI, 2015.

Bram Bakker. Reinforcement Learning with Long Short-Term Memory. In *Advances in Neural* Information Processing Systems, volume 14. MIT Press, 2001.

Borja Balle, Xavier Carreras, Franco M. Luque, and Ariadna Quattoni. Spectral learning of weighted automata. *Machine Learning*, 96(1):33–63, 2014. ISSN 1573-0565. doi: 10.1007/
s10994-013-5416-x. URL https://doi.org/10.1007/s10994-013-5416-x.

Manuel Baum, Matthew Bernstein, Roberto Martin-Martin, Sebastian Hofer, Johannes Kulick, Marc ¨
Toussaint, Alex Kacelnik, and Oliver Brock. Opening a lockbox through physical exploration. In 2017 IEEE-RAS 17th International Conference on Humanoid Robotics (Humanoids), pp. 461– 467, November 2017. doi: 10.1109/HUMANOIDS.2017.8246913.

Byron Boots, Sajid M Siddiqi, and Geoffrey J Gordon. Closing the learning-planning loop with predictive state representations. *The International Journal of Robotics Research*, 30(7):954–966, June 2011. ISSN 0278-3649. doi: 10.1177/0278364911404092.

Michael Bowling, Peter McCracken, Michael James, James Neufeld, and Dana Wilkinson. Learning predictive state representations using non-blind policies. In *Proceedings of the 23rd International* Conference on Machine Learning - ICML '06, pp. 129–136, Pittsburgh, Pennsylvania, 2006. ACM Press. ISBN 978-1-59593-383-6. doi: 10.1145/1143844.1143861.

Ronen I. Brafman and Giuseppe De Giacomo. Regular Decision Processes: A Model for Non-
Markovian Domains. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pp. 5516–5522, Macao, China, August 2019. International Joint Conferences on Artificial Intelligence Organization. doi: 10.24963/ijcai.2019/766.

J.W. Carlyle and A. Paz. Realizations by stochastic finite automata. Journal of Computer and System Sciences, 5(1):26–40, February 1971. ISSN 00220000. doi: 10.1016/S0022-0000(71)80005-3.

Thomas Dean, Dana Angluin, Kenneth Basye, Sean Engelson, Leslie Kaelbling, Evangelos Kokkevis, and Oded Maron. Inferring finite automata with stochastic output functions and an application to map learning. *Machine Learning*, 18(1):81–108, January 1995. ISSN 1573-0565. doi: 10.1007/BF00993822.

Caelan Reed Garrett, Chris Paxton, Tomas Lozano-P ´ erez, Leslie Pack Kaelbling, ´
and Dieter Fox. Online Replanning in Belief Space for Partially Observable Task and Motion Problems. In Proceedings of the IEEE International Conference on Robotics and Automation, pp. 5678–5684, 2020. doi: 10.1109/ ICRA40945.2020.9196681. URL https://ieeexplore.ieee.org/abstract/
document/9196681?casa_token=KrItGweWHQsAAAAA:HsjlznrkmKBAd_ 9dmaMT68bgkzvi7nd7AtMhA6Cp9gsym9-jP0VIKjqyA4EV9vE8n2Nu6gVLsA.

Zhaohan Daniel Guo, Shayan Doroudi, and Emma Brunskill. A PAC RL Algorithm for Episodic POMDPs. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, pp. 510–518. PMLR, May 2016.

Haoze He, Daniel Kressner, and Bor Plestenjak. Randomized methods for computing joint eigenvalues, with applications to multiparameter eigenvalue problems and root finding. Numerical Algorithms, October 2024. ISSN 1572-9265. doi: 10.1007/s11075-024-01971-0.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 B L. Ho and R. E. Kalman. Editorial: Effective construction of linear state-variable models from input/output functions: Die Konstruktion von linearen Modeilen in der Darstellung durch Zustandsvariable aus den Beziehungen fur Ein- und Ausgangsgr ¨ oßen. ¨ *at - Automatisierungstechnik*, 14(1-12):545–548, December 1966. ISSN 2196-677X. doi: 10.1524/auto.1966.14.112.545.

Daniel Hsu, Sham M. Kakade, and Tong Zhang. A spectral algorithm for learning Hidden Markov Models. *Journal of Computer and System Sciences*, 78(5):1460–1480, September 2012. ISSN 0022-0000. doi: 10.1016/j.jcss.2011.12.025.

Masoumeh T. Izadi and Doina Precup. Point-Based Planning for Predictive State Representations. In Sabine Bergler (ed.), *Advances in Artificial Intelligence*, pp. 126–137, Berlin, Heidelberg, 2008. Springer. ISBN 978-3-540-68825-9. doi: 10.1007/978-3-540-68825-9 13.

Chi Jin, Sham Kakade, Akshay Krishnamurthy, and Qinghua Liu. Sample-Efficient Reinforcement Learning of Undercomplete POMDPs. In *Advances in Neural Information Processing Systems*, volume 33, pp. 18530–18539. Curran Associates, Inc., 2020.

Leslie Pack Kaelbling and Tomas Lozano-P ´ erez. Integrated task and motion planning in belief space. ´
The International Journal of Robotics Research, 32(9–10):1194–1227, 2013. ISSN 0278-3649, 1741-3176. doi: 10.1177/0278364913484072. URL http://journals.sagepub.com/ doi/10.1177/0278364913484072.

Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101(1):99–134, May 1998. ISSN 0004-3702. doi: 10.1016/S0004-3702(98)00023-X.

John M. Lee. Sard's theorem. In *Introduction to Smooth Manifolds*, pp. 125–149. Springer New York, New York, NY, 2012. ISBN 978-1-4419-9982-5. doi: 10.1007/978-1-4419-9982-5 6.

Michael Littman and Richard S Sutton. Predictive Representations of State. In Advances in Neural Information Processing Systems, volume 14. MIT Press, 2001. URL https://proceedings.neurips.cc/paper_files/paper/2001/hash/ 1e4d36177d71bbb3558e43af9577d70e-Abstract.html.