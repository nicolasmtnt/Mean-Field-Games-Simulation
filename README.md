# Mean Field Games Simulation and Introduction: a Reinforcement Learning Approach in Python

## 1 Mean Field Game in the Static Setting

Consider a game with $N$ players, denoted as $\{1, \dots, N\}$. 

### Action Space
Each player $i \in \{1, \dots, N\}$ can choose an action $a^i$ from the action space $\mathcal{A}$. The collection of actions from all players is referred to as the **population profile of actions**, denoted by:
$$
\underline{a} = \{a^1, a^2, \dots, a^N\} \in \mathcal{A}^N
$$

### Cost Function
Each player $i$ has a cost function $f^i(\underline{a})$, where:
$$
f^i : \mathcal{A}^N \to \mathbb{R}
$$
The goal of each player is to minimize their cost by selecting an optimal action:
$$
\min_{a^i} f^i(\underline{a}) \quad \forall i \in \{1, \dots, N\}.
$$

### Static Setting
In the static setting, the **state** of each player is fixed and does not change over time. Therefore, actions are chosen once, and the population profile remains static. 

---

### Remark
1. The main difference between the static and dynamic settings is that in the dynamic setting, the state of each player evolves over time based on the actions taken. As a result, the population profile $\underline{a}$ must adapt to these state changes. This dynamic aspect will be explored in the next section.

2. In this setup, we consider **pure strategies**, meaning that the actions chosen by each player are deterministic. No randomness is involved in the decision-making process of any player. This contrasts with mixed strategies, where players would choose actions based on probabilities.


### Definition: Nash Equilibrium (NE)

A population profile $\hat{\underline{a}} = \{\hat{a}^1, \dots, \hat{a}^N\} \in \mathcal{A}^N$ is a **Nash Equilibrium (NE)** if:
$$
f^i(\hat{\underline{a}}) \leq f^i(a^i, \hat{\underline{a}}^{-i}) \quad \forall a^i \in \mathcal{A}, \, \forall i \in \{1, \dots, N\}.
$$

Here, $\hat{\underline{a}}^{-i}$ represents the actions of all players except player $i$, that is:
$$
\hat{\underline{a}}^{-i} = \{\hat{a}^1, \dots, \hat{a}^{i-1}, \hat{a}^{i+1}, \dots, \hat{a}^N\}.
$$

In simple terms, at a Nash Equilibrium, no player can reduce their cost by changing their action while the actions of all other players remain the same.


### Examples of Nash Equilibrium

Here are a few examples to illustrate Nash Equilibrium in different settings:

---

#### 1. **Target Position with No Interaction**
Each player chooses an action $a^i$ to minimize the distance to a fixed target position $a^* \in \mathcal{A}$:
$$
f^i(\underline{a}) = |a^i - a^*|
$$

- **Equilibrium:** All players independently choose $a^i = a^*$.
- **Explanation:** Since there is no interaction between players, each player's cost is minimized by moving directly to the target position.

---

#### 2. **Attraction Through the Mean**
Players are attracted to the mean position of the group, aiming to minimize the distance between their action and the average action:
$$
f^i(\underline{a}) = |a^i - \bar{a}|
$$
where $\bar{a} = \frac{1}{N} \sum_{j=1}^N a^j$ is the mean of the population profile.

- **Equilibrium:** Any population profile $\underline{a}$ where all players choose the same action, $a^i = c$ for some constant $c$, is a Nash Equilibrium.
- **Explanation:** When all players act identically, the mean $\bar{a}$ equals their action, minimizing the cost for everyone. However, the solution is **not unique**, as this works for any constant $c$.

---

#### 3. **Group Aversion**
Players aim to avoid choosing the same action as others, minimizing the number of players taking the same action:
$$
f^i(\underline{a}) = \sum_{j \neq i} \mathbf{1}_{a^j = a^i}
$$
where $\mathbf{1}_{a^j = a^i}$ is an indicator function that equals 1 if $a^j = a^i$ and 0 otherwise.

- **Equilibrium:** Players spread out across the action space $\mathcal{A}$ to ensure no two players choose the same action, if the size of $\mathcal{A}$ allows it.
- **Explanation:** Each player minimizes their cost by choosing a unique action, avoiding overlap with others.

---

#### 4. **Rock-Paper-Scissors**
In the classic rock-paper-scissors game, each player $i$ chooses an action $a^i \in \mathcal{A} = \{\text{Rock}, \text{Paper}, \text{Scissors}\}$. The payoff function $f^i(\underline{a})$ depends on the actions of both players and is defined by the rules:

- Rock beats Scissors: $f^i(\underline{a}) = 1$ if $a^i = \text{Rock}$ and $a^j = \text{Scissors}$.
- Scissors beats Paper: $f^i(\underline{a}) = 1$ if $a^i = \text{Scissors}$ and $a^j = \text{Paper}$.
- Paper beats Rock: $f^i(\underline{a}) = 1$ if $a^i = \text{Paper}$ and $a^j = \text{Rock}$.
- Otherwise, $f^i(\underline{a}) = 0$ in case of a draw or $-1$ if the other player wins.

- **Pure Strategy Solution:** There is no Nash Equilibrium in pure strategies. For any choice of $a^i$, the other player $j$ can always switch their action to increase their own payoff, $f^j(\underline{a})$.
- **Explanation:** The cyclical nature of rock-paper-scissors ensures that no single pair of actions maximizes payoffs for both players simultaneously in pure strategies. However, a mixed strategy Nash Equilibrium exists, where each player chooses each action with equal probability, $\frac{1}{3}$.

#### 5. **Prisoner's Dilemma**
In the **Prisoner's Dilemma**, two players decide whether to cooperate ($C$) or defect ($D$). The payoff function $f^i(\underline{a})$ for player $i$ represents the utility they want to **maximize**, and it depends on the combination of actions:

- If both cooperate: $f^i(\underline{a}) = 3$.
- If one defects while the other cooperates: The defector gets $f^i(\underline{a}) = 5$, and the cooperator gets $f^i(\underline{a}) = 1$.
- If both defect: $f^i(\underline{a}) = 2$.

|               | Cooperate ($C$) | Defect ($D$)   |
|---------------|------------------|----------------|
| **Cooperate** | $3, 3$          | $1, 5$         |
| **Defect**    | $5, 1$          | $2, 2$         |

- **Nash Equilibrium:** Both players choose $a^i = D$ (defect).
- **Explanation:** If one player cooperates, the other minimizes their cost by defecting. Since defecting is always the better response regardless of the other player's choice, mutual defection is the Nash Equilibrium. However, it is not Pareto-optimal, as mutual cooperation results in a better outcome for both players.



### Mixed Strategy Nash Equilibrium

In a **mixed strategy** setting, each player does not choose a deterministic action $a^i \in \mathcal{A}$ but instead selects actions according to a probability distribution $\pi^i \in \mathcal{P}(\mathcal{A})$, where $\mathcal{P}(\mathcal{A})$ denotes the set of all probability measures over the action space $\mathcal{A}$. 

The **population profile of strategies** is then represented by:
$$
\underline{\pi} = \{\pi^1, \pi^2, \dots, \pi^N\} \in \mathcal{P}(\mathcal{A})^N,
$$
and a realization of actions from these strategies is denoted by:
$$
\underline{a} = \{a^1, a^2, \dots, a^N\} \sim \underline{\pi}.
$$

### Empirical Distribution of Actions
Given the population profile of actions $\underline{a}$, the **empirical distribution** of actions is:
$$
\mu_{\underline{a}} = \frac{1}{N} \sum_{j=1}^N \delta_{a^j},
$$
where $\delta_{a^j}$ is the Dirac measure centered at $a^j$. 

As $N \to \infty$, this empirical distribution $\mu_{\underline{a}}$ becomes less random and converges to a deterministic measure reflecting the population's mixed strategies. This limit is governed by the Law of Large Numbers.

### Cost Function with Mixed Strategies
The cost for player $i$ is now defined as the expected value of their payoff under the distribution of all players' strategies:
$$
J^i(\underline{\pi}) = \mathbb{E}_{\underline{a} \sim \underline{\pi}}[f^i(\underline{a})].
$$

Here:
- $\underline{\pi}$ governs the random realization of the population profile $\underline{a}$.
- $f^i(\underline{a})$ is the payoff function for player $i$ given the realized actions $\underline{a}$.

### Nash Equilibrium in Mixed Strategies
A mixed strategy profile $\hat{\underline{\pi}} = \{\hat{\pi}^1, \dots, \hat{\pi}^N\}$ is a Nash Equilibrium if:
$$
J^i(\hat{\underline{\pi}}) \leq J^i(\pi^i, \hat{\underline{\pi}}^{-i}) \quad \forall \pi^i \in \mathcal{P}(\mathcal{A}), \, \forall i \in \{1, \dots, N\}.
$$

- $\hat{\underline{\pi}}^{-i}$ represents the strategies of all players except player $i$.
- Player $i$ cannot decrease their expected cost by unilaterally changing their mixed strategy $\pi^i$.




### Static Mean Field Game

In a **static mean field game**, we analyze the interaction of a large number of players under the assumptions of **homogeneity** and **anonymity**. 

---

#### Homogeneity and Anonymity
- **Homogeneity**: All players share the same cost function.
- **Anonymity**: Each player's cost depends on their own action and the overall distribution of actions, not on the identity of other players.

The cost function for a representative player is defined as:
$$
f: \mathcal{A} \times \mathcal{P}(\mathcal{A}) \to \mathbb{R},
$$
where $\mathcal{P}(\mathcal{A})$ is the set of all probability measures on the action space $\mathcal{A}$. 

For a finite population of $N$ players, the cost for player $i$ is:
$$
f^i(\underline{a}) = f(a^i, \mu_{\underline{a}}),
$$
where:
- $a^i$ is the action of player $i$.
- $\mu_{\underline{a}} = \frac{1}{N} \sum_{j=1}^N \delta_{a^j}$ is the **empirical distribution** of the population's actions.
- Note that $\mu_{\underline{a}}$ is different from the **mean action** $\frac{1}{N} \sum_{j=1}^N a^j$, which is a single value and does not capture the full distribution.

---

#### Passing to the Limit
As the number of players $N \to \infty$, the empirical distribution $\mu_{\underline{a}}$ converges to a **population distribution of actions** $\pi' \in \mathcal{P}(\mathcal{A})$. This limit removes randomness from the empirical distribution, reflecting the mean field interaction.

---

#### Goal for Each Player
In the mean field setting, each player minimizes their expected cost given the population distribution $\pi'$:
$$
J(\pi, \pi') = \mathbb{E}_{a \sim \pi}[f(a, \pi')],
$$
where:
- $\pi \in \mathcal{P}(\mathcal{A})$ is the player's strategy.
- $\pi'$ is the population distribution of actions.
- The expectation $\mathbb{E}_{a \sim \pi}$ is taken over the player's action distribution $\pi$.

The goal is to find the strategy $\pi$ that minimizes $J(\pi, \pi')$ given the population distribution $\pi'$.

---

### Static Mean Field Game

In a **static mean field game**, we analyze the interaction of a large number of players under the assumptions of **homogeneity** and **anonymity**. 

---

#### Homogeneity and Anonymity
- **Homogeneity**: All players share the same cost function.
- **Anonymity**: Each player's cost depends on their own action and the overall distribution of actions, not on the identity of other players.

The cost function for a representative player is defined as:
$$
f: \mathcal{A} \times \mathcal{P}(\mathcal{A}) \to \mathbb{R},
$$
where $\mathcal{P}(\mathcal{A})$ is the set of all probability measures on the action space $\mathcal{A}$. 

For a finite population of $N$ players, the cost for player $i$ is:
$$
f^i(\underline{a}) = f(a^i, \mu_{\underline{a}}),
$$
where:
- $a^i$ is the action of player $i$.
- $\mu_{\underline{a}} = \frac{1}{N} \sum_{j=1}^N \delta_{a^j}$ is the **empirical distribution** of the population's actions.
- Note that $\mu_{\underline{a}}$ is different from the **mean action** $\frac{1}{N} \sum_{j=1}^N a^j$, which is a single value and does not capture the full distribution.

---

#### Passing to the Limit
As the number of players $N \to \infty$, the empirical distribution $\mu_{\underline{a}}$ converges to a **population distribution of actions** $\pi' \in \mathcal{P}(\mathcal{A})$. This limit removes randomness from the empirical distribution, reflecting the mean field interaction.

---

#### Definition: Mean Field Nash Equilibrium
A **Mean Field Nash Equilibrium** is a pair $(\hat{\pi}, \pi')$ where:
1. The player minimizes their expected cost given the population distribution $\pi'$:
   $$
   \hat{\pi} = \arg\min_{\pi \in \mathcal{P}(\mathcal{A})} J(\pi, \pi'),
   $$
   where the expected cost is:
   $$
   J(\pi, \pi') = \mathbb{E}_{a \sim \tilde{\pi}}[f(a, \pi')].
   $$
2. The player's optimal strategy $\hat{\pi}$ matches the population distribution $\pi'$:
   $$
   \hat{\pi} = \pi'.
   $$

---

#### Fixed Point Formulation
The Mean Field Nash Equilibrium can equivalently be characterized as a **fixed point**:
$$
\pi' \in  \arg\min_{\pi \in \mathcal{P}(\mathcal{A})} \mathbb{E}_{a \sim \pi}[f(a, \pi')].
$$

This fixed point formulation highlights that the equilibrium strategy $\pi'$ must simultaneously minimize the expected cost and align with the population distribution.

### $\epsilon$-Nash Equilibrium

An **$\epsilon$-Nash Equilibrium** is a relaxed version of the Nash Equilibrium concept, where each player’s strategy is allowed to be approximately optimal within an error margin $\epsilon \geq 0$.

---

#### Definition
A mixed strategy profile $\hat{\underline{\pi}} = \{\hat{\pi}^1, \dots, \hat{\pi}^N\}$ is an **$\epsilon$-Nash Equilibrium** if for every player $i$:
$$
J(\hat{\pi}^i, \hat{\pi}^{-i}) \leq J(\pi^i, \hat{\pi}^{-i}) + \epsilon \quad \forall \pi^i \in \mathcal{P}(\mathcal{A}),
$$
where:
- $J(\pi^i, \hat{\pi}^{-i})$ is the expected cost for player $i$ when they use strategy $\pi^i$ and all other players use $\hat{\pi}^{-i}$.
- $\hat{\pi}^{-i} = \{\hat{\pi}^1, \dots, \hat{\pi}^{i-1}, \hat{\pi}^{i+1}, \dots, \hat{\pi}^N\}$ denotes the strategies of all other players.

---

#### Interpretation
- In an $\epsilon$-Nash Equilibrium, no player can unilaterally change their strategy to improve their cost by more than $\epsilon$.
- When $\epsilon = 0$, the $\epsilon$-Nash Equilibrium reduces to the standard Nash Equilibrium.

---

#### Relevance in Mean Field Games
In mean field games with large populations, $\epsilon$-Nash Equilibria become particularly important:
1. **Approximation of Nash Equilibria**: In practical applications, solving for an exact Nash Equilibrium might be infeasible, and an $\epsilon$-Nash Equilibrium provides a good approximation.
2. **Scalability**: As the number of players $N \to \infty$, the $\epsilon$ value can often decrease, reflecting how individual deviations have diminishing effects in large populations.

![epsilon Nash Equilibrium](src/img/epsNash.png "Optional Title")


The slide provides a proof sketch for an **Approximate Nash Equilibrium (NE)** in a mean field game setting. Here's a breakdown and explanation:

---

### 1. **Proof Idea**  
The goal is to compare the cost of the $N$-player game to the mean field game cost. Specifically:
$$
J^i(\hat{\pi}, \hat{\pi}^{-i}) - J^i(\pi, \hat{\pi}^{-i}),
$$
is decomposed as:
$$
= J^i(\hat{\pi}, \hat{\pi}^{-i}) - J(\hat{\pi}, \hat{\pi}) + J(\hat{\pi}, \hat{\pi}) - J(\pi, \hat{\pi}) + J(\pi, \hat{\pi}) - J^i(\pi, \hat{\pi}^{-i}),
$$
into three parts:
1. The first term relates to differences between the $N$-player game cost and the mean field cost.
2. The second term captures the difference between mean field costs for $\hat{\pi}$ and $\pi$.
3. The third term accounts for the difference between the mean field cost and the $N$-player cost for $\pi$.

---

### 2. **Notation**
- $\hat{a} = \mathbb{E}_{a \sim \hat{\pi}}[a]$: Expected action under $\hat{\pi}$.
- $\tilde{a} = \frac{1}{N} \sum_{j=1}^N a^j$: Empirical average of actions.
- $\phi(a, \tilde{a})$: Represents the cost function for a player $i$ based on their action $a$ and the empirical distribution $\tilde{a}$.

---

### 3. **Key Observations**
- **Lipschitz property of $\phi$:**
  The cost function $\phi$ is assumed to be Lipschitz, meaning:
  $$
  |\phi(a^i, \tilde{a}) - \phi(a^i, \hat{a})| \leq C|\tilde{a} - \hat{a}|,
  $$
  where $C$ is the Lipschitz constant.

- **Approximation of $\tilde{a}$ by $\hat{a}$:**
  The empirical mean $\tilde{a}$ converges to the population mean $\hat{a}$ as $N \to \infty$. The deviation is of the order $O(1/N)$.

---
