# Q-Learning

**Type:** Reinforcement Learning  
**Category:** Policy-based Algorithm

## Overview  
Q-Learning is a model-free reinforcement learning algorithm used to find the optimal action-selection policy for any given finite Markov decision process. It iterates over states and actions to maximize the expected value of the total reward over any and all successive steps, starting from the current state.

## Key Concepts
- **Q-value (Action-Value):** Measures the worth of taking a specific action from a specific state.
- **Learning Rate (α):** Determines to what extent newly acquired information overrides old information.
- **Discount Factor (γ):** Balances immediate and future rewards.

## How It Works  
Q-Learning updates the Q-values for each state-action pair using the Bellman equation as a recursive update:
1. Initialize the Q-values arbitrarily.
2. Observe the current state.
3. Choose and perform an action.
4. Measure the reward and update the Q-value for the state-action pair based on the reward received and the highest Q-value of the next state.

## Mathematical Model (Optional)  
`Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))`

## Pseudocode  
```plaintext
Initialize Q-values Q(s, a) arbitrarily
For each episode:
    Initialize state s
    For each step in episode:
        Choose action a from s using policy derived from Q (e.g., ε-greedy)
        Take action a, observe reward r, and next state s'
        Update Q(s, a): Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
        s = s'
    end for
end for
```

## Implementation (Python)
```python
def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.random() > epsilon:
                action = np.argmax(Q[state])
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
    return Q
```

## Applications  
Commonly used in gaming, robotics for navigation and decision-making tasks, and various control systems.

## Strengths
- Does not require a model of the environment.
- Can handle problems with stochastic transitions and rewards.

## Limitations
- Requires a lot of memory for storing Q-values if the state or action space is large.
- Slow to converge in environments with many states or actions.

## References and Further Reading  
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- Watkins, C. J. C. H. (1989). *Learning from delayed rewards*. Ph.D. dissertation, Cambridge University.
