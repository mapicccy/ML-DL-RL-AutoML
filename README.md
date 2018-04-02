# ML-DL-RL-AutoML
machine learning/deep learning/reinforcement learning/autoML

## ML

## DL

## RL
### Deep-Q-Network
algorithm:

![dqn_algorithm](./images/dqn_algorithm.png)

result on CartPole:

result on MountainCar: ##learning rate = 0.01, gamma = 0.9, episodes = 3000, steps of every episode = 2000, start learning steps = 200

![run_mountain_car_on_dqn](./images/run_mountain_car_on_dqn.PNG)

result on MountainCar: ##learning rate = 0.01, gamma = 0.9, episodes = 3000, steps of every episode = 2000, start learning steps = 100

![run_mountain_car_on_dqn](./images/run_mountain_car_on_dqn_diff.PNG)

result on MountainCar: ##learning rate = 0.001, gamma = 0.9, episodes = 3000, steps of every episode = 2000, start learning steps = 0, set_replace_target_iter = 200

![run_mountain_car_on_dqn_no_iter](./images/run_mountain_car_on_dqn_no_iter.PNG)

we can see that different start_learning_step leads to totally different results

### Actor-Critic
algorithm:

result:
learning rate of Actor = 0.001, learning rate of Critic = 0.01, Gamma = 0.8
episodes = 3000, steps of every episode = 1000

![actor-critic](./images/actor-critic.png)

function run_double build positive and negative network, but the result is still not so good as positive only network

the result of function run_double train positive network only:
![run_double_positive_only](./images/run_double_positive_only.png)

the result of function run_double train positive and negative network:
![run_double_positive_and negative](./images/run_double_positive_and_negative.png)

## AutoML
