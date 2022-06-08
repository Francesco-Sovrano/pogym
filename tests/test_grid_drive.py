import time
from pogym.envs.grid_drive import GridDrive

env = GridDrive(culture_level="Easy", partial_observability=True)

def run_one_episode (env):
	env.seed(38)
	env.reset()
	sum_reward = 0
	done = False
	while not done:
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)
		sum_reward += reward
		env.render()
		time.sleep(0.25)
	return sum_reward

sum_reward = run_one_episode(env)