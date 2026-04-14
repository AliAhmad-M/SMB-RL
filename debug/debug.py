import matplotlib.pyplot as plt
from environments.smb_env import make_smb_env

env = make_smb_env()

# Reset
state = env.reset()

# Move right
for _ in range(24):
    state, reward, done, trunc, info = env.step(0)
    if done:
        state = env.reset()

plt.figure(figsize=(5, 5))
plt.imshow(state[-1], cmap='gray')
plt.title(f"Mario at x_pos: {info['x_pos']}")
plt.axis('off')
plt.savefig("debug_output.png")