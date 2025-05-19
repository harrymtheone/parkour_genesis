import torch
from matplotlib import pyplot as plt

data = torch.load('no_song.zip', weights_only=True, map_location='cpu')
data = data[:100]

#########################  GRU  #########################
# proprio = data[:, :50]
# hidden = data[:, 50:-13]
# actions = data[:, -13:]
#
# hidden_zero = torch.zeros(1, 1, 128)
# model = torch.jit.load('/home/harry/projects/parkour_genesis/scripts/dream_gru/traced/t1_dream_017r1_7000_jit.pt')
# actions_pred = torch.zeros_like(actions)
# with torch.inference_mode():
#     for i in range(1000):
#         act, hidden_zero = model(proprio[i:i + 1], hidden_zero)
#         actions_pred[i:i + 1] = act
# actions_pred = actions_pred.numpy()
#
# x = torch.arange(1000)
#
# # Setup subplot grid (7 rows, 2 columns)
# fig, axes = plt.subplots(7, 2, figsize=(12, 16))
#
# # axes[0, 0].plot(x, actions[:, 0])
# # axes[0, 0].plot(x, actions_pred[:, 0])
# #
# # for i in range(6):
# #     axes[i + 1, 0].plot(x, actions[:, 1 + 2 * i])
# #     axes[i + 1, 1].plot(x, actions[:, 1 + 2 * i])
# #     axes[i + 1, 0].plot(x, actions_pred[:, 1 + 2 * i])
# #     axes[i + 1, 1].plot(x, actions_pred[:, 1 + 2 * i])
#
# # base_ang_vel
# # axes[0, 0].plot(x, proprio[:, 0])
# # axes[1, 0].plot(x, proprio[:, 1])
# # axes[2, 0].plot(x, proprio[:, 2])
#
# # gravity
# # axes[0, 0].plot(x, proprio[:, 3])
# # axes[1, 0].plot(x, proprio[:, 4])
# # axes[2, 0].plot(x, proprio[:, 5])
#
#
# # axes[0, 0].plot(x, proprio[:, 37])
# #
# # for i in range(6):
# #     axes[i + 1, 0].plot(x, proprio[:, 38 + 2 * i])
# #     axes[i + 1, 1].plot(x, proprio[:, 38 + 2 * i + 1])
#
# plt.tight_layout()
# plt.show()

#########################  SG  #########################
proprio = data[:, :47]
hidden = data[:, 47:-12]
actions = data[:, -12:]

x = torch.arange(len(data))

# Setup subplot grid (7 rows, 2 columns)
fig, axes = plt.subplots(7, 2, figsize=(12, 16))

# # commands
# axes[0, 0].plot(x, proprio[:, 0])
# axes[1, 0].plot(x, proprio[:, 1])
# axes[2, 0].plot(x, proprio[:, 2])
# axes[3, 0].plot(x, proprio[:, 3])
# axes[4, 0].plot(x, proprio[:, 4])

# # dof_pos
# for i in range(6):
#     axes[i, 0].plot(x, proprio[:, 5 + 2 * i])
#     axes[i, 1].plot(x, proprio[:, 5 + 2 * i + 1])

# # dof_vel
# for i in range(6):
#     axes[i, 0].plot(x, proprio[:, 17 + 2 * i])
#     axes[i, 1].plot(x, proprio[:, 17 + 2 * i + 1])

# # last actions
# for i in range(6):
#     axes[i, 0].plot(x, proprio[:, 29 + 2 * i])
#     axes[i, 1].plot(x, proprio[:, 29 + 1 + 2 * i])

# # commands
# axes[0, 0].plot(x, proprio[:, 41])
# axes[1, 0].plot(x, proprio[:, 42])
# axes[2, 0].plot(x, proprio[:, 43])

# # base euler
# axes[0, 0].plot(x, proprio[:, 44])
# axes[1, 0].plot(x, proprio[:, 45])
# axes[2, 0].plot(x, proprio[:, 46])

plt.tight_layout()
plt.show()
