import os
import re
import matplotlib.pyplot as plt

# Directory containing the files
directory = "./"  # Change if needed

# Pattern to extract config name and train loss values
file_pattern = re.compile(r"products_(.+)\.txt")
loss_pattern = re.compile(r"Epoch: \d+, Train Loss: ([\d\.]+)")

# Dictionary to store losses per config
losses_dict = {}

# Iterate over all files in the directory
for filename in os.listdir(directory):
    match = file_pattern.match(filename)
    if match:
        config_name = match.group(1)
        losses = []

        # Read the file and extract losses
        with open(os.path.join(directory, filename), "r") as file:
            for line in file:
                loss_match = loss_pattern.search(line)
                if loss_match:
                    losses.append(float(loss_match.group(1)))

        # Store the extracted losses
        if losses:
            losses_dict[config_name] = losses

# Plot the losses
plt.figure(figsize=(10, 6))
for config, losses in losses_dict.items():
    plt.plot(losses, label=config)

plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.title("Training Loss per Configuration")
plt.legend()
plt.grid(True)

plt.savefig("val.png")
