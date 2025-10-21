sudo apt-get update

########################################################
# git
########################################################
sudo apt-get install -y git

########################################################
# docker
########################################################

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again
sudo apt-get update

# Install Docker Engine
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (allows running docker without sudo)
sudo usermod -aG docker $USER

# Apply the group changes for the current session
newgrp docker

# Verify installation (now running without sudo)
docker --version

echo "Docker installed successfully! You can now use docker without sudo."
echo "Note: If 'newgrp docker' didn't work, log out and log back in for group changes to take effect."