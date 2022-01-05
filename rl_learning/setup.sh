sudo apt-get update
sudo umount /media/ssd
sudo mkdir /media/ssd
{ echo "y"; } | sudo mkfs.ext4 /dev/sda4
sudo mount /dev/sda4 /media/ssd/
sudo chmod 777 -R /media/ssd/
df -h | grep ssd
sudo apt-get install python3.7
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
sudo update-alternatives --config python3
sudo apt-get install python3-pip
pip3 install --upgrade pip
pip3 install networkx
pip3 install simpy
pip3 install scipy
pip3 install numpy
pip3 install pandas
sudo apt-get install python3-opencv
sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip3 install --upgrade tensorflow==1.14
pip3 install stable_baselines
git fetch --all
git reset --hard origin/master

# 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 
