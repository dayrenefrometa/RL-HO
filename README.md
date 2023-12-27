# Introduction

This repository contains the simulator code of a prediction-model-assisted reinforcement learning algorithm for handover decision-making in hybrid LiFi and WiFi networks (HLWNets). The algorithm is called RL-HO, and it is detailed in a paper published in the 
Journal of Optical Communications and Networking under the title: "A prediction-model-assisted reinforcement learning algorithm for handover decision-making in hybrid LiFi and WiFi networks". A link to the publication will be added soon.

# Instructions

The code is divided into two folders: 
1. Dataset_Generation contains the scripts to generate the dataset for trajectory prediction. There are three scripts for different groups of APs that must be executed one after the other (the order does not matter).

   Note:
    * The first script executed will produce a dataset.csv file, and the other scripts will append the data to the end of the existing dataset.csv file.
    * The produced dataset.csv file is an input for the main script implementing the RL-HO algorithm.
3. RL-HO_Algorithm contains the main script implementing the RL-HO algorithm.

   Note:
    * For a better understanding of the code, please, refer to our paper.
  
# References

"A prediction-model-assisted reinforcement learning algorithm for handover decision-making in hybrid LiFi and WiFi networks", D Frómeta Fonseca, B Genovés Guzmán, GL Mertena, R Bian, H Haas, and D Giustiniano, Journal of Optical Communications and Networking, 2023.

# License

This code is released under GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.


