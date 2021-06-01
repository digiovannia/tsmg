# TSMG

Code for replicating results of DiGiovanni and Tewari (2021).

Procedure to replicate Figures 2 and 4:

1.) Run python TSMG.py for each of the following inputs when prompted. Order doesn’t matter, and experiments for the different games are split up so that experiments can be run in parallel.  The script will create the necessary subfolders in your current directory. Some plots will be generated as well, to visualize the individual (not averaged) runs, but these are not used in Figs. 2/4.

  1.1.) 1 2
  
  1.2.) 1 p
  
  1.3.) 1 b
  
  1.4.) 1 g
  
  1.5.) 0 1 2
  
  1.6.) 0 1 p
  
  1.7.) 0 1 b
  
  1.8.) 0 1 g
  
  1.9.) 0 0 1 2
  
  1.10.) 0 0 1 p
  
  1.11.) 0 0 1 b
  
  1.12.) 0 0 1 g
  
  1.13.) 0 0 0 1 2
  
  1.14.) 0 0 0 1 p
  
  1.15.) 0 0 0 1 b
  
  1.16.) 0 0 0 1 g

2.) Making sure TSMG.py is in your current directory and that that directory is “multi”, with subfolders containing all the data files produced in step 1, run python result_plots.py. Responses to the prompts are irrelevant (this is a side effect of importing TSMG). The plots will appear in each corresponding folder constructed in step 1.
