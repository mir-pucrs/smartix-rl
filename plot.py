import PyGnuplot as gp

with open("plots/averages_rewards_history.gnu") as f: 
# with open("plots/averages_episode_reward.gnu") as f: 
    gp.c(f.read())
    gp.pdf('rewards_history_plot.pdf')