import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    plt.figure(figsize=(6,6))
    x = [0,1,2,3]
    plt.xlim([-0.01, 3.01])
    xlabels = ['0','1','2','3']
    plt.xticks(x, xlabels)
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Number of observations", fontsize=20)
    plt.ylabel("Exploration performance", fontsize=20)
    plt.savefig('../talk/talk_blank_perf.pdf')

    plt.figure(figsize=(6,6))
    x = [0,1,2,3]
    ye = [0.5, 0.75, 0.5, 0.5]
    yt = [0.5, 0.75, 1, 1]
    plt.plot(x, ye, 'r-')
    plt.xlim([-0.01, 3.01])
    xlabels = ['0','1','2','3']
    plt.xticks(x, xlabels)
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Number of observations", fontsize=20)
    plt.ylabel("Exploration performance", fontsize=20)
    plt.savefig('../talk/talk_exploration_perf.pdf')

    plt.figure(figsize=(6,6))
    x = [0,1,2,3]
    ye = [0.5, 0.75, 1, 1]
    yt = [0.5, 0.75, 1, 1]
    plt.plot(x, ye, 'r-')
    plt.xlim([-0.01, 3.01])
    xlabels = ['0','1','2','3']
    plt.xticks(x, xlabels)
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Number of observations", fontsize=20)
    plt.ylabel("Teaching performance", fontsize=20)
    plt.savefig('../talk/talk_teaching_perf.pdf')

    plt.figure(figsize=(6,6))
    x = [0,1,2,3]
    ye = [0.5, 0.75, 0.5, 0.5]
    yt = [0.5, 0.75, 1, 1]
    plt.hold(True)
    plt.plot([0,1], [0,1], 'k:', alpha=.5)
    plt.plot(ye, yt, 'r-')
    plt.hold(False)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Exploration performance", fontsize=20)
    plt.ylabel("Teaching performance", fontsize=20)
    plt.savefig('../talk/talk_evst.pdf')
