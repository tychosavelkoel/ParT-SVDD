import matplotlib.pyplot as plt
import numpy as np

def plot_distance(radius, distance, title = 'Distance'):
    """ 
    DESCRIPTION
     
    Plot the curve of distance
    --------------------------------------------------------------- 
     
    """ 
    n = len(distance)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    radius = np.ones((n, 1))*radius

    print("make plot")
        
    #for i in range(len(distance))
    #   mask.append(distance[i] <= radius[i])
        
    ax.plot(distance,
            color='k',
            linestyle=':',
            marker='o',
            linewidth=1,
            markeredgecolor='k',
            markerfacecolor='C4',
            markersize=6)
            
    ax.plot(radius, 
            color='r',
            linestyle='-', 
            marker='None',
            linewidth=3, 
            markeredgecolor='k',
            markerfacecolor='w', 
            markersize=6)
    
    ax.set_xlabel('Samples')
    ax.set_ylabel('Squared distance')
        
    ax.legend(["Distance", "Radius"], 
               ncol=1, loc=0, 
               edgecolor='black', 
               markerscale=1, fancybox=True)
    ax.yaxis.grid()
    plt.savefig(title + '.png')
    plt.savefig(title + '.pdf')

def plot_loss(loss, loss_val, ylabel = 'Loss', title = 'loss'):
    n = len(loss)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(loss, 
            color='r',
            linestyle='-', 
            marker='None',
            linewidth=3, 
            markeredgecolor='k',
            markerfacecolor='w', 
            markersize=6)

    ax.plot(loss_val, 
            color='b',
            linestyle='-', 
            marker='None',
            linewidth=3, 
            markeredgecolor='k',
            markerfacecolor='w', 
            markersize=6)
       
    ax.set_xlabel('Epoch')
    ax.set_xlim(0,len(loss))
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    
    ax.legend(["train", "val"], 
               ncol=1, loc=0, 
               edgecolor='black', 
               markerscale=1, fancybox=True)

    ax.yaxis.grid()
    plt.savefig(title + '.png')
    plt.savefig(title + '.pdf')