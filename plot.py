# Plot Graph
def make_graph(actual,result,title,n_days,plt):
    '''Return the plot
    
    Keyword arguments:
    actual -- Actual Result
    result -- Predicted Value
    title -- Name of classifier
    n_days -- Days ahead
    plt -- Matplotlib object
    '''
    plt.plot(actual,label='Actual Value')
    plt.plot(result,color='r',label='Predicted Value')
    plt.xlabel('Number of Days')
    plt.ylabel('Target')
    plt.legend()
    plt.title(title+" (" + n_days + " days ahead)")
