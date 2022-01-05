def Save_history_to_h5(DIR_OUT, file_name, extension, patience, tvalues_check):
    '''
    The function save the 2D-array of the loss histories in an .h5 file.
    
    DIR_OUT: directory where to save the output file
    file_name: output file name
    extension: label to be appended to the file_name
    
    No return.
    '''
    epochs_check = []
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
        
    log_file = DIR_OUT+file_name+extension+'.h5'
    print(log_file)
    f = h5py.File(log_file,"w")
    for i in range(tvalues_check.shape[1]):
        f.create_dataset(str(epochs_check[i]), data=tvalues_check[:, i], compression='gzip')
    f.close()
    print('Saved to file: ' +file_name+extension)
    return

def Read_history_from_h5(DIR_IN, file_name, extension, patience, epochs):
    '''
    The function creates a 2D-array from a .h5 file.
    
    DIR_OUT: directory where to save the input file
    file_name: input file name
    extension: label to be appended to the file_name
    
    The function returns a 2D-array with final shape (nr toys, nr check points). 
    '''
    
    tvalues_check = np.array([])
    epochs_check = []
    
    for i in range(int(epochs/patience)):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
        
    log_file = DIR_IN+file_name+extension+'.h5'
    print(log_file)

    f = h5py.File(log_file,"r")
    for i in range(len(epochs_check)):
        # the t distribution at each check point is named by the epoch number
        t = f.get(str(epochs_check[i]))
        t = np.array(t)
        t = np.expand_dims(t, axis=1)
        if not i:
            tvalues_check = t
        else:
            tvalues_check = np.concatenate((tvalues_check, t), axis=1)
    f.close()
    print(tvalues_check.shape)
    return tvalues_check
    
def Plot_Analysis_tdistribution(output_path, title, tvalues_BkgOnly, tvalues, dof, rmin, rmax, bins=35, verbose=0, save=0):
    '''
    The function creates the plot for the comparison of two samples of toys at the end of the training.
    tvalues_BkgOnly: t distribution for the sample with BKG-only events.
    tvalues: t distribution for the sample with Sig+Bkg events.
    dof: number of degrees of freedom of the reference chi2 distribution.
    
    '''
    fig, ax = plt.subplots()
    chisq = np.random.chisquare(dof, 5000000)
    ax.hist(chisq, bins=bins, range=(rmin, rmax), density = True, histtype = 'step', linewidth=2, color='darkgreen')
    ax.hist(tvalues_BkgOnly, bins=bins, range=(rmin, rmax), density = True, alpha = 0.7, edgecolor='blue')
    ax.hist(tvalues, bins=bins, range=(rmin, rmax), density= True, alpha = 0.7, edgecolor='red')
    ax.legend(["$\chi^2$ with "+str(dof)+" dof",'Data samples following SM','Data samples containing New Physics'], loc='upper right')
    ax.set_ylabel('Probability')
    ax.set_xlabel("t")
    ax.set_title(title)
    #compute significance
    quantiles=np.percentile(tvalues, [16., 50., 84.])
    q50=quantiles[1]
    q16=quantiles[0]
    q84=quantiles[2]
    counts50 = np.sum((tvalues_BkgOnly > q50).astype(int))
    counts16 = np.sum((tvalues_BkgOnly > q16).astype(int))
    counts84 = np.sum((tvalues_BkgOnly > q84).astype(int))
    
    p_val50 = counts50*1./len(tvalues_BkgOnly)
    p_val16 = counts16*1./len(tvalues_BkgOnly)
    p_val84 = counts84*1./len(tvalues_BkgOnly)
    
    chisq = np.random.chisquare(dof, 100000000)
    integral50 = (chisq > q50).sum()/float(len(chisq))
    integral16 = (chisq > q16).sum()/float(len(chisq))
    integral84 = (chisq > q84).sum()/float(len(chisq))
    
    print("Bkg-only median: %f" %np.median(tvalues_BkgOnly))
    print("Bkg-only mean: %f" %np.mean(tvalues_BkgOnly))
    print("Bkg-only RMS: %f" %math.sqrt(np.var(tvalues_BkgOnly)))
    print("Sig+Bkg median: %f" %np.median(tvalues))
    print("Sig+Bkg quantile16: %f" %q16)
    print("Sig+Bkg quantile84: %f" %q84)
    print("Sig+Bkg mean: %f" %np.mean(tvalues))
    print("Sig+Bkg RMS: %f" %math.sqrt(np.var(tvalues)))
    
    print("p-value %f with 68 %% CL [%f, %f]" %(p_val50, p_val16, p_val84))
    print("number of sigmas: %f with 68%% CL [%f, %f]" %(norm.ppf(1.-p_val50), norm.ppf(1.-p_val16), norm.ppf(1.-p_val84)))
    print("p-value assuming %i dof chi square: %f" %(dof, integral50))
    print("number of sigmas assuming %i dof chi square: %f with 68 %% CL [%f, %f]" %(dof, norm.ppf(1.-integral50), norm.ppf(1.-integral16), norm.ppf(1.-integral84)))
    textstr = "Bkg-only median: %f\nSig+Bkg median: %f\nSignificance: %f $\sigma$\nTh Significance: %f $\sigma$" %(np.median(tvalues_BkgOnly), np.median(tvalues), norm.ppf(1.-p_val50), norm.ppf(1.-integral50))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)

    # place a text box in upper left in axes coords
    ax.text(0.5, 0.65, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
    if verbose:
        plt.show()
    fig.savefig(output_path+title+'_t_distributions_dof'+str(dof)+'.pdf')
    plt.close(fig)
    return 

def get_percentiles_Zscore(t, df, percentage_list=[], verbose=False):
    '''
    For a given test statistic sample (t), it returns the percentile and the corresponding Z-score for each percentage given in percentage_list.
    '''
    p = np.percentile(t, percentage_list)
    z = norm.ppf(chi2.cdf(p, df))
    if verbose:
        for i in range(p.shape[0]):
            print('%s percentile: %s, Z-score: %s'%(str(np.around(percentage_list[i], 2)), str(np.around(p[i], 2)), str(np.around(z[i], 2)) ))
    return p, z

def get_percentage_from_Zscore (t, df, Zscore_star_list=[], verbose=False):
    '''
    For a given test statistic sample (t), it returns the percentage of toys with Zscore greater or equal to Z-score-star for each Z-score-star in Zscore_star_list.
    '''
    t_star_list = chi2.ppf(norm.cdf(np.array(Zscore_star_list)),df)
    percentage  = np.array([np.sum(t>t_star)*1./t.shape[0] for t_star in t_star_list])
    if verbose:
        for i in range(percentage.shape[0]):
            print('Z-score > %s: t > %s, percentage: %s'%(str(np.around(Zscore_star_list[i], 2)), str(np.around(t_star_list[i], 2)), str(np.around(percentage[i], 2)) ))
    return t_star_list, percentage

def plot_1distribution(t, df, xmin=0, xmax=300, nbins=10, save=False, output_path='', save_name='', label=''):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution (df must be specified!). 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    Z_obs     = norm.ppf(chi2.cdf(np.median(t), df))
    t_obs_err = 1.2533*np.std(t)*1./np.sqrt(t.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t)-t_obs_err, df))
    label  = 'sample %s\nsize: %i \nmedian: %s, std: %s\n'%(label, t.shape[0], str(np.around(np.median(t), 2)),str(np.around(np.std(t), 2)))
    label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    binswidth = (xmax-xmin)*1./nbins
    h = plt.hist(t, weights=np.ones_like(t)*1./(t.shape[0]*binswidth), color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'$\chi^2$('+str(df)+')')
    font = font_manager.FontProperties(family='serif', size=14) 
    plt.legend(prop=font)
    plt.xlabel('t', fontsize=18, fontname="serif")
    plt.ylabel('Probability', fontsize=18, fontname="serif")
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(output_path+ save_name+'_distribution.pdf')
    plt.show()
    plt.close(fig)
    return

def plot_2distribution(t1, t2, df, xmin=0, xmax=300, nbins=10, save=False, output_path='', label1='1', label2='2', save_name=''):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution (df must be specified!).
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    binswidth = (xmax-xmin)*1./nbins
    # t1
    Z_obs     = norm.ppf(chi2.cdf(np.median(t1), df))
    t_obs_err = 1.2533*np.std(t1)*1./np.sqrt(t1.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t1)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t1)-t_obs_err, df))
    label  = 'sample %s\nsize: %i\nmedian: %s\nstd: %s\n'%(label1, t1.shape[0], str(np.around(np.median(t1), 2)),str(np.around(np.std(t1), 2)))
    label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    h = plt.hist(t1, weights=np.ones_like(t1)*1./(t1.shape[0]*binswidth), color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t1.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    # t2
    Z_obs     = norm.ppf(chi2.cdf(np.median(t2), df))
    t_obs_err = 1.2533*np.std(t2)*1./np.sqrt(t2.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t2)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t2)-t_obs_err, df))
    label  = 'sample %s\nsize: %i\nmedian: %s\nstd: %s\n'%(label2, t2.shape[0], str(np.around(np.median(t2), 2)),str(np.around(np.std(t2), 2)))
    label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    h = plt.hist(t2, weights=np.ones_like(t2)*1./(t2.shape[0]*binswidth), color='#8dd3c7', ec='seagreen',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t2.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='seagreen', marker='o', ls='')
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'$\chi^2$('+str(df)+')')
    font = font_manager.FontProperties(family='serif', size=14) #weight='bold', style='normal', )
    plt.legend(ncol=1, loc='upper right', prop=font)
    plt.xlabel('t', fontsize=14, fontname="serif")
    plt.ylabel('Probability', fontsize=14, fontname="serif")
    plt.ylim(0., np.max(chi2.pdf(x, df))*1.3)
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(output_path+ save_name+'_2distribution.pdf')
    plt.show()
    plt.close()
    return

def compute_df(input_size, hidden_layers, output_size=1):
    """
    Compute degrees of freedom of a neural net (number of trainable params)
    Parameters
    ----------
    input_size : int
        size of the input layer
    hidden_layers : list
        list specifiying size of hidden layers
    latentsize : int
        number of hidden units for each layer
    Returns
    -------
    df : int
        degrees of freedom
    """
    
    # nn_arch = [input_size] + [latentsize for i in range(hidden_layers)] + \
    #     [output_size]
    
    nn_arch = [input_size] + hidden_layers + [output_size]
    
    df = sum(map(lambda x, y : x*(y+1), nn_arch[1:], nn_arch[:-1]))
    
    return df

def Extract_Tail(tvalues_check, patience, cut=95):
    tail_distribution = np.array([])
    normal_distribution = np.array([])
    epochs_check = []
    size = tvalues_check.shape[0]
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
        
    for i in range(tvalues_check.shape[1]):
        tvalues = np.sort(tvalues_check[:, i])
        percentile_cut = int(cut*0.01*size)
        bulk_distribution_i = tvalues[:percentile_cut]
        bulk_distribution_i = np.expand_dims(bulk_distribution_i, axis=1)
        tail_distribution_i = tvalues[percentile_cut:]
        tail_distribution_i = np.expand_dims(tail_distribution_i, axis=1)
        if not i:
            tail_distribution = tail_distribution_i
            bulk_distribution = bulk_distribution_i
        else:
            tail_distribution = np.concatenate((tail_distribution, tail_distribution_i), axis=1)
            bulk_distribution = np.concatenate((bulk_distribution, bulk_distribution_i), axis=1)
    print('Tail distributions shape')
    print(tail_distribution.shape)
    print('Bulk distributions shape')
    print(bulk_distribution.shape)
    return tail_distribution, bulk_distribution

def Plot_Percentiles(tvalues_check, patience=1, title='', ylabel='t', ymax=300, ymin=0, save=False, output_path=''):
    '''
    The function creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    
    patience: interval between two check points (epochs).
    tvalues_check: array of t=-2*loss, shape = (N_toys, N_check_points)
    '''
    colors = [
    'seagreen',
    'mediumseagreen',
    'lightseagreen',
    '#2c7fb8',
    'midnightblue',
    ]
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    epochs_check = []
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
    
    fig=plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    quantiles=[2.5, 25, 50, 75, 97.5]
    percentiles=np.array([])
    plt.xlabel('Epoch', fontsize=16, fontname="serif")
    plt.ylabel(ylabel, fontsize=16, fontname="serif")
    plt.ylim(ymin, ymax)
    for i in range(tvalues_check.shape[1]):
        percentiles_i = np.percentile(tvalues_check[:, i], quantiles)
        #print(percentiles_i.shape)
        percentiles_i = np.expand_dims(percentiles_i, axis=1)
        #print(percentiles_i.shape)
        if not i:
            percentiles = percentiles_i.T
        else:
            percentiles = np.concatenate((percentiles, percentiles_i.T))
    legend=[]
    print(percentiles.shape)
    for j in range(percentiles.shape[1]):
        plt.plot(epochs_check, percentiles[:, j], marker='.', color=colors[j])
        legend.append(str(quantiles[j])+' % quantile')
    plt.legend(legend, fontsize=16)
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    plt.grid()
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            fig.savefig(output_path+title+'_PlotPercentiles.pdf')
    plt.show()
    plt.close(fig)
    return

def Plot_Percentiles_ref(tvalues_check, dof, patience=1, title='', wc=None, ymax=300, ymin=0, save=False, output_path=''):
    '''
    The funcion creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    The percentile lines for the target chi2 distribution (dof required!) are shown as a reference.
    patience: interval between two check points (epochs).
    tvalues_check: array of t=-2*loss, shape = (N_toys, N_check_points)
    '''
    colors = [
    'seagreen',
    'mediumseagreen',
    'lightseagreen',
    '#2c7fb8',
    'midnightblue',
    ]
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    epochs_check = []
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
    
    fig=plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    quantiles=[2.5, 25, 50, 75, 97.5]
    percentiles=np.array([])
    plt.xlabel('Training Epochs', fontsize=16, fontname="serif")
    plt.ylabel('t', fontsize=16, fontname="serif")
    plt.ylim(ymin, ymax)
    if wc != None:
        plt.title('Weight Clipping = '+wc, fontsize=16,  fontname="serif")
    for i in range(tvalues_check.shape[1]):
        percentiles_i = np.percentile(tvalues_check[:, i], quantiles)
        #print(percentiles_i.shape)
        percentiles_i = np.expand_dims(percentiles_i, axis=1)
        #print(percentiles_i.shape)
        if not i:
            percentiles = percentiles_i.T
        else:
            percentiles = np.concatenate((percentiles, percentiles_i.T))
    legend=[]
    #print(percentiles.shape)
    for j in range(percentiles.shape[1]):
        plt.plot(epochs_check, percentiles[:, j], marker='.', linewidth=3, color=colors[j])
        #print(percentiles[:, j])
        legend.append(str(quantiles[j])+' % quantile')
    for j in range(percentiles.shape[1]):
        plt.plot(epochs_check, chi2.ppf(quantiles[j]/100., df=dof, loc=0, scale=1)*np.ones_like(epochs_check),
                color=colors[j], ls='--', linewidth=1)
        #print( chi2.ppf(quantiles[j]/100., df=dof, loc=0, scale=1))
        if j==0:
            legend.append("Target "+r"$\chi^2(dof=$"+str(dof)+")")
    font = font_manager.FontProperties(family='serif', size=16)         
    plt.legend(legend, prop=font)
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            fig.savefig(output_path+title+'_PlotPercentiles.pdf')
    plt.show()
    plt.close(fig)
    return

def plot_alpha_scores(t1,t2, label1, label2, df, Zscore_star_list=[2, 3, 5], verbose=False, save=False, output_path='', title=''):
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    y1, y2 = get_percentage_from_Zscore (t1, df, Zscore_star_list)
    plt.errorbar(Zscore_star_list, y2, yerr=y2*np.sqrt((1.+1./y2)/t1.shape[0]), lw=1.5, ms=10, capsize=5, capthick=3, elinewidth=3,
                 color='#2c7fb8', ls='--', marker='o', label=label1)
    for i, txt in enumerate(y2):
        plt.annotate(str(np.around(txt, 2)), (Zscore_star_list[i]+0.025, 0.025+y2[i]), 
                     color='#2c7fb8', fontname='serif', fontsize=18)
    y1, y2 = get_percentage_from_Zscore (t2, df, Zscore_star_list)
    plt.errorbar(Zscore_star_list, y2, yerr=y2*np.sqrt((1.+1./y2)/t2.shape[0]), lw=1.5, ms=10, capsize=5, capthick=3, elinewidth=3,
                 color='seagreen', ls='--', marker='o', label=label2)
    for i, txt in enumerate(y2):
        plt.annotate(str(np.around(txt, 2)), (Zscore_star_list[i]+0.025, -0.05+y2[i]), 
                     color='seagreen', fontname='serif', fontsize=18)
    plt.xlabel('Z-score', fontsize=20, fontname='serif')
    plt.ylabel(r"P($\alpha$)", fontsize=20, fontname='serif')
    plt.ylim(-0.075, 1.075)
    plt.xlim(Zscore_star_list[0]-0.3, Zscore_star_list[-1]+0.3)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
               [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
               fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    font = font_manager.FontProperties(family='serif', size=18)         
    plt.legend(ncol=1, loc='upper right', prop=font)
    plt.grid()
    if verbose:
        plt.show()
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            fig.savefig(output_path+title+'_alpha_plot.pdf')
    
    plt.close()
    return

def plot_reconstruction(DATA, weight_DATA, REF, weight_REF, 
                        tau_OBS, delta_OBS, df,
                        output_tau_ref, output_delta_ref,
                        feature_labels, bins_code, xlabel_code, 
                        save=False, save_path='', save_name=''):
    
    Zscore=norm.ppf(chi2.cdf(tau_OBS-delta_OBS, df))
    
    plt_i = 0
    for key in feature_labels:
        bins = bins_code[key]
        plt.rcParams["font.family"] = "serif"
        plt.style.use('classic')
        fig = plt.figure(figsize=(10, 10))                                                                                                                                            
        fig.patch.set_facecolor('white')                                                                                                                                              
        ax1 = fig.add_axes([0.1, 0.43, 0.8, 0.5])        
        hD = plt.hist(DATA[:, plt_i],weights=weight_DATA, 
                      bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=4)
        hR = plt.hist(REF[:, plt_i], weights=weight_REF, color='#a6cee3',     
                      ec='#1f78b4', bins=bins, lw=1, label='REFERENCE')
        hN = plt.hist(REF[:, plt_i], weights=np.exp(output_tau_ref[:, 0])*weight_REF, 
                      histtype='step', bins=bins, lw=0)
        hN2= plt.hist(REF[:, plt_i], weights=np.exp(output_delta_ref[:, 0])*weight_REF, 
                      histtype='step', bins=bins, lw=0)
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0],  yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0],  edgecolor='black', label=r'$\tau$ RECO',   color='#b2df8a', lw=1, s=30, zorder=2)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN2[0], edgecolor='black', label=r'$\Delta$ RECO', color='#33a02c', lw=1, s=30, zorder=1)
        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18) 
        title  = r'$\tau(D,\,A)$='+str(np.around(tau_OBS, 2))
        title += r', $\Delta(D,\,A)$='+str(np.around(delta_OBS, 2))
        title += ', Z-score='+str(np.around(Zscore, 2))
        l.set_title(title=title, prop=font)
        #l.set_title(title=r'$\tau(D)$='+str(np.around(tau_OBS, 2))+r', $\Delta(D)$='+str(np.around(delta_OBS, 2)), prop=font)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3]) 
        x = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/hR[0], yerr=np.sqrt(hD[0])/hR[0], ls='', marker='o', label ='DATA/REF', color='black')
        plt.plot(x, hN[0]/hR[0], label =r'$\tau$ RECO/REF', color='#b2df8a', lw=3)
        plt.plot(x, hN2[0]/hR[0], ls='--', label =r'$\Delta$ RECO/REF', color='#33a02c', lw=3)
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)
        plt.xlabel(xlabel_code[key], fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')
        plt.ylim(0., ymax_code[key])
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])

        plt.grid()
        if save:
            if save_path=='':
                print('argument save_path is not defined. The figure will not be saved.')
            else:
                plt.savefig('%s/%s_%sReconstruction.pdf'%(save_path, save_name, xlabel_code[key]))
        plt.show()
        plt.close()
        plt_i+=1
    return

def Read_from_h5(path, title, extension, epochs_check):
    log_file = path+title+extension
    print(log_file)
    tvalues_check = np.array([])
    f = h5py.File(log_file,"r")
    for i in range(len(epochs_check)):
        t=f.get(str(epochs_check[i]))
        t=np.array(t)
        t=np.expand_dims(t, axis=1)
        print(t.shape)
        if not i:
            tvalues_check = t
        else:
            tvalues_check = np.concatenate((tvalues_check, t), axis=1)
    f.close()
    print(tvalues_check.shape)
    return tvalues_check