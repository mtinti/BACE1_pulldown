import warnings
warnings.filterwarnings("ignore")
#define helping function
import os
from tqdm import tqdm_notebook
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from adjustText import adjust_text
from matplotlib.lines import Line2D
from Bio import SeqIO
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import matplotlib
import inspect, re
plt.style.use('ggplot')

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]

def compare_sets(s1=set(),s2=set(),name1='s1',name2='s2'):
    common = len(set(s1) & set(s2))
    uS1 = len(set(s1) - set(s2))
    uS2 = len(set(s2) - set(s1))
    res = pd.DataFrame(columns=[name1,name2])
    res.loc['size',:]=[len(s1),len(s2)]
    res.loc['common',:]=[common,common]
    res.loc['unique',:]=[uS1,uS2]
    str_report="""
    {lenS1} in {s1}
    {lenS2} in {s2} 
    {common} in common
    {uS1} unique {s1}
    {uS2} unique in {s2}     
    """.format(
        s1 = res.columns[0],
        s2 = res.columns[1],
        lenS1 = res.loc['size',res.columns[0]],
        lenS2 = res.loc['size',res.columns[1]],
        common=res.loc['common',res.columns[0]],
        uS1 = res.loc['unique',res.columns[0]],
        uS2 = res.loc['unique',res.columns[1]],
              )
    return str_report,res




def quantileNormalize(df_input, keep_na=True):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        norm = [rank[i] for i in t]
        if keep_na == True:
            norm = [np.nan if np.isnan(a) else b for a,b in zip(df[col],norm)]
        df[col] =  norm             
    return df


#get only the gene id from
#the new TryTripDB format
def clean_id(temp_id):
    temp_id = temp_id.split(':')[0]
    if temp_id.count('.')>2:
        temp_id = '.'.join(temp_id.split('.')[0:3])
    return temp_id

#helper function to print out
#the protein removed at each threshold
def print_result(start_df_shape, shape_before, df, what):
    removed = shape_before[0]- df.shape[0]
    removed_from_beginning = start_df_shape[0]-df.shape[0]
    if removed > 0:
        print ('removed ',removed, 'Protein Groups by:',what )  
        print ('tot ', removed_from_beginning, ' entries removed' )
        print ('---------------')
    else:
        print (what)
        print ('nothing removed')
        print ('---------------')

#remove rubbish entires from a
#maxquant output
def clean_df(df, id_by_site=True, rev_database=True, 
             contaminant=True, score=False, unique_pep_threshold=2):  
    before,start = df.shape,df.shape
    print('starting from:', before)
    if id_by_site:
        #remove Only identified by site
        before,start = df.shape,df.shape
        col = 'Only identified by site'
        df = df[df[col] != '+'] 
        print_result(start, before, df, col)
    
    if rev_database:
        #remove hits from reverse database
        before = df.shape
        col = 'Reverse'
        df = df[df[col] != '+']
        print_result(start, before, df, col)
     

    if contaminant:
        #remove contaminants (mainly keratine and bsa)
        before = df.shape
        col = 'Potential contaminant'
        df = df[df[col] != '+']
        print_result(start, before, df, col)

    if score:
        before = df.shape
        col = 'Score'
        df = df[df[col] >= score]
        print_result(start, before, df, col)
        
        
    ##remove protein groups with less thatn 2 unique peptides
    before = df.shape
    col = 'Peptide counts (unique)'
    df['unique_int'] = [int(n.split(';')[0]) for n in df[col]]
    df = df[df['unique_int'] >= unique_pep_threshold]
    print_result(start, before, df, col)
    return df  




#extract the description from the fasta headers
#of the proten group
def make_desc(n, lookfor='gene_product'):
    temp_dict = {}
    n=str(n)
    if 'sp|' in n:
        item_list = n.split(';')
        desc = []
        for n in item_list[0].split(' '):
            if '=' not in n and 'sp|' not in n:
                desc.append(n)
            if '=' in n:
                break
        desc = ' '.join(desc)
        return desc
           
    item_list = n.split(' | ')
    for n in item_list:
        if '=' in n:
            key = n.split('=')[0].strip()
            value= n.split('=')[1].strip()
            temp_dict[key]=value

                  
    return temp_dict.get(lookfor,'none')


#rename some of maxquant output 
#columns
def mod_df(df, desc_from_id=False, desc_value='gene_product' ):
    df['Gene_id'] = [clean_id(n.split(':')[0].split(';')[0])
                     for n in df['Protein IDs']]
    df['desc'] = df['Fasta headers'].apply(make_desc, lookfor=desc_value)
    return df
   

#create a dictionary id -> description
#from a trytripdb fasta file
def make_desc_dict(path_to_file):
    desc_dict = {}
    with open(path_to_file, "r") as handle:
        a=0
        for record in SeqIO.parse(handle, "fasta"):
            a+=1
            temp_id = clean_id(record.id).strip()
            temp_desc = record.description.split('|')[4].strip()
            desc_dict[temp_id]=temp_desc
    return desc_dict



                           
#make pca plot from pandas df
def make_pca(in_df, palette, ax, top=500, 
             color_dictionary=False, do_adjust_text=False):
    
    cols = in_df.columns
    
    
    sorted_mean = in_df.mean(axis=1).sort_values()
    select = sorted_mean.tail(top)
    #print(top)
    in_df = in_df.loc[select.index.values]
    
    pca = PCA(n_components=2)
    pca.fit(in_df)
    
    temp_df = pd.DataFrame()
    temp_df['pc_1']=pca.components_[0]
    temp_df['pc_2']=pca.components_[1]
    temp_df.index = cols
    print(pca.explained_variance_ratio_)
    temp_df['color']=palette
    #fig,ax=plt.subplots(figsize=(12,6))
    temp_df.plot(kind='scatter',x='pc_1', y='pc_2',s=30, c=temp_df['color'], ax=ax)
    #print(temp_df.index.values)

    for color in temp_df['color'].unique():
        c_data = temp_df[temp_df['color']==color].iloc[0]
        ax.scatter(x=c_data.pc_1, y=c_data.pc_2, c=color, label=color,s=30)
        ax.legend(title='Groups',loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    
    texts = [ax.text(temp_df.iloc[i]['pc_1'], 
                       temp_df.iloc[i]['pc_2'],
                       cols[i])
                       for i in range(temp_df.shape[0])]
    
    if do_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'),ax=ax)
    ax.set_title('PCA', size=14)
    ax.set_xlabel('PC1_{:.3f}'.format(pca.explained_variance_ratio_[0]),size=12)
    ax.set_ylabel('PC2_{:.3f}'.format(pca.explained_variance_ratio_[1]),size=12)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)
    
    if color_dictionary:
        print(color_dictionary)
        handles, labels = ax.get_legend_handles_labels()
        labels = [ color_dictionary[l] for l in labels]
        ax.legend(handles=handles, labels=labels, 
        title='Groups',loc='center left', bbox_to_anchor=(1, 0.9))    
    return ax
#make mds plot from pandas df  
def make_mds(in_df, palette, ax, top=500, 
             color_dictionary=False,do_adjust_text=True):
    cols = in_df.columns
    
    
    
    sorted_mean = in_df.mean(axis=1).sort_values()
    select = sorted_mean.tail(top)
    #print(top)
    in_df = in_df.loc[select.index.values]
    
    pca = MDS(n_components=2, metric=True)
    temp_df = pd.DataFrame(pca.fit_transform(in_df.T),
                                 index=cols, 
                           columns =['pc_1', 'pc_2'] )
    
    temp_df['color']=palette
    
    temp_df.plot(kind='scatter',x='pc_1', y='pc_2', s=50, c=temp_df['color'], ax=ax)
    #print(temp_df.head())
    for color in temp_df['color'].unique():
        c_data = temp_df[temp_df['color']==color].iloc[0]
        ax.scatter(x=c_data.pc_1, y=c_data.pc_2, c=color, label=color,s=50)
        ax.legend(title='Groups',loc='center left', bbox_to_anchor=(1, 0.5))
      
    
    #.plot(kind='scatter',x='pc_1', y='pc_2', s=50, c=temp_df['color'], ax=ax)
    
    
    
    
    texts = [ax.text(temp_df.iloc[i]['pc_1'], 
                       temp_df.iloc[i]['pc_2'],
                       cols[i])
                       for i in range(temp_df.shape[0])]
    if do_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'),ax=ax)
    ax.set_title('MDS',size=14)
    ax.set_xlabel('DIM_1',size=12)
    ax.set_ylabel('DIM_2',size=12)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)
    if color_dictionary:
        print(color_dictionary)
        handles, labels = ax.get_legend_handles_labels()
        labels = [ color_dictionary[l] for l in labels]
        ax.legend(handles=handles, labels=labels, 
        title='Groups',loc='center left', bbox_to_anchor=(1, 0.9))
    return ax


#format legend of hist plots 
#with lines instead of boxes
def hist_legend(ax, title = False):
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    ax.legend(handles=new_handles, labels=labels, 
    title=title,loc='center left', bbox_to_anchor=(1, 0.5))  


#get a random distribution of numbers 
#around the minimum value 
#of a columns (greather than zero)
#with small std
def get_random(in_col, strategy):
    if strategy == 'small':
        mean_random = in_col[in_col>0].min()
        std_random = mean_random*0.25
        random_values = np.random.normal(mean_random, 
                                         scale=std_random, 
                                         size=in_col.shape[0])
    if strategy == 'median':
        pass
        
    return  random_values

#add a small random value to each element
#of a cloumn, optionally plots the distribution
#of the random values
def impute(in_col, ax=False, strategy='small'):
    random_values = get_random(in_col, strategy=strategy)
    if ax:
        np.log10(pd.Series(random_values)).plot(kind='hist',histtype='step', 
                          density=True,ax=ax,label=in_col.name)  
    
    fake_col = in_col.copy()
    fake_col = fake_col+random_values
    index = in_col[in_col==0].index.values 
    in_col.loc[index] = fake_col.loc[index] 
    return in_col    

#replace missing values with zeros
def replace_nan(col):
    col = col.replace('NaN', np.nan)
    col = col.fillna(0)
    return col

#normalization of dataframe
#to account for uneven loading
def norm_loading_TMT(df):
    col_sum = df.sum(axis=0)
    print(col_sum)
    target = np.mean(col_sum)
    print(target)
    norm_facs = target / col_sum
    print(norm_facs)
    data_norm = df.multiply(norm_facs, axis=1)
    return  data_norm

def norm_loading(df):
    col_sum = df.median(axis=0)
    print(col_sum)
    target = np.mean(col_sum)
    print(target)
    norm_facs = target / col_sum
    print(norm_facs)
    data_norm = df.multiply(norm_facs, axis=1)
    return data_norm

#essentially a scatter plot with the option 
#of annootating group of genes 
def make_vulcano(df, ax, x='-Log10PValue', 
                 y='Log2FC',
                 fc_col = 'Log2FC',
                 fc_limit=False,
                 
                 pval_col = 'PValue',
                 pval_limit=False,
                 
                 annot_index=pd.Series(), 
                 annot_names=pd.Series(),
                 title='Volcano',
                 legend_title='',
                 
                 label_for_selection = None,
                 label_for_all = None,
                 add_text = True,
                 do_adjust_text=True,
                 text_size = 8,
                 rolling_mean = False,
                 alpha_main=0.05,
                point_size_selection=1,
                point_size_all=1,
                fontdict=None,
                expand_text=None,
                force_text=None,
                expand_points=None,
                change_color=None,
                color_main = 'b'):
    

    if fc_limit and pval_limit:
        upper = df[df[fc_col]>fc_limit].copy()
        lower = df[df[fc_col]<-fc_limit].copy()
        
        upper = upper[upper[pval_col]<pval_limit]
        lower = lower[lower[pval_col]<pval_limit]
         
    elif pval_limit:
        upper = df[df[pval_col]<pval_limit].copy()
        lower = df[df[pval_col]<pval_limit].copy()
        
    elif fc_limit:
        upper = df[df[fc_col]>fc_limit].copy()
        lower = df[df[fc_col]<-fc_limit].copy()

    else: 
        print('no selection')    
            
    
    to_remove = []
    if 'upper' in locals() and upper.shape[0]>0:
        #print(upper.head())
        upper.plot(
        kind='scatter',x=x,y=y, ax=ax, 
        c='r', label='Bigger Than {fc_limit}'.format(fc_limit=fc_limit), alpha=0.5, zorder=5)
        to_remove.append(upper)
        
    if 'lower' in locals() and lower.shape[0]>0:     
        lower.plot(
        kind='scatter',x=x,y=y, ax=ax, 
        c='g', label='Lower Than {fc_limit}'.format(fc_limit=fc_limit), alpha=0.5, zorder=5)       
        to_remove.append(lower)


    if len(annot_index) > 0:
        df.loc[annot_index].plot(kind='scatter', x=x, y=y, c='r', 
                                 s=point_size_selection, ax=ax, 
                                 label=label_for_selection, alpha=1, zorder=10)
        to_remove.append(df.loc[annot_index])
        
    if change_color:
        df.loc[change_color[0]].plot(kind='scatter', x=x, y=y, c=change_color[1], 
                                 s=change_color[2], ax=ax, 
                                 label=label_for_selection, alpha=1, zorder=10)
        to_remove.append(df.loc[change_color[0]])        
                

 

    
    if len(to_remove)>0:
        to_remove=pd.concat(to_remove)
        idx = df.index.difference(to_remove.index)
        df.loc[idx].plot(kind='scatter', x=x, y=y, ax=ax, 
                         alpha=alpha_main, c=color_main, zorder=1, label=label_for_all,
                        s=point_size_all)
    else:
        df.plot(kind='scatter', x=x, y=y, ax=ax, 
                alpha=alpha_main,c=color_main, zorder=1,label=label_for_all,s=point_size_all)
    
    if rolling_mean:
        df = df.sort_values(x,ascending=False)
        df['rolling_mean'] = df[y].rolling(100).mean()
        print(df.head())
        temp = df[['rolling_mean',x]]
        temp=temp.dropna()
        temp.plot(ax=ax, x=x, y='rolling_mean', label = 'rolling mean', c='r',alpha=0.3)
    
        ax.set_xlim(df[x].min()-df[x].min()*0.01,
                 df[x].max()+df[x].min()*0.01)

    
    if add_text:
        texts = [ax.text(df.loc[i][x], df.loc[i][y],name, fontsize=text_size,fontdict=fontdict)
                               for i,name in zip(annot_index,annot_names)]
        #print(texts)
        if do_adjust_text:
            #print('adjusting text')
            if not expand_text:
                expand_text=(1.1, 1.1)
            if not force_text:
                force_text=(0.1, 0.2)
            if not expand_points:
                expand_points=(1.05, 1.2)

            adjust_text(texts, arrowprops=dict(arrowstyle='-',
                                               color='red',lw=0.8),
                                               force_text=force_text,
                                               va='bottom',
                                               lim=1000,
                                               expand_text=expand_text,
                                               autoalign='xy',
                                               #only_move={'points':'x', 'text':'x'},
                        ax=ax)

    
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, 0.8), title=legend_title)
    
    ax.set_title(title)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)
    return ax



#helper function to visualize the correlation between experiments
def plot_correlation(df, figname='corr_prot'):
    #function to annotate the axes with
    #the pearson correlation coefficent
    def corrfunc(x, y, **kws):
        corr = np.corrcoef(x, y)
        r = corr[0][1]
        ax = plt.gca()
        ax.annotate("p = {:.2f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)
    
    #prepare the seaborn grid and plot
    g = sns.PairGrid(df.dropna(), palette=["red"], height=1.8, aspect=1.5)
    g.map_upper(plt.scatter, s=5)
    g.map_diag(sns.distplot, kde=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(corrfunc)
    sns.set(font_scale=1.1)
    
    
    
def concordance_correlation_coefficient(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.  
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    >>> from sklearn.metrics import concordance_correlation_coefficient
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator
    
    
    
    
class IRS():
    """
     Internal Reference Scaling for multibach TMT
     
    cols = ['Reporter intensity corrected {}'.format(n) for n in range(0,10)]
    experiments = ['E5014','E5015','E5016']
    data=df[[b + ' '+ a for a in experiments for b in cols ]]
    data.columns  = [str(b) + '_'+ a for a in experiments for b in range(1,11)]
    dataIRS =IRS(data=data,
                experiments=experiments,
                chaneels = range(1,11)) 
     dataIRS.norm_loading()
     dataIRS.norm_irs()
    """
    def __init__(
        self,
        data=pd.DataFrame(),
        experiments=[],
        chaneels = []
    ):
        self.data= data
        self.experiments=experiments
        self.chaneels = []
        self.columns = []
        for e in experiments:
            temp = []
            for c in chaneels:
                temp.append('{c}_{e}'.format(c=c, e=e))
            self.columns.append(temp)
                    
    def norm_loading(self):
        data = self.data.copy()
        sum_of_columns = []
        for cols in self.columns:
            #sum of columns for each experiments
            sum_of_columns.append(data[cols].sum(axis=0))
        target = np.mean(sum_of_columns)
        norm_factors = [target / n for n in sum_of_columns]
        
        for cols, nf in zip(self.columns, norm_factors):
            data[cols]=data[cols].multiply(nf, axis=1) 
        self.data_nl = data
        
    def norm_irs(self):
        data = self.data_nl.copy()
        irs = []
        for exp, cols in zip(self.experiments, self.columns):
            temp = data[cols].sum(axis=1)
            temp.name=exp                  
            irs.append(temp)
        irs=pd.concat(irs,axis=1)
        #geometric mean of the sum intensity of all the proteins
        irs['average']=np.exp(np.log(irs.replace(0,np.nan)).mean(axis=1))#, skipna=True))
        print(irs.head())    
        
        norm_factors = []
        for exp in self.experiments:
            norm_factors.append(irs['average'] / irs[exp])
        
        for cols, nf in zip(self.columns, norm_factors):
            data[cols] = data[cols].multiply(nf, axis=0)
            
        self.data_irs = data
        #print(data.head()) 

class CV():
    '''
    cols = ['Reporter intensity corrected {}'.format(n) for n in range(0,10)]
    experiments = ['E5014','E5015','E5016']
    data=df[[b + ' '+ a for a in experiments for b in cols ]]
    data.columns  = [str(b) + '_'+ a for a in experiments for b in range(1,11)]

    groups = {}
    colors = {}
    for n in range(1,11):
        temp = []
        for exp in experiments:
            temp.append('{n}_{exp}'.format(n=n,exp=exp))
        groups[n]=temp
        colors[n]='b'
    {1: ['1_E5014', '1_E5015', '1_E5016'],
     2: ['2_E5014', '2_E5015', '2_E5016'],
     3: ['3_E5014', '3_E5015', '3_E5016'],
     4: ['4_E5014', '4_E5015', '4_E5016'],
     5: ['5_E5014', '5_E5015', '5_E5016'],
     6: ['6_E5014', '6_E5015', '6_E5016'],
     7: ['7_E5014', '7_E5015', '7_E5016'],
     8: ['8_E5014', '8_E5015', '8_E5016'],
     9: ['9_E5014', '9_E5015', '9_E5016'],
     10: ['10_E5014', '10_E5015', '10_E5016']}
    '''
    def __init__(
        self,
        data,
        groups = {},
        
    ):
            self.data = data
            self.groups = groups
            
    
    def compute(self):
        data = self.data.copy()
        cv_means = []
        cv_stds = []
        cvs = []
        groups = self.groups
        for group in groups:
            #print(group,groups[group])
            #if group == 1:
               #print(data[groups[group]].head())
            temp = data[groups[group]].replace(0,np.nan).mean(axis=1, skipna=True)
            cv_means.append(temp)
            #print(temp)
            temp = data[groups[group]].replace(0,np.nan).std(axis=1, skipna=True)
            cv_stds.append(temp)

        for std,mean, group in zip(cv_stds, cv_means, groups):
            temp = std/mean
            temp.name=group
            cvs.append(temp)
        
        cvs = pd.concat(cvs, axis=1)
        self.cv = cvs
        #print(cvs.head())          