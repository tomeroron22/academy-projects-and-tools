import numpy as np
from scipy.stats import chi2
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy as sp
from scipy.stats import binom
from plotly import graph_objects as go


#### General parameters and calculations according to model ####
gene_probs = np.array([
                [[1,0,0],      [0.5,0.5,0],      [0,1,0]],     #DD
                [[0.5,0.5,0],  [0.25,0.5,0.25],  [0,0.5,0.5]], #DC
                [[0,1,0],      [0,0.5,0.5],      [0,0,1]]])    #CC

def calc_c(pl, pldc):
    """
    Calculate the incidence of allele C based on the given parameters.

    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :return: p(C) - The incidence of allele C.
    """
    if pldc == 0.25:
        return 2*pl
    
    return (-2 * pldc + 2 * np.sqrt(pldc **2 +(0.5 - 2*pldc)*pl))/(2*(0.5 -2*pldc)) 

def p_gen_phen(C, pl, pldc, gen, H):
    """
    Calculate the conditional probability of having the genotype given the phenotype.

    :param C: p(C) - The incidence of allele C.
    :type C: float
    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param gen: The genotype: 0=DD, 1=DC, 2=CC.
    :type gen: int
    :param H: The phenotype: 0=right-handed, 1=left-handed.
    :type H: int
    :return: p(G|H) - The conditional probability of having the genotype given the phenotype.
    :rtype: float
    """

    if gen == 0:
        if H == 0:
            return (1-C)**2/(1-pl)
        return 0
        
    if gen == 2:
        if H == 0:
            return C**2       * 0.5     /(1-pl)
        return C**2       * 0.5     /pl

    if gen == 1:
        if H == 0:
            return 2*(C-C**2) * (1-pldc)/(1-pl)

        return 2*(C-C**2) * pldc / pl

#### Meaure parms from data ####
def calc_measured_fam(data):
    """
    Estimate the probability p(L_m) for progeny and parents based on the given dataset.
    :param data: dataset to estimate p(L_m) from.
    :return: A tuple containing two probabilities:
             - p(L_m) for progeny (the manifest proportion of left-handers in the progeny generation).
             - p(L_m) for parents (the manifest proportion of left-handers in the parents generation).
    :rtype: tuple[float, float]
    """
    l_child = 0
    child_cnt = 0

    l_par = 0
    par_count = 0

    cols = len(data[0])/3
    
    for size,row in enumerate(data):

        child_cnt += np.nansum(row) * (1+ size)
        par_count += np.nansum(row) * 2
    
        for col, cell in enumerate(row):
            count = np.nansum(cell)
            
            l_par += count * (col//cols)
            
            l_child += count * (col % cols)  
    return l_child/child_cnt, l_par/par_count

def calc_measured_twin(data):
    """
    Estimate the probability p(L_m) for progeny based on the given dataset.
    :param data: The dataset used for estimating p(L_m).Should be a 2D array-like structure.
    :type data: ArrayLike
    :return: The estimate of p(L_m) for the twins' dataset.
    """
    cnt = np.nansum(data) *2
    l = data[1] + 2 * data[2]
    return l/cnt


#### Family data probs ####

def p_left(pldc,g1,g2):
    """
    Calculate the conditional probability of being left-handed given the parental genotypes.

    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param g1: The genotype of parent 1: 0=DD, 1=DC, 2=CC.
    :type gen: int
    :param g2: The genotype of parent 2: 0=DD, 1=DC, 2=CC.
    :type gen: int
    :return: p(L|G1xG2) - The conditional probability of being left-hander given the parental genotypes.
    :rtype: float
    """
    pl_gen = np.array([0,pldc,0.5])
    return np.sum(pl_gen * gene_probs[g1,g2])

def fam_prob(pldc,g1,g2, size):
    """
    Calculate the conditional probability of having n left-handed children in a family given the parents genotype.

    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param g1: The genotype of parent 1: 0=DD, 1=DC, 2=CC.
    :type gen: int
    :param g2: The genotype of parent 2: 0=DD, 1=DC, 2=CC.
    :type gen: int
    :param size: The number of children in the family
    :type size: int
    :return: An array of p(n|G1xG2) - The conditional probability of having n children presenting left-handedness given the parents genotypes.
    :rtype: float
    """
    pl_fam = p_left(pldc,g1,g2)
    return np.array([binom.pmf(i, size, pl_fam) for i in range(size+1)])

def family_probs(C,pl, pldc, size):
    """
    Calculate the list of binomial probabilities for a given family size and parameters.

    :param C: p(C) - The incidence of allele C.
    :type C: float
    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param size: The family size.
    :type size: int
    :return: An array of binomial probabilities where n = family size and success = number of left-handed children.
    :rtype: numpy.ndarray
    """
    probs = np.zeros((size+1)*3)
    for g1 in range(3):
        for g2 in range(3):
            fam_gene_prob = fam_prob(pldc,g1,g2,size) # Array 
            p_g1_r = p_gen_phen(C,pl,pldc,g1,0) # P(G1|Right)
            p_g1_l = p_gen_phen(C,pl,pldc,g1,1) # P(G1|Left)
            p_g2_r = p_gen_phen(C,pl,pldc,g2,0) # P(G2|Right)
            p_g2_l = p_gen_phen(C,pl,pldc,g2,1) # P(G2|Left)
            
            probs[:size+1] +=  fam_gene_prob * p_g1_r* p_g2_r   # RxR parents
            probs[1+size:2*size+2] += fam_gene_prob * p_g1_r* p_g2_l # RxL parents
            probs[-size-1:] += fam_gene_prob * p_g1_l * p_g2_l # LxL parents
    return probs

#### Twins data probs ####

def mono_twins_phen_parental_gen(pldc, pl, phen1, phen2, gen1, gen2):
    """
    Calculate the probability of having MZ twins based on parents' genotype.

    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :param phen1: Phenotype of twin 1: 0=right-handed, 1=left-handed.
    :type phen1: int
    :param phen2: Phenotype of twin 2: 0=right-handed, 1=left-handed.
    :type phen2: int
    :param gen1: Genotype of parent 1: 0=DD, 1=DC, 2=CC.
    :type gen1: int
    :param gen2: Genotype of parent 2: 0=DD, 1=DC, 2=CC.
    :type gen2: int
    :return: p(H1xH2|G1xG2) - The probability of having monozygotic twins presenting phenotypes (H1xH2) given the parents genotypes.
    :rtype: float
    """
    # Probability matrix for phenotype based on genotype

    phen_probs =np.array([
#                  R          L
                  [1,         0], # DD
                  [1-pldc, pldc], # DC
                  [0.5,     0.5]  # CC
        ])
    return np.nansum(gene_probs[gen1,gen2, :] * phen_probs[:,phen1] * phen_probs[:,phen2])

def di_twins_phen_parental_gen(pldc, pl, phen1, phen2, gen1, gen2):
    """
    Calculate the probability of having dizygotic twins based on parents' genotype.

    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :param phen1: Phenotype of twin 1: 0=right-handed, 1=left-handed.
    :type phen1: int
    :param phen2: Phenotype of twin 2: 0=right-handed, 1=left-handed.
    :type phen2: int
    :param gen1: Genotype of parent 1: 0=DD, 1=DC, 2=CC.
    :type gen1: int
    :param gen2: Genotype of parent 2: 0=DD, 1=DC, 2=CC.
    :type gen2: int
    :return: p(H1xH2|G1xG2) The probability of having dizygotic twins presenting phenotypes (H1xH2) given the parents genotypes.
    :rtype: float
    """
    # Probability matrix for phenotype based on genotype

    phen_probs =np.array([
#                  R          L
                  [1,         0], # DD
                  [1-pldc, pldc], # DC
                  [0.5,     0.5]  # CC
        ])
    return np.sum(gene_probs[gen1,gen2,:] * phen_probs[:,phen1]) * np.sum(gene_probs[gen1,gen2,:]  *phen_probs[:,phen2])
    
def twins_phen_parental_phen(C, pldc, pl, phen1, phen2, parent_phen1, parent_phen2, type):
    """
    Calculate the probability of having twins based on parental phenotypes.

    :param C: p(C) - The incidence of allele C.
    :type C: float
    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :param phen1: The phenotype of the first twin: 0=right-handed, 1=left-handed.
    :type phen1: int
    :param phen2: The phenotype of the second twin: 0=right-handed, 1=left-handed.
    :type phen2: int
    :param parent_phen1: The phenotype of parent 1: 0=right-handed, 1=left-handed.
    :type parent_phen1: int
    :param parent_phen2: The phenotype of parent 2: 0=right-handed, 1=left-handed.
    :type parent_phen2: int
    :param type: type of the twins 1=Mz, 2=DZ
    :type type: int
    :return: The probability of having twins with the given parental phenotypes and twin phenotypes.
    :rtype: float
    """
    prob = 0
    for p1_gen in range(3):

        # p1_phen = prob of phenotype given genotype of parent1
        p1_phen = p_gen_phen(C,pl,pldc,p1_gen, parent_phen1)

        for p2_gen in range(3):

            # p2_phen = prob of phenotype given genotype of parent2
            p2_phen = p_gen_phen(C,pl,pldc,p2_gen, parent_phen2)
            if type == 1:
                p = mono_twins_phen_parental_gen(pldc,pl, phen1,phen2,p1_gen,p2_gen)
            if type == 2:
                p = di_twins_phen_parental_gen(pldc,pl, phen1,phen2,p1_gen,p2_gen)
            prob +=  p * p1_phen * p2_phen
    return prob

def twins_prob(C, pldc, pl, type):
    """
    Calculate the probability of having the given phenotype combination in twins.

    :param C: p(C) - The incidence of allele C.
    :type C: float
    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :param phen1: The phenotype of the first twin: 0=right-handed, 1=left-handed.
    :type phen1: int
    :param phen2: The phenotype of the second twin: 0=right-handed, 1=left-handed.
    :type phen2: int
    :param type: type of the twins 1=Mz, 2=DZ
    :type type: int
    :return: probability of the given phenotype combination in twins
    """
    rr,rl,ll = 0,0,0
    for parent_phen1 in range(2):
        if parent_phen1 == 0:
            phen1_prob = 1-pl
        else:
            phen1_prob = pl
        for parent_phen2 in range(2):
            if parent_phen2 == 0:
                phen2_prob = 1-pl
            else:
                phen2_prob = pl
            rr += twins_phen_parental_phen(C, pldc, pl, 0, 0, parent_phen1, parent_phen2, type) * phen1_prob * phen2_prob
            rl += twins_phen_parental_phen(C, pldc, pl, 0, 1, parent_phen1, parent_phen2, type) * phen1_prob * phen2_prob
            ll += twins_phen_parental_phen(C, pldc, pl, 1, 1, parent_phen1, parent_phen2, type) * phen1_prob * phen2_prob
    return np.array([rr, 2*rl,ll ])


#### Correctin matrices ####

def make_P(true, measured ,size=1):
    """
    Generate a probability matrix based on the true and measured incidences of left-handedness.

    :param true:p(L_t) - The true incidence of left-handedness.
    :type true: float
    :param measured: p(L_m) - The measured incidence of left-handedness .
    :type measured: float
    :param size: number of children in family (default 1)
    :return: A (size+1)x(size+1) probability matrix P
    :rtype: numpy.ndarray
    """

    # u = p(Lm|Rt)
    # v = p(Rm|Lt)
    if true > measured:
        v = 1 - (measured/true) 
        u = 0
    elif true < measured:
        v = 0
        u = (measured - true)/(1 - true)
    else:
        u = v = 0

    mat = np.zeros((size+1,size+1))
    for true_left in range(size+1):
        for meas_left in range(size+1):
            if meas_left < true_left:
                prob = np.power(1-u,size- true_left) * sp.special.comb(true_left, meas_left) * np.power(1-v, meas_left) * np.power(v, true_left-meas_left)

            elif meas_left > true_left:
                prob = sp.special.comb(size - true_left, size - meas_left) * np.power(1-u,size- meas_left) * np.power(u,meas_left-true_left) * np.power(1-v, true_left)
            
            else: 
                prob = np.power(1-u, size - true_left) * np.power(1-v, true_left)
            mat[true_left, meas_left] = prob
            
    return mat

def make_Q(true, measured):
    """
    Construct a 3x3 matrix Q based on the input matrix R.

    :param true: The true incidence of left-handedness (p(L_t)).
    :type true: float
    :param measured: The measured incidence of left-handedness (p(L_m)).
    :type measured: float
    :return: A 3x3 probability matrix Q.
    :rtype: numpy.ndarray
    """

    # v = p(Rt/Rm),   u = p(Rt/Lm)
    if true > measured:
        v = (true - measured)/(1 - measured)
        u = 0
    else:
        v = 0
        u = 1 - true / measured

    return np.array([
        [np.power(1-v,2), 2*v*(1-v),np.power(v,2)   ],
        [        u      ,   1-v-u  ,         v      ],
        [np.power(u,2)  , 2*u*(1-u), np.power(1-u,2)]
    ])

def make_T(pl, pldc, type, size=1):
    """
    Generate a matrix T based on the model parameters.

    :param pl: p(L_t)-The true incidence of left-handedness.
    :type pl: float
    :param pldc: p(L|DC)-The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param type: The type of matrix: triplets=1 ,monozygotic=2, dizygotic=3.
    :type type: int
    :param size: family size (number of children) default = 1
    :type size: int
    :return: A probability matrix T.
    :rtype: numpy.ndarray
    """
    C = calc_c(pl,pldc)
    if type == 1:
        probs = family_probs(C, pl,pldc,size)
            
    else:
         probs = twins_prob(C, pldc,pl, (type+1)//2)

    return np.reshape(probs, (3,-1))

def make_M(progeny, parents, pl, pldc, type, size=1):
    """
    Generate the matrix M reflecting the conditional probability of manifest handedness.

    :param progeny: p(L_m) progeny-The measured handedness of the progeny.
    :type progeny: float
    :param parents: p(L_m) parental-The measured handedness of the parents if type different from 1 ignored.
    :type parents: float
    :param pl: p(L_t)-The true incidence of left-handedness.
    :type pl: float
    :param pldc: p(L|DC)-The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param type: The type of data: 1 for family data, 2 for monozygotic twins, 3 for dizygotic twins.
    :type type: int
    :param size: number of children in family (used only in family data) 
    :type size: int
    :return: The matrix M reflecting the conditional probability of manifest handedness.
    :rtype: numpy.ndarray
    """
    assert type in {1,2,3}, "illeagal type"
   

    if type != 1:
        P = make_P(pl, progeny, 2)
        Q = np.eye(1)
        T = make_T(pl, pldc, type).reshape(1,-1)

    else:
        P = make_P(pl,progeny, size)
        Q = make_Q(pl, parents)
        T = make_T(pl, pldc, type, size)


    tp = np.matmul(T,P)

    m = np.matmul(Q,tp)

    return m


#### Expected values ####
def calc_exp_table(table, measured, pl, pldc, rows_per_dataset, type):
    """
    Calculate the expected table and expected probabilities based on the observed table, measured manifast, and model parameters.

    :param table: The input table containing the observed handedness data.
    :param measured: The table of p(L_m) estimated from the observed data.
    :param pl: The true incidence of left-handedness.
    :type pl: float
    :param pldc: The probability of manifesting left-handedness in heterozygous.
    :type pldc: float
    :param rows_per_dataset: The number of rows per dataset in the table.
    :type rows_per_dataset: int
    :param type: The type of data: 1 for family data, 2 for monozygotic twins, 3 for dizygotic twins.
    :type type: int
    :return: The expected table and the expected empirical probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    exp = []
    probs = []
    
    for i in range(len(table)//rows_per_dataset):
        dataset = table[i*rows_per_dataset: (i+1)*rows_per_dataset]
        for fam_size,row in enumerate(dataset):
            if type == 1:
                M = make_M(measured[i,0],measured[i,1], pl, pldc, type, fam_size + 1)
            else:
                M = make_M(measured[i],0, pl, pldc, type)   
            e,p = calc_exp_row(M,row,fam_size+1, type)
            exp.append(e)
            probs.append(p)
    return np.vstack(exp), np.vstack(probs)

def calc_exp_row(M, row, fam_size, type):
    """
    Calculate the expected row values and probabilities based on the input parameters.

    :param M: The matrix M containing the conditional probabilities.
    :param row: The input row from the table.
    :param fam_size: The family size.
    :type fam_size: int
    :param type: The type of data: 1 for family data, 2 for monozygotic twins, 3 for dizygotic twins.
    :type type: int
    :return: The expected row values and probabilities.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    cols = fam_size +1
    n = len(row)//3
    prob_row = M.flatten()
    if type == 1:
        prob_row = np.concatenate([M[0], [0]*(n-cols), M[1], [0]*(n-cols), M[2], [0]*(n-cols)])
        exp_row=  np.concatenate( [M[0] * np.nansum(row[:n]),[np.nan]*(n-cols), M[1] * np.nansum(row[n:-n]),[np.nan]*(n-cols), M[2] * np.nansum(row[-n:]), [np.nan]*(n-cols)])

    else:
        exp_row = M.flatten() * np.nansum(row)

    return exp_row, prob_row

#### Likelihood function ####

def log_like(data, probs):
    """
    Calculate the log-likelihood based on the observed data and expected distribution.

    :param data: The observed data.
    :type data: numpy.ndarray
    :param probs: The expected probabilities.
    :type probs: numpy.ndarray
    :return: The log-likelihood value.
    :rtype: float
    """
    
    res = sp.special.xlogy(data, probs)
    return np.nansum(res)

def log_likelihood_model(table_list, measured_tables, rows_in_dataset_list, type_list, pldc, pl):
    """
    Calculate the log-likelihood of the model based on the observed tables and model parameters.

    :param table_list: List of observed tables.
    :param measured_tables: List of p(Lm) from the observed tables.
    :param pldc: p(L|DC) - The probability of manifesting left-handedness in heterozygotes.
    :type pldc: float
    :param pl: p(L_t) - The true incidence of left-handedness.
    :type pl: float
    :return: The log-likelihood of the model.
    :rtype: float
    """
    res =0 
    for i in range(len(table_list)):
        exp_probs = calc_exp_table(table_list[i], measured_tables[i], pl, pldc, rows_in_dataset_list[i], type_list[i])[1]
        res += np.nansum([log_like(table_list[i], exp_probs)])

    return res

#### Grid creation ####


def make_grid(tables, type_list, rows_in_dataset_list, measured_tables):
    """
    Create grid of the models parameters and the support score for each coordinate. 

    :param tables: List of observed tables.
    :type tables: List
    :param type_list: List of the types of each table in tables list (1=familial datasets, 2= MZ twins datasets, 3= Dz twins datasets).
    :type type_list: List
    :param rows_in_dataset_list:  List of the number of rows for each datasets  presented in table that belongs to tables list
    :type rows_in_dataset_list: List
    """
    
    L_DC = np.arange(0, 0.525, 0.025)
    L_T = np.arange(0.02, 0.2025, 0.0025)
    val_lst = np.empty((len(L_DC), len(L_T)))
    max_likelihood = -np.inf
    max_cord = (0.04,0.1)

    for pldc_idx, pldc in enumerate(L_DC):
        for pl_idx,pl in enumerate(L_T):
            cur_val = log_likelihood_model(tables, measured_tables, rows_in_dataset_list, type_list, pldc, pl)
            val_lst[pldc_idx, pl_idx] = cur_val
            if cur_val > max_likelihood:
                max_likelihood = cur_val
                max_cord = (pldc,pl)



    PL, PLDC = np.meshgrid(L_T, L_DC)
    vals = np.array(val_lst).reshape(PL.shape)
    res = opt.minimize(lambda params: -log_likelihood_model(tables, measured_tables, rows_in_dataset_list, type_list, params[0], params[1]), x0=max_cord, bounds=[(0,0.5),(0.02, 0.2)], method='L-BFGS-B')
    return PL,PLDC, vals, res.x, -res.fun

def run_model(tables, type_list, rows_in_dataset_list, measured_tables):
    

    res = opt.minimize(lambda params: -log_likelihood_model(tables, measured_tables, rows_in_dataset_list, type_list, params[0], params[1]), x0=(0.25,0.01), bounds=[(0,0.5),(0.02, 0.2)], method='L-BFGS-B')
    return res.x



