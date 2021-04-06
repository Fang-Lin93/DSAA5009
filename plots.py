import pickle
import seaborn as sns
import pandas as pd
import torch
from matplotlib import pyplot as plt
from data_processing import FamilyData, NonElderlyData
from models import ForecastModel

years = list(range(2006, 2026))

def plot_figures():
    """
    Forecasting of the total population: Fam & pp
    """
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    fam_truth = FamilyData.dataset.total
    pp_truth = NonElderlyData.dataset.total

    with open(f'results/fam_pred_total.pickle', 'rb') as file:
        fam_pred = pickle.load(file)
    with open(f'results/pp_pred_total.pickle', 'rb') as file:
        pp_pred = pickle.load(file)

    fam_pred = [fam_truth[0]] + [round(i, 3) for i in fam_pred]
    pp_pred = [pp_truth[0]] + [round(i, 3) for i in pp_pred]

    sns.lineplot(x=years[:len(fam_truth)], y=fam_truth, color='b', ax=ax[0], label='Truth')
    sns.lineplot(x=years[:len(fam_pred)], y=fam_pred, color='g', ax=ax[0], label='Prediction')
    sns.lineplot(x=years[:len(pp_truth)], y=pp_truth, color='b', ax=ax[1], label='Truth')
    sns.lineplot(x=years[:len(pp_pred)], y=pp_pred, color='g', ax=ax[1], label='Prediction')
    ax[0].set_xlim(2005, 2026)
    ax[0].set_xticks(range(2005, 2026))
    ax[1].set_xlim(2005, 2026)
    ax[1].set_xticks(range(2005, 2026))
    ax[0].set_ylabel('Population (10K)')
    ax[1].set_ylabel('Population (10K)')
    ax[0].set_title('Non-elderly families with children < 18')
    ax[1].set_title('Non-elderly people')

    fig.show()

    """
    Forecasting of the total density
    """

    with open(f'results/fam_pred_dens.pickle', 'rb') as file:
        pred_fam_dens = pickle.load(file)
    with open(f'results/pp_pred_dens.pickle', 'rb') as file:
        pred_pp_dens = pickle.load(file)
    with open(f'cleaned_data/fam_density.pickle', 'rb') as file:
        true_fam_dens = pickle.load(file)
    with open(f'cleaned_data/pp_density.pickle', 'rb') as file:
        true_pp_dens = pickle.load(file)

    fam_x = ['<1.30', '1.30-1.49', '1.50-1.84', '>1.85']
    pp_x = ['<0.5', '0.5-0.74', '0.75-0.99', '1.00-1.24', '1.25-1.49',
            '1.50-1.74', '1.75-1.84', '1.85-1.99', '2.00-2.99', '3.00-3.99',
            '4.00-4.99', '>5.00']
    pred_fam_dens = [true_fam_dens[0]] + pred_fam_dens
    pred_pp_dens = [true_pp_dens[0]] + pred_pp_dens
    #  Family
    fig, ax = plt.subplots(4, 5, figsize=(24, 16))
    for i in range(len(pred_fam_dens)):
        ax_ = ax[i // 5][i % 5]
        sns.lineplot(x=fam_x, y=pred_fam_dens[i], ax=ax_, color='g', label='Prediction')
        for x, y in zip(fam_x, pred_fam_dens[i]):
            ax_.annotate(xy=(x, y + 0.02), text=round(y.item(), 3), color='g')
        if i < len(true_fam_dens):
            sns.lineplot(x=fam_x, y=true_fam_dens[i], ax=ax_, color='b', label='Truth')
            for x, y in zip(fam_x, true_fam_dens[i]):
                ax_.annotate(xy=(x, y - 0.02), text=round(y.item(), 3), color='b')
        ax_.set_xlabel(f'{2006 + i}')

    fig.suptitle('PL density of non-elderly families with children < 18', fontsize=36)
    fig.show()

    #  None-Elderly people
    fig, ax = plt.subplots(4, 5, figsize=(24, 16))

    for i in range(len(pred_pp_dens)):
        ax_ = ax[i // 5][i % 5]
        sns.lineplot(x=pp_x, y=pred_pp_dens[i], ax=ax_, color='g', label='Prediction')
        for x, y in zip(pp_x, pred_pp_dens[i]):
            ax_.annotate(xy=(x, y + 0.02), text=round(y.item(), 3), color='g', fontsize=6)
        if i < len(true_pp_dens):
            sns.lineplot(x=pp_x, y=true_pp_dens[i], ax=ax_, color='b', label='Truth')
            for x, y in zip(pp_x, true_pp_dens[i]):
                ax_.annotate(xy=(x, y - 0.02), text=round(y.item(), 3), color='b', fontsize=6)
        ax_.set_xlabel(f'{2006 + i}')
        ax_.set_xticks(ax_.get_xticks())
        ax_.set_xticklabels(['<0.5', ' ', ' ', ' ', '1.25-1.49',
                             ' ', ' ', ' ', '2.00-2.99', ' ',
                             ' ', '>5.00'])

    fig.suptitle('PL densities of Non-elderly people', fontsize=36)
    fig.show()

    """
    Compare expansion v.s. non-expansion, 2013, 2016, 2019, 2022, 2025
    """
    c_yrs = [2013, 2016, 2019, 2022, 2025]
    # Eligibility for Medicaid
    fig, ax = plt.subplots(5, 1, figsize=(24, 16))
    c_ids = None
    for i_, y_ in enumerate(c_yrs):
        df = pd.read_csv(f'results/fam_{y_}.csv', index_col=0)
        if i_ == 0:
            df = df.sort_values(by='<0.36', ascending=False)
            c_ids = df.index.to_list()
        else:
            df = df.loc[c_ids]
        sns.lineplot(x=df.index.to_list(), y=df['<0.36'], ax=ax[i_], label='Non-Expansion')
        sns.barplot(x=df.index.to_list(), y=df['<1.38'], ax=ax[i_], label='Expansion')
        if i_ < len(c_yrs) - 1:
            ax[i_].set_xticklabels([])
        else:
            ax[i_].set_xticklabels(ax[i_].get_xticklabels(), rotation=90)
        ax[i_].legend()
        ax[i_].set_xlabel(y_)
        ax[i_].set_ylabel('Number of people')
    fig.suptitle('Eligibility for Medicaid forecasting', fontsize=32)
    fig.show()

    # Subsidizing health insurances
    fig, ax = plt.subplots(5, 1, figsize=(24, 16))
    c_ids = None
    for i_, y_ in enumerate(c_yrs):
        df = pd.read_csv(f'results/pp_{y_}.csv', index_col=0)
        if i_ == 0:
            df = df.sort_values(by='1.00-4.00', ascending=False)
            c_ids = df.index.to_list()
        else:
            df = df.loc[c_ids]
        sns.barplot(x=df.index.to_list(), y=df['1.00-4.00'], ax=ax[i_], label='Non-Expansion/Expansion')
        if i_ < len(c_yrs) - 1:
            ax[i_].set_xticklabels([])
        else:
            ax[i_].set_xticklabels(ax[i_].get_xticklabels(), rotation=90)
        ax[i_].legend()
        ax[i_].set_xlabel(y_)
        ax[i_].set_ylabel('Number of people')
    fig.suptitle('Subsidizing health insurances forecasting', fontsize=32)
    fig.show()

    # Both Eligibility for Medicaid & Subsidizing health insurances
    fig, ax = plt.subplots(5, 1, figsize=(24, 16))
    c_ids = None
    for i_, y_ in enumerate(c_yrs):
        df = pd.read_csv(f'results/pp_{y_}.csv', index_col=0)
        if i_ == 0:
            df = df.sort_values(by='1.00-1.38', ascending=False)
            c_ids = df.index.to_list()
        else:
            df = df.loc[c_ids]
        sns.barplot(x=df.index.to_list(), y=df['1.00-1.38'], ax=ax[i_], label='Expansion')
        if i_ < len(c_yrs) - 1:
            ax[i_].set_xticklabels([])
        else:
            ax[i_].set_xticklabels(ax[i_].get_xticklabels(), rotation=90)
        ax[i_].legend()
        ax[i_].set_xlabel(y_)
        ax[i_].set_ylabel('Number of people')
    fig.suptitle('Eligible for both Medicaid and the subsidizing health insurances forecasting', fontsize=32)
    fig.show()

    """
    Low rank check
    """
    family_model_config = {
        'tag': 'fam',
        'input_dim': 4,
        'alpha': 1e-3,
        'beta': 1e-5
    }
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    #  without training
    F_model = ForecastModel(**family_model_config)
    para = list(F_model.decoder.parameters())[2]
    eig, _ = torch.eig(para.transpose(1, 0).matmul(para))
    eig = [i[0].item() for i in eig]
    eig.sort(reverse=True)
    sns.scatterplot(x=range(len(eig)), y=eig, s=2, ax=ax[0])

    #  after training
    F_model = ForecastModel(**family_model_config)
    F_model.load_model()
    para = list(F_model.decoder.parameters())[2]
    eig, _ = torch.eig(para.transpose(1, 0).matmul(para))
    eig = [i[0].item() for i in eig]
    eig.sort(reverse=True)
    sns.scatterplot(x=range(len(eig)), y=eig, s=10, ax=ax[1])
    ax[0].set_xlabel('Without training')
    ax[1].set_xlabel('After training')
    ax[0].set_ylabel('Eigenvalues')
    ax[1].set_ylabel('Eigenvalues')

    fig.suptitle('Low rank check of NN', fontsize=32)
    fig.show()


if __name__ == '__main__':
    plot_figures()















