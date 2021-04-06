

import torch
import pickle
from abc import ABC
import pandas as pd
from torch.utils.data import Dataset, DataLoader

d_family = [f'cleaned_data/data_family_{str(_)}.csv' for _ in range(2006, 2013)]
d_people = [f'cleaned_data/data_people_{str(_)}.csv' for _ in range(2006, 2013)]

with open('cleaned_data/county_list.pickle', 'rb') as file:
    County_list = pickle.load(file)

adults_prop = pd.read_csv('cleaned_data/household_adults_prop.csv')['percentage'].to_numpy()


def process_data(data_ids, tag=''):
    res_total, res_density, res_county = [], [], []
    for file_ in data_ids:
        data_ = pd.read_csv(file_)
        total_ = data_.loc[0][1:].sum()
        res_total.append(total_/10000)
        res_density.append(torch.FloatTensor((data_.iloc[0, 1:] / total_).to_list()))
        county_ = data_.drop(0, axis=0).drop('county', axis=1).to_numpy().reshape(-1)
        res_county.append(torch.FloatTensor(county_)/10000)

    with open(f'cleaned_data/{tag}_total.pickle', 'wb') as file:
        pickle.dump(res_total, file)

    with open(f'cleaned_data/{tag}_density.pickle', 'wb') as file:
        pickle.dump(res_density, file)

    with open(f'cleaned_data/{tag}_county.pickle', 'wb') as file:
        pickle.dump(res_county, file)

    return res_total, res_density, res_county


def rescale_bins(county_count, tag=''):
    """
    county_density is a torch.tensor of shape (county_size, col)
    :return the dataframe wite
    """
    county_count = county_count.view(159, -1)
    res = torch.zeros((159, 2))
    cols = ['0', '1']
    if tag == 'fam':  # <0.36 and <1.38
        cols = ['<0.36', '<1.38']
        res[:, 0] = 0.36 / 1.3 * county_count[:, 0]
        res[:, 1] = county_count[:, 0] + 0.8 * county_count[:, 1]

    if tag == 'pp':  # 0.37-1.00 and 1.00-4.00
        cols = ['1.00-1.38', '1.00-4.00']
        res[:, 0] = (county_count[:, 3] + 0.13/0.24*county_count[:, 4])*adults_prop
        res[:, 1] = county_count[:, 3:10].sum(dim=1)
    # here it should be converted to counts
    res = pd.DataFrame(res.numpy()*10000, columns=cols, index=County_list).astype(int)

    return res


class MyData(Dataset, ABC):
    def __init__(self, tag=''):
        with open(f'cleaned_data/{tag}_total.pickle', 'rb') as file:
            self.total = pickle.load(file)

        with open(f'cleaned_data/{tag}_density.pickle', 'rb') as file:
            self.density = pickle.load(file)

        with open(f'cleaned_data/{tag}_county.pickle', 'rb') as file:
            self.county = pickle.load(file)

    def __len__(self):
        return len(self.total)-1

    def __getitem__(self, index):
        """
        :return current density, next density, next total, next county
        """
        return self.density[index], self.density[index+1], self.total[index+1], self.county[index+1]


process_data(d_family, 'fam')
process_data(d_people, 'pp')

FamilyData = DataLoader(MyData('fam'), batch_size=1, shuffle=False)
NonElderlyData = DataLoader(MyData('pp'), batch_size=1, shuffle=False)


















