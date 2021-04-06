import pickle
import os
from data_processing import FamilyData, NonElderlyData, rescale_bins
from models import ForecastModel

family_model_config = {
    'tag': 'fam',
    'input_dim': 4,
    'alpha': 1e-3,
    'beta': 1e-5
}

NE_people_model_config = {
    'tag': 'pp',
    'input_dim': 12,
    'alpha': 1e-3,
    'beta': 1e-5
}

F_model = ForecastModel(**family_model_config)
NP_model = ForecastModel(**NE_people_model_config)


def train():
    F_model.train_model(FamilyData)
    NP_model.train_model(NonElderlyData)


def forecast(model, tag: str, ahead: int):
    """
    start: density of 2006
    """
    if not os.path.exists('results'):
        os.mkdir('results')

    res = []
    start = None
    if tag == 'fam':
        start = FamilyData.dataset[0][0]
    if tag == 'pp':
        start = NonElderlyData.dataset[0][0]
    pred_dens, pred_county, pred_total = model.forecast(start, ahead)
    for i_, p_ in enumerate(pred_county[6:]):  #
        pd_ = rescale_bins(p_, tag=tag)
        pd_.to_csv(f'results/{tag}_{2013 + i_}.csv')
        res.append(pd_)


    with open(f'results/{tag}_pred_dens.pickle', 'wb') as file:
        pickle.dump(pred_dens, file)

    with open(f'results/{tag}_pred_total.pickle', 'wb') as file:
        pickle.dump(pred_total, file)

    return pred_dens, res, pred_total


if __name__ == '__main__':
    train()
    F_model.save_model()
    NP_model.save_model()
    forecast(F_model, tag='fam', ahead=19)
    forecast(NP_model, tag='pp', ahead=19)
