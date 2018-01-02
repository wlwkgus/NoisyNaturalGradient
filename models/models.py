from models.ffg_model import FFGModel
from models.mvg_model import MVGModel


def create_model(opt):
    if opt.model == 'FFGModel':
        opt.option = 'FFG'
        return FFGModel(opt)
    elif opt.model == 'MVGModel':
        opt.option = 'MVG'
        return MVGModel(opt)
    else:
        raise Exception("This implementation only supports FFG, MVG Models.")
