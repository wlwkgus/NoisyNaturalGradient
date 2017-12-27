from models.ffg_model import FFGModel


def create_model(opt):
    if opt.model == 'FFGModel':
        return FFGModel(opt)
    else:
        raise Exception("This implementation only supports ALI, GibbsNet.")
