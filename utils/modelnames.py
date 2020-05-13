from models.mmp import mmp
from models.acf import acf
from models.bpr import bpr
from models.dvae import dvae
models = {
    "MMP": mmp,
    "ACF": acf,
    'BPR': bpr,
    "DVAE": dvae
}
