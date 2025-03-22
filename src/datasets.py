import torch as th
import tqdm as tq
import h5py as h5
import pyvista as pv

from torch.utils.data import Dataset, DataLoader
from utils import convert_to_spec


class SignalsSet(Dataset):

    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params
        self._signals_ = h5.File(self.params["path"], "r")["tracings"]
    
    def __len__(self) -> int:
        return self._signals_.shape[0]

    def __getitem__(self, idx: int) -> None:
        
        sample = th.Tensor(self._signals_[idx])
        spec = convert_to_spec(sample, spec_size=self.params["spec_size"])
        return spec




if __name__ == "__main__":

    plt = pv.Plotter()
    dataset = SignalsSet({
        "path": "C:\\Users\\1\\Downloads\\exams_part0\\exams_part0.hdf5",
        "spec_size": (129, 129)
    })
    loader = DataLoader(
        dataset=dataset,
        batch_size=320,
        shuffle=True
    )

    sample = next(iter(loader))[0]
    plt.add_volume(sample.permute((1, 2, 0)).numpy(), opacity="sigmoid")
    plt.show()
        
    