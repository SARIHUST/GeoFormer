import torch
from datasets.scannetv2_fs_inst import FSInstDataset
from model.geoformer.geoformer_fs import GeoFormerFS

if __name__ == '__main__':
    dataset = FSInstDataset()
    loader = dataset.trainLoader()
    device = torch.device('cuda')
    model = GeoFormerFS()
    for iteration, batch in enumerate(loader):
        support_dict, query_dict, _ = batch
        output = model(support_dict, query_dict)
        k = 1