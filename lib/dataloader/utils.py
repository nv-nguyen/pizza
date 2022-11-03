from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def init_dataloader(dict_dataloader, use_distributed, batch_size, num_workers):
    names = dict_dataloader.keys()
    print("Names dataloader {}".format(names))
    results = {}
    for name in names:
        if use_distributed:
            if name == "train":
                sampler = DistributedSampler(dict_dataloader[name])
                train_sampler = sampler
                shuffle = False
            else:
                sampler = DistributedSampler(dict_dataloader[name])
                shuffle = False
        else:
            sampler = None
            train_sampler = None
            shuffle = True
        dataloader = DataLoader(dict_dataloader[name], batch_size=batch_size,
                                num_workers=num_workers, drop_last=False,
                                sampler=sampler, shuffle=shuffle, pin_memory=True)
        results[name] = dataloader
    return train_sampler, results
