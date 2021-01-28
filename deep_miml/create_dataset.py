import argparse

import torch
import torchvision.transforms as transforms

from deep_miml.cifar_bags import MIMLBagsData, get_test_data, get_train_val_split_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create and Save Multi-Instance Multi-Label Dataset"
    )

    parser.add_argument(
        "--num_bags_train",
        type=int,
        default=2000,
        metavar="NTrain",
        help="number of bags in training set",
    )

    parser.add_argument(
        "--num_bags_test",
        type=int,
        default=100,
        metavar="NTest",
        help="number of bags in test set",
    )

    parser.add_argument(
        "--num_bags_val",
        type=int,
        default=100,
        metavar="NVal",
        help="number of bags in validation set",
    )

    parser.add_argument(
        "--root",
        type=str,
        default="../data",
        help="location where CIFAR-10 is downloaded",
    )

    parser.add_argument(
        "--save_file_to",
        type=str,
        default="../data/miml_data2000.pt",
        help="location to save the file",
    )

    parser.add_argument(
        "--cpu_workers", type=int, default=4, metavar="S", help="Number of workers"
    )

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--m",
        type=int,
        default=4,
        help="Upper bound on number of instances per category per bag",
    )

    args = parser.parse_args()

    # Img transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # train and validation raw data
    train_dat, val_dat = get_train_val_split_data(
        root=args.root, transform=transform, num_workers=args.cpu_workers
    )

    # test raw data
    test_dat = get_test_data(
        root=args.root, transform=transform, num_workers=args.cpu_workers
    )

    image_datasets = {}
    # image_datasets['train'] = MIMLBagsData(train_dat, # tuple
    # 									 num_bag=args.num_bags_train,
    # 									 seed=args.seed,
    # 									 # TODO SW: consider making m { 'cata' : 10, 'catb' : 4, ...}
    # 									 m=args.m, # Upper bound on number of instances per category per bag
    # 									 category_list = None, # len(category_list) is N
    # 									 transform=transform,
    # 									 num_workers=args.cpu_workers)

    # image_datasets['val'] = MIMLBagsData(val_dat, # tuple
    # 					num_bag=args.num_bags_val,
    # 					seed=args.seed,
    # 					 # TODO SW: consider making m { 'cata' : 10, 'catb' : 4, ...}
    # 					m=args.m, # Upper bound on number of instances per category per bag
    # 					category_list = None, # len(category_list) is N
    # 					transform=transform,
    # 					num_workers=args.cpu_workers)

    image_datasets["test"] = MIMLBagsData(
        test_dat,  # tuple
        num_bag=args.num_bags_test,
        seed=args.seed,
        # TODO SW: consider making m { 'cata' : 10, 'catb' : 4, ...}
        m=args.m,  # Upper bound on number of instances per category per bag
        category_list=None,  # len(category_list) is N
        transform=transform,
        num_workers=args.cpu_workers,
    )

    torch.save(image_datasets, args.save_file_to)

    # with open('../data/mimil_dataset.pkl', 'wb') as pickle_file:
    #    # cPickle.dump(all, pickle_file)
    # 	pickle.dump(obj=image_datasets, file=pickle_file)
