from data.dataloader import MyDataLoader

dl = MyDataLoader()

train, validation, test = dl.load_few_shot_dataset('omniglot')

for spt_x, spt_y, qry_x, qry_y in train:
    print(len(spt_x[0]), len(spt_y), len(qry_x), len(qry_y))
    break

