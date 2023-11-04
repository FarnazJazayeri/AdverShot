from AdverShot.data.dataloader import MyDataLoader
from learner import Learner
from meta import Meta

problem_params = dict(
    n_way = 5,

)
hyper_params = dict(
    update_lr=0.01,
    meta_lr=0.01,
    n_way=5,

)
if __name__ == '__main__':
    my_data_loader = MyDataLoader()
    train, validation, test = my_data_loader.load_omniglot()
    learner = Learner()
    meta_learner = Meta(hyper_params, learner)
    meta_learner.fit(train)

    loss, accuracy, adv_loss, adv_accuracy = meta_learner.evaluate(test)
