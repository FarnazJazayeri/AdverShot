from data.dataloader import MyDataLoader
from model.maml_net import MAMLNet
from model.proto_net import GnericProtoNet
from model.meta import Meta
from tools.attacks.pgd import PGD

problem_params = dict(
    num_tasks=1000,
    n_way=5,
    k_shot_spt=5,
    k_shot_qry=1,
)
hyper_params = dict(
    update_lr=0.01,
    meta_lr=0.01,
    n_way=5,
    adv_reg_param=0.001,
    update_steps=100,
    update_steps_test=100,
)
if __name__ == '__main__':
    dl = MyDataLoader(
        num_tasks=problem_params['num_tasks'],
        n_way=problem_params['n_way'],
        k_shot_spt=problem_params['k_shot_spt'],
        k_shot_qry=problem_params['k_shot_qry'],
    )
    train_dl, validation_dl, test_dl = dl.load_few_shot_dataset('omniglot')
    learner = MAMLNet(n_way=problem_params['n_way'])
    attacker = PGD(learner)

    meta_learner = Meta(
        learner=learner,
        update_lr=hyper_params['update_lr'],
        meta_lr=hyper_params['meta_lr'],
        adv_reg_param=hyper_params['adv_reg_param'],
        update_steps=hyper_params['update_steps'],
        update_steps_test=hyper_params['update_steps_test'],
        attacker=attacker,
    )
    meta_learner.fit(train_dl, validation_dl)

    loss, accuracy, adv_loss, adv_accuracy = meta_learner.evaluate(test_dl)
