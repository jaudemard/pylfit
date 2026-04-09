import os 
import pylfit 

if __name__ == "__main__":

    data_path = "./tests/datasets/mammalian.csv"
    targets = ["CycD_t","CycE_t","Rb_t","E2F_t","CycA_t","p27_t","Cdc20_t","UbcH10_t","Cdh1_t","CycB_t"]
    features = ["CycD_t_1","CycE_t_1","Rb_t_1","E2F_t_1","CycA_t_1","p27_t_1","Cdc20_t_1","UbcH10_t_1","Cdh1_t_1","CycB_t_1"]
    dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_csv(path = data_path, feature_names=features, target_names=targets)

    background_knowledge = [
        "CycD_t(0) :- CycD_t_1(0).",
        # "CycD_t(0) :- CycD_t_1(1).",
        "UbcH10_t(1) :- Cdh1_t_1(0).",
        "CycB_t(1) :- Cdc20_t_1(0), Cdh1_t_1(0).",
        "CycA_t(1) :- Cdc20_t_1(0), Cdh1_t_1(0), E2F_t_1(1), Rb_t_1(0).",
        #"p27_t(1) :- CycA_t_1(0), CycB_t_1(0), CycD_t_1(0)"
        ]

    background_rules = [pylfit.objects.Rule.from_string(bk_rule, dataset.features, dataset.targets) for bk_rule in background_knowledge]


    model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula")

    model.fit(dataset=dataset, options= {"background_rules" : background_rules})

    print(model.summary())



