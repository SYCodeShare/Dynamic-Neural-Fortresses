import os
script_path = os.path.abspath(__file__)
proj_path = os.path.dirname(os.path.dirname(script_path))

dev_id=0
p_v="CUBS200"
f_v="resnet50"
queryset="Caltech256"
oeset="Indoor67"
oe_lamb=1.0
vic_dir=f"models/victim/{p_v}-{f_v}-train-nodefense"
budget=23000 #50000
ori_batch_size=32
lr=0.01
lr_step=10
lr_gamma=0.5
epochs=30
training_batch_size=32

pretrained="imagenet"

query_list = ['random']
attack_list = ['naive','top1'] 
defense_list = ['none']

for policy in query_list:
    for attack in attack_list:
        
        defense_aware=0
        
        if attack == 'top1':
            hardlabel=1
        else:
            hardlabel=0
        
        recover_table_size=1000000
        recover_norm=1
        recover_tolerance=0.01
        concentration_factor=8.0
        recover_proc=5
        recover_params=f"'table_size:{recover_table_size};recover_norm:{recover_norm};tolerance:{recover_tolerance};concentration_factor:{concentration_factor};recover_proc:{recover_proc}'"
        semi_train_weight=0.0
        semi_dataset=queryset
        transform=0
        qpi=1

        policy_suffix=f"_{attack}"

        for defense in defense_list:
            batch_size=ori_batch_size
            quantize=0
            quantize_epsilon=0.0
            optim=0
            ydist="l1"
            frozen=0
            quantize_args=f"'epsilon:{quantize_epsilon};ydist:{ydist};optim:{optim};frozen:{frozen};ordered_quantization:1'"       
            if defense == 'none':          
                strat="none"
                out_dir=f"models/final_bb_dist/{p_v}-{f_v}/{policy}{policy_suffix}-{queryset}-B{budget}/none"
             
                defense_args=f"'out_path:{out_dir}'"
                    
            command_eval = f"python defenses/victim/eval.py {vic_dir} {strat} {defense_args} --quantize {quantize} --quantize_args {quantize_args} --out_dir {out_dir} --batch_size {batch_size} -d {dev_id} > eval_{policy}_{attack}_{defense}.txt"
            status = os.system(command_eval)
            if status != 0:
                raise RuntimeError("Fail to evaluate the protected accuracy for defense {}".format(defense))
            
            if policy == 'random':         
                command_transfer = f"python defenses/adversary/transfer.py {policy} {vic_dir} {strat} {defense_args} --out_dir {out_dir} --batch_size {batch_size} -d {dev_id} --queryset {queryset} --budget {budget} --quantize {quantize} --quantize_args {quantize_args} --defense_aware {defense_aware} --recover_args {recover_params} --hardlabel {hardlabel} --train_transform {transform} --qpi {qpi} > transfer_{policy}_{attack}_{defense}.txt"
                f_v="vgg19_bn"                        
                command_train = f"python defenses/adversary/train.py {out_dir} {f_v} {p_v} --budgets {budget} -e {epochs} -b {training_batch_size} --lr {lr} --lr_step {lr_step} --lr_gamma {lr_gamma} -d {dev_id} -w 4 --pretrained {pretrained} --vic_dir {vic_dir} --semitrainweight {semi_train_weight} --semidataset {semi_dataset} > train_{policy}_{attack}_{defense}.txt"
                status = os.system(command_transfer)
                if status != 0:
                    if not os.path.exists(os.path.join(out_dir,'params_transfer.json')):
                        raise RuntimeError("Fail to generate transfer set with attack {} and defense {}".format('random_'+attack,defense))
                status = os.system(command_train)
                if status != 0:
                    raise RuntimeError("Fail to train the substitute model with attack {} and defense {}".format('random_'+attack,defense))
