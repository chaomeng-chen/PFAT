with open("Dp/N_100/Epochs_4_Batches_25/K_10_本算法模型diff"+"/model_diff.txt", 'r') as f:
    acc = [[] for _ in range(11)]
    for j in range(25):
        line = f.readline()
        if line[:5]=="round":
            model = f.readline()
            model = model.split(" ")
            acc[0].append(float(model[3]))
            for i in range(1,11):
                model = f.readline()
                model = model.split(" ")
                acc[i].append(float(model[4]))

    with open("Dp/N_100/Epochs_4_Batches_25/K_10_本算法模型diff"+"/var1_diff.csv", 'w') as wt:
        for i in range(11):
            for every in acc[i]:
                wt.write(str(every))
                wt.write(",")
            wt.write("\n")