from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED,as_completed
import threading
import time
import numpy as np
import tensorflow as tf
import math
import random
import copy
from Helper_Functions import Vname_to_FeedPname, Vname_to_Pname, check_validaity_of_FLAGS, create_save_dir, \
    global_step_creator, load_from_directory_or_initialize, bring_Accountant_up_to_date, save_progress, \
    save_progress_K_model, load_from_directory_or_initialize_K_model, WeightsAccountant, print_loss_and_accuracy, \
    print_new_comm_round, PrivAgent, Flag

from compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
from compute_noise_from_budget_lib import compute_noise

# model_num
# MODEL_NUM=9
lock=threading.Lock() #申请一把锁
finished_thread_Count=0

def Single_user_train(sess, assignments, set_global_step, model, participating_clients, clients_epoch,
                      increase_global_step, FLAGS,data_placeholder,label_placeholder,data_set_asarray,label_set_asarray,
                      s, c, train_op, real_round, acc):
    with sess.as_default():
        with sess.graph.as_default():
            choose_model = random.randrange(FLAGS.MODEL_NUM)
            data_ind = np.split(np.asarray(participating_clients[c]), FLAGS.b, 0)
            replaceModel = random.randrange(FLAGS.MODEL_NUM)

            clients_epoch[replaceModel][s[c]] = clients_epoch[choose_model][s[c]] + FLAGS.e

            # print('user: {} start'.format(c))
            if FLAGS.method==0:
                target_sigma = 1.5/FLAGS.b*((clients_epoch[replaceModel][s[c]]*FLAGS.b*math.log(1e3))**0.5)/FLAGS.THV

                # target_sigma = compute_noise(participating_clients[c].shape[0],
                #                          participating_clients[c].shape[0] / FLAGS.b, 1,
                #                          clients_epoch[replaceModel][s[c]], 1e-3, 1e-6, c)
            else:
                target_sigma = 0

            # lock.acquire()  # 加锁
            sess.run(assignments + [set_global_step], feed_dict=model[choose_model])
            # print(tf.trainable_variables())
            for e in range(int(FLAGS.e)):
                for step in range(len(data_ind)):
                    real_step = sess.run(increase_global_step)
                    batch_ind = data_ind[step]

                    feed_dict = {str(data_placeholder.name): data_set_asarray[[int(j) for j in batch_ind]],
                                 str(label_placeholder.name): label_set_asarray[[int(j) for j in batch_ind]]}
                    _ = sess.run([train_op], feed_dict=feed_dict)

            weights_accountant = WeightsAccountant(sess, model[choose_model], target_sigma, real_round,
                                                   participating_clients[c].shape[0] * FLAGS.e)
            newModel = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, [])

            model[replaceModel] = newModel
            model[replaceModel]['global_step_placeholder:0'] = real_step

            # lock.acquire()  # 加锁
            global finished_thread_Count
            finished_thread_Count= finished_thread_Count + 1
            # lock.release()
            # print('user: {} finish'.format(c))
    return

def run_differentially_private_federated_averaging(loss, train_op, eval_correct, data, data_placeholder,
                                                   label_placeholder, privacy_agent=None, b=10, e=4,
                                                   record_privacy=True, m=0, sigma=0, eps=8, save_dir=None,
                                                   log_dir=None, max_comm_rounds=3000, gm=True,
                                                   saver_func=create_save_dir, save_params=False, MODEL_NUM=10,
                                                   THV=1, C=0.01, method=0, isTest=False):

    """
    This function will simulate a federated learning setting and enable differential privacy tracking. It will detect
    all trainable tensorflow variables in the tensorflow graph and simulate a decentralized learning process where these
    variables are learned through clients that only have access to their own data set.
    This function must therefore be run inside a Graph as follows:
    --------------------------------------------------------------------------------------------------------------------

    with tf.Graph().as_default():

        train_op, eval_correct, loss, data_placeholder, labels_placeholder = Some_function_that_builds_TF_graph()

        Accuracy_accountant, Delta_accountant, model = \
            run_differentially_private_federated_averaging(loss, train_op, eval_correct, DATA, data_placeholder,
                                                           labels_placeholder)
    --------------------------------------------------------------------------------------------------------------------
    The graph that train_op, loss and eval_op belong to should have a global_step variable.

    :param loss:                TENSORFLOW node that computes the current loss
    :param train_op:            TENSORFLOW Training_op
    :param eval_correct:        TENSORFLOW node that evaluates the number of correct predictions
    :param data:                A class instance with attributes:
                                .data_set       : The training data stored in a list or numpy array.
                                .label_set      : The training labels stored in a list or numpy array.
                                                  The indices should correspond to .data_set. This means a single index
                                                  corresponds to a data(x)-label(y) pair used for training:
                                                  (x_i, y_i) = (data.data_set(i),data.label_set(i))
                                .client_set     : A nested list or numpy array. len(data.client_set) is the total
                                                  number of clients. for any j, data.client_set[j] is a list (or array)
                                                  holding indices. these indices specify the data points that client j
                                                  holds.
                                                  i.e. if i \in data.client_set[j], then client j owns (x_i, y_i)
                                .vali_data_set  : The validation data stored in a list or numpy array.
                                .vali_label_set : The validation labels stored in a list or numpy array.
    :param data_placeholder:    The placeholder from the tensorflow graph that is used to feed the model with data
    :param label_placeholder:   The placeholder from the tensorflow graph that is used to feed the model with labels
    :param privacy_agent:       A class instance that has callabels .get_m(r) .get_Sigma(r) .get_bound(), where r is the
                                communication round.
    :param b:                   Batchsize
    :param e:                   Epochs to run on each client
    :param record_privacy:      Whether to record the privacy or not
    :param m:                   If specified, a privacyAgent is not used, instead the parameter is kept constant
    :param sigma:               If specified, a privacyAgent is not used, instead the parameter is kept constant
    :param eps:                 The epsilon for epsilon-delta privacy
    :param save_dir:            Directory to store the process
    :param log_dir:             Directory to store the graph
    :param max_comm_rounds:     The maximum number of allowed communication rounds
    :param gm:                  Whether to use a Gaussian Mechanism or not.
    :param saver_func:          A function that specifies where and how to save progress: Note that the usual tensorflow
                                tracking will not work
    :param save_params:         save all weights_throughout training.

    :return:

    """

    # If no privacy agent was specified, the default privacy agent is used.
    if not privacy_agent:
        if method==0:
            nstr="完善版本算法模型diff"
        elif method==1:
            nstr="完善版无噪声"
        elif method==2:
            nstr="完善版联邦平均"

        privacy_agent = PrivAgent(len(data.client_set), 'K_'+str(MODEL_NUM)+'_'+nstr)


    # A Flags instance is created that will fuse all specified parameters and default those that are not specified.
    FLAGS = Flag(len(data.client_set), b, e, record_privacy, m, sigma, eps, save_dir, log_dir, max_comm_rounds, gm,
                 privacy_agent, MODEL_NUM, THV, C, method)
    # Check whether the specified parameters make sense.
    FLAGS = check_validaity_of_FLAGS(FLAGS)

    # At this point, FLAGS.save_dir specifies both; where we save progress and where we assume the data is stored
    save_dir = saver_func(FLAGS)

    # This function will retrieve the variable associated to the global step and create nodes that serve to
    # increase and reset it to a certain value.
    increase_global_step, set_global_step = global_step_creator()

    # - model_placeholder : a dictionary in which there is a placeholder stored for every trainable variable defined
    #                       in the tensorflow graph. Each placeholder corresponds to one trainable variable and has
    #                       the same shape and dtype as that variable. in addition, the placeholder has the same
    #                       name as the Variable, but a '_placeholder:0' added to it. The keys of the dictionary
    #                       correspond to the name of the respective placeholder
    model_placeholder = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                                 [tf.placeholder(name=Vname_to_Pname(var),
                                                 shape=var.shape,
                                                 dtype=tf.float32)
                                  for var in tf.trainable_variables()]))

    # - assignments : Is a list of nodes. when run, all trainable variables are set to the value specified through
    #                 the placeholders in 'model_placeholder'.

    assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)]) for var in
                   tf.trainable_variables()]

    # load_from_directory_or_initialize checks whether there is a model at 'save_dir' corresponding to the one we
    # are building. If so, training is resumed, if not, it returns:  - model = []
    #                                                                - accuracy_accountant = []
    #                                                                - delta_accountant = []
    #                                                                - real_round = 0
    # And initializes a Differential_Privacy_Accountant as acc

    # model, accuracy_accountant, delta_accountant, acc, real_round, FLAGS, computed_deltas = \
    #     load_from_directory_or_initialize(save_dir, FLAGS)
    model, accuracy_accountant, clients_epoch,timeList, acc, real_round, FLAGS = \
        load_from_directory_or_initialize_K_model(save_dir, FLAGS)

    m = int(FLAGS.m)
    sigma = float(FLAGS.sigma)
    # - m : amount of clients participating in a round
    # - sigma : variable for the Gaussian Mechanism.
    # Both will only be used if no Privacy_Agent is deployed.

    ################################################################################################################
    # Test Part
    # if isTest==True:
    #     assert (model is not None)
    #     sess = tf.Session()
    #
    #     if method == 0 or method == 1:
    #         # 1.0/2.0 每轮测试选择所有模型的平均模型进行测试
    #         average_model = copy.deepcopy(model[0])
    #         for k in model[0].keys():
    #             for i in range(1, FLAGS.MODEL_NUM):
    #                 average_model[k] = average_model[k] + model[i][k]
    #             average_model[k] = average_model[k] / FLAGS.MODEL_NUM
    #
    #         # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
    #         sess.run(assignments, feed_dict=average_model)
    #     else:
    #         sess.run(assignments, feed_dict=model)
    #
    #     feed_dict = {str(data_placeholder.name): np.asarray(data.x_vali),
    #                  str(label_placeholder.name): np.asarray(data.y_vali)}
    #
    #     # compute the loss on the validation set.
    #     global_loss = sess.run(loss, feed_dict=feed_dict)
    #     count = sess.run(eval_correct, feed_dict=feed_dict)
    #     accuracy = float(count) / float(len(data.y_vali))
    #     accuracy_accountant.append(accuracy)
    #
    #     print("平均模型准确率")
    #     print_loss_and_accuracy(global_loss, accuracy)
    #
    #     for i in range(FLAGS.MODEL_NUM):
    #         sess.run(assignments, feed_dict=model[i])
    #         feed_dict = {str(data_placeholder.name): np.asarray(data.x_vali),
    #                      str(label_placeholder.name): np.asarray(data.y_vali)}
    #
    #         # compute the loss on the validation set.
    #         global_loss = sess.run(loss, feed_dict=feed_dict)
    #         count = sess.run(eval_correct, feed_dict=feed_dict)
    #         accuracy = float(count) / float(len(data.y_vali))
    #         accuracy_accountant.append(accuracy)
    #
    #         print("第"+str(i)+"个模型准确率")
    #         print_loss_and_accuracy(global_loss, accuracy)
    #     return

    ################################################################################################################

    # Usual Tensorflow...

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    ################################################################################################################

    # If there was no loadable model, we initialize a model:
    # - model : dictionary having as keys the names of the placeholders associated to each variable. It will serve
    #           as a feed_dict to assign values to the placeholders which are used to set the variables to
    #           specific values.

    if not model:
        # 0.0 训练开始前初始化K个模型参数
        if method==2:
            model=dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                                  [sess.run(var) for var in tf.trainable_variables()]))
            model['global_step_placeholder:0'] = 0
        else:
            for i in range(FLAGS.MODEL_NUM):
                model.append(dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                                                  [sess.run(var) for var in tf.trainable_variables()])))
                model[i]['global_step_placeholder:0'] = 0

        sess.run(init)
        # end 0.0

        # original 只有一个模型
        # model = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
        #                  [sess.run(var) for var in tf.trainable_variables()]))
        # model['global_step_placeholder:0'] = 0
        # original

        real_round = 0

        weights_accountant = []

    # If a model is loaded, and we are not relearning it (relearning means that we once already finished such a model
    # and we are learning it again to average the outcomes), we have to get the privacy accountant up to date. This
    # means, that we have to iterate the privacy accountant over all the m, sigmas that correspond to already completed
    # communication

    if not FLAGS.relearn and real_round > 0:
        bring_Accountant_up_to_date(acc, sess, real_round, privacy_agent, FLAGS)

    ################################################################################################################

    # This is where the actual communication rounds start:

    data_set_asarray = np.asarray(data.sorted_x_train)
    label_set_asarray = np.asarray(data.sorted_y_train)
    time_start = time.time()
    for r in range(18):

        # First, we check whether we are loading a model, if so, we have to skip the first allocation, as it took place
        # already.

        # 1.0 每轮选择一个模型，数据集非IID，非噪声梯度更新
        # choose one model
        # choose_model = random.randrange(MODEL_NUM)
        # end 1.0

        # compute result
        # need use average model
        # but dont care
        if not (FLAGS.loaded and r == 0):
            if method==0 or method==1 :
                # 1.0/2.0 每轮测试选择所有模型的平均模型进行测试
                average_model = copy.deepcopy(model[0])
                for k in model[0].keys():
                    for i in range(1, FLAGS.MODEL_NUM):
                        average_model[k] = average_model[k] + model[i][k]
                    average_model[k]=average_model[k] / FLAGS.MODEL_NUM

                # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
                sess.run(assignments, feed_dict=average_model)
            else:
                sess.run(assignments, feed_dict=model)
            # end 1.0

            # original 只有一个模型
            #sess.run(assignments, feed_dict=model)
            # end original

            # create a feed-dict holding the validation set.

            feed_dict = {str(data_placeholder.name): np.asarray(data.x_vali),
                         str(label_placeholder.name): np.asarray(data.y_vali)}

            # compute the loss on the validation set.
            global_loss = sess.run(loss, feed_dict=feed_dict)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            accuracy = float(count) / float(len(data.y_vali))
            accuracy_accountant.append(accuracy)
            timeList.append(int(time.time()-time_start))

            print_loss_and_accuracy(global_loss, accuracy)
            save_progress_K_model(save_dir, model, accuracy_accountant, privacy_agent, clients_epoch,timeList, FLAGS)

            if isTest==True:
                with open(save_dir+"/model_diff.txt", 'a+') as f:
                    f.write("round"+str(real_round)+":\n")
                    f.write(str(accuracy)+" "+str(global_loss))
                    for k in model[0].keys():
                        mu_Test = np.mean(average_model[k])
                        sigma_Test = np.var(average_model[k])
                        f.write(" "+str(mu_Test)+" "+str(sigma_Test))
                    f.write("\n")

                    for i in range(FLAGS.MODEL_NUM):
                        sess.run(assignments, feed_dict=model[i])
                        feed_dict = {str(data_placeholder.name): np.asarray(data.x_vali),
                                     str(label_placeholder.name): np.asarray(data.y_vali)}

                        # compute the loss on the validation set.
                        global_loss = sess.run(loss, feed_dict=feed_dict)
                        count = sess.run(eval_correct, feed_dict=feed_dict)
                        accuracy = float(count) / float(len(data.y_vali))
                        accuracy_accountant.append(accuracy)
                        f.write(str(accuracy) + " " + str(global_loss) + " ")
                        for k in model[0].keys():
                            mu_Test = np.mean(model[i][k])
                            sigma_Test = np.var(model[i][k])
                            f.write(" " + str(mu_Test) + " " + str(sigma_Test))
                        f.write("\n")

        # original 计算隐私边界并自动停止
        # if delta_accountant[-1] > privacy_agent.get_bound() or math.isnan(delta_accountant[-1]):
        #     print('************** The last step exhausted the privacy budget **************')
        #     if not math.isnan(delta_accountant[-1]):
        #         try:
        #             None
        #         finally:
        #             save_progress(save_dir, model, delta_accountant + [float('nan')],
        #                           accuracy_accountant + [float('nan')], privacy_agent, FLAGS)
        #         return accuracy_accountant, delta_accountant, model
        # else:
        #     try:
        #         None
        #     finally:
        # end original
            #save_progress(save_dir, model, delta_accountant, accuracy_accountant, privacy_agent, FLAGS)

        ############################################################################################################
        # Start of a new communication round

        real_round = real_round + 1

        print_new_comm_round(real_round)

        if FLAGS.priv_agent:
            m = int(privacy_agent.get_m(int(real_round)))
            sigma = privacy_agent.get_Sigma(int(real_round))

        print('Clients participating: ' + str(m))

        # Randomly choose a total of m (out of n) client-indices that participate in this round
        # randomly permute a range-list of length n: [1,2,3...n] --> [5,2,7..3]
        # perm = np.random.permutation(FLAGS.n)
        perm = np.arange(FLAGS.n)

        # Use the first m entries of the permuted list to decide which clients (and their sets) will participate in
        # this round. participating_clients is therefore a nested list of length m. participating_clients[i] should be
        # a list of integers that specify which data points are held by client i. Note that this nested list is a
        # mapping only. the actual data is stored in data.data_set.
        s = perm[0:m].tolist()
        participating_clients = [data.client_set[k] for k in s]

        if method==0:
            for c in range(5):
                participating_clients[c]=participating_clients[c][:225]
            for c in range(5,100):
                participating_clients[c]=participating_clients[c][:475]

        global finished_thread_Count
        finished_thread_Count=0

        # For each client c (out of the m chosen ones):
        if method == 2:
            for c in range(m):
                sess.run(assignments + [set_global_step], feed_dict=model)
                data_ind = np.split(np.asarray(participating_clients[c]), FLAGS.b, 0)

                for e in range(int(FLAGS.e)):
                    for step in range(len(data_ind)):
                        real_step = sess.run(increase_global_step)
                        batch_ind = data_ind[step]

                        feed_dict = {str(data_placeholder.name): data_set_asarray[[int(j) for j in batch_ind]],
                                     str(label_placeholder.name): label_set_asarray[[int(j) for j in batch_ind]]}
                        _ = sess.run([train_op], feed_dict=feed_dict)

                if c == 0:
                    weights_accountant = WeightsAccountant(sess, model, sigma, real_round, participating_clients[c].shape[0] * FLAGS.e)
                else:
                    weights_accountant.allocate(sess)

            model = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, [])
            model['global_step_placeholder:0'] = real_step
            # 2.0 每台机器选择一个模型，数据集非IID，非噪声梯度更新
            # choose one model
            # choose_model = random.randrange(FLAGS.MODEL_NUM)
            # Assign the global model and set the global step. This is obsolete when the first client trains,
            # but as soon as the next client trains, all progress allocated before, has to be discarded and the
            # trainable variables reset to the values specified in 'model'
            # sess.run(assignments + [set_global_step], feed_dict=model[choose_model])
            # end 1.0

            # original 只有一个模型
            #sess.run(assignments + [set_global_step], feed_dict=model)
            # end original

            # allocate a list, holding data indices associated to client c and split into batches.
            # data_ind = np.split(np.asarray(participating_clients[c]), FLAGS.b, 0)

            # 2.0 每台机器选择一个模型，噪声和非噪声都有
            # replaceModel = random.randrange(FLAGS.MODEL_NUM)



            # clients_epoch[replaceModel][s[c]] = clients_epoch[choose_model][s[c]] + FLAGS.e

            # target_sigma = compute_noise(participating_clients[c].shape[0],
            #                              participating_clients[c].shape[0] / FLAGS.b, 1,
            #                              clients_epoch[replaceModel][s[c]], 1e-3, 1e-6)
            #
            # sess.run(assignments + [set_global_step], feed_dict=model[choose_model])
            # #end 2.0
            #
            # # e = Epoch
            # for e in range(int(FLAGS.e)):
            #     for step in range(len(data_ind)):
            #         # increase the global_step count (it's used for the learning rate.)
            #         real_step = sess.run(increase_global_step)
            #         # batch_ind holds the indices of the current batch
            #         batch_ind = data_ind[step]
            #
            #         # Fill a feed dictionary with the actual set of data and labels using the data and labels associated
            #         # to the indices stored in batch_ind:
            #
            #         # 1.0
            #         # for j in batch_ind:
            #         #     feed_dict = {str(data_placeholder.name): data_set_asarray[[int(j)]],
            #         #                               str(label_placeholder.name): label_set_asarray[[int(j)]]}
            #         #     sess.run(assignments + [set_global_step], feed_dict=model[choose_model])
            #         #     _ = sess.run([train_op], feed_dict=feed_dict)
            #         #     if j == batch_ind[0]:
            #         #         # If we just trained the first client in a comm_round, We override the old weights_accountant (or,
            #         #         # if this was the first comm_round, we allocate a new one. The Weights_accountant keeps track of
            #         #         # all client updates throughout a communication round.
            #         #         weights_accountant = WeightsAccountant(sess, model[choose_model], target_sigma, real_round,
            #         #                                                participating_clients[c].shape[0])
            #         #     else:
            #         #         # Allocate the client update, if this is not the first client in a communication round
            #         #         weights_accountant.allocate(sess)
            #         #
            #         # newModel = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, [])
            #         # model[replaceModel] = newModel
            #         # model[replaceModel]['global_step_placeholder:0'] = real_step
            #         # end 1.0
            #         # original
            #         feed_dict = {str(data_placeholder.name): data_set_asarray[[int(j) for j in batch_ind]],
            #                      str(label_placeholder.name): label_set_asarray[[int(j) for j in batch_ind]]}
            #         # Run one optimization step.
            #         _ = sess.run([train_op], feed_dict=feed_dict)
            #         # end original
            #
            #         # original/1.0 每轮选择一个模型，数据集非IID，非噪声梯度更新
            #         # if c == 0:
            #         #
            #         #     # If we just trained the first client in a comm_round, We override the old weights_accountant (or,
            #         #     # if this was the first comm_round, we allocate a new one. The Weights_accountant keeps track of
            #         #     # all client updates throughout a communication round.
            #         #     weights_accountant = WeightsAccountant(sess, model[choose_model], sigma, real_round)
            #         # else:
            #         #     # Allocate the client update, if this is not the first client in a communication round
            #         #     weights_accountant.allocate(sess)
            #         # end 1.0
            #
            #         # 2.0 每台机器选择一个模型，噪声和非噪声都有
            #         # if step==0 and e==0:
            #         #     weights_accountant = WeightsAccountant(sess, model[choose_model], target_sigma, real_round, participating_clients[c].shape[0])
            #         # else:
            #         #     weights_accountant.allocate(sess)
            #         # newModel = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, computed_deltas)
            #
            # weights_accountant = WeightsAccountant(sess, model[choose_model], target_sigma, real_round,
            #                                        participating_clients[c].shape[0]*FLAGS.e)
            # newModel = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, [])
            #
            # model[replaceModel] = newModel
            #         # original
            #         # delta_accountant.append(delta)
            #         # end original
            #         # if c==0:
            #         #     target_sigma = compute_noise(participating_clients[c].shape[0], participating_clients[c].shape[0]/FLAGS.b, 1, clients_epoch[replaceModel][s[c]], 1e-3, 1e-6)
            #         #     eps, _ = compute_dp_sgd_privacy(participating_clients[c].shape[0], participating_clients[c].shape[0]/FLAGS.b, sigma, clients_epoch[s[c]], 1e-3)
            # model[replaceModel]['global_step_placeholder:0'] = real_step
                    # end 2.0
            # newModel = Single_user_train(sess, assignments, set_global_step, model, participating_clients,
            #                              clients_epoch,
            #                              choose_model, replaceModel,
            #                              data_ind, increase_global_step, FLAGS, data_placeholder, label_placeholder,
            #                              data_set_asarray, label_set_asarray,
            #                              s, c, train_op, real_round, acc)
        # End of a communication round
        ############################################################################################################
        elif method==0:
            for c in range(m):
                Single_user_train(sess, assignments, set_global_step, model, participating_clients, clients_epoch,
                          increase_global_step, FLAGS, data_placeholder, label_placeholder, data_set_asarray,label_set_asarray,
                          s, c, train_op, real_round, acc)
            # for c in range(m):
            #     threading.Thread(target=Single_user_train, args=(sess, assignments, set_global_step, model, participating_clients, clients_epoch,
            #           increase_global_step, FLAGS, data_placeholder, label_placeholder, data_set_asarray,label_set_asarray,
            #           s, c, train_op, real_round, acc)).start()
            # Single_user_train(sess, assignments, set_global_step, model, participating_clients, clients_epoch,
            #           increase_global_step, FLAGS, data_placeholder, label_placeholder, data_set_asarray,label_set_asarray,
            #           s, c, train_op, real_round, acc)
            # task = pool.submit(Single_user_train, sess, assignments, set_global_step, model, participating_clients, clients_epoch,
            #         increase_global_step, FLAGS, data_placeholder, label_placeholder, data_set_asarray,
            #         label_set_asarray, s, c, train_op, real_round, acc)
            # model = task.result()

            # while finished_thread_Count!=m:
            #     pass
        else:
            for c in range(m):
                Single_user_train(sess, assignments, set_global_step, model, participating_clients, clients_epoch,
                          increase_global_step, FLAGS, data_placeholder, label_placeholder, data_set_asarray,label_set_asarray,
                          s, c, train_op, real_round, acc)
        print('......Communication round %s completed' % str(real_round))
        # Compute a new model according to the updates and the Gaussian mechanism specifications from FLAGS
        # Also, if computed_deltas is an empty list, compute delta; the probability of Epsilon-Differential Privacy
        # being broken by allocating the model. If computed_deltas is passed, instead of computing delta, the
        # pre-computed vaue is used.

        # original 每轮训练选择一个模型
        #model, delta = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, computed_deltas)
        #end original

        # 1.0 每轮训练选择一个模型
        #replace model
        #newModel, delta = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, computed_deltas)
        #replaceModel = random.randrange(MODEL_NUM)
        #model[replaceModel] = newModel

        # append delta to a list.
        # delta_accountant.append(delta)

        # Set the global_step to the current step of the last client, such that the next clients can feed it into
        # the learning rate.
        #model[replaceModel]['global_step_placeholder:0'] = real_step

        # PRINT the progress and stage of affairs.
        #print(' - Epsilon-Delta Privacy:' + str([FLAGS.eps, delta]))
        # end 1.0

        if save_params:
            weights_accountant.save_params(save_dir)
    time_end = time.time()
    with open(save_dir+'/time.txt', 'w') as f:
        f.write(str(time_end-time_start))  # 文件的写操作
    return [], [], []
