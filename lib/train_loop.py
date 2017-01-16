import lib

import numpy as np
import theano
import theano.tensor as T
import lasagne

import time
import itertools
import collections

def train_loop(
    inputs,
    cost,
    train_data,
    times,
    prints=None,
    inject_total_iters=False,
    test_data=None,
    callback=None,
    optimizer=lasagne.updates.adam,
    save_params=False,
    nan_guard=False
    ):

    params = lib.search(cost, lambda x: hasattr(x, 'param'))
    lib.print_params_info(params)

    grads = T.grad(cost, wrt=params, disconnected_inputs='warn')

    grads = [T.clip(g, lib.floatX(-1), lib.floatX(1)) for g in grads]

    updates = optimizer(grads, params)

    if prints is None:
        prints = [('cost', cost)]
    else:
        prints = [('cost', cost)] + prints

    print "Compiling train function..."
    if nan_guard:
        from theano.compile.nanguardmode import NanGuardMode
        mode = NanGuardMode(
            nan_is_error=True, 
            inf_is_error=True, 
            big_is_error=True
        )
    else:
        mode = None
    train_fn = theano.function(
        inputs,
        [p[1] for p in prints],
        updates=updates,
        on_unused_input='warn',
        mode=mode
    )

    print "Compiling eval function..."
    eval_fn = theano.function(
        inputs,
        [p[1] for p in prints],
        on_unused_input='warn'
    )

    print "Training!"

    total_iters = 0
    total_seconds = 0.
    last_print = 0
    last_gen = 0

    if len(times) >= 4:
        gen_every = times[3]
    else:
        gen_every = times[1]

    if len(times) >= 5:
        early_stop = times[4]
        if len(times) >= 6:
            early_stop_min = times[5]
        else:
            early_stop_min = 0
    else:
        early_stop = None
        early_stop_min = None

    best_test_cost = np.inf
    best_test_cost_iter = 0.

    all_outputs = []
    all_stats = []
    for epoch in itertools.count():

        generator = train_data()
        while True:
            try:
                inputs = generator.next()
            except StopIteration:
                break

            if inject_total_iters:
                inputs = [np.int32(total_iters)] + list(inputs)

            start_time = time.time()
            outputs = train_fn(*inputs)
            total_seconds += time.time() - start_time
            total_iters += 1

            all_outputs.append(outputs)

            if total_iters == 1:
                try: # This only matters on Ishaan's computer
                    import experiment_tools
                    experiment_tools.register_crash_notifier()
                except ImportError:
                    pass

            if (times[0]=='iters' and total_iters-last_print == times[1]) or \
                (times[0]=='seconds' and total_seconds-last_print >= times[1]):

                mean_outputs = np.array(all_outputs).mean(axis=0)

                if test_data is not None:
                    if inject_total_iters:
                        test_outputs = [
                            eval_fn(np.int32(total_iters), *inputs)
                            for inputs in test_data()
                        ]
                    else:
                        test_outputs = [
                            eval_fn(*inputs) 
                            for inputs in test_data()
                        ]
                    test_mean_outputs = np.array(test_outputs).mean(axis=0)

                stats = collections.OrderedDict()
                stats['epoch'] = epoch
                stats['iters'] = total_iters
                for i,p in enumerate(prints):
                    stats['train '+p[0]] = mean_outputs[i]
                if test_data is not None:
                    for i,p in enumerate(prints):
                        stats['test '+p[0]] = test_mean_outputs[i]
                stats['secs'] = total_seconds
                stats['secs/iter'] = total_seconds / total_iters

                if test_data != None and (stats['test cost'] < best_test_cost or (early_stop_min != None and total_iters <= early_stop_min)):
                    best_test_cost = stats['test cost']
                    best_test_cost_iter = total_iters

                print_str = ""
                for k,v in stats.items():
                    if isinstance(v, int):
                        print_str += "{}:{}\t".format(k,v)
                    else:
                        print_str += "{}:{:.4f}\t".format(k,v)
                print print_str[:-1] # omit the last \t

                all_stats.append(stats)

                all_outputs = []
                last_print += times[1]

            if (times[0]=='iters' and total_iters-last_gen==gen_every) or \
                (times[0]=='seconds' and total_seconds-last_gen >= gen_every):
                tag = "iters{}_time{}".format(total_iters, total_seconds)
                if callback is not None:
                    callback(tag)
                if save_params:
                    lib.save_params('params_{}.pkl'.format(tag))

                last_gen += gen_every

            if (times[0]=='iters' and total_iters == times[2]) or \
                (times[0]=='seconds' and total_seconds >= times[2]) or \
                (test_data != None and early_stop != None and total_iters > (3*early_stop) and (total_iters-best_test_cost_iter) > early_stop):

                if (test_data != None and early_stop != None and total_iters > (3*early_stop) and (total_iters-best_test_cost_iter) > early_stop):
                    print "Early stop! Best test cost was {} at iter {}".format(best_test_cost, best_test_cost_iter)

                print "Done!"

                try: # This only matters on Ishaan's computer
                    import experiment_tools
                    experiment_tools.send_sms("done!")
                except ImportError:
                    pass

                return all_stats