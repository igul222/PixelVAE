import tflib as lib

import numpy as np
import tensorflow as tf

import collections
import cPickle as pickle
import json
import locale
import os
import time
import shutil

locale.setlocale(locale.LC_ALL, '')

PARAMS_FILE = 'params.ckpt'
TRAIN_LOOP_FILE = 'train_loop.pkl'
TRAIN_OUTPUT_FILE = 'train_output.ndjson'

def train_loop(
    session,
    inputs,
    cost,
    train_data,
    stop_after,
    prints=[],
    test_data=None,
    test_every=None,
    callback=None,
    callback_every=None,
    inject_iteration=False,
    bn_vars=None,
    bn_stats_iters=1000,
    before_test=None,
    optimizer=tf.train.AdamOptimizer(),
    save_every=1000,
    save_checkpoints=False
    ):

    prints = [('cost', cost)] + prints

    grads_and_vars = optimizer.compute_gradients(
        cost,
        colocate_gradients_with_ops=True
    )

    print "Params:"
    total_param_count = 0
    for g, v in grads_and_vars:
        shape = v.get_shape()
        shape_str = ",".join([str(x) for x in v.get_shape()])

        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count

        if g == None:
            print "\t{} ({}) [no grad!]".format(v.name, shape_str)
        else:
            print "\t{} ({})".format(v.name, shape_str)
    print "Total param count: {}".format(
        locale.format("%d", total_param_count, grouping=True)
    )

    # for i in xrange(len(grads_and_vars)):
    #     g, v = grads_and_vars[i]
    #     if g == None:
    #         grads_and_vars[i] = (tf.zeros_like(v), v)
    #     else:
    #         grads_and_vars[i] = (tf.clip_by_value(g, -5., 5.), v)

    grads = [g for g,v in grads_and_vars]
    _vars = [v for g,v in grads_and_vars]

    global_norm = tf.global_norm(grads)
    prints = prints + [('gradnorm', global_norm)]

    grads, global_norm = tf.clip_by_global_norm(grads, 5.0, use_norm=global_norm)
    grads_and_vars = zip(grads, _vars)

    train_op = optimizer.apply_gradients(grads_and_vars)

    def train_fn(input_vals):
        feed_dict = {sym:real for sym, real in zip(inputs, input_vals)}
        if bn_vars is not None:
            feed_dict[bn_vars[0]] = True
            feed_dict[bn_vars[1]] = 0
        return session.run(
            [p[1] for p in prints] + [train_op],
            feed_dict=feed_dict
        )[:-1]

    def bn_stats_fn(input_vals, iter_):
        feed_dict = {sym:real for sym, real in zip(inputs, input_vals)}
        feed_dict[bn_vars[0]] = True
        feed_dict[bn_vars[1]] = iter_
        return session.run(
            [p[1] for p in prints],
            feed_dict=feed_dict
        )

    def eval_fn(input_vals):
        feed_dict = {sym:real for sym, real in zip(inputs, input_vals)}
        if bn_vars is not None:
            feed_dict[bn_vars[0]] = False
            feed_dict[bn_vars[1]] = 0
        return session.run(
            [p[1] for p in prints],
            feed_dict=feed_dict
        )

    _vars = {
        'epoch': 0,
        'iteration': 0,
        'seconds': 0.,
        'last_callback': 0,
        'last_test': 0
    }

    train_generator = train_data()

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    if os.path.isfile(TRAIN_LOOP_FILE):
        print "Resuming interrupted train loop session"
        with open(TRAIN_LOOP_FILE, 'r') as f:
            _vars = pickle.load(f)
        saver.restore(session, os.getcwd()+"/"+PARAMS_FILE)

        print "Fast-fowarding dataset generator"
        dataset_iters = 0
        while dataset_iters < _vars['iteration']:
            try:
                train_generator.next()
            except StopIteration:
                train_generator = train_data()
                train_generator.next()
            dataset_iters += 1
    else:
        print "Initializing variables..."
        session.run(tf.initialize_all_variables())
        print "done!"

    train_output_entries = [[]]
    
    def log(outputs, test, _vars, extra_things_to_print):
        entry = collections.OrderedDict()
        for key in ['epoch', 'iteration', 'seconds']:
            entry[key] = _vars[key]
        for i,p in enumerate(prints):
            if test:
                entry['test '+p[0]] = outputs[i]
            else:
                entry['train '+p[0]] = outputs[i]

        train_output_entries[0].append(entry)

        to_print = entry.items()
        to_print.extend(extra_things_to_print)
        print_str = ""
        for k,v in to_print:
            if isinstance(v, int):
                print_str += "{}:{}\t".format(k,v)
            else:
                print_str += "{}:{:.4f}\t".format(k,v)
        print print_str[:-1] # omit the last \t

    def save_train_output_and_params(iteration):
        print "Saving things..."

        if save_checkpoints:
            # Saving weights takes a while. There's a risk of interruption during
            # this time, leaving the weights file corrupt. Oh well.

            start_time = time.time()
            saver.save(session, PARAMS_FILE)
            print "saver.save time: {}".format(time.time() - start_time)

            start_time = time.time()
            with open(TRAIN_LOOP_FILE, 'w') as f:
                pickle.dump(_vars, f)
            print "_vars pickle dump time: {}".format(time.time() - start_time)

        start_time = time.time()
        with open(TRAIN_OUTPUT_FILE, 'a') as f:
            for entry in train_output_entries[0]:
                for k,v in entry.items():
                    if isinstance(v, np.generic):
                        entry[k] = np.asscalar(v)
                f.write(json.dumps(entry) + "\n")
        print "ndjson write time: {}".format(time.time() - start_time)

        train_output_entries[0] = []

    while True:

        if _vars['iteration'] == stop_after:
            save_train_output_and_params(_vars['iteration'])

            print "Done!"

            try: # This only matters on Ishaan's computer
                import experiment_tools
                experiment_tools.send_sms("done!")
            except ImportError:
                pass

            break

        data_load_start_time = time.time()
        try:
            input_vals = train_generator.next()
        except StopIteration:
            train_generator = train_data()
            input_vals = train_generator.next()
            train_generator.next()
            _vars['epoch'] += 1
        data_load_time = time.time() - data_load_start_time

        if inject_iteration:
            input_vals = [np.int32(_vars['iteration'])] + list(input_vals)

        start_time = time.time()
        outputs = train_fn(input_vals)
        run_time = time.time() - start_time

        _vars['seconds'] += run_time
        _vars['iteration'] += 1

        log(outputs, False, _vars, [('iter time', run_time), ('data time', data_load_time)])

        if ((test_data is not None) and _vars['iteration'] % test_every == (test_every-1)) or ((callback is not None) and _vars['iteration'] % callback_every == (callback_every-1)):
            if inject_iteration:

                if bn_vars is not None: # If using batchnorm, run over a bunch of training data first to make the running-average stats good.
                    _train_gen = train_data()
                    for i in xrange(bn_stats_iters):
                        try:
                            bn_stats_fn([np.int32(_vars['iteration'])] + list(_train_gen.next()), i)
                        except StopIteration:
                            _train_gen = train_data()
                            bn_stats_fn([np.int32(_vars['iteration'])] + list(_train_gen.next()), i)

            else:

                if bn_vars is not None: # If using batchnorm, run over a bunch of training data first to make the running-average stats good.
                    _train_gen = train_data()
                    for i in xrange(bn_stats_iters):
                        try:
                            bn_stats_fn(list(_train_gen.next()), i)
                        except StopIteration:
                            _train_gen = train_data()
                            bn_stats_fn(list(_train_gen.next()), i)


            if (test_data is not None) and _vars['iteration'] % test_every == (test_every-1):
                if inject_iteration:

                    test_outputs = [
                        eval_fn([np.int32(_vars['iteration'])] + list(input_vals))
                        for input_vals in test_data()
                    ]

                else:

                    test_outputs = [
                        eval_fn(list(input_vals))
                        for input_vals in test_data()
                    ]
                mean_test_outputs = np.array(test_outputs).mean(axis=0)

                log(mean_test_outputs, True, _vars, [])

            if (callback is not None) and _vars['iteration'] % callback_every == (callback_every-1):
                tag = "iter{}".format(_vars['iteration'])
                callback(tag)

        if _vars['iteration'] % save_every == (save_every-1):
            save_train_output_and_params(_vars['iteration'])