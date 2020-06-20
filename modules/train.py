import os
import time
import datetime

import infolog
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from modules.models import create_model
from modules.utils import ValueWindow
from modules.lr_scheduler import WarmUpSchedule
from modules.step import valid_step
from modules.decode import greedy_decode
from modules.loss import ctc_loss, ctc_label_error_rate
from text_syllable import index_token, token_index
import matplotlib.pyplot as plt

log = infolog.log


def time_string():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def train(log_dir, args):
    save_dir = os.path.join(log_dir, 'pretrained')
    valid_dir = os.path.join(log_dir, 'valid-dir')
    tensorboard_dir = os.path.join(log_dir, 'events', time_string())
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'model')
    input_path = os.path.join(args.base_dir, args.training_input)
    valid_path = os.path.join(args.base_dir, args.validation_input)

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_path))
    log('Using model: {}'.format(args.model))

    # Start by setting a seed for repeatability
    tf.random.set_seed(args.random_seed)

    # To find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(False)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Set up data feeder
    from datasets.feeder import dataset
    train_dataset, valid_dataset, train_steps, valid_steps = dataset(input_path, args)

    # Track the model
    train_summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    valid_summary_writer = tf.summary.create_file_writer(tensorboard_dir)

    # metrics to measure the loss of the model
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    train_ler = tf.keras.metrics.Mean(name='train_ler')
    valid_ler = tf.keras.metrics.Mean(name='valid_ler')

    # Set up model
    speech_model = create_model(args.model, save_dir, args)

    summary_list = list()
    speech_model.model.summary(line_length=180, print_fn=lambda x: summary_list.append(x))
    for summary in summary_list:
        log(summary)

    tf.keras.utils.plot_model(speech_model.model, os.path.join(log_dir, 'model.png'), show_shapes=True)

    learning_rate = WarmUpSchedule(args.num_units_per_lstm)
    opt = Adam(learning_rate, beta_1=args.adam_beta1, beta_2=args.adam_beta2, epsilon=args.adam_epsilon)

    temp_learning_rate = WarmUpSchedule(args.num_units_per_lstm, int(train_steps * 5))
    plt.plot(temp_learning_rate(tf.range(50000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, encoder=speech_model.model)
    manager = tf.train.CheckpointManager(checkpoint, directory=save_dir, max_to_keep=5)

    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
        checkpoint.restore(manager.latest_checkpoint)
    else:
        log('No model to load at {}'.format(save_dir), slack=True)
        log('Starting new training!', slack=True)
    eval_best_loss = np.inf

    # Book keeping
    patience_count = 0
    time_window = ValueWindow(100)

    log('Speech Recognition training set to a maximum of {} epochs'.format(args.train_epochs))

    def create_lengths(input, label):
        input_lengths = tf.reduce_sum(tf.cast(tf.logical_not(tf.math.equal(input, 0)), tf.int32), axis=-2)
        label_lengths = tf.reduce_sum(tf.cast(tf.logical_not(tf.math.equal(label, 0)), tf.int32), axis=-1)
        return input_lengths[:, 0], label_lengths

    train_step_signature = [tf.TensorSpec(shape=(None, None, 80), dtype=tf.float32),
                            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

    @tf.function(input_signature=train_step_signature)
    def train_step(input, label):
        input_len, label_len = create_lengths(input, label)
        with tf.GradientTape() as tape:
            logit = speech_model.model(input, training=True)
            loss = ctc_loss(label, logit, input_len, label_len)
        grads = tape.gradient(loss, speech_model.model.trainable_variables)
        if args.clip_gradients:
            clipped_grads, _ = tf.clip_by_global_norm(grads, args.clip_gradients)
        else:
            clipped_grads = grads
        opt.apply_gradients(zip(clipped_grads, speech_model.model.trainable_variables))
        ler = ctc_label_error_rate(label, logit, input_len, label_len)
        train_loss.update_state(loss)
        train_ler.update_state(ler)

    # Train
    for epoch in range(args.train_epochs):
        # show the current epoch number
        log("[INFO] starting epoch {}/{}...".format(1 + epoch, args.train_epochs))
        epochStart = time.time()

        train_loss.reset_states()
        train_ler.reset_states()
        valid_loss.reset_states()
        valid_ler.reset_states()

        # loop over the data in batch size increments
        for (batch, (input, label)) in enumerate(train_dataset):
            start_time = time.time()
            # take a step
            train_step(input, label)
            # book keeping
            time_window.append(time.time() - start_time)
            message = '[Epoch {:3d}] [Step {:7d}] [{:.3f} sec/step, loss={:.5f}, ler={:.5f}]'.format(
                epoch + (batch / train_steps), int(checkpoint.step), time_window.average,
                train_loss.result(), train_ler.result())

            log(message)
            checkpoint.step.assign_add(1)

            if train_loss.result() > 1e15 or np.isnan(train_loss.result()):
                log('Loss exploded to {:.5f} at step {}'.format(train_loss.result(), int(checkpoint.step)))
                raise Exception('Loss exploded')

            if int(checkpoint.step) % 1000 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(), step=int(checkpoint.step))
                    tf.summary.scalar('train_ler', train_ler.result(), step=int(checkpoint.step))

        if (1 + epoch) % args.eval_interval == 0:
            # Run eval and save eval stats
            log('\nRunning evaluation at epoch {}'.format(epoch))
            for (batch, (input, label)) in enumerate(valid_dataset):
                input_len, label_len = create_lengths(input, label)
                # take a step
                valid_logit = valid_step(input, label, input_len, label_len, speech_model, valid_loss, valid_ler)
                if batch % (valid_steps // 10) == 0:
                    decoded = greedy_decode(tf.expand_dims(valid_logit[0], axis=0), tf.expand_dims(input_len[0], axis=-1)[tf.newaxis, ...])
                    decoded = ''.join([index_token[x] for x in decoded])
                    original = ''.join([index_token[x] for x in label.numpy()[0]])
                    log('Original: %s' % original)
                    log('Decoded: %s' % decoded)

            log('Eval loss & ler for global step {}: {:.3f}, {:.3f}'.format(int(checkpoint.step),
                                                                            valid_loss.result(),
                                                                            valid_ler.result()))

            with valid_summary_writer.as_default():
                tf.summary.scalar('valid_loss', valid_loss.result(), step=int(checkpoint.step))
                tf.summary.scalar('valid_ler', valid_ler.result(), step=int(checkpoint.step))

            # Save model and current global step
            save_path = manager.save()
            log("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

            if valid_loss.result() < eval_best_loss:
                # Save model and current global step
                save_path = manager.save()
                log("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
                log('Validation loss improved from {:.2f} to {:.2f}'.format(eval_best_loss, valid_loss.result()))
                eval_best_loss = valid_loss.result()
                patience_count = 0
            else:
                patience_count += 1
                log('Patience: {} times'.format(patience_count))
                if patience_count == args.patience:
                    log('Validation loss has not been improved for {} times, early stopping'.format(
                        args.patience))
                    log('Training complete after {} global steps!'.format(int(checkpoint.step)), slack=True)
                    return save_dir

        elapsed = (time.time() - epochStart) / 60.0
        log("one epoch took {:.4} minutes".format(elapsed))

    log('Separation training complete after {} epochs!'.format(args.train_epochs), slack=True)

    return save_dir


def sr_train(args, log_dir):
    return train(log_dir, args)
