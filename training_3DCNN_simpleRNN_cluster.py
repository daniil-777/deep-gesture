"""
Training 3DCNN + Simple RNN on original data on cluster
"""

import os
import time
import json

import tensorflow as tf
from dataset_3DCNN_simpleRNN_cluster import Dataset_3DCNN_plus_Simple_RNN
from model_3DCNN_simpleRNN_cluster import Complete_Model


def main(config):

    #############
    # Data
    #############

    # Each <key,value> pair in `training_placeholders` and `validation_placeholders` corresponds to a TF placeholder.
    # Create input placeholders for training data.
    training_file_pattern = os.path.join(
        "/cluster/scratch/beluis/data/", "training-??-of-??"
    )
    training_dataset = Dataset_3DCNN_plus_Simple_RNN(
        data_path=training_file_pattern,
        batch_size=config["batch_size"],
        window_size=config["window_size"],
        shuffle=True,
        normalize=config["normalize_data"],
    )
    training_iterator = training_dataset.get_iterator()
    training_placeholders = training_dataset.get_tf_samples()

    # Create input placeholders for validation data.
    validation_file_pattern = os.path.join(config["data_dir"], "validation-??-of-??")
    validation_dataset = Dataset_3DCNN_plus_Simple_RNN(
        data_path=validation_file_pattern,
        batch_size=config["batch_size"],
        window_size=config["window_size"],
        shuffle=False,
        normalize=config["normalize_data"],
    )
    validation_iterator = validation_dataset.get_iterator()
    validation_placeholders = validation_dataset.get_tf_samples()

    # Using RGB modality.
    # training_input_layer = training_placeholders['rgb']
    # validation_input_layer = validation_placeholders['rgb']

    ##################
    # Training Model
    ##################
    # Create separate graphs for training and validation.
    # Training graph.
    with tf.name_scope("Training"):
        # Create model
        train_model = Complete_Model(
            config=config["Complete_Model"],
            placeholders=training_placeholders,
            mode="training",
        )
        # pretraining!!!
        tf.train.init_from_checkpoint(
            "pretrained_models/3dcnn/1593162915_experiment-name/",
            {
                v.name.split(":")[0]: v
                for v in tf.contrib.framework.get_variables_to_restore()
            },
        )
        train_model.build_graph()
        train_model.build_loss()

        print("\n# of parameters: %s" % train_model.get_num_parameters())

        ##############
        # Optimization
        ##############
        global_step = tf.Variable(1, name="global_step", trainable=False)
        if config["learning_rate_type"] == "exponential":
            learning_rate = tf.train.exponential_decay(
                config["learning_rate"],
                global_step=global_step,
                decay_steps=500,
                decay_rate=0.97,
                staircase=False,
            )
        elif config["learning_rate_type"] == "fixed":
            learning_rate = config["learning_rate"]
        else:
            raise Exception("Invalid learning rate type")

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(train_model.loss, global_step=global_step)

    ###################
    # Validation Model
    ###################
    with tf.name_scope("Validation"):
        # Create model
        valid_model = Complete_Model(
            config=config["Complete_Model"],
            placeholders=validation_placeholders,
            mode="validation",
        )
        valid_model.build_graph()
        valid_model.build_loss()

    ##############
    # Monitoring
    ##############
    # Create placeholders to provide tensorflow average loss and accuracy.
    loss_avg_pl = tf.placeholder(tf.float32, name="loss_avg_pl")
    accuracy_avg_pl = tf.placeholder(tf.float32, name="accuracy_avg_pl")

    # Create summary ops for monitoring the training.
    # Each summary op annotates a node in the computational graph and plots evaluation results.
    summary_train_loss = tf.summary.scalar("loss", train_model.loss)
    summary_train_acc = tf.summary.scalar(
        "accuracy_training", train_model.batch_accuracy
    )
    summary_avg_accuracy = tf.summary.scalar("accuracy_avg", accuracy_avg_pl)
    summary_avg_loss = tf.summary.scalar("loss_avg", loss_avg_pl)
    summary_learning_rate = tf.summary.scalar("learning_rate", learning_rate)

    # Group summaries. summaries_training is used during training and reported after every step.
    summaries_training = tf.summary.merge(
        [summary_train_loss, summary_train_acc, summary_learning_rate]
    )
    # summaries_evaluation is used by both training and validation in order to report the performance on the dataset.
    summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

    # Create session object
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    )

    """
    # Create session object
    session = tf.Session()
    """

    # Add the ops to initialize variables.
    init_op = tf.group(
        tf.global_variables_initializer(), tf.local_variables_initializer()
    )
    # Actually initialize the variables
    session.run(init_op)

    # Register summary ops.
    train_summary_dir = os.path.join(config["model_dir"], "summary", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)
    valid_summary_dir = os.path.join(config["model_dir"], "summary", "valid")
    valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, session.graph)

    # Create a saver for saving checkpoints.
    saver = tf.train.Saver(
        var_list=tf.trainable_variables(), max_to_keep=3, save_relative_paths=True
    )

    # Define counters in order to accumulate measurements.
    counter_correct_predictions_training = 0.0
    counter_loss_training = 0.0
    counter_correct_predictions_validation = 0.0
    counter_loss_validation = 0.0

    # Save configuration in json formats.
    json.dump(
        config,
        open(os.path.join(config["model_dir"], "config.json"), "w"),
        indent=4,
        sort_keys=True,
    )

    ##########################
    # Training Loop
    ##########################
    session.run(training_iterator.initializer)
    session.run(validation_iterator.initializer)
    epoch = 0
    step = 0
    stop_signal = False
    """
    # UNCOMMENT THIS PART FOR EARLY STOPPING
    valid_performance = []
    valid_performance_counter = 0
    tolerance_steps = 3 # Set tolerance steps
    initial_warmup = 5000
    """
    while not stop_signal:
        # Run training for some steps and then run evaluation on the validation split.
        for i in range(config["evaluate_every_step"]):
            try:
                step += 1
                start_time = time.perf_counter()
                # Run the optimizer to update weights.
                # Note that "train_op" is responsible from updating network weights.
                # Only the operations that are fed are evaluated.
                # Run the optimizer to update weights.
                train_summary, num_correct_predictions, loss, _ = session.run(
                    [
                        summaries_training,
                        train_model.num_correct_predictions,
                        train_model.loss,
                        train_op,
                    ],
                    feed_dict={},
                )
                # Update counters.
                counter_correct_predictions_training += num_correct_predictions
                counter_loss_training += loss
                # Write summary data.
                train_summary_writer.add_summary(train_summary, step)

                # Report training performance
                if (step % config["print_every_step"]) == 0:
                    # To get a smoother loss plot, we calculate average performance.
                    accuracy_avg = counter_correct_predictions_training / (
                        config["batch_size"] * config["print_every_step"]
                    )
                    loss_avg = counter_loss_training / (config["print_every_step"])
                    # Feed average performance.
                    summary_report = session.run(
                        summaries_evaluation,
                        feed_dict={
                            accuracy_avg_pl: accuracy_avg,
                            loss_avg_pl: loss_avg,
                        },
                    )
                    train_summary_writer.add_summary(summary_report, step)
                    time_elapsed = (time.perf_counter() - start_time) / config[
                        "print_every_step"
                    ]
                    print(
                        "[Train/%d] Accuracy: %.3f, Loss: %.3f, time/step = %.3f"
                        % (step, accuracy_avg, loss_avg, time_elapsed)
                    )
                    counter_correct_predictions_training = 0.0
                    counter_loss_training = 0.0

            except tf.errors.OutOfRangeError:
                # Dataset iterator throws an exception after all samples are used (i.e., epoch). Reinitialize the
                # iterator to start a new epoch.
                session.run(training_iterator.initializer)
                print("done epoch", epoch)
                epoch += 1

                if epoch >= config["num_epochs"]:
                    stop_signal = True
                    break

        # Evaluate the model.
        validation_step = 0
        start_time = time.perf_counter()
        try:
            # Here we evaluate our model on entire validation split.
            # Validation model fetches the data samples from the iterator every session.run that evaluates an
            # op using input sample. We don't need to do anything else.
            while True:
                # Calculate average validation accuracy.
                num_correct_predictions, loss = session.run(
                    [valid_model.num_correct_predictions, valid_model.loss]
                )
                # Update counters.
                counter_correct_predictions_validation += num_correct_predictions
                counter_loss_validation += loss
                validation_step += 1

        except tf.errors.OutOfRangeError:
            # Report validation performance
            accuracy_avg = counter_correct_predictions_validation / (
                config["batch_size"] * validation_step
            )
            loss_avg = counter_loss_validation / validation_step
            summary_report = session.run(
                summaries_evaluation,
                feed_dict={accuracy_avg_pl: accuracy_avg, loss_avg_pl: loss_avg},
            )
            valid_summary_writer.add_summary(summary_report, step)
            time_elapsed = (time.perf_counter() - start_time) / validation_step
            print(
                "[Valid/%d] Accuracy: %.3f, Loss: %.3f, time/step = %.3f"
                % (step, accuracy_avg, loss_avg, time_elapsed)
            )
            counter_correct_predictions_validation = 0.0
            counter_loss_validation = 0.0

            """
            # UNCOMMENT THIS PART FOR EARLY STOPPING
            valid_performance.append(loss_avg)
            valid_performance_counter += 1
            if valid_performance_counter >= initial_warmup:
                c = valid_performance_counter
                checks = [0] * tolerance_steps
                for i in range(tolerance_steps):
                    deviation = (valid_performance[c - (i+1)] - valid_performance[c - (i+2)])
                    if deviation >= 0:
                        checks[i] = 1
                if (abs(sum(checks) - tolerance_steps) < 0.0001) :
                    stop_signal = True
             """
            # Initialize the validation data iterator for the next evaluation round.
            session.run(validation_iterator.initializer)

        if (step % config["checkpoint_every_step"]) == 0:
            ckpt_save_path = saver.save(
                session, os.path.join(config["model_dir"], "model"), global_step
            )
            print("Model saved in file: %s" % ckpt_save_path)

    session.close()
    """
    # Evaluate model after training and create submission file.
    tf.reset_default_graph()
    from restore_and_evaluate import main as evaluate
    config['checkpoint_id'] = None
    evaluate(config)
    """


if __name__ == "__main__":
    from config_3DCNN_simpleRNN_cluster import config

    main(config)
